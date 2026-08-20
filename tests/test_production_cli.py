import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import production_cli
from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict
from services.evidence_sufficiency_policy import EvidenceSufficiencyAssessment
from services.production_execution import (
    RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE,
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    OperationalFailure,
    OperationalFailureCode,
    OperationalFailureStage,
    ProductionExecutionOutcome,
    ProductionExecutionStatus,
)
from services.production_result import ProductionResult


class FakeExecutionService:
    def __init__(self, *, outcome=None, error=None):
        self.outcome = outcome
        self.error = error
        self.calls = []

    def execute(self, session_id, claim, video):
        self.calls.append((session_id, claim, video))
        if self.error is not None:
            raise self.error
        return self.outcome


class RecordingFactory:
    def __init__(self, services=None, error=None):
        self.services = list(services or [])
        self.error = error
        self.calls = []

    def __call__(self, config_path):
        self.calls.append(config_path)
        if self.error is not None:
            raise self.error
        return self.services.pop(0)


class BrokenSerializationOutcome(ProductionExecutionOutcome):
    def to_json(self):
        raise ValueError("serialization secret at /private/result.json")


class ProductionCLITests(unittest.TestCase):
    ARGS = [
        "--config",
        "deployment.json",
        "--session-id",
        "session-1",
        "--claim",
        "claim",
        "--video",
        "video.mp4",
    ]

    @staticmethod
    def _production_result(model_verdict):
        completed = model_verdict in {ModelVerdict.FAKE, ModelVerdict.REAL}
        display_verdict = {
            ModelVerdict.FAKE: DisplayVerdict.FAKE,
            ModelVerdict.REAL: DisplayVerdict.REAL,
            ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
        }[model_verdict]
        evidence_status = (
            EvidenceStatus.SUFFICIENT
            if completed
            else EvidenceStatus.INSUFFICIENT
        )
        sufficiency = EvidenceSufficiencyAssessment(
            status=evidence_status,
            reason_code=(
                "frozen_g1_evidence_available_and_model_completed"
                if completed
                else "no_frozen_g1_eligible_evidence"
            ),
            model_was_run=completed,
            g1_exposure_count=1 if completed else 0,
            transcript_exposure_count=1 if completed else 0,
            ocr_exposure_count=0,
            visual_unit_count=0,
            top_k_count=1 if completed else 0,
            supplemental_visual_present=False,
        )
        return ProductionResult(
            schema_version=1,
            session_id="session-1",
            claim="  exact claim 汉语  ",
            model_verdict=model_verdict,
            display_verdict=display_verdict,
            evidence_status=evidence_status,
            sample_logits=(
                (("fake", 1.0), ("real", -1.0)) if completed else ()
            ),
            probabilities=(
                (("fake", 0.8), ("real", 0.2)) if completed else ()
            ),
            class_winners=(("fake", "unit-1"),) if completed else (),
            checkpoint_sha256="checkpoint" if completed else None,
            sufficiency=sufficiency,
            g1_exposure_units=(),
            g1_top_k_explanation_unit_ids=("unit-1",) if completed else (),
            visual_supplemental_units=(),
            runtime_ms=12.5,
        )

    @classmethod
    def _success(cls, model_verdict):
        return ProductionExecutionOutcome(
            schema_version=1,
            status=ProductionExecutionStatus.SUCCESS,
            result=cls._production_result(model_verdict),
            failure=None,
        )

    @staticmethod
    def _failure(stage):
        if stage is OperationalFailureStage.RUNTIME:
            code = OperationalFailureCode.RUNTIME_EXECUTION_FAILED
            message = RUNTIME_FAILURE_PUBLIC_MESSAGE
        else:
            code = OperationalFailureCode.RESULT_PACKAGING_FAILED
            message = RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE
        return ProductionExecutionOutcome(
            schema_version=1,
            status=ProductionExecutionStatus.FAILURE,
            result=None,
            failure=OperationalFailure(
                stage=stage,
                code=code,
                exception_type="RuntimeError",
                public_message=message,
            ),
        )

    def _run(self, service, args=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        factory = RecordingFactory([service])
        code = production_cli.run_cli(
            self.ARGS if args is None else args,
            service_factory=factory,
            stdout=stdout,
            stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue(), factory

    def test_parser_exposes_exactly_four_required_production_options(self):
        actions = [
            action
            for action in production_cli.build_parser()._actions
            if action.dest != "help"
        ]
        self.assertEqual(
            {action.dest for action in actions},
            {"config", "session_id", "claim", "video"},
        )
        self.assertTrue(all(action.required for action in actions))

    def test_each_production_argument_is_required(self):
        parser = production_cli.build_parser()
        pairs = [self.ARGS[index : index + 2] for index in range(0, 8, 2)]
        for omitted in range(4):
            with self.subTest(omitted=pairs[omitted][0]):
                args = [item for index, pair in enumerate(pairs) if index != omitted for item in pair]
                with patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parser.parse_args(args)
                self.assertEqual(raised.exception.code, production_cli.EXIT_CLI_ERROR)

    def test_scientific_and_deployment_override_options_do_not_exist(self):
        options = {
            option
            for action in production_cli.build_parser()._actions
            for option in action.option_strings
        }
        forbidden = {
            "--device",
            "--model",
            "--top-k",
            "--max-units",
            "--pooling",
            "--threshold",
            "--visual-g1",
            "--max-new-tokens",
            "--candidate-frames",
            "--output",
            "--json-output",
            "--log-file",
        }
        self.assertTrue(forbidden.isdisjoint(options))

    def test_exit_code_constants_are_exact(self):
        self.assertEqual(production_cli.EXIT_SUCCESS, 0)
        self.assertEqual(production_cli.EXIT_EXECUTION_FAILURE, 1)
        self.assertEqual(production_cli.EXIT_CLI_ERROR, 2)

    def test_default_factory_is_production_execution_service_from_json(self):
        service = FakeExecutionService(outcome=self._success(ModelVerdict.FAKE))
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "app.production_cli.ProductionExecutionService.from_json",
            return_value=service,
        ) as from_json:
            code = production_cli.run_cli(
                self.ARGS,
                stdout=stdout,
                stderr=stderr,
            )
        from_json.assert_called_once_with(Path("deployment.json"))
        self.assertEqual(code, 0)

    def test_one_path_factory_construction_and_one_execute_call(self):
        service = FakeExecutionService(outcome=self._success(ModelVerdict.FAKE))
        code, _, _, factory = self._run(service)
        self.assertIsInstance(factory.calls[0], Path)
        self.assertEqual(factory.calls, [Path("deployment.json")])
        self.assertEqual(service.calls, [("session-1", "claim", "video.mp4")])
        self.assertIsInstance(code, int)

    def test_request_arguments_are_passed_unchanged_without_cli_validation(self):
        service = FakeExecutionService(outcome=self._success(ModelVerdict.FAKE))
        args = [
            "--config",
            "/missing/config.json",
            "--session-id",
            " unsafe/session ",
            "--claim",
            "  Exact Unicode 汉语 Claim  ",
            "--video",
            "../missing video.mp4",
        ]
        code, _, _, factory = self._run(service, args)
        self.assertEqual(code, 0)
        self.assertEqual(factory.calls, [Path("/missing/config.json")])
        self.assertEqual(
            service.calls,
            [(" unsafe/session ", "  Exact Unicode 汉语 Claim  ", "../missing video.mp4")],
        )

    def test_fake_real_and_nei_success_print_exact_json_and_return_zero(self):
        for verdict in (ModelVerdict.FAKE, ModelVerdict.REAL, ModelVerdict.NOT_RUN):
            with self.subTest(verdict=verdict):
                outcome = self._success(verdict)
                code, stdout, stderr, _ = self._run(
                    FakeExecutionService(outcome=outcome)
                )
                self.assertEqual(code, production_cli.EXIT_SUCCESS)
                self.assertEqual(stdout, outcome.to_json() + "\n")
                self.assertEqual(stderr, "")

    def test_success_exit_code_does_not_depend_on_model_verdict(self):
        codes = []
        for verdict in (ModelVerdict.FAKE, ModelVerdict.REAL, ModelVerdict.NOT_RUN):
            code, _, _, _ = self._run(
                FakeExecutionService(outcome=self._success(verdict))
            )
            codes.append(code)
        self.assertEqual(codes, [0, 0, 0])

    def test_both_operational_failure_stages_print_exact_json_and_return_one(self):
        for stage in (
            OperationalFailureStage.RUNTIME,
            OperationalFailureStage.RESULT_PACKAGING,
        ):
            with self.subTest(stage=stage):
                outcome = self._failure(stage)
                code, stdout, stderr, _ = self._run(
                    FakeExecutionService(outcome=outcome)
                )
                self.assertEqual(code, production_cli.EXIT_EXECUTION_FAILURE)
                self.assertEqual(stdout, outcome.to_json() + "\n")
                self.assertEqual(stderr, "")

    def test_obtained_outcome_is_the_only_stdout_document(self):
        outcome = self._success(ModelVerdict.FAKE)
        _, stdout, _, _ = self._run(FakeExecutionService(outcome=outcome))
        self.assertEqual(stdout.count("\n"), 1)
        self.assertEqual(stdout, outcome.to_json() + "\n")

    def test_initialization_exceptions_are_redacted_with_no_stdout(self):
        unsafe = "/secret/deployment/model/config.json"
        for error in (
            RuntimeError(unsafe),
            ValueError(unsafe),
            FileNotFoundError(unsafe),
        ):
            with self.subTest(error=type(error).__name__):
                stdout = io.StringIO()
                stderr = io.StringIO()
                code = production_cli.run_cli(
                    self.ARGS,
                    service_factory=RecordingFactory(error=error),
                    stdout=stdout,
                    stderr=stderr,
                )
                self.assertEqual(code, production_cli.EXIT_CLI_ERROR)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(
                    stderr.getvalue(),
                    production_cli.INITIALIZATION_FAILURE_MESSAGE + "\n",
                )
                self.assertNotIn(unsafe, stderr.getvalue())

    def test_unexpected_execute_exception_is_redacted_and_does_not_become_nei(self):
        unsafe = "internal failure at /private/model.bin"
        code, stdout, stderr, _ = self._run(
            FakeExecutionService(error=RuntimeError(unsafe))
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            production_cli.INITIALIZATION_FAILURE_MESSAGE + "\n",
        )
        self.assertNotIn(unsafe, stderr)
        self.assertNotIn("NEI", stderr)

    def test_factory_does_not_catch_base_exceptions(self):
        for error in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    production_cli.run_cli(
                        self.ARGS,
                        service_factory=RecordingFactory(error=error),
                        stdout=io.StringIO(),
                        stderr=io.StringIO(),
                    )

    def test_execute_does_not_catch_base_exceptions(self):
        for error in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    self._run(FakeExecutionService(error=error))

    def test_invalid_outcome_type_is_safe_cli_error(self):
        secret = "/private/internal/result"
        code, stdout, stderr, _ = self._run(
            FakeExecutionService(outcome={"raw_result": secret})
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            production_cli.INITIALIZATION_FAILURE_MESSAGE + "\n",
        )
        self.assertNotIn(secret, stderr)

    def test_unexpected_serialization_exception_is_redacted(self):
        valid = self._success(ModelVerdict.FAKE)
        outcome = BrokenSerializationOutcome(
            schema_version=valid.schema_version,
            status=valid.status,
            result=valid.result,
            failure=valid.failure,
        )
        code, stdout, stderr, _ = self._run(
            FakeExecutionService(outcome=outcome)
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            production_cli.INITIALIZATION_FAILURE_MESSAGE + "\n",
        )
        self.assertNotIn("serialization secret", stderr)

    def test_repeated_runs_construct_independent_services_once_each(self):
        first = FakeExecutionService(outcome=self._success(ModelVerdict.FAKE))
        second = FakeExecutionService(outcome=self._success(ModelVerdict.REAL))
        factory = RecordingFactory([first, second])
        for _ in range(2):
            self.assertEqual(
                production_cli.run_cli(
                    self.ARGS,
                    service_factory=factory,
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                ),
                0,
            )
        self.assertEqual(factory.calls, [Path("deployment.json")] * 2)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_cli_writes_no_result_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = list(self.ARGS)
            args[1] = str(root / "missing-config.json")
            before = list(root.rglob("*"))
            code, _, _, _ = self._run(
                FakeExecutionService(outcome=self._success(ModelVerdict.FAKE)),
                args,
            )
            after = list(root.rglob("*"))
        self.assertEqual(code, 0)
        self.assertEqual(after, before)

    def test_cli_source_has_only_the_execution_service_dependency(self):
        source = inspect.getsource(production_cli)
        self.assertIn("from services.production_execution import", source)
        forbidden = (
            "FrozenG1Runner",
            "VideoMultimodalRunner",
            "ProductionRuntimeFactory",
            "ProductionResultBuilder",
            "EvidenceSufficiencyPolicy",
            "subprocess",
            "Validation/",
            "Test/",
        )
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, source)

    def test_cli_source_contains_no_probability_or_logit_threshold(self):
        source = inspect.getsource(production_cli).lower()
        for value in ("probability", "logit", "threshold"):
            with self.subTest(value=value):
                self.assertNotIn(value, source)

    def test_main_raises_system_exit_with_run_cli_code(self):
        with patch("app.production_cli.run_cli", return_value=1) as run_cli:
            with self.assertRaises(SystemExit) as raised:
                production_cli.main()
        self.assertEqual(raised.exception.code, 1)
        run_cli.assert_called_once_with()

    def test_task06f_serialization_is_forwarded_without_changes(self):
        outcome = self._success(ModelVerdict.NOT_RUN)
        _, stdout, _, _ = self._run(FakeExecutionService(outcome=outcome))
        self.assertEqual(stdout[:-1], outcome.to_json())


if __name__ == "__main__":
    unittest.main()
