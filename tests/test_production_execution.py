import inspect
import json
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict
from services.evidence_sufficiency_policy import EvidenceSufficiencyAssessment
from services.frozen_g1_runner import FrozenG1Runner
from services.paddle_ocr_service import PaddleOCRService
from services.production_execution import (
    RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE,
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    OperationalFailure,
    OperationalFailureCode,
    OperationalFailureStage,
    ProductionExecutionOutcome,
    ProductionExecutionService,
    ProductionExecutionStatus,
)
from services.production_result import ProductionResult, ProductionResultBuilder
from services.production_runtime import ProductionRuntime
from services.qwen_visual_observer import QwenVisualObserver
from services.siglip_visual_retriever import SigLIPVisualRetriever
from services.video_multimodal_runner import VideoMultimodalRunner
from services.whisper_asr_service import WhisperASRService


class FakeRuntime:
    def __init__(self, outcome=None, error=None):
        self.outcome = outcome
        self.error = error
        self.calls = []

    def run(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        if self.error is not None:
            raise self.error
        return self.outcome


class FakeBuilder:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def build(self, internal):
        self.calls.append(internal)
        if self.error is not None:
            raise self.error
        return self.result


class ProductionExecutionTests(unittest.TestCase):
    @staticmethod
    def _production_result(model_verdict=ModelVerdict.FAKE):
        completed = model_verdict in {ModelVerdict.FAKE, ModelVerdict.REAL}
        display_verdict = (
            DisplayVerdict.FAKE
            if model_verdict is ModelVerdict.FAKE
            else DisplayVerdict.REAL
            if model_verdict is ModelVerdict.REAL
            else DisplayVerdict.NEI
        )
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
            visual_unit_count=1,
            top_k_count=1 if completed else 0,
            supplemental_visual_present=True,
        )
        return ProductionResult(
            schema_version=1,
            session_id="session-1",
            claim="  exact claim  ",
            model_verdict=model_verdict,
            display_verdict=display_verdict,
            evidence_status=evidence_status,
            sample_logits=(
                (("fake", 1.25), ("real", -0.25)) if completed else ()
            ),
            probabilities=(
                (("fake", 0.75), ("real", 0.25)) if completed else ()
            ),
            class_winners=(("fake", "unit-1"),) if completed else (),
            checkpoint_sha256="checkpoint" if completed else None,
            sufficiency=sufficiency,
            g1_exposure_units=(),
            g1_top_k_explanation_unit_ids=("unit-1",) if completed else (),
            visual_supplemental_units=(),
            runtime_ms=123.456,
        )

    @staticmethod
    def _runtime_failure(exception_type="RuntimeError"):
        return OperationalFailure(
            stage=OperationalFailureStage.RUNTIME,
            code=OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
            exception_type=exception_type,
            public_message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
        )

    @staticmethod
    def _failure_outcome(exception_type="RuntimeError"):
        return ProductionExecutionOutcome(
            schema_version=1,
            status=ProductionExecutionStatus.FAILURE,
            result=None,
            failure=ProductionExecutionTests._runtime_failure(exception_type),
        )

    @staticmethod
    def _recursive_keys(value):
        keys = set()
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                keys.update(ProductionExecutionTests._recursive_keys(item))
        elif isinstance(value, list):
            for item in value:
                keys.update(ProductionExecutionTests._recursive_keys(item))
        return keys

    def test_public_enum_values_are_exact(self):
        self.assertEqual(
            {item.value for item in ProductionExecutionStatus},
            {"success", "failure"},
        )
        self.assertEqual(
            {item.value for item in OperationalFailureStage},
            {"runtime", "result_packaging"},
        )
        self.assertEqual(
            {item.value for item in OperationalFailureCode},
            {"runtime_execution_failed", "result_packaging_failed"},
        )

    def test_public_contracts_are_frozen(self):
        failure = self._runtime_failure()
        outcome = self._failure_outcome()
        with self.assertRaises(FrozenInstanceError):
            failure.code = OperationalFailureCode.RESULT_PACKAGING_FAILED
        with self.assertRaises(FrozenInstanceError):
            outcome.status = ProductionExecutionStatus.SUCCESS

    def test_outcome_schema_version_must_be_exact_integer_one(self):
        result = self._production_result()
        for invalid in (0, 2, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    ProductionExecutionOutcome(
                        schema_version=invalid,
                        status=ProductionExecutionStatus.SUCCESS,
                        result=result,
                        failure=None,
                    )

    def test_outcome_success_and_failure_invariants(self):
        result = self._production_result()
        failure = self._runtime_failure()
        invalid = (
            (ProductionExecutionStatus.SUCCESS, None, None),
            (ProductionExecutionStatus.SUCCESS, result, failure),
            (ProductionExecutionStatus.FAILURE, None, None),
            (ProductionExecutionStatus.FAILURE, result, failure),
        )
        for status, outcome_result, outcome_failure in invalid:
            with self.subTest(status=status, result=outcome_result):
                with self.assertRaises(ValueError):
                    ProductionExecutionOutcome(
                        schema_version=1,
                        status=status,
                        result=outcome_result,
                        failure=outcome_failure,
                    )

    def test_failure_code_and_fixed_message_must_match_stage(self):
        invalid = (
            (
                OperationalFailureStage.RUNTIME,
                OperationalFailureCode.RESULT_PACKAGING_FAILED,
                RUNTIME_FAILURE_PUBLIC_MESSAGE,
            ),
            (
                OperationalFailureStage.RUNTIME,
                OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                "unsafe exception text",
            ),
        )
        for stage, code, message in invalid:
            with self.subTest(code=code, message=message):
                with self.assertRaises(ValueError):
                    OperationalFailure(
                        stage=stage,
                        code=code,
                        exception_type="RuntimeError",
                        public_message=message,
                    )

    def test_failure_and_outcome_are_json_serializable(self):
        failure = self._runtime_failure()
        outcome = self._failure_outcome()
        self.assertIsInstance(json.dumps(failure.to_dict()), str)
        self.assertEqual(json.loads(outcome.to_json()), outcome.to_dict())

    def test_outcome_to_json_is_deterministic_and_writes_no_file(self):
        outcomes = (
            ProductionExecutionOutcome(
                schema_version=1,
                status=ProductionExecutionStatus.SUCCESS,
                result=self._production_result(),
                failure=None,
            ),
            self._failure_outcome(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = list(root.rglob("*"))
            for outcome in outcomes:
                self.assertEqual(outcome.to_json(), outcome.to_json())
            after = list(root.rglob("*"))
        self.assertEqual(after, before)

    def test_constructor_is_lazy_and_retains_exact_dependencies(self):
        runtime = FakeRuntime()
        builder = FakeBuilder()
        service = ProductionExecutionService(runtime, result_builder=builder)
        self.assertIs(service.runtime, runtime)
        self.assertIs(service.result_builder, builder)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(builder.calls, [])

    def test_constructor_creates_default_builder_without_execution(self):
        runtime = FakeRuntime()
        service = ProductionExecutionService(runtime)
        self.assertIsInstance(service.result_builder, ProductionResultBuilder)
        self.assertEqual(runtime.calls, [])

    def test_default_builder_receives_runtime_cache_root_for_evidence_images(self):
        runtime = FakeRuntime()
        runtime.config = SimpleNamespace(cache_root=Path("/deployment/cache"))

        service = ProductionExecutionService(runtime)

        self.assertEqual(
            service.result_builder.evidence_root,
            Path("/deployment/cache").resolve(),
        )
        self.assertEqual(runtime.calls, [])

    def test_constructor_rejects_incompatible_dependencies(self):
        with self.assertRaisesRegex(TypeError, "runtime"):
            ProductionExecutionService(object())
        with self.assertRaisesRegex(TypeError, "result_builder"):
            ProductionExecutionService(FakeRuntime(), result_builder=object())

    def test_from_json_constructs_one_lazy_runtime(self):
        runtime = FakeRuntime()
        config_path = Path("/deployment/config.json")
        with patch.object(
            ProductionRuntime,
            "from_json",
            return_value=runtime,
        ) as from_json:
            service = ProductionExecutionService.from_json(config_path)
        from_json.assert_called_once_with(config_path)
        self.assertIs(service.runtime, runtime)
        self.assertEqual(runtime.calls, [])

    def test_fake_real_and_nei_are_all_successful_execution_outcomes(self):
        for verdict in (
            ModelVerdict.FAKE,
            ModelVerdict.REAL,
            ModelVerdict.NOT_RUN,
        ):
            with self.subTest(verdict=verdict):
                internal = object()
                public = self._production_result(verdict)
                runtime = FakeRuntime(outcome=internal)
                builder = FakeBuilder(result=public)
                outcome = ProductionExecutionService(
                    runtime,
                    result_builder=builder,
                ).execute("session-1", "claim", "video.mp4")
                self.assertIs(outcome.status, ProductionExecutionStatus.SUCCESS)
                self.assertIs(outcome.result, public)
                self.assertIsNone(outcome.failure)
                self.assertEqual(runtime.calls, [("session-1", "claim", "video.mp4")])
                self.assertEqual(builder.calls, [internal])

    def test_success_delegates_exactly_once_and_returns_exact_result(self):
        internal = object()
        public = self._production_result()
        runtime = FakeRuntime(outcome=internal)
        builder = FakeBuilder(result=public)
        service = ProductionExecutionService(runtime, result_builder=builder)
        outcome = service.execute("Session.Exact", "  claim  ", Path("video.mp4"))
        self.assertEqual(
            runtime.calls,
            [("Session.Exact", "  claim  ", Path("video.mp4"))],
        )
        self.assertEqual(builder.calls, [internal])
        self.assertIs(outcome.result, public)

    def test_runtime_failures_use_one_coarse_safe_classification(self):
        exceptions = (
            RuntimeError("runtime secret"),
            ValueError("validation secret"),
            FileNotFoundError("asset secret"),
            OSError("worker secret"),
        )
        for error in exceptions:
            with self.subTest(error_type=type(error).__name__):
                runtime = FakeRuntime(error=error)
                builder = FakeBuilder(result=self._production_result())
                outcome = ProductionExecutionService(
                    runtime,
                    result_builder=builder,
                ).execute("session", "claim", "video.mp4")
                self.assertIs(outcome.status, ProductionExecutionStatus.FAILURE)
                self.assertIsNone(outcome.result)
                self.assertEqual(builder.calls, [])
                self.assertIs(
                    outcome.failure.stage,
                    OperationalFailureStage.RUNTIME,
                )
                self.assertIs(
                    outcome.failure.code,
                    OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                )
                self.assertEqual(
                    outcome.failure.exception_type,
                    type(error).__name__,
                )
                self.assertEqual(
                    outcome.failure.public_message,
                    RUNTIME_FAILURE_PUBLIC_MESSAGE,
                )

    def test_packaging_failure_uses_safe_packaging_classification(self):
        for error in (
            RuntimeError("packaging secret"),
            ValueError("malformed result secret"),
        ):
            with self.subTest(error_type=type(error).__name__):
                internal = object()
                runtime = FakeRuntime(outcome=internal)
                builder = FakeBuilder(error=error)
                outcome = ProductionExecutionService(
                    runtime,
                    result_builder=builder,
                ).execute("session", "claim", "video.mp4")
                self.assertIs(outcome.status, ProductionExecutionStatus.FAILURE)
                self.assertIsNone(outcome.result)
                self.assertEqual(builder.calls, [internal])
                self.assertIs(
                    outcome.failure.stage,
                    OperationalFailureStage.RESULT_PACKAGING,
                )
                self.assertIs(
                    outcome.failure.code,
                    OperationalFailureCode.RESULT_PACKAGING_FAILED,
                )
                self.assertEqual(
                    outcome.failure.exception_type,
                    type(error).__name__,
                )
                self.assertEqual(
                    outcome.failure.public_message,
                    RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE,
                )

    def test_runtime_failure_redacts_messages_paths_traceback_cause_and_stderr(self):
        cause = FileNotFoundError("/scr/user/private/model.safetensors")
        error = RuntimeError(
            "worker failed at /secret/server/runtime/cache/request.json; "
            "stderr=CUDA error from /home/user/private/model"
        )
        error.__cause__ = cause
        outcome = ProductionExecutionService(
            FakeRuntime(error=error),
            result_builder=FakeBuilder(result=self._production_result()),
        ).execute("session", "claim", "video.mp4")
        encoded = outcome.to_json()
        forbidden = (
            "/secret/server/",
            "/scr/user/",
            "/home/user/",
            "worker failed at",
            "stderr=",
            "CUDA error",
            "model.safetensors",
        )
        for secret in forbidden:
            with self.subTest(secret=secret):
                self.assertNotIn(secret, encoded)
        self.assertEqual(
            set(outcome.failure.to_dict()),
            {"stage", "code", "exception_type", "public_message"},
        )

    def test_packaging_failure_redacts_exception_text_and_paths(self):
        error = ValueError("malformed data from /home/user/private/file.json")
        outcome = ProductionExecutionService(
            FakeRuntime(outcome=object()),
            result_builder=FakeBuilder(error=error),
        ).execute("session", "claim", "video.mp4")
        encoded = outcome.to_json()
        self.assertNotIn("malformed data from", encoded)
        self.assertNotIn("/home/user/", encoded)

    def test_failure_payload_contains_no_scientific_result_or_nei_fields(self):
        outcome = ProductionExecutionService(
            FakeRuntime(error=RuntimeError("NOT_RUN NEI fake real")),
            result_builder=FakeBuilder(result=self._production_result()),
        ).execute("session", "claim", "video.mp4")
        payload = outcome.to_dict()
        keys = self._recursive_keys(payload)
        forbidden_keys = {
            "model_verdict",
            "display_verdict",
            "evidence_status",
            "sample_logits",
            "probabilities",
            "class_winners",
            "checkpoint_sha256",
            "sufficiency",
            "evidence",
        }
        self.assertTrue(forbidden_keys.isdisjoint(keys))
        self.assertNotIn("NEI", outcome.to_json())
        self.assertNotIn("not_run", outcome.to_json())

    def test_keyboard_interrupt_and_system_exit_are_not_caught(self):
        for control_exception in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(exception=type(control_exception).__name__):
                service = ProductionExecutionService(
                    FakeRuntime(error=control_exception),
                    result_builder=FakeBuilder(result=self._production_result()),
                )
                with self.assertRaises(type(control_exception)):
                    service.execute("session", "claim", "video.mp4")

    def test_packaging_base_exceptions_are_not_caught(self):
        for control_exception in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(exception=type(control_exception).__name__):
                service = ProductionExecutionService(
                    FakeRuntime(outcome=object()),
                    result_builder=FakeBuilder(error=control_exception),
                )
                with self.assertRaises(type(control_exception)):
                    service.execute("session", "claim", "video.mp4")

    def test_repeated_execute_reuses_exact_runtime_and_builder(self):
        runtime = FakeRuntime(outcome=object())
        builder = FakeBuilder(result=self._production_result())
        service = ProductionExecutionService(runtime, result_builder=builder)
        service.execute("session-a", "claim A", "a.mp4")
        service.execute("session-b", "claim B", "b.mp4")
        self.assertIs(service.runtime, runtime)
        self.assertIs(service.result_builder, builder)
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(len(builder.calls), 2)

    def test_execute_does_not_duplicate_runtime_request_validation(self):
        runtime = FakeRuntime(outcome=object())
        builder = FakeBuilder(result=self._production_result())
        outcome = ProductionExecutionService(
            runtime,
            result_builder=builder,
        ).execute("../unsafe", " ", "/missing/video.mp4")
        self.assertIs(outcome.status, ProductionExecutionStatus.SUCCESS)
        self.assertEqual(
            runtime.calls,
            [("../unsafe", " ", "/missing/video.mp4")],
        )

    def test_execute_calls_no_closed_runner_model_or_subprocess_directly(self):
        runtime = FakeRuntime(outcome=object())
        builder = FakeBuilder(result=self._production_result())
        service = ProductionExecutionService(runtime, result_builder=builder)
        with (
            patch.object(FrozenG1Runner, "run", side_effect=AssertionError),
            patch.object(VideoMultimodalRunner, "run", side_effect=AssertionError),
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
            patch.object(PaddleOCRService, "predict", side_effect=AssertionError),
            patch.object(subprocess, "run", side_effect=AssertionError),
        ):
            outcome = service.execute("session", "claim", "video.mp4")
        self.assertIs(outcome.status, ProductionExecutionStatus.SUCCESS)

    def test_execute_writes_no_files_and_does_not_mutate_result(self):
        public = self._production_result()
        before_payload = public.to_dict()
        service = ProductionExecutionService(
            FakeRuntime(outcome=object()),
            result_builder=FakeBuilder(result=public),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before_files = list(root.rglob("*"))
            outcome = service.execute("session", "claim", "video.mp4")
            after_files = list(root.rglob("*"))
        self.assertIs(outcome.result, public)
        self.assertEqual(public.to_dict(), before_payload)
        self.assertEqual(after_files, before_files)

    def test_execute_uses_no_exception_text_classification_or_score_threshold(self):
        source = inspect.getsource(ProductionExecutionService)
        self.assertNotIn("str(exc)", source)
        self.assertNotIn("isinstance(exc", source)
        self.assertNotIn("probabilities", source)
        self.assertNotIn("logits", source)

    def test_successful_nei_and_operational_failure_remain_distinct(self):
        nei_result = self._production_result(ModelVerdict.NOT_RUN)
        success = ProductionExecutionService(
            FakeRuntime(outcome=object()),
            result_builder=FakeBuilder(result=nei_result),
        ).execute("session", "claim", "video.mp4")
        failure = ProductionExecutionService(
            FakeRuntime(error=RuntimeError("failure")),
            result_builder=FakeBuilder(result=nei_result),
        ).execute("session", "claim", "video.mp4")
        self.assertIs(success.status, ProductionExecutionStatus.SUCCESS)
        self.assertIs(success.result.model_verdict, ModelVerdict.NOT_RUN)
        self.assertIsNone(success.failure)
        self.assertIs(failure.status, ProductionExecutionStatus.FAILURE)
        self.assertIsNone(failure.result)
        self.assertIsNotNone(failure.failure)

    def test_failure_serialization_is_deterministic(self):
        service = ProductionExecutionService(
            FakeRuntime(error=RuntimeError("secret changes nothing")),
            result_builder=FakeBuilder(result=self._production_result()),
        )
        first = service.execute("session", "claim", "video.mp4")
        second = service.execute("session", "claim", "video.mp4")
        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())


if __name__ == "__main__":
    unittest.main()
