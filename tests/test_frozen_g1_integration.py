import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.phase4a_request_adapter import build_phase4a_request
from adapters.phase4a_response_adapter import (
    FROZEN_CHECKPOINT_SHA256,
    parse_phase4a_prediction,
)
from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict, RuntimeUnit, SourceType
from services.frozen_g1_runner import FrozenG1Runner, FrozenG1RunnerConfig


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_prediction():
    return json.loads((FIXTURES / "phase4a_prediction.json").read_text(encoding="utf-8"))


def fixture_units():
    return [
        RuntimeUnit("text-unit", SourceType.TEXT, "  claim text  ", eligible_for_frozen_g1=True),
        RuntimeUnit(
            "transcript-unit", SourceType.TRANSCRIPT, "spoken words", eligible_for_frozen_g1=True
        ),
        RuntimeUnit("visual-unit", SourceType.VISUAL_OBSERVATION, "visible scene"),
        RuntimeUnit("ocr-unit", SourceType.OCR, "frame words", eligible_for_frozen_g1=True),
        RuntimeUnit("ineligible-text", SourceType.TEXT, "supplement", eligible_for_frozen_g1=False),
    ]


def prediction_for(unit_ids, input_unit_count=None):
    outputs = []
    for index, unit_id in enumerate(unit_ids):
        outputs.append(
            {
                "unit_id": unit_id,
                "modality": "text",
                "selection_score": 1.0 - index / 100.0,
                "veracity_logits": {"fake": float(index), "real": float(100 - index)},
            }
        )
    return {
        "case_id": "phase4a-session",
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "claim": "the claim",
        "variant": "G1_text_ocr",
        "contract_version": "1.0.0",
        "pooling": "max",
        "maximum_units_per_sample": 24,
        "max_length": 256,
        "selection_head_affects_sample_pooling": False,
        "topk_is_only_prediction_basis": False,
        "prediction": "real",
        "prediction_id": 1,
        "sample_logits": {"fake": float(len(unit_ids) - 1), "real": 100.0},
        "probabilities": {"fake": 0.1, "real": 0.9},
        "dropped_visual_unit_count": 0,
        "input_unit_count_before_contract": input_unit_count or len(outputs),
        "model_exposed_unit_count": len(outputs),
        "truncated_unit_count": (input_unit_count or len(outputs)) - len(outputs),
        "unit_outputs": outputs,
        "top_k_selection_units": [
            {"unit_id": unit_id, "selection_score": outputs[index]["selection_score"]}
            for index, unit_id in enumerate(unit_ids[:5])
        ],
        "max_pool_winner_by_class": {
            "fake": dict(outputs[-1]),
            "real": dict(outputs[0]),
        },
    }


class SubprocessStub:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(json.dumps(self.prediction) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class FrozenG1IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.config = FrozenG1RunnerConfig(
            unirumor_root=self.base / "external-unirumor",
            python_executable=str(self.base / "venv" / "bin" / "python"),
            phase4a_infer=self.base / "external-unirumor" / "phase4a_infer.py",
            phase4a_config=self.base / "external-unirumor" / "phase4a.json",
            device="cuda:0",
            timeout_seconds=45,
            cache_root=self.base / "cache",
            output_root=self.base / "outputs",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, units=None, prediction=None):
        stub = SubprocessStub(prediction or fixture_prediction())
        result = FrozenG1Runner(self.config, subprocess_run=stub).run(
            "phase4a-session", "the claim", units or fixture_units()
        )
        return result, stub

    def parse_fixture(self, payload=None):
        return parse_phase4a_prediction(
            payload or fixture_prediction(),
            ["text-unit", "transcript-unit", "ocr-unit"],
            expected_case_id="phase4a-session",
            expected_claim="the claim",
        )

    def test_exact_candidate_mapping_and_preserved_ids(self):
        request = build_phase4a_request("sample-1", "claim", fixture_units())
        self.assertEqual("sample-1", request["case_id"])
        self.assertEqual("external", request["dataset"])
        self.assertEqual("claim", request["claim"])
        self.assertEqual(
            [
                {
                    "unit_id": "text-unit",
                    "unit_type": "text",
                    "modality": "text",
                    "text": "  claim text  ",
                },
                {
                    "unit_id": "transcript-unit",
                    "unit_type": "transcript",
                    "modality": "text",
                    "text": "spoken words",
                },
                {
                    "unit_id": "ocr-unit",
                    "unit_type": "ocr",
                    "modality": "ocr",
                    "text": "frame words",
                },
            ],
            request["candidate_units"],
        )
        self.assertNotIn("transcript", request)
        self.assertNotIn("ocr_text", request)
        self.assertNotIn("sample_id", request)

    def test_visual_and_ineligible_units_are_excluded(self):
        request = build_phase4a_request("sample-1", "claim", fixture_units())
        ids = [item["unit_id"] for item in request["candidate_units"]]
        self.assertNotIn("visual-unit", ids)
        self.assertNotIn("ineligible-text", ids)

    def test_adapter_does_not_truncate_to_24(self):
        units = [
            RuntimeUnit(f"unit-{index}", SourceType.TEXT, str(index), eligible_for_frozen_g1=True)
            for index in range(30)
        ]
        request = build_phase4a_request("sample-1", "claim", units)
        self.assertEqual(30, len(request["candidate_units"]))
        self.assertEqual([unit.unit_id for unit in units], [item["unit_id"] for item in request["candidate_units"]])

    def test_adapter_rejects_blank_text_and_duplicate_candidate_ids(self):
        blank = RuntimeUnit("blank", SourceType.TEXT, " \t", eligible_for_frozen_g1=True)
        with self.assertRaisesRegex(ValueError, "blank text"):
            build_phase4a_request("sample", "claim", [blank])
        duplicates = [
            RuntimeUnit("same", SourceType.TEXT, "one", eligible_for_frozen_g1=True),
            RuntimeUnit("same", SourceType.OCR, "two", eligible_for_frozen_g1=True),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_phase4a_request("sample", "claim", duplicates)

    def test_runner_rejects_duplicate_ids_across_eligible_and_visual_units(self):
        units = [
            RuntimeUnit("shared", SourceType.TEXT, "eligible", eligible_for_frozen_g1=True),
            RuntimeUnit("shared", SourceType.VISUAL_OBSERVATION, "excluded visual"),
        ]

        def fail_if_called(*args, **kwargs):
            raise AssertionError("subprocess must not run")

        with self.assertRaisesRegex(ValueError, "duplicate RuntimeUnit ID"):
            FrozenG1Runner(self.config, subprocess_run=fail_if_called).run(
                "duplicate-session", "claim", units
            )

    def test_zero_eligible_returns_nei_without_subprocess(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("subprocess must not run")

        units = [RuntimeUnit("visual", SourceType.VISUAL_OBSERVATION, "scene")]
        result = FrozenG1Runner(self.config, subprocess_run=fail_if_called).run(
            "zero-session", "claim", units
        )
        self.assertEqual(ModelVerdict.NOT_RUN, result.model_verdict)
        self.assertEqual(EvidenceStatus.INSUFFICIENT, result.evidence_status)
        self.assertEqual(DisplayVerdict.NEI, result.display_verdict)
        self.assertEqual({}, result.sample_logits)
        self.assertEqual({}, result.probabilities)
        self.assertIs(units[0], result.all_units[0])

    def test_zero_eligible_clears_stale_scores_from_returned_units(self):
        units = [
            RuntimeUnit(
                "visual",
                SourceType.VISUAL_OBSERVATION,
                "scene",
                selection_score=0.9,
                logits={"fake": 2.0, "real": 1.0},
            ),
            RuntimeUnit(
                "ineligible",
                SourceType.TEXT,
                "supplement",
                selection_score=0.8,
                logits={"fake": 1.0, "real": 2.0},
            ),
        ]
        result = FrozenG1Runner(
            self.config,
            subprocess_run=lambda *args, **kwargs: self.fail("subprocess must not run"),
        ).run("stale-nei", "claim", units)
        self.assertEqual(ModelVerdict.NOT_RUN, result.model_verdict)
        self.assertTrue(
            all(unit.selection_score is None and unit.logits is None for unit in result.all_units)
        )

    def test_subprocess_command_uses_argument_list_without_shell(self):
        _, stub = self.run_fixture()
        self.assertEqual(1, len(stub.calls))
        command, kwargs = stub.calls[0]
        self.assertIsInstance(command, list)
        self.assertEqual(
            [
                self.config.python_executable,
                "-u",
                str(self.config.phase4a_infer),
                "--config",
                str(self.config.phase4a_config),
                "--project-root",
                str(self.config.unirumor_root),
                "--input",
                str(self.config.cache_root / "phase4a_requests" / "phase4a-session.jsonl"),
                "--output",
                str(self.config.output_root / "phase4a_predictions" / "phase4a-session.jsonl"),
                "--device",
                "cuda:0",
            ],
            command,
        )
        self.assertIs(False, kwargs["shell"])
        self.assertNotIn("--drop-unsupported-visual", command)
        self.assertEqual(45, kwargs["timeout"])

    def test_stale_prediction_is_removed_before_subprocess(self):
        prediction_path = self.config.output_root / "phase4a_predictions" / "phase4a-session.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(fixture_prediction()) + "\n", encoding="utf-8")

        def successful_without_output(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with self.assertRaisesRegex(ValueError, "was not created"):
            FrozenG1Runner(self.config, subprocess_run=successful_without_output).run(
                "phase4a-session", "the claim", fixture_units()
            )
        self.assertFalse(prediction_path.exists())

    def test_valid_response_contract_is_accepted(self):
        parsed = self.parse_fixture()
        self.assertEqual("real", parsed.prediction)
        self.assertEqual({"fake": "text-unit", "real": "ocr-unit"}, parsed.class_winners)

    def test_golden_probe_winner_shape_without_standalone_logit_is_accepted(self):
        payload = fixture_prediction()
        for winner in payload["max_pool_winner_by_class"].values():
            self.assertEqual(
                {"unit_id", "modality", "selection_score", "veracity_logits"}, set(winner)
            )
            self.assertNotIn("logit", winner)
        parsed = self.parse_fixture(payload)
        self.assertEqual({"fake": "text-unit", "real": "ocr-unit"}, parsed.class_winners)

    def test_contract_field_violations_are_rejected(self):
        candidate_ids = ["text-unit", "transcript-unit", "ocr-unit"]
        violations = [
            ("variant", "wrong"),
            ("contract_version", "2.0.0"),
            ("maximum_units_per_sample", 25),
            ("max_length", 512),
            ("dropped_visual_unit_count", 1),
            ("prediction", "NEI"),
            ("input_unit_count_before_contract", 2),
            ("model_exposed_unit_count", 2),
            ("truncated_unit_count", 1),
        ]
        for field, value in violations:
            payload = fixture_prediction()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_phase4a_prediction(
                    payload, candidate_ids, "phase4a-session", "the claim"
                )

    def test_response_provenance_mismatches_are_rejected(self):
        for field, value, message in (
            ("case_id", "wrong-case", "case_id"),
            ("claim", "wrong claim", "claim"),
        ):
            payload = fixture_prediction()
            payload[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.parse_fixture(payload)

    def test_wrong_checkpoint_is_rejected(self):
        payload = fixture_prediction()
        payload["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            self.parse_fixture(payload)

    def test_wrong_pooling_and_prediction_flags_are_rejected(self):
        for field, value in (
            ("pooling", "mean"),
            ("selection_head_affects_sample_pooling", True),
            ("topk_is_only_prediction_basis", True),
        ):
            payload = fixture_prediction()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse_fixture(payload)

    def test_all_unit_class_wise_max_is_recomputed(self):
        payload = fixture_prediction()
        payload["unit_outputs"][1]["veracity_logits"]["fake"] = 9.0
        with self.assertRaisesRegex(ValueError, "class-wise maxima"):
            self.parse_fixture(payload)

    def test_phase4a_accounting_counts_must_be_nonnegative_integers(self):
        candidate_ids = ["text-unit", "transcript-unit", "ocr-unit"]
        for field, value in (
            ("input_unit_count_before_contract", -1),
            ("model_exposed_unit_count", -1),
            ("truncated_unit_count", -1),
            ("input_unit_count_before_contract", 3.0),
        ):
            payload = fixture_prediction()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                ValueError, "nonnegative integer"
            ):
                parse_phase4a_prediction(
                    payload, candidate_ids, "phase4a-session", "the claim"
                )

    def test_class_winner_id_must_identify_a_unit_with_the_max_logit(self):
        payload = fixture_prediction()
        payload["max_pool_winner_by_class"]["fake"]["unit_id"] = "transcript-unit"
        with self.assertRaisesRegex(ValueError, "class-winner unit"):
            self.parse_fixture(payload)

    def test_response_maps_to_verification_result_and_units(self):
        units = fixture_units()
        result, _ = self.run_fixture(units=units)
        self.assertEqual(ModelVerdict.REAL, result.model_verdict)
        self.assertEqual(DisplayVerdict.REAL, result.display_verdict)
        self.assertEqual(EvidenceStatus.SUFFICIENT, result.evidence_status)
        self.assertEqual({"fake": 2.0, "real": 2.5}, result.sample_logits)
        self.assertEqual(FROZEN_CHECKPOINT_SHA256, result.checkpoint_sha256)
        self.assertEqual([unit.unit_id for unit in units], [unit.unit_id for unit in result.all_units])
        by_id = {unit.unit_id: unit for unit in result.all_units}
        self.assertEqual(0.9, by_id["text-unit"].selection_score)
        self.assertEqual({"fake": 2.0, "real": 0.1}, by_id["text-unit"].logits)
        self.assertNotIn("MOCK_NON_SCIENTIFIC_OUTPUT", result.warnings)

    def test_top_k_explanation_mapping_preserves_response_order(self):
        result, _ = self.run_fixture()
        self.assertEqual(
            ["text-unit", "transcript-unit", "ocr-unit"],
            [unit.unit_id for unit in result.top_k_units],
        )
        self.assertEqual({"fake": "text-unit", "real": "ocr-unit"}, result.class_winners)

    def test_visual_and_officially_truncated_units_remain_unscored(self):
        eligible = [
            RuntimeUnit(f"unit-{index:02d}", SourceType.TEXT, str(index), eligible_for_frozen_g1=True)
            for index in range(25)
        ]
        visual = RuntimeUnit("visual", SourceType.VISUAL_OBSERVATION, "scene")
        units = eligible + [visual]
        exposed_ids = [unit.unit_id for unit in eligible[:24]]
        result, _ = self.run_fixture(
            units=units, prediction=prediction_for(exposed_ids, input_unit_count=25)
        )
        self.assertEqual([unit.unit_id for unit in units], [unit.unit_id for unit in result.all_units])
        self.assertTrue(all(unit.logits is not None for unit in result.all_units[:24]))
        self.assertIsNone(result.all_units[24].selection_score)
        self.assertIsNone(result.all_units[24].logits)
        self.assertIsNone(result.all_units[25].selection_score)
        self.assertIsNone(result.all_units[25].logits)
        self.assertIn("official Phase4A truncation occurred", result.warnings)
        self.assertIn("supplemental visual units excluded from Frozen G1", result.warnings)


if __name__ == "__main__":
    unittest.main()
