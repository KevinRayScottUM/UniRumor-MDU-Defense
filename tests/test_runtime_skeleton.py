import json
import tempfile
import unittest
from pathlib import Path

from pipeline import RuntimeConfig, RuntimeOrchestrator
from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict, SourceType, VerificationRequest
from services.mock_models import aggregate_all_evaluated


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_request() -> VerificationRequest:
    return VerificationRequest.from_dict(
        json.loads((FIXTURES / "mock_request.json").read_text(encoding="utf-8"))
    )


class RuntimeSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.config = RuntimeConfig(self.base / "cache", self.base / "outputs")

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self):
        orchestrator = RuntimeOrchestrator(self.config)
        result = orchestrator.run(fixture_request())
        return orchestrator, result

    def test_stage_transitions_complete_in_contract_order(self):
        _, result = self.run_fixture()
        self.assertEqual(list(range(9)), [stage.sequence for stage in result.pipeline_stages])
        self.assertEqual([name.value for name in __import__("schemas").StageName], [s.name.value for s in result.pipeline_stages])
        self.assertTrue(all(stage.status.value == "completed" for stage in result.pipeline_stages))

    def test_all_nine_stages_emit_completion_logs(self):
        _, result = self.run_fixture()
        log_path = self.config.output_root / "logs" / f"{result.session_id}.jsonl"
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        completions = [record for record in records if record["event"] == "stage_completed"]
        self.assertEqual(9, len(completions))
        self.assertEqual(
            [stage.name.value for stage in result.pipeline_stages],
            [record["payload"]["stage"] for record in completions],
        )

    def test_repeated_run_is_deterministic(self):
        _, first = self.run_fixture()
        _, second = self.run_fixture()
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_asr_ocr_and_visual_units_keep_visual_excluded(self):
        _, result = self.run_fixture()
        source_types = {unit.source_type for unit in result.all_units}
        self.assertTrue({SourceType.TRANSCRIPT, SourceType.OCR, SourceType.VISUAL_OBSERVATION} <= source_types)
        visual = [unit for unit in result.all_units if unit.source_type is SourceType.VISUAL_OBSERVATION]
        self.assertTrue(visual)
        self.assertTrue(all(not unit.eligible_for_frozen_g1 and unit.logits is None for unit in visual))

    def test_sample_logits_are_all_evaluated_unit_maxima(self):
        _, result = self.run_fixture()
        evaluated = [unit for unit in result.all_units if unit.logits is not None]
        for label in ("fake", "real"):
            self.assertEqual(max(unit.logits[label] for unit in evaluated), result.sample_logits[label])
            self.assertEqual(
                max(evaluated, key=lambda unit: (unit.logits[label], unit.unit_id)).unit_id,
                result.class_winners[label],
            )

    def test_top_k_is_explanation_only(self):
        request = fixture_request()
        request.claim = "top k independence claim 0"
        request.transcript_segments = [{"text": f"deterministic unit {index}"} for index in range(24)]
        request.ocr_observations = []
        request.visual_inputs = []
        result = RuntimeOrchestrator(self.config).run(request)
        top_ids = {unit.unit_id for unit in result.top_k_units}
        predicted_winner = result.class_winners[result.model_verdict.value]
        self.assertNotIn(predicted_winner, top_ids)
        self.assertEqual(5, len(result.top_k_units))
        self.assertEqual(
            max(unit.logits[result.model_verdict.value] for unit in result.all_units),
            result.sample_logits[result.model_verdict.value],
        )

    def test_aggregate_does_not_use_top_k_for_prediction(self):
        request = fixture_request()
        request.transcript_segments = [{"text": f"aggregation unit {index}"} for index in range(8)]
        request.ocr_observations = []
        request.visual_inputs = []
        result = RuntimeOrchestrator(self.config).run(request)
        evaluated = [unit for unit in result.all_units if unit.logits is not None]
        for index, unit in enumerate(evaluated):
            unit.selection_score = 1.0 - index / 100.0
            unit.logits = {"fake": 0.0, "real": 0.0}
        evaluated[-1].logits = {"fake": 9.0, "real": -9.0}
        pooled = aggregate_all_evaluated(evaluated, 5)
        self.assertEqual(ModelVerdict.FAKE, pooled.model_verdict)
        self.assertNotIn(evaluated[-1].unit_id, {unit.unit_id for unit in pooled.top_k_units})

    def test_only_first_24_eligible_units_are_evaluated(self):
        request = fixture_request()
        request.transcript_segments = [{"text": f"unit {index}"} for index in range(30)]
        request.ocr_observations = []
        request.visual_inputs = []
        result = RuntimeOrchestrator(self.config).run(request)
        self.assertEqual(24, sum(unit.logits is not None for unit in result.all_units))
        self.assertTrue(all(unit.logits is None for unit in result.all_units[24:]))

    def test_zero_eligible_units_uses_engineering_nei_path(self):
        request = VerificationRequest(
            claim="visual only",
            request_id="nei-session",
            visual_inputs=[{"frame_id": "f", "observation": "visual content"}],
        )
        result = RuntimeOrchestrator(self.config).run(request)
        self.assertEqual(ModelVerdict.NOT_RUN, result.model_verdict)
        self.assertEqual(EvidenceStatus.INSUFFICIENT, result.evidence_status)
        self.assertEqual(DisplayVerdict.NEI, result.display_verdict)
        self.assertEqual({}, result.sample_logits)
        self.assertEqual({}, result.probabilities)
        self.assertNotIn("NEI", result.probabilities)

    def test_cache_log_and_result_files_are_created(self):
        orchestrator, result = self.run_fixture()
        self.assertTrue((self.config.cache_root / "sessions" / f"{result.session_id}.json").is_file())
        self.assertTrue((self.config.cache_root / "units" / f"{result.session_id}.json").is_file())
        self.assertTrue((self.config.output_root / "logs" / f"{result.session_id}.jsonl").is_file())
        self.assertTrue(orchestrator.last_result_path.is_file())
        saved = json.loads(orchestrator.last_result_path.read_text(encoding="utf-8"))
        self.assertEqual(result.to_dict(), saved)

    def test_runtime_creates_nothing_outside_configured_roots(self):
        untouched = self.base / "sentinel.txt"
        untouched.write_text("unchanged", encoding="utf-8")
        before = {path.relative_to(self.base) for path in self.base.rglob("*")}
        self.run_fixture()
        after = {path.relative_to(self.base) for path in self.base.rglob("*")}
        created = after - before
        self.assertTrue(created)
        self.assertTrue(all(parts.parts[0] in {"cache", "outputs"} for parts in created))
        self.assertEqual("unchanged", untouched.read_text(encoding="utf-8"))

    def test_warning_and_checkpoint_boundary(self):
        orchestrator, result = self.run_fixture()
        self.assertIn("MOCK_NON_SCIENTIFIC_OUTPUT", result.warnings)
        self.assertIsNone(result.checkpoint_sha256)
        self.assertEqual("forbidden", orchestrator.registry.get("g1").checkpoint_access)


if __name__ == "__main__":
    unittest.main()
