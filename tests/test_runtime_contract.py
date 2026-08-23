import json
import unittest
from pathlib import Path

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    PipelineStage,
    RuntimeUnit,
    StageName,
    StageStatus,
    VerificationRequest,
    VerificationResult,
)


FIXTURES = Path(__file__).parent / "fixtures"


class RuntimeContractTests(unittest.TestCase):
    def test_request_json_round_trip(self):
        data = json.loads((FIXTURES / "mock_request.json").read_text(encoding="utf-8"))
        request = VerificationRequest.from_dict(data)
        encoded = json.dumps(request.to_dict(), sort_keys=True)
        restored = VerificationRequest.from_dict(json.loads(encoded))
        self.assertEqual(request.to_dict(), restored.to_dict())

    def test_unit_json_round_trip(self):
        data = json.loads((FIXTURES / "mock_units.json").read_text(encoding="utf-8"))
        units = [RuntimeUnit.from_dict(item) for item in data]
        encoded = json.dumps([unit.to_dict() for unit in units], sort_keys=True)
        restored = [RuntimeUnit.from_dict(item) for item in json.loads(encoded)]
        self.assertEqual([unit.to_dict() for unit in units], [unit.to_dict() for unit in restored])
        self.assertFalse(restored[1].eligible_for_frozen_g1)

    def test_visual_unit_cannot_be_frozen_g1_eligible(self):
        visual_data = json.loads((FIXTURES / "mock_units.json").read_text(encoding="utf-8"))[1]
        visual_data["eligible_for_frozen_g1"] = True
        with self.assertRaisesRegex(ValueError, "visual_observation"):
            RuntimeUnit.from_dict(visual_data)

    def test_result_json_round_trip_and_binary_classes(self):
        unit_data = json.loads((FIXTURES / "mock_units.json").read_text(encoding="utf-8"))[0]
        stage = PipelineStage(StageName.MOCK_G1, 0, StageStatus.COMPLETED)
        result = VerificationResult(
            session_id="round-trip",
            claim="claim",
            model_verdict=ModelVerdict.FAKE,
            display_verdict=DisplayVerdict.FAKE,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 1.0, "real": -1.0},
            probabilities={"fake": 0.88, "real": 0.12},
            all_units=[RuntimeUnit.from_dict(unit_data)],
            top_k_units=[RuntimeUnit.from_dict(unit_data)],
            class_winners={"fake": "text-000", "real": "text-000"},
            pipeline_stages=[stage],
            warnings=["MOCK_NON_SCIENTIFIC_OUTPUT"],
        )
        payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))
        restored = VerificationResult.from_dict(payload)
        self.assertEqual(result.to_dict(), restored.to_dict())
        self.assertEqual({"fake", "real"}, set(restored.probabilities))
        self.assertNotIn("NEI", restored.probabilities)

    def test_all_verdict_states_serialize_with_explicit_display_mapping(self):
        cases = (
            (
                ModelVerdict.FAKE,
                DisplayVerdict.FAKE,
                EvidenceStatus.SUFFICIENT,
                {"fake": 1.0, "real": 0.0},
                {"fake": 0.75, "real": 0.25},
            ),
            (
                ModelVerdict.REAL,
                DisplayVerdict.REAL,
                EvidenceStatus.SUFFICIENT,
                {"fake": 0.0, "real": 1.0},
                {"fake": 0.25, "real": 0.75},
            ),
            (
                ModelVerdict.NOT_RUN,
                DisplayVerdict.NEI,
                EvidenceStatus.INSUFFICIENT,
                {},
                {},
            ),
        )
        for model_verdict, display_verdict, evidence_status, logits, probabilities in cases:
            with self.subTest(model_verdict=model_verdict):
                result = VerificationResult(
                    session_id="verdict-serialization",
                    claim="claim",
                    model_verdict=model_verdict,
                    display_verdict=display_verdict,
                    evidence_status=evidence_status,
                    sample_logits=logits,
                    probabilities=probabilities,
                    all_units=[],
                    top_k_units=[],
                    class_winners={},
                    pipeline_stages=[],
                )
                payload = result.to_dict()
                self.assertEqual(model_verdict.value, payload["model_verdict"])
                self.assertEqual(display_verdict.value, payload["display_verdict"])
                self.assertEqual(result, VerificationResult.from_dict(payload))

    def test_mock_result_fixture_json_round_trip(self):
        data = json.loads((FIXTURES / "mock_result.json").read_text(encoding="utf-8"))
        result = VerificationResult.from_dict(data)
        encoded = json.dumps(result.to_dict(), sort_keys=True)
        restored = VerificationResult.from_dict(json.loads(encoded))
        self.assertEqual(result.to_dict(), restored.to_dict())

    def test_binary_result_invariants(self):
        base = {
            "session_id": "invariant",
            "claim": "claim",
            "model_verdict": ModelVerdict.FAKE,
            "display_verdict": DisplayVerdict.FAKE,
            "evidence_status": EvidenceStatus.SUFFICIENT,
            "sample_logits": {"fake": 1.0, "real": 0.0},
            "probabilities": {"fake": 0.75, "real": 0.25},
            "all_units": [],
            "top_k_units": [],
            "class_winners": {},
            "pipeline_stages": [],
        }
        invalid_overrides = [
            {"sample_logits": {"fake": 1.0, "real": 0.0, "NEI": 2.0}},
            {"probabilities": {"fake": 0.75}},
            {"display_verdict": DisplayVerdict.REAL},
            {
                "model_verdict": ModelVerdict.NOT_RUN,
                "display_verdict": DisplayVerdict.NEI,
                "evidence_status": EvidenceStatus.INSUFFICIENT,
                "sample_logits": {"fake": 1.0, "real": 0.0},
                "probabilities": {},
            },
            {
                "model_verdict": ModelVerdict.NOT_RUN,
                "display_verdict": DisplayVerdict.NEI,
                "evidence_status": EvidenceStatus.SUFFICIENT,
                "sample_logits": {},
                "probabilities": {},
            },
            {
                "model_verdict": ModelVerdict.NOT_RUN,
                "display_verdict": DisplayVerdict.FAKE,
                "evidence_status": EvidenceStatus.INSUFFICIENT,
                "sample_logits": {},
                "probabilities": {},
            },
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                VerificationResult(**dict(base, **overrides))

    def test_stage_transition_contract(self):
        stage = PipelineStage(StageName.REQUEST, 0)
        stage.transition(StageStatus.RUNNING)
        stage.transition(StageStatus.COMPLETED)
        with self.assertRaises(ValueError):
            stage.transition(StageStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
