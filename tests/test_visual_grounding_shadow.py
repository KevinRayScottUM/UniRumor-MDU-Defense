import ast
from pathlib import Path
import unittest
from unittest import mock

import services.visual_grounding_shadow as shadow_module
from adapters.visual_grounding_adapter import VisualGroundingAdapter
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    GroundedVisualUnit,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
    VerificationResult,
    VisualObservationSnapshot,
)
from services.claim_consistency_gate import ConsistencyResult
from services.production_result import ProductionResultBuilder
from services.video_multimodal_runner import VideoMultimodalRunner
from services.visual_grounding_shadow import (
    VISUAL_GROUNDING_SHADOW_FAILURE_WARNING,
    VisualGroundingShadowRunner,
)


class VisualGroundingShadowTests(unittest.TestCase):
    @staticmethod
    def visual_unit():
        return RuntimeUnit(
            unit_id="visual_0123456789abcdef0123",
            source_type=SourceType.VISUAL_OBSERVATION,
            text="players are visible on a basketball court",
            start_time=1.5,
            end_time=3.5,
            frame_id="F001",
            frame_path="/private/frames/F001.jpg",
            bbox=None,
            confidence=None,
            producer="Qwen/Qwen2.5-VL-7B-Instruct",
            provenance=UnitProvenance(
                source_uri="/private/uploads/video.mp4",
                source_index=0,
                extraction_method="qwen_claim_blind_visual_observer",
                details={
                    "observation_type": "scene",
                    "frame_ids": ["F001", "F002"],
                    "evidence_refs": ["F001", "F002"],
                    "referenced_frames": [
                        {
                            "frame_id": "F001",
                            "frame_path": "/private/frames/F001.jpg",
                            "frame_index": 10,
                            "timestamp_sec": 1.5,
                            "frame_rank": 1,
                            "image_sha256": "a" * 64,
                            "retrieval_score": 0.8,
                            "retrieval_rank": 1,
                        },
                        {
                            "frame_id": "F002",
                            "frame_path": "/private/frames/F002.jpg",
                            "frame_index": 20,
                            "timestamp_sec": 3.5,
                            "frame_rank": 2,
                            "image_sha256": "b" * 64,
                            "retrieval_score": 0.7,
                            "retrieval_rank": 2,
                        },
                    ],
                    "siglip_model_id": "google/siglip-so400m-patch14-384",
                    "siglip_revision": "siglip-revision",
                    "qwen_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "qwen_revision": "qwen-revision",
                    "prompt_policy": "claim_blind_visible_atomic_facts_no_ocr_no_inference",
                    "recovery_mode": "canonical_object",
                    "raw_generation_sha256": "c" * 64,
                },
            ),
            eligible_for_frozen_g1=False,
            selection_score=None,
            logits=None,
        )

    @staticmethod
    def text_unit():
        return RuntimeUnit(
            unit_id="text-1",
            source_type=SourceType.TEXT,
            text="The video shows a basketball game.",
            eligible_for_frozen_g1=True,
        )

    @staticmethod
    def verification(all_units):
        return VerificationResult(
            session_id="session",
            claim="The video shows a basketball game.",
            model_verdict=ModelVerdict.REAL,
            display_verdict=DisplayVerdict.REAL,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 0.1, "real": 0.9},
            probabilities={"fake": 0.2, "real": 0.8},
            all_units=list(all_units),
            top_k_units=[],
            class_winners={},
            pipeline_stages=[],
            warnings=[],
        )

    def build_runner(self, shadow_runner=None):
        text = self.text_unit()
        visual = self.visual_unit()
        text_result = mock.Mock()
        text_result.g1_exposure_units = [text]
        text_result.warnings = []
        text_result.to_dict.return_value = {"kind": "text_ocr"}
        visual_result = mock.Mock()
        visual_result.runtime_units = [visual]
        visual_result.warnings = []
        visual_result.to_dict.return_value = {"kind": "visual"}
        text_runner = mock.Mock()
        text_runner.run.return_value = text_result
        visual_runner = mock.Mock()
        visual_runner.run.return_value = visual_result
        frozen = mock.Mock()
        expected = self.verification([text, visual])
        frozen.run.return_value = expected
        gate = mock.Mock()
        gate.evaluate.return_value = ConsistencyResult.UNKNOWN
        runner = VideoMultimodalRunner(
            text_runner,
            visual_runner,
            frozen,
            claim_consistency_gate=gate,
            visual_grounding_shadow_runner=shadow_runner,
        )
        return runner, frozen, expected, text, visual

    def test_shadow_runner_wraps_adapter_independently(self):
        source = self.visual_unit()
        snapshot = VisualObservationSnapshot.from_runtime_unit(source)
        expected = VisualGroundingAdapter().ground([snapshot])
        adapter = mock.Mock()
        adapter.ground.return_value = expected
        runner = VisualGroundingShadowRunner(adapter)

        actual = runner.run([snapshot])

        adapter.ground.assert_called_once_with([snapshot])
        self.assertEqual(expected, actual)
        self.assertTrue(all(isinstance(unit, GroundedVisualUnit) for unit in actual))

    def test_visual_observations_create_shadow_units(self):
        runner, frozen, expected, text, visual = self.build_runner()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertIs(expected, result.verification_result)
        self.assertEqual(1, len(result.visual_grounding_shadow_units))
        shadow = result.visual_grounding_shadow_units[0]
        self.assertIsInstance(shadow, GroundedVisualUnit)
        self.assertEqual(visual.unit_id, shadow.source_observation_id)
        self.assertEqual(visual.text, shadow.text_observation)
        frozen.run.assert_called_once_with(
            "session",
            "The video shows a basketball game.",
            [text, visual],
        )

    def test_shadow_units_never_enter_authoritative_runtime_lists(self):
        runner, frozen, _, text, visual = self.build_runner()
        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertEqual([text], result.g1_exposure_units)
        self.assertEqual([visual], result.visual_units)
        self.assertEqual([text, visual], result.all_runtime_units)
        self.assertEqual(2, len(result.all_runtime_units))
        self.assertTrue(all(isinstance(unit, RuntimeUnit) for unit in result.all_runtime_units))
        self.assertFalse(
            any(
                isinstance(unit, GroundedVisualUnit)
                for unit in frozen.run.call_args.args[2]
            )
        )
        self.assertFalse(
            any(
                isinstance(unit, GroundedVisualUnit)
                for unit in result.g1_exposure_units
            )
        )

    def test_existing_verdict_and_frozen_invocation_are_unchanged(self):
        runner, frozen, expected, text, visual = self.build_runner()
        expected_verdict = expected.to_dict()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertIs(expected, result.verification_result)
        self.assertEqual(expected_verdict, result.verification_result.to_dict())
        frozen.run.assert_called_once_with(
            "session",
            "The video shows a basketball game.",
            [text, visual],
        )

    def test_shadow_serialization_is_deterministic_and_separate(self):
        runner, _, _, _, _ = self.build_runner()
        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )
        first = result.to_dict()["visual_grounding_shadow_units"]
        second = result.to_dict()["visual_grounding_shadow_units"]
        self.assertEqual(first, second)
        self.assertEqual(
            result.visual_grounding_shadow_units[0].to_dict(), first[0]
        )
        self.assertNotIn("visual_grounding_shadow_units", result.verification_result.to_dict())

    def test_adapter_failure_does_not_change_verification(self):
        adapter = mock.Mock()
        adapter.ground.side_effect = ValueError("private internal grounding detail")
        shadow_runner = VisualGroundingShadowRunner(adapter)
        runner, frozen, expected, text, visual = self.build_runner(shadow_runner)
        expected_verdict = expected.to_dict()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertIs(expected, result.verification_result)
        self.assertEqual(expected_verdict, result.verification_result.to_dict())
        self.assertEqual([], result.visual_grounding_shadow_units)
        self.assertIn(VISUAL_GROUNDING_SHADOW_FAILURE_WARNING, result.warnings)
        self.assertNotIn("private internal grounding detail", result.warnings)
        frozen.run.assert_called_once_with(
            "session",
            "The video shows a basketball game.",
            [text, visual],
        )

    def test_mutating_snapshot_then_raising_cannot_change_authoritative_state(self):
        class MutatingFailureAdapter:
            def ground(self, snapshots):
                object.__setattr__(snapshots[0], "observation_text", "tampered")
                object.__setattr__(
                    snapshots[0].frame_references[0], "frame_id", "tampered-frame"
                )
                raise RuntimeError("adversarial shadow failure")

        shadow_runner = VisualGroundingShadowRunner(MutatingFailureAdapter())
        runner, frozen, expected, text, visual = self.build_runner(shadow_runner)
        visual_before = visual.to_dict()
        verification_before = expected.to_dict()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertEqual(visual_before, visual.to_dict())
        self.assertEqual(verification_before, result.verification_result.to_dict())
        self.assertEqual([], result.visual_grounding_shadow_units)
        self.assertIn(VISUAL_GROUNDING_SHADOW_FAILURE_WARNING, result.warnings)
        production_result = ProductionResultBuilder().build(result)
        self.assertEqual(DisplayVerdict.REAL, production_result.display_verdict)
        frozen.run.assert_called_once_with(
            "session",
            "The video shows a basketball game.",
            [text, visual],
        )

    def test_mutating_snapshot_and_returning_artifact_leaves_units_unchanged(self):
        class MutatingSuccessAdapter:
            def ground(self, snapshots):
                grounded = VisualGroundingAdapter().ground(snapshots)
                object.__setattr__(snapshots[0], "unit_id", "tampered-unit")
                object.__setattr__(snapshots[0], "observation_text", "tampered")
                return grounded

        shadow_runner = VisualGroundingShadowRunner(MutatingSuccessAdapter())
        runner, _, expected, _, visual = self.build_runner(shadow_runner)
        visual_before = visual.to_dict()
        verification_before = expected.to_dict()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertEqual(visual_before, visual.to_dict())
        self.assertEqual(verification_before, result.verification_result.to_dict())
        self.assertEqual(1, len(result.visual_grounding_shadow_units))
        self.assertEqual(
            visual.unit_id,
            result.visual_grounding_shadow_units[0].source_observation_id,
        )

    def test_nested_provenance_has_no_shared_references(self):
        class NestedMutationAdapter:
            def __init__(self):
                self.snapshot = None

            def ground(self, snapshots):
                self.snapshot = snapshots[0]
                grounded = VisualGroundingAdapter().ground(snapshots)
                object.__setattr__(
                    snapshots[0].frame_references[0], "frame_id", "tampered-frame"
                )
                object.__setattr__(snapshots[0], "frame_ids", ("tampered-frame",))
                return grounded

        adapter = NestedMutationAdapter()
        shadow_runner = VisualGroundingShadowRunner(adapter)
        runner, _, _, _, visual = self.build_runner(shadow_runner)
        provenance_before = visual.provenance.to_dict()

        result = runner.run(
            "session", "The video shows a basketball game.", Path("video.mp4")
        )

        self.assertEqual(provenance_before, visual.provenance.to_dict())
        self.assertIsNot(
            adapter.snapshot.frame_references,
            visual.provenance.details["referenced_frames"],
        )
        self.assertIsNot(
            adapter.snapshot.frame_references[0],
            visual.provenance.details["referenced_frames"][0],
        )
        self.assertEqual(
            ["F001", "F002"], visual.provenance.details["frame_ids"]
        )
        self.assertEqual(1, len(result.visual_grounding_shadow_units))

    def test_shadow_service_has_no_prediction_dependencies(self):
        source_path = Path(shadow_module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = set()
        imported_names = set()
        runtime_unit_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if name == "RuntimeUnit":
                    runtime_unit_calls.append(node)
        self.assertFalse(runtime_unit_calls)
        self.assertTrue(
            {
                "FrozenG1Runner",
                "VerificationResult",
                "ModelVerdict",
                "DisplayVerdict",
                "EvidenceStatus",
            }.isdisjoint(imported_names)
        )
        self.assertFalse(
            any(
                module.startswith(
                    (
                        "services.frozen_g1_runner",
                        "adapters.phase4a_request_adapter",
                        "adapters.phase4a_response_adapter",
                    )
                )
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
