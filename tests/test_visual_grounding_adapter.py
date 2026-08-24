import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import unittest

import adapters.visual_grounding_adapter as adapter_module
from adapters.visual_grounding_adapter import VisualGroundingAdapter
from schemas import (
    GroundedVisualUnit,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
    VisualObservationSnapshot,
)


class VisualGroundingAdapterTests(unittest.TestCase):
    @staticmethod
    def referenced_frame(
        frame_id="F001",
        frame_index=10,
        timestamp_sec=1.5,
        frame_rank=1,
        retrieval_rank=1,
        image_sha256="a" * 64,
    ):
        return {
            "frame_id": frame_id,
            "frame_path": f"/private/frames/{frame_id}.jpg",
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "frame_rank": frame_rank,
            "image_sha256": image_sha256,
            "retrieval_score": 0.75,
            "retrieval_rank": retrieval_rank,
        }

    @classmethod
    def visual_unit(cls, text="players are visible on a basketball court"):
        frames = [
            cls.referenced_frame(),
            cls.referenced_frame(
                frame_id="F002",
                frame_index=20,
                timestamp_sec=3.5,
                frame_rank=2,
                retrieval_rank=2,
                image_sha256="b" * 64,
            ),
        ]
        return RuntimeUnit(
            unit_id="visual_0123456789abcdef0123",
            source_type=SourceType.VISUAL_OBSERVATION,
            text=text,
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
                    "referenced_frames": frames,
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

    @classmethod
    def visual_snapshot(cls, text="players are visible on a basketball court"):
        return VisualObservationSnapshot.from_runtime_unit(cls.visual_unit(text=text))

    def test_accepts_visual_observation(self):
        grounded = VisualGroundingAdapter().ground([self.visual_snapshot()])
        self.assertEqual(1, len(grounded))
        self.assertIsInstance(grounded[0], GroundedVisualUnit)

    def test_rejects_non_visual_runtime_units(self):
        for source_type in (
            SourceType.TEXT,
            SourceType.OCR,
            SourceType.TRANSCRIPT,
        ):
            with self.subTest(source_type=source_type):
                unit = self.visual_unit()
                unit.source_type = source_type
                with self.assertRaisesRegex(ValueError, "source_type visual_observation"):
                    VisualObservationSnapshot.from_runtime_unit(unit)

    def test_rejects_already_grounded_and_unknown_inputs(self):
        grounded = VisualGroundingAdapter().ground([self.visual_snapshot()])[0]
        for value in (grounded, object()):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(
                TypeError, "VisualObservationSnapshot"
            ):
                VisualGroundingAdapter().ground([value])

    def test_preserves_text_exactly(self):
        text = "  players are visible on a basketball court  "
        grounded = VisualGroundingAdapter().ground([self.visual_snapshot(text=text)])[0]
        self.assertEqual(text, grounded.text_observation)

    def test_preserves_frames_timestamps_and_provenance(self):
        source = self.visual_unit()
        snapshot = VisualObservationSnapshot.from_runtime_unit(source)
        grounded = VisualGroundingAdapter().ground([snapshot])[0]
        self.assertEqual(source.unit_id, grounded.source_observation_id)
        self.assertEqual(1.5, grounded.start_timestamp_seconds)
        self.assertEqual(3.5, grounded.end_timestamp_seconds)
        self.assertEqual(
            ("F001", "F002"),
            tuple(reference.frame_id for reference in grounded.frame_references),
        )
        self.assertEqual(
            (10, 20),
            tuple(reference.frame_index for reference in grounded.frame_references),
        )
        self.assertEqual(
            ("a" * 64, "b" * 64),
            tuple(reference.image_sha256 for reference in grounded.frame_references),
        )
        self.assertEqual(source.provenance.source_index, grounded.lineage.source_index)
        self.assertEqual(
            source.provenance.extraction_method, grounded.lineage.extraction_method
        )
        self.assertEqual(("F001", "F002"), grounded.lineage.frame_ids)
        self.assertEqual(("F001", "F002"), grounded.lineage.evidence_refs)
        self.assertEqual("c" * 64, grounded.lineage.raw_generation_sha256)
        self.assertEqual(
            source.provenance.details["qwen_revision"],
            grounded.model_identity.qwen_revision,
        )
        self.assertRegex(
            grounded.lineage.source_observation_sha256, r"^[0-9a-f]{64}$"
        )

    def test_same_input_produces_identical_artifact_and_hash(self):
        source = self.visual_unit()
        snapshot = VisualObservationSnapshot.from_runtime_unit(source)
        adapter = VisualGroundingAdapter()
        first = adapter.ground([snapshot])[0]
        second = adapter.ground([snapshot])[0]
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.unit_id, second.unit_id)
        self.assertEqual(first.identity_sha256(), second.identity_sha256())

    def test_adapter_does_not_mutate_runtime_unit(self):
        source = self.visual_unit()
        before = source.to_dict()
        snapshot = VisualObservationSnapshot.from_runtime_unit(source)
        VisualGroundingAdapter().ground([snapshot])
        self.assertEqual(before, source.to_dict())
        self.assertEqual(
            {"text", "transcript", "ocr", "visual_observation"},
            {source_type.value for source_type in SourceType},
        )

    def test_visual_snapshot_is_immutable_and_slot_restricted(self):
        snapshot = self.visual_snapshot()
        with self.assertRaises(FrozenInstanceError):
            snapshot.observation_text = "tampered"
        with self.assertRaises(FrozenInstanceError):
            snapshot.frame_ids = ("tampered",)
        self.assertFalse(hasattr(snapshot, "__dict__"))

    def test_output_has_no_prediction_or_path_fields(self):
        snapshot = self.visual_snapshot()
        grounded = VisualGroundingAdapter().ground([snapshot])[0]
        forbidden = {
            "verdict",
            "model_verdict",
            "display_verdict",
            "logits",
            "probabilities",
            "selection_score",
            "confidence",
            "eligible_for_frozen_g1",
            "frame_path",
            "source_uri",
            "checkpoint_path",
        }
        self.assertTrue(forbidden.isdisjoint({item.name for item in fields(snapshot)}))
        self.assertTrue(forbidden.isdisjoint({item.name for item in fields(grounded)}))

        def nested_keys(value):
            if isinstance(value, dict):
                result = set(value)
                for nested in value.values():
                    result.update(nested_keys(nested))
                return result
            if isinstance(value, list):
                result = set()
                for nested in value:
                    result.update(nested_keys(nested))
                return result
            return set()

        self.assertTrue(forbidden.isdisjoint(nested_keys(grounded.to_dict())))
        self.assertNotIn("/private/", grounded.to_json())

    def test_adapter_has_no_inference_imports_or_runtime_unit_construction(self):
        source_path = Path(adapter_module.__file__)
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
                module.startswith(("services.", "adapters.phase4a"))
                for module in imported_modules
            )
        )

    def test_invalid_visual_observation_data_is_rejected_explicitly(self):
        cases = []

        empty_text = self.visual_unit()
        empty_text.text = "   "
        cases.append((empty_text, "text must be a non-empty string"))

        missing_id = self.visual_unit()
        missing_id.unit_id = ""
        cases.append((missing_id, "unit_id must be a non-empty string"))

        missing_frames = self.visual_unit()
        missing_frames.provenance.details["referenced_frames"] = []
        cases.append((missing_frames, "referenced_frames must be a non-empty"))

        invalid_timestamp = self.visual_unit()
        invalid_timestamp.provenance.details["referenced_frames"][0][
            "timestamp_sec"
        ] = -1.0
        cases.append((invalid_timestamp, "timestamp_seconds"))

        for source, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                VisualObservationSnapshot.from_runtime_unit(source)

    def test_predictive_visual_input_is_rejected(self):
        cases = (
            ("eligible_for_frozen_g1", True, "ineligible"),
            ("selection_score", 0.8, "selection_score"),
            ("logits", {"fake": 1.0, "real": 0.0}, "logits"),
            ("confidence", 0.9, "confidence"),
        )
        for field_name, value, message in cases:
            with self.subTest(field_name=field_name):
                unit = self.visual_unit()
                setattr(unit, field_name, value)
                with self.assertRaisesRegex(ValueError, message):
                    VisualObservationSnapshot.from_runtime_unit(unit)


if __name__ == "__main__":
    unittest.main()
