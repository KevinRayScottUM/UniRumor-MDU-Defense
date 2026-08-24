import ast
from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path
import unittest

import schemas.grounded_visual_unit as grounded_visual_module
from schemas import (
    GROUNDED_VISUAL_ARTIFACT_TYPE,
    GROUNDED_VISUAL_SCHEMA_VERSION,
    GroundedFrameReference,
    GroundedVisualUnit,
    GroundingLineage,
    GroundingModelIdentity,
    RuntimeUnit,
)


class GroundedVisualUnitTests(unittest.TestCase):
    @staticmethod
    def frame(
        frame_id="F001",
        frame_index=10,
        timestamp_seconds=1.5,
        frame_rank=1,
        retrieval_rank=1,
        image_sha256="a" * 64,
    ):
        return GroundedFrameReference(
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            frame_rank=frame_rank,
            retrieval_rank=retrieval_rank,
            image_sha256=image_sha256,
        )

    @staticmethod
    def model_identity():
        return GroundingModelIdentity(
            siglip_model_id="google/siglip-so400m-patch14-384",
            siglip_revision="siglip-revision",
            qwen_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            qwen_revision="qwen-revision",
            adapter_id="visual-grounding-shadow-adapter",
            adapter_version="1.0.0",
        )

    @staticmethod
    def lineage(frame_ids=("F001", "F002"), evidence_refs=("F001", "F002")):
        return GroundingLineage(
            source_observation_id="visual_0123456789abcdef0123",
            source_observation_sha256="b" * 64,
            source_index=0,
            extraction_method="qwen_claim_blind_visual_observer",
            observation_type="temporal_change",
            frame_ids=frame_ids,
            evidence_refs=evidence_refs,
            raw_generation_sha256="c" * 64,
            recovery_mode="canonical_object",
            retrieval_policy_id="claim_conditioned_siglip_top4",
            observer_policy_id="claim_blind_visible_atomic_facts_no_ocr_no_inference",
        )

    @classmethod
    def unit(cls, text="A player moves from one side of the court to the other."):
        frames = (
            cls.frame(),
            cls.frame(
                frame_id="F002",
                frame_index=20,
                timestamp_seconds=3.5,
                frame_rank=2,
                retrieval_rank=2,
                image_sha256="d" * 64,
            ),
        )
        return GroundedVisualUnit.create(
            source_observation_id="visual_0123456789abcdef0123",
            text_observation=text,
            start_timestamp_seconds=1.5,
            end_timestamp_seconds=3.5,
            frame_references=frames,
            model_identity=cls.model_identity(),
            prompt_policy="claim_blind_visible_atomic_facts_no_ocr_no_inference",
            lineage=cls.lineage(),
        )

    def test_object_and_nested_objects_are_immutable(self):
        unit = self.unit()
        with self.assertRaises(FrozenInstanceError):
            unit.text_observation = "changed"
        with self.assertRaises(FrozenInstanceError):
            unit.frame_references[0].frame_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            unit.model_identity.adapter_version = "changed"
        with self.assertRaises(FrozenInstanceError):
            unit.lineage.recovery_mode = "changed"
        with self.assertRaises(AttributeError):
            unit.frame_references.append(self.frame())

    def test_tuple_ordering_is_preserved(self):
        unit = self.unit()
        self.assertEqual(("F001", "F002"), tuple(item.frame_id for item in unit.frame_references))
        self.assertEqual(["F001", "F002"], unit.to_dict()["lineage"]["frame_ids"])
        self.assertEqual(
            ["F001", "F002"],
            [item["frame_id"] for item in unit.to_dict()["frame_references"]],
        )

    def test_serialization_is_deterministic_and_round_trips(self):
        first = self.unit()
        second = self.unit()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.to_dict(), json.loads(first.to_json()))
        self.assertEqual(first, GroundedVisualUnit.from_dict(first.to_dict()))

    def test_sha256_identity_is_deterministic(self):
        first = self.unit()
        second = self.unit()
        self.assertEqual(first.unit_id, second.unit_id)
        self.assertEqual(first.identity_sha256(), second.identity_sha256())
        self.assertRegex(first.identity_sha256(), r"^[0-9a-f]{64}$")
        self.assertEqual(f"gvu_{first.identity_sha256()}", first.unit_id)

    def test_changed_content_changes_identity(self):
        first = self.unit()
        changed = self.unit(text="Two players stand near a basketball hoop.")
        self.assertNotEqual(first.unit_id, changed.unit_id)
        self.assertNotEqual(first.identity_sha256(), changed.identity_sha256())

    def test_prediction_fields_are_absent(self):
        forbidden = {
            "verdict",
            "model_verdict",
            "display_verdict",
            "logits",
            "probabilities",
            "selection_score",
            "confidence",
            "eligible_for_frozen_g1",
        }
        declared = {item.name for item in fields(GroundedVisualUnit)}
        self.assertTrue(forbidden.isdisjoint(declared))

        def keys(value):
            if isinstance(value, dict):
                found = set(value)
                for item in value.values():
                    found.update(keys(item))
                return found
            if isinstance(value, list):
                found = set()
                for item in value:
                    found.update(keys(item))
                return found
            return set()

        self.assertTrue(forbidden.isdisjoint(keys(self.unit().to_dict())))

    def test_schema_has_no_forbidden_imports(self):
        source_path = Path(grounded_visual_module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = set()
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)
        self.assertTrue(
            {
                "services.frozen_g1_runner",
                "adapters.phase4a_request_adapter",
                "adapters.phase4a_response_adapter",
                "schemas.result",
            }.isdisjoint(imported_modules)
        )
        self.assertTrue(
            {
                "FrozenG1Runner",
                "RuntimeUnit",
                "SourceType",
                "ModelVerdict",
                "DisplayVerdict",
                "EvidenceStatus",
            }.isdisjoint(imported_names)
        )

    def test_grounded_visual_unit_does_not_inherit_runtime_unit(self):
        self.assertFalse(issubclass(GroundedVisualUnit, RuntimeUnit))

    def test_empty_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "text_observation"):
            self.unit(text="   ")

    def test_empty_frame_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame_references"):
            GroundedVisualUnit.create(
                source_observation_id="visual_0123456789abcdef0123",
                text_observation="A player stands on a court.",
                start_timestamp_seconds=1.5,
                end_timestamp_seconds=3.5,
                frame_references=(),
                model_identity=self.model_identity(),
                prompt_policy="claim_blind_visible_atomic_facts_no_ocr_no_inference",
                lineage=self.lineage(),
            )

    def test_invalid_timestamps_are_rejected(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "timestamp_seconds"
            ):
                self.frame(timestamp_seconds=value)

    def test_invalid_sha256_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            self.frame(image_sha256="not-a-sha256")
        with self.assertRaisesRegex(ValueError, "source_observation_sha256"):
            GroundingLineage(
                source_observation_id="visual_0123456789abcdef0123",
                source_observation_sha256="not-a-sha256",
                source_index=0,
                extraction_method="observer",
                observation_type="scene",
                frame_ids=("F001",),
                evidence_refs=("F001",),
                raw_generation_sha256="c" * 64,
                recovery_mode="canonical_object",
                retrieval_policy_id="claim_conditioned_siglip_top4",
                observer_policy_id="claim_blind",
            )

    def test_schema_constants_are_stable_strings(self):
        self.assertEqual("1", GROUNDED_VISUAL_SCHEMA_VERSION)
        self.assertEqual("grounded_visual_unit", GROUNDED_VISUAL_ARTIFACT_TYPE)


if __name__ == "__main__":
    unittest.main()
