import base64
import inspect
import json
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from unittest.mock import patch

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
    VerificationResult,
)
from services.evidence_sufficiency_policy import EvidenceSufficiencyPolicy
from services.frozen_g1_runner import FrozenG1Runner
from services.paddle_ocr_service import PaddleOCRService
from services.production_result import (
    ProductionEvidenceUnit,
    ProductionResult,
    ProductionResultBuilder,
)
from services.production_runtime import ProductionRuntime
from services.qwen_visual_observer import QwenVisualObserver
from services.siglip_visual_retriever import (
    SigLIPRetrievalResult,
    SigLIPVisualRetriever,
    VisualFrame,
)
from services.video_asr_runner import VideoASRResult
from services.video_multimodal_runner import (
    VideoMultimodalResult,
    VideoMultimodalRunner,
)
from services.video_ocr_runner import VideoOCRResult
from services.video_text_ocr_runner import VideoTextOCRResult
from services.video_visual_runner import VideoVisualResult
from services.whisper_asr_service import WhisperASRService


class ProductionResultTests(unittest.TestCase):
    def setUp(self):
        self.builder = ProductionResultBuilder()

    @staticmethod
    def _g1_unit(unit_id, source_type=SourceType.TRANSCRIPT, index=0):
        return RuntimeUnit(
            unit_id=unit_id,
            source_type=source_type,
            text=f"evidence {unit_id}",
            start_time=index + 0.125,
            end_time=index + 0.875,
            frame_id=f"OCR{index:03d}" if source_type is SourceType.OCR else None,
            frame_path=(
                "/secret/server/ocr-frame.jpg"
                if source_type is SourceType.OCR
                else None
            ),
            bbox=[1.125, 2.25, 30.5, 40.875]
            if source_type is SourceType.OCR
            else None,
            confidence=0.9123456789012345
            if source_type is SourceType.OCR
            else None,
            producer="fixture-g1-producer",
            provenance=UnitProvenance(
                source_uri="/scr/user/private/dataset/video.mp4",
                source_index=index,
                extraction_method="fixture_g1_extraction",
                details={
                    "source_unit_ids": [f"raw-{index}-a", f"raw-{index}-b"],
                    "private_model_path": "/secret/server/model.bin",
                },
            ),
            eligible_for_frozen_g1=True,
            selection_score=0.1234567890123456 + index,
            logits={
                "real": -0.1234567890123456 - index,
                "fake": 1.2345678901234567 + index,
            },
        )

    @staticmethod
    def _visual_unit(unit_id="visual-1", index=0):
        return RuntimeUnit(
            unit_id=unit_id,
            source_type=SourceType.VISUAL_OBSERVATION,
            text=f"visible fact {unit_id}",
            start_time=index + 1.5,
            end_time=index + 2.5,
            frame_id=f"F{index + 1:03d}",
            frame_path="/secret/server/frame.jpg",
            bbox=None,
            confidence=None,
            producer="fixture-visual-producer",
            provenance=UnitProvenance(
                source_uri="/secret/server/video.mp4",
                source_index=index,
                extraction_method="qwen_claim_blind_visual_observer",
                details={
                    "frame_ids": ["F001", "F002"],
                    "evidence_refs": ["F001"],
                    "observation_type": "scene",
                    "referenced_frames": [
                        {
                            "frame_id": "F001",
                            "frame_path": "/secret/server/referenced.jpg",
                        }
                    ],
                    "some_unapproved_internal_field": "must-not-leak",
                },
            ),
            eligible_for_frozen_g1=False,
            selection_score=None,
            logits=None,
        )

    def _result(
        self,
        *,
        g1_units=None,
        visual_units=None,
        model_verdict=ModelVerdict.FAKE,
        display_verdict=DisplayVerdict.FAKE,
        evidence_status=EvidenceStatus.SUFFICIENT,
        top_k_units=None,
        session_id="Session.API-1",
        claim="  Exact Unicode claim 汉语  ",
    ):
        g1_units = (
            [
                self._g1_unit("transcript-z", SourceType.TRANSCRIPT, 0),
                self._g1_unit("ocr-a", SourceType.OCR, 1),
            ]
            if g1_units is None
            else list(g1_units)
        )
        visual_units = (
            [self._visual_unit("visual-z", 0), self._visual_unit("visual-a", 1)]
            if visual_units is None
            else list(visual_units)
        )
        top_k_units = (
            list(reversed(g1_units)) if top_k_units is None else list(top_k_units)
        )
        completed = model_verdict in {ModelVerdict.FAKE, ModelVerdict.REAL}
        all_units = g1_units + visual_units
        verification = VerificationResult(
            session_id=session_id,
            claim=claim,
            model_verdict=model_verdict,
            display_verdict=display_verdict,
            evidence_status=evidence_status,
            sample_logits=(
                {
                    "real": -0.9876543210987654,
                    "fake": 1.2345678901234567,
                }
                if completed
                else {}
            ),
            probabilities=(
                {
                    "real": 0.12345678901234568,
                    "fake": 0.8765432109876543,
                }
                if completed
                else {}
            ),
            all_units=g1_units,
            top_k_units=top_k_units,
            class_winners=(
                {"real": g1_units[-1].unit_id, "fake": g1_units[0].unit_id}
                if completed
                else {}
            ),
            pipeline_stages=[],
            warnings=["internal verification warning /secret/server/runtime.log"],
            checkpoint_sha256=("checkpoint-fixture" if completed else None),
            runtime_ms=111.11111111111111,
        )
        asr_result = VideoASRResult(
            session_id=session_id,
            claim=claim,
            video_metadata={"source": "/secret/server/video.mp4"},
            asr_text="",
            asr_segments=[],
            runtime_units=[
                unit
                for unit in g1_units
                if unit.source_type is SourceType.TRANSCRIPT
            ],
        )
        ocr_result = VideoOCRResult(
            session_id=session_id,
            video_path="/secret/server/video.mp4",
            sampled_frames=[],
            raw_ocr_artifacts=[],
            ocr_units=[
                unit for unit in g1_units if unit.source_type is SourceType.OCR
            ],
        )
        text_ocr_result = VideoTextOCRResult(
            session_id=session_id,
            claim=claim,
            asr_result=asr_result,
            ocr_result=ocr_result,
            raw_asr_units=list(asr_result.runtime_units),
            raw_ocr_artifacts=[],
            ocr_runtime_units=list(ocr_result.ocr_units),
            g1_exposure_units=g1_units,
            warnings=["internal text warning /secret/server/text.log"],
        )
        selected_frame = VisualFrame(
            frame_id="F001",
            frame_path=Path("/secret/server/selected/F001.jpg"),
            frame_index=1,
            timestamp_sec=1.5,
            frame_rank=0,
            image_sha256="image-fixture",
        )
        visual_result = VideoVisualResult(
            session_id=session_id,
            claim=claim,
            video_path="/secret/server/video.mp4",
            retrieval_result=SigLIPRetrievalResult(
                candidate_frames=[selected_frame],
                selected_frames=[selected_frame],
                claim_token_audit={},
            ),
            observation_result=None,
            runtime_units=visual_units,
            warnings=["internal visual warning /secret/server/visual.log"],
        )
        return VideoMultimodalResult(
            session_id=session_id,
            claim=claim,
            text_ocr_result=text_ocr_result,
            visual_result=visual_result,
            g1_exposure_units=g1_units,
            visual_units=visual_units,
            all_runtime_units=all_units,
            verification_result=verification,
            warnings=["internal aggregate warning /secret/server/aggregate.log"],
            runtime_ms=987.6543210987654,
        )

    @staticmethod
    def _recursive_keys(value):
        keys = set()
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                keys.update(ProductionResultTests._recursive_keys(item))
        elif isinstance(value, list):
            for item in value:
                keys.update(ProductionResultTests._recursive_keys(item))
        return keys

    def test_builder_accepts_result_and_preserves_identity_fields(self):
        internal = self._result()
        public = self.builder.build(internal)
        self.assertIsInstance(public, ProductionResult)
        self.assertEqual(public.schema_version, 1)
        self.assertEqual(public.session_id, internal.session_id)
        self.assertEqual(public.claim, internal.claim)
        self.assertEqual(public.runtime_ms, internal.runtime_ms)

    def test_builder_rejects_non_multimodal_result(self):
        for invalid in ({}, None, object()):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "VideoMultimodalResult"):
                    self.builder.build(invalid)

    def test_scientific_values_are_preserved_exactly_without_rounding(self):
        internal = self._result()
        public = self.builder.build(internal)
        verification = internal.verification_result
        self.assertIs(public.model_verdict, verification.model_verdict)
        self.assertIs(public.display_verdict, verification.display_verdict)
        self.assertIs(public.evidence_status, verification.evidence_status)
        self.assertEqual(dict(public.sample_logits), verification.sample_logits)
        self.assertEqual(dict(public.probabilities), verification.probabilities)
        self.assertEqual(dict(public.class_winners), verification.class_winners)
        self.assertEqual(public.checkpoint_sha256, verification.checkpoint_sha256)
        payload = public.to_dict()["verdict"]
        self.assertEqual(payload["sample_logits"], verification.sample_logits)
        self.assertEqual(payload["probabilities"], verification.probabilities)

    def test_builder_derives_authoritative_sufficiency_assessment(self):
        internal = self._result()
        expected = EvidenceSufficiencyPolicy().assess(internal)
        with patch(
            "services.production_result.EvidenceSufficiencyPolicy.assess",
            return_value=expected,
        ) as assess:
            public = self.builder.build(internal)
        assess.assert_called_once_with(internal)
        self.assertIs(public.sufficiency, expected)
        self.assertEqual(public.to_dict()["sufficiency"], expected.to_dict())

    def test_builder_has_no_sufficiency_override_or_threshold_arguments(self):
        parameters = list(inspect.signature(ProductionResultBuilder.build).parameters)
        self.assertEqual(parameters, ["self", "result"])

    def test_g1_exposure_preserves_exact_order_without_truncation(self):
        units = [self._g1_unit(f"unit-{index:02d}", index=index) for index in range(9)]
        internal = self._result(
            g1_units=units,
            visual_units=[],
            top_k_units=[units[8], units[2], units[5]],
        )
        public = self.builder.build(internal)
        self.assertEqual(
            [unit.unit_id for unit in public.g1_exposure_units],
            [unit.unit_id for unit in units],
        )
        self.assertEqual(len(public.g1_exposure_units), len(units))

    def test_top_k_is_ordered_explanation_ids_only(self):
        internal = self._result()
        public = self.builder.build(internal)
        expected_ids = tuple(
            unit.unit_id for unit in internal.verification_result.top_k_units
        )
        self.assertEqual(public.g1_top_k_explanation_unit_ids, expected_ids)
        evidence = public.to_dict()["evidence"]
        self.assertEqual(
            evidence["g1_top_k_explanation_unit_ids"], list(expected_ids)
        )
        self.assertNotIn("top_k_units", evidence)
        self.assertTrue(
            all(isinstance(unit_id, str) for unit_id in expected_ids)
        )

    def test_evidence_payload_separates_g1_and_visual_without_ambiguous_list(self):
        public = self.builder.build(self._result())
        evidence = public.to_dict()["evidence"]
        self.assertEqual(
            set(evidence),
            {
                "g1_exposure_units",
                "g1_top_k_explanation_unit_ids",
                "visual_supplemental_units",
            },
        )
        self.assertNotIn("all_runtime_units", public.to_dict())
        self.assertEqual(
            [item["unit_id"] for item in evidence["g1_exposure_units"]],
            ["transcript-z", "ocr-a"],
        )
        self.assertEqual(
            [item["unit_id"] for item in evidence["visual_supplemental_units"]],
            ["visual-z", "visual-a"],
        )

    def test_public_evidence_is_immutable_copy_without_runtime_unit_reference(self):
        internal = self._result()
        public = self.builder.build(internal)
        evidence_field_names = {field.name for field in fields(ProductionEvidenceUnit)}
        self.assertNotIn("runtime_unit", evidence_field_names)
        first_internal = internal.g1_exposure_units[0]
        first_public = public.g1_exposure_units[0]
        first_internal.text = "mutated"
        first_internal.logits["fake"] = 999.0
        first_internal.provenance.details["source_unit_ids"].append("mutated")
        self.assertNotEqual(first_public.text, first_internal.text)
        self.assertNotEqual(dict(first_public.logits), first_internal.logits)
        self.assertNotIn("mutated", first_public.source_unit_ids)
        with self.assertRaises(FrozenInstanceError):
            first_public.text = "cannot mutate"

    def test_public_evidence_serializes_fields_and_preserves_numeric_values(self):
        public = self.builder.build(self._result())
        ocr = public.g1_exposure_units[1].to_dict()
        visual = public.visual_supplemental_units[0].to_dict()
        self.assertEqual(ocr["source_type"], "ocr")
        self.assertEqual(ocr["bbox"], [1.125, 2.25, 30.5, 40.875])
        self.assertEqual(
            ocr["logits"],
            {"fake": 2.234567890123457, "real": -1.1234567890123457},
        )
        self.assertEqual(ocr["selection_score"], 1.1234567890123457)
        self.assertEqual(visual["frame_id"], "F001")
        self.assertIsNone(visual["selection_score"])
        self.assertIsNone(visual["logits"])
        self.assertIsNone(visual["confidence"])

    def test_curated_provenance_fields_are_preserved(self):
        public = self.builder.build(self._result())
        g1 = public.g1_exposure_units[0].to_dict()
        visual = public.visual_supplemental_units[0].to_dict()
        self.assertEqual(g1["extraction_method"], "fixture_g1_extraction")
        self.assertEqual(g1["source_index"], 0)
        self.assertEqual(g1["source_unit_ids"], ["raw-0-a", "raw-0-b"])
        self.assertEqual(visual["frame_ids"], ["F001", "F002"])
        self.assertEqual(visual["evidence_refs"], ["F001"])
        self.assertEqual(visual["observation_type"], "scene")

    def test_malformed_curated_sequence_metadata_is_rejected(self):
        malformed_values = ("F001", ["F001", 2], {"F001": True}, None)
        for field_name in ("frame_ids", "evidence_refs", "source_unit_ids"):
            for malformed in malformed_values:
                with self.subTest(field_name=field_name, malformed=malformed):
                    visual = self._visual_unit()
                    visual.provenance.details[field_name] = malformed
                    internal = self._result(
                        visual_units=[visual],
                    )
                    with self.assertRaisesRegex(ValueError, field_name):
                        self.builder.build(internal)

    def test_malformed_observation_type_is_rejected(self):
        visual = self._visual_unit()
        visual.provenance.details["observation_type"] = ["scene"]
        with self.assertRaisesRegex(ValueError, "observation_type"):
            self.builder.build(self._result(visual_units=[visual]))

    def test_public_payload_omits_local_paths_and_internal_provenance(self):
        payload = self.builder.build(self._result()).to_dict()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        forbidden_values = (
            "/secret/server/frame.jpg",
            "/secret/server/video.mp4",
            "/secret/server/referenced.jpg",
            "/secret/server/selected/F001.jpg",
            "/scr/user/private/dataset/video.mp4",
            "/secret/server/model.bin",
        )
        for forbidden in forbidden_values:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, encoded)
        keys = self._recursive_keys(payload)
        self.assertNotIn("frame_path", keys)
        self.assertNotIn("source_uri", keys)
        self.assertNotIn("referenced_frames", keys)
        self.assertNotIn("some_unapproved_internal_field", keys)
        self.assertNotIn("private_model_path", keys)

    def test_path_safe_evidence_frames_embed_real_images_and_ocr_regions(self):
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
            "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory)
            frame_path = cache_root / "frame.png"
            frame_path.write_bytes(png)
            ocr = self._g1_unit("ocr-grounded", SourceType.OCR, 7)
            ocr.frame_path = str(frame_path)
            ocr.provenance.details["accepted_detections"] = [
                {
                    "text": "CPAC",
                    "confidence": 0.97,
                    "runtime_bbox": [1, 2, 30, 20],
                },
                {
                    "text": "2018",
                    "confidence": 0.93,
                    "runtime_bbox": [32, 2, 58, 20],
                },
            ]
            visual = self._visual_unit("visual-grounded", 0)
            visual.provenance.details["referenced_frames"] = [
                {
                    "frame_id": "F001",
                    "frame_path": str(frame_path),
                    "frame_index": 42,
                    "timestamp_sec": 2.5,
                }
            ]

            public = ProductionResultBuilder(evidence_root=cache_root).build(
                self._result(
                    g1_units=[ocr],
                    visual_units=[visual],
                    top_k_units=[ocr],
                )
            )

        ocr_frame = public.g1_exposure_units[0].to_dict()["evidence_frames"][0]
        visual_frame = public.visual_supplemental_units[0].to_dict()[
            "evidence_frames"
        ][0]
        self.assertTrue(ocr_frame["original_image"].startswith("data:image/png;base64,"))
        self.assertEqual(["CPAC", "2018"], [item["text"] for item in ocr_frame["regions"]])
        self.assertEqual([1.0, 2.0, 30.0, 20.0], ocr_frame["regions"][0]["bbox"])
        self.assertIsNone(ocr_frame["annotated_image"])
        self.assertTrue(visual_frame["original_image"].startswith("data:image/png;base64,"))
        self.assertEqual(visual_frame["frame_index"], 42)
        self.assertEqual(visual_frame["timestamp"], 2.5)
        self.assertEqual(visual_frame["regions"], [])
        encoded = json.dumps(public.to_dict())
        self.assertNotIn(str(frame_path), encoded)

    def test_evidence_image_outside_configured_root_is_never_embedded(self):
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
            "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_root = root / "cache"
            cache_root.mkdir()
            outside = root / "private.png"
            outside.write_bytes(png)
            ocr = self._g1_unit("ocr-outside", SourceType.OCR, 3)
            ocr.frame_path = str(outside)

            public = ProductionResultBuilder(evidence_root=cache_root).build(
                self._result(
                    g1_units=[ocr],
                    visual_units=[],
                    top_k_units=[ocr],
                )
            )

        frame = public.g1_exposure_units[0].to_dict()["evidence_frames"][0]
        self.assertIsNone(frame["original_image"])
        self.assertNotIn(str(outside), json.dumps(public.to_dict()))

    def test_builder_never_uses_internal_to_dict_contracts(self):
        internal = self._result()
        with (
            patch.object(
                VideoMultimodalResult,
                "to_dict",
                side_effect=AssertionError,
            ),
            patch.object(RuntimeUnit, "to_dict", side_effect=AssertionError),
        ):
            public = self.builder.build(internal)
        self.assertIsInstance(public, ProductionResult)

    def test_internal_warnings_are_omitted_and_unchanged(self):
        internal = self._result()
        before_result = list(internal.warnings)
        before_verification = list(internal.verification_result.warnings)
        payload = self.builder.build(internal).to_dict()
        self.assertNotIn("warnings", self._recursive_keys(payload))
        self.assertEqual(internal.warnings, before_result)
        self.assertEqual(internal.verification_result.warnings, before_verification)

    def test_to_dict_and_to_json_are_json_safe_and_deterministic(self):
        public = self.builder.build(self._result())
        payload = public.to_dict()
        encoded = public.to_json()
        self.assertEqual(json.loads(encoded), payload)
        self.assertEqual(encoded, public.to_json())
        self.assertEqual(public.to_dict(), self.builder.build(self._result()).to_dict())

    def test_to_json_writes_no_file(self):
        public = self.builder.build(self._result())
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = list(root.rglob("*"))
            encoded = public.to_json()
            after = list(root.rglob("*"))
        self.assertIsInstance(encoded, str)
        self.assertEqual(after, before)

    def test_fake_and_real_results_package_without_changing_verdict(self):
        cases = (
            (ModelVerdict.FAKE, DisplayVerdict.FAKE),
            (ModelVerdict.REAL, DisplayVerdict.REAL),
        )
        for model_verdict, display_verdict in cases:
            with self.subTest(model_verdict=model_verdict):
                public = self.builder.build(
                    self._result(
                        model_verdict=model_verdict,
                        display_verdict=display_verdict,
                    )
                )
                self.assertIs(public.model_verdict, model_verdict)
                self.assertIs(public.display_verdict, display_verdict)

    def test_successful_nei_serializes_without_invented_scores(self):
        internal = self._result(
            g1_units=[],
            model_verdict=ModelVerdict.NOT_RUN,
            display_verdict=DisplayVerdict.NEI,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            top_k_units=[],
        )
        public = self.builder.build(internal)
        payload = public.to_dict()
        self.assertEqual(payload["verdict"]["model_verdict"], "not_run")
        self.assertEqual(payload["verdict"]["display_verdict"], "NEI")
        self.assertEqual(payload["verdict"]["sample_logits"], {})
        self.assertEqual(payload["verdict"]["probabilities"], {})
        self.assertEqual(payload["sufficiency"]["status"], "insufficient")
        self.assertEqual(payload["evidence"]["g1_exposure_units"], [])
        self.assertEqual(len(payload["evidence"]["visual_supplemental_units"]), 2)

    def test_builder_does_not_mutate_internal_result(self):
        internal = self._result()
        before = json.dumps(internal.to_dict(), ensure_ascii=False, sort_keys=True)
        self.builder.build(internal)
        after = json.dumps(internal.to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(after, before)

    def test_repeated_build_is_deterministic(self):
        internal = self._result()
        first = self.builder.build(internal)
        second = self.builder.build(internal)
        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())

    def test_builder_calls_no_runtime_model_or_subprocess(self):
        internal = self._result()
        with (
            patch.object(ProductionRuntime, "run", side_effect=AssertionError),
            patch.object(VideoMultimodalRunner, "run", side_effect=AssertionError),
            patch.object(FrozenG1Runner, "run", side_effect=AssertionError),
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(
                SigLIPVisualRetriever, "retrieve", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
            patch.object(QwenVisualObserver, "observe", side_effect=AssertionError),
            patch.object(PaddleOCRService, "predict", side_effect=AssertionError),
            patch.object(subprocess, "run", side_effect=AssertionError),
        ):
            public = self.builder.build(internal)
        self.assertIsInstance(public, ProductionResult)

    def test_builder_writes_no_files(self):
        internal = self._result()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = list(root.rglob("*"))
            self.builder.build(internal)
            after = list(root.rglob("*"))
        self.assertEqual(after, before)

    def test_visual_only_stays_supplemental_and_cannot_authorize_g1(self):
        internal = self._result(
            g1_units=[],
            model_verdict=ModelVerdict.NOT_RUN,
            display_verdict=DisplayVerdict.NEI,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            top_k_units=[],
        )
        public = self.builder.build(internal)
        self.assertEqual(public.g1_exposure_units, ())
        self.assertEqual(public.g1_top_k_explanation_unit_ids, ())
        self.assertTrue(public.visual_supplemental_units)
        self.assertTrue(
            all(
                not unit.eligible_for_frozen_g1
                for unit in public.visual_supplemental_units
            )
        )
        self.assertEqual(public.sufficiency.status, EvidenceStatus.INSUFFICIENT)

    def test_payload_never_names_top_k_as_prediction_basis(self):
        payload = self.builder.build(self._result()).to_dict()
        keys = self._recursive_keys(payload)
        self.assertIn("g1_top_k_explanation_unit_ids", keys)
        self.assertNotIn("prediction_units", keys)
        self.assertNotIn("decision_evidence", keys)
        self.assertNotIn("units_used_for_prediction", keys)


if __name__ == "__main__":
    unittest.main()
