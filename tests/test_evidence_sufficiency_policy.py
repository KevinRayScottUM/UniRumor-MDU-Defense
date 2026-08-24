import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    VerificationResult,
)
from services.evidence_sufficiency_policy import (
    EvidenceSufficiencyAssessment,
    EvidenceSufficiencyPolicy,
)
from services.frozen_g1_runner import FrozenG1Runner
from services.production_runtime import ProductionRuntime
from services.qwen_visual_observer import QwenVisualObserver
from services.siglip_visual_retriever import (
    SigLIPRetrievalResult,
    SigLIPVisualRetriever,
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


_UNSET = object()


class EvidenceSufficiencyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = EvidenceSufficiencyPolicy()

    @staticmethod
    def _unit(
        unit_id,
        source_type,
        *,
        eligible=True,
        selection_score=None,
        logits=None,
        confidence=None,
    ):
        return RuntimeUnit(
            unit_id=unit_id,
            source_type=source_type,
            text=f"evidence {unit_id}",
            confidence=confidence,
            producer="test-fixture",
            eligible_for_frozen_g1=eligible,
            selection_score=selection_score,
            logits=logits,
        )

    def _transcript(self, unit_id="transcript-1", *, eligible=True):
        return self._unit(
            unit_id,
            SourceType.TRANSCRIPT,
            eligible=eligible,
            selection_score=0.8,
            logits={"fake": 1.2, "real": -0.2},
        )

    def _ocr(self, unit_id="ocr-1", *, eligible=True):
        return self._unit(
            unit_id,
            SourceType.OCR,
            eligible=eligible,
            selection_score=0.7,
            logits={"fake": 0.6, "real": 0.1},
            confidence=0.95,
        )

    def _visual(
        self,
        unit_id="visual-1",
        *,
        selection_score=None,
        logits=None,
        confidence=None,
    ):
        return self._unit(
            unit_id,
            SourceType.VISUAL_OBSERVATION,
            eligible=False,
            selection_score=selection_score,
            logits=logits,
            confidence=confidence,
        )

    def _result(
        self,
        *,
        g1_units=None,
        visual_units=None,
        all_runtime_units=None,
        verification_all_units=None,
        top_k_units=None,
        model_verdict=_UNSET,
        display_verdict=_UNSET,
        evidence_status=_UNSET,
        session_id="session-1",
        claim="  exact claim  ",
        verification_session_id=None,
        verification_claim=None,
    ):
        g1_units = (
            [self._transcript(), self._ocr()]
            if g1_units is None
            else list(g1_units)
        )
        visual_units = (
            [self._visual()] if visual_units is None else list(visual_units)
        )
        composed = g1_units + visual_units
        all_runtime_units = (
            composed if all_runtime_units is None else list(all_runtime_units)
        )
        verification_all_units = (
            g1_units
            if verification_all_units is None
            else list(verification_all_units)
        )
        top_k_units = (
            list(g1_units[:2]) if top_k_units is None else list(top_k_units)
        )

        if model_verdict is _UNSET:
            model_verdict = ModelVerdict.FAKE if g1_units else ModelVerdict.NOT_RUN
        if display_verdict is _UNSET:
            display_verdict = (
                DisplayVerdict.FAKE
                if model_verdict is ModelVerdict.FAKE
                else DisplayVerdict.REAL
                if model_verdict is ModelVerdict.REAL
                else DisplayVerdict.NEI
            )
        if evidence_status is _UNSET:
            evidence_status = (
                EvidenceStatus.INSUFFICIENT
                if model_verdict is ModelVerdict.NOT_RUN
                else EvidenceStatus.SUFFICIENT
            )
        binary_completed = model_verdict in {
            ModelVerdict.FAKE,
            ModelVerdict.REAL,
        }
        verification = VerificationResult(
            session_id=(
                session_id
                if verification_session_id is None
                else verification_session_id
            ),
            claim=claim if verification_claim is None else verification_claim,
            model_verdict=model_verdict,
            display_verdict=display_verdict,
            evidence_status=evidence_status,
            sample_logits=(
                {"fake": 1.2, "real": 0.2} if binary_completed else {}
            ),
            probabilities=(
                {"fake": 0.73, "real": 0.27} if binary_completed else {}
            ),
            all_units=verification_all_units,
            top_k_units=top_k_units,
            class_winners={},
            pipeline_stages=[],
            warnings=["fixture warning"],
        )
        asr_result = VideoASRResult(
            session_id=session_id,
            claim=claim,
            video_metadata={},
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
            video_path="/read-only/source.mp4",
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
            g1_exposure_units=list(g1_units),
        )
        visual_result = VideoVisualResult(
            session_id=session_id,
            claim=claim,
            video_path="/read-only/source.mp4",
            retrieval_result=SigLIPRetrievalResult(
                candidate_frames=[],
                selected_frames=[],
                claim_token_audit={},
            ),
            observation_result=None,
            runtime_units=list(visual_units),
        )
        return VideoMultimodalResult(
            session_id=session_id,
            claim=claim,
            text_ocr_result=text_ocr_result,
            visual_result=visual_result,
            g1_exposure_units=g1_units,
            visual_units=visual_units,
            all_runtime_units=all_runtime_units,
            verification_result=verification,
            warnings=["fixture warning"],
        )

    def test_valid_sufficient_result_reports_exact_status_reason_and_model_state(self):
        assessment = self.policy.assess(self._result())
        self.assertEqual(assessment.status, EvidenceStatus.SUFFICIENT)
        self.assertEqual(
            assessment.reason_code,
            "frozen_g1_evidence_available_and_model_completed",
        )
        self.assertTrue(assessment.model_was_run)

    def test_valid_sufficient_counts_actual_evidence_structure(self):
        assessment = self.policy.assess(self._result())
        self.assertEqual(assessment.g1_exposure_count, 2)
        self.assertEqual(assessment.transcript_exposure_count, 1)
        self.assertEqual(assessment.ocr_exposure_count, 1)
        self.assertEqual(assessment.visual_unit_count, 1)
        self.assertEqual(assessment.top_k_count, 2)
        self.assertTrue(assessment.supplemental_visual_present)

    def test_sufficient_result_without_visual_units_remains_sufficient(self):
        assessment = self.policy.assess(self._result(visual_units=[]))
        self.assertEqual(assessment.status, EvidenceStatus.SUFFICIENT)
        self.assertEqual(assessment.visual_unit_count, 0)
        self.assertFalse(assessment.supplemental_visual_present)

    def test_valid_real_completion_is_sufficient(self):
        assessment = self.policy.assess(
            self._result(
                model_verdict=ModelVerdict.REAL,
                display_verdict=DisplayVerdict.REAL,
            )
        )
        self.assertEqual(assessment.status, EvidenceStatus.SUFFICIENT)
        self.assertTrue(assessment.model_was_run)

    def test_valid_insufficient_result_without_units(self):
        assessment = self.policy.assess(
            self._result(g1_units=[], visual_units=[], top_k_units=[])
        )
        self.assertEqual(assessment.status, EvidenceStatus.INSUFFICIENT)
        self.assertEqual(assessment.g1_exposure_count, 0)
        self.assertEqual(assessment.visual_unit_count, 0)
        self.assertFalse(assessment.model_was_run)

    def test_visual_only_result_remains_insufficient_with_exact_reason(self):
        assessment = self.policy.assess(
            self._result(g1_units=[], top_k_units=[])
        )
        self.assertEqual(assessment.status, EvidenceStatus.INSUFFICIENT)
        self.assertEqual(
            assessment.reason_code,
            "no_frozen_g1_eligible_evidence",
        )
        self.assertFalse(assessment.model_was_run)
        self.assertEqual(assessment.visual_unit_count, 1)
        self.assertTrue(assessment.supplemental_visual_present)

    def test_policy_rejects_non_multimodal_result(self):
        for invalid in ({}, None, object()):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "VideoMultimodalResult"):
                    self.policy.assess(invalid)

    def test_policy_rejects_visual_inside_g1_exposure(self):
        visual = self._visual()
        result = self._result(
            g1_units=[visual],
            visual_units=[],
            top_k_units=[],
        )
        with self.assertRaisesRegex(ValueError, "eligible non-visual"):
            self.policy.assess(result)

    def test_policy_rejects_ineligible_g1_exposure(self):
        result = self._result(
            g1_units=[self._transcript(eligible=False)],
            visual_units=[],
            top_k_units=[],
        )
        with self.assertRaisesRegex(ValueError, "eligible non-visual"):
            self.policy.assess(result)

    def test_policy_rejects_non_visual_inside_visual_units(self):
        non_visual = self._unit(
            "text-supplement",
            SourceType.TEXT,
            eligible=False,
        )
        result = self._result(visual_units=[non_visual])
        with self.assertRaisesRegex(ValueError, "only visual observations"):
            self.policy.assess(result)

    def test_policy_rejects_visual_marked_g1_eligible_via_bypass(self):
        visual = self._visual()
        visual.eligible_for_frozen_g1 = True
        result = self._result(g1_units=[], visual_units=[visual], top_k_units=[])
        with self.assertRaisesRegex(ValueError, "cannot be eligible"):
            self.policy.assess(result)

    def test_policy_rejects_visual_scientific_fields(self):
        invalid_visuals = (
            (self._visual(selection_score=0.1), "selection scores"),
            (self._visual(logits={"fake": 0.1, "real": 0.2}), "logits"),
            (self._visual(confidence=0.9), "confidence"),
        )
        for visual, message in invalid_visuals:
            with self.subTest(message=message):
                result = self._result(
                    g1_units=[],
                    visual_units=[visual],
                    top_k_units=[],
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.policy.assess(result)

    def test_policy_rejects_duplicate_ids_across_composed_units(self):
        transcript = self._transcript("duplicate")
        visual = self._visual("duplicate")
        result = self._result(g1_units=[transcript], visual_units=[visual])
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            self.policy.assess(result)

    def test_policy_rejects_all_runtime_unit_order_or_content_mismatch(self):
        transcript = self._transcript()
        visual = self._visual()
        mismatches = (
            [visual, transcript],
            [transcript],
            [transcript, visual, self._ocr("extra")],
        )
        for all_units in mismatches:
            with self.subTest(ids=[unit.unit_id for unit in all_units]):
                result = self._result(
                    g1_units=[transcript],
                    visual_units=[visual],
                    all_runtime_units=all_units,
                    verification_all_units=all_units,
                )
                with self.assertRaisesRegex(ValueError, "all_runtime_units"):
                    self.policy.assess(result)

    def test_policy_rejects_verification_all_units_mismatch(self):
        transcript = self._transcript()
        visual = self._visual()
        result = self._result(
            g1_units=[transcript],
            visual_units=[visual],
            verification_all_units=[transcript, visual],
        )
        with self.assertRaisesRegex(ValueError, "verification_result.all_units"):
            self.policy.assess(result)

    def test_policy_rejects_visual_or_unknown_unit_in_top_k(self):
        transcript = self._transcript()
        visual = self._visual()
        unknown = self._transcript("unknown")
        for invalid_top_k in ([visual], [unknown]):
            with self.subTest(unit_id=invalid_top_k[0].unit_id):
                result = self._result(
                    g1_units=[transcript],
                    visual_units=[visual],
                    top_k_units=invalid_top_k,
                )
                with self.assertRaisesRegex(ValueError, "top_k_units"):
                    self.policy.assess(result)

    def test_policy_rejects_session_or_exact_claim_mismatch(self):
        mismatches = (
            {"verification_session_id": "different-session"},
            {"verification_claim": "exact claim"},
        )
        for overrides in mismatches:
            with self.subTest(overrides=overrides):
                result = self._result(**overrides)
                with self.assertRaisesRegex(ValueError, "must match"):
                    self.policy.assess(result)

    def test_g1_evidence_with_not_run_nei_is_rejected(self):
        result = self._result(
            model_verdict=ModelVerdict.NOT_RUN,
            display_verdict=DisplayVerdict.NEI,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            top_k_units=[],
        )
        with self.assertRaisesRegex(ValueError, "completed binary"):
            self.policy.assess(result)

    def test_g1_evidence_rejects_inconsistent_status_or_display(self):
        insufficient = self._result(evidence_status=EvidenceStatus.INSUFFICIENT)
        wrong_display = self._result()
        wrong_display.verification_result.display_verdict = DisplayVerdict.REAL
        cases = (
            (insufficient, "sufficient verification status"),
            (wrong_display, "display verdict"),
        )
        for result, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.policy.assess(result)

    def test_no_g1_evidence_with_binary_result_is_rejected(self):
        cases = (
            (ModelVerdict.FAKE, DisplayVerdict.FAKE),
            (ModelVerdict.REAL, DisplayVerdict.REAL),
        )
        for model_verdict, display_verdict in cases:
            with self.subTest(model_verdict=model_verdict):
                result = self._result(
                    g1_units=[],
                    top_k_units=[],
                    model_verdict=model_verdict,
                    display_verdict=display_verdict,
                    evidence_status=EvidenceStatus.SUFFICIENT,
                )
                with self.assertRaisesRegex(ValueError, "not_run"):
                    self.policy.assess(result)

    def test_one_g1_unit_and_fewer_than_five_top_k_is_sufficient(self):
        transcript = self._transcript()
        assessment = self.policy.assess(
            self._result(
                g1_units=[transcript],
                visual_units=[],
                top_k_units=[transcript],
            )
        )
        self.assertEqual(assessment.status, EvidenceStatus.SUFFICIENT)
        self.assertEqual(assessment.g1_exposure_count, 1)
        self.assertEqual(assessment.top_k_count, 1)

    def test_eligible_text_counts_as_g1_but_not_transcript_or_ocr(self):
        text_unit = self._unit("text-1", SourceType.TEXT, eligible=True)
        assessment = self.policy.assess(
            self._result(
                g1_units=[text_unit],
                visual_units=[],
                top_k_units=[text_unit],
            )
        )
        self.assertEqual(assessment.g1_exposure_count, 1)
        self.assertEqual(assessment.transcript_exposure_count, 0)
        self.assertEqual(assessment.ocr_exposure_count, 0)
        self.assertEqual(assessment.status, EvidenceStatus.SUFFICIENT)

    def test_assessment_is_frozen_and_serializes_to_primitives(self):
        assessment = self.policy.assess(self._result())
        payload = assessment.to_dict()
        encoded = json.dumps(payload, sort_keys=True)
        self.assertIsInstance(encoded, str)
        self.assertEqual(payload["status"], "sufficient")
        self.assertNotIn("logits", payload)
        self.assertNotIn("probabilities", payload)
        self.assertTrue(
            all(
                value is None
                or isinstance(value, (str, bool, int, float))
                for value in payload.values()
            )
        )
        with self.assertRaises(FrozenInstanceError):
            assessment.status = EvidenceStatus.INSUFFICIENT

    def test_assess_does_not_mutate_input_result(self):
        result = self._result()
        before = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
        self.policy.assess(result)
        after = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(after, before)

    def test_assess_calls_no_runtime_runner_or_model_loader(self):
        result = self._result()
        with (
            patch.object(ProductionRuntime, "run", side_effect=AssertionError),
            patch.object(VideoMultimodalRunner, "run", side_effect=AssertionError),
            patch.object(FrozenG1Runner, "run", side_effect=AssertionError),
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
        ):
            assessment = self.policy.assess(result)
        self.assertIsInstance(assessment, EvidenceSufficiencyAssessment)

    def test_assess_writes_no_files(self):
        result = self._result()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before = list(root.rglob("*"))
            self.policy.assess(result)
            after = list(root.rglob("*"))
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
