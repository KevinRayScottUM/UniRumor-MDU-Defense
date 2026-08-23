"""Structural evidence-sufficiency assessment for completed video results."""

from dataclasses import dataclass
from typing import Any, Dict

from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict, SourceType
from services.video_multimodal_runner import VideoMultimodalResult


@dataclass(frozen=True)
class EvidenceSufficiencyAssessment:
    status: EvidenceStatus
    reason_code: str
    model_was_run: bool
    g1_exposure_count: int
    transcript_exposure_count: int
    ocr_exposure_count: int
    visual_unit_count: int
    top_k_count: int
    supplemental_visual_present: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "model_was_run": self.model_was_run,
            "g1_exposure_count": self.g1_exposure_count,
            "transcript_exposure_count": self.transcript_exposure_count,
            "ocr_exposure_count": self.ocr_exposure_count,
            "visual_unit_count": self.visual_unit_count,
            "top_k_count": self.top_k_count,
            "supplemental_visual_present": self.supplemental_visual_present,
        }


class EvidenceSufficiencyPolicy:
    SUFFICIENT_REASON = "frozen_g1_evidence_available_and_model_completed"
    INSUFFICIENT_REASON = "no_frozen_g1_eligible_evidence"

    def assess(
        self,
        result: VideoMultimodalResult,
    ) -> EvidenceSufficiencyAssessment:
        if not isinstance(result, VideoMultimodalResult):
            raise TypeError("result must be a VideoMultimodalResult")

        g1_units = result.g1_exposure_units
        visual_units = result.visual_units
        allowed_g1_sources = {
            SourceType.TEXT,
            SourceType.TRANSCRIPT,
            SourceType.OCR,
        }
        for unit in g1_units:
            if (
                unit.source_type not in allowed_g1_sources
                or not unit.eligible_for_frozen_g1
            ):
                raise ValueError(
                    "g1_exposure_units must contain only eligible non-visual units"
                )

        for unit in visual_units:
            if unit.source_type is not SourceType.VISUAL_OBSERVATION:
                raise ValueError("visual_units must contain only visual observations")
            if unit.eligible_for_frozen_g1:
                raise ValueError("visual units cannot be eligible for Frozen G1")
            if unit.selection_score is not None:
                raise ValueError("visual units cannot carry selection scores")
            if unit.logits is not None:
                raise ValueError("visual units cannot carry logits")
            if unit.confidence is not None:
                raise ValueError("visual units cannot carry confidence")

        composed_units = list(g1_units) + list(visual_units)
        composed_ids = [unit.unit_id for unit in composed_units]
        if len(set(composed_ids)) != len(composed_ids):
            raise ValueError("RuntimeUnit IDs must be unique across composed evidence")
        if [unit.unit_id for unit in result.all_runtime_units] != composed_ids:
            raise ValueError(
                "all_runtime_units must equal g1_exposure_units plus visual_units"
            )

        verification = result.verification_result
        if [unit.unit_id for unit in verification.all_units] != composed_ids:
            raise ValueError(
                "verification_result.all_units must match all_runtime_units"
            )

        g1_ids = {unit.unit_id for unit in g1_units}
        for unit in verification.top_k_units:
            if (
                unit.unit_id not in g1_ids
                or unit.source_type is SourceType.VISUAL_OBSERVATION
                or not unit.eligible_for_frozen_g1
            ):
                raise ValueError("top_k_units must belong to Frozen G1 exposure")

        if result.session_id != verification.session_id:
            raise ValueError("result and verification session_id must match")
        if result.claim != verification.claim:
            raise ValueError("result and verification claim must match")

        g1_exposure_count = len(g1_units)
        transcript_exposure_count = sum(
            unit.source_type is SourceType.TRANSCRIPT for unit in g1_units
        )
        ocr_exposure_count = sum(
            unit.source_type is SourceType.OCR for unit in g1_units
        )
        visual_unit_count = len(visual_units)
        expected_display = {
            ModelVerdict.FAKE: DisplayVerdict.FAKE,
            ModelVerdict.REAL: DisplayVerdict.REAL,
            ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
        }[verification.model_verdict]

        if g1_exposure_count:
            if verification.model_verdict not in {
                ModelVerdict.FAKE,
                ModelVerdict.REAL,
            }:
                raise ValueError(
                    "Frozen G1 evidence requires a completed binary model verdict"
                )
            if verification.evidence_status is not EvidenceStatus.SUFFICIENT:
                raise ValueError(
                    "Frozen G1 evidence requires sufficient verification status"
                )
            if verification.display_verdict is not expected_display:
                raise ValueError(
                    "display verdict must match the completed Frozen G1 verdict"
                )
            status = EvidenceStatus.SUFFICIENT
            reason_code = self.SUFFICIENT_REASON
            model_was_run = True
        else:
            if verification.model_verdict is not ModelVerdict.NOT_RUN:
                raise ValueError(
                    "no Frozen G1 evidence requires a not_run model verdict"
                )
            if verification.evidence_status is not EvidenceStatus.INSUFFICIENT:
                raise ValueError(
                    "no Frozen G1 evidence requires insufficient verification status"
                )
            if verification.display_verdict is not expected_display:
                raise ValueError(
                    "no Frozen G1 evidence requires the NEI display verdict"
                )
            status = EvidenceStatus.INSUFFICIENT
            reason_code = self.INSUFFICIENT_REASON
            model_was_run = False

        return EvidenceSufficiencyAssessment(
            status=status,
            reason_code=reason_code,
            model_was_run=model_was_run,
            g1_exposure_count=g1_exposure_count,
            transcript_exposure_count=transcript_exposure_count,
            ocr_exposure_count=ocr_exposure_count,
            visual_unit_count=visual_unit_count,
            top_k_count=len(verification.top_k_units),
            supplemental_visual_present=visual_unit_count > 0,
        )
