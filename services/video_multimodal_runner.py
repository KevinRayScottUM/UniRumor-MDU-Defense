"""Compose text/OCR verification with supplemental real visual observations."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    GroundedVisualUnit,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    VerificationResult,
    VisualObservationSnapshot,
)
from services.claim_consistency_gate import (
    CLAIM_VIDEO_MISMATCH_WARNING,
    ClaimConsistencyGate,
    ConsistencyResult,
)
from services.video_text_ocr_runner import VideoTextOCRResult
from services.video_visual_runner import VideoVisualResult
from services.visual_grounding_shadow import (
    VISUAL_GROUNDING_SHADOW_FAILURE_WARNING,
    VisualGroundingShadowRunner,
)


@dataclass
class VideoMultimodalResult:
    session_id: str
    claim: str
    text_ocr_result: VideoTextOCRResult
    visual_result: VideoVisualResult
    g1_exposure_units: List[RuntimeUnit]
    visual_units: List[RuntimeUnit]
    all_runtime_units: List[RuntimeUnit]
    verification_result: VerificationResult
    visual_grounding_shadow_units: List[GroundedVisualUnit] = field(
        default_factory=list
    )
    warnings: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claim": self.claim,
            "text_ocr_result": self.text_ocr_result.to_dict(),
            "visual_result": self.visual_result.to_dict(),
            "g1_exposure_units": [unit.to_dict() for unit in self.g1_exposure_units],
            "visual_units": [unit.to_dict() for unit in self.visual_units],
            "all_runtime_units": [unit.to_dict() for unit in self.all_runtime_units],
            "verification_result": self.verification_result.to_dict(),
            "visual_grounding_shadow_units": [
                unit.to_dict() for unit in self.visual_grounding_shadow_units
            ],
            "warnings": list(self.warnings),
            "runtime_ms": self.runtime_ms,
        }


class VideoMultimodalRunner:
    def __init__(
        self,
        video_text_ocr_runner: Any,
        video_visual_runner: Any,
        frozen_g1_runner: Any,
        claim_consistency_gate: Any = None,
        visual_grounding_shadow_runner: Any = None,
    ) -> None:
        self.video_text_ocr_runner = video_text_ocr_runner
        self.video_visual_runner = video_visual_runner
        self.frozen_g1_runner = frozen_g1_runner
        self.claim_consistency_gate = (
            claim_consistency_gate
            if claim_consistency_gate is not None
            else ClaimConsistencyGate()
        )
        self.visual_grounding_shadow_runner = (
            visual_grounding_shadow_runner
            if visual_grounding_shadow_runner is not None
            else VisualGroundingShadowRunner()
        )

    @staticmethod
    def _insufficient_nei(
        session_id: str,
        claim: str,
        g1_units: List[RuntimeUnit],
        warnings: List[str],
    ) -> VerificationResult:
        return VerificationResult(
            session_id=session_id,
            claim=claim,
            model_verdict=ModelVerdict.NOT_RUN,
            display_verdict=DisplayVerdict.NEI,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            sample_logits={},
            probabilities={},
            all_units=g1_units,
            top_k_units=[],
            class_winners={},
            pipeline_stages=[],
            warnings=list(warnings),
        )

    def run(
        self, session_id: str, claim: str, video_path: Path
    ) -> VideoMultimodalResult:
        started = time.perf_counter()
        text_ocr_result = self.video_text_ocr_runner.run(
            session_id, claim, video_path, run_frozen_g1=False
        )
        visual_result = self.video_visual_runner.run(session_id, claim, video_path)
        g1_units = list(text_ocr_result.g1_exposure_units)
        visual_units = list(visual_result.runtime_units)
        if any(unit.source_type is SourceType.VISUAL_OBSERVATION for unit in g1_units):
            raise ValueError("visual observations cannot enter composed G1 exposure")
        for unit in visual_units:
            if (
                unit.source_type is not SourceType.VISUAL_OBSERVATION
                or unit.eligible_for_frozen_g1
                or unit.selection_score is not None
                or unit.logits is not None
                or unit.confidence is not None
            ):
                raise ValueError("invalid supplemental visual RuntimeUnit contract")
        all_units = g1_units + visual_units
        warnings = list(text_ocr_result.warnings) + list(visual_result.warnings)
        if g1_units:
            consistency_result = self.claim_consistency_gate.evaluate(
                claim=claim,
                transcript_units=[
                    unit
                    for unit in g1_units
                    if unit.source_type is SourceType.TRANSCRIPT
                ],
                ocr_units=[
                    unit
                    for unit in g1_units
                    if unit.source_type is SourceType.OCR
                ],
                visual_units=visual_units,
            )
            if consistency_result is ConsistencyResult.MISMATCH:
                warnings.append(CLAIM_VIDEO_MISMATCH_WARNING)
                verification = self._insufficient_nei(
                    session_id, claim, g1_units, warnings
                )
            elif consistency_result in {
                ConsistencyResult.PASS,
                ConsistencyResult.UNKNOWN,
            }:
                verification = self.frozen_g1_runner.run(
                    session_id, claim, g1_units
                )
            else:
                raise ValueError("invalid claim consistency result")
        else:
            warnings.append("visual-only evidence cannot run Frozen G1")
            verification = self._insufficient_nei(
                session_id, claim, g1_units, warnings
            )
        visual_grounding_shadow_units = []
        try:
            visual_snapshots = [
                VisualObservationSnapshot.from_runtime_unit(unit, index=index)
                for index, unit in enumerate(visual_units)
            ]
            visual_grounding_shadow_units = self.visual_grounding_shadow_runner.run(
                visual_snapshots
            )
        except Exception:
            warnings.append(VISUAL_GROUNDING_SHADOW_FAILURE_WARNING)
        return VideoMultimodalResult(
            session_id=session_id,
            claim=claim,
            text_ocr_result=text_ocr_result,
            visual_result=visual_result,
            g1_exposure_units=g1_units,
            visual_units=visual_units,
            all_runtime_units=all_units,
            verification_result=verification,
            visual_grounding_shadow_units=visual_grounding_shadow_units,
            warnings=warnings,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
