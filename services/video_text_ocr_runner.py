"""Compose existing video ASR with OCR and optional external Frozen G1."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    VerificationResult,
)
from services.multimodal_exposure_composer import MultimodalExposureComposer
from services.paddle_ocr_service import OCRFrameResult
from services.video_asr_runner import VideoASRResult
from services.video_ocr_runner import VideoOCRResult


@dataclass
class VideoTextOCRResult:
    session_id: str
    claim: str
    asr_result: VideoASRResult
    ocr_result: VideoOCRResult
    raw_asr_units: List[RuntimeUnit]
    raw_ocr_artifacts: List[OCRFrameResult]
    ocr_runtime_units: List[RuntimeUnit]
    g1_exposure_units: List[RuntimeUnit]
    verification_result: Optional[VerificationResult] = None
    warnings: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claim": self.claim,
            "asr_result": self.asr_result.to_dict(),
            "ocr_result": self.ocr_result.to_dict(),
            "raw_asr_units": [unit.to_dict() for unit in self.raw_asr_units],
            "raw_ocr_artifacts": [
                artifact.to_dict() for artifact in self.raw_ocr_artifacts
            ],
            "ocr_runtime_units": [
                unit.to_dict() for unit in self.ocr_runtime_units
            ],
            "g1_exposure_units": [
                unit.to_dict() for unit in self.g1_exposure_units
            ],
            "verification_result": (
                self.verification_result.to_dict()
                if self.verification_result is not None
                else None
            ),
            "warnings": list(self.warnings),
            "runtime_ms": self.runtime_ms,
        }


class VideoTextOCRRunner:
    def __init__(
        self,
        video_asr_runner: Any,
        video_ocr_runner: Any,
        exposure_composer: Optional[MultimodalExposureComposer] = None,
        frozen_g1_runner: Any = None,
    ) -> None:
        self.video_asr_runner = video_asr_runner
        self.video_ocr_runner = video_ocr_runner
        self.exposure_composer = exposure_composer or MultimodalExposureComposer()
        self.frozen_g1_runner = frozen_g1_runner

    @staticmethod
    def _insufficient(
        session_id: str, claim: str, warnings: List[str]
    ) -> VerificationResult:
        return VerificationResult(
            session_id=session_id,
            claim=claim,
            model_verdict=ModelVerdict.NOT_RUN,
            display_verdict=DisplayVerdict.NEI,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            sample_logits={},
            probabilities={},
            all_units=[],
            top_k_units=[],
            class_winners={},
            pipeline_stages=[],
            warnings=list(warnings),
        )

    def run(
        self,
        session_id: str,
        claim: str,
        video_path: Path,
        run_frozen_g1: bool = False,
    ) -> VideoTextOCRResult:
        started = time.perf_counter()
        asr_result = self.video_asr_runner.run(
            session_id, claim, video_path, run_frozen_g1=False
        )
        ocr_result = self.video_ocr_runner.run(session_id, video_path)
        raw_asr_units = list(asr_result.runtime_units)
        raw_ocr_units = list(ocr_result.ocr_units)
        exposure_units = self.exposure_composer.compose(
            raw_asr_units, raw_ocr_units
        )
        warnings = list(asr_result.warnings) + list(ocr_result.warnings)
        verification_result = None
        if not exposure_units:
            warnings.append("ASR and OCR produced no Frozen G1 evidence")
            verification_result = self._insufficient(session_id, claim, warnings)
        elif run_frozen_g1:
            if self.frozen_g1_runner is None:
                raise ValueError("FrozenG1Runner is required for the optional G1 handoff")
            verification_result = self.frozen_g1_runner.run(
                session_id, claim, exposure_units
            )
        return VideoTextOCRResult(
            session_id=session_id,
            claim=claim,
            asr_result=asr_result,
            ocr_result=ocr_result,
            raw_asr_units=raw_asr_units,
            raw_ocr_artifacts=list(ocr_result.raw_ocr_artifacts),
            ocr_runtime_units=raw_ocr_units,
            g1_exposure_units=exposure_units,
            verification_result=verification_result,
            warnings=warnings,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )

    def run_with_frozen_g1(
        self, session_id: str, claim: str, video_path: Path
    ) -> VideoTextOCRResult:
        return self.run(session_id, claim, video_path, run_frozen_g1=True)
