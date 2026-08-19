"""Compose video decoding, local Whisper ASR, and optional Frozen G1 inference."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.transcript_unit_adapter import TranscriptUnitAdapter
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    VerificationResult,
)


@dataclass
class VideoASRResult:
    session_id: str
    claim: str
    video_metadata: Dict[str, Any]
    asr_text: str
    asr_segments: List[Dict[str, Any]]
    runtime_units: List[RuntimeUnit]
    warnings: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0
    verification_result: Optional[VerificationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claim": self.claim,
            "video_metadata": dict(self.video_metadata),
            "asr_text": self.asr_text,
            "asr_segments": [dict(segment) for segment in self.asr_segments],
            "runtime_units": [unit.to_dict() for unit in self.runtime_units],
            "warnings": list(self.warnings),
            "runtime_ms": self.runtime_ms,
            "verification_result": (
                self.verification_result.to_dict()
                if self.verification_result is not None
                else None
            ),
        }


class VideoASRRunner:
    def __init__(
        self,
        decoder: Any,
        asr_service: Any,
        transcript_adapter: Optional[TranscriptUnitAdapter] = None,
        frozen_g1_runner: Any = None,
    ) -> None:
        self.decoder = decoder
        self.asr_service = asr_service
        self.transcript_adapter = transcript_adapter or TranscriptUnitAdapter()
        self.frozen_g1_runner = frozen_g1_runner

    @staticmethod
    def _insufficient_result(
        session_id: str,
        claim: str,
        units: List[RuntimeUnit],
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
            all_units=units,
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
    ) -> VideoASRResult:
        if not str(session_id).strip():
            raise ValueError("session_id is required")
        if not str(claim).strip():
            raise ValueError("claim is required")
        started = time.perf_counter()
        path = Path(video_path)
        decoded = self.decoder.decode(path)
        segments = self.asr_service.transcribe(
            decoded.waveform,
            sample_rate=decoded.sample_rate,
        )
        units = self.transcript_adapter.convert(segments, source_uri=str(path))
        warnings = []
        verification_result = None
        if not units:
            warnings.append("ASR produced no valid transcript units")
            verification_result = self._insufficient_result(
                session_id, claim, units, warnings
            )
        elif run_frozen_g1:
            if self.frozen_g1_runner is None:
                raise ValueError("FrozenG1Runner is required for the optional G1 handoff")
            verification_result = self.frozen_g1_runner.run(session_id, claim, units)

        runtime_ms = (time.perf_counter() - started) * 1000.0
        metadata = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "audio_duration_seconds": decoded.duration_seconds,
            "audio_sample_rate": decoded.sample_rate,
            "audio_sample_count": int(decoded.waveform.size),
        }
        return VideoASRResult(
            session_id=session_id,
            claim=claim,
            video_metadata=metadata,
            asr_text=" ".join(unit.text for unit in units),
            asr_segments=segments,
            runtime_units=units,
            warnings=warnings,
            runtime_ms=runtime_ms,
            verification_result=verification_result,
        )

    def run_with_frozen_g1(
        self, session_id: str, claim: str, video_path: Path
    ) -> VideoASRResult:
        return self.run(session_id, claim, video_path, run_frozen_g1=True)
