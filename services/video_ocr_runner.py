"""Sample video frames, run isolated PP-OCRv5, and build OCR RuntimeUnits."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.ocr_unit_adapter import OCRUnitAdapter
from schemas import RuntimeUnit
from services.paddle_ocr_service import OCRFrameResult
from services.video_frame_sampler import SampledVideoFrame


@dataclass
class VideoOCRResult:
    session_id: str
    video_path: str
    sampled_frames: List[SampledVideoFrame]
    raw_ocr_artifacts: List[OCRFrameResult]
    ocr_units: List[RuntimeUnit]
    warnings: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_path": self.video_path,
            "sampled_frames": [frame.to_dict() for frame in self.sampled_frames],
            "raw_ocr_artifacts": [
                result.to_dict() for result in self.raw_ocr_artifacts
            ],
            "ocr_units": [unit.to_dict() for unit in self.ocr_units],
            "warnings": list(self.warnings),
            "runtime_ms": self.runtime_ms,
        }


class VideoOCRRunner:
    def __init__(
        self,
        frame_sampler: Any,
        ocr_service: Any,
        ocr_adapter: Optional[OCRUnitAdapter] = None,
    ) -> None:
        self.frame_sampler = frame_sampler
        self.ocr_service = ocr_service
        self.ocr_adapter = ocr_adapter or OCRUnitAdapter()

    def run(self, session_id: str, video_path: Path) -> VideoOCRResult:
        if not str(session_id).strip():
            raise ValueError("session_id is required")
        started = time.perf_counter()
        path = Path(video_path)
        sampled_frames = self.frame_sampler.sample(session_id, path)
        raw_results = (
            self.ocr_service.predict(session_id, sampled_frames)
            if sampled_frames
            else []
        )
        units = self.ocr_adapter.convert(raw_results, source_uri=str(path))
        warnings = []
        if not sampled_frames:
            warnings.append("video produced no sampled frames")
        return VideoOCRResult(
            session_id=session_id,
            video_path=str(path),
            sampled_frames=sampled_frames,
            raw_ocr_artifacts=raw_results,
            ocr_units=units,
            warnings=warnings,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
