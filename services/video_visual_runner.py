"""Run SigLIP retrieval and claim-blind Qwen observation over one video."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.visual_observation_adapter import VisualObservationAdapter
from schemas import RuntimeUnit
from services.qwen_visual_observer import QwenVisualObservationResult
from services.siglip_visual_retriever import SigLIPRetrievalResult


@dataclass
class VideoVisualResult:
    session_id: str
    claim: str
    video_path: str
    retrieval_result: SigLIPRetrievalResult
    observation_result: Optional[QwenVisualObservationResult]
    runtime_units: List[RuntimeUnit]
    warnings: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claim": self.claim,
            "video_path": self.video_path,
            "retrieval_result": self.retrieval_result.to_dict(),
            "observation_result": (
                self.observation_result.to_dict()
                if self.observation_result is not None
                else None
            ),
            "runtime_units": [unit.to_dict() for unit in self.runtime_units],
            "warnings": list(self.warnings),
            "runtime_ms": self.runtime_ms,
        }


class VideoVisualRunner:
    def __init__(
        self,
        retriever: Any,
        observer: Any,
        adapter: Optional[VisualObservationAdapter] = None,
    ) -> None:
        self.retriever = retriever
        self.observer = observer
        self.adapter = adapter or VisualObservationAdapter()

    def run(self, session_id: str, claim: str, video_path: Path) -> VideoVisualResult:
        if not str(session_id).strip():
            raise ValueError("session_id is required")
        if not str(claim).strip():
            raise ValueError("claim is required")
        started = time.perf_counter()
        path = Path(video_path)
        retrieval = self.retriever.retrieve(
            claim=claim, video_path=path, session_id=session_id
        )
        warnings = []
        observation_result = None
        units = []
        if retrieval.selected_frames:
            selected_frames = sorted(
                retrieval.selected_frames,
                key=lambda frame: (
                    frame.timestamp_sec is None,
                    frame.timestamp_sec if frame.timestamp_sec is not None else 0.0,
                    frame.frame_index,
                    str(frame.frame_path),
                ),
            )
            observation_result = self.observer.observe(selected_frames)
            units = self.adapter.convert(
                observation_result.observations,
                selected_frames,
                recovery_mode=observation_result.recovery_mode,
                raw_generation_sha256=observation_result.raw_generation_sha256,
                source_uri=str(path),
            )
            if observation_result.rejected_observation_count:
                warnings.append(
                    f"rejected {observation_result.rejected_observation_count} unsafe visual observations"
                )
        else:
            warnings.append("SigLIP retrieval produced no visual frames")
        return VideoVisualResult(
            session_id=session_id,
            claim=claim,
            video_path=str(path),
            retrieval_result=retrieval,
            observation_result=observation_result,
            runtime_units=units,
            warnings=warnings,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
        )
