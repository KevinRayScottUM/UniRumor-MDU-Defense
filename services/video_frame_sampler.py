"""Deterministically sample and extract video frames beneath a cache root."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from services.cache_manager import safe_target


DEFAULT_FRAMES_PER_VIDEO = 8


def sample_frame_indices(frame_count: int, n: int) -> List[int]:
    """Historical UniRumor frame-index sampling rule."""
    if frame_count <= 0 or n <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    if n == 1:
        return [frame_count // 2]

    out = []
    seen = set()
    for i in range(n):
        ratio = (i + 1) / (n + 1)
        idx = int(round(ratio * (frame_count - 1)))
        idx = max(0, min(frame_count - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


@dataclass(frozen=True)
class SampledVideoFrame:
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    frame_id: str
    frame_path: Path

    def to_dict(self):
        return {
            "frame_rank": self.frame_rank,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "frame_id": self.frame_id,
            "frame_path": str(self.frame_path),
        }


class VideoFrameSampler:
    def __init__(
        self,
        cache_root: Path,
        frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
        av_module: Any = None,
        frame_writer: Optional[Callable[[Any, Path], None]] = None,
    ) -> None:
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be positive")
        self.cache_root = Path(cache_root).resolve()
        self.frames_per_video = frames_per_video
        self._av = av_module
        self._frame_writer = frame_writer or self._write_jpeg

    @staticmethod
    def _write_jpeg(frame: Any, target: Path) -> None:
        frame.to_image().save(str(target), format="JPEG")

    def _open_video(self, path: Path):
        av_module = self._av or importlib.import_module("av")
        container = av_module.open(str(path))
        streams = list(container.streams.video)
        if not streams:
            container.close()
            raise ValueError("video has no video stream")
        return container, streams[0]

    @staticmethod
    def _fps(stream: Any) -> float:
        rate = getattr(stream, "average_rate", None)
        if rate is None:
            rate = getattr(stream, "base_rate", None)
        fps = float(rate) if rate is not None else 0.0
        if fps <= 0:
            raise ValueError("video stream has no positive frame rate")
        return fps

    def sample(self, session_id: str, video_path: Path) -> List[SampledVideoFrame]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"video path is not a regular file: {path}")

        container, stream = self._open_video(path)
        try:
            frame_count = int(getattr(stream, "frames", 0) or 0)
            fps = self._fps(stream)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(stream))
        finally:
            container.close()

        indices = sample_frame_indices(frame_count, self.frames_per_video)
        if not indices:
            return []
        wanted = set(indices)
        extracted = {}
        container, stream = self._open_video(path)
        try:
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index not in wanted:
                    continue
                frame_rank = indices.index(frame_index)
                frame_id = f"frame_{frame_rank:04d}_{frame_index:06d}"
                target = safe_target(
                    self.cache_root,
                    "ocr_frames",
                    f"{session_id}_{frame_id}.jpg",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                self._frame_writer(frame, target)
                extracted[frame_index] = SampledVideoFrame(
                    frame_rank=frame_rank,
                    frame_index=frame_index,
                    timestamp_sec=float(frame_index) / fps,
                    frame_id=frame_id,
                    frame_path=target,
                )
                if len(extracted) == len(indices):
                    break
        finally:
            container.close()
        missing = [index for index in indices if index not in extracted]
        if missing:
            raise ValueError(f"video ended before sampled frame indices: {missing}")
        return [extracted[index] for index in indices]
