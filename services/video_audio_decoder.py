"""Decode a video's audio stream to an in-memory 16 kHz mono waveform."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class DecodedAudio:
    waveform: Any
    duration_seconds: float
    sample_rate: int = TARGET_SAMPLE_RATE


class VideoAudioDecoder:
    """Thin PyAV decoder that never creates an intermediate audio file."""

    def __init__(
        self,
        max_duration_seconds: Optional[float] = None,
        av_module: Any = None,
        numpy_module: Any = None,
    ) -> None:
        if max_duration_seconds is not None and max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive when configured")
        self.max_duration_seconds = max_duration_seconds
        self._av = av_module
        self._numpy = numpy_module

    def _dependencies(self):
        av_module = self._av or importlib.import_module("av")
        numpy_module = self._numpy or importlib.import_module("numpy")
        return av_module, numpy_module

    def decode(self, video_path: Path) -> DecodedAudio:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"video path is not a regular file: {path}")

        av_module, numpy_module = self._dependencies()
        container = av_module.open(str(path))
        try:
            audio_streams = list(container.streams.audio)
            if not audio_streams:
                raise ValueError("video has no audio stream")

            resampler = av_module.AudioResampler(
                format="fltp",
                layout="mono",
                rate=TARGET_SAMPLE_RATE,
            )
            chunks = []
            sample_count = 0

            def append_resampled(frames) -> None:
                nonlocal sample_count
                if frames is None:
                    return
                if not isinstance(frames, (list, tuple)):
                    frames = [frames]
                for frame in frames:
                    values = numpy_module.asarray(
                        frame.to_ndarray(), dtype=numpy_module.float32
                    ).reshape(-1)
                    if values.size == 0:
                        continue
                    chunks.append(values)
                    sample_count += int(values.size)
                    if (
                        self.max_duration_seconds is not None
                        and sample_count
                        > int(self.max_duration_seconds * TARGET_SAMPLE_RATE)
                    ):
                        raise ValueError(
                            "decoded audio exceeds configured maximum duration"
                        )

            for frame in container.decode(audio_streams[0]):
                append_resampled(resampler.resample(frame))
            append_resampled(resampler.resample(None))
        finally:
            container.close()

        if not chunks:
            raise ValueError("video audio stream decoded to an empty waveform")
        waveform = numpy_module.ascontiguousarray(
            numpy_module.concatenate(chunks), dtype=numpy_module.float32
        )
        numpy_module.clip(waveform, -1.0, 1.0, out=waveform)
        if waveform.size == 0:
            raise ValueError("video audio stream decoded to an empty waveform")
        return DecodedAudio(
            waveform=waveform,
            duration_seconds=float(waveform.size) / TARGET_SAMPLE_RATE,
        )
