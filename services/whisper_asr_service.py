"""Local-only Whisper inference and ASR segment normalization."""

import hashlib
import importlib
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_FROZEN_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
WHISPER_SAFETENSORS_SHA256 = (
    "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
)


@dataclass(frozen=True)
class WhisperASRConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: str = "float16"
    verify_asset_sha256: bool = False
    expected_safetensors_sha256: str = WHISPER_SAFETENSORS_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if not self.device:
            raise ValueError("ASR device is required")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError(f"unsupported ASR dtype: {self.dtype}")


class WhisperASRService:
    def __init__(
        self,
        config: WhisperASRConfig,
        transformers_module: Any = None,
        torch_module: Any = None,
    ) -> None:
        self.config = config
        self._transformers = transformers_module
        self._torch = torch_module
        self._pipeline = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_asset(self) -> None:
        weights = self.config.model_path / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Whisper model.safetensors not found: {weights}")
        actual = self._sha256(weights)
        if actual != self.config.expected_safetensors_sha256:
            raise ValueError(
                "Whisper model.safetensors SHA256 mismatch: "
                f"expected {self.config.expected_safetensors_sha256}, got {actual}"
            )

    def _dependencies(self):
        transformers_module = self._transformers or importlib.import_module("transformers")
        torch_module = self._torch or importlib.import_module("torch")
        return transformers_module, torch_module

    def _device_and_dtype(self, torch_module: Any) -> Tuple[str, Any]:
        device = self.config.device
        cuda_available = bool(torch_module.cuda.is_available())
        if device == "auto":
            device = "cuda:0" if cuda_available else "cpu"
        elif device.startswith("cuda") and not cuda_available:
            raise RuntimeError(f"configured ASR device is unavailable: {device}")

        dtype_name = self.config.dtype
        if device == "cpu" and dtype_name == "float16":
            dtype_name = "float32"
        return device, getattr(torch_module, dtype_name)

    @staticmethod
    def _with_dtype(factory, args, kwargs, dtype):
        try:
            return factory(*args, **dict(kwargs, dtype=dtype))
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            return factory(*args, **dict(kwargs, torch_dtype=dtype))

    def load(self) -> None:
        if self._pipeline is not None:
            return
        if not self.config.model_path.is_dir():
            raise FileNotFoundError(
                f"configured local Whisper model directory not found: {self.config.model_path}"
            )
        if self.config.verify_asset_sha256:
            self.verify_asset()

        transformers_module, torch_module = self._dependencies()
        device, dtype = self._device_and_dtype(torch_module)
        local_path = str(self.config.model_path)
        processor = transformers_module.AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True,
        )
        model = self._with_dtype(
            transformers_module.AutoModelForSpeechSeq2Seq.from_pretrained,
            (local_path,),
            {"local_files_only": True},
            dtype,
        )
        pipeline_kwargs = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "device": device,
        }
        self._pipeline = self._with_dtype(
            transformers_module.pipeline,
            ("automatic-speech-recognition",),
            pipeline_kwargs,
            dtype,
        )

    @staticmethod
    def _timestamp_pair(value: Any) -> Tuple[Optional[float], Optional[float]]:
        if value is None:
            return None, None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("ASR segment timestamp must be a two-item sequence or null")
        normalized = []
        for item in value:
            if item is None:
                normalized.append(None)
                continue
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError("ASR timestamps must be numeric or null")
            numeric = float(item)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("ASR timestamps must be finite and nonnegative")
            normalized.append(numeric)
        start, end = normalized
        if start is not None and end is not None and end < start:
            raise ValueError("ASR segment end timestamp precedes its start")
        return start, end

    @classmethod
    def normalize_output(cls, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(output, dict):
            raise ValueError("Whisper output must be a mapping")
        chunks = output.get("chunks")
        if chunks is None:
            text = str(output.get("text", "")).strip()
            chunks = [] if not text else [{"text": text, "timestamp": None}]
        if not isinstance(chunks, list):
            raise ValueError("Whisper chunks must be a list")

        segments = []
        previous_start = None
        previous_end = None
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("each Whisper chunk must be a mapping")
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            start, end = cls._timestamp_pair(chunk.get("timestamp"))
            if start is not None and previous_start is not None and start < previous_start:
                raise ValueError("ASR segment start timestamps are non-monotonic")
            if end is not None and previous_end is not None and end < previous_end:
                raise ValueError("ASR segment end timestamps are non-monotonic")
            segments.append(
                {
                    "segment_index": len(segments),
                    "text": text,
                    "start_time": start,
                    "end_time": end,
                }
            )
            if start is not None:
                previous_start = start
            if end is not None:
                previous_end = end
        return segments

    def transcribe(self, waveform: Any, sample_rate: int = 16_000) -> List[Dict[str, Any]]:
        if sample_rate != 16_000:
            raise ValueError("WhisperASRService requires a 16 kHz waveform")
        self.load()
        output = self._pipeline(
            {"raw": waveform, "sampling_rate": sample_rate},
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
        return self.normalize_output(output)
