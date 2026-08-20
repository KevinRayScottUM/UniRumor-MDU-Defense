"""Strict deployment-only configuration for the real production runtime."""

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set


SCHEMA_VERSION = 1
PRODUCTION_PROFILE = "production"
WHISPER_DTYPES = {"float16", "float32", "bfloat16"}


def _nonblank(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _positive_timeout(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be positive")
    return float(value)


def _absolute_path(value: Any, field_name: str, base_dir: Path) -> Path:
    raw = _nonblank(value, field_name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _frozen_path(value: Any, field_name: str) -> Path:
    raw = str(value) if isinstance(value, Path) else value
    return _absolute_path(raw, field_name, Path.cwd().resolve())


def _strict_section(
    value: Any,
    section_name: str,
    required_keys: Set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section_name} must be a JSON object")
    keys = set(value)
    unknown = keys - required_keys
    missing = required_keys - keys
    if unknown:
        raise ValueError(
            f"unknown {section_name} fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"missing {section_name} fields: {', '.join(sorted(missing))}"
        )
    return value


@dataclass(frozen=True)
class WhisperRuntimeConfig:
    model_path: Path
    device: str
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", _frozen_path(self.model_path, "whisper.model_path"))
        object.__setattr__(self, "device", _nonblank(self.device, "whisper.device"))
        if not isinstance(self.dtype, str) or self.dtype not in WHISPER_DTYPES:
            raise ValueError(f"unsupported whisper.dtype: {self.dtype}")


@dataclass(frozen=True)
class OCRRuntimeConfig:
    detector_model_path: Path
    recognizer_model_path: Path
    python_executable: str
    device: str
    cudnn8_library_path: Path
    timeout_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detector_model_path",
            _frozen_path(self.detector_model_path, "ocr.detector_model_path"),
        )
        object.__setattr__(
            self,
            "recognizer_model_path",
            _frozen_path(self.recognizer_model_path, "ocr.recognizer_model_path"),
        )
        object.__setattr__(
            self,
            "cudnn8_library_path",
            _frozen_path(self.cudnn8_library_path, "ocr.cudnn8_library_path"),
        )
        object.__setattr__(
            self,
            "python_executable",
            _nonblank(self.python_executable, "ocr.python_executable"),
        )
        object.__setattr__(self, "device", _nonblank(self.device, "ocr.device"))
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_timeout(self.timeout_seconds, "ocr.timeout_seconds"),
        )


@dataclass(frozen=True)
class SigLIPRuntimeConfig:
    model_path: Path
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", _frozen_path(self.model_path, "siglip.model_path"))
        object.__setattr__(self, "device", _nonblank(self.device, "siglip.device"))


@dataclass(frozen=True)
class QwenRuntimeConfig:
    model_path: Path
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", _frozen_path(self.model_path, "qwen.model_path"))
        object.__setattr__(self, "device", _nonblank(self.device, "qwen.device"))


@dataclass(frozen=True)
class FrozenG1RuntimeConfig:
    unirumor_root: Path
    python_executable: str
    phase4a_infer: Path
    phase4a_config: Path
    device: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unirumor_root",
            _frozen_path(self.unirumor_root, "frozen_g1.unirumor_root"),
        )
        object.__setattr__(
            self,
            "phase4a_infer",
            _frozen_path(self.phase4a_infer, "frozen_g1.phase4a_infer"),
        )
        object.__setattr__(
            self,
            "phase4a_config",
            _frozen_path(self.phase4a_config, "frozen_g1.phase4a_config"),
        )
        object.__setattr__(
            self,
            "python_executable",
            _nonblank(self.python_executable, "frozen_g1.python_executable"),
        )
        object.__setattr__(
            self, "device", _nonblank(self.device, "frozen_g1.device")
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_timeout(
                self.timeout_seconds, "frozen_g1.timeout_seconds"
            ),
        )


@dataclass(frozen=True)
class ProductionRuntimeConfig:
    schema_version: int
    profile: str
    cache_root: Path
    output_root: Path
    whisper: WhisperRuntimeConfig
    ocr: OCRRuntimeConfig
    siglip: SigLIPRuntimeConfig
    qwen: QwenRuntimeConfig
    frozen_g1: FrozenG1RuntimeConfig

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("schema_version must equal 1")
        if self.profile != PRODUCTION_PROFILE:
            raise ValueError("profile must equal production")
        object.__setattr__(
            self, "cache_root", _frozen_path(self.cache_root, "cache_root")
        )
        object.__setattr__(
            self, "output_root", _frozen_path(self.output_root, "output_root")
        )
        if self.cache_root == self.output_root:
            raise ValueError("cache_root and output_root must be distinct")
        nested_types = (
            ("whisper", self.whisper, WhisperRuntimeConfig),
            ("ocr", self.ocr, OCRRuntimeConfig),
            ("siglip", self.siglip, SigLIPRuntimeConfig),
            ("qwen", self.qwen, QwenRuntimeConfig),
            ("frozen_g1", self.frozen_g1, FrozenG1RuntimeConfig),
        )
        for field_name, value, expected_type in nested_types:
            if not isinstance(value, expected_type):
                raise ValueError(f"{field_name} has the wrong configuration type")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Optional[Path] = None,
    ) -> "ProductionRuntimeConfig":
        root = Path.cwd().resolve() if base_dir is None else Path(base_dir).resolve()
        top = _strict_section(
            payload,
            "top-level",
            {
                "schema_version",
                "profile",
                "cache_root",
                "output_root",
                "whisper",
                "ocr",
                "siglip",
                "qwen",
                "frozen_g1",
            },
        )
        whisper = _strict_section(
            top["whisper"], "whisper", {"model_path", "device", "dtype"}
        )
        ocr = _strict_section(
            top["ocr"],
            "ocr",
            {
                "detector_model_path",
                "recognizer_model_path",
                "python_executable",
                "device",
                "cudnn8_library_path",
                "timeout_seconds",
            },
        )
        siglip = _strict_section(
            top["siglip"], "siglip", {"model_path", "device"}
        )
        qwen = _strict_section(
            top["qwen"], "qwen", {"model_path", "device"}
        )
        frozen_g1 = _strict_section(
            top["frozen_g1"],
            "frozen_g1",
            {
                "unirumor_root",
                "python_executable",
                "phase4a_infer",
                "phase4a_config",
                "device",
                "timeout_seconds",
            },
        )
        return cls(
            schema_version=top["schema_version"],
            profile=top["profile"],
            cache_root=_absolute_path(top["cache_root"], "cache_root", root),
            output_root=_absolute_path(top["output_root"], "output_root", root),
            whisper=WhisperRuntimeConfig(
                model_path=_absolute_path(
                    whisper["model_path"], "whisper.model_path", root
                ),
                device=whisper["device"],
                dtype=whisper["dtype"],
            ),
            ocr=OCRRuntimeConfig(
                detector_model_path=_absolute_path(
                    ocr["detector_model_path"], "ocr.detector_model_path", root
                ),
                recognizer_model_path=_absolute_path(
                    ocr["recognizer_model_path"],
                    "ocr.recognizer_model_path",
                    root,
                ),
                python_executable=ocr["python_executable"],
                device=ocr["device"],
                cudnn8_library_path=_absolute_path(
                    ocr["cudnn8_library_path"],
                    "ocr.cudnn8_library_path",
                    root,
                ),
                timeout_seconds=ocr["timeout_seconds"],
            ),
            siglip=SigLIPRuntimeConfig(
                model_path=_absolute_path(
                    siglip["model_path"], "siglip.model_path", root
                ),
                device=siglip["device"],
            ),
            qwen=QwenRuntimeConfig(
                model_path=_absolute_path(
                    qwen["model_path"], "qwen.model_path", root
                ),
                device=qwen["device"],
            ),
            frozen_g1=FrozenG1RuntimeConfig(
                unirumor_root=_absolute_path(
                    frozen_g1["unirumor_root"],
                    "frozen_g1.unirumor_root",
                    root,
                ),
                python_executable=frozen_g1["python_executable"],
                phase4a_infer=_absolute_path(
                    frozen_g1["phase4a_infer"],
                    "frozen_g1.phase4a_infer",
                    root,
                ),
                phase4a_config=_absolute_path(
                    frozen_g1["phase4a_config"],
                    "frozen_g1.phase4a_config",
                    root,
                ),
                device=frozen_g1["device"],
                timeout_seconds=frozen_g1["timeout_seconds"],
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> "ProductionRuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed production runtime JSON: {config_path}") from exc
        return cls.from_dict(payload, base_dir=config_path.parent.parent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "cache_root": str(self.cache_root),
            "output_root": str(self.output_root),
            "whisper": {
                "model_path": str(self.whisper.model_path),
                "device": self.whisper.device,
                "dtype": self.whisper.dtype,
            },
            "ocr": {
                "detector_model_path": str(self.ocr.detector_model_path),
                "recognizer_model_path": str(self.ocr.recognizer_model_path),
                "python_executable": self.ocr.python_executable,
                "device": self.ocr.device,
                "cudnn8_library_path": str(self.ocr.cudnn8_library_path),
                "timeout_seconds": self.ocr.timeout_seconds,
            },
            "siglip": {
                "model_path": str(self.siglip.model_path),
                "device": self.siglip.device,
            },
            "qwen": {
                "model_path": str(self.qwen.model_path),
                "device": self.qwen.device,
            },
            "frozen_g1": {
                "unirumor_root": str(self.frozen_g1.unirumor_root),
                "python_executable": self.frozen_g1.python_executable,
                "phase4a_infer": str(self.frozen_g1.phase4a_infer),
                "phase4a_config": str(self.frozen_g1.phase4a_config),
                "device": self.frozen_g1.device,
                "timeout_seconds": self.frozen_g1.timeout_seconds,
            },
        }
