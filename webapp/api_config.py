"""Frozen server-side configuration for the Task07 HTTP boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from urllib.parse import urlsplit


DEFAULT_POLL_AFTER_MS = 3000
DEFAULT_RETRY_AFTER_SECONDS = 3
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_UPLOAD_BYTES = 90 * 1024 * 1024


def validate_web_runtime_root(value: Union[str, Path]) -> Path:
    """Return the canonical existing non-symlink directory used for ownership."""

    if not isinstance(value, (str, Path)):
        raise TypeError("web_runtime_root must be a string or Path")
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError("web_runtime_root must not be a symlink")
    if not path.exists():
        raise ValueError("web_runtime_root must exist")
    if not path.is_dir():
        raise ValueError("web_runtime_root must be a directory")
    return path.resolve(strict=True)


def _validated_origins(values: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed_origins must be an iterable of origins")
    try:
        candidates = tuple(values)
    except TypeError:
        raise TypeError("allowed_origins must be an iterable of origins") from None

    origins = []
    seen = set()
    for origin in candidates:
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("allowed_origins must not contain blank origins")
        if origin != origin.strip():
            raise ValueError("allowed_origins must use exact origins")
        if "*" in origin:
            raise ValueError("wildcard origins are forbidden")

        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("allowed_origins must contain explicit HTTP origins")
        try:
            _ = parsed.port
        except ValueError:
            raise ValueError(
                "allowed_origins must contain explicit HTTP origins"
            ) from None

        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return tuple(origins)


def _positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_seconds(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class APIConfig:
    """Deployment controls supplied only by the server operator."""

    web_runtime_root: Path
    allowed_origins: Tuple[str, ...] = ()
    poll_after_ms: int = DEFAULT_POLL_AFTER_MS
    retry_after_seconds: int = DEFAULT_RETRY_AFTER_SECONDS
    graceful_shutdown_timeout_seconds: float = (
        DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
    )
    production_runtime_config_path: Optional[Path] = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "web_runtime_root",
            validate_web_runtime_root(self.web_runtime_root),
        )
        object.__setattr__(
            self,
            "allowed_origins",
            _validated_origins(self.allowed_origins),
        )
        object.__setattr__(
            self,
            "poll_after_ms",
            _positive_integer(self.poll_after_ms, "poll_after_ms"),
        )
        object.__setattr__(
            self,
            "retry_after_seconds",
            _positive_integer(
                self.retry_after_seconds,
                "retry_after_seconds",
            ),
        )
        object.__setattr__(
            self,
            "graceful_shutdown_timeout_seconds",
            _positive_seconds(
                self.graceful_shutdown_timeout_seconds,
                "graceful_shutdown_timeout_seconds",
            ),
        )
        object.__setattr__(
            self,
            "max_upload_bytes",
            _positive_integer(self.max_upload_bytes, "max_upload_bytes"),
        )
        if self.max_upload_bytes > DEFAULT_MAX_UPLOAD_BYTES:
            raise ValueError(
                "max_upload_bytes must not exceed the 90 MiB baseline limit"
            )
        if self.production_runtime_config_path is not None:
            if not isinstance(self.production_runtime_config_path, (str, Path)):
                raise TypeError(
                    "production_runtime_config_path must be a string or Path"
                )
            object.__setattr__(
                self,
                "production_runtime_config_path",
                Path(self.production_runtime_config_path).expanduser().resolve(),
            )


WebAPIConfig = APIConfig


__all__ = [
    "APIConfig",
    "DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_POLL_AFTER_MS",
    "DEFAULT_RETRY_AFTER_SECONDS",
    "WebAPIConfig",
    "validate_web_runtime_root",
]
