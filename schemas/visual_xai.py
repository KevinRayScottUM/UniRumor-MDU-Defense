"""Immutable, prediction-free contracts for supplemental visual attribution."""

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


VISUAL_XAI_SCHEMA_VERSION = "1"
VISUAL_XAI_ARTIFACT_TYPE = "visual_observation_attribution"
QWEN_OCCLUSION_METHOD = "qwen_occlusion_logprob_v1"
QWEN_OCCLUSION_BASELINE = "gaussian_blur_region_v1"
VISUAL_XAI_DISCLAIMER = (
    "This is a post-hoc perturbation attribution of the Visual Observer. "
    "It does not affect the authoritative verification verdict."
)
VISUAL_XAI_BOUNDARY = (
    "Supplemental visual XAI is explanatory only and does not participate "
    "in the Frozen G1 verdict."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUS_VALUES = {"available", "unavailable"}
_SCOPE_VALUES = {"observation", "phrase"}


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _matrix(
    value: Any,
    field_name: str,
    *,
    rows: int,
    columns: int,
    normalized: bool = False,
) -> Tuple[Tuple[float, ...], ...]:
    if not isinstance(value, tuple) or len(value) != rows:
        raise ValueError(f"{field_name} must contain exactly {rows} rows")
    output = []
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != columns:
            raise ValueError(
                f"{field_name}[{row_index}] must contain exactly {columns} columns"
            )
        normalized_row = tuple(
            _finite_number(item, f"{field_name}[{row_index}]") for item in row
        )
        if normalized and any(not 0.0 <= item <= 1.0 for item in normalized_row):
            raise ValueError(f"{field_name} values must be within [0, 1]")
        output.append(normalized_row)
    return tuple(output)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class VisualTargetSpan:
    """A character span inside the fixed original Qwen generation target."""

    span_id: str
    scope: str
    label: str
    start_character: int
    end_character: int

    def __post_init__(self) -> None:
        _require_string(self.span_id, "span_id")
        if self.scope not in _SCOPE_VALUES:
            raise ValueError("scope must be observation or phrase")
        _require_string(self.label, "label")
        if type(self.start_character) is not int or self.start_character < 0:
            raise ValueError("start_character must be a non-negative integer")
        if (
            type(self.end_character) is not int
            or self.end_character <= self.start_character
        ):
            raise ValueError("end_character must follow start_character")


@dataclass(frozen=True, slots=True)
class VisualTargetScore:
    """Teacher-forced log-probability sums for requested target spans."""

    span_log_probabilities: Tuple[Tuple[str, float], ...]
    span_token_counts: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.span_log_probabilities, tuple):
            raise TypeError("span_log_probabilities must be a tuple")
        if not isinstance(self.span_token_counts, tuple):
            raise TypeError("span_token_counts must be a tuple")
        probability_ids = []
        for span_id, value in self.span_log_probabilities:
            probability_ids.append(_require_string(span_id, "span_id"))
            _finite_number(value, "span_log_probability")
        count_ids = []
        for span_id, value in self.span_token_counts:
            count_ids.append(_require_string(span_id, "span_id"))
            if type(value) is not int or value < 1:
                raise ValueError("span_token_count must be a positive integer")
        if len(set(probability_ids)) != len(probability_ids):
            raise ValueError("span_log_probabilities contains duplicate span IDs")
        if set(probability_ids) != set(count_ids):
            raise ValueError("score and token-count span IDs must match")

    def log_probability(self, span_id: str) -> float:
        return dict(self.span_log_probabilities)[span_id]

    def token_count(self, span_id: str) -> int:
        return dict(self.span_token_counts)[span_id]


@dataclass(frozen=True, slots=True)
class VisualAttributionMap:
    """One sentence- or phrase-level spatial perturbation attribution map."""

    map_id: str
    scope: str
    label: str
    target_start_character: int
    target_end_character: int
    target_token_count: int
    baseline_target_log_probability: float
    raw_importance: Tuple[Tuple[float, ...], ...]
    normalized_importance: Tuple[Tuple[float, ...], ...]
    overlay_image_path: Optional[Path] = None

    def __post_init__(self) -> None:
        _require_string(self.map_id, "map_id")
        if self.scope not in _SCOPE_VALUES:
            raise ValueError("scope must be observation or phrase")
        _require_string(self.label, "label")
        if (
            type(self.target_start_character) is not int
            or self.target_start_character < 0
        ):
            raise ValueError("target_start_character must be non-negative")
        if (
            type(self.target_end_character) is not int
            or self.target_end_character <= self.target_start_character
        ):
            raise ValueError("target_end_character must follow target_start_character")
        if type(self.target_token_count) is not int or self.target_token_count < 1:
            raise ValueError("target_token_count must be positive")
        object.__setattr__(
            self,
            "baseline_target_log_probability",
            _finite_number(
                self.baseline_target_log_probability,
                "baseline_target_log_probability",
            ),
        )
        rows = len(self.raw_importance)
        if rows < 1:
            raise ValueError("raw_importance must not be empty")
        columns = len(self.raw_importance[0]) if self.raw_importance[0] else 0
        if columns < 1:
            raise ValueError("raw_importance rows must not be empty")
        object.__setattr__(
            self,
            "raw_importance",
            _matrix(
                self.raw_importance,
                "raw_importance",
                rows=rows,
                columns=columns,
            ),
        )
        object.__setattr__(
            self,
            "normalized_importance",
            _matrix(
                self.normalized_importance,
                "normalized_importance",
                rows=rows,
                columns=columns,
                normalized=True,
            ),
        )
        if self.overlay_image_path is not None:
            if not isinstance(self.overlay_image_path, Path):
                raise TypeError("overlay_image_path must be a Path or None")
            object.__setattr__(self, "overlay_image_path", Path(self.overlay_image_path))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize public-safe metadata; local image paths are deliberately absent."""

        return {
            "map_id": self.map_id,
            "scope": self.scope,
            "label": self.label,
            "target_start_character": self.target_start_character,
            "target_end_character": self.target_end_character,
            "target_token_count": self.target_token_count,
            "baseline_target_log_probability": self.baseline_target_log_probability,
            "raw_importance": [list(row) for row in self.raw_importance],
            "normalized_importance": [
                list(row) for row in self.normalized_importance
            ],
        }


@dataclass(frozen=True, slots=True)
class VisualAttributionArtifact:
    """Auditable supplemental artifact with no prediction semantics."""

    schema_version: str
    artifact_type: str
    artifact_id: str
    status: str
    unavailable_reason: Optional[str]
    method: str
    model_id: str
    model_revision: str
    model_fingerprint: str
    source_frame_id: str
    source_frame_index: int
    source_timestamp_seconds: float
    source_frame_sha256: str
    observation_unit_id: str
    observation_text: str
    observation_text_sha256: str
    raw_generation_sha256: str
    grid_rows: int
    grid_columns: int
    occlusion_baseline: str
    configuration_version: str
    phrase_policy: str
    maps: Tuple[VisualAttributionMap, ...]
    cache_key: str

    def __post_init__(self) -> None:
        if self.schema_version != VISUAL_XAI_SCHEMA_VERSION:
            raise ValueError("invalid visual XAI schema version")
        if self.artifact_type != VISUAL_XAI_ARTIFACT_TYPE:
            raise ValueError("invalid visual XAI artifact type")
        _require_sha256(self.artifact_id, "artifact_id")
        if self.status not in _STATUS_VALUES:
            raise ValueError("status must be available or unavailable")
        if self.status == "available":
            if self.unavailable_reason is not None or not self.maps:
                raise ValueError("available attribution requires maps and no reason")
        elif self.unavailable_reason is None or self.maps:
            raise ValueError("unavailable attribution requires a reason and no maps")
        if self.unavailable_reason is not None:
            _require_string(self.unavailable_reason, "unavailable_reason")
        for field_name in (
            "method",
            "model_id",
            "model_revision",
            "model_fingerprint",
            "source_frame_id",
            "observation_unit_id",
            "observation_text",
            "occlusion_baseline",
            "configuration_version",
            "phrase_policy",
        ):
            _require_string(getattr(self, field_name), field_name)
        _require_sha256(self.source_frame_sha256, "source_frame_sha256")
        _require_sha256(self.observation_text_sha256, "observation_text_sha256")
        _require_sha256(self.raw_generation_sha256, "raw_generation_sha256")
        _require_sha256(self.cache_key, "cache_key")
        if type(self.source_frame_index) is not int or self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative")
        object.__setattr__(
            self,
            "source_timestamp_seconds",
            _finite_number(self.source_timestamp_seconds, "source_timestamp_seconds"),
        )
        if self.source_timestamp_seconds < 0:
            raise ValueError("source_timestamp_seconds must be non-negative")
        for field_name in ("grid_rows", "grid_columns"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.maps, tuple) or not all(
            isinstance(item, VisualAttributionMap) for item in self.maps
        ):
            raise TypeError("maps must contain only VisualAttributionMap values")
        if self.maps:
            ids = [item.map_id for item in self.maps]
            if len(ids) != len(set(ids)):
                raise ValueError("attribution map IDs must be unique")
            for item in self.maps:
                if (
                    len(item.raw_importance) != self.grid_rows
                    or len(item.raw_importance[0]) != self.grid_columns
                ):
                    raise ValueError("attribution map dimensions do not match artifact")

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "status": self.status,
            "unavailable_reason": self.unavailable_reason,
            "method": self.method,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "source_frame_id": self.source_frame_id,
            "source_frame_index": self.source_frame_index,
            "source_timestamp_seconds": self.source_timestamp_seconds,
            "source_frame_sha256": self.source_frame_sha256,
            "observation_unit_id": self.observation_unit_id,
            "observation_text": self.observation_text,
            "observation_text_sha256": self.observation_text_sha256,
            "raw_generation_sha256": self.raw_generation_sha256,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "occlusion_baseline": self.occlusion_baseline,
            "configuration_version": self.configuration_version,
            "phrase_policy": self.phrase_policy,
            "maps": [item.to_dict() for item in self.maps],
            "cache_key": self.cache_key,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"artifact_id": self.artifact_id, **self.identity_payload()}

    @staticmethod
    def compute_identity(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
