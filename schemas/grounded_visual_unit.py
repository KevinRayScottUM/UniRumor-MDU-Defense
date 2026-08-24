"""Immutable shadow-only visual-grounding audit contracts.

These objects are deliberately separate from ``RuntimeUnit`` and contain no
prediction, eligibility, scoring, or verdict fields.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Tuple


GROUNDED_VISUAL_SCHEMA_VERSION = "1"
GROUNDED_VISUAL_ARTIFACT_TYPE = "grounded_visual_unit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNIT_ID = re.compile(r"^gvu_[0-9a-f]{64}$")


def _require_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_nonnegative_integer(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_positive_integer(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_timestamp(value: Any, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _require_sha256(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_string_tuple(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


def _require_exact_keys(
    data: Mapping[str, Any], expected: Tuple[str, ...], type_name: str
) -> None:
    actual = set(data)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unexpected = sorted(actual - expected_set)
        raise ValueError(
            f"{type_name} fields do not match the schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class GroundedFrameReference:
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    frame_rank: int
    retrieval_rank: int
    image_sha256: str

    def __post_init__(self) -> None:
        _require_string(self.frame_id, "frame_id")
        _require_nonnegative_integer(self.frame_index, "frame_index")
        object.__setattr__(
            self,
            "timestamp_seconds",
            _require_timestamp(self.timestamp_seconds, "timestamp_seconds"),
        )
        _require_nonnegative_integer(self.frame_rank, "frame_rank")
        _require_positive_integer(self.retrieval_rank, "retrieval_rank")
        _require_sha256(self.image_sha256, "image_sha256")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_rank": self.frame_rank,
            "retrieval_rank": self.retrieval_rank,
            "image_sha256": self.image_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundedFrameReference":
        expected = (
            "frame_id",
            "frame_index",
            "timestamp_seconds",
            "frame_rank",
            "retrieval_rank",
            "image_sha256",
        )
        _require_exact_keys(data, expected, cls.__name__)
        return cls(**{field_name: data[field_name] for field_name in expected})


@dataclass(frozen=True)
class GroundingModelIdentity:
    siglip_model_id: str
    siglip_revision: str
    qwen_model_id: str
    qwen_revision: str
    adapter_id: str
    adapter_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "siglip_model_id",
            "siglip_revision",
            "qwen_model_id",
            "qwen_revision",
            "adapter_id",
            "adapter_version",
        ):
            _require_string(getattr(self, field_name), field_name)

    def to_dict(self) -> Dict[str, str]:
        return {
            "siglip_model_id": self.siglip_model_id,
            "siglip_revision": self.siglip_revision,
            "qwen_model_id": self.qwen_model_id,
            "qwen_revision": self.qwen_revision,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundingModelIdentity":
        expected = (
            "siglip_model_id",
            "siglip_revision",
            "qwen_model_id",
            "qwen_revision",
            "adapter_id",
            "adapter_version",
        )
        _require_exact_keys(data, expected, cls.__name__)
        return cls(**{field_name: data[field_name] for field_name in expected})


@dataclass(frozen=True)
class GroundingLineage:
    source_observation_id: str
    source_observation_sha256: str
    source_index: int
    extraction_method: str
    observation_type: str
    frame_ids: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    raw_generation_sha256: str
    recovery_mode: str
    retrieval_policy_id: str
    observer_policy_id: str

    def __post_init__(self) -> None:
        _require_string(self.source_observation_id, "source_observation_id")
        _require_sha256(self.source_observation_sha256, "source_observation_sha256")
        _require_nonnegative_integer(self.source_index, "source_index")
        for field_name in (
            "extraction_method",
            "observation_type",
            "recovery_mode",
            "retrieval_policy_id",
            "observer_policy_id",
        ):
            _require_string(getattr(self, field_name), field_name)
        _require_string_tuple(self.frame_ids, "frame_ids")
        _require_string_tuple(self.evidence_refs, "evidence_refs")
        _require_sha256(self.raw_generation_sha256, "raw_generation_sha256")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_observation_id": self.source_observation_id,
            "source_observation_sha256": self.source_observation_sha256,
            "source_index": self.source_index,
            "extraction_method": self.extraction_method,
            "observation_type": self.observation_type,
            "frame_ids": list(self.frame_ids),
            "evidence_refs": list(self.evidence_refs),
            "raw_generation_sha256": self.raw_generation_sha256,
            "recovery_mode": self.recovery_mode,
            "retrieval_policy_id": self.retrieval_policy_id,
            "observer_policy_id": self.observer_policy_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundingLineage":
        expected = (
            "source_observation_id",
            "source_observation_sha256",
            "source_index",
            "extraction_method",
            "observation_type",
            "frame_ids",
            "evidence_refs",
            "raw_generation_sha256",
            "recovery_mode",
            "retrieval_policy_id",
            "observer_policy_id",
        )
        _require_exact_keys(data, expected, cls.__name__)
        return cls(
            source_observation_id=data["source_observation_id"],
            source_observation_sha256=data["source_observation_sha256"],
            source_index=data["source_index"],
            extraction_method=data["extraction_method"],
            observation_type=data["observation_type"],
            frame_ids=tuple(data["frame_ids"]),
            evidence_refs=tuple(data["evidence_refs"]),
            raw_generation_sha256=data["raw_generation_sha256"],
            recovery_mode=data["recovery_mode"],
            retrieval_policy_id=data["retrieval_policy_id"],
            observer_policy_id=data["observer_policy_id"],
        )


@dataclass(frozen=True)
class GroundedVisualUnit:
    schema_version: str
    artifact_type: str
    unit_id: str
    source_observation_id: str
    text_observation: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    frame_references: Tuple[GroundedFrameReference, ...]
    model_identity: GroundingModelIdentity
    prompt_policy: str
    lineage: GroundingLineage

    def __post_init__(self) -> None:
        if self.schema_version != GROUNDED_VISUAL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {GROUNDED_VISUAL_SCHEMA_VERSION!r}"
            )
        if self.artifact_type != GROUNDED_VISUAL_ARTIFACT_TYPE:
            raise ValueError(
                f"artifact_type must equal {GROUNDED_VISUAL_ARTIFACT_TYPE!r}"
            )
        if not isinstance(self.unit_id, str) or _UNIT_ID.fullmatch(self.unit_id) is None:
            raise ValueError("unit_id must be gvu_ followed by a SHA-256 hex digest")
        _require_string(self.source_observation_id, "source_observation_id")
        _require_string(self.text_observation, "text_observation")
        object.__setattr__(
            self,
            "start_timestamp_seconds",
            _require_timestamp(
                self.start_timestamp_seconds, "start_timestamp_seconds"
            ),
        )
        object.__setattr__(
            self,
            "end_timestamp_seconds",
            _require_timestamp(self.end_timestamp_seconds, "end_timestamp_seconds"),
        )
        if self.end_timestamp_seconds < self.start_timestamp_seconds:
            raise ValueError(
                "end_timestamp_seconds must not precede start_timestamp_seconds"
            )
        if not isinstance(self.frame_references, tuple) or not self.frame_references:
            raise ValueError("frame_references must be a non-empty tuple")
        if not all(
            isinstance(reference, GroundedFrameReference)
            for reference in self.frame_references
        ):
            raise TypeError(
                "frame_references must contain only GroundedFrameReference objects"
            )
        if not isinstance(self.model_identity, GroundingModelIdentity):
            raise TypeError("model_identity must be a GroundingModelIdentity")
        _require_string(self.prompt_policy, "prompt_policy")
        if not isinstance(self.lineage, GroundingLineage):
            raise TypeError("lineage must be a GroundingLineage")
        if self.source_observation_id != self.lineage.source_observation_id:
            raise ValueError(
                "source_observation_id must match lineage.source_observation_id"
            )
        frame_ids = tuple(reference.frame_id for reference in self.frame_references)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame_references must have unique frame IDs")
        referenced_ids = set(frame_ids)
        lineage_ids = set(self.lineage.frame_ids) | set(self.lineage.evidence_refs)
        if referenced_ids != lineage_ids:
            raise ValueError(
                "frame_references must exactly cover lineage frame IDs and evidence refs"
            )
        frame_timestamps = tuple(
            reference.timestamp_seconds for reference in self.frame_references
        )
        if self.start_timestamp_seconds != min(frame_timestamps):
            raise ValueError(
                "start_timestamp_seconds must match the earliest frame reference"
            )
        if self.end_timestamp_seconds != max(frame_timestamps):
            raise ValueError(
                "end_timestamp_seconds must match the latest frame reference"
            )
        expected_unit_id = self.deterministic_unit_id(
            schema_version=self.schema_version,
            artifact_type=self.artifact_type,
            source_observation_id=self.source_observation_id,
            text_observation=self.text_observation,
            start_timestamp_seconds=self.start_timestamp_seconds,
            end_timestamp_seconds=self.end_timestamp_seconds,
            frame_references=self.frame_references,
            model_identity=self.model_identity,
            prompt_policy=self.prompt_policy,
            lineage=self.lineage,
        )
        if self.unit_id != expected_unit_id:
            raise ValueError("unit_id does not match the deterministic content identity")

    @staticmethod
    def _identity_payload(
        *,
        schema_version: str,
        artifact_type: str,
        source_observation_id: str,
        text_observation: str,
        start_timestamp_seconds: float,
        end_timestamp_seconds: float,
        frame_references: Tuple[GroundedFrameReference, ...],
        model_identity: GroundingModelIdentity,
        prompt_policy: str,
        lineage: GroundingLineage,
    ) -> Dict[str, Any]:
        return {
            "schema_version": schema_version,
            "artifact_type": artifact_type,
            "source_observation_id": source_observation_id,
            "text_observation": text_observation,
            "start_timestamp_seconds": float(start_timestamp_seconds),
            "end_timestamp_seconds": float(end_timestamp_seconds),
            "frame_references": [
                reference.to_dict() for reference in frame_references
            ],
            "model_identity": model_identity.to_dict(),
            "prompt_policy": prompt_policy,
            "lineage": lineage.to_dict(),
        }

    @classmethod
    def deterministic_unit_id(
        cls,
        *,
        schema_version: str,
        artifact_type: str,
        source_observation_id: str,
        text_observation: str,
        start_timestamp_seconds: float,
        end_timestamp_seconds: float,
        frame_references: Tuple[GroundedFrameReference, ...],
        model_identity: GroundingModelIdentity,
        prompt_policy: str,
        lineage: GroundingLineage,
    ) -> str:
        payload = cls._identity_payload(
            schema_version=schema_version,
            artifact_type=artifact_type,
            source_observation_id=source_observation_id,
            text_observation=text_observation,
            start_timestamp_seconds=start_timestamp_seconds,
            end_timestamp_seconds=end_timestamp_seconds,
            frame_references=frame_references,
            model_identity=model_identity,
            prompt_policy=prompt_policy,
            lineage=lineage,
        )
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"gvu_{digest}"

    @classmethod
    def create(
        cls,
        *,
        source_observation_id: str,
        text_observation: str,
        start_timestamp_seconds: float,
        end_timestamp_seconds: float,
        frame_references: Tuple[GroundedFrameReference, ...],
        model_identity: GroundingModelIdentity,
        prompt_policy: str,
        lineage: GroundingLineage,
        schema_version: str = GROUNDED_VISUAL_SCHEMA_VERSION,
        artifact_type: str = GROUNDED_VISUAL_ARTIFACT_TYPE,
    ) -> "GroundedVisualUnit":
        unit_id = cls.deterministic_unit_id(
            schema_version=schema_version,
            artifact_type=artifact_type,
            source_observation_id=source_observation_id,
            text_observation=text_observation,
            start_timestamp_seconds=start_timestamp_seconds,
            end_timestamp_seconds=end_timestamp_seconds,
            frame_references=frame_references,
            model_identity=model_identity,
            prompt_policy=prompt_policy,
            lineage=lineage,
        )
        return cls(
            schema_version=schema_version,
            artifact_type=artifact_type,
            unit_id=unit_id,
            source_observation_id=source_observation_id,
            text_observation=text_observation,
            start_timestamp_seconds=start_timestamp_seconds,
            end_timestamp_seconds=end_timestamp_seconds,
            frame_references=frame_references,
            model_identity=model_identity,
            prompt_policy=prompt_policy,
            lineage=lineage,
        )

    def identity_sha256(self) -> str:
        return self.unit_id.removeprefix("gvu_")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "unit_id": self.unit_id,
            "source_observation_id": self.source_observation_id,
            "text_observation": self.text_observation,
            "start_timestamp_seconds": self.start_timestamp_seconds,
            "end_timestamp_seconds": self.end_timestamp_seconds,
            "frame_references": [
                reference.to_dict() for reference in self.frame_references
            ],
            "model_identity": self.model_identity.to_dict(),
            "prompt_policy": self.prompt_policy,
            "lineage": self.lineage.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundedVisualUnit":
        expected = (
            "schema_version",
            "artifact_type",
            "unit_id",
            "source_observation_id",
            "text_observation",
            "start_timestamp_seconds",
            "end_timestamp_seconds",
            "frame_references",
            "model_identity",
            "prompt_policy",
            "lineage",
        )
        _require_exact_keys(data, expected, cls.__name__)
        return cls(
            schema_version=data["schema_version"],
            artifact_type=data["artifact_type"],
            unit_id=data["unit_id"],
            source_observation_id=data["source_observation_id"],
            text_observation=data["text_observation"],
            start_timestamp_seconds=data["start_timestamp_seconds"],
            end_timestamp_seconds=data["end_timestamp_seconds"],
            frame_references=tuple(
                GroundedFrameReference.from_dict(item)
                for item in data["frame_references"]
            ),
            model_identity=GroundingModelIdentity.from_dict(data["model_identity"]),
            prompt_policy=data["prompt_policy"],
            lineage=GroundingLineage.from_dict(data["lineage"]),
        )
