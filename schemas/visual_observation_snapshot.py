"""Immutable, prediction-free snapshots of visual RuntimeUnit observations."""

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Mapping, Tuple

from .grounded_visual_unit import GroundedFrameReference
from .provenance import UnitProvenance
from .unit import RuntimeUnit, SourceType


DEFAULT_RETRIEVAL_POLICY_ID = "claim_conditioned_siglip_top4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DETAIL_FIELDS = {
    "observation_type",
    "frame_ids",
    "evidence_refs",
    "referenced_frames",
    "siglip_model_id",
    "siglip_revision",
    "qwen_model_id",
    "qwen_revision",
    "prompt_policy",
    "recovery_mode",
    "raw_generation_sha256",
}
_OPTIONAL_DETAIL_FIELDS = {
    "retrieval_policy_id",
    "observer_policy_id",
}


def _require_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_timestamp(value: Any, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _require_string_tuple(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class VisualObservationSnapshot:
    """Safe projection containing only metadata needed by shadow grounding."""

    unit_id: str
    observation_text: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    primary_frame_id: str
    frame_references: Tuple[GroundedFrameReference, ...]
    source_index: int
    extraction_method: str
    observation_type: str
    frame_ids: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    siglip_model_id: str
    siglip_revision: str
    qwen_model_id: str
    qwen_revision: str
    prompt_policy: str
    raw_generation_sha256: str
    recovery_mode: str
    retrieval_policy_id: str
    observer_policy_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "unit_id",
            "observation_text",
            "primary_frame_id",
            "extraction_method",
            "observation_type",
            "siglip_model_id",
            "siglip_revision",
            "qwen_model_id",
            "qwen_revision",
            "prompt_policy",
            "recovery_mode",
            "retrieval_policy_id",
            "observer_policy_id",
        ):
            _require_string(getattr(self, field_name), field_name)
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
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("source_index must be a non-negative integer")
        _require_string_tuple(self.frame_ids, "frame_ids")
        _require_string_tuple(self.evidence_refs, "evidence_refs")
        if not isinstance(self.raw_generation_sha256, str) or _SHA256.fullmatch(
            self.raw_generation_sha256
        ) is None:
            raise ValueError(
                "raw_generation_sha256 must be a lowercase SHA-256 hex digest"
            )

        reference_ids = tuple(
            reference.frame_id for reference in self.frame_references
        )
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("frame_references must have unique frame IDs")
        if self.primary_frame_id not in set(reference_ids):
            raise ValueError("primary_frame_id must identify a referenced frame")
        if set(reference_ids) != set(self.frame_ids) | set(self.evidence_refs):
            raise ValueError(
                "frame_references must exactly cover frame_ids and evidence_refs"
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

    @staticmethod
    def _invalid(index: int, message: str) -> ValueError:
        return ValueError(f"visual_units[{index}] {message}")

    @classmethod
    def _string(cls, value: Any, index: int, field_name: str) -> str:
        try:
            _require_string(value, field_name)
        except ValueError as exc:
            raise cls._invalid(index, str(exc)) from exc
        return value

    @classmethod
    def _string_tuple(
        cls, value: Any, index: int, field_name: str
    ) -> Tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise cls._invalid(index, f"{field_name} must be a non-empty sequence")
        result = tuple(value)
        for item_index, item in enumerate(result):
            cls._string(item, index, f"{field_name}[{item_index}]")
        return result

    @classmethod
    def _frame_references(
        cls, details: Mapping[str, Any], index: int
    ) -> Tuple[GroundedFrameReference, ...]:
        raw_frames = details["referenced_frames"]
        if not isinstance(raw_frames, (list, tuple)) or not raw_frames:
            raise cls._invalid(index, "referenced_frames must be a non-empty sequence")
        references = []
        required = {
            "frame_id",
            "frame_index",
            "timestamp_sec",
            "frame_rank",
            "retrieval_rank",
            "image_sha256",
        }
        for frame_index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, Mapping):
                raise cls._invalid(
                    index, f"referenced_frames[{frame_index}] must be a mapping"
                )
            missing = sorted(required - set(raw_frame))
            if missing:
                raise cls._invalid(
                    index,
                    f"referenced_frames[{frame_index}] is missing metadata: {missing}",
                )
            try:
                references.append(
                    GroundedFrameReference(
                        frame_id=raw_frame["frame_id"],
                        frame_index=raw_frame["frame_index"],
                        timestamp_seconds=raw_frame["timestamp_sec"],
                        frame_rank=raw_frame["frame_rank"],
                        retrieval_rank=raw_frame["retrieval_rank"],
                        image_sha256=raw_frame["image_sha256"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise cls._invalid(
                    index,
                    f"referenced_frames[{frame_index}] is invalid: {exc}",
                ) from exc
        return tuple(references)

    @classmethod
    def from_runtime_unit(
        cls, unit: RuntimeUnit, *, index: int = 0
    ) -> "VisualObservationSnapshot":
        """Copy a validated visual observation without retaining mutable aliases."""

        if not isinstance(unit, RuntimeUnit):
            raise TypeError(
                f"visual_units[{index}] must be an existing RuntimeUnit visual observation"
            )
        if unit.source_type is not SourceType.VISUAL_OBSERVATION:
            source_name = getattr(unit.source_type, "value", repr(unit.source_type))
            raise cls._invalid(
                index,
                "must have source_type visual_observation; "
                f"received {source_name!r}",
            )
        if unit.eligible_for_frozen_g1:
            raise cls._invalid(index, "must remain ineligible for Frozen G1")
        if unit.selection_score is not None:
            raise cls._invalid(index, "must not carry selection_score")
        if unit.logits is not None:
            raise cls._invalid(index, "must not carry logits")
        if unit.confidence is not None:
            raise cls._invalid(index, "must not carry confidence")
        if unit.bbox is not None:
            raise cls._invalid(index, "must not carry unsupported bbox provenance")
        if unit.start_time is None or unit.end_time is None:
            raise cls._invalid(index, "must provide start_time and end_time")
        if not isinstance(unit.provenance, UnitProvenance):
            raise cls._invalid(index, "must provide UnitProvenance")
        if unit.provenance.source_index is None:
            raise cls._invalid(index, "provenance.source_index is required")
        if not isinstance(unit.provenance.details, Mapping):
            raise cls._invalid(index, "provenance.details must be a mapping")

        details = unit.provenance.details
        missing = sorted(_REQUIRED_DETAIL_FIELDS - set(details))
        unexpected = sorted(
            set(details) - _REQUIRED_DETAIL_FIELDS - _OPTIONAL_DETAIL_FIELDS
        )
        if missing or unexpected:
            raise cls._invalid(
                index,
                "provenance details do not match the visual observation contract; "
                f"missing={missing}, unexpected={unexpected}",
            )

        unit_id = cls._string(unit.unit_id, index, "unit_id")
        observation_text = cls._string(unit.text, index, "text")
        primary_frame_id = cls._string(unit.frame_id, index, "frame_id")
        extraction_method = cls._string(
            unit.provenance.extraction_method,
            index,
            "provenance.extraction_method",
        )
        frame_ids = cls._string_tuple(details["frame_ids"], index, "frame_ids")
        evidence_refs = cls._string_tuple(
            details["evidence_refs"], index, "evidence_refs"
        )
        frame_references = cls._frame_references(details, index)
        qwen_model_id = cls._string(
            details["qwen_model_id"], index, "qwen_model_id"
        )
        if unit.producer != qwen_model_id:
            raise cls._invalid(index, "producer must match provenance qwen_model_id")

        try:
            return cls(
                unit_id=unit_id,
                observation_text=observation_text,
                start_timestamp_seconds=unit.start_time,
                end_timestamp_seconds=unit.end_time,
                primary_frame_id=primary_frame_id,
                frame_references=frame_references,
                source_index=unit.provenance.source_index,
                extraction_method=extraction_method,
                observation_type=cls._string(
                    details["observation_type"], index, "observation_type"
                ),
                frame_ids=frame_ids,
                evidence_refs=evidence_refs,
                siglip_model_id=cls._string(
                    details["siglip_model_id"], index, "siglip_model_id"
                ),
                siglip_revision=cls._string(
                    details["siglip_revision"], index, "siglip_revision"
                ),
                qwen_model_id=qwen_model_id,
                qwen_revision=cls._string(
                    details["qwen_revision"], index, "qwen_revision"
                ),
                prompt_policy=cls._string(
                    details["prompt_policy"], index, "prompt_policy"
                ),
                raw_generation_sha256=cls._string(
                    details["raw_generation_sha256"],
                    index,
                    "raw_generation_sha256",
                ),
                recovery_mode=cls._string(
                    details["recovery_mode"], index, "recovery_mode"
                ),
                retrieval_policy_id=cls._string(
                    details.get(
                        "retrieval_policy_id", DEFAULT_RETRIEVAL_POLICY_ID
                    ),
                    index,
                    "retrieval_policy_id",
                ),
                observer_policy_id=cls._string(
                    details.get("observer_policy_id", details["prompt_policy"]),
                    index,
                    "observer_policy_id",
                ),
            )
        except (TypeError, ValueError) as exc:
            if str(exc).startswith(f"visual_units[{index}]"):
                raise
            raise cls._invalid(index, f"cannot be snapshotted: {exc}") from exc

    def source_identity_payload(self) -> Dict[str, Any]:
        """Return the path-free deterministic payload used for lineage hashing."""

        return {
            "source_observation_id": self.unit_id,
            "text_observation": self.observation_text,
            "start_timestamp_seconds": self.start_timestamp_seconds,
            "end_timestamp_seconds": self.end_timestamp_seconds,
            "producer": self.qwen_model_id,
            "source_index": self.source_index,
            "extraction_method": self.extraction_method,
            "observation_type": self.observation_type,
            "frame_ids": list(self.frame_ids),
            "evidence_refs": list(self.evidence_refs),
            "frame_references": [
                reference.to_dict() for reference in self.frame_references
            ],
            "siglip_model_id": self.siglip_model_id,
            "siglip_revision": self.siglip_revision,
            "qwen_model_id": self.qwen_model_id,
            "qwen_revision": self.qwen_revision,
            "prompt_policy": self.prompt_policy,
            "recovery_mode": self.recovery_mode,
            "raw_generation_sha256": self.raw_generation_sha256,
            "retrieval_policy_id": self.retrieval_policy_id,
            "observer_policy_id": self.observer_policy_id,
        }
