"""Immutable, label-free schemas for the independent audit cohort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


IMPLEMENTATION_REVISION = "step2.6r-3b1-r1-v1"
SCHEMA_VERSION = 1
SAMPLING_SALT = "step2.6r-3b1-independent-audit-v1"
REVIEWER_A_SALT = "step2.6r-3b1-reviewer-a-v1"
REVIEWER_B_SALT = "step2.6r-3b1-reviewer-b-v1"
SUPPORTED_DATASETS = ("GroundLie360", "TRUE-3MFact")
EXPECTED_SOURCE_COUNTS = {"GroundLie360": 1636, "TRUE-3MFact": 2242}
TARGET_DATASET_COUNTS = {"GroundLie360": 15, "TRUE-3MFact": 15}
AUTHORITATIVE_TRAIN_SHA256 = (
    "e807535556441434df0ef53a37921c0bdac5e27215ed045104ac08f38275e406"
)

SEALED_CHALLENGE_IDS = frozenset(
    {
        "GroundLie360:13025004",
        "TRUE-3MFact:10145403",
        "TRUE-3MFact:10258205",
        "TRUE-3MFact:10372904",
        "TRUE-3MFact:10455808",
        "TRUE-3MFact:10865013",
    }
)
STAGE_A_IDS = frozenset(
    {
        "GroundLie360:13199900",
        "GroundLie360:13296704",
        "GroundLie360:13310803",
        "GroundLie360:13359007",
        "GroundLie360:13364604",
        "GroundLie360:13443602",
        "GroundLie360:13494602",
    }
)

PUBLIC_REVIEW_COLUMNS = (
    "review_case_id",
    "claim",
    "review_unit_id",
    "candidate_text",
    "direct_relevance_label",
    "review_confidence",
    "review_note",
)
DIRECT_RELEVANCE_LABELS = ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE")
REVIEW_CONFIDENCE_LABELS = ("HIGH", "MEDIUM", "LOW")


class AuditSchemaError(ValueError):
    """Raised when a score-free audit record violates its frozen schema."""


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditSchemaError(f"{field} must be a nonblank string")
    return value


@dataclass(frozen=True)
class AuditCandidate:
    unit_id: str
    unit_type: str
    modality: str
    text: str
    original_candidate_position: int

    def __post_init__(self) -> None:
        _nonblank(self.unit_id, "unit_id")
        _nonblank(self.unit_type, "unit_type")
        _nonblank(self.modality, "modality")
        _nonblank(self.text, "text")
        if type(self.original_candidate_position) is not int or self.original_candidate_position < 0:
            raise AuditSchemaError("original_candidate_position must be nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "modality": self.modality,
            "text": self.text,
            "original_candidate_position": self.original_candidate_position,
        }


@dataclass(frozen=True)
class AuditCase:
    audit_case_id: str
    dataset: str
    canonical_case_id: str
    original_case_id: str
    claim: str
    candidates: Tuple[AuditCandidate, ...]
    sampling_hash: str

    def __post_init__(self) -> None:
        _nonblank(self.audit_case_id, "audit_case_id")
        if self.dataset not in SUPPORTED_DATASETS:
            raise AuditSchemaError("dataset is not supported")
        _nonblank(self.canonical_case_id, "canonical_case_id")
        _nonblank(self.original_case_id, "original_case_id")
        _nonblank(self.claim, "claim")
        if not 6 <= len(self.candidates) <= 24:
            raise AuditSchemaError("candidate count must be between 6 and 24")
        ids = tuple(candidate.unit_id for candidate in self.candidates)
        if len(ids) != len(set(ids)):
            raise AuditSchemaError("candidate unit IDs must be unique")
        expected_positions = tuple(range(len(self.candidates)))
        actual_positions = tuple(
            candidate.original_candidate_position for candidate in self.candidates
        )
        if actual_positions != expected_positions:
            raise AuditSchemaError("candidate positions must preserve exposure order")
        if len(self.sampling_hash) != 64:
            raise AuditSchemaError("sampling_hash must be SHA-256")

    def request_dict(self) -> Dict[str, Any]:
        return {
            "audit_case_id": self.audit_case_id,
            "dataset": self.dataset,
            "canonical_case_id": self.canonical_case_id,
            "original_case_id": self.original_case_id,
            "claim": self.claim,
            "candidate_units": [candidate.to_dict() for candidate in self.candidates],
        }

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "canonical_case_id": self.canonical_case_id,
            "original_case_id": self.original_case_id,
            "sampling_hash": self.sampling_hash,
            "model_exposed_unit_count": len(self.candidates),
            "candidate_unit_ids_in_original_order": [c.unit_id for c in self.candidates],
            "candidate_unit_types_in_original_order": [c.unit_type for c in self.candidates],
            "candidate_modalities_in_original_order": [c.modality for c in self.candidates],
        }
