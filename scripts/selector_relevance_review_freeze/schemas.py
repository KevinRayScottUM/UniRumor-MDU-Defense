"""Immutable, score-free schemas for Step 2.6R-3B2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


IMPLEMENTATION_REVISION = "step2.6r-3b2-v1"
SOURCE_IMPLEMENTATION_REVISION = "step2.6r-3b1-r2-v1"
SCHEMA_VERSION = 1
EXPECTED_CASE_COUNT = 30
EXPECTED_UNIT_COUNT = 289
COVERAGE_GATE_MINIMUM = 24
ADJUDICATION_SALT = "step2.6r-3b2-adjudication-v1"

RELEVANCE_LABELS = ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE")
CONFIDENCE_LABELS = ("HIGH", "MEDIUM", "LOW")
REVIEW_COLUMNS = (
    "review_case_id",
    "claim",
    "review_unit_id",
    "candidate_text",
    "direct_relevance_label",
    "review_confidence",
    "review_note",
)
ADJUDICATION_COLUMNS = (
    "adjudication_case_id",
    "claim",
    "adjudication_unit_id",
    "candidate_text",
    "final_relevance_label",
    "adjudication_confidence",
    "adjudication_note",
)
FINAL_GOLD_FIELDS = (
    "dataset",
    "canonical_case_id",
    "unit_id",
    "original_candidate_position",
    "final_relevance_label",
    "binary_direct_relevance_target",
    "resolution_source",
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


class ReviewFreezeError(RuntimeError):
    """Raised when a review or gold-freeze boundary fails closed."""


def binary_direct_target(label: str) -> int:
    if label not in RELEVANCE_LABELS:
        raise ReviewFreezeError("relevance label is invalid")
    return 1 if label == "DIRECT" else 0


@dataclass(frozen=True)
class FrozenReviewRow:
    review_case_id: str
    claim: str
    review_unit_id: str
    candidate_text: str
    direct_relevance_label: str
    review_confidence: str
    review_note: str


@dataclass(frozen=True)
class UnderlyingUnit:
    dataset: str
    canonical_case_id: str
    original_case_id: str
    unit_id: str
    unit_type: str
    modality: str
    original_candidate_position: int

    @property
    def key(self) -> Tuple[str, str]:
        return (self.canonical_case_id, self.unit_id)


@dataclass(frozen=True)
class AlignedReviewRow:
    underlying: UnderlyingUnit
    claim: str
    candidate_text: str
    reviewer_a: FrozenReviewRow
    reviewer_b: FrozenReviewRow


@dataclass(frozen=True)
class AgreementResult:
    report: Mapping[str, object]
    by_case_rows: Tuple[Mapping[str, object], ...]
    ledger_rows: Tuple[Mapping[str, object], ...]
    aligned_rows: Tuple[AlignedReviewRow, ...]
