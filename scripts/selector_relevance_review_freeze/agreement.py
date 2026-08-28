"""Private mapping alignment and descriptive reviewer agreement metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .review_loader import PublicCohort, ValidatedReview
from .schemas import (
    CONFIDENCE_LABELS,
    EXPECTED_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    RELEVANCE_LABELS,
    AgreementResult,
    AlignedReviewRow,
    ReviewFreezeError,
    UnderlyingUnit,
    binary_direct_target,
)


def cohen_kappa(
    labels_a: Sequence[str], labels_b: Sequence[str], vocabulary: Sequence[str]
) -> Optional[float]:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ReviewFreezeError("Cohen kappa inputs must be nonempty and aligned")
    if any(value not in vocabulary for value in (*labels_a, *labels_b)):
        raise ReviewFreezeError("Cohen kappa input label is invalid")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        counts_a[label] * counts_b[label] for label in vocabulary
    ) / (total * total)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _mapping_index(
    cohort: PublicCohort,
    values: Any,
    reviewer: str,
) -> Mapping[Tuple[str, str], UnderlyingUnit]:
    if not isinstance(values, list) or len(values) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError(f"Reviewer {reviewer} private mapping count changed")
    index: Dict[Tuple[str, str], UnderlyingUnit] = {}
    underlying_seen = set()
    for row in values:
        if not isinstance(row, Mapping) or row.get("reviewer") != reviewer:
            raise ReviewFreezeError(f"Reviewer {reviewer} private mapping row is invalid")
        blind_key = (row.get("review_case_id"), row.get("review_unit_id"))
        if not all(isinstance(value, str) and value for value in blind_key):
            raise ReviewFreezeError(f"Reviewer {reviewer} blind mapping ID is invalid")
        if blind_key in index:
            raise ReviewFreezeError(f"Reviewer {reviewer} blind mapping ID is duplicated")
        position = row.get("original_candidate_position")
        values_required = (
            row.get("dataset"),
            row.get("canonical_case_id"),
            row.get("original_case_id"),
            row.get("unit_id"),
            row.get("unit_type"),
            row.get("modality"),
        )
        if (
            not all(isinstance(value, str) and value for value in values_required)
            or type(position) is not int
            or position < 0
        ):
            raise ReviewFreezeError(f"Reviewer {reviewer} underlying mapping is invalid")
        unit = UnderlyingUnit(
            dataset=values_required[0],
            canonical_case_id=values_required[1],
            original_case_id=values_required[2],
            unit_id=values_required[3],
            unit_type=values_required[4],
            modality=values_required[5],
            original_candidate_position=position,
        )
        authoritative = cohort.selected_units.get(unit.key)
        if authoritative != unit:
            raise ReviewFreezeError(
                f"Reviewer {reviewer} mapping differs from selected cohort"
            )
        if unit.key in underlying_seen:
            raise ReviewFreezeError(
                f"Reviewer {reviewer} underlying unit mapping is duplicated"
            )
        underlying_seen.add(unit.key)
        index[blind_key] = unit
    if underlying_seen != set(cohort.selected_units):
        raise ReviewFreezeError(
            f"Reviewer {reviewer} mapping does not cover the selected cohort"
        )
    return index


def align_reviews(
    cohort: PublicCohort,
    reviewer_a: ValidatedReview,
    reviewer_b: ValidatedReview,
    private_mapping: Mapping[str, Any],
) -> Tuple[AlignedReviewRow, ...]:
    mapping_a = _mapping_index(cohort, private_mapping.get("reviewer_A"), "A")
    mapping_b = _mapping_index(cohort, private_mapping.get("reviewer_B"), "B")
    rows_by_underlying: Dict[str, Dict[Tuple[str, str], Tuple[Any, Any]]] = {
        "A": {},
        "B": {},
    }
    for review in (reviewer_a, reviewer_b):
        mapping = mapping_a if review.reviewer == "A" else mapping_b
        for row in review.rows:
            blind_key = (row.review_case_id, row.review_unit_id)
            unit = mapping.get(blind_key)
            if unit is None:
                raise ReviewFreezeError(
                    f"Reviewer {review.reviewer} completed row is not privately mapped"
                )
            rows_by_underlying[review.reviewer][unit.key] = (row, unit)
    if set(rows_by_underlying["A"]) != set(rows_by_underlying["B"]):
        raise ReviewFreezeError("Reviewer A/B underlying mapped unit sets differ")

    aligned = []
    for key in sorted(
        cohort.selected_units,
        key=lambda item: (
            item[0],
            cohort.selected_units[item].original_candidate_position,
            item[1],
        ),
    ):
        row_a, unit_a = rows_by_underlying["A"][key]
        row_b, unit_b = rows_by_underlying["B"][key]
        if unit_a != unit_b:
            raise ReviewFreezeError("Reviewer A/B underlying metadata differs")
        if row_a.claim != row_b.claim or row_a.candidate_text != row_b.candidate_text:
            raise ReviewFreezeError("Reviewer A/B semantic review content differs")
        aligned.append(
            AlignedReviewRow(
                underlying=unit_a,
                claim=row_a.claim,
                candidate_text=row_a.candidate_text,
                reviewer_a=row_a,
                reviewer_b=row_b,
            )
        )
    if len(aligned) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError("aligned review count is not 289")
    return tuple(aligned)


def compute_agreement(aligned: Sequence[AlignedReviewRow]) -> AgreementResult:
    if len(aligned) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError("agreement input count is not 289")
    labels_a = [row.reviewer_a.direct_relevance_label for row in aligned]
    labels_b = [row.reviewer_b.direct_relevance_label for row in aligned]
    binary_a = ["DIRECT" if binary_direct_target(label) else "NON_DIRECT" for label in labels_a]
    binary_b = ["DIRECT" if binary_direct_target(label) else "NON_DIRECT" for label in labels_b]
    exact_agreement = sum(a == b for a, b in zip(labels_a, labels_b))
    binary_agreement = sum(a == b for a, b in zip(binary_a, binary_b))

    by_case: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total_unit_count": 0, "agreement_count": 0, "disagreement_count": 0}
    )
    ledger = []
    for row in aligned:
        case = by_case[row.underlying.canonical_case_id]
        case["dataset"] = row.underlying.dataset
        case["canonical_case_id"] = row.underlying.canonical_case_id
        case["total_unit_count"] += 1
        agreement = (
            row.reviewer_a.direct_relevance_label
            == row.reviewer_b.direct_relevance_label
        )
        case["agreement_count" if agreement else "disagreement_count"] += 1
        ledger.append(
            {
                "dataset": row.underlying.dataset,
                "canonical_case_id": row.underlying.canonical_case_id,
                "unit_id": row.underlying.unit_id,
                "original_candidate_position": row.underlying.original_candidate_position,
                "reviewer_a_label": row.reviewer_a.direct_relevance_label,
                "reviewer_a_confidence": row.reviewer_a.review_confidence,
                "reviewer_b_label": row.reviewer_b.direct_relevance_label,
                "reviewer_b_confidence": row.reviewer_b.review_confidence,
                "agreement": agreement,
                "pre_adjudication_status": "AGREED" if agreement else "NEEDS_ADJUDICATION",
            }
        )
    if len(by_case) != EXPECTED_CASE_COUNT:
        raise ReviewFreezeError("agreement case count is not 30")
    by_case_rows = tuple(by_case[key] for key in sorted(by_case))
    four_class_kappa = cohen_kappa(labels_a, labels_b, RELEVANCE_LABELS)
    binary_kappa = cohen_kappa(
        binary_a, binary_b, ("DIRECT", "NON_DIRECT")
    )
    report: Mapping[str, object] = {
        "total_unit_count": len(aligned),
        "exact_four_class_agreement_count": exact_agreement,
        "exact_four_class_disagreement_count": len(aligned) - exact_agreement,
        "exact_four_class_agreement_rate": exact_agreement / len(aligned),
        "binary_DIRECT_vs_nonDIRECT_agreement_count": binary_agreement,
        "binary_DIRECT_vs_nonDIRECT_disagreement_count": len(aligned) - binary_agreement,
        "binary_DIRECT_vs_nonDIRECT_agreement_rate": binary_agreement / len(aligned),
        "Cohen_kappa_four_class": four_class_kappa,
        "cohen_kappa_four_class_defined": four_class_kappa is not None,
        "Cohen_kappa_binary": binary_kappa,
        "cohen_kappa_binary_defined": binary_kappa is not None,
        "reviewer_A_label_counts": {
            label: Counter(labels_a)[label] for label in RELEVANCE_LABELS
        },
        "reviewer_B_label_counts": {
            label: Counter(labels_b)[label] for label in RELEVANCE_LABELS
        },
        "reviewer_A_confidence_counts": {
            value: Counter(row.reviewer_a.review_confidence for row in aligned)[value]
            for value in CONFIDENCE_LABELS
        },
        "reviewer_B_confidence_counts": {
            value: Counter(row.reviewer_b.review_confidence for row in aligned)[value]
            for value in CONFIDENCE_LABELS
        },
        "agreement_counts_by_case": {
            row["canonical_case_id"]: row["agreement_count"] for row in by_case_rows
        },
        "disagreement_counts_by_case": {
            row["canonical_case_id"]: row["disagreement_count"] for row in by_case_rows
        },
    }
    return AgreementResult(
        report=report,
        by_case_rows=by_case_rows,
        ledger_rows=tuple(ledger),
        aligned_rows=tuple(aligned),
    )
