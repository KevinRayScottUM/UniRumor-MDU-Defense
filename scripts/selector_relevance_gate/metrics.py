"""Deterministic held-out direct-relevance ranking metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple


METRIC_NAMES = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
)


@dataclass(frozen=True)
class HeldoutRanking:
    reference_id: str
    case_id: str
    dataset: str
    reference_modality: str
    candidate_unit_ids: Tuple[str, ...]
    positive_unit_ids: Tuple[str, ...]
    selection_scores: Tuple[float, ...]

    def __post_init__(self) -> None:
        for value, field in (
            (self.reference_id, "reference_id"),
            (self.case_id, "case_id"),
            (self.dataset, "dataset"),
            (self.reference_modality, "reference_modality"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonblank")
        size = len(self.candidate_unit_ids)
        if size == 0 or len(set(self.candidate_unit_ids)) != size:
            raise ValueError("candidate unit IDs must be nonempty and unique")
        if len(self.selection_scores) != size:
            raise ValueError("selection score count must match candidates")
        if any(not math.isfinite(float(value)) for value in self.selection_scores):
            raise ValueError("selection scores must be finite")
        if not self.positive_unit_ids or len(set(self.positive_unit_ids)) != len(
            self.positive_unit_ids
        ):
            raise ValueError("positive unit IDs must be nonempty and unique")
        if not set(self.positive_unit_ids) <= set(self.candidate_unit_ids):
            raise ValueError("positive unit ID is missing from candidate pool")


def ranked_unit_ids(example: HeldoutRanking) -> Tuple[str, ...]:
    """Rank by score descending with original candidate order as the tie break."""

    order = sorted(
        range(len(example.candidate_unit_ids)),
        key=lambda index: (-float(example.selection_scores[index]), index),
    )
    return tuple(example.candidate_unit_ids[index] for index in order)


def reference_metrics(example: HeldoutRanking) -> Mapping[str, float | int]:
    order = ranked_unit_ids(example)
    positives = set(example.positive_unit_ids)
    relevant_ranks = tuple(
        rank for rank, unit_id in enumerate(order, start=1) if unit_id in positives
    )
    best_rank = min(relevant_ranks)

    def recall_at(k: int) -> float:
        return 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= 5
    )
    ideal_count = min(len(positives), 5)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1)
    )
    return {
        "best_positive_rank": best_rank,
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "mrr": 1.0 / best_rank,
        "ndcg_at_5": dcg / ideal_dcg,
    }


def _macro(examples: Sequence[HeldoutRanking]) -> Mapping[str, float]:
    if not examples:
        raise ValueError("ranking evaluation requires at least one reference")
    totals: Dict[str, float] = {name: 0.0 for name in METRIC_NAMES}
    for example in examples:
        values = reference_metrics(example)
        for name in METRIC_NAMES:
            totals[name] += float(values[name])
    return {name: totals[name] / len(examples) for name in METRIC_NAMES}


def grouped_metrics(
    examples: Sequence[HeldoutRanking],
) -> Mapping[str, object]:
    by_case: Dict[str, list[HeldoutRanking]] = {}
    by_dataset: Dict[str, list[HeldoutRanking]] = {}
    by_modality: Dict[str, list[HeldoutRanking]] = {}
    for example in examples:
        by_case.setdefault(example.case_id, []).append(example)
        by_dataset.setdefault(example.dataset, []).append(example)
        by_modality.setdefault(example.reference_modality, []).append(example)
    return {
        "overall": _macro(examples),
        "by_underlying_case": {
            key: _macro(value) for key, value in sorted(by_case.items())
        },
        "by_dataset": {
            key: _macro(value) for key, value in sorted(by_dataset.items())
        },
        "by_reference_modality": {
            key: _macro(value) for key, value in sorted(by_modality.items())
        },
    }
