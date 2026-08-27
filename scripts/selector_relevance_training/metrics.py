"""Deterministic macro ranking metrics for direct-relevance calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple


METRIC_NAMES = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
)


@dataclass(frozen=True)
class RankingExample:
    """Scores and binary direct-relevance targets for one calibration example."""

    calibration_example_id: str
    source_dataset: str
    expected_modality: str
    candidate_unit_ids: Tuple[str, ...]
    relevance_targets: Tuple[int, ...]
    selection_scores: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.calibration_example_id:
            raise ValueError("calibration_example_id must be nonblank")
        if not self.source_dataset:
            raise ValueError("source_dataset must be nonblank")
        if self.expected_modality not in {"OCR", "TRANSCRIPT"}:
            raise ValueError("expected_modality must be OCR or TRANSCRIPT")
        size = len(self.candidate_unit_ids)
        if size == 0:
            raise ValueError("ranking example must contain candidates")
        if len(set(self.candidate_unit_ids)) != size:
            raise ValueError("ranking candidate unit IDs must be unique")
        if len(self.relevance_targets) != size or len(self.selection_scores) != size:
            raise ValueError("ranking fields must have identical lengths")
        if any(type(target) is not int or target not in {0, 1} for target in self.relevance_targets):
            raise ValueError("relevance targets must be binary integers")
        if not any(self.relevance_targets):
            raise ValueError("ranking example must contain a positive target")
        if any(not math.isfinite(float(score)) for score in self.selection_scores):
            raise ValueError("selection scores must be finite")


def _example_metrics(example: RankingExample) -> Mapping[str, float]:
    order = sorted(
        range(len(example.selection_scores)),
        key=lambda index: (-float(example.selection_scores[index]), index),
    )
    relevant_count = sum(example.relevance_targets)

    def recall_at(k: int) -> float:
        hits = sum(example.relevance_targets[index] for index in order[:k])
        return hits / relevant_count

    first_relevant_rank = next(
        rank
        for rank, candidate_index in enumerate(order, start=1)
        if example.relevance_targets[candidate_index] == 1
    )
    dcg = sum(
        example.relevance_targets[candidate_index] / math.log2(rank + 1)
        for rank, candidate_index in enumerate(order[:5], start=1)
    )
    ideal_hits = min(relevant_count, 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "mrr": 1.0 / first_relevant_rank,
        "ndcg_at_5": dcg / idcg,
    }


def evaluate_ranking(examples: Sequence[RankingExample]) -> Mapping[str, float]:
    """Macro-average the preregistered metrics over calibration examples."""

    if not examples:
        raise ValueError("ranking evaluation requires at least one example")
    totals: Dict[str, float] = {name: 0.0 for name in METRIC_NAMES}
    for example in examples:
        values = _example_metrics(example)
        for name in METRIC_NAMES:
            totals[name] += values[name]
    return {name: totals[name] / len(examples) for name in METRIC_NAMES}


def grouped_ranking_metrics(
    examples: Sequence[RankingExample],
) -> Mapping[str, object]:
    """Report overall metrics and deterministic dataset/modality slices."""

    by_dataset: Dict[str, list[RankingExample]] = {}
    by_modality: Dict[str, list[RankingExample]] = {}
    for example in examples:
        by_dataset.setdefault(example.source_dataset, []).append(example)
        by_modality.setdefault(example.expected_modality, []).append(example)
    return {
        "overall": evaluate_ranking(examples),
        "by_dataset": {
            name: evaluate_ranking(group)
            for name, group in sorted(by_dataset.items())
        },
        "by_anchor_modality": {
            name: evaluate_ranking(group)
            for name, group in sorted(by_modality.items())
        },
    }


def finite_metrics(metrics: Mapping[str, object]) -> bool:
    """Return whether every nested metric is a finite real number."""

    for value in metrics.values():
        if isinstance(value, Mapping):
            if not finite_metrics(value):
                return False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        elif not math.isfinite(float(value)):
            return False
    return True


def mean_and_population_std(values: Iterable[float]) -> Mapping[str, float]:
    items = tuple(float(value) for value in values)
    if not items or any(not math.isfinite(item) for item in items):
        raise ValueError("summary values must be a nonempty finite sequence")
    mean = sum(items) / len(items)
    variance = sum((item - mean) ** 2 for item in items) / len(items)
    return {"mean": mean, "std": math.sqrt(variance)}
