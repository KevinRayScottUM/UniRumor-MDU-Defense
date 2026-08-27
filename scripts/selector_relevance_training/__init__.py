"""Step 2.6R-2 selector-only direct-relevance calibration."""

from .metrics import RankingExample, evaluate_ranking, grouped_ranking_metrics
from .trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    AUTHORITATIVE_SOURCE_HASHES,
    SelectorTrainingError,
    TrainingProtocol,
    run_selector_calibration,
)

__all__ = [
    "AUTHORITATIVE_CHECKPOINT_SHA256",
    "AUTHORITATIVE_SOURCE_HASHES",
    "RankingExample",
    "SelectorTrainingError",
    "TrainingProtocol",
    "evaluate_ranking",
    "grouped_ranking_metrics",
    "run_selector_calibration",
]
