"""Read-only Step 2.6R-3 selector relevance and invariance evaluation."""

from .evaluator import EvaluationError, run_heldout_gate, run_invariance_smoke
from .schemas import EvaluationRequest, EvaluationUnit, PredictionSnapshot

__all__ = [
    "EvaluationError",
    "EvaluationRequest",
    "EvaluationUnit",
    "PredictionSnapshot",
    "run_heldout_gate",
    "run_invariance_smoke",
]
