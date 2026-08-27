"""Immutable, label-free evaluation contracts for Step 2.6R-3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class EvaluationUnit:
    unit_id: str
    unit_type: str
    modality: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("unit_id must be nonblank")
        if self.unit_type not in {"text", "transcript", "ocr"}:
            raise ValueError("unit_type must be text, transcript, or ocr")
        expected_modality = "ocr" if self.unit_type == "ocr" else "text"
        if self.modality != expected_modality:
            raise ValueError("unit_type and modality are inconsistent")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("unit text must be nonblank")

    def to_dict(self) -> Mapping[str, str]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "modality": self.modality,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: str
    case_id: str
    dataset: str
    claim: str
    candidate_units: Tuple[EvaluationUnit, ...]
    reference_id: Optional[str] = None
    positive_unit_ids: Tuple[str, ...] = ()
    reference_modality: Optional[str] = None
    source_audit_artifact_path: Optional[str] = None
    source_audit_artifact_sha256: Optional[str] = None
    prior_original_best_positive_rank: Optional[int] = None
    prior_original_top5_unit_ids: Tuple[str, ...] = ()
    prior_candidate_unit_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, field in (
            (self.request_id, "request_id"),
            (self.case_id, "case_id"),
            (self.dataset, "dataset"),
            (self.claim, "claim"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonblank")
        if not self.candidate_units or len(self.candidate_units) > 24:
            raise ValueError("candidate count must be between 1 and 24")
        ids = self.candidate_unit_ids
        if len(set(ids)) != len(ids):
            raise ValueError("candidate unit IDs must be unique")
        if self.positive_unit_ids and not set(self.positive_unit_ids) <= set(ids):
            raise ValueError("positive unit ID is missing from candidate pool")

    @property
    def candidate_unit_ids(self) -> Tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.candidate_units)

    def collator_item(self) -> Mapping[str, object]:
        """Return authoritative input plus an unused structural label."""

        return {
            "case_id": self.request_id,
            "label": 0,
            "claim": self.claim,
            "dataset": self.dataset,
            "units": [dict(unit.to_dict()) for unit in self.candidate_units],
        }


@dataclass(frozen=True)
class PredictionSnapshot:
    candidate_unit_ids: Tuple[str, ...]
    selection_scores: Tuple[float, ...]
    unit_veracity_logits: Tuple[Tuple[float, float], ...]
    sample_logits: Tuple[float, float]
    probabilities: Tuple[float, float]
    prediction: str
    top_k_unit_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.candidate_unit_ids)
        if size == 0 or len(set(self.candidate_unit_ids)) != size:
            raise ValueError("snapshot candidate IDs must be nonempty and unique")
        if len(self.selection_scores) != size or len(self.unit_veracity_logits) != size:
            raise ValueError("snapshot unit outputs must match candidate count")
        numbers = list(self.selection_scores) + list(self.sample_logits) + list(
            self.probabilities
        )
        numbers.extend(value for pair in self.unit_veracity_logits for value in pair)
        if any(not math.isfinite(float(value)) for value in numbers):
            raise ValueError("snapshot numeric outputs must be finite")
        if self.prediction not in {"fake", "real"}:
            raise ValueError("snapshot prediction must be fake or real")
        if len(set(self.top_k_unit_ids)) != len(self.top_k_unit_ids):
            raise ValueError("snapshot Top-k IDs must be unique")
        if not set(self.top_k_unit_ids) <= set(self.candidate_unit_ids):
            raise ValueError("snapshot Top-k contains an unknown unit ID")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "candidate_unit_ids": list(self.candidate_unit_ids),
            "selection_scores": list(self.selection_scores),
            "unit_veracity_logits": [
                {"fake": pair[0], "real": pair[1]}
                for pair in self.unit_veracity_logits
            ],
            "sample_logits": {
                "fake": self.sample_logits[0],
                "real": self.sample_logits[1],
            },
            "probabilities": {
                "fake": self.probabilities[0],
                "real": self.probabilities[1],
            },
            "prediction": self.prediction,
            "top_k_unit_ids": list(self.top_k_unit_ids),
        }
