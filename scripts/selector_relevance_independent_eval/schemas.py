"""Frozen schemas and constants for Step 2.6R-3B3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from scripts.selector_relevance_gate.schemas import EvaluationRequest, EvaluationUnit


IMPLEMENTATION_REVISION = "step2.6r-3b3-v1"
SOURCE_3B1_REVISION = "step2.6r-3b1-r2-v1"
SOURCE_3B2_REVISION = "step2.6r-3b2-v1"

EXPECTED_CASE_COUNT = 30
EXPECTED_UNIT_COUNT = 289
EXPECTED_EVALUABLE_CASE_COUNT = 28
EXPECTED_ZERO_DIRECT_CASE_COUNT = 2
COVERAGE_GATE_MINIMUM = 24
REQUIRED_SEED = 42

EXPECTED_DATASET_COUNTS = {
    "GroundLie360": {"total_case_count": 15, "evaluable_case_count": 13},
    "TRUE-3MFact": {"total_case_count": 15, "evaluable_case_count": 15},
}

BASE_G1_SHA256 = (
    "b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9"
)
CALIBRATED_SELECTOR_SHA256 = (
    "10cd426a97b61f14097145efcc3e67ca4eb381b7d4c6588a3c733c5955cb7687"
)
ALLOWED_STATE_DIFFERENCES = frozenset(
    {"selection_head.weight", "selection_head.bias"}
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
RELEVANCE_LABELS = ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE")
RESOLUTION_SOURCES = ("REVIEWER_AGREEMENT", "INDEPENDENT_ADJUDICATION")
METRIC_NAMES = (
    "mrr",
    "ndcg_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
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


class IndependentEvaluationError(RuntimeError):
    """An input, protocol, runtime, or immutability failure."""


@dataclass(frozen=True)
class IndependentCase:
    audit_case_id: str
    dataset: str
    canonical_case_id: str
    claim: str
    candidate_units: Tuple[EvaluationUnit, ...]
    positive_unit_ids: Tuple[str, ...]

    @property
    def candidate_unit_ids(self) -> Tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.candidate_units)

    @property
    def evaluable(self) -> bool:
        return bool(self.positive_unit_ids)

    def evaluation_request(self) -> EvaluationRequest:
        return EvaluationRequest(
            request_id=self.audit_case_id,
            case_id=self.canonical_case_id,
            dataset=self.dataset,
            claim=self.claim,
            candidate_units=self.candidate_units,
            positive_unit_ids=self.positive_unit_ids,
        )

    def manifest_row(self) -> Mapping[str, Any]:
        return {
            "dataset": self.dataset,
            "canonical_case_id": self.canonical_case_id,
            "audit_case_id": self.audit_case_id,
            "candidate_count": len(self.candidate_units),
            "candidate_unit_ids_in_original_order": list(self.candidate_unit_ids),
            "candidate_unit_types_in_original_order": [
                unit.unit_type for unit in self.candidate_units
            ],
            "candidate_modalities_in_original_order": [
                unit.modality for unit in self.candidate_units
            ],
            "positive_unit_ids": list(self.positive_unit_ids),
            "direct_positive_count": len(self.positive_unit_ids),
            "evaluable": self.evaluable,
        }


@dataclass(frozen=True)
class PreparedInputs:
    cases: Tuple[IndependentCase, ...]
    project_root: Path
    source_lock: Mapping[str, Any]
    case_manifest: Mapping[str, Any]
    preregistration_lock: Mapping[str, Any]
    selector_artifact_lock: Mapping[str, Any]
    immutable_file_hashes: Mapping[str, str]
    training_artifacts: Any
    checkpoint_path: Path
