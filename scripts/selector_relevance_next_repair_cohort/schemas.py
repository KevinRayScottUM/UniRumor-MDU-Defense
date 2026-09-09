"""Engineering contracts; scientific values come from the frozen 3C1 loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from scripts.selector_relevance_next_repair_protocol.protocol import (
    PREREGISTRATION_PATH, PREREGISTRATION_SHA256, load_preregistration,
)
from scripts.selector_relevance_next_repair_protocol.schemas import ProtocolError
from scripts.selector_relevance_independent_audit.schemas import (
    AUTHORITATIVE_TRAIN_SHA256, EXPECTED_SOURCE_COUNTS,
    SEALED_CHALLENGE_IDS, STAGE_A_IDS,
    TARGET_DATASET_COUNTS as REVEALED_AUDIT_DATASET_COUNTS,
)

IMPLEMENTATION_REVISION = "step2.6r-3c2a-v1"
CLOSURE_HASHES = {
    "step3b3_scientific_summary.json":
        "6ed3401614f68e58ae0efb3a1671f9f2f7b6b8b653f2100660199ee72938fdf0",
    "step3b3_closure_manifest.json":
        "71732d7c2b84856ed735cc6d81e7da8af3495fdfba49d9b71cd57df7ed92f81f",
}
CLOSURE_STATUS = "STEP_2_6R_3B3_CLOSED_VALID_SCIENTIFIC_FAIL"
REVIEWER_SALTS = {
    "A": "step2.6r-3c2a-reviewer-a-v1",
    "B": "step2.6r-3c2a-reviewer-b-v1",
}
PUBLIC_COLUMNS = (
    "review_case_id", "claim", "review_unit_id", "candidate_text",
    "direct_relevance_label", "review_confidence", "review_note",
)
CANDIDATE_FIELDS = ("unit_id", "unit_type", "modality", "text")
IDENTITY_FIELDS = frozenset({
    "dataset", "source_dataset", "case_id", "sample_id", "id",
    "original_case_id", "source_case_id", "canonical_case_id",
    "canonical_underlying_case_id", "split",
})
FORBIDDEN_ARTIFACTS = frozenset({
    "ranking_scores.jsonl", "per_case_ranking_metrics.jsonl", "selector_metrics.json",
    "repair_verification_gate_report.json", "one_shot_evaluation_report.json",
})


class CohortError(ProtocolError):
    """Invalid input, integrity failure or contract drift (exit 2)."""


@dataclass(frozen=True)
class Inputs:
    project_root: Path
    phase4a_config: Path
    source_3b1_cohort_dir: Path
    step3b3_closure_dir: Path
    neutral_dir: Path
    stage_a_invariance_report: Path
    future_exclusion_manifests: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class Case:
    dataset: str
    canonical_case_id: str
    original_case_id: str
    source_row_index: int
    claim: str
    # Candidate tuples prevent accidental mutation between exposure and selection.
    candidates: Tuple[Tuple[str, str, str, str], ...]

    def units(self):
        return [dict(zip(CANDIDATE_FIELDS, unit), original_candidate_position=index)
                for index, unit in enumerate(self.candidates)]


@dataclass(frozen=True)
class Selection:
    case: Case
    repair_split: str
    sampling_hash: str
    split_hash: str

    def request(self) -> Mapping[str, Any]:
        return {
            "development_case_id": "development-" + self.sampling_hash,
            "dataset": self.case.dataset,
            "canonical_case_id": self.case.canonical_case_id,
            "original_case_id": self.case.original_case_id,
            "repair_split": self.repair_split,
            "claim": self.case.claim,
            "candidate_units": self.case.units(),
        }

    def manifest(self) -> Mapping[str, Any]:
        return {
            "dataset": self.case.dataset,
            "canonical_case_id": self.case.canonical_case_id,
            "original_case_id": self.case.original_case_id,
            "repair_split": self.repair_split,
            "source_row_index": self.case.source_row_index,
            "sampling_hash": self.sampling_hash,
            "split_hash": self.split_hash,
            "exposed_candidate_count": len(self.case.candidates),
            "candidate_unit_ids_in_original_order": [u[0] for u in self.case.candidates],
            "candidate_unit_types_in_original_order": [u[1] for u in self.case.candidates],
            "candidate_modalities_in_original_order": [u[2] for u in self.case.candidates],
        }
