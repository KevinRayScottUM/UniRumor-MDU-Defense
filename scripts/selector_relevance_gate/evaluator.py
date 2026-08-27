"""Pure Step 2.6R-3 invariance and held-out relevance gate orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from scripts.selector_relevance_training.trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    IMPLEMENTATION_REVISION as TRAINING_IMPLEMENTATION_REVISION,
    SELECTOR_ID,
    sha256_file,
)

from .heldout_loader import (
    CPAC_REFERENCE_ID,
    EXPECTED_HELDOUT_CASE_IDS,
    calibration_overlap_count,
)
from .metrics import HeldoutRanking, grouped_metrics, ranked_unit_ids, reference_metrics
from .runtime import DEPLOYMENT_CANDIDATE_SEED, TrainingArtifacts
from .schemas import EvaluationRequest, PredictionSnapshot


IMPLEMENTATION_REVISION = "step2.6r-3-v1"
INVARIANCE_TOLERANCE = 1e-6
SELECTION_CHANGE_THRESHOLD = 1e-8
CPAC_CASE_ID = "GroundLie360:13025004"


class EvaluationError(RuntimeError):
    """Raised when an evaluation boundary cannot be proven."""


class EvaluationRuntime(Protocol):
    encoder_hash: str
    veracity_head_hash: str
    original_selection_head_hash: str
    calibrated_selection_head_hash: str
    state_difference_names: Tuple[str, ...]

    def evaluate(
        self, request: EvaluationRequest, *, state: str
    ) -> PredictionSnapshot: ...

    def assert_immutable(self) -> None: ...


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, value: Any) -> str:
    content = _json_bytes(value)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    content = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_sidecar(path: Path, digest: str) -> None:
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")


def _prepare_output(output_dir: Path) -> Tuple[Path, Path]:
    output = Path(output_dir).expanduser().resolve()
    if any(part.casefold() in {"validation", "test"} for part in output.parts):
        raise EvaluationError("output cannot be inside Formal Validation/Test")
    if output.exists():
        raise EvaluationError("evaluation output directory must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".selector-gate-", dir=str(output.parent)))
    return output, staging


def _maximum_pair_difference(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> float:
    return max(
        (
            abs(float(left_pair[index]) - float(right_pair[index]))
            for left_pair, right_pair in zip(left, right)
            for index in (0, 1)
        ),
        default=0.0,
    )


def compare_prediction_pair(
    request: EvaluationRequest,
    original: PredictionSnapshot,
    calibrated: PredictionSnapshot,
) -> Mapping[str, Any]:
    original_ids = original.candidate_unit_ids
    calibrated_ids = calibrated.candidate_unit_ids
    candidate_id_match = set(original_ids) == set(calibrated_ids) == set(
        request.candidate_unit_ids
    )
    candidate_order_match = original_ids == calibrated_ids == request.candidate_unit_ids
    if candidate_id_match:
        original_by_id = {
            unit_id: values
            for unit_id, values in zip(original_ids, original.unit_veracity_logits)
        }
        calibrated_by_id = {
            unit_id: values
            for unit_id, values in zip(calibrated_ids, calibrated.unit_veracity_logits)
        }
        unit_difference = _maximum_pair_difference(
            tuple(original_by_id[unit_id] for unit_id in request.candidate_unit_ids),
            tuple(calibrated_by_id[unit_id] for unit_id in request.candidate_unit_ids),
        )
        original_scores = {
            unit_id: score
            for unit_id, score in zip(original_ids, original.selection_scores)
        }
        calibrated_scores = {
            unit_id: score
            for unit_id, score in zip(calibrated_ids, calibrated.selection_scores)
        }
        maximum_selection_difference = max(
            abs(original_scores[unit_id] - calibrated_scores[unit_id])
            for unit_id in request.candidate_unit_ids
        )
    else:
        unit_difference = 0.0
        maximum_selection_difference = 0.0
    sample_difference = max(
        abs(original.sample_logits[index] - calibrated.sample_logits[index])
        for index in (0, 1)
    )
    probability_difference = max(
        abs(original.probabilities[index] - calibrated.probabilities[index])
        for index in (0, 1)
    )
    prediction_match = original.prediction == calibrated.prediction
    passed = bool(
        candidate_id_match
        and candidate_order_match
        and unit_difference <= INVARIANCE_TOLERANCE
        and sample_difference <= INVARIANCE_TOLERANCE
        and probability_difference <= INVARIANCE_TOLERANCE
        and prediction_match
    )
    return {
        "request_id": request.request_id,
        "case_id": request.case_id,
        "candidate_unit_ids": list(request.candidate_unit_ids),
        "candidate_ids_identical": candidate_id_match,
        "candidate_order_identical": candidate_order_match,
        "maximum_unit_veracity_logit_difference": unit_difference,
        "maximum_sample_logit_difference": sample_difference,
        "maximum_probability_difference": probability_difference,
        "prediction_identical": prediction_match,
        "maximum_selection_score_difference": maximum_selection_difference,
        "selection_score_changed": (
            maximum_selection_difference > SELECTION_CHANGE_THRESHOLD
        ),
        "original_selection_scores": list(original.selection_scores),
        "calibrated_selection_scores": list(calibrated.selection_scores),
        "original_top_k_unit_ids": list(original.top_k_unit_ids),
        "calibrated_top_k_unit_ids": list(calibrated.top_k_unit_ids),
        "original_prediction": original.prediction,
        "calibrated_prediction": calibrated.prediction,
        "prediction_invariant": passed,
    }


def summarize_invariance(
    comparisons: Sequence[Mapping[str, Any]], *, expected_count: Optional[int] = None
) -> Mapping[str, Any]:
    if not comparisons:
        raise EvaluationError("prediction invariance requires evaluation requests")
    summary = {
        "request_count": len(comparisons),
        "candidate_id_mismatch_count": sum(
            not bool(item["candidate_ids_identical"]) for item in comparisons
        ),
        "candidate_order_mismatch_count": sum(
            not bool(item["candidate_order_identical"]) for item in comparisons
        ),
        "maximum_unit_veracity_logit_difference": max(
            float(item["maximum_unit_veracity_logit_difference"])
            for item in comparisons
        ),
        "maximum_sample_logit_difference": max(
            float(item["maximum_sample_logit_difference"]) for item in comparisons
        ),
        "maximum_probability_difference": max(
            float(item["maximum_probability_difference"]) for item in comparisons
        ),
        "prediction_mismatch_count": sum(
            not bool(item["prediction_identical"]) for item in comparisons
        ),
        "selection_scores_changed": any(
            bool(item["selection_score_changed"]) for item in comparisons
        ),
    }
    count_ok = expected_count is None or len(comparisons) == expected_count
    summary["prediction_invariance_gate"] = bool(
        count_ok
        and summary["candidate_id_mismatch_count"] == 0
        and summary["candidate_order_mismatch_count"] == 0
        and summary["maximum_unit_veracity_logit_difference"]
        <= INVARIANCE_TOLERANCE
        and summary["maximum_sample_logit_difference"] <= INVARIANCE_TOLERANCE
        and summary["maximum_probability_difference"] <= INVARIANCE_TOLERANCE
        and summary["prediction_mismatch_count"] == 0
    )
    return summary


def _runtime_boundary(runtime: EvaluationRuntime) -> Mapping[str, Any]:
    differences = tuple(runtime.state_difference_names)
    if not differences or not set(differences) <= {
        "selection_head.weight",
        "selection_head.bias",
    }:
        raise EvaluationError("runtime state-difference whitelist failed")
    selection_changed = (
        runtime.original_selection_head_hash != runtime.calibrated_selection_head_hash
    )
    if not selection_changed:
        raise EvaluationError("selection-head hash did not change")
    return {
        "encoder_hash_original": runtime.encoder_hash,
        "encoder_hash_calibrated": runtime.encoder_hash,
        "veracity_head_hash_original": runtime.veracity_head_hash,
        "veracity_head_hash_calibrated": runtime.veracity_head_hash,
        "original_selection_head_hash": runtime.original_selection_head_hash,
        "calibrated_selection_head_hash": runtime.calibrated_selection_head_hash,
        "state_difference_names": list(differences),
        "encoder_hash_unchanged": True,
        "veracity_head_hash_unchanged": True,
        "selection_head_hash_changed": True,
    }


def _assert_files_unchanged(file_hashes: Mapping[Path, str]) -> None:
    for path, expected in file_hashes.items():
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file() or sha256_file(resolved) != expected:
            raise EvaluationError(f"immutable evaluation input changed: {resolved.name}")


def _evaluate_pairs(
    requests: Sequence[EvaluationRequest], runtime: EvaluationRuntime
) -> Tuple[Tuple[Mapping[str, Any], ...], Mapping[str, PredictionSnapshot], Mapping[str, PredictionSnapshot]]:
    comparisons = []
    originals: Dict[str, PredictionSnapshot] = {}
    calibrated: Dict[str, PredictionSnapshot] = {}
    for request in requests:
        original = runtime.evaluate(request, state="original")
        current = runtime.evaluate(request, state="calibrated")
        originals[request.request_id] = original
        calibrated[request.request_id] = current
        comparison = dict(compare_prediction_pair(request, original, current))
        comparison["original_outputs"] = dict(original.to_dict())
        comparison["calibrated_outputs"] = dict(current.to_dict())
        comparisons.append(comparison)
    runtime.assert_immutable()
    return tuple(comparisons), originals, calibrated


def run_invariance_smoke(
    *,
    requests: Sequence[EvaluationRequest],
    phase4a_replay_sha256: str,
    training_artifacts: TrainingArtifacts,
    runtime: EvaluationRuntime,
    output_dir: Path,
    immutable_input_hashes: Optional[Mapping[Path, str]] = None,
) -> Mapping[str, Any]:
    if len(requests) != 8:
        raise EvaluationError("Stage A requires exactly eight Phase4A replay requests")
    if any(item.case_id in EXPECTED_HELDOUT_CASE_IDS for item in requests):
        raise EvaluationError("Stage A cannot access held-out relevance cases")
    boundary = _runtime_boundary(runtime)
    comparisons, _, _ = _evaluate_pairs(requests, runtime)
    _assert_files_unchanged(
        immutable_input_hashes or training_artifacts.immutable_file_hashes
    )
    invariance = summarize_invariance(comparisons, expected_count=8)
    gate = bool(
        invariance["prediction_invariance_gate"]
        and invariance["selection_scores_changed"]
    )
    report = {
        "status": (
            "PREDICTION_INVARIANCE_SMOKE_PASS"
            if gate
            else "PREDICTION_INVARIANCE_SMOKE_FAIL"
        ),
        "implementation_revision": IMPLEMENTATION_REVISION,
        "training_implementation_revision": TRAINING_IMPLEMENTATION_REVISION,
        "selector_id": SELECTOR_ID,
        "deployment_candidate_seed": DEPLOYMENT_CANDIDATE_SEED,
        "deployment_candidate_selector_sha256": training_artifacts.selector_sha256,
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "phase4a_replay_artifact_sha256": phase4a_replay_sha256,
        "exact_phase4a_replay_request_count": len(requests),
        "exact_phase4a_replay_request_set_used": True,
        **boundary,
        **invariance,
        "heldout_relevance_cases_accessed": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "veracity_labels_inspected": False,
        "training_started": False,
        "optimizer_created": False,
        "production_or_model_code_changed": False,
        "public_demo_changed": False,
        "frozen_g1_checkpoint_unchanged": True,
        "selector_artifact_unchanged": True,
        "neutral_source_artifacts_unchanged": True,
    }
    output, staging = _prepare_output(output_dir)
    try:
        comparisons_sha = _write_jsonl(
            staging / "phase4a_replay_comparison.jsonl", comparisons
        )
        _write_sidecar(staging / "phase4a_replay_comparison.jsonl", comparisons_sha)
        report_sha = _write_json(
            staging / "prediction_invariance_smoke_report.json", report
        )
        _write_sidecar(
            staging / "prediction_invariance_smoke_report.json", report_sha
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def verify_approved_invariance_report(
    path: Path, training_artifacts: TrainingArtifacts
) -> Tuple[str, Mapping[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in {"validation", "test"} for part in resolved.parts):
        raise EvaluationError("approved report cannot be under Formal Validation/Test")
    if not resolved.is_file():
        raise EvaluationError("approved invariance-smoke report is missing")
    actual_sha = sha256_file(resolved)
    sidecar = resolved.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != actual_sha:
        raise EvaluationError("approved invariance-smoke report SHA mismatch")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("approved invariance-smoke report is malformed") from exc
    expected = {
        "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "training_implementation_revision": TRAINING_IMPLEMENTATION_REVISION,
        "selector_id": SELECTOR_ID,
        "deployment_candidate_seed": DEPLOYMENT_CANDIDATE_SEED,
        "deployment_candidate_selector_sha256": training_artifacts.selector_sha256,
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "prediction_invariance_gate": True,
        "selection_scores_changed": True,
        "heldout_relevance_cases_accessed": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "veracity_labels_inspected": False,
        "training_started": False,
        "optimizer_created": False,
    }
    if not isinstance(report, Mapping):
        raise EvaluationError("approved invariance-smoke report must be an object")
    for field, value in expected.items():
        if report.get(field) != value:
            raise EvaluationError(f"approved invariance-smoke mismatch: {field}")
    return actual_sha, report


def _ranking(request: EvaluationRequest, snapshot: PredictionSnapshot) -> HeldoutRanking:
    return HeldoutRanking(
        reference_id=request.reference_id or request.request_id,
        case_id=request.case_id,
        dataset=request.dataset,
        reference_modality=request.reference_modality or "UNKNOWN",
        candidate_unit_ids=snapshot.candidate_unit_ids,
        positive_unit_ids=request.positive_unit_ids,
        selection_scores=snapshot.selection_scores,
    )


def _heldout_manifest(references: Sequence[EvaluationRequest]) -> Mapping[str, Any]:
    rows = []
    for item in references:
        claim_sha = hashlib.sha256(item.claim.encode("utf-8")).hexdigest()
        ids_sha = hashlib.sha256(
            json.dumps(
                list(item.candidate_unit_ids), separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "case_id": item.case_id,
                "source_dataset": item.dataset,
                "reference_id": item.reference_id,
                "original_evaluation_claim_sha256": claim_sha,
                "candidate_id_list_sha256": ids_sha,
                "positive_unit_ids": list(item.positive_unit_ids),
                "reference_modality": item.reference_modality,
                "source_audit_artifact_path": item.source_audit_artifact_path,
                "source_audit_artifact_sha256": item.source_audit_artifact_sha256,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "heldout_relevance_reference_manifest",
        "reference_count": len(rows),
        "references": rows,
        "fake_real_ground_truth_included": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
    }


def _heldout_results(
    references: Sequence[EvaluationRequest],
    originals: Mapping[str, PredictionSnapshot],
    calibrated: Mapping[str, PredictionSnapshot],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> Tuple[Tuple[Mapping[str, Any], ...], Tuple[HeldoutRanking, ...], Tuple[HeldoutRanking, ...], bool]:
    rows = []
    original_rankings = []
    calibrated_rankings = []
    baseline_verified = True
    for request in references:
        original = _ranking(request, originals[request.request_id])
        current = _ranking(request, calibrated[request.request_id])
        original_rankings.append(original)
        calibrated_rankings.append(current)
        original_values = reference_metrics(original)
        current_values = reference_metrics(current)
        original_order = ranked_unit_ids(original)
        current_order = ranked_unit_ids(current)
        comparable = bool(
            original.candidate_unit_ids == request.prior_candidate_unit_ids
            and original_values["best_positive_rank"]
            == request.prior_original_best_positive_rank
            and tuple(original_order[:5]) == request.prior_original_top5_unit_ids
        )
        baseline_verified = baseline_verified and comparable
        positive_original_top5 = set(request.positive_unit_ids) & set(original_order[:5])
        rows.append(
            {
                "case_id": request.case_id,
                "reference_id": request.reference_id,
                "candidate_count": len(request.candidate_units),
                "positive_unit_ids": list(request.positive_unit_ids),
                "original_best_positive_rank": original_values["best_positive_rank"],
                "calibrated_best_positive_rank": current_values["best_positive_rank"],
                "rank_delta": int(original_values["best_positive_rank"])
                - int(current_values["best_positive_rank"]),
                "original_recall_at_1": original_values["recall_at_1"],
                "calibrated_recall_at_1": current_values["recall_at_1"],
                "original_recall_at_3": original_values["recall_at_3"],
                "calibrated_recall_at_3": current_values["recall_at_3"],
                "original_recall_at_5": original_values["recall_at_5"],
                "calibrated_recall_at_5": current_values["recall_at_5"],
                "original_mrr": original_values["mrr"],
                "calibrated_mrr": current_values["mrr"],
                "original_ndcg_at_5": original_values["ndcg_at_5"],
                "calibrated_ndcg_at_5": current_values["ndcg_at_5"],
                "original_top5_unit_ids": list(original_order[:5]),
                "calibrated_top5_unit_ids": list(current_order[:5]),
                "selection_score_changed": comparisons[request.request_id][
                    "selection_score_changed"
                ],
                "prediction_invariant": comparisons[request.request_id][
                    "prediction_invariant"
                ],
                "baseline_replay_verified": comparable,
                "positive_original_top5_preserved": positive_original_top5
                <= set(current_order[:5]),
            }
        )
    return (
        tuple(rows),
        tuple(original_rankings),
        tuple(calibrated_rankings),
        baseline_verified,
    )


def run_heldout_gate(
    *,
    references: Sequence[EvaluationRequest],
    heldout_reference_sha256: str,
    approved_invariance_smoke_path: Path,
    training_artifacts: TrainingArtifacts,
    runtime: EvaluationRuntime,
    output_dir: Path,
    immutable_input_hashes: Optional[Mapping[Path, str]] = None,
) -> Mapping[str, Any]:
    approved_sha, _ = verify_approved_invariance_report(
        approved_invariance_smoke_path, training_artifacts
    )
    case_ids = tuple(sorted({item.case_id for item in references}))
    if set(case_ids) != set(EXPECTED_HELDOUT_CASE_IDS):
        raise EvaluationError("held-out case identity mismatch")
    overlap = calibration_overlap_count(
        case_ids, training_artifacts.calibration_case_ids
    )
    if overlap:
        raise EvaluationError("held-out cases overlap neutral calibration Train/Dev")
    if any(
        item.claim.casefold().startswith('the relevant content states "')
        for item in references
    ):
        raise EvaluationError("neutral template is forbidden for held-out evaluation")
    boundary = _runtime_boundary(runtime)
    comparisons, originals, calibrated = _evaluate_pairs(references, runtime)
    source_hashes = {
        Path(item.source_audit_artifact_path): str(item.source_audit_artifact_sha256)
        for item in references
        if item.source_audit_artifact_path and item.source_audit_artifact_sha256
    }
    _assert_files_unchanged(
        immutable_input_hashes
        or {
            **training_artifacts.immutable_file_hashes,
            Path(approved_invariance_smoke_path): approved_sha,
            **source_hashes,
        }
    )
    comparison_by_id = {item["request_id"]: item for item in comparisons}
    invariance = summarize_invariance(comparisons)
    rows, original_rankings, calibrated_rankings, baseline_verified = _heldout_results(
        references,
        originals,
        calibrated,
        comparison_by_id,
    )
    original_metrics = grouped_metrics(original_rankings)
    calibrated_metrics = grouped_metrics(calibrated_rankings)
    original_overall = original_metrics["overall"]
    calibrated_overall = calibrated_metrics["overall"]
    improvement_count = sum(int(row["rank_delta"]) > 0 for row in rows)
    equal_count = sum(int(row["rank_delta"]) == 0 for row in rows)
    regression_count = sum(int(row["rank_delta"]) < 0 for row in rows)
    cpac_rows = [
        row
        for row in rows
        if row["case_id"] == CPAC_CASE_ID
        and row["reference_id"] == CPAC_REFERENCE_ID
    ]
    if len(cpac_rows) != 1:
        raise EvaluationError("designated CPAC reference is missing")
    cpac_top5 = all(int(row["calibrated_best_positive_rank"]) <= 5 for row in cpac_rows)
    top5_positive_preserved = all(
        bool(row["positive_original_top5_preserved"]) for row in rows
    )
    relevance_gate = bool(
        baseline_verified
        and float(calibrated_overall["mrr"]) > float(original_overall["mrr"])
        and float(calibrated_overall["ndcg_at_5"])
        > float(original_overall["ndcg_at_5"])
        and all(
            float(calibrated_overall[name]) >= float(original_overall[name])
            for name in ("recall_at_1", "recall_at_3", "recall_at_5")
        )
        and regression_count == 0
        and improvement_count >= 2
        and cpac_top5
        and top5_positive_preserved
    )
    prediction_gate = bool(invariance["prediction_invariance_gate"])
    deployment_eligible = relevance_gate and prediction_gate
    if not baseline_verified:
        status = "BASELINE_REPLAY_PROTOCOL_MISMATCH"
    elif not prediction_gate:
        status = "PREDICTION_INVARIANCE_FAIL"
    elif not relevance_gate:
        status = "HELDOUT_RELEVANCE_FAIL"
    else:
        status = "HELDOUT_RELEVANCE_AND_INVARIANCE_PASS"
    report = {
        "status": status,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "selector_id": SELECTOR_ID,
        "deployment_candidate_seed": DEPLOYMENT_CANDIDATE_SEED,
        "deployment_candidate_selector_sha256": training_artifacts.selector_sha256,
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "approved_invariance_smoke_sha256": approved_sha,
        "heldout_reference_artifact_sha256": heldout_reference_sha256,
        "heldout_case_count": len(case_ids),
        "heldout_reference_count": len(references),
        "heldout_case_ids": list(case_ids),
        "calibration_overlap_count": overlap,
        "neutral_template_used_for_heldout_evaluation": False,
        "original_challenge_claims_preserved": True,
        "candidate_exposure_changed": False,
        "sample_pooling_changed": False,
        "tokenizer_contract_changed": False,
        "baseline_replay_verified": baseline_verified,
        "original_selector_metrics": original_metrics,
        "calibrated_selector_metrics": calibrated_metrics,
        "mrr_delta": float(calibrated_overall["mrr"])
        - float(original_overall["mrr"]),
        "ndcg_at_5_delta": float(calibrated_overall["ndcg_at_5"])
        - float(original_overall["ndcg_at_5"]),
        "recall_at_1_delta": float(calibrated_overall["recall_at_1"])
        - float(original_overall["recall_at_1"]),
        "recall_at_3_delta": float(calibrated_overall["recall_at_3"])
        - float(original_overall["recall_at_3"]),
        "recall_at_5_delta": float(calibrated_overall["recall_at_5"])
        - float(original_overall["recall_at_5"]),
        "reference_rank_improvement_count": improvement_count,
        "reference_rank_equal_count": equal_count,
        "reference_rank_regression_count": regression_count,
        "cpac_original_best_positive_rank": min(
            int(row["original_best_positive_rank"]) for row in cpac_rows
        ),
        "cpac_calibrated_best_positive_rank": min(
            int(row["calibrated_best_positive_rank"]) for row in cpac_rows
        ),
        "cpac_top5_after_calibration": cpac_top5,
        "prediction_candidate_id_mismatch_count": invariance[
            "candidate_id_mismatch_count"
        ],
        "prediction_candidate_order_mismatch_count": invariance[
            "candidate_order_mismatch_count"
        ],
        "maximum_unit_veracity_logit_difference": invariance[
            "maximum_unit_veracity_logit_difference"
        ],
        "maximum_sample_logit_difference": invariance[
            "maximum_sample_logit_difference"
        ],
        "maximum_probability_difference": invariance[
            "maximum_probability_difference"
        ],
        "prediction_mismatch_count": invariance["prediction_mismatch_count"],
        **boundary,
        "selection_scores_changed": invariance["selection_scores_changed"],
        "heldout_relevance_gate": relevance_gate,
        "prediction_invariance_gate": prediction_gate,
        "deployment_eligible": deployment_eligible,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "veracity_labels_inspected": False,
        "training_started": False,
        "optimizer_created": False,
        "production_or_model_code_changed": False,
        "public_demo_changed": False,
        "frozen_g1_checkpoint_unchanged": True,
        "selector_artifact_unchanged": True,
        "neutral_source_artifacts_unchanged": True,
        "heldout_source_artifacts_unchanged": True,
    }
    output, staging = _prepare_output(output_dir)
    try:
        manifest_sha = _write_json(
            staging / "heldout_reference_manifest.json",
            _heldout_manifest(references),
        )
        _write_sidecar(staging / "heldout_reference_manifest.json", manifest_sha)
        results_sha = _write_jsonl(staging / "heldout_case_results.jsonl", rows)
        _write_sidecar(staging / "heldout_case_results.jsonl", results_sha)
        _write_json(staging / "original_selector_metrics.json", original_metrics)
        _write_json(staging / "calibrated_selector_metrics.json", calibrated_metrics)
        _write_json(
            staging / "prediction_invariance_report.json",
            {**invariance, "comparisons": list(comparisons)},
        )
        report_sha = _write_json(
            staging / "heldout_relevance_gate_report.json", report
        )
        _write_sidecar(staging / "heldout_relevance_gate_report.json", report_sha)
        (staging / "dataset_card.md").write_text(
            "# Step 2.6R-3 Held-out Relevance Challenge\n\n"
            "This is repair-verification on the pre-existing held-out relevance "
            "challenge set. It is not Formal Test, an untouched final test, or a "
            "population-level generalization benchmark. No statistical significance "
            "claim is made. The original challenge claims and audited positive unit "
            "IDs are preserved; the neutral calibration template is not used.\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
