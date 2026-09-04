"""Atomic preflight and one-shot execution for Step 2.6R-3B3."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from scripts.selector_relevance_gate.runtime import (
    DICCEvaluationRuntime,
    TrainingArtifacts,
    validate_training_artifacts,
)
from scripts.selector_relevance_gate.schemas import PredictionSnapshot

from .metrics_adapter import (
    aggregate_metrics,
    calculate_repair_gate,
    evaluate_case_scores,
)
from .schemas import (
    ALLOWED_STATE_DIFFERENCES,
    BASE_G1_SHA256,
    CALIBRATED_SELECTOR_SHA256,
    EXPECTED_CASE_COUNT,
    EXPECTED_EVALUABLE_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    IMPLEMENTATION_REVISION,
    REQUIRED_SEED,
    IndependentEvaluationError,
    PreparedInputs,
)
from .source_loader import (
    assert_hashes_unchanged,
    json_bytes,
    jsonl_bytes,
    prepare_inputs,
    safe_path,
    sha256_file,
)


_PREFLIGHT_ARTIFACTS = (
    "evaluation_preflight_source_lock.json",
    "evaluation_case_manifest.json",
    "preregistration_lock.json",
    "selector_artifact_lock.json",
    "one_shot_preflight_report.json",
)
_EVALUATION_ARTIFACTS = (
    "evaluation_source_lock.json",
    "selector_state_lock.json",
    "ranking_scores.jsonl",
    "per_case_ranking_metrics.jsonl",
    "selector_metrics.json",
    "repair_verification_gate_report.json",
    "one_shot_evaluation_report.json",
)


class EvaluationRuntime(Protocol):
    encoder_hash: str
    veracity_head_hash: str
    original_selection_head_hash: str
    calibrated_selection_head_hash: str
    state_difference_names: Tuple[str, ...]

    def evaluate(self, request: Any, *, state: str) -> PredictionSnapshot: ...

    def assert_immutable(self) -> None: ...


def _output_target(path: Path, field: str) -> Path:
    output = safe_path(path, field)
    if output.exists():
        raise IndependentEvaluationError(f"{field} already exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _artifact_payload(name: str, value: Any) -> bytes:
    return jsonl_bytes(value) if name.endswith(".jsonl") else json_bytes(value)


def _freeze_directory(
    output: Path,
    artifacts: Mapping[str, Any],
    *,
    expected_names: Sequence[str],
) -> Mapping[str, str]:
    if tuple(artifacts) != tuple(expected_names):
        raise IndependentEvaluationError("atomic output artifact schema changed")
    if output.exists():
        raise IndependentEvaluationError("authoritative output appeared during execution")
    staging = Path(tempfile.mkdtemp(prefix=".step2.6r-3b3-", dir=output.parent))
    digests: Dict[str, str] = {}
    try:
        for name, value in artifacts.items():
            payload = _artifact_payload(name, value)
            path = staging / name
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
            digests[name] = digest
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return digests


def run_preflight(
    *,
    cohort_dir: Path,
    final_gold_dir: Path,
    stage_a_invariance_report: Path,
    project_root: Path,
    phase4a_config: Path,
    neutral_dir: Path,
    training_dir: Path,
    output_dir: Path,
    training_validator: Callable[[Path, Path], TrainingArtifacts] = validate_training_artifacts,
) -> Mapping[str, Any]:
    """Freeze a score-free approval; this function never creates a runtime."""

    output = _output_target(output_dir, "preflight output directory")
    prepared = prepare_inputs(
        cohort_dir=cohort_dir,
        final_gold_dir=final_gold_dir,
        stage_a_invariance_report=stage_a_invariance_report,
        project_root=project_root,
        phase4a_config=phase4a_config,
        neutral_dir=neutral_dir,
        training_dir=training_dir,
        training_validator=training_validator,
    )
    assert_hashes_unchanged(prepared.immutable_file_hashes)
    source_sha = hashlib.sha256(json_bytes(prepared.source_lock)).hexdigest()
    case_sha = hashlib.sha256(json_bytes(prepared.case_manifest)).hexdigest()
    prereg_sha = hashlib.sha256(json_bytes(prepared.preregistration_lock)).hexdigest()
    selector_sha = hashlib.sha256(json_bytes(prepared.selector_artifact_lock)).hexdigest()
    report = {
        "status": "INDEPENDENT_SELECTOR_ONE_SHOT_PREFLIGHT_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "evaluation_preflight_source_lock_sha256": source_sha,
        "evaluation_case_manifest_sha256": case_sha,
        "preregistration_lock_sha256": prereg_sha,
        "selector_artifact_lock_sha256": selector_sha,
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": EXPECTED_UNIT_COUNT,
        "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
        "required_seed": REQUIRED_SEED,
        "selector_sha256": CALIBRATED_SELECTOR_SHA256,
        "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
        "model_loaded": False,
        "checkpoint_loaded_for_execution": False,
        "selector_scoring_performed": False,
        "training_started": False,
        "optimizer_created": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
        "production_or_ui_changed": False,
    }
    artifacts = {
        "evaluation_preflight_source_lock.json": prepared.source_lock,
        "evaluation_case_manifest.json": prepared.case_manifest,
        "preregistration_lock.json": prepared.preregistration_lock,
        "selector_artifact_lock.json": prepared.selector_artifact_lock,
        "one_shot_preflight_report.json": report,
    }
    _freeze_directory(output, artifacts, expected_names=_PREFLIGHT_ARTIFACTS)
    return report


def _verify_file_and_sidecar(path: Path, field: str) -> str:
    if not path.is_file():
        raise IndependentEvaluationError(f"approved preflight {field} is missing")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise IndependentEvaluationError(f"approved preflight {field} sidecar is missing")
    actual = sha256_file(path)
    try:
        declared = sidecar.read_text(encoding="utf-8").strip().casefold()
    except (OSError, UnicodeError) as exc:
        raise IndependentEvaluationError(
            f"approved preflight {field} sidecar is unreadable"
        ) from exc
    if declared != actual:
        raise IndependentEvaluationError(f"approved preflight {field} SHA mismatch")
    return actual


def _load_approved_preflight(path: Path) -> Mapping[str, Mapping[str, Any]]:
    report_path = safe_path(path, "approved preflight report")
    if report_path.name != "one_shot_preflight_report.json":
        raise IndependentEvaluationError("approved preflight report filename changed")
    directory = report_path.parent
    expected_files = {
        name for artifact in _PREFLIGHT_ARTIFACTS for name in (artifact, str(Path(artifact).with_suffix(".sha256")))
    }
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != expected_files:
        raise IndependentEvaluationError("approved preflight artifact set changed")
    payloads: Dict[str, Mapping[str, Any]] = {}
    digests = {}
    for name in _PREFLIGHT_ARTIFACTS:
        artifact = directory / name
        digests[name] = _verify_file_and_sidecar(artifact, name)
        try:
            value = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise IndependentEvaluationError(
                f"approved preflight {name} is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise IndependentEvaluationError(f"approved preflight {name} is not an object")
        payloads[name] = value
    report = payloads["one_shot_preflight_report.json"]
    required = {
        "status": "INDEPENDENT_SELECTOR_ONE_SHOT_PREFLIGHT_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": EXPECTED_UNIT_COUNT,
        "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
        "required_seed": REQUIRED_SEED,
        "selector_sha256": CALIBRATED_SELECTOR_SHA256,
        "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
        "model_loaded": False,
        "checkpoint_loaded_for_execution": False,
        "selector_scoring_performed": False,
        "training_started": False,
        "optimizer_created": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
        "production_or_ui_changed": False,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise IndependentEvaluationError(
                f"approved preflight report contract failed: {field}"
            )
    pointers = {
        "evaluation_preflight_source_lock_sha256": "evaluation_preflight_source_lock.json",
        "evaluation_case_manifest_sha256": "evaluation_case_manifest.json",
        "preregistration_lock_sha256": "preregistration_lock.json",
        "selector_artifact_lock_sha256": "selector_artifact_lock.json",
    }
    for field, name in pointers.items():
        if report.get(field) != digests[name]:
            raise IndependentEvaluationError(
                f"approved preflight report hash pointer failed: {field}"
            )
    return payloads


def _same_preflight(current: PreparedInputs, saved: Mapping[str, Mapping[str, Any]]) -> None:
    comparisons = {
        "evaluation_preflight_source_lock.json": current.source_lock,
        "evaluation_case_manifest.json": current.case_manifest,
        "preregistration_lock.json": current.preregistration_lock,
        "selector_artifact_lock.json": current.selector_artifact_lock,
    }
    for name, value in comparisons.items():
        if saved.get(name) != value:
            raise IndependentEvaluationError(
                f"current frozen inputs differ from approved preflight: {name}"
            )


def _runtime_boundary(runtime: EvaluationRuntime) -> Tuple[bool, Mapping[str, Any]]:
    differences = tuple(runtime.state_difference_names)
    architecture_pass = bool(
        differences
        and len(set(differences)) == len(differences)
        and set(differences) <= ALLOWED_STATE_DIFFERENCES
        and runtime.original_selection_head_hash
        != runtime.calibrated_selection_head_hash
        and isinstance(runtime.encoder_hash, str)
        and bool(runtime.encoder_hash)
        and isinstance(runtime.veracity_head_hash, str)
        and bool(runtime.veracity_head_hash)
    )
    if not architecture_pass:
        raise IndependentEvaluationError("runtime state-difference whitelist failed")
    return architecture_pass, {
        "status": "INDEPENDENT_SELECTOR_STATE_LOCK_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "required_seed": REQUIRED_SEED,
        "selector_sha256": CALIBRATED_SELECTOR_SHA256,
        "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
        "allowed_state_differences": sorted(ALLOWED_STATE_DIFFERENCES),
        "observed_state_differences": list(differences),
        "only_selection_head_weight_bias_may_differ": True,
        "architecture_condition_pass": True,
        "encoder_hash_original": runtime.encoder_hash,
        "encoder_hash_calibrated": runtime.encoder_hash,
        "veracity_head_hash_original": runtime.veracity_head_hash,
        "veracity_head_hash_calibrated": runtime.veracity_head_hash,
        "original_selection_head_hash": runtime.original_selection_head_hash,
        "calibrated_selection_head_hash": runtime.calibrated_selection_head_hash,
        "training_started": False,
        "optimizer_created": False,
        "model_parameter_update_performed": False,
    }


def run_one_shot_evaluation(
    *,
    cohort_dir: Path,
    final_gold_dir: Path,
    approved_preflight_report: Path,
    stage_a_invariance_report: Path,
    project_root: Path,
    phase4a_config: Path,
    neutral_dir: Path,
    training_dir: Path,
    device: str,
    output_dir: Path,
    training_validator: Callable[[Path, Path], TrainingArtifacts] = validate_training_artifacts,
    runtime_factory: Callable[..., EvaluationRuntime] = DICCEvaluationRuntime,
) -> Mapping[str, Any]:
    """Perform the sole valid scoring run and freeze PASS or scientific FAIL."""

    output = _output_target(output_dir, "one-shot evaluation output directory")
    approved = _load_approved_preflight(approved_preflight_report)
    current = prepare_inputs(
        cohort_dir=cohort_dir,
        final_gold_dir=final_gold_dir,
        stage_a_invariance_report=stage_a_invariance_report,
        project_root=project_root,
        phase4a_config=phase4a_config,
        neutral_dir=neutral_dir,
        training_dir=training_dir,
        training_validator=training_validator,
    )
    _same_preflight(current, approved)
    assert_hashes_unchanged(current.immutable_file_hashes)
    if not isinstance(device, str) or not device.strip():
        raise IndependentEvaluationError("one-shot device must be nonblank")

    # No model/runtime object is created until every preflight artifact and current
    # immutable source has been revalidated above.
    try:
        runtime = runtime_factory(
            project_root=current.project_root,
            phase4a_config_path=Path(phase4a_config).expanduser().resolve(),
            training_artifacts=current.training_artifacts,
            device=device.strip(),
        )
    except IndependentEvaluationError:
        raise
    except Exception as exc:
        raise IndependentEvaluationError("authoritative evaluation runtime failed") from exc
    architecture_pass, selector_state_lock = _runtime_boundary(runtime)

    score_rows = []
    case_rows = []
    for case in current.cases:
        request = case.evaluation_request()
        try:
            original = runtime.evaluate(request, state="original")
            calibrated = runtime.evaluate(request, state="calibrated")
        except Exception as exc:
            raise IndependentEvaluationError("selector scoring failed") from exc
        if (
            original.candidate_unit_ids != case.candidate_unit_ids
            or calibrated.candidate_unit_ids != case.candidate_unit_ids
        ):
            raise IndependentEvaluationError("runtime changed candidate IDs/order")
        score_row, case_row = evaluate_case_scores(
            case,
            original_scores=original.selection_scores,
            calibrated_scores=calibrated.selection_scores,
        )
        score_rows.append(score_row)
        case_rows.append(case_row)
    try:
        runtime.assert_immutable()
    except Exception as exc:
        raise IndependentEvaluationError("runtime immutability assertion failed") from exc
    assert_hashes_unchanged(current.immutable_file_hashes)

    metric_values = aggregate_metrics(case_rows)
    selector_metrics = {
        "status": "INDEPENDENT_SELECTOR_RANKING_METRICS_FROZEN",
        "implementation_revision": IMPLEMENTATION_REVISION,
        **metric_values,
        "metric_macro_denominator": "evaluable cases only",
        "zero_direct_cases_included_in_denominator": False,
    }
    gate = calculate_repair_gate(
        metric_values,
        evaluable_case_count=EXPECTED_EVALUABLE_CASE_COUNT,
        seed=REQUIRED_SEED,
        architecture_condition_pass=architecture_pass,
    )
    gate_report = {
        "implementation_revision": IMPLEMENTATION_REVISION,
        **gate,
    }
    preflight_path = Path(approved_preflight_report).expanduser().resolve()
    evaluation_source_lock = {
        "status": "INDEPENDENT_SELECTOR_EVALUATION_SOURCE_LOCK_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "approved_preflight_report": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "preflight_reverified_before_model_loading": True,
        "immutable_input_sha256": dict(current.immutable_file_hashes),
        "immutable_inputs_reverified_after_scoring": True,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
    }
    report = {
        "status": gate["status"],
        "implementation_revision": IMPLEMENTATION_REVISION,
        "scientific_result_valid": True,
        "repair_verification_pass": gate["repair_verification_pass"],
        "deployment_remains_blocked": gate["deployment_remains_blocked"],
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": EXPECTED_UNIT_COUNT,
        "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
        "zero_direct_positive_case_count": EXPECTED_CASE_COUNT
        - EXPECTED_EVALUABLE_CASE_COUNT,
        "required_seed": REQUIRED_SEED,
        "selector_sha256": CALIBRATED_SELECTOR_SHA256,
        "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
        "model_loaded": True,
        "selector_scoring_performed": True,
        "training_started": False,
        "optimizer_created": False,
        "post_audit_retraining_or_model_selection_performed": False,
        "seed_43_or_44_scored": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
        "production_switch_performed": False,
        "public_demo_changed": False,
        "interpretation": "independent Train-derived repair verification only",
    }
    artifacts = {
        "evaluation_source_lock.json": evaluation_source_lock,
        "selector_state_lock.json": selector_state_lock,
        "ranking_scores.jsonl": score_rows,
        "per_case_ranking_metrics.jsonl": case_rows,
        "selector_metrics.json": selector_metrics,
        "repair_verification_gate_report.json": gate_report,
        "one_shot_evaluation_report.json": report,
    }
    _freeze_directory(output, artifacts, expected_names=_EVALUATION_ARTIFACTS)
    return report
