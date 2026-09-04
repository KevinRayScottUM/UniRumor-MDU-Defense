"""Score-free loading and hash verification for the frozen 3B3 inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from scripts.selector_relevance_gate.runtime import (
    TrainingArtifacts,
    validate_training_artifacts,
)
from scripts.selector_relevance_gate.schemas import EvaluationUnit
from scripts.selector_relevance_training.dicc_backend import (
    MAXIMUM_UNITS_PER_SAMPLE,
    MAX_LENGTH,
    MODEL_NAME,
    POOLING,
    _checkpoint_path,
    _require_config_value,
)

from .schemas import (
    ALLOWED_STATE_DIFFERENCES,
    BASE_G1_SHA256,
    CALIBRATED_SELECTOR_SHA256,
    COVERAGE_GATE_MINIMUM,
    EXPECTED_CASE_COUNT,
    EXPECTED_DATASET_COUNTS,
    EXPECTED_EVALUABLE_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    EXPECTED_ZERO_DIRECT_CASE_COUNT,
    FINAL_GOLD_FIELDS,
    IMPLEMENTATION_REVISION,
    IndependentCase,
    IndependentEvaluationError,
    PreparedInputs,
    RELEVANCE_LABELS,
    REQUIRED_SEED,
    RESOLUTION_SOURCES,
    SEALED_CHALLENGE_IDS,
    SOURCE_3B1_REVISION,
    SOURCE_3B2_REVISION,
)


_COHORT_ARTIFACTS = (
    "build_report.json",
    "cohort_source_lock.json",
    "eligibility_inventory.json",
    "selected_case_manifest.json",
    "independent_relevance_audit_requests.jsonl",
    "independent_audit_preregistration.json",
)
_GOLD_ARTIFACTS = (
    "final_gold_source_lock.json",
    "final_relevance_gold.jsonl",
    "review_resolution_ledger.jsonl",
    "coverage_report.json",
    "final_gold_manifest.json",
    "final_gold_freeze_report.json",
    "adjudication_frozen.csv",
    "adjudication_provenance.json",
)
_RESTRICTED_PATH_PARTS = {"validation", "test", "formalvalidation", "formaltest"}
_HISTORICAL_PATH_MARKERS = {
    "cpac",
    "heldout",
    "historicalchallenge",
    "step25b",
    "sixcasechallenge",
}
_REQUEST_FIELDS = {
    "audit_case_id",
    "dataset",
    "canonical_case_id",
    "original_case_id",
    "claim",
    "candidate_units",
}
_CANDIDATE_FIELDS = {
    "unit_id",
    "unit_type",
    "modality",
    "text",
    "original_candidate_position",
}
_SELECTED_CASE_FIELDS = {
    "dataset",
    "canonical_case_id",
    "original_case_id",
    "sampling_hash",
    "model_exposed_unit_count",
    "candidate_unit_ids_in_original_order",
    "candidate_unit_types_in_original_order",
    "candidate_modalities_in_original_order",
}
_AUTHORITATIVE_RUNTIME_SOURCES = {
    "phase3_model": Path("MDU/scripts/clip12_phase3_common/clip12p3_model.py"),
    "phase4a_engine": Path(
        "MDU/scripts/clip12_phase4a_inference_handoff/clip12p4a_engine.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def safe_path(path: Path, field: str, *, reject_historical: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized = {
        re.sub(r"[^a-z0-9]", "", part.casefold()) for part in resolved.parts
    }
    if normalized & _RESTRICTED_PATH_PARTS:
        raise IndependentEvaluationError(
            f"{field} must not reference Formal Validation/Test"
        )
    if reject_historical and normalized & _HISTORICAL_PATH_MARKERS:
        raise IndependentEvaluationError(
            f"{field} must not reference the historical six-case/CPAC gate"
        )
    return resolved


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentEvaluationError(f"{field} is unavailable or malformed") from exc
    if not isinstance(value, Mapping):
        raise IndependentEvaluationError(f"{field} must be a JSON object")
    return value


def _read_jsonl(path: Path, field: str) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise IndependentEvaluationError(
                        f"{field} row {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentEvaluationError(f"{field} is unavailable or malformed") from exc
    return tuple(rows)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _lock_file(
    path: Path,
    field: str,
    immutable_hashes: Dict[str, str],
) -> Mapping[str, str]:
    if not path.is_file():
        raise IndependentEvaluationError(f"{field} is missing")
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        raise IndependentEvaluationError(f"{field} SHA sidecar is missing")
    actual = sha256_file(path)
    try:
        declared = sidecar.read_text(encoding="utf-8").strip().casefold()
    except (OSError, UnicodeError) as exc:
        raise IndependentEvaluationError(f"{field} SHA sidecar is unreadable") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", declared) or declared != actual:
        raise IndependentEvaluationError(f"{field} SHA-256 mismatch")
    immutable_hashes[str(path)] = actual
    immutable_hashes[str(sidecar)] = sha256_file(sidecar)
    return {
        "path": str(path),
        "sha256": actual,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": immutable_hashes[str(sidecar)],
    }


def _plain_lock(
    path: Path,
    field: str,
    immutable_hashes: Dict[str, str],
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, str]:
    if not path.is_file():
        raise IndependentEvaluationError(f"{field} is missing")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise IndependentEvaluationError(f"{field} SHA-256 mismatch")
    immutable_hashes[str(path)] = actual
    return {"path": str(path), "sha256": actual}


def _require_fields(
    payload: Mapping[str, Any], expected: Mapping[str, Any], field: str
) -> None:
    for name, value in expected.items():
        if payload.get(name) != value:
            raise IndependentEvaluationError(f"{field} contract failed: {name}")


def _frozen_protocol() -> Mapping[str, Any]:
    return {
        "metrics": ["MRR", "NDCG@5", "Recall@1", "Recall@3", "Recall@5"],
        "report_groups": ["overall", "GroundLie360", "TRUE-3MFact"],
        "stable_original_candidate_order_for_score_ties": True,
        "zero_direct_positive_cases_retained": True,
        "evaluable_case_rule": ">=1 DIRECT unit",
        "coverage_gate": {
            "minimum_evaluable_case_count": 24,
            "frozen_total_case_count": 30,
            "failure_status": "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_INSUFFICIENT",
            "resampling_permitted": False,
        },
        "repair_verification_gate": {
            "minimum_evaluable_case_count": 24,
            "calibrated_mrr_strictly_greater_than_original": True,
            "calibrated_ndcg_at_5_strictly_greater_than_original": True,
            "minimum_absolute_mrr_or_ndcg_at_5_improvement": 0.05,
            "maximum_recall_at_5_decrease": 0.02,
            "maximum_groundlie_mrr_decrease": 0.05,
            "maximum_true3m_mrr_decrease": 0.05,
            "prediction_architecture_unchanged": True,
            "required_seed": 42,
            "post_audit_retraining_or_model_selection_permitted": False,
            "all_conditions_required": True,
            "failure_action": "deployment remains blocked",
        },
    }


def _validate_preregistration(
    payload: Mapping[str, Any], digest: str
) -> Mapping[str, Any]:
    _require_fields(
        payload,
        {
            "status": "PREREGISTERED_BEFORE_REVIEW_AND_SCORING",
            "implementation_revision": SOURCE_3B1_REVISION,
            "deployment_candidate_seed": REQUIRED_SEED,
            "direct_relevance_binary_mapping": {
                "DIRECT": 1,
                "RELATED": 0,
                "IRRELEVANT": 0,
                "UNREADABLE": 0,
            },
        },
        "3B1 preregistration",
    )
    if payload.get("future_step_2_6r_3b3") != _frozen_protocol():
        raise IndependentEvaluationError("3B1 preregistered 3B3 protocol changed")
    prohibitions = payload.get("prohibitions")
    required_prohibitions = {
        "training": True,
        "calibration": True,
        "seed_42_43_44_selection": True,
        "selector_architecture_change": True,
        "threshold_tuning": True,
        "iterative_reuse_after_scores": True,
    }
    if not isinstance(prohibitions, Mapping) or any(
        prohibitions.get(name) is not value
        for name, value in required_prohibitions.items()
    ):
        raise IndependentEvaluationError("3B1 preregistration prohibitions changed")
    return {
        "status": "INDEPENDENT_AUDIT_PREREGISTRATION_LOCKED",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_implementation_revision": SOURCE_3B1_REVISION,
        "source_preregistration_sha256": digest,
        "deployment_candidate_seed": REQUIRED_SEED,
        "protocol": _frozen_protocol(),
        "direct_relevance_binary_mapping": {
            "DIRECT": 1,
            "RELATED": 0,
            "IRRELEVANT": 0,
            "UNREADABLE": 0,
        },
        "post_audit_retraining_or_model_selection_permitted": False,
    }


def _load_manifest_cases(
    selected: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    _require_fields(
        selected,
        {
            "status": "FROZEN",
            "implementation_revision": SOURCE_3B1_REVISION,
            "sampling_salt": "step2.6r-3b1-independent-audit-v1",
        },
        "3B1 selected-case manifest",
    )
    rows = selected.get("selected_cases")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise IndependentEvaluationError("3B1 selected-case count is not 30")
    order = []
    by_id: Dict[str, Mapping[str, Any]] = {}
    dataset_counts: Counter[str] = Counter()
    unit_count = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SELECTED_CASE_FIELDS:
            raise IndependentEvaluationError("3B1 selected-case row is invalid")
        dataset = row.get("dataset")
        canonical = row.get("canonical_case_id")
        original = row.get("original_case_id")
        count = row.get("model_exposed_unit_count")
        unit_ids = row.get("candidate_unit_ids_in_original_order")
        unit_types = row.get("candidate_unit_types_in_original_order")
        modalities = row.get("candidate_modalities_in_original_order")
        if (
            dataset not in EXPECTED_DATASET_COUNTS
            or not isinstance(canonical, str)
            or not canonical.startswith(dataset + ":")
            or not isinstance(original, str)
            or not original
            or canonical in by_id
            or type(count) is not int
            or not 6 <= count <= 24
            or not isinstance(unit_ids, list)
            or not isinstance(unit_types, list)
            or not isinstance(modalities, list)
            or not (len(unit_ids) == len(unit_types) == len(modalities) == count)
            or len(set(unit_ids)) != count
            or not isinstance(row.get("sampling_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sampling_hash"])
        ):
            raise IndependentEvaluationError(
                "3B1 selected-case identity/accounting changed"
            )
        # This identity-only manifest is checked before any request claim/text is read.
        if canonical in SEALED_CHALLENGE_IDS:
            raise IndependentEvaluationError(
                "3B1 cohort overlaps the sealed historical six-case challenge"
            )
        order.append(canonical)
        by_id[canonical] = row
        dataset_counts[dataset] += 1
        unit_count += count
    expected_totals = Counter(
        {name: values["total_case_count"] for name, values in EXPECTED_DATASET_COUNTS.items()}
    )
    if dataset_counts != expected_totals or unit_count != EXPECTED_UNIT_COUNT:
        raise IndependentEvaluationError("3B1 selected cohort composition changed")
    return tuple(order), by_id


def _load_requests(
    path: Path,
    manifest_order: Sequence[str],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Tuple[str, Tuple[EvaluationUnit, ...], str]]:
    rows = _read_jsonl(path, "3B1 independent audit requests")
    if len(rows) != EXPECTED_CASE_COUNT:
        raise IndependentEvaluationError("3B1 request count is not 30")
    by_id = {}
    observed_order = []
    audit_case_ids = set()
    for row in rows:
        if set(row) != _REQUEST_FIELDS:
            raise IndependentEvaluationError("3B1 request schema changed")
        canonical = row.get("canonical_case_id")
        dataset = row.get("dataset")
        original = row.get("original_case_id")
        audit_case_id = row.get("audit_case_id")
        claim = row.get("claim")
        candidates = row.get("candidate_units")
        manifest = manifest_by_id.get(canonical)
        if (
            manifest is None
            or canonical in by_id
            or dataset != manifest.get("dataset")
            or original != manifest.get("original_case_id")
            or not isinstance(audit_case_id, str)
            or not audit_case_id
            or audit_case_id in audit_case_ids
            or not isinstance(claim, str)
            or not claim.strip()
            or not isinstance(candidates, list)
        ):
            raise IndependentEvaluationError("3B1 request identity/claim changed")
        expected_ids = manifest["candidate_unit_ids_in_original_order"]
        expected_types = manifest["candidate_unit_types_in_original_order"]
        expected_modalities = manifest["candidate_modalities_in_original_order"]
        if len(candidates) != len(expected_ids):
            raise IndependentEvaluationError("3B1 request candidate count changed")
        units = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_FIELDS:
                raise IndependentEvaluationError("3B1 request candidate schema changed")
            if (
                candidate.get("unit_id") != expected_ids[position]
                or candidate.get("unit_type") != expected_types[position]
                or candidate.get("modality") != expected_modalities[position]
                or candidate.get("original_candidate_position") != position
            ):
                raise IndependentEvaluationError(
                    "3B1 request candidate metadata/order changed"
                )
            try:
                units.append(
                    EvaluationUnit(
                        unit_id=candidate["unit_id"],
                        unit_type=candidate["unit_type"],
                        modality=candidate["modality"],
                        text=candidate["text"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise IndependentEvaluationError(
                    "3B1 request candidate content is invalid"
                ) from exc
        observed_order.append(canonical)
        audit_case_ids.add(audit_case_id)
        by_id[canonical] = (claim, tuple(units), audit_case_id)
    if tuple(observed_order) != tuple(manifest_order):
        raise IndependentEvaluationError("3B1 request case order changed")
    return by_id


def _validate_coverage(
    coverage: Mapping[str, Any],
    gold_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Tuple[str, ...]], Mapping[str, Mapping[str, int]]]:
    labels_by_case: Dict[str, list[str]] = defaultdict(list)
    datasets = {}
    for row in gold_rows:
        canonical = str(row["canonical_case_id"])
        labels_by_case[canonical].append(str(row["final_relevance_label"]))
        prior = datasets.setdefault(canonical, row["dataset"])
        if prior != row["dataset"]:
            raise IndependentEvaluationError("3B2 case dataset identity changed")
    per_dataset_total: Counter[str] = Counter()
    per_dataset_evaluable: Counter[str] = Counter()
    positives: Dict[str, Tuple[str, ...]] = {}
    computed_case_coverage: Dict[str, Mapping[str, int]] = {}
    rows_by_case: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        rows_by_case[str(row["canonical_case_id"])].append(row)
    for canonical, rows in rows_by_case.items():
        dataset = str(rows[0]["dataset"])
        direct_ids = tuple(
            str(row["unit_id"])
            for row in rows
            if row["final_relevance_label"] == "DIRECT"
        )
        positives[canonical] = direct_ids
        per_dataset_total[dataset] += 1
        per_dataset_evaluable[dataset] += int(bool(direct_ids))
        counts = Counter(str(row["final_relevance_label"]) for row in rows)
        computed_case_coverage[canonical] = {
            "candidate_count": len(rows),
            "DIRECT_count": counts["DIRECT"],
            "RELATED_count": counts["RELATED"],
            "IRRELEVANT_count": counts["IRRELEVANT"],
            "UNREADABLE_count": counts["UNREADABLE"],
        }
    evaluable = sum(per_dataset_evaluable.values())
    expected_summary = {
        "status": "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_PASS",
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": EXPECTED_UNIT_COUNT,
        "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
        "zero_direct_positive_case_count": EXPECTED_ZERO_DIRECT_CASE_COUNT,
        "coverage_rate": EXPECTED_EVALUABLE_CASE_COUNT / EXPECTED_CASE_COUNT,
        "coverage_gate_minimum": COVERAGE_GATE_MINIMUM,
        "coverage_gate_pass": True,
        "resampling_performed": False,
    }
    _require_fields(coverage, expected_summary, "3B2 coverage report")
    if evaluable != EXPECTED_EVALUABLE_CASE_COUNT:
        raise IndependentEvaluationError("3B2 computed evaluable case count is not 28")
    expected_dataset = {
        dataset: {
            "total_case_count": per_dataset_total[dataset],
            "evaluable_case_count": per_dataset_evaluable[dataset],
        }
        for dataset in EXPECTED_DATASET_COUNTS
    }
    if expected_dataset != EXPECTED_DATASET_COUNTS or coverage.get("per_dataset") != expected_dataset:
        raise IndependentEvaluationError("3B2 dataset coverage changed")
    case_rows = coverage.get("case_coverage")
    if not isinstance(case_rows, list) or len(case_rows) != EXPECTED_CASE_COUNT:
        raise IndependentEvaluationError("3B2 case coverage ledger changed")
    seen = set()
    for row in case_rows:
        if not isinstance(row, Mapping):
            raise IndependentEvaluationError("3B2 case coverage row is invalid")
        canonical = row.get("canonical_case_id")
        computed = computed_case_coverage.get(canonical)
        if canonical in seen or computed is None:
            raise IndependentEvaluationError("3B2 case coverage identities changed")
        seen.add(canonical)
        expected = {
            "dataset": datasets[canonical],
            "canonical_case_id": canonical,
            **computed,
            "has_DIRECT": computed["DIRECT_count"] >= 1,
        }
        if dict(row) != expected:
            raise IndependentEvaluationError("3B2 case coverage content changed")
    return positives, computed_case_coverage


def _load_gold(
    final_gold_dir: Path,
    locks: Mapping[str, Mapping[str, str]],
    cohort_locks: Mapping[str, Mapping[str, str]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Tuple[Mapping[str, Any], ...], Mapping[str, Tuple[str, ...]]]:
    source_lock = _read_json(
        final_gold_dir / "final_gold_source_lock.json", "3B2 final-gold source lock"
    )
    _require_fields(
        source_lock,
        {
            "status": "PASS",
            "implementation_revision": SOURCE_3B2_REVISION,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
        },
        "3B2 final-gold source lock",
    )
    recorded_cohort = source_lock.get("cohort_public_artifacts")
    if not isinstance(recorded_cohort, Mapping):
        raise IndependentEvaluationError("3B2 source lock lacks 3B1 hash chain")
    for name in _COHORT_ARTIFACTS:
        recorded = recorded_cohort.get(name)
        if not isinstance(recorded, Mapping) or recorded.get("sha256") != cohort_locks[name]["sha256"]:
            raise IndependentEvaluationError(
                f"3B2 source lock disagrees with 3B1 artifact: {name}"
            )

    manifest = _read_json(
        final_gold_dir / "final_gold_manifest.json", "3B2 final-gold manifest"
    )
    _require_fields(
        manifest,
        {
            "status": "FINAL_RELEVANCE_GOLD_FROZEN",
            "implementation_revision": SOURCE_3B2_REVISION,
            "frozen_case_count": EXPECTED_CASE_COUNT,
            "frozen_unit_count": EXPECTED_UNIT_COUNT,
            "final_gold_fields": list(FINAL_GOLD_FIELDS),
            "adjudication_used": True,
            "coverage_gate_pass": True,
            "resampling_performed": False,
        },
        "3B2 final-gold manifest",
    )
    pointer_fields = {
        "final_relevance_gold_sha256": "final_relevance_gold.jsonl",
        "review_resolution_ledger_sha256": "review_resolution_ledger.jsonl",
        "coverage_report_sha256": "coverage_report.json",
        "source_lock_sha256": "final_gold_source_lock.json",
        "adjudication_frozen_csv_sha256": "adjudication_frozen.csv",
        "adjudication_provenance_sha256": "adjudication_provenance.json",
    }
    for field, name in pointer_fields.items():
        if manifest.get(field) != locks[name]["sha256"]:
            raise IndependentEvaluationError(f"3B2 manifest hash pointer failed: {field}")

    freeze_report = _read_json(
        final_gold_dir / "final_gold_freeze_report.json",
        "3B2 final-gold freeze report",
    )
    _require_fields(
        freeze_report,
        {
            "status": "FINAL_RELEVANCE_GOLD_FREEZE_PASS",
            "implementation_revision": SOURCE_3B2_REVISION,
            "frozen_case_count": EXPECTED_CASE_COUNT,
            "frozen_unit_count": EXPECTED_UNIT_COUNT,
            "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
            "coverage_gate_minimum": COVERAGE_GATE_MINIMUM,
            "coverage_gate_pass": True,
            "selector_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
            "resampling_performed": False,
            "step_3b3_executed": False,
        },
        "3B2 final-gold freeze report",
    )
    if freeze_report.get("final_gold_manifest_sha256") != locks["final_gold_manifest.json"]["sha256"]:
        raise IndependentEvaluationError("3B2 freeze report manifest pointer failed")

    rows = _read_jsonl(
        final_gold_dir / "final_relevance_gold.jsonl", "3B2 final relevance gold"
    )
    if len(rows) != EXPECTED_UNIT_COUNT:
        raise IndependentEvaluationError("3B2 final gold unit count is not 289")
    seen = set()
    by_case_positions: Dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if set(row) != set(FINAL_GOLD_FIELDS):
            raise IndependentEvaluationError("3B2 final gold schema changed")
        dataset = row.get("dataset")
        canonical = row.get("canonical_case_id")
        unit_id = row.get("unit_id")
        position = row.get("original_candidate_position")
        label = row.get("final_relevance_label")
        target = row.get("binary_direct_relevance_target")
        resolution = row.get("resolution_source")
        key = (dataset, canonical, unit_id)
        manifest_case = manifest_by_id.get(canonical)
        if (
            manifest_case is None
            or dataset != manifest_case.get("dataset")
            or key in seen
            or type(position) is not int
            or position < 0
            or label not in RELEVANCE_LABELS
            or target != (1 if label == "DIRECT" else 0)
            or resolution not in RESOLUTION_SOURCES
        ):
            raise IndependentEvaluationError("3B2 final gold row is invalid")
        ids = manifest_case["candidate_unit_ids_in_original_order"]
        if position >= len(ids) or unit_id != ids[position]:
            raise IndependentEvaluationError("3B2 final gold candidate position changed")
        seen.add(key)
        by_case_positions[canonical].append(position)
    if len(by_case_positions) != EXPECTED_CASE_COUNT or any(
        positions != list(range(len(positions)))
        for positions in by_case_positions.values()
    ):
        raise IndependentEvaluationError("3B2 final gold candidate set/order changed")
    resolution_rows = _read_jsonl(
        final_gold_dir / "review_resolution_ledger.jsonl",
        "3B2 review resolution ledger",
    )
    if len(resolution_rows) != EXPECTED_UNIT_COUNT:
        raise IndependentEvaluationError("3B2 resolution ledger count is not 289")
    coverage = _read_json(final_gold_dir / "coverage_report.json", "3B2 coverage report")
    positives, _ = _validate_coverage(coverage, rows)
    return rows, positives


def _validate_stage_a(
    report: Mapping[str, Any], selector_sha: str
) -> None:
    _require_fields(
        report,
        {
            "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
            "implementation_revision": "step2.6r-3-v1",
            "deployment_candidate_seed": REQUIRED_SEED,
            "deployment_candidate_selector_sha256": selector_sha,
            "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
            "prediction_invariance_gate": True,
            "selection_scores_changed": True,
            "encoder_hash_unchanged": True,
            "veracity_head_hash_unchanged": True,
            "selection_head_hash_changed": True,
            "heldout_relevance_cases_accessed": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "veracity_labels_inspected": False,
            "training_started": False,
            "optimizer_created": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
        },
        "Stage-A prediction-invariance report",
    )
    differences = report.get("state_difference_names")
    if (
        not isinstance(differences, list)
        or not differences
        or not set(differences) <= ALLOWED_STATE_DIFFERENCES
    ):
        raise IndependentEvaluationError("Stage-A state-difference whitelist failed")


def _load_config(path: Path) -> Mapping[str, Any]:
    return _read_json(path, "Phase4A configuration")


def _record_existing(
    path: Path,
    field: str,
    hashes: Dict[str, str],
) -> Mapping[str, str]:
    return _plain_lock(path.resolve(), field, hashes)


def _validate_project_root(project_root: Path) -> Mapping[str, Path]:
    """Require the UniRumor root layout consumed by the authoritative runtime."""

    if not project_root.is_dir():
        raise IndependentEvaluationError("UniRumor project root is missing")
    sources = {
        name: (project_root / relative_path).resolve()
        for name, relative_path in _AUTHORITATIVE_RUNTIME_SOURCES.items()
    }
    if any(not path.is_file() for path in sources.values()):
        raise IndependentEvaluationError(
            "project root does not contain the authoritative UniRumor runtime layout"
        )
    return sources


def _validate_phase4a_config(
    phase4a_config: Path,
    project_root: Path,
) -> Path:
    config = _load_config(phase4a_config)
    try:
        _require_config_value(
            config, {"model_name", "backbone"}, MODEL_NAME, "model_name"
        )
        _require_config_value(
            config,
            {"maximum_units_per_sample", "max_units"},
            MAXIMUM_UNITS_PER_SAMPLE,
            "maximum_units_per_sample",
        )
        _require_config_value(config, {"max_length"}, MAX_LENGTH, "max_length")
        _require_config_value(config, {"pooling"}, POOLING, "pooling")
    except Exception as exc:
        raise IndependentEvaluationError(
            "Phase4A configuration does not freeze the authoritative G1 contract"
        ) from exc
    try:
        checkpoint = Path(_checkpoint_path(config, project_root)).resolve()
        return safe_path(checkpoint, "Frozen G1 checkpoint")
    except Exception as exc:
        raise IndependentEvaluationError(
            "Phase4A configuration cannot resolve the Frozen G1 checkpoint"
        ) from exc


def _selector_metadata_lock(
    *,
    training: TrainingArtifacts,
    neutral_dir: Path,
    phase4a_config: Path,
    project_root: Path,
    cohort_source_lock: Mapping[str, Any],
    stage_a_report: Mapping[str, Any],
    hashes: Dict[str, str],
) -> Tuple[Mapping[str, Any], Path]:
    if training.selector_sha256 != CALIBRATED_SELECTOR_SHA256:
        raise IndependentEvaluationError("seed-42 selector SHA lock failed")
    selector_path = Path(training.selector_path).resolve()
    selector_lock = _plain_lock(
        selector_path,
        "seed-42 selector head",
        hashes,
        expected_sha256=CALIBRATED_SELECTOR_SHA256,
    )
    report = training.training_report
    _require_fields(
        report,
        {
            "status": "PASS",
            "run_mode": "full",
            "base_frozen_g1_checkpoint_sha256": BASE_G1_SHA256,
        },
        "closed selector training report",
    )
    summary = report.get("multi_seed_summary")
    if not isinstance(summary, Mapping) or summary.get("future_deployment_candidate_seed") != REQUIRED_SEED:
        raise IndependentEvaluationError("training report does not preregister seed 42")

    config_lock = _plain_lock(phase4a_config, "Phase4A configuration", hashes)
    artifacts = cohort_source_lock.get("artifacts")
    phase4a_source = artifacts.get("phase4a_configuration") if isinstance(artifacts, Mapping) else None
    if not isinstance(phase4a_source, Mapping) or phase4a_source.get("sha256") != config_lock["sha256"]:
        raise IndependentEvaluationError("Phase4A configuration differs from 3B1 lock")
    checkpoint = _validate_phase4a_config(phase4a_config, project_root)
    checkpoint_lock = _plain_lock(
        checkpoint,
        "Frozen G1 checkpoint",
        hashes,
        expected_sha256=BASE_G1_SHA256,
    )

    locked_files: Dict[str, Mapping[str, str]] = {
        "phase4a_configuration": config_lock,
        "frozen_g1_checkpoint": checkpoint_lock,
        "seed_42_selector_head": selector_lock,
    }
    training_dir = Path(training.training_dir).resolve()
    for label, path in (
        ("training_report", training_dir / "training_report.json"),
        ("multi_seed_summary", training_dir / "multi_seed_summary.json"),
    ):
        locked_files[label] = _record_existing(path, label, hashes)
    report_sidecar = training_dir / "training_report.json.sha256"
    locked_files["training_report_sha256_sidecar"] = _record_existing(
        report_sidecar, "training report SHA sidecar", hashes
    )
    neutral_names = (
        "neutral_calibration_train.jsonl",
        "neutral_calibration_train.sha256",
        "neutral_calibration_dev.jsonl",
        "neutral_calibration_dev.sha256",
        "neutral_revision_manifest.json",
        "neutral_revision_manifest.sha256",
        "neutral_build_report.json",
    )
    for name in neutral_names:
        locked_files[f"neutral/{name}"] = _record_existing(
            neutral_dir / name, f"neutral artifact {name}", hashes
        )
    for path, expected in training.immutable_file_hashes.items():
        resolved = Path(path).resolve()
        actual = sha256_file(resolved) if resolved.is_file() else None
        if actual != expected:
            raise IndependentEvaluationError(
                f"training validator immutable hash changed: {resolved.name}"
            )
        hashes[str(resolved)] = actual
    _validate_stage_a(stage_a_report, training.selector_sha256)
    return (
        {
            "status": "INDEPENDENT_SELECTOR_ARTIFACT_LOCK_PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "required_seed": REQUIRED_SEED,
            "selector_sha256": training.selector_sha256,
            "base_frozen_g1_checkpoint_sha256": checkpoint_lock["sha256"],
            "allowed_state_differences": sorted(ALLOWED_STATE_DIFFERENCES),
            "training_started": False,
            "optimizer_created": False,
            "post_audit_model_selection_performed": False,
            "immutable_artifacts": dict(sorted(locked_files.items())),
        },
        checkpoint,
    )


def prepare_inputs(
    *,
    cohort_dir: Path,
    final_gold_dir: Path,
    stage_a_invariance_report: Path,
    project_root: Path,
    phase4a_config: Path,
    neutral_dir: Path,
    training_dir: Path,
    training_validator: Callable[[Path, Path], TrainingArtifacts] = validate_training_artifacts,
) -> PreparedInputs:
    """Validate every frozen source without loading Torch or executing a model."""

    cohort = safe_path(cohort_dir, "3B1 cohort directory", reject_historical=True)
    gold = safe_path(final_gold_dir, "3B2 final-gold directory", reject_historical=True)
    stage_a = safe_path(stage_a_invariance_report, "Stage-A invariance report")
    project = safe_path(project_root, "project root")
    config = safe_path(phase4a_config, "Phase4A configuration")
    neutral = safe_path(neutral_dir, "neutral calibration directory")
    training_root = safe_path(training_dir, "training artifact directory")
    if not cohort.is_dir() or not gold.is_dir():
        raise IndependentEvaluationError("required source directory is missing")
    authoritative_sources = _validate_project_root(project)

    hashes: Dict[str, str] = {}
    cohort_locks: Dict[str, Mapping[str, str]] = {}
    # Identity-only metadata is opened before the request text artifact.
    for name in _COHORT_ARTIFACTS:
        if name == "independent_relevance_audit_requests.jsonl":
            continue
        cohort_locks[name] = _lock_file(
            cohort / name, f"3B1 {name}", hashes
        )
    selected = _read_json(
        cohort / "selected_case_manifest.json", "3B1 selected-case manifest"
    )
    manifest_order, manifest_by_id = _load_manifest_cases(selected)
    cohort_locks["independent_relevance_audit_requests.jsonl"] = _lock_file(
        cohort / "independent_relevance_audit_requests.jsonl",
        "3B1 independent audit requests",
        hashes,
    )

    build = _read_json(cohort / "build_report.json", "3B1 build report")
    _require_fields(
        build,
        {
            "status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS",
            "implementation_revision": SOURCE_3B1_REVISION,
            "authoritative_g1_case_count": 3878,
            "groundlie_source_case_count": 1636,
            "true3m_source_case_count": 2242,
            "selected_groundlie_count": 15,
            "selected_true3m_count": 15,
            "selected_total_count": EXPECTED_CASE_COUNT,
            "selected_candidate_unit_count": EXPECTED_UNIT_COUNT,
            "selection_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_heldout_reference_content_accessed": False,
        },
        "3B1 build report",
    )
    build_pointers = {
        "cohort_source_lock_sha256": "cohort_source_lock.json",
        "eligibility_inventory_sha256": "eligibility_inventory.json",
        "selected_case_manifest_sha256": "selected_case_manifest.json",
        "independent_relevance_audit_requests_sha256": "independent_relevance_audit_requests.jsonl",
        "preregistration_sha256": "independent_audit_preregistration.json",
    }
    for field, name in build_pointers.items():
        if build.get(field) != cohort_locks[name]["sha256"]:
            raise IndependentEvaluationError(f"3B1 build report hash pointer failed: {field}")
    inventory = _read_json(
        cohort / "eligibility_inventory.json", "3B1 eligibility inventory"
    )
    _require_fields(
        inventory,
        {
            "authoritative_source_case_count": 3878,
            "authoritative_groundlie_case_count": 1636,
            "authoritative_true3m_case_count": 2242,
            "selected_groundlie_count": 15,
            "selected_true3m_count": 15,
            "selected_total_count": EXPECTED_CASE_COUNT,
        },
        "3B1 eligibility inventory",
    )
    cohort_source = _read_json(
        cohort / "cohort_source_lock.json", "3B1 cohort source lock"
    )
    _require_fields(
        cohort_source,
        {
            "status": "PASS",
            "implementation_revision": SOURCE_3B1_REVISION,
            "sealed_historical_reference_artifacts_opened": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
        },
        "3B1 cohort source lock",
    )
    preregistration = _read_json(
        cohort / "independent_audit_preregistration.json", "3B1 preregistration"
    )
    preregistration_lock = _validate_preregistration(
        preregistration,
        cohort_locks["independent_audit_preregistration.json"]["sha256"],
    )
    requests_by_id = _load_requests(
        cohort / "independent_relevance_audit_requests.jsonl",
        manifest_order,
        manifest_by_id,
    )

    gold_locks: Dict[str, Mapping[str, str]] = {}
    for name in _GOLD_ARTIFACTS:
        gold_locks[name] = _lock_file(gold / name, f"3B2 {name}", hashes)
    gold_rows, positives = _load_gold(
        gold, gold_locks, cohort_locks, manifest_by_id
    )
    gold_by_case: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        gold_by_case[str(row["canonical_case_id"])].append(row)

    cases = []
    for canonical in manifest_order:
        claim, units, audit_case_id = requests_by_id[canonical]
        gold_case = gold_by_case.get(canonical)
        if gold_case is None or tuple(row["unit_id"] for row in gold_case) != tuple(
            unit.unit_id for unit in units
        ):
            raise IndependentEvaluationError("3B1-to-3B2 unit join is not one-to-one")
        cases.append(
            IndependentCase(
                audit_case_id=audit_case_id,
                dataset=str(gold_case[0]["dataset"]),
                canonical_case_id=canonical,
                claim=claim,
                candidate_units=units,
                positive_unit_ids=positives[canonical],
            )
        )

    stage_a_lock = _lock_file(stage_a, "Stage-A invariance report", hashes)
    stage_a_payload = _read_json(stage_a, "Stage-A invariance report")
    try:
        training_artifacts = training_validator(training_root, neutral)
    except IndependentEvaluationError:
        raise
    except Exception as exc:
        raise IndependentEvaluationError("closed training artifacts are invalid") from exc
    selector_artifact_lock, checkpoint = _selector_metadata_lock(
        training=training_artifacts,
        neutral_dir=neutral,
        phase4a_config=config,
        project_root=project,
        cohort_source_lock=cohort_source,
        stage_a_report=stage_a_payload,
        hashes=hashes,
    )
    source_lock = {
        "status": "INDEPENDENT_SELECTOR_ONE_SHOT_SOURCE_LOCK_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_root": str(project),
        "authoritative_runtime_sources": {
            name: _record_existing(path, f"authoritative runtime source {name}", hashes)
            for name, path in sorted(authoritative_sources.items())
        },
        "cohort_artifacts": dict(sorted(cohort_locks.items())),
        "final_gold_artifacts": dict(sorted(gold_locks.items())),
        "stage_a_invariance_report": stage_a_lock,
        "phase4a_configuration_sha256": hashes[str(config)],
        "frozen_g1_checkpoint_sha256": hashes[str(checkpoint)],
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
    }
    case_manifest = {
        "status": "INDEPENDENT_SELECTOR_EVALUATION_CASES_FROZEN",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": EXPECTED_UNIT_COUNT,
        "evaluable_case_count": EXPECTED_EVALUABLE_CASE_COUNT,
        "zero_direct_positive_case_count": EXPECTED_ZERO_DIRECT_CASE_COUNT,
        "per_dataset": EXPECTED_DATASET_COUNTS,
        "stable_original_candidate_order_for_score_ties": True,
        "zero_direct_positive_cases_retained": True,
        "positive_unit_definition": "all and only final_relevance_label == DIRECT",
        "cases": [case.manifest_row() for case in cases],
    }
    return PreparedInputs(
        cases=tuple(cases),
        project_root=project,
        source_lock=source_lock,
        case_manifest=case_manifest,
        preregistration_lock=preregistration_lock,
        selector_artifact_lock=selector_artifact_lock,
        immutable_file_hashes=dict(sorted(hashes.items())),
        training_artifacts=training_artifacts,
        checkpoint_path=checkpoint,
    )


def assert_hashes_unchanged(hashes: Mapping[str, str]) -> None:
    for raw_path, expected in hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise IndependentEvaluationError(
                f"immutable scientific input changed: {path.name}"
            )
