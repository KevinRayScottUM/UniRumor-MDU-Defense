"""Build the score-blind, Train-derived independent relevance-audit cohort.

The module is standard-library-only. It never imports a selector, checkpoint,
Torch, an optimizer, or model inference. Candidate exposure is delegated to
the real Phase4A request normalizer, except for explicit fixture adapters in
focused unit tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from scripts.selector_fidelity_audit.cross_case import (
    canonicalize_underlying_case_id,
)
from scripts.selector_relevance_calibration.dataset_builder import (
    ExposureResult,
    Phase4ANormalizationExposureAdapter,
    SourceCase,
    assess_source_case_provenance,
    iter_jsonl_rows,
    sha256_file,
    verify_train_lock,
)
from scripts.selector_relevance_training.trainer import AUTHORITATIVE_SOURCE_HASHES

from .blinding import build_review_packet, write_review_packet
from .schemas import (
    AUTHORITATIVE_TRAIN_SHA256,
    EXPECTED_SOURCE_COUNTS,
    IMPLEMENTATION_REVISION,
    REVIEWER_A_SALT,
    REVIEWER_B_SALT,
    SAMPLING_SALT,
    SEALED_CHALLENGE_IDS,
    STAGE_A_IDS,
    SUPPORTED_DATASETS,
    TARGET_DATASET_COUNTS,
    AuditCandidate,
    AuditCase,
)


EXPECTED_CALIBRATION_COUNTS = {"GroundLie360": 570, "TRUE-3MFact": 736}
EXPECTED_STAGE_A_COUNT = 7
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESTRICTED_PATH_PARTS = {
    "validation",
    "test",
    "formalvalidation",
    "formaltest",
}
_FORBIDDEN_PROJECTED_KEYS = frozenset(
    {
        "selection_score",
        "selection_scores",
        "selection_probability",
        "selection_probabilities",
        "selector_score",
        "selector_scores",
        "selector_logits",
        "selector_rank",
        "original_selector_rank",
        "calibrated_selector_rank",
        "veracity_logits",
        "sample_logits",
        "probabilities",
        "prediction",
        "predictions",
        "prediction_id",
        "top_k",
        "topk",
        "top_k_selection_units",
        "rank",
    }
)
_VISUAL_MARKERS = frozenset(
    {
        "visual",
        "visual_observation",
        "grounded_visual_unit",
        "image",
        "video_frame",
    }
)


class IndependentAuditBuildError(RuntimeError):
    """Raised when the independent audit cannot preserve its frozen protocol."""


class ExposureAdapter(Protocol):
    def normalize(self, request: Mapping[str, Any]) -> ExposureResult:
        ...


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    build_report: Mapping[str, Any]


@dataclass(frozen=True)
class _SourceRecord:
    dataset: str
    original_case_id: str
    canonical_case_id: str
    claim: str
    candidates: Tuple[Mapping[str, Any], ...]
    row_index: int
    source_split: str


def _require_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
        raise IndependentAuditBuildError(f"{field} must be a SHA-256 digest")
    return value.casefold()


def _safe_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in _RESTRICTED_PATH_PARTS for part in resolved.parts):
        raise IndependentAuditBuildError(
            f"{field} must not reference Formal Validation/Test"
        )
    return resolved


def _verify_file(path: Path, expected_sha256: str, field: str) -> Tuple[Path, str]:
    resolved = _safe_path(path, field)
    if not resolved.is_file():
        raise IndependentAuditBuildError(f"{field} is missing")
    expected = _require_digest(expected_sha256, f"{field} expected SHA-256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise IndependentAuditBuildError(f"{field} SHA-256 mismatch")
    return resolved, actual


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentAuditBuildError(f"{field} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise IndependentAuditBuildError(f"{field} must be a JSON object")
    return value


def _read_jsonl(path: Path, field: str) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise IndependentAuditBuildError(
                        f"{field} row {index} must be an object"
                    )
                yield index, value
    except json.JSONDecodeError as exc:
        raise IndependentAuditBuildError(f"{field} contains malformed JSONL") from exc


def _identity(row: Mapping[str, Any], field: str) -> Tuple[str, str, str]:
    dataset = row.get("dataset", row.get("source_dataset"))
    case_id = row.get("case_id", row.get("source_case_id"))
    if case_id is None:
        case_id = row.get("original_case_id", row.get("sample_id", row.get("id")))
    canonical = row.get(
        "canonical_case_id", row.get("canonical_underlying_case_id")
    )
    if not isinstance(dataset, str) or dataset not in SUPPORTED_DATASETS:
        raise IndependentAuditBuildError(f"{field} dataset is not supported")
    if canonical is None:
        if not isinstance(case_id, (str, int)) or not str(case_id).strip():
            raise IndependentAuditBuildError(f"{field} case identity is missing")
        canonical = canonicalize_underlying_case_id(dataset, str(case_id).strip())
    if not isinstance(canonical, str) or not canonical.strip():
        raise IndependentAuditBuildError(f"{field} canonical identity is missing")
    if not isinstance(case_id, (str, int)) or not str(case_id).strip():
        parts = canonical.split(":", 1)
        case_id = parts[1] if len(parts) == 2 else canonical
    expected = canonicalize_underlying_case_id(dataset, str(case_id).strip())
    if canonical.strip() != expected:
        raise IndependentAuditBuildError(f"{field} identity is inconsistent")
    return dataset, str(case_id).strip(), canonical.strip()


def _contains_forbidden_projected_key(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_PROJECTED_KEYS:
                return str(key)
            match = _contains_forbidden_projected_key(nested)
            if match is not None:
                return match
    elif isinstance(value, (list, tuple)):
        for nested in value:
            match = _contains_forbidden_projected_key(nested)
            if match is not None:
                return match
    return None


def _load_calibration_exclusions(
    neutral_dir: Path,
    *,
    expected_hashes: Mapping[str, str],
    expected_counts: Mapping[str, int],
) -> Tuple[frozenset[str], Mapping[str, Mapping[str, str]]]:
    directory = _safe_path(neutral_dir, "neutral calibration source")
    if not directory.is_dir():
        raise IndependentAuditBuildError("neutral calibration source is missing")
    locks: Dict[str, Mapping[str, str]] = {}
    for name, expected in expected_hashes.items():
        path, actual = _verify_file(directory / name, expected, f"neutral {name}")
        sidecar = directory / (name.rsplit(".", 1)[0] + ".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip().casefold() != actual:
            raise IndependentAuditBuildError(f"neutral {name} SHA sidecar mismatch")
        locks[name] = {"path": str(path), "sha256": actual}
    report = _read_json(directory / "neutral_build_report.json", "neutral build report")
    if report.get("status") != "PASS" or report.get("implementation_revision") != "step2.6r-1d-v1":
        raise IndependentAuditBuildError("neutral artifacts are not the closed revision")
    identities: Dict[str, str] = {}
    split_sets = []
    for name in ("neutral_calibration_train.jsonl", "neutral_calibration_dev.jsonl"):
        current = set()
        for index, row in _read_jsonl(directory / name, name):
            dataset, _, canonical = _identity(row, f"{name} row {index}")
            prior = identities.setdefault(canonical, dataset)
            if prior != dataset:
                raise IndependentAuditBuildError("neutral identity dataset changed")
            current.add(canonical)
        split_sets.append(current)
    if split_sets[0] & split_sets[1]:
        raise IndependentAuditBuildError("neutral Train/Dev case overlap")
    counts = Counter(identities.values())
    if dict(counts) != dict(expected_counts):
        raise IndependentAuditBuildError("neutral calibration identity counts mismatch")
    return frozenset(identities), locks


def _load_stage_a_exclusions(
    *,
    report_path: Path,
    report_sha256: str,
    replay_path: Path,
    replay_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    expected_ids: frozenset[str],
) -> Tuple[frozenset[str], Mapping[str, Mapping[str, str]]]:
    report_path, report_actual = _verify_file(
        report_path, report_sha256, "Stage-A prediction-invariance report"
    )
    replay_path, replay_actual = _verify_file(
        replay_path, replay_sha256, "Stage-A normalized replay"
    )
    manifest_path, manifest_actual = _verify_file(
        manifest_path, manifest_sha256, "Stage-A normalized replay manifest"
    )
    report = _read_json(report_path, "Stage-A prediction-invariance report")
    if report.get("status") != "PREDICTION_INVARIANCE_SMOKE_PASS":
        raise IndependentAuditBuildError("Stage-A report status is not PASS")
    required_exact = {
        "exact_phase4a_replay_request_count": EXPECTED_STAGE_A_COUNT,
        "candidate_id_mismatch_count": 0,
        "candidate_order_mismatch_count": 0,
        "maximum_unit_veracity_logit_difference": 0.0,
        "maximum_sample_logit_difference": 0.0,
        "maximum_probability_difference": 0.0,
        "prediction_mismatch_count": 0,
        "encoder_hash_unchanged": True,
        "veracity_head_hash_unchanged": True,
        "heldout_relevance_cases_accessed": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "training_started": False,
        "optimizer_created": False,
    }
    for field, expected in required_exact.items():
        if report.get(field) != expected:
            raise IndependentAuditBuildError(f"Stage-A report gate failed: {field}")
    if report.get("phase4a_replay_artifact_sha256") != replay_actual:
        raise IndependentAuditBuildError("Stage-A report replay SHA mismatch")
    if report.get("phase4a_replay_manifest_sha256") != manifest_actual:
        raise IndependentAuditBuildError("Stage-A report manifest SHA mismatch")
    manifest = _read_json(manifest_path, "Stage-A normalized replay manifest")
    if manifest.get("status") != "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS":
        raise IndependentAuditBuildError("Stage-A replay manifest status is not PASS")
    if manifest.get("normalized_artifact_sha256") != replay_actual:
        raise IndependentAuditBuildError("Stage-A replay manifest SHA mismatch")
    identities = set()
    for index, row in _read_jsonl(replay_path, "Stage-A normalized replay"):
        dataset, _, canonical = _identity(row, f"Stage-A replay row {index}")
        if dataset != "GroundLie360":
            raise IndependentAuditBuildError("Stage-A dataset is unexpected")
        identities.add(canonical)
    if frozenset(identities) != expected_ids:
        raise IndependentAuditBuildError("Stage-A replay identities mismatch")
    locks = {
        "stage_a_prediction_invariance_report": {
            "path": str(report_path),
            "sha256": report_actual,
        },
        "stage_a_normalized_replay": {
            "path": str(replay_path),
            "sha256": replay_actual,
        },
        "stage_a_normalized_replay_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_actual,
        },
    }
    return frozenset(identities), locks


def _load_additional_exclusions(paths: Sequence[Path]) -> Tuple[frozenset[str], Mapping[str, Mapping[str, str]]]:
    identities = set()
    locks: Dict[str, Mapping[str, str]] = {}
    for index, raw_path in enumerate(paths):
        path = _safe_path(raw_path, "additional identity-only exclusion manifest")
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise IndependentAuditBuildError("additional exclusion manifest or sidecar missing")
        actual = sha256_file(path)
        if sidecar.read_text(encoding="utf-8").strip().casefold() != actual:
            raise IndependentAuditBuildError("additional exclusion manifest SHA mismatch")
        payload = _read_json(path, "additional identity-only exclusion manifest")
        values = payload.get("canonical_case_ids")
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise IndependentAuditBuildError("additional exclusion manifest is not identity-only")
        for value in values:
            dataset = value.split(":", 1)[0]
            if dataset not in SUPPORTED_DATASETS or ":" not in value:
                raise IndependentAuditBuildError("additional exclusion identity is invalid")
            identities.add(value)
        locks[f"additional_exclusion_manifest_{index}"] = {
            "path": str(path),
            "sha256": actual,
        }
    return frozenset(identities), locks


def _source_record(row: Mapping[str, Any], index: int) -> _SourceRecord:
    dataset, original, canonical = _identity(row, f"Train row {index}")
    split = row.get("split", "train")
    if isinstance(split, str) and split.strip() and split.strip().casefold() != "train":
        raise IndependentAuditBuildError("authoritative source contains non-Train row")
    claim = row.get("claim")
    candidates = row.get("candidate_units")
    if not isinstance(claim, str) or not claim.strip():
        raise IndependentAuditBuildError("Train row claim is missing")
    if not isinstance(candidates, list) or not all(isinstance(item, Mapping) for item in candidates):
        raise IndependentAuditBuildError("Train row candidate_units are invalid")
    forbidden = _contains_forbidden_projected_key(row)
    if forbidden is not None:
        raise IndependentAuditBuildError(f"forbidden selector/model output supplied: {forbidden}")
    return _SourceRecord(
        dataset=dataset,
        original_case_id=original,
        canonical_case_id=canonical,
        claim=claim,
        candidates=tuple(candidates),
        row_index=index,
        source_split="train",
    )


def _expose(record: _SourceRecord, train_sha256: str, adapter: ExposureAdapter) -> Optional[Tuple[AuditCandidate, ...]]:
    source_case = SourceCase(
        source_dataset=record.dataset,
        source_case_id=record.original_case_id,
        canonical_underlying_case_id=record.canonical_case_id,
        source_split=record.source_split,
        source_row_index=record.row_index,
        request={},
        candidates=record.candidates,
    )
    provenance = assess_source_case_provenance(source_case, train_sha256)
    if provenance.ambiguous:
        raise IndependentAuditBuildError("ambiguous candidate provenance is forbidden")
    request = {
        "dataset": record.dataset,
        "case_id": record.original_case_id,
        "claim": record.claim,
        "candidate_units": [
            {
                "unit_id": item.get("unit_id"),
                "unit_type": item.get("unit_type"),
                "modality": item.get("modality"),
                "text": item.get("text"),
            }
            for item in record.candidates
        ],
    }
    try:
        exposed = adapter.normalize(request)
    except Exception:
        return None
    if not isinstance(exposed, ExposureResult):
        raise IndependentAuditBuildError("exposure adapter returned an invalid result")
    if len(exposed.candidate_units) > 24:
        raise IndependentAuditBuildError("Phase4A exposure exceeded 24 candidates")
    candidates = []
    seen = set()
    for position, item in enumerate(exposed.candidate_units):
        forbidden = _contains_forbidden_projected_key(item)
        if forbidden is not None:
            raise IndependentAuditBuildError(f"Phase4A exposure contains forbidden field: {forbidden}")
        unit_id = item.get("unit_id")
        unit_type = item.get("unit_type")
        modality = item.get("modality")
        text = item.get("text")
        values = (unit_id, unit_type, modality, text)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise IndependentAuditBuildError("exposed candidate fields must be nonblank")
        if unit_type.casefold() in _VISUAL_MARKERS or modality.casefold() in _VISUAL_MARKERS:
            raise IndependentAuditBuildError("visual candidate survived Phase4A normalization")
        if modality.casefold() not in {"text", "ocr"}:
            raise IndependentAuditBuildError("unsupported modality survived Phase4A normalization")
        if unit_id in seen:
            raise IndependentAuditBuildError("exposed candidate unit IDs are not unique")
        seen.add(unit_id)
        candidates.append(
            AuditCandidate(
                unit_id=unit_id,
                unit_type=unit_type,
                modality=modality,
                text=text,
                original_candidate_position=position,
            )
        )
    return tuple(candidates)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for value in values
    )


def _write_artifact(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _preregistration() -> Mapping[str, Any]:
    return {
        "status": "PREREGISTERED_BEFORE_REVIEW_AND_SCORING",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "cohort_description": "independent score-blind Train-derived direct-relevance audit cohort",
        "deployment_candidate_seed": 42,
        "direct_relevance_binary_mapping": {
            "DIRECT": 1,
            "RELATED": 0,
            "IRRELEVANT": 0,
            "UNREADABLE": 0,
        },
        "future_step_2_6r_3b2": {
            "name": "Independent Review Freeze and Agreement Audit",
            "selector_scoring_permitted_before_completion": False,
            "required_freeze": [
                "Reviewer A completed CSV and provenance hash",
                "Reviewer B completed CSV and provenance hash",
                "hidden mapping application",
                "review agreement computation",
            ],
            "disagreement_rule": "NEEDS_ADJUDICATION",
            "automatic_disagreement_resolution": False,
            "adjudication_selector_score_blind": True,
        },
        "future_step_2_6r_3b3": {
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
        },
        "prohibitions": {
            "training": True,
            "calibration": True,
            "seed_42_43_44_selection": True,
            "selector_architecture_change": True,
            "threshold_tuning": True,
            "iterative_reuse_after_scores": True,
        },
        "interpretation_boundary": "independent Train-derived repair verification; not Formal Validation/Test or population-level generalization",
    }


def build_cohort(
    *,
    project_root: Path,
    phase3a_train_lock_report: Path,
    phase3a_train_lock_report_sha256: str,
    phase4a_config_path: Path,
    phase4a_config_sha256: str,
    neutral_dir: Path,
    stage_a_report_path: Path,
    stage_a_report_sha256: str,
    stage_a_replay_path: Path,
    stage_a_replay_sha256: str,
    stage_a_replay_manifest_path: Path,
    stage_a_replay_manifest_sha256: str,
    output_dir: Path,
    additional_exclusion_manifests: Sequence[Path] = (),
    exposure_adapter: Optional[ExposureAdapter] = None,
    expected_train_sha256: str = AUTHORITATIVE_TRAIN_SHA256,
    expected_source_counts: Mapping[str, int] = EXPECTED_SOURCE_COUNTS,
    expected_neutral_hashes: Mapping[str, str] = AUTHORITATIVE_SOURCE_HASHES,
    expected_calibration_counts: Mapping[str, int] = EXPECTED_CALIBRATION_COUNTS,
    target_dataset_counts: Mapping[str, int] = TARGET_DATASET_COUNTS,
) -> BuildResult:
    """Build and atomically freeze the independent audit artifacts."""

    project_root = Path(project_root).expanduser().resolve()
    output = _safe_path(output_dir, "output directory")
    if output.exists():
        raise IndependentAuditBuildError("output directory already exists")
    report_path, report_hash = _verify_file(
        phase3a_train_lock_report,
        phase3a_train_lock_report_sha256,
        "Phase3A Train-lock report",
    )
    try:
        train_lock = verify_train_lock(
            project_root, report_path, expected_sha256=expected_train_sha256
        )
    except Exception as exc:
        raise IndependentAuditBuildError("authoritative Train lock failed") from exc
    config_path, config_hash = _verify_file(
        phase4a_config_path, phase4a_config_sha256, "Phase4A configuration"
    )
    calibration_ids, neutral_locks = _load_calibration_exclusions(
        neutral_dir,
        expected_hashes=expected_neutral_hashes,
        expected_counts=expected_calibration_counts,
    )
    stage_a_ids, stage_a_locks = _load_stage_a_exclusions(
        report_path=stage_a_report_path,
        report_sha256=stage_a_report_sha256,
        replay_path=stage_a_replay_path,
        replay_sha256=stage_a_replay_sha256,
        manifest_path=stage_a_replay_manifest_path,
        manifest_sha256=stage_a_replay_manifest_sha256,
        expected_ids=STAGE_A_IDS,
    )
    additional_ids, additional_locks = _load_additional_exclusions(
        additional_exclusion_manifests
    )
    if exposure_adapter is None:
        try:
            adapter: ExposureAdapter = Phase4ANormalizationExposureAdapter.from_project_root(
                project_root, config_path
            )
        except Exception as exc:
            raise IndependentAuditBuildError(
                "real Phase4A exposure adapter is required; no fallback is permitted"
            ) from exc
    else:
        adapter = exposure_adapter

    raw_counts: Counter[str] = Counter()
    seen_ids = set()
    exclusion_counts: Counter[str] = Counter()
    remaining_records = []
    for index, row in iter_jsonl_rows(train_lock.source_path):
        dataset, _, canonical = _identity(row, f"Train row {index}")
        raw_counts[dataset] += 1
        if canonical in seen_ids:
            raise IndependentAuditBuildError("authoritative Train case IDs are not unique")
        seen_ids.add(canonical)
        # The sealed set is checked first; claim/candidate fields are never inspected.
        if canonical in SEALED_CHALLENGE_IDS:
            exclusion_counts["sealed"] += 1
            continue
        if canonical in stage_a_ids:
            exclusion_counts["stage_a"] += 1
            continue
        if canonical in calibration_ids:
            exclusion_counts["calibration"] += 1
            continue
        if canonical in additional_ids:
            exclusion_counts["additional"] += 1
            continue
        remaining_records.append(_source_record(row, index))
    if len(seen_ids) != sum(expected_source_counts.values()):
        raise IndependentAuditBuildError("authoritative source case count mismatch")
    if dict(raw_counts) != dict(expected_source_counts):
        raise IndependentAuditBuildError("authoritative source dataset counts mismatch")
    if exclusion_counts["calibration"] != sum(expected_calibration_counts.values()):
        raise IndependentAuditBuildError("calibration exclusion count mismatch")
    if exclusion_counts["sealed"] != len(SEALED_CHALLENGE_IDS):
        raise IndependentAuditBuildError("sealed challenge exclusion count mismatch")
    if exclusion_counts["stage_a"] != len(STAGE_A_IDS):
        raise IndependentAuditBuildError("Stage-A exclusion count mismatch")

    exposure_success = 0
    exposure_failure = 0
    below_six = 0
    eligible_pairs: Dict[str, list[Tuple[_SourceRecord, Tuple[AuditCandidate, ...]]]] = {
        dataset: [] for dataset in SUPPORTED_DATASETS
    }
    for record in remaining_records:
        candidates = _expose(record, train_lock.source_sha256, adapter)
        if candidates is None:
            exposure_failure += 1
            continue
        exposure_success += 1
        if len(candidates) < 6:
            below_six += 1
            continue
        eligible_pairs[record.dataset].append((record, candidates))

    selected = []
    for dataset in SUPPORTED_DATASETS:
        ranked = sorted(
            eligible_pairs[dataset],
            key=lambda pair: (
                _hash_text(f"{SAMPLING_SALT}|{dataset}|{pair[0].canonical_case_id}"),
                pair[0].canonical_case_id,
            ),
        )
        target = target_dataset_counts[dataset]
        if len(ranked) < target:
            raise IndependentAuditBuildError(f"{dataset} has fewer than {target} eligible cases")
        for record, candidates in ranked[:target]:
            sampling_hash = _hash_text(
                f"{SAMPLING_SALT}|{dataset}|{record.canonical_case_id}"
            )
            selected.append(
                AuditCase(
                    audit_case_id="audit-" + sampling_hash[:24],
                    dataset=dataset,
                    canonical_case_id=record.canonical_case_id,
                    original_case_id=record.original_case_id,
                    claim=record.claim,
                    candidates=candidates,
                    sampling_hash=sampling_hash,
                )
            )
    expected_selected_total = sum(target_dataset_counts.values())
    selected_counts = Counter(case.dataset for case in selected)
    if len(selected) != expected_selected_total or dict(selected_counts) != dict(target_dataset_counts):
        raise IndependentAuditBuildError("selected cohort counts mismatch")
    forbidden_ids = calibration_ids | SEALED_CHALLENGE_IDS | stage_a_ids | additional_ids
    if any(case.canonical_case_id in forbidden_ids for case in selected):
        raise IndependentAuditBuildError("an excluded identity entered the cohort")

    reviewer_a = build_review_packet(selected, reviewer="A", salt=REVIEWER_A_SALT)
    reviewer_b = build_review_packet(selected, reviewer="B", salt=REVIEWER_B_SALT)
    if set(reviewer_a.underlying_unit_keys) != set(reviewer_b.underlying_unit_keys):
        raise IndependentAuditBuildError("review packets do not cover identical units")
    if reviewer_a.case_order == reviewer_b.case_order:
        raise IndependentAuditBuildError("independent reviewer case ordering did not differ")
    if [row["review_unit_id"] for row in reviewer_a.rows] == [row["review_unit_id"] for row in reviewer_b.rows]:
        raise IndependentAuditBuildError("independent reviewer IDs did not differ")

    inventory = {
        "authoritative_source_case_count": len(seen_ids),
        "authoritative_groundlie_case_count": raw_counts["GroundLie360"],
        "authoritative_true3m_case_count": raw_counts["TRUE-3MFact"],
        "calibration_exclusion_count": exclusion_counts["calibration"],
        "sealed_challenge_exclusion_count": exclusion_counts["sealed"],
        "stage_a_exclusion_count": exclusion_counts["stage_a"],
        "additional_prior_audit_exclusion_count": exclusion_counts["additional"],
        "remaining_after_exclusions": len(remaining_records),
        "phase4a_exposure_success_count": exposure_success,
        "phase4a_exposure_failure_count": exposure_failure,
        "candidate_count_below_6_count": below_six,
        "candidate_count_6_to_24_count": sum(len(items) for items in eligible_pairs.values()),
        "visual_rejection_count": 0,
        "eligible_groundlie_count": len(eligible_pairs["GroundLie360"]),
        "eligible_true3m_count": len(eligible_pairs["TRUE-3MFact"]),
        "selected_groundlie_count": selected_counts["GroundLie360"],
        "selected_true3m_count": selected_counts["TRUE-3MFact"],
        "selected_total_count": len(selected),
    }
    source_lock = {
        "status": "PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "artifacts": {
            "phase3a_train_lock_report": {"path": str(report_path), "sha256": report_hash},
            "authoritative_g1_train": {"path": str(train_lock.source_path), "sha256": train_lock.source_sha256},
            "phase4a_configuration": {"path": str(config_path), "sha256": config_hash},
            **neutral_locks,
            **stage_a_locks,
            **additional_locks,
        },
        "sealed_historical_reference_artifacts_opened": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
    }
    preregistration = _preregistration()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".independent-audit-", dir=output.parent))
    try:
        source_lock_sha = _write_artifact(staging / "cohort_source_lock.json", _json_bytes(source_lock))
        inventory_sha = _write_artifact(staging / "eligibility_inventory.json", _json_bytes(inventory))
        selected_manifest = {
            "status": "FROZEN",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "sampling_salt": SAMPLING_SALT,
            "selected_cases": [case.manifest_dict() for case in selected],
        }
        selected_manifest_sha = _write_artifact(staging / "selected_case_manifest.json", _json_bytes(selected_manifest))
        requests_sha = _write_artifact(
            staging / "independent_relevance_audit_requests.jsonl",
            _jsonl_bytes([case.request_dict() for case in selected]),
        )
        reviewer_a_sha = write_review_packet(staging / "reviewer_A", reviewer_a)
        reviewer_b_sha = write_review_packet(staging / "reviewer_B", reviewer_b)
        mapping_payload = {
            "status": "PRIVATE_FROZEN_MAPPING",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "reviewer_A": list(reviewer_a.mapping),
            "reviewer_B": list(reviewer_b.mapping),
        }
        private_mapping_sha = _write_artifact(staging / "private_review_mapping.json", _json_bytes(mapping_payload))
        preregistration_sha = _write_artifact(
            staging / "independent_audit_preregistration.json",
            _json_bytes(preregistration),
        )
        candidate_counts = [len(case.candidates) for case in selected]
        build_report = {
            "status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "authoritative_g1_sha256": train_lock.source_sha256,
            "authoritative_g1_case_count": len(seen_ids),
            "groundlie_source_case_count": raw_counts["GroundLie360"],
            "true3m_source_case_count": raw_counts["TRUE-3MFact"],
            "calibration_exclusion_count": exclusion_counts["calibration"],
            "sealed_challenge_exclusion_count": exclusion_counts["sealed"],
            "stage_a_exclusion_count": exclusion_counts["stage_a"],
            "additional_prior_audit_exclusion_count": exclusion_counts["additional"],
            "eligible_groundlie_count": len(eligible_pairs["GroundLie360"]),
            "eligible_true3m_count": len(eligible_pairs["TRUE-3MFact"]),
            "selected_groundlie_count": selected_counts["GroundLie360"],
            "selected_true3m_count": selected_counts["TRUE-3MFact"],
            "selected_total_count": len(selected),
            "minimum_candidate_count": min(candidate_counts),
            "maximum_candidate_count": max(candidate_counts),
            "selected_candidate_unit_count": sum(candidate_counts),
            "reviewer_a_row_count": len(reviewer_a.rows),
            "reviewer_b_row_count": len(reviewer_b.rows),
            "reviewer_a_packet_sha256": reviewer_a_sha,
            "reviewer_b_packet_sha256": reviewer_b_sha,
            "private_mapping_sha256": private_mapping_sha,
            "preregistration_sha256": preregistration_sha,
            "cohort_source_lock_sha256": source_lock_sha,
            "eligibility_inventory_sha256": inventory_sha,
            "selected_case_manifest_sha256": selected_manifest_sha,
            "independent_relevance_audit_requests_sha256": requests_sha,
            "veracity_labels_used_for_sampling": False,
            "veracity_labels_emitted": False,
            "veracity_labels_inspected_for_selection": False,
            "selection_scores_accessed": False,
            "original_selector_accessed": False,
            "calibrated_selector_accessed": False,
            "veracity_logits_accessed": False,
            "predictions_accessed": False,
            "checkpoint_loaded": False,
            "model_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_heldout_reference_content_accessed": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
        }
        build_report_sha = _write_artifact(staging / "build_report.json", _json_bytes(build_report))
        if build_report_sha != sha256_file(staging / "build_report.json"):
            raise IndependentAuditBuildError("build report hash changed")
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BuildResult(output_dir=output, build_report=build_report)
