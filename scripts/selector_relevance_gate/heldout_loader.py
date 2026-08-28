"""Fail-closed loaders for label-free replay and held-out audit references."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .phase4a_normalizer import (
    AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
    CPAC_CANONICAL_CASE_ID,
    EXCLUSION_REASON,
    EXPECTED_HISTORICAL_PHASE4A_COUNT,
    EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT,
    EXPECTED_STAGE_A_REQUEST_COUNT,
    IMPLEMENTATION_REVISION as NORMALIZATION_IMPLEMENTATION_REVISION,
    NORMALIZED_SCHEMA_FIELDS,
    Phase4ANormalizationError,
    PROTECTED_HELDOUT_CASE_IDS,
    SOURCE_SCHEMA_FIELDS,
    canonical_underlying_case_id,
    request_content_sha256,
)
from .schemas import EvaluationRequest, EvaluationUnit


EXPECTED_HELDOUT_CASE_IDS = (
    "GroundLie360:13025004",
    "TRUE-3MFact:10145403",
    "TRUE-3MFact:10258205",
    "TRUE-3MFact:10372904",
    "TRUE-3MFact:10455808",
    "TRUE-3MFact:10865013",
)
CPAC_REFERENCE_ID = "ocr_01_direct_full_banner"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "label",
    "labels",
    "ground_truth",
    "ground_truth_label",
    "veracity_label",
    "veracity_labels",
    "model_verdict",
    "display_verdict",
    "prediction",
    "prediction_id",
    "veracity_logits",
    "sample_logits",
    "probabilities",
    "selection_score",
}


class ReferenceInputError(ValueError):
    """Raised when an evaluation input violates the preregistered boundary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in {"validation", "test"} for part in resolved.parts):
        raise ReferenceInputError(f"{field} cannot access Formal Validation/Test")
    if not resolved.is_file():
        raise ReferenceInputError(f"{field} is missing")
    return resolved


def _expected_sha(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
        raise ReferenceInputError(f"{field} must be an independently supplied SHA-256")
    return value.casefold()


def _read_verified_json(path: Path, expected_sha256: str, field: str) -> Tuple[Path, Any]:
    resolved = _resolve_input(path, field)
    expected = _expected_sha(expected_sha256, f"{field} SHA-256")
    if sha256_file(resolved) != expected:
        raise ReferenceInputError(f"{field} SHA-256 mismatch")
    try:
        return resolved, json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceInputError(f"{field} is malformed JSON") from exc


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceInputError(f"{field} must be nonblank")
    if value != value.strip():
        raise ReferenceInputError(f"{field} must preserve exact surrounding whitespace")
    return value


def _string_tuple(value: Any, field: str, *, allow_empty: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ReferenceInputError(f"{field} must be a list")
    items = tuple(_nonblank(item, field) for item in value)
    if not allow_empty and not items:
        raise ReferenceInputError(f"{field} must not be empty")
    if len(set(items)) != len(items):
        raise ReferenceInputError(f"{field} contains duplicate IDs")
    return items


def _dataset(value: Any, field: str) -> str:
    dataset = _nonblank(value, field)
    if dataset.casefold() in {"validation", "test"}:
        raise ReferenceInputError("Formal Validation/Test dataset labels are forbidden")
    return dataset


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_LABEL_KEYS:
                return True
            if _contains_forbidden_label_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _parse_units(value: Any, field: str) -> Tuple[EvaluationUnit, ...]:
    if not isinstance(value, list) or not value:
        raise ReferenceInputError(f"{field} must be a nonempty list")
    units = []
    allowed = {"unit_id", "unit_type", "modality", "text"}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != allowed:
            raise ReferenceInputError(f"{field}[{index}] has an invalid schema")
        try:
            units.append(
                EvaluationUnit(
                    unit_id=_nonblank(item.get("unit_id"), f"{field}[{index}].unit_id"),
                    unit_type=_nonblank(
                        item.get("unit_type"), f"{field}[{index}].unit_type"
                    ),
                    modality=_nonblank(
                        item.get("modality"), f"{field}[{index}].modality"
                    ),
                    text=_nonblank(item.get("text"), f"{field}[{index}].text"),
                )
            )
        except ValueError as exc:
            raise ReferenceInputError(str(exc)) from exc
    return tuple(units)


def _derive_expected_normalized_protocol(
    source_path: Path,
) -> Tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Re-project the verified historical source to bind Stage A to its content."""

    try:
        source_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceInputError("historical Phase4A source artifact is malformed") from exc
    if len(source_rows) != EXPECTED_HISTORICAL_PHASE4A_COUNT:
        raise ReferenceInputError(
            "historical Phase4A source artifact must contain exactly 8 requests"
        )
    if _contains_forbidden_label_key(source_rows):
        raise ReferenceInputError(
            "historical Phase4A source artifact contains forbidden labels or outputs"
        )

    normalized_rows = []
    excluded_records = []
    retained_records = []
    for row_index, row in enumerate(source_rows):
        if not isinstance(row, Mapping) or set(row) != set(SOURCE_SCHEMA_FIELDS):
            raise ReferenceInputError(
                f"historical Phase4A source request {row_index} has invalid schema"
            )
        historical_case_id = _nonblank(
            row.get("case_id"), f"historical[{row_index}].case_id"
        )
        source_case_id = _nonblank(
            row.get("source_case_id"), f"historical[{row_index}].source_case_id"
        )
        dataset = _dataset(row.get("dataset"), f"historical[{row_index}].dataset")
        claim = _nonblank(row.get("claim"), f"historical[{row_index}].claim")
        _parse_units(
            row.get("candidate_units"), f"historical[{row_index}].candidate_units"
        )
        try:
            canonical = canonical_underlying_case_id(
                historical_case_id, source_case_id
            )
        except Phase4ANormalizationError as exc:
            raise ReferenceInputError(str(exc)) from exc
        if canonical.rsplit(":", 1)[0] != dataset:
            raise ReferenceInputError(
                "historical Phase4A dataset and source identity disagree"
            )
        normalized = {
            "case_id": historical_case_id,
            "dataset": dataset,
            "claim": claim,
            "candidate_units": row["candidate_units"],
        }
        record = {
            "historical_case_id": historical_case_id,
            "source_case_id": source_case_id,
            "canonical_underlying_case_id": canonical,
            "row_index": row_index,
            "request_content_sha256": request_content_sha256(normalized),
        }
        if canonical in PROTECTED_HELDOUT_CASE_IDS:
            excluded_records.append({**record, "exclusion_reason": EXCLUSION_REASON})
        else:
            normalized_rows.append(normalized)
            retained_records.append(record)
    if (
        len(excluded_records) != EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT
        or excluded_records[0]["canonical_underlying_case_id"]
        != CPAC_CANONICAL_CASE_ID
        or len(retained_records) != EXPECTED_STAGE_A_REQUEST_COUNT
    ):
        raise ReferenceInputError(
            "historical Phase4A source does not produce the approved 8-to-7 projection"
        )
    return normalized_rows, excluded_records, retained_records


def load_phase4a_replay_requests(
    path: Path,
    *,
    expected_sha256: str,
    manifest_path: Path,
    manifest_expected_sha256: str,
) -> Tuple[str, str, Path, Tuple[EvaluationRequest, ...]]:
    resolved = _resolve_input(path, "authoritative Phase4A replay artifact")
    expected = _expected_sha(
        expected_sha256, "authoritative Phase4A replay artifact SHA-256"
    )
    actual_sha = sha256_file(resolved)
    if actual_sha != expected:
        raise ReferenceInputError("authoritative Phase4A replay artifact SHA-256 mismatch")
    if resolved.suffix.casefold() != ".jsonl":
        raise ReferenceInputError("normalized Phase4A replay artifact must be JSONL")
    try:
        rows = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        payload: Any = rows
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceInputError("Phase4A replay artifact is malformed") from exc
    if _contains_forbidden_label_key(payload):
        raise ReferenceInputError("Phase4A replay artifact contains forbidden labels or outputs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_STAGE_A_REQUEST_COUNT:
        raise ReferenceInputError(
            "normalized Phase4A replay artifact must contain exactly 7 requests"
        )

    manifest_resolved, manifest = _read_verified_json(
        manifest_path,
        manifest_expected_sha256,
        "Phase4A normalization manifest",
    )
    if not isinstance(manifest, Mapping):
        raise ReferenceInputError("Phase4A normalization manifest must be an object")
    required_manifest = {
        "status": "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS",
        "implementation_revision": NORMALIZATION_IMPLEMENTATION_REVISION,
        "artifact_type": "phase4a_invariance_request_normalization_manifest",
        "source_artifact_sha256": AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
        "normalized_artifact_sha256": actual_sha,
        "source_request_count": EXPECTED_HISTORICAL_PHASE4A_COUNT,
        "historical_labels_included": False,
        "historical_validation_rows_loaded": 0,
        "historical_test_rows_loaded": 0,
        "protected_heldout_case_ids": list(PROTECTED_HELDOUT_CASE_IDS),
        "overlap_count": EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT,
        "excluded_request_count": EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT,
        "retained_request_count": EXPECTED_STAGE_A_REQUEST_COUNT,
        "source_schema_fields": list(SOURCE_SCHEMA_FIELDS),
        "normalized_schema_fields": list(NORMALIZED_SCHEMA_FIELDS),
        "source_case_id_removed": True,
        "claims_changed_count": 0,
        "candidate_content_changed_count": 0,
        "candidate_id_changed_count": 0,
        "candidate_order_changed_count": 0,
        "unit_type_changed_count": 0,
        "modality_changed_count": 0,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "selector_loaded": False,
        "training_started": False,
        "optimizer_created": False,
        "production_or_model_code_changed": False,
    }
    for field, expected_value in required_manifest.items():
        if manifest.get(field) != expected_value:
            raise ReferenceInputError(
                f"Phase4A normalization manifest mismatch: {field}"
            )
    if set(manifest) != set(required_manifest) | {
        "source_artifact_path",
        "excluded_requests",
        "retained_requests",
    }:
        raise ReferenceInputError("Phase4A normalization manifest schema is invalid")
    for field in (
        "source_request_count",
        "historical_validation_rows_loaded",
        "historical_test_rows_loaded",
        "overlap_count",
        "excluded_request_count",
        "retained_request_count",
        "claims_changed_count",
        "candidate_content_changed_count",
        "candidate_id_changed_count",
        "candidate_order_changed_count",
        "unit_type_changed_count",
        "modality_changed_count",
    ):
        if type(manifest.get(field)) is not int:
            raise ReferenceInputError(
                f"Phase4A normalization manifest count is invalid: {field}"
            )
    source_path = _resolve_input(
        Path(_nonblank(manifest.get("source_artifact_path"), "source_artifact_path")),
        "historical Phase4A source artifact",
    )
    if sha256_file(source_path) != AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256:
        raise ReferenceInputError("historical Phase4A source artifact SHA mismatch")
    expected_rows, expected_excluded, expected_retained = (
        _derive_expected_normalized_protocol(source_path)
    )
    if rows != expected_rows:
        raise ReferenceInputError(
            "normalized Phase4A requests do not match the historical source projection"
        )
    excluded = manifest.get("excluded_requests")
    retained = manifest.get("retained_requests")
    if not isinstance(excluded, list) or len(excluded) != 1:
        raise ReferenceInputError("normalization manifest excluded-request proof is invalid")
    if not isinstance(retained, list) or len(retained) != EXPECTED_STAGE_A_REQUEST_COUNT:
        raise ReferenceInputError("normalization manifest retained-request proof is invalid")
    excluded_record = excluded[0]
    required_record_fields = {
        "historical_case_id",
        "source_case_id",
        "canonical_underlying_case_id",
        "row_index",
        "request_content_sha256",
    }
    if (
        not isinstance(excluded_record, Mapping)
        or set(excluded_record) != required_record_fields | {"exclusion_reason"}
        or excluded_record.get("canonical_underlying_case_id")
        != CPAC_CANONICAL_CASE_ID
        or excluded_record.get("exclusion_reason") != EXCLUSION_REASON
    ):
        raise ReferenceInputError("normalization manifest CPAC exclusion is invalid")
    if type(excluded_record.get("row_index")) is not int:
        raise ReferenceInputError("normalization manifest CPAC row index is invalid")

    requests = []
    retained_by_index = {}
    for record in retained:
        if not isinstance(record, Mapping) or set(record) != required_record_fields:
            raise ReferenceInputError("normalization manifest retained record is invalid")
        row_index = record.get("row_index")
        if type(row_index) is not int or not 0 <= row_index < EXPECTED_HISTORICAL_PHASE4A_COUNT:
            raise ReferenceInputError("normalization manifest retained row index is invalid")
        canonical = record.get("canonical_underlying_case_id")
        if canonical in PROTECTED_HELDOUT_CASE_IDS:
            raise ReferenceInputError("Stage A retained a protected held-out case")
        if row_index in retained_by_index:
            raise ReferenceInputError("normalization manifest duplicates a row index")
        retained_by_index[row_index] = record
    excluded_index = excluded_record["row_index"]
    if set(retained_by_index) | {excluded_index} != set(
        range(EXPECTED_HISTORICAL_PHASE4A_COUNT)
    ):
        raise ReferenceInputError("normalization manifest row accounting is incomplete")
    ordered_retained = tuple(retained_by_index[index] for index in sorted(retained_by_index))
    if tuple(retained) != ordered_retained:
        raise ReferenceInputError("normalization manifest retained order is invalid")
    if excluded != expected_excluded or retained != expected_retained:
        raise ReferenceInputError(
            "normalization manifest does not match the historical source projection"
        )

    native_allowed = set(NORMALIZED_SCHEMA_FIELDS)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != native_allowed:
            raise ReferenceInputError(f"Phase4A replay request {index} has invalid schema")
        case_id = _nonblank(row.get("case_id"), f"requests[{index}].case_id")
        record = ordered_retained[index]
        if record.get("historical_case_id") != case_id:
            raise ReferenceInputError("normalization manifest historical case mismatch")
        if record.get("request_content_sha256") != request_content_sha256(row):
            raise ReferenceInputError("normalized request content is not manifest-approved")
        if (
            case_id in EXPECTED_HELDOUT_CASE_IDS
            or record.get("canonical_underlying_case_id") in EXPECTED_HELDOUT_CASE_IDS
        ):
            raise ReferenceInputError("Stage A attempted to access a held-out relevance case")
        try:
            requests.append(
                EvaluationRequest(
                    request_id=case_id,
                    case_id=case_id,
                    dataset=_dataset(row.get("dataset"), f"requests[{index}].dataset"),
                    claim=_nonblank(row.get("claim"), f"requests[{index}].claim"),
                    candidate_units=_parse_units(
                        row.get("candidate_units"), f"requests[{index}].candidate_units"
                    ),
                )
            )
        except ValueError as exc:
            raise ReferenceInputError(str(exc)) from exc
    ids = tuple(item.request_id for item in requests)
    if len(set(ids)) != len(ids):
        raise ReferenceInputError("Phase4A replay request IDs must be unique")
    return actual_sha, sha256_file(manifest_resolved), source_path, tuple(requests)


def _verify_source_artifact(path_value: Any, sha_value: Any, field: str) -> Tuple[str, str]:
    path = _resolve_input(Path(_nonblank(path_value, f"{field}.path")), field)
    expected = _expected_sha(sha_value, f"{field}.sha256")
    if sha256_file(path) != expected:
        raise ReferenceInputError(f"{field} SHA-256 mismatch")
    return str(path), expected


def load_heldout_references(
    path: Path,
    *,
    expected_sha256: str,
) -> Tuple[str, Tuple[EvaluationRequest, ...]]:
    resolved, payload = _read_verified_json(
        path, expected_sha256, "held-out relevance reference artifact"
    )
    if not isinstance(payload, Mapping):
        raise ReferenceInputError("held-out reference artifact must be an object")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != (
        "preexisting_heldout_relevance_challenge_references"
    ):
        raise ReferenceInputError("held-out reference artifact contract is incompatible")
    if _contains_forbidden_label_key(payload):
        raise ReferenceInputError("held-out reference artifact contains forbidden labels or outputs")
    rows = payload.get("references")
    if not isinstance(rows, list) or not rows:
        raise ReferenceInputError("held-out reference artifact contains no references")
    references = []
    allowed = {
        "reference_id",
        "case_id",
        "dataset",
        "claim",
        "candidate_units",
        "positive_unit_ids",
        "reference_modality",
        "source_audit_artifact_path",
        "source_audit_artifact_sha256",
        "prior_original_best_positive_rank",
        "prior_original_top5_unit_ids",
        "prior_candidate_unit_ids",
    }
    for index, row in enumerate(rows):
        field = f"references[{index}]"
        if not isinstance(row, Mapping) or set(row) != allowed:
            raise ReferenceInputError(f"{field} has an invalid schema")
        claim = _nonblank(row.get("claim"), f"{field}.claim")
        if claim.casefold().startswith('the relevant content states "'):
            raise ReferenceInputError(
                "neutral synthetic claims are forbidden in held-out evaluation"
            )
        case_id = _nonblank(row.get("case_id"), f"{field}.case_id")
        if case_id not in EXPECTED_HELDOUT_CASE_IDS:
            raise ReferenceInputError("held-out case identity mismatch")
        source_path, source_sha = _verify_source_artifact(
            row.get("source_audit_artifact_path"),
            row.get("source_audit_artifact_sha256"),
            f"{field}.source_audit_artifact",
        )
        units = _parse_units(row.get("candidate_units"), f"{field}.candidate_units")
        positive_ids = _string_tuple(
            row.get("positive_unit_ids"), f"{field}.positive_unit_ids"
        )
        prior_candidate_ids = _string_tuple(
            row.get("prior_candidate_unit_ids"), f"{field}.prior_candidate_unit_ids"
        )
        if prior_candidate_ids != tuple(unit.unit_id for unit in units):
            raise ReferenceInputError("prior candidate IDs/order differ from reference pool")
        prior_rank = row.get("prior_original_best_positive_rank")
        if type(prior_rank) is not int or not 1 <= prior_rank <= len(units):
            raise ReferenceInputError("prior original best-positive rank is invalid")
        prior_top5 = _string_tuple(
            row.get("prior_original_top5_unit_ids"),
            f"{field}.prior_original_top5_unit_ids",
        )
        if len(prior_top5) > 5 or not set(prior_top5) <= set(prior_candidate_ids):
            raise ReferenceInputError("prior original Top-5 is invalid")
        try:
            references.append(
                EvaluationRequest(
                    request_id=_nonblank(
                        row.get("reference_id"), f"{field}.reference_id"
                    ),
                    reference_id=_nonblank(
                        row.get("reference_id"), f"{field}.reference_id"
                    ),
                    case_id=case_id,
                    dataset=_dataset(row.get("dataset"), f"{field}.dataset"),
                    claim=claim,
                    candidate_units=units,
                    positive_unit_ids=positive_ids,
                    reference_modality=_nonblank(
                        row.get("reference_modality"), f"{field}.reference_modality"
                    ).upper(),
                    source_audit_artifact_path=source_path,
                    source_audit_artifact_sha256=source_sha,
                    prior_original_best_positive_rank=prior_rank,
                    prior_original_top5_unit_ids=prior_top5,
                    prior_candidate_unit_ids=prior_candidate_ids,
                )
            )
        except ValueError as exc:
            raise ReferenceInputError(str(exc)) from exc
    case_ids = {item.case_id for item in references}
    if case_ids != set(EXPECTED_HELDOUT_CASE_IDS):
        raise ReferenceInputError("held-out reference artifact must cover exactly six cases")
    reference_ids = tuple(item.reference_id for item in references)
    if len(set(reference_ids)) != len(reference_ids):
        raise ReferenceInputError("held-out reference IDs must be unique")
    if not any(
        item.case_id == EXPECTED_HELDOUT_CASE_IDS[0]
        and item.reference_id == CPAC_REFERENCE_ID
        for item in references
    ):
        raise ReferenceInputError("designated CPAC direct-OCR reference is missing")
    return sha256_file(resolved), tuple(references)


def calibration_overlap_count(
    heldout_case_ids: Iterable[str], calibration_case_ids: Iterable[str]
) -> int:
    return len(set(heldout_case_ids) & set(calibration_case_ids))
