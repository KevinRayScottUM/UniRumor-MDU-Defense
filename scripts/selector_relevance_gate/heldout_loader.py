"""Fail-closed loaders for label-free replay and held-out audit references."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .schemas import EvaluationRequest, EvaluationUnit


EXPECTED_HELDOUT_CASE_IDS = (
    "GroundLie360:13025004",
    "TRUE-3MFact:10145403",
    "TRUE-3MFact:10258205",
    "TRUE-3MFact:10372904",
    "TRUE-3MFact:10455808",
    "TRUE-3MFact:10865013",
)
EXPECTED_PHASE4A_REPLAY_COUNT = 8
CPAC_REFERENCE_ID = "ocr_01_direct_full_banner"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "label",
    "labels",
    "ground_truth",
    "ground_truth_label",
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
                    ).casefold(),
                    modality=_nonblank(
                        item.get("modality"), f"{field}[{index}].modality"
                    ).casefold(),
                    text=_nonblank(item.get("text"), f"{field}[{index}].text"),
                )
            )
        except ValueError as exc:
            raise ReferenceInputError(str(exc)) from exc
    return tuple(units)


def load_phase4a_replay_requests(
    path: Path,
    *,
    expected_sha256: str,
) -> Tuple[str, Tuple[EvaluationRequest, ...]]:
    resolved = _resolve_input(path, "authoritative Phase4A replay artifact")
    expected = _expected_sha(
        expected_sha256, "authoritative Phase4A replay artifact SHA-256"
    )
    actual_sha = sha256_file(resolved)
    if actual_sha != expected:
        raise ReferenceInputError("authoritative Phase4A replay artifact SHA-256 mismatch")
    try:
        if resolved.suffix.casefold() == ".jsonl":
            rows = [
                json.loads(line)
                for line in resolved.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            payload: Any = rows
        else:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ReferenceInputError("Phase4A replay artifact must be an object")
            if payload.get("schema_version") != 1 or payload.get("artifact_type") != (
                "phase4a_label_free_replay_requests"
            ):
                raise ReferenceInputError("Phase4A replay artifact contract is incompatible")
            rows = payload.get("requests")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceInputError("Phase4A replay artifact is malformed") from exc
    if _contains_forbidden_label_key(payload):
        raise ReferenceInputError("Phase4A replay artifact contains forbidden labels or outputs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_PHASE4A_REPLAY_COUNT:
        raise ReferenceInputError("Phase4A replay artifact must contain exactly 8 requests")
    requests = []
    wrapper_allowed = {"request_id", "case_id", "dataset", "claim", "candidate_units"}
    native_allowed = {"case_id", "dataset", "claim", "candidate_units"}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or frozenset(row) not in {
            frozenset(wrapper_allowed),
            frozenset(native_allowed),
        }:
            raise ReferenceInputError(f"Phase4A replay request {index} has invalid schema")
        case_id = _nonblank(row.get("case_id"), f"requests[{index}].case_id")
        if case_id in EXPECTED_HELDOUT_CASE_IDS:
            raise ReferenceInputError("Stage A attempted to access a held-out relevance case")
        try:
            requests.append(
                EvaluationRequest(
                    request_id=(
                        _nonblank(
                            row.get("request_id"), f"requests[{index}].request_id"
                        )
                        if "request_id" in row
                        else case_id
                    ),
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
    return actual_sha, tuple(requests)


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
