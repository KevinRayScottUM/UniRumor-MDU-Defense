"""Result-blind normalization of the historical Phase4A smoke requests."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


IMPLEMENTATION_REVISION = "step2.6r-3a0-v1"
AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256 = (
    "356ee750c7b95de37e5d14b481e2f5f8fb5ae1e3805ee922d016fcb0a3ab2178"
)
EXPECTED_HISTORICAL_PHASE4A_COUNT = 8
EXPECTED_STAGE_A_REQUEST_COUNT = 7
EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT = 1
EXCLUSION_REASON = "PREEXISTING_HELDOUT_RELEVANCE_CHALLENGE"
CPAC_CANONICAL_CASE_ID = "GroundLie360:13025004"
EXPECTED_HISTORICAL_CASE_IDS = (
    CPAC_CANONICAL_CASE_ID,
    "GroundLie360:13199900",
    "GroundLie360:13296704",
    "GroundLie360:13310803",
    "GroundLie360:13359007",
    "GroundLie360:13364604",
    "GroundLie360:13443602",
    "GroundLie360:13494602",
)
PROTECTED_HELDOUT_CASE_IDS = (
    CPAC_CANONICAL_CASE_ID,
    "TRUE-3MFact:10145403",
    "TRUE-3MFact:10258205",
    "TRUE-3MFact:10372904",
    "TRUE-3MFact:10455808",
    "TRUE-3MFact:10865013",
)
SOURCE_SCHEMA_FIELDS = (
    "case_id",
    "dataset",
    "claim",
    "candidate_units",
    "source_case_id",
)
NORMALIZED_SCHEMA_FIELDS = (
    "case_id",
    "dataset",
    "claim",
    "candidate_units",
)
_UNIT_FIELDS = ("unit_id", "unit_type", "modality", "text")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_FIELDS = frozenset(
    {
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
        "selection_scores",
    }
)
_RESTRICTED_PATH_PARTS = frozenset(
    {"validation", "test", "formalvalidation", "formaltest"}
)
_ALLOWED_CANONICAL_DATASETS = frozenset({"GroundLie360", "TRUE-3MFact"})


class Phase4ANormalizationError(ValueError):
    """Raised when the historical-to-normalized protocol cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in _RESTRICTED_PATH_PARTS for part in resolved.parts):
        raise Phase4ANormalizationError(
            f"{field} cannot access Formal Validation/Test"
        )
    if not resolved.is_file():
        raise Phase4ANormalizationError(f"{field} is missing")
    return resolved


def _resolve_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in _RESTRICTED_PATH_PARTS for part in resolved.parts):
        raise Phase4ANormalizationError(
            "normalization output cannot be inside Formal Validation/Test"
        )
    if resolved.exists():
        raise Phase4ANormalizationError("normalization output directory must not exist")
    return resolved


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase4ANormalizationError(f"{field} must be nonblank")
    return value


def _contains_forbidden_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_FIELDS:
                return True
            if _contains_forbidden_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def canonical_underlying_case_id(
    historical_case_id: str, source_case_id: str
) -> str:
    """Resolve only the two explicitly supported historical identity forms."""

    historical = _nonblank(historical_case_id, "historical case_id")
    source = _nonblank(source_case_id, "source_case_id")
    if any(
        token.casefold() in _RESTRICTED_PATH_PARTS
        for token in (*historical.split(":"), *source.split(":"))
    ):
        raise Phase4ANormalizationError("historical identity references Validation/Test")
    value = source
    if value.startswith("smoke::"):
        value = value[len("smoke::") :]
    parts = value.split(":")
    if len(parts) == 3 and parts[1] == "train":
        dataset, _, numeric_id = parts
    elif len(parts) == 2:
        dataset, numeric_id = parts
    else:
        raise Phase4ANormalizationError("source_case_id has an unsupported identity form")
    if dataset not in _ALLOWED_CANONICAL_DATASETS or not numeric_id.isdigit():
        raise Phase4ANormalizationError("source_case_id cannot be canonicalized narrowly")
    canonical = f"{dataset}:{numeric_id}"
    historical_suffix = historical[len("smoke::") :] if historical.startswith("smoke::") else historical
    if historical_suffix not in {source, canonical}:
        raise Phase4ANormalizationError("case_id and source_case_id provenance disagree")
    return canonical


def _validated_candidates(value: Any, row_index: int) -> list[Mapping[str, str]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 24:
        raise Phase4ANormalizationError(
            f"row {row_index} candidate_units must contain between 1 and 24 units"
        )
    candidates = []
    unit_ids = []
    for unit_index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != set(_UNIT_FIELDS):
            raise Phase4ANormalizationError(
                f"row {row_index} candidate {unit_index} has an invalid schema"
            )
        candidate = {
            field: _nonblank(
                item.get(field), f"row {row_index} candidate {unit_index} {field}"
            )
            for field in _UNIT_FIELDS
        }
        if candidate["unit_type"] not in {"text", "transcript", "ocr"}:
            raise Phase4ANormalizationError("historical candidate unit_type is invalid")
        expected_modality = "ocr" if candidate["unit_type"] == "ocr" else "text"
        if candidate["modality"] != expected_modality:
            raise Phase4ANormalizationError(
                "historical candidate unit_type/modality are inconsistent"
            )
        candidates.append(candidate)
        unit_ids.append(candidate["unit_id"])
    if len(set(unit_ids)) != len(unit_ids):
        raise Phase4ANormalizationError("historical candidate IDs are not unique")
    return candidates


def request_content_sha256(row: Mapping[str, Any]) -> str:
    content = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _write_with_sidecar(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    _sidecar_path(path).write_text(digest + "\n", encoding="utf-8")
    return digest


def _load_historical_rows(path: Path) -> Tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = tuple(json.loads(line) for line in lines if line.strip())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4ANormalizationError(
            "historical Phase4A request artifact is malformed JSONL"
        ) from exc
    if len(rows) != EXPECTED_HISTORICAL_PHASE4A_COUNT:
        raise Phase4ANormalizationError(
            "historical Phase4A request artifact must contain exactly 8 rows"
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise Phase4ANormalizationError("historical Phase4A rows must be objects")
    if _contains_forbidden_field(rows):
        raise Phase4ANormalizationError(
            "historical Phase4A request artifact contains forbidden labels or outputs"
        )
    return rows


def prepare_invariance_requests(
    *,
    source_artifact: Path,
    source_sha256: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Project the immutable historical eight requests to the approved seven."""

    if (
        not isinstance(source_sha256, str)
        or not _SHA256_RE.fullmatch(source_sha256.casefold())
        or source_sha256.casefold() != AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256
    ):
        raise Phase4ANormalizationError(
            "historical Phase4A SHA-256 is not the authoritative value"
        )
    source = _resolve_file(source_artifact, "historical Phase4A request artifact")
    source_sha_before = sha256_file(source)
    if source_sha_before != AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256:
        raise Phase4ANormalizationError("historical Phase4A source SHA-256 mismatch")
    rows = _load_historical_rows(source)

    normalized_rows = []
    excluded_records = []
    retained_records = []
    canonical_ids = []
    change_counts = {
        "claims_changed_count": 0,
        "candidate_content_changed_count": 0,
        "candidate_id_changed_count": 0,
        "candidate_order_changed_count": 0,
        "unit_type_changed_count": 0,
        "modality_changed_count": 0,
    }
    for row_index, row in enumerate(rows):
        if set(row) != set(SOURCE_SCHEMA_FIELDS):
            raise Phase4ANormalizationError(
                f"historical row {row_index} does not match the exact source schema"
            )
        historical_case_id = _nonblank(row.get("case_id"), f"row {row_index} case_id")
        source_case_id = _nonblank(
            row.get("source_case_id"), f"row {row_index} source_case_id"
        )
        dataset = _nonblank(row.get("dataset"), f"row {row_index} dataset")
        if dataset.casefold() in _RESTRICTED_PATH_PARTS:
            raise Phase4ANormalizationError("historical dataset is Validation/Test")
        claim = _nonblank(row.get("claim"), f"row {row_index} claim")
        candidates = _validated_candidates(row.get("candidate_units"), row_index)
        canonical = canonical_underlying_case_id(historical_case_id, source_case_id)
        if canonical.rsplit(":", 1)[0] != dataset:
            raise Phase4ANormalizationError(
                "historical dataset and source_case_id provenance disagree"
            )
        canonical_ids.append(canonical)
        normalized = {
            "case_id": historical_case_id,
            "dataset": dataset,
            "claim": claim,
            "candidate_units": candidates,
        }
        source_candidates = row["candidate_units"]
        change_counts["claims_changed_count"] += int(
            claim.encode("utf-8") != normalized["claim"].encode("utf-8")
        )
        change_counts["candidate_content_changed_count"] += int(
            tuple(item["text"] for item in source_candidates)
            != tuple(item["text"] for item in candidates)
        )
        source_ids = tuple(item["unit_id"] for item in source_candidates)
        normalized_ids = tuple(item["unit_id"] for item in candidates)
        change_counts["candidate_id_changed_count"] += int(
            set(source_ids) != set(normalized_ids)
        )
        change_counts["candidate_order_changed_count"] += int(
            source_ids != normalized_ids
        )
        change_counts["unit_type_changed_count"] += int(
            tuple(item["unit_type"] for item in source_candidates)
            != tuple(item["unit_type"] for item in candidates)
        )
        change_counts["modality_changed_count"] += int(
            tuple(item["modality"] for item in source_candidates)
            != tuple(item["modality"] for item in candidates)
        )
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

    true_heldout = set(PROTECTED_HELDOUT_CASE_IDS[1:]) & set(canonical_ids)
    if true_heldout:
        raise Phase4ANormalizationError(
            "historical source contains a TRUE-3MFact held-out relevance case"
        )
    if CPAC_CANONICAL_CASE_ID not in canonical_ids:
        raise Phase4ANormalizationError("historical source is missing the CPAC case")
    if len(excluded_records) != EXPECTED_STAGE_A_EXCLUDED_HELDOUT_COUNT:
        raise Phase4ANormalizationError(
            "historical source must contain exactly one protected overlap"
        )
    if tuple(sorted(canonical_ids)) != tuple(sorted(EXPECTED_HISTORICAL_CASE_IDS)):
        raise Phase4ANormalizationError(
            "historical Phase4A underlying case set is not authoritative"
        )
    if len(normalized_rows) != EXPECTED_STAGE_A_REQUEST_COUNT:
        raise Phase4ANormalizationError(
            "normalized Stage-A request set must contain exactly 7 rows"
        )
    if any(change_counts.values()):
        raise Phase4ANormalizationError("field projection changed scientific content")

    output = _resolve_output(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase4a-normalization-", dir=output.parent))
    try:
        normalized_path = staging / "phase4a_invariance_requests.jsonl"
        normalized_sha = _write_with_sidecar(
            normalized_path, _jsonl_bytes(normalized_rows)
        )
        manifest = {
            "status": "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "artifact_type": "phase4a_invariance_request_normalization_manifest",
            "source_artifact_path": str(source),
            "source_artifact_sha256": source_sha_before,
            "normalized_artifact_sha256": normalized_sha,
            "source_request_count": len(rows),
            "historical_labels_included": False,
            "historical_validation_rows_loaded": 0,
            "historical_test_rows_loaded": 0,
            "protected_heldout_case_ids": list(PROTECTED_HELDOUT_CASE_IDS),
            "overlap_count": len(excluded_records),
            "excluded_request_count": len(excluded_records),
            "excluded_requests": excluded_records,
            "retained_request_count": len(retained_records),
            "retained_requests": retained_records,
            "source_schema_fields": list(SOURCE_SCHEMA_FIELDS),
            "normalized_schema_fields": list(NORMALIZED_SCHEMA_FIELDS),
            "source_case_id_removed": True,
            **change_counts,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "selector_loaded": False,
            "training_started": False,
            "optimizer_created": False,
            "production_or_model_code_changed": False,
        }
        _write_with_sidecar(
            staging / "phase4a_invariance_request_manifest.json",
            _json_bytes(manifest),
        )
        if sha256_file(source) != source_sha_before:
            raise Phase4ANormalizationError(
                "historical Phase4A source changed during normalization"
            )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
