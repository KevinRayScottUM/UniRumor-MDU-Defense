"""Fail-closed public review loading and delayed private mapping alignment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .schemas import (
    CONFIDENCE_LABELS,
    EXPECTED_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    REVIEW_COLUMNS,
    RELEVANCE_LABELS,
    SEALED_CHALLENGE_IDS,
    SOURCE_IMPLEMENTATION_REVISION,
    FrozenReviewRow,
    ReviewFreezeError,
    UnderlyingUnit,
)


_RESTRICTED_PATH_PARTS = {
    "validation",
    "test",
    "formalvalidation",
    "formaltest",
}
_PUBLIC_ROOT_ARTIFACTS = (
    "build_report.json",
    "cohort_source_lock.json",
    "eligibility_inventory.json",
    "selected_case_manifest.json",
    "independent_relevance_audit_requests.jsonl",
    "independent_audit_preregistration.json",
)
_PRIVATE_MAPPING = "private_review_mapping.json"
_PUBLIC_PACKET_FILES = (
    "README_REVIEWER.md",
    "REVIEW_MANIFEST.json",
    "relevance_review_template.csv",
)
_IMMUTABLE_REVIEW_FIELDS = (
    "review_case_id",
    "claim",
    "review_unit_id",
    "candidate_text",
)
_REVIEW_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "reviewer",
        "review_type",
        "completed_csv_sha256",
        "row_count",
        "case_count",
        "web_search_used",
        "external_sources_used",
        "other_reviewer_output_accessed",
        "dataset_identity_accessed",
        "modality_identity_accessed",
        "selector_outputs_accessed",
        "veracity_labels_accessed",
        "formal_validation_accessed",
        "formal_test_accessed",
        "completed_all_rows",
    }
)


@dataclass(frozen=True)
class PublicCohort:
    directory: Path
    public_source_locks: Mapping[str, Mapping[str, str]]
    build_report: Mapping[str, Any]
    selected_manifest: Mapping[str, Any]
    selected_units: Mapping[Tuple[str, str], UnderlyingUnit]
    selected_case_ids: frozenset[str]
    templates: Mapping[str, Tuple[Mapping[str, str], ...]]
    notes_required: Mapping[str, bool]


@dataclass(frozen=True)
class ValidatedReview:
    reviewer: str
    completed_path: Path
    completed_sha256: str
    completed_bytes: bytes
    provenance_path: Path
    provenance_sha256: str
    provenance_bytes: bytes
    provenance: Mapping[str, Any]
    rows: Tuple[FrozenReviewRow, ...]


def safe_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(
        re.sub(r"[^a-z0-9]", "", part.casefold()) in _RESTRICTED_PATH_PARTS
        for part in resolved.parts
    ):
        raise ReviewFreezeError(f"{field} must not reference Formal Validation/Test")
    return resolved


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        for value in values
    )


def csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_artifact(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    digest = sha256_bytes(payload)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewFreezeError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ReviewFreezeError(f"{field} must be a JSON object")
    return payload


def _read_csv(path: Path, field: str) -> Tuple[Tuple[str, ...], Tuple[Mapping[str, str], ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewFreezeError(f"{field} is not valid UTF-8 CSV") from exc
    return columns, rows


def _verify_sidecar(path: Path, field: str) -> Mapping[str, str]:
    if not path.is_file():
        raise ReviewFreezeError(f"{field} is missing")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ReviewFreezeError(f"{field} SHA sidecar is missing")
    expected = sidecar.read_text(encoding="utf-8").strip().casefold()
    actual = sha256_file(path)
    if len(expected) != 64 or expected != actual:
        raise ReviewFreezeError(f"{field} SHA-256 mismatch")
    return {"path": str(path), "sha256": actual}


def _validate_public_packet(
    directory: Path, reviewer: str
) -> Tuple[
    Tuple[Mapping[str, str], ...],
    bool,
    Mapping[str, Mapping[str, str]],
]:
    if not directory.is_dir():
        raise ReviewFreezeError(f"Reviewer {reviewer} packet is missing")
    if tuple(sorted(path.name for path in directory.iterdir())) != tuple(
        sorted(_PUBLIC_PACKET_FILES)
    ):
        raise ReviewFreezeError(f"Reviewer {reviewer} public packet files changed")
    manifest = _read_json(
        directory / "REVIEW_MANIFEST.json", f"Reviewer {reviewer} manifest"
    )
    required = {
        "status": "READY_FOR_INDEPENDENT_REVIEW",
        "implementation_revision": SOURCE_IMPLEMENTATION_REVISION,
        "reviewer": reviewer,
        "case_count": EXPECTED_CASE_COUNT,
        "row_count": EXPECTED_UNIT_COUNT,
        "public_columns": list(REVIEW_COLUMNS),
        "allowed_direct_relevance_labels": list(RELEVANCE_LABELS),
        "allowed_review_confidence": list(CONFIDENCE_LABELS),
        "review_fields_initially_blank": True,
        "dataset_blind": True,
        "modality_blind": True,
        "selector_blind": True,
        "veracity_label_blind": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ReviewFreezeError(
                f"Reviewer {reviewer} manifest contract failed: {field}"
            )
    readme = (directory / "README_REVIEWER.md").read_bytes()
    template_path = directory / "relevance_review_template.csv"
    template = template_path.read_bytes()
    if manifest.get("instructions_sha256") != sha256_bytes(readme):
        raise ReviewFreezeError(f"Reviewer {reviewer} instructions SHA mismatch")
    if manifest.get("template_sha256") != sha256_bytes(template):
        raise ReviewFreezeError(f"Reviewer {reviewer} template SHA mismatch")
    columns, rows = _read_csv(template_path, f"Reviewer {reviewer} template")
    if columns != REVIEW_COLUMNS or len(rows) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError(f"Reviewer {reviewer} template schema changed")
    if len({row["review_case_id"] for row in rows}) != EXPECTED_CASE_COUNT:
        raise ReviewFreezeError(f"Reviewer {reviewer} template case count changed")
    seen = set()
    for row in rows:
        key = (row["review_case_id"], row["review_unit_id"])
        if key in seen:
            raise ReviewFreezeError(f"Reviewer {reviewer} template contains duplicate IDs")
        seen.add(key)
        if any(row[field] for field in (
            "direct_relevance_label", "review_confidence", "review_note"
        )):
            raise ReviewFreezeError(f"Reviewer {reviewer} template is not blank")
    notes_required = b"review_note` for every row" in readme
    locks = {
        name: {
            "path": str(directory / name),
            "sha256": sha256_file(directory / name),
        }
        for name in _PUBLIC_PACKET_FILES
    }
    return rows, notes_required, locks


def load_public_cohort(cohort_dir: Path) -> PublicCohort:
    """Validate public 3B1-R2 inputs without opening the private mapping."""

    directory = safe_path(cohort_dir, "3B1-R2 cohort directory")
    if not directory.is_dir():
        raise ReviewFreezeError("3B1-R2 cohort directory is missing")
    locks: Dict[str, Mapping[str, str]] = {}
    for name in _PUBLIC_ROOT_ARTIFACTS:
        locks[name] = _verify_sidecar(directory / name, f"3B1-R2 {name}")

    report = _read_json(directory / "build_report.json", "3B1-R2 build report")
    required_report = {
        "status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS",
        "implementation_revision": SOURCE_IMPLEMENTATION_REVISION,
        "selected_total_count": EXPECTED_CASE_COUNT,
        "selected_groundlie_count": 15,
        "selected_true3m_count": 15,
        "selected_candidate_unit_count": EXPECTED_UNIT_COUNT,
        "reviewer_a_row_count": EXPECTED_UNIT_COUNT,
        "reviewer_b_row_count": EXPECTED_UNIT_COUNT,
        "selection_scores_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "training_started": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_heldout_reference_content_accessed": False,
    }
    for field, expected in required_report.items():
        if report.get(field) != expected:
            raise ReviewFreezeError(f"3B1-R2 build contract failed: {field}")

    selected = _read_json(
        directory / "selected_case_manifest.json", "selected case manifest"
    )
    if selected.get("status") != "FROZEN" or selected.get(
        "implementation_revision"
    ) != SOURCE_IMPLEMENTATION_REVISION or selected.get(
        "sampling_salt"
    ) != "step2.6r-3b1-independent-audit-v1":
        raise ReviewFreezeError("selected case manifest contract changed")
    cases = selected.get("selected_cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ReviewFreezeError("selected case count is not 30")
    selected_units: Dict[Tuple[str, str], UnderlyingUnit] = {}
    selected_case_ids = set()
    selected_dataset_counts: Counter[str] = Counter()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ReviewFreezeError("selected case manifest row is invalid")
        dataset = case.get("dataset")
        canonical = case.get("canonical_case_id")
        original = case.get("original_case_id")
        unit_ids = case.get("candidate_unit_ids_in_original_order")
        unit_types = case.get("candidate_unit_types_in_original_order")
        modalities = case.get("candidate_modalities_in_original_order")
        count = case.get("model_exposed_unit_count")
        if (
            not all(isinstance(value, str) and value for value in (dataset, canonical, original))
            or canonical in selected_case_ids
            or canonical in SEALED_CHALLENGE_IDS
            or not isinstance(unit_ids, list)
            or not isinstance(unit_types, list)
            or not isinstance(modalities, list)
            or not (len(unit_ids) == len(unit_types) == len(modalities) == count)
        ):
            raise ReviewFreezeError("selected case manifest identity/accounting changed")
        selected_case_ids.add(canonical)
        selected_dataset_counts[dataset] += 1
        for position, (unit_id, unit_type, modality) in enumerate(
            zip(unit_ids, unit_types, modalities)
        ):
            if not all(
                isinstance(value, str) and value
                for value in (unit_id, unit_type, modality)
            ):
                raise ReviewFreezeError("selected unit metadata is invalid")
            unit = UnderlyingUnit(
                dataset=dataset,
                canonical_case_id=canonical,
                original_case_id=original,
                unit_id=unit_id,
                unit_type=unit_type,
                modality=modality,
                original_candidate_position=position,
            )
            if unit.key in selected_units:
                raise ReviewFreezeError("selected underlying unit is duplicated")
            selected_units[unit.key] = unit
    if len(selected_units) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError("selected underlying unit count is not 289")
    if selected_dataset_counts != Counter(
        {"GroundLie360": 15, "TRUE-3MFact": 15}
    ):
        raise ReviewFreezeError("selected dataset case counts changed")

    preregistration = _read_json(
        directory / "independent_audit_preregistration.json",
        "independent audit preregistration",
    )
    future = preregistration.get("future_step_2_6r_3b3")
    gate = future.get("coverage_gate") if isinstance(future, Mapping) else None
    if not isinstance(gate, Mapping) or {
        "minimum_evaluable_case_count": gate.get("minimum_evaluable_case_count"),
        "frozen_total_case_count": gate.get("frozen_total_case_count"),
        "resampling_permitted": gate.get("resampling_permitted"),
    } != {
        "minimum_evaluable_case_count": 24,
        "frozen_total_case_count": 30,
        "resampling_permitted": False,
    }:
        raise ReviewFreezeError("3B1 preregistered coverage gate changed")

    templates = {}
    notes_required = {}
    for reviewer in ("A", "B"):
        rows, required, packet_locks = _validate_public_packet(
            directory / f"reviewer_{reviewer}", reviewer
        )
        templates[reviewer] = rows
        notes_required[reviewer] = required
        for name, lock in packet_locks.items():
            locks[f"reviewer_{reviewer}/{name}"] = lock
    return PublicCohort(
        directory=directory,
        public_source_locks=locks,
        build_report=report,
        selected_manifest=selected,
        selected_units=selected_units,
        selected_case_ids=frozenset(selected_case_ids),
        templates=templates,
        notes_required=notes_required,
    )


def validate_review_return(
    cohort: PublicCohort,
    *,
    reviewer: str,
    completed_path: Path,
    provenance_path: Path,
    frozen_copy: bool = False,
) -> ValidatedReview:
    if reviewer not in {"A", "B"}:
        raise ReviewFreezeError("reviewer must be A or B")
    completed = safe_path(completed_path, f"Reviewer {reviewer} completed CSV")
    provenance_file = safe_path(
        provenance_path, f"Reviewer {reviewer} provenance"
    )
    expected_completed_name = (
        f"reviewer_{reviewer}_frozen.csv"
        if frozen_copy
        else f"STEP26R3B2_REVIEWER_{reviewer}_completed.csv"
    )
    expected_provenance_name = (
        f"reviewer_{reviewer}_provenance.json"
        if frozen_copy
        else f"STEP26R3B2_REVIEWER_{reviewer}_provenance.json"
    )
    if (
        completed.name != expected_completed_name
        or provenance_file.name != expected_provenance_name
    ):
        raise ReviewFreezeError(f"Reviewer {reviewer} return filename is invalid")
    if not completed.is_file() or not provenance_file.is_file():
        raise ReviewFreezeError(f"Reviewer {reviewer} return is missing")
    columns, rows = _read_csv(completed, f"Reviewer {reviewer} completed CSV")
    if columns != REVIEW_COLUMNS or len(rows) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError(f"Reviewer {reviewer} completed CSV schema changed")
    template = cohort.templates[reviewer]
    frozen_rows = []
    seen = set()
    for index, (row, original) in enumerate(zip(rows, template)):
        for field in _IMMUTABLE_REVIEW_FIELDS:
            if row[field] != original[field]:
                raise ReviewFreezeError(
                    f"Reviewer {reviewer} immutable field changed at row {index}: {field}"
                )
        key = (row["review_case_id"], row["review_unit_id"])
        if key in seen:
            raise ReviewFreezeError(f"Reviewer {reviewer} review ID is duplicated")
        seen.add(key)
        label = row["direct_relevance_label"]
        confidence = row["review_confidence"]
        if not label:
            raise ReviewFreezeError(f"Reviewer {reviewer} relevance label is blank")
        if label not in RELEVANCE_LABELS:
            raise ReviewFreezeError(f"Reviewer {reviewer} relevance label is invalid")
        if not confidence:
            raise ReviewFreezeError(f"Reviewer {reviewer} confidence is blank")
        if confidence not in CONFIDENCE_LABELS:
            raise ReviewFreezeError(f"Reviewer {reviewer} confidence is invalid")
        if cohort.notes_required[reviewer] and not row["review_note"]:
            raise ReviewFreezeError(f"Reviewer {reviewer} review note is required")
        frozen_rows.append(FrozenReviewRow(**row))

    completed_bytes = completed.read_bytes()
    completed_sha = sha256_bytes(completed_bytes)
    provenance_bytes = provenance_file.read_bytes()
    provenance = _read_json(provenance_file, f"Reviewer {reviewer} provenance")
    if set(provenance) != _REVIEW_PROVENANCE_FIELDS:
        raise ReviewFreezeError(f"Reviewer {reviewer} provenance schema changed")
    required = {
        "schema_version": 1,
        "stage": "step2.6r-3b2",
        "reviewer": reviewer,
        "review_type": "independent_score_blind_direct_relevance_annotation",
        "completed_csv_sha256": completed_sha,
        "row_count": EXPECTED_UNIT_COUNT,
        "case_count": EXPECTED_CASE_COUNT,
        "web_search_used": False,
        "external_sources_used": False,
        "other_reviewer_output_accessed": False,
        "dataset_identity_accessed": False,
        "modality_identity_accessed": False,
        "selector_outputs_accessed": False,
        "veracity_labels_accessed": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "completed_all_rows": True,
    }
    for field, expected in required.items():
        if provenance.get(field) != expected:
            raise ReviewFreezeError(
                f"Reviewer {reviewer} provenance contract failed: {field}"
            )
    return ValidatedReview(
        reviewer=reviewer,
        completed_path=completed,
        completed_sha256=completed_sha,
        completed_bytes=completed_bytes,
        provenance_path=provenance_file,
        provenance_sha256=sha256_bytes(provenance_bytes),
        provenance_bytes=provenance_bytes,
        provenance=provenance,
        rows=tuple(frozen_rows),
    )


def load_private_mapping(
    cohort: PublicCohort,
    reviewer_a: ValidatedReview,
    reviewer_b: ValidatedReview,
) -> Tuple[Mapping[str, Any], Mapping[str, str]]:
    """Open the private mapping only after both public returns are validated."""

    if reviewer_a.reviewer != "A" or reviewer_b.reviewer != "B":
        raise ReviewFreezeError("both independently validated reviews are required")
    path = cohort.directory / _PRIVATE_MAPPING
    lock = _verify_sidecar(path, "3B1-R2 private review mapping")
    payload = _read_json(path, "3B1-R2 private review mapping")
    if payload.get("status") != "PRIVATE_FROZEN_MAPPING" or payload.get(
        "implementation_revision"
    ) != SOURCE_IMPLEMENTATION_REVISION:
        raise ReviewFreezeError("private review mapping contract changed")
    return payload, lock
