"""Phase 3B2-B: freeze adjudication and final direct-relevance gold."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .adjudication import coverage_report, validate_adjudication_return
from .agreement import align_reviews, compute_agreement
from .review_loader import (
    json_bytes,
    jsonl_bytes,
    load_private_mapping,
    load_public_cohort,
    safe_path,
    sha256_bytes,
    sha256_file,
    validate_review_return,
    write_artifact,
)
from .schemas import (
    ADJUDICATION_COLUMNS,
    EXPECTED_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    FINAL_GOLD_FIELDS,
    IMPLEMENTATION_REVISION,
    RELEVANCE_LABELS,
    ReviewFreezeError,
    binary_direct_target,
)


_REVIEW_FREEZE_ARTIFACTS = (
    "review_source_lock.json",
    "reviewer_A_frozen.csv",
    "reviewer_A_provenance.json",
    "reviewer_B_frozen.csv",
    "reviewer_B_provenance.json",
    "agreement_report.json",
    "agreement_by_case.csv",
    "review_resolution_pre_adjudication.jsonl",
    "private_agreement_mapping.json",
    "review_freeze_report.json",
)


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewFreezeError(f"{field} is invalid") from exc
    if not isinstance(value, Mapping):
        raise ReviewFreezeError(f"{field} must be an object")
    return value


def _read_jsonl(path: Path, field: str) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ReviewFreezeError(f"{field} row is invalid")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewFreezeError(f"{field} is invalid") from exc
    return tuple(rows)


def _verify_artifact(path: Path, field: str) -> Mapping[str, str]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ReviewFreezeError(f"{field} or SHA sidecar is missing")
    expected = sidecar.read_text(encoding="utf-8").strip().casefold()
    actual = sha256_file(path)
    if expected != actual:
        raise ReviewFreezeError(f"{field} SHA-256 mismatch")
    return {"path": str(path), "sha256": actual}


def _validate_adjudication_packet(
    review_dir: Path,
    disagreement_count: int,
    expected_composite_sha256: str,
) -> None:
    packet_dir = review_dir / "adjudication_packet"
    expected_files = {
        "README_ADJUDICATOR.md",
        "ADJUDICATION_MANIFEST.json",
        "adjudication_template.csv",
    }
    if not packet_dir.is_dir() or {
        path.name for path in packet_dir.iterdir()
    } != expected_files:
        raise ReviewFreezeError("adjudication packet files changed")
    manifest = _read_json(
        packet_dir / "ADJUDICATION_MANIFEST.json", "adjudication manifest"
    )
    required = {
        "status": "READY_FOR_INDEPENDENT_ADJUDICATION",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "row_count": disagreement_count,
        "public_columns": list(ADJUDICATION_COLUMNS),
        "reviewer_labels_blind": True,
        "reviewer_confidence_blind": True,
        "reviewer_notes_blind": True,
        "dataset_blind": True,
        "modality_blind": True,
        "underlying_identity_blind": True,
        "selector_blind": True,
        "veracity_label_blind": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ReviewFreezeError(f"adjudication packet contract failed: {field}")
    instructions = (packet_dir / "README_ADJUDICATOR.md").read_bytes()
    manifest_bytes = (packet_dir / "ADJUDICATION_MANIFEST.json").read_bytes()
    template_path = packet_dir / "adjudication_template.csv"
    template = template_path.read_bytes()
    if manifest.get("instructions_sha256") != sha256_bytes(instructions):
        raise ReviewFreezeError("adjudication instructions SHA mismatch")
    if manifest.get("template_sha256") != sha256_bytes(template):
        raise ReviewFreezeError("adjudication template SHA mismatch")
    if sha256_bytes(instructions + manifest_bytes + template) != expected_composite_sha256:
        raise ReviewFreezeError("adjudication packet composite SHA mismatch")
    try:
        with template_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewFreezeError("adjudication template is invalid") from exc
    if columns != ADJUDICATION_COLUMNS or len(rows) != disagreement_count:
        raise ReviewFreezeError("adjudication template schema/count changed")
    seen = set()
    for row in rows:
        blind = (row["adjudication_case_id"], row["adjudication_unit_id"])
        if (
            blind in seen
            or not row["adjudication_case_id"].startswith("ADJ-C-")
            or not row["adjudication_unit_id"].startswith("ADJ-U-")
            or any(
                row[field]
                for field in (
                    "final_relevance_label",
                    "adjudication_confidence",
                    "adjudication_note",
                )
            )
        ):
            raise ReviewFreezeError("adjudication template is not uniquely blank/blinded")
        seen.add(blind)


def freeze_adjudication(
    *,
    cohort_dir: Path,
    review_freeze_dir: Path,
    output_dir: Path,
    adjudication_completed: Optional[Path] = None,
    adjudication_provenance: Optional[Path] = None,
) -> Path:
    cohort = load_public_cohort(cohort_dir)
    review_dir = safe_path(review_freeze_dir, "review-freeze directory")
    output = safe_path(output_dir, "final-gold output directory")
    if not review_dir.is_dir():
        raise ReviewFreezeError("review-freeze directory is missing")
    if output.exists():
        raise ReviewFreezeError("final-gold output directory already exists")

    review_locks: Dict[str, Mapping[str, str]] = {}
    for name in _REVIEW_FREEZE_ARTIFACTS:
        review_locks[name] = _verify_artifact(
            review_dir / name, f"review-freeze {name}"
        )
    freeze_report = _read_json(
        review_dir / "review_freeze_report.json", "review freeze report"
    )
    if freeze_report.get("status") != "INDEPENDENT_REVIEW_FREEZE_AND_AGREEMENT_PASS" or freeze_report.get(
        "implementation_revision"
    ) != IMPLEMENTATION_REVISION:
        raise ReviewFreezeError("review freeze report contract changed")
    disagreement_count = freeze_report.get("disagreement_count")
    if type(disagreement_count) is not int or disagreement_count < 0:
        raise ReviewFreezeError("review disagreement count is invalid")
    source_lock = _read_json(
        review_dir / "review_source_lock.json", "review source lock"
    )
    required_source_lock = {
        "status": "PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_3b1_revision": "step2.6r-3b1-r2-v1",
        "private_mapping_opened_after_both_reviews_validated_and_frozen": True,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_historical_reference_content_accessed": False,
    }
    for field, expected in required_source_lock.items():
        if source_lock.get(field) != expected:
            raise ReviewFreezeError(f"review source lock contract failed: {field}")
    report_hash_fields = {
        "reviewer_A_frozen.csv": "reviewer_a_frozen_csv_sha256",
        "reviewer_A_provenance.json": "reviewer_a_provenance_sha256",
        "reviewer_B_frozen.csv": "reviewer_b_frozen_csv_sha256",
        "reviewer_B_provenance.json": "reviewer_b_provenance_sha256",
        "review_source_lock.json": "review_source_lock_sha256",
        "agreement_report.json": "agreement_report_sha256",
        "agreement_by_case.csv": "agreement_by_case_sha256",
        "review_resolution_pre_adjudication.jsonl": "pre_adjudication_ledger_sha256",
        "private_agreement_mapping.json": "private_agreement_mapping_sha256",
    }
    for name, field in report_hash_fields.items():
        if freeze_report.get(field) != review_locks[name]["sha256"]:
            raise ReviewFreezeError(f"review freeze report hash pointer failed: {field}")
    agreement_report = _read_json(
        review_dir / "agreement_report.json", "agreement report"
    )
    if (
        agreement_report.get("status")
        != "INDEPENDENT_REVIEW_AGREEMENT_AUDIT_COMPLETE"
        or agreement_report.get("implementation_revision")
        != IMPLEMENTATION_REVISION
        or agreement_report.get("total_unit_count") != EXPECTED_UNIT_COUNT
        or agreement_report.get("exact_four_class_disagreement_count")
        != disagreement_count
    ):
        raise ReviewFreezeError("agreement report contract changed")
    frozen_review_a = validate_review_return(
        cohort,
        reviewer="A",
        completed_path=review_dir / "reviewer_A_frozen.csv",
        provenance_path=review_dir / "reviewer_A_provenance.json",
        frozen_copy=True,
    )
    frozen_review_b = validate_review_return(
        cohort,
        reviewer="B",
        completed_path=review_dir / "reviewer_B_frozen.csv",
        provenance_path=review_dir / "reviewer_B_provenance.json",
        frozen_copy=True,
    )
    frozen_private_mapping, frozen_private_mapping_lock = load_private_mapping(
        cohort, frozen_review_a, frozen_review_b
    )
    recomputed_agreement = compute_agreement(
        align_reviews(
            cohort,
            frozen_review_a,
            frozen_review_b,
            frozen_private_mapping,
        )
    )
    for field, expected in recomputed_agreement.report.items():
        if agreement_report.get(field) != expected:
            raise ReviewFreezeError(f"agreement report recomputation failed: {field}")
    ledger = _read_jsonl(
        review_dir / "review_resolution_pre_adjudication.jsonl",
        "pre-adjudication ledger",
    )
    if len(ledger) != EXPECTED_UNIT_COUNT:
        raise ReviewFreezeError("pre-adjudication ledger count is not 289")
    if ledger != recomputed_agreement.ledger_rows:
        raise ReviewFreezeError("pre-adjudication ledger recomputation failed")
    private_agreement = _read_json(
        review_dir / "private_agreement_mapping.json",
        "private agreement mapping",
    )
    private_rows = private_agreement.get("rows")
    if (
        private_agreement.get("status") != "PRIVATE_AGREEMENT_MAPPING_FROZEN"
        or private_agreement.get("implementation_revision") != IMPLEMENTATION_REVISION
        or not isinstance(private_rows, list)
        or len(private_rows) != EXPECTED_UNIT_COUNT
    ):
        raise ReviewFreezeError("private agreement mapping count is not 289")
    private_keys = set()
    aligned_by_key = {
        row.underlying.key: row for row in recomputed_agreement.aligned_rows
    }
    exact_private_agreement_fields = {
        "dataset",
        "canonical_case_id",
        "original_case_id",
        "unit_id",
        "unit_type",
        "modality",
        "original_candidate_position",
        "claim",
        "candidate_text",
        "reviewer_a_review_case_id",
        "reviewer_a_review_unit_id",
        "reviewer_b_review_case_id",
        "reviewer_b_review_unit_id",
    }
    for row in private_rows:
        if not isinstance(row, Mapping) or set(row) != exact_private_agreement_fields:
            raise ReviewFreezeError("private agreement mapping row is invalid")
        key = (row.get("canonical_case_id"), row.get("unit_id"))
        unit = cohort.selected_units.get(key)
        aligned_row = aligned_by_key.get(key)
        if (
            unit is None
            or aligned_row is None
            or key in private_keys
            or row.get("dataset") != unit.dataset
            or row.get("original_case_id") != unit.original_case_id
            or row.get("unit_type") != unit.unit_type
            or row.get("modality") != unit.modality
            or row.get("original_candidate_position")
            != unit.original_candidate_position
            or row.get("claim") != aligned_row.claim
            or row.get("candidate_text") != aligned_row.candidate_text
            or row.get("reviewer_a_review_case_id")
            != aligned_row.reviewer_a.review_case_id
            or row.get("reviewer_a_review_unit_id")
            != aligned_row.reviewer_a.review_unit_id
            or row.get("reviewer_b_review_case_id")
            != aligned_row.reviewer_b.review_case_id
            or row.get("reviewer_b_review_unit_id")
            != aligned_row.reviewer_b.review_unit_id
        ):
            raise ReviewFreezeError("private agreement mapping differs from 3B1")
        private_keys.add(key)
    if private_keys != set(cohort.selected_units):
        raise ReviewFreezeError("private agreement mapping unit set differs from 3B1")

    adjudication = None
    private_adjudication_lock = None
    if disagreement_count:
        if adjudication_completed is None or adjudication_provenance is None:
            raise ReviewFreezeError("adjudication return is required for disagreements")
        private_mapping_path = review_dir / "private_adjudication_mapping.json"
        private_adjudication_lock = _verify_artifact(
            private_mapping_path, "private adjudication mapping"
        )
        private_adjudication = _read_json(
            private_mapping_path, "private adjudication mapping"
        )
        mapping_rows = private_adjudication.get("rows")
        if (
            private_adjudication.get("status")
            != "PRIVATE_ADJUDICATION_MAPPING_FROZEN"
            or private_adjudication.get("implementation_revision")
            != IMPLEMENTATION_REVISION
            or private_adjudication.get("adjudication_salt")
            != "step2.6r-3b2-adjudication-v1"
            or not isinstance(mapping_rows, list)
            or len(mapping_rows) != disagreement_count
        ):
            raise ReviewFreezeError("private adjudication mapping count changed")
        exact_mapping_fields = {
            "adjudication_case_id",
            "adjudication_unit_id",
            "dataset",
            "canonical_case_id",
            "original_case_id",
            "unit_id",
            "unit_type",
            "modality",
            "original_candidate_position",
        }
        if any(
            not isinstance(row, Mapping) or set(row) != exact_mapping_fields
            for row in mapping_rows
        ):
            raise ReviewFreezeError("private adjudication mapping schema changed")
        for row in mapping_rows:
            unit = cohort.selected_units.get(
                (row["canonical_case_id"], row["unit_id"])
            )
            if (
                unit is None
                or row["dataset"] != unit.dataset
                or row["original_case_id"] != unit.original_case_id
                or row["unit_type"] != unit.unit_type
                or row["modality"] != unit.modality
                or row["original_candidate_position"]
                != unit.original_candidate_position
            ):
                raise ReviewFreezeError(
                    "private adjudication mapping differs from 3B1"
                )
        composite_sha = freeze_report.get("adjudication_packet_composite_sha256")
        if not isinstance(composite_sha, str) or len(composite_sha) != 64:
            raise ReviewFreezeError("adjudication packet hash pointer is invalid")
        if freeze_report.get("private_adjudication_mapping_sha256") != private_adjudication_lock["sha256"]:
            raise ReviewFreezeError("private adjudication mapping hash pointer failed")
        _validate_adjudication_packet(
            review_dir, disagreement_count, composite_sha
        )
        adjudication = validate_adjudication_return(
            template_path=review_dir
            / "adjudication_packet"
            / "adjudication_template.csv",
            private_mapping=mapping_rows,
            completed_path=adjudication_completed,
            provenance_path=adjudication_provenance,
        )
    elif adjudication_completed is not None or adjudication_provenance is not None:
        raise ReviewFreezeError("adjudication must not be supplied for zero disagreements")

    selected_keys = set(cohort.selected_units)
    ledger_by_key = {}
    for row in ledger:
        key = (row.get("canonical_case_id"), row.get("unit_id"))
        if key in ledger_by_key or key not in selected_keys:
            raise ReviewFreezeError("pre-adjudication ledger unit set changed")
        ledger_by_key[key] = row
    if set(ledger_by_key) != selected_keys:
        raise ReviewFreezeError("pre-adjudication ledger unit set differs from 3B1")

    adjudicated_labels = (
        dict(adjudication.labels_by_underlying) if adjudication is not None else {}
    )
    disagreement_keys = {
        key
        for key, row in ledger_by_key.items()
        if row.get("pre_adjudication_status") == "NEEDS_ADJUDICATION"
    }
    if len(disagreement_keys) != disagreement_count:
        raise ReviewFreezeError("frozen disagreement ledger count changed")
    if set(adjudicated_labels) != disagreement_keys:
        raise ReviewFreezeError("adjudication does not resolve exactly the disagreements")

    final_rows = []
    resolution_ledger = []
    for key in sorted(
        selected_keys,
        key=lambda item: (
            item[0],
            cohort.selected_units[item].original_candidate_position,
            item[1],
        ),
    ):
        source = ledger_by_key[key]
        unit = cohort.selected_units[key]
        agreement = source.get("agreement") is True
        if agreement:
            if (
                source.get("pre_adjudication_status") != "AGREED"
                or source.get("reviewer_a_label") != source.get("reviewer_b_label")
            ):
                raise ReviewFreezeError("agreed row contract changed")
            label = source["reviewer_a_label"]
            resolution_source = "REVIEWER_AGREEMENT"
        else:
            if source.get("pre_adjudication_status") != "NEEDS_ADJUDICATION":
                raise ReviewFreezeError("disagreement row contract changed")
            label = adjudicated_labels[key]
            resolution_source = "INDEPENDENT_ADJUDICATION"
        if label not in RELEVANCE_LABELS:
            raise ReviewFreezeError("final relevance label is invalid")
        final = {
            "dataset": unit.dataset,
            "canonical_case_id": unit.canonical_case_id,
            "unit_id": unit.unit_id,
            "original_candidate_position": unit.original_candidate_position,
            "final_relevance_label": label,
            "binary_direct_relevance_target": binary_direct_target(label),
            "resolution_source": resolution_source,
        }
        if tuple(final) != FINAL_GOLD_FIELDS:
            raise ReviewFreezeError("final gold schema changed")
        final_rows.append(final)
        resolution_ledger.append(
            {
                **source,
                "final_relevance_label": label,
                "binary_direct_relevance_target": binary_direct_target(label),
                "resolution_source": resolution_source,
            }
        )
    if len(final_rows) != EXPECTED_UNIT_COUNT or len(
        {row["canonical_case_id"] for row in final_rows}
    ) != EXPECTED_CASE_COUNT:
        raise ReviewFreezeError("final gold count invariants failed")
    coverage = coverage_report(final_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".final-gold-", dir=output.parent))
    try:
        source_lock = {
            "status": "PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "cohort_dir": str(cohort.directory),
            "review_freeze_dir": str(review_dir),
            "cohort_public_artifacts": cohort.public_source_locks,
            "cohort_private_review_mapping": frozen_private_mapping_lock,
            "review_freeze_artifacts": review_locks,
            "private_adjudication_mapping": private_adjudication_lock,
            "adjudication_completed_input": (
                {
                    "path": str(adjudication.completed_path),
                    "sha256": adjudication.completed_sha256,
                }
                if adjudication is not None
                else None
            ),
            "adjudication_provenance_input": (
                {
                    "path": str(adjudication.provenance_path),
                    "sha256": adjudication.provenance_sha256,
                }
                if adjudication is not None
                else None
            ),
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
        }
        source_lock_sha = write_artifact(
            staging / "final_gold_source_lock.json", json_bytes(source_lock)
        )
        final_gold_sha = write_artifact(
            staging / "final_relevance_gold.jsonl", jsonl_bytes(final_rows)
        )
        resolution_sha = write_artifact(
            staging / "review_resolution_ledger.jsonl",
            jsonl_bytes(resolution_ledger),
        )
        coverage_sha = write_artifact(
            staging / "coverage_report.json", json_bytes(coverage)
        )
        adjudication_csv_sha = None
        adjudication_provenance_sha = None
        if adjudication is not None:
            adjudication_csv_sha = write_artifact(
                staging / "adjudication_frozen.csv", adjudication.completed_bytes
            )
            adjudication_provenance_sha = write_artifact(
                staging / "adjudication_provenance.json",
                adjudication.provenance_bytes,
            )
        manifest = {
            "status": "FINAL_RELEVANCE_GOLD_FROZEN",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "frozen_case_count": EXPECTED_CASE_COUNT,
            "frozen_unit_count": EXPECTED_UNIT_COUNT,
            "final_gold_fields": list(FINAL_GOLD_FIELDS),
            "final_relevance_gold_sha256": final_gold_sha,
            "review_resolution_ledger_sha256": resolution_sha,
            "coverage_report_sha256": coverage_sha,
            "source_lock_sha256": source_lock_sha,
            "adjudication_used": adjudication is not None,
            "adjudication_frozen_csv_sha256": adjudication_csv_sha,
            "adjudication_provenance_sha256": adjudication_provenance_sha,
            "coverage_gate_pass": coverage["coverage_gate_pass"],
            "resampling_performed": False,
        }
        manifest_sha = write_artifact(
            staging / "final_gold_manifest.json", json_bytes(manifest)
        )
        freeze_status = (
            "FINAL_RELEVANCE_GOLD_FREEZE_PASS"
            if coverage["coverage_gate_pass"]
            else "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_INSUFFICIENT"
        )
        freeze_report = {
            "status": freeze_status,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "frozen_case_count": EXPECTED_CASE_COUNT,
            "frozen_unit_count": EXPECTED_UNIT_COUNT,
            "evaluable_case_count": coverage["evaluable_case_count"],
            "coverage_gate_minimum": coverage["coverage_gate_minimum"],
            "coverage_gate_pass": coverage["coverage_gate_pass"],
            "final_gold_manifest_sha256": manifest_sha,
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
        }
        write_artifact(
            staging / "final_gold_freeze_report.json", json_bytes(freeze_report)
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze adjudication and final direct-relevance gold."
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--review-freeze-dir", type=Path, required=True)
    parser.add_argument("--adjudication-completed", type=Path)
    parser.add_argument("--adjudication-provenance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = freeze_adjudication(
            cohort_dir=args.cohort_dir,
            review_freeze_dir=args.review_freeze_dir,
            adjudication_completed=args.adjudication_completed,
            adjudication_provenance=args.adjudication_provenance,
            output_dir=args.output_dir,
        )
    except ReviewFreezeError as exc:
        print(f"adjudication freeze failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
