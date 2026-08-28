"""Deterministic blinded adjudication and final-gold coverage helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .review_loader import (
    csv_bytes,
    safe_path,
    sha256_bytes,
)
from .schemas import (
    ADJUDICATION_COLUMNS,
    ADJUDICATION_SALT,
    CONFIDENCE_LABELS,
    COVERAGE_GATE_MINIMUM,
    EXPECTED_CASE_COUNT,
    IMPLEMENTATION_REVISION,
    RELEVANCE_LABELS,
    AlignedReviewRow,
    ReviewFreezeError,
    binary_direct_target,
)


ADJUDICATOR_INSTRUCTIONS = """# Independent Direct-Relevance Adjudication

This packet contains only rows where two independent reviewers disagreed.

- Judge semantic direct relevance independently.
- Do not infer fake/real truthfulness.
- Do not search the web or use external evidence.
- Do not infer dataset, modality, source identity, or prior reviewer decisions.
- DIRECT is verification-relevant and is not necessarily supportive.
- RELATED is not DIRECT.
- Use only DIRECT, RELATED, IRRELEVANT, or UNREADABLE.
- Use HIGH, MEDIUM, or LOW confidence.
- Complete final_relevance_label, adjudication_confidence, and adjudication_note for every row.
"""


@dataclass(frozen=True)
class AdjudicationPacket:
    rows: Tuple[Mapping[str, str], ...]
    mapping: Tuple[Mapping[str, Any], ...]
    case_count: int


@dataclass(frozen=True)
class ValidatedAdjudication:
    completed_path: Path
    completed_sha256: str
    completed_bytes: bytes
    provenance_path: Path
    provenance_sha256: str
    provenance_bytes: bytes
    provenance: Mapping[str, Any]
    labels_by_underlying: Mapping[Tuple[str, str], str]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_adjudication_packet(
    aligned_rows: Sequence[AlignedReviewRow],
) -> AdjudicationPacket:
    disagreements = [
        row
        for row in aligned_rows
        if row.reviewer_a.direct_relevance_label
        != row.reviewer_b.direct_relevance_label
    ]
    by_case: Dict[str, list[AlignedReviewRow]] = defaultdict(list)
    for row in disagreements:
        by_case[row.underlying.canonical_case_id].append(row)
    ordered_cases = sorted(
        by_case,
        key=lambda case_id: (
            _digest(f"{ADJUDICATION_SALT}|case-order|{case_id}"),
            case_id,
        ),
    )
    rows = []
    mapping = []
    unit_counter = 0
    for case_index, canonical in enumerate(ordered_cases, start=1):
        adjudication_case_id = f"ADJ-C-{case_index:03d}"
        ordered_units = sorted(
            by_case[canonical],
            key=lambda row: (
                _digest(
                    f"{ADJUDICATION_SALT}|unit-order|{canonical}|{row.underlying.unit_id}"
                ),
                row.underlying.unit_id,
            ),
        )
        for row in ordered_units:
            unit_counter += 1
            adjudication_unit_id = f"ADJ-U-{unit_counter:04d}"
            rows.append(
                {
                    "adjudication_case_id": adjudication_case_id,
                    "claim": row.claim,
                    "adjudication_unit_id": adjudication_unit_id,
                    "candidate_text": row.candidate_text,
                    "final_relevance_label": "",
                    "adjudication_confidence": "",
                    "adjudication_note": "",
                }
            )
            mapping.append(
                {
                    "adjudication_case_id": adjudication_case_id,
                    "adjudication_unit_id": adjudication_unit_id,
                    "dataset": row.underlying.dataset,
                    "canonical_case_id": row.underlying.canonical_case_id,
                    "original_case_id": row.underlying.original_case_id,
                    "unit_id": row.underlying.unit_id,
                    "unit_type": row.underlying.unit_type,
                    "modality": row.underlying.modality,
                    "original_candidate_position": row.underlying.original_candidate_position,
                }
            )
    return AdjudicationPacket(
        rows=tuple(rows), mapping=tuple(mapping), case_count=len(ordered_cases)
    )


def adjudication_manifest(packet: AdjudicationPacket, template: bytes) -> Mapping[str, Any]:
    instructions = ADJUDICATOR_INSTRUCTIONS.encode("utf-8")
    return {
        "status": "READY_FOR_INDEPENDENT_ADJUDICATION",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "artifact_type": "blinded_direct_relevance_adjudication_packet",
        "row_count": len(packet.rows),
        "case_count": packet.case_count,
        "public_columns": list(ADJUDICATION_COLUMNS),
        "allowed_final_relevance_labels": list(RELEVANCE_LABELS),
        "allowed_adjudication_confidence": list(CONFIDENCE_LABELS),
        "template_sha256": sha256_bytes(template),
        "instructions_sha256": sha256_bytes(instructions),
        "reviewer_labels_blind": True,
        "reviewer_confidence_blind": True,
        "reviewer_notes_blind": True,
        "dataset_blind": True,
        "modality_blind": True,
        "underlying_identity_blind": True,
        "selector_blind": True,
        "veracity_label_blind": True,
    }


def _read_csv(path: Path) -> Tuple[Tuple[str, ...], Tuple[Mapping[str, str], ...]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            return tuple(reader.fieldnames or ()), tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewFreezeError("adjudication completed CSV is invalid") from exc


def validate_adjudication_return(
    *,
    template_path: Path,
    private_mapping: Sequence[Mapping[str, Any]],
    completed_path: Path,
    provenance_path: Path,
) -> ValidatedAdjudication:
    template_file = safe_path(template_path, "adjudication template")
    completed = safe_path(completed_path, "adjudication completed CSV")
    provenance_file = safe_path(provenance_path, "adjudication provenance")
    if (
        completed.name != "STEP26R3B2_ADJUDICATION_completed.csv"
        or provenance_file.name != "STEP26R3B2_ADJUDICATION_provenance.json"
    ):
        raise ReviewFreezeError("adjudication return filename is invalid")
    if not template_file.is_file() or not completed.is_file() or not provenance_file.is_file():
        raise ReviewFreezeError("adjudication return or template is missing")
    template_columns, template_rows = _read_csv(template_file)
    columns, rows = _read_csv(completed)
    if template_columns != ADJUDICATION_COLUMNS or columns != ADJUDICATION_COLUMNS:
        raise ReviewFreezeError("adjudication CSV schema changed")
    if len(rows) != len(template_rows) or len(rows) != len(private_mapping):
        raise ReviewFreezeError("adjudication row count differs from disagreement count")
    blind_mapping = {}
    underlying_seen = set()
    for item in private_mapping:
        blind = (item.get("adjudication_case_id"), item.get("adjudication_unit_id"))
        underlying = (item.get("canonical_case_id"), item.get("unit_id"))
        if (
            not all(isinstance(value, str) and value for value in (*blind, *underlying))
            or blind in blind_mapping
            or underlying in underlying_seen
        ):
            raise ReviewFreezeError("private adjudication mapping is not one-to-one")
        blind_mapping[blind] = underlying
        underlying_seen.add(underlying)

    labels = {}
    immutable = (
        "adjudication_case_id",
        "claim",
        "adjudication_unit_id",
        "candidate_text",
    )
    for index, (row, original) in enumerate(zip(rows, template_rows)):
        if any(row[field] != original[field] for field in immutable):
            raise ReviewFreezeError(
                f"adjudication immutable field changed at row {index}"
            )
        label = row["final_relevance_label"]
        confidence = row["adjudication_confidence"]
        if label not in RELEVANCE_LABELS:
            raise ReviewFreezeError("adjudication final relevance label is invalid")
        if confidence not in CONFIDENCE_LABELS:
            raise ReviewFreezeError("adjudication confidence is invalid")
        if not row["adjudication_note"]:
            raise ReviewFreezeError("adjudication note is required")
        blind = (row["adjudication_case_id"], row["adjudication_unit_id"])
        underlying = blind_mapping.get(blind)
        if underlying is None or underlying in labels:
            raise ReviewFreezeError("adjudication row is not uniquely mapped")
        labels[underlying] = label

    completed_bytes = completed.read_bytes()
    completed_sha = sha256_bytes(completed_bytes)
    provenance_bytes = provenance_file.read_bytes()
    try:
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewFreezeError("adjudication provenance is invalid") from exc
    required_fields = {
        "schema_version",
        "stage",
        "reviewer",
        "review_type",
        "completed_csv_sha256",
        "row_count",
        "case_count",
        "web_search_used",
        "external_sources_used",
        "reviewer_a_labels_accessed",
        "reviewer_b_labels_accessed",
        "dataset_identity_accessed",
        "modality_identity_accessed",
        "selector_outputs_accessed",
        "veracity_labels_accessed",
        "formal_validation_accessed",
        "formal_test_accessed",
        "completed_all_rows",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required_fields:
        raise ReviewFreezeError("adjudication provenance schema changed")
    required = {
        "schema_version": 1,
        "stage": "step2.6r-3b2",
        "reviewer": "ADJUDICATOR",
        "review_type": "independent_score_blind_direct_relevance_adjudication",
        "completed_csv_sha256": completed_sha,
        "row_count": len(rows),
        "case_count": len({row["adjudication_case_id"] for row in rows}),
        "web_search_used": False,
        "external_sources_used": False,
        "reviewer_a_labels_accessed": False,
        "reviewer_b_labels_accessed": False,
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
            raise ReviewFreezeError(f"adjudication provenance contract failed: {field}")
    return ValidatedAdjudication(
        completed_path=completed,
        completed_sha256=completed_sha,
        completed_bytes=completed_bytes,
        provenance_path=provenance_file,
        provenance_sha256=sha256_bytes(provenance_bytes),
        provenance_bytes=provenance_bytes,
        provenance=provenance,
        labels_by_underlying=labels,
    )


def coverage_report(final_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_case: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in final_rows:
        label = row.get("final_relevance_label")
        if label not in RELEVANCE_LABELS:
            raise ReviewFreezeError("final gold relevance label is invalid")
        by_case[(row["dataset"], row["canonical_case_id"])][label] += 1
    if len(by_case) != EXPECTED_CASE_COUNT:
        raise ReviewFreezeError("final gold case count is not 30")
    case_rows = []
    dataset_total = Counter()
    dataset_evaluable = Counter()
    for (dataset, canonical), counts in sorted(by_case.items()):
        candidate_count = sum(counts.values())
        has_direct = counts["DIRECT"] >= 1
        dataset_total[dataset] += 1
        dataset_evaluable[dataset] += int(has_direct)
        case_rows.append(
            {
                "dataset": dataset,
                "canonical_case_id": canonical,
                "candidate_count": candidate_count,
                "DIRECT_count": counts["DIRECT"],
                "RELATED_count": counts["RELATED"],
                "IRRELEVANT_count": counts["IRRELEVANT"],
                "UNREADABLE_count": counts["UNREADABLE"],
                "has_DIRECT": has_direct,
            }
        )
    evaluable = sum(dataset_evaluable.values())
    if dataset_total != Counter({"GroundLie360": 15, "TRUE-3MFact": 15}):
        raise ReviewFreezeError("final gold dataset case counts changed")
    passed = evaluable >= COVERAGE_GATE_MINIMUM
    return {
        "status": (
            "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_PASS"
            if passed
            else "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_INSUFFICIENT"
        ),
        "frozen_case_count": EXPECTED_CASE_COUNT,
        "frozen_unit_count": len(final_rows),
        "evaluable_case_count": evaluable,
        "zero_direct_positive_case_count": EXPECTED_CASE_COUNT - evaluable,
        "coverage_rate": evaluable / EXPECTED_CASE_COUNT,
        "coverage_gate_minimum": COVERAGE_GATE_MINIMUM,
        "coverage_gate_pass": passed,
        "per_dataset": {
            dataset: {
                "total_case_count": dataset_total[dataset],
                "evaluable_case_count": dataset_evaluable[dataset],
            }
            for dataset in ("GroundLie360", "TRUE-3MFact")
        },
        "case_coverage": case_rows,
        "resampling_performed": False,
    }
