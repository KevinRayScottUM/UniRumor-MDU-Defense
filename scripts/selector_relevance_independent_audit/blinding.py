"""Deterministic independent public-packet blinding."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .schemas import (
    DIRECT_RELEVANCE_LABELS,
    IMPLEMENTATION_REVISION,
    PUBLIC_REVIEW_COLUMNS,
    REVIEW_CONFIDENCE_LABELS,
    AuditCase,
)


PUBLIC_PACKET_FILES = (
    "README_REVIEWER.md",
    "REVIEW_MANIFEST.json",
    "relevance_review_template.csv",
)


REVIEWER_INSTRUCTIONS = """# Independent Direct-Relevance Review

This packet is an independently blinded semantic direct-relevance review.

## Rules

- Judge only semantic direct relevance to the claim.
- Do not infer a fake/real label.
- Do not search the web or use external factual knowledge.
- Do not infer hidden dataset or source identity.
- Do not guess whether text is OCR or transcript.
- Do not rank candidates against each other; label each candidate independently.
- DIRECT means verification-relevant and is not necessarily supportive.
- RELATED is not DIRECT.
- Use UNREADABLE when text quality prevents reliable judgment.
- Complete `direct_relevance_label`, `review_confidence`, and `review_note` for every row.

## Labels

- DIRECT: directly addresses a material proposition, entity-event relationship, factual assertion, or verification-relevant claim component.
- RELATED: topically related or contextual, but not directly probative for a material claim proposition.
- IRRELEVANT: does not materially help assess the claim.
- UNREADABLE: too corrupted, fragmentary, or unintelligible for reliable semantic judgment.

## Confidence

Use HIGH, MEDIUM, or LOW. Confidence is descriptive only.
"""


@dataclass(frozen=True)
class ReviewPacket:
    reviewer: str
    rows: Tuple[Mapping[str, str], ...]
    mapping: Tuple[Mapping[str, Any], ...]
    case_order: Tuple[str, ...]
    underlying_unit_keys: Tuple[Tuple[str, str], ...]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_review_packet(
    cases: Sequence[AuditCase], *, reviewer: str, salt: str
) -> ReviewPacket:
    if reviewer not in {"A", "B"}:
        raise ValueError("reviewer must be A or B")
    ordered_cases = sorted(
        cases,
        key=lambda case: (
            _digest(f"{salt}|case-order|{case.audit_case_id}"),
            case.audit_case_id,
        ),
    )
    rows = []
    mapping = []
    underlying = []
    case_order = []
    unit_counter = 0
    for case_index, case in enumerate(ordered_cases, start=1):
        review_case_id = f"{reviewer}-C-{case_index:03d}"
        case_order.append(case.audit_case_id)
        ordered_units = sorted(
            case.candidates,
            key=lambda candidate: (
                _digest(
                    f"{salt}|unit-order|{case.audit_case_id}|{candidate.unit_id}"
                ),
                candidate.unit_id,
            ),
        )
        for candidate in ordered_units:
            unit_counter += 1
            review_unit_id = f"{reviewer}-U-{unit_counter:04d}"
            rows.append(
                {
                    "review_case_id": review_case_id,
                    "claim": case.claim,
                    "review_unit_id": review_unit_id,
                    "candidate_text": candidate.text,
                    "direct_relevance_label": "",
                    "review_confidence": "",
                    "review_note": "",
                }
            )
            mapping.append(
                {
                    "reviewer": reviewer,
                    "review_case_id": review_case_id,
                    "review_unit_id": review_unit_id,
                    "audit_case_id": case.audit_case_id,
                    "dataset": case.dataset,
                    "canonical_case_id": case.canonical_case_id,
                    "original_case_id": case.original_case_id,
                    "unit_id": candidate.unit_id,
                    "unit_type": candidate.unit_type,
                    "modality": candidate.modality,
                    "original_candidate_position": candidate.original_candidate_position,
                }
            )
            underlying.append((case.audit_case_id, candidate.unit_id))
    return ReviewPacket(
        reviewer=reviewer,
        rows=tuple(rows),
        mapping=tuple(mapping),
        case_order=tuple(case_order),
        underlying_unit_keys=tuple(underlying),
    )


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=PUBLIC_REVIEW_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def packet_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in sorted(PUBLIC_PACKET_FILES):
        payload = (directory / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def write_review_packet(directory: Path, packet: ReviewPacket) -> str:
    directory.mkdir(parents=False, exist_ok=False)
    readme = REVIEWER_INSTRUCTIONS.encode("utf-8")
    template = _csv_bytes(packet.rows)
    manifest: Dict[str, Any] = {
        "status": "READY_FOR_INDEPENDENT_REVIEW",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "artifact_type": "blinded_direct_relevance_review_packet",
        "reviewer": packet.reviewer,
        "case_count": len(packet.case_order),
        "row_count": len(packet.rows),
        "public_columns": list(PUBLIC_REVIEW_COLUMNS),
        "allowed_direct_relevance_labels": list(DIRECT_RELEVANCE_LABELS),
        "allowed_review_confidence": list(REVIEW_CONFIDENCE_LABELS),
        "template_sha256": hashlib.sha256(template).hexdigest(),
        "instructions_sha256": hashlib.sha256(readme).hexdigest(),
        "review_fields_initially_blank": True,
        "dataset_blind": True,
        "modality_blind": True,
        "selector_blind": True,
        "veracity_label_blind": True,
    }
    (directory / "README_REVIEWER.md").write_bytes(readme)
    (directory / "relevance_review_template.csv").write_bytes(template)
    (directory / "REVIEW_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if tuple(sorted(path.name for path in directory.iterdir())) != tuple(
        sorted(PUBLIC_PACKET_FILES)
    ):
        raise ValueError("public review packet contains unexpected files")
    return packet_digest(directory)
