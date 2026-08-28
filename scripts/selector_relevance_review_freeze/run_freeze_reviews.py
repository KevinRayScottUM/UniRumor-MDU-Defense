"""Phase 3B2-A: freeze two reviews, audit agreement, and prepare adjudication."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .adjudication import (
    ADJUDICATOR_INSTRUCTIONS,
    adjudication_manifest,
    build_adjudication_packet,
)
from .agreement import align_reviews, compute_agreement
from .review_loader import (
    csv_bytes,
    json_bytes,
    jsonl_bytes,
    load_private_mapping,
    load_public_cohort,
    safe_path,
    sha256_bytes,
    validate_review_return,
    write_artifact,
)
from .schemas import (
    ADJUDICATION_COLUMNS,
    EXPECTED_CASE_COUNT,
    EXPECTED_UNIT_COUNT,
    IMPLEMENTATION_REVISION,
    ReviewFreezeError,
)


def freeze_reviews(
    *,
    cohort_dir: Path,
    reviewer_a_completed: Path,
    reviewer_a_provenance: Path,
    reviewer_b_completed: Path,
    reviewer_b_provenance: Path,
    output_dir: Path,
) -> Path:
    output = safe_path(output_dir, "review-freeze output directory")
    if output.exists():
        raise ReviewFreezeError("review-freeze output directory already exists")

    # Public source and review checks are deliberately complete before the
    # private 3B1 mapping is opened.
    cohort = load_public_cohort(cohort_dir)
    review_a = validate_review_return(
        cohort,
        reviewer="A",
        completed_path=reviewer_a_completed,
        provenance_path=reviewer_a_provenance,
    )
    review_b = validate_review_return(
        cohort,
        reviewer="B",
        completed_path=reviewer_b_completed,
        provenance_path=reviewer_b_provenance,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-freeze-", dir=output.parent))
    try:
        reviewer_a_csv_sha = write_artifact(
            staging / "reviewer_A_frozen.csv", review_a.completed_bytes
        )
        reviewer_a_provenance_sha = write_artifact(
            staging / "reviewer_A_provenance.json", review_a.provenance_bytes
        )
        reviewer_b_csv_sha = write_artifact(
            staging / "reviewer_B_frozen.csv", review_b.completed_bytes
        )
        reviewer_b_provenance_sha = write_artifact(
            staging / "reviewer_B_provenance.json", review_b.provenance_bytes
        )

        private_mapping, private_mapping_lock = load_private_mapping(
            cohort, review_a, review_b
        )
        aligned = align_reviews(cohort, review_a, review_b, private_mapping)
        agreement = compute_agreement(aligned)
        disagreement_count = agreement.report[
            "exact_four_class_disagreement_count"
        ]

        source_lock = {
            "status": "PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "source_3b1_revision": cohort.build_report["implementation_revision"],
            "public_source_artifacts": cohort.public_source_locks,
            "private_review_mapping": private_mapping_lock,
            "reviewer_A_completed_input": {
                "path": str(review_a.completed_path),
                "sha256": review_a.completed_sha256,
            },
            "reviewer_A_provenance_input": {
                "path": str(review_a.provenance_path),
                "sha256": review_a.provenance_sha256,
            },
            "reviewer_B_completed_input": {
                "path": str(review_b.completed_path),
                "sha256": review_b.completed_sha256,
            },
            "reviewer_B_provenance_input": {
                "path": str(review_b.provenance_path),
                "sha256": review_b.provenance_sha256,
            },
            "private_mapping_opened_after_both_reviews_validated_and_frozen": True,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
        }
        source_lock_sha = write_artifact(
            staging / "review_source_lock.json", json_bytes(source_lock)
        )

        agreement_report = {
            "status": "INDEPENDENT_REVIEW_AGREEMENT_AUDIT_COMPLETE",
            "implementation_revision": IMPLEMENTATION_REVISION,
            **agreement.report,
            "agreement_threshold_applied": False,
            "confidence_weighting_applied": False,
            "selector_scores_accessed": False,
            "veracity_labels_accessed": False,
        }
        agreement_report_sha = write_artifact(
            staging / "agreement_report.json", json_bytes(agreement_report)
        )
        agreement_by_case_sha = write_artifact(
            staging / "agreement_by_case.csv",
            csv_bytes(
                (
                    "dataset",
                    "canonical_case_id",
                    "total_unit_count",
                    "agreement_count",
                    "disagreement_count",
                ),
                agreement.by_case_rows,
            ),
        )
        pre_adjudication_sha = write_artifact(
            staging / "review_resolution_pre_adjudication.jsonl",
            jsonl_bytes(agreement.ledger_rows),
        )
        private_agreement_mapping = {
            "status": "PRIVATE_AGREEMENT_MAPPING_FROZEN",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "rows": [
                {
                    "dataset": row.underlying.dataset,
                    "canonical_case_id": row.underlying.canonical_case_id,
                    "original_case_id": row.underlying.original_case_id,
                    "unit_id": row.underlying.unit_id,
                    "unit_type": row.underlying.unit_type,
                    "modality": row.underlying.modality,
                    "original_candidate_position": row.underlying.original_candidate_position,
                    "claim": row.claim,
                    "candidate_text": row.candidate_text,
                    "reviewer_a_review_case_id": row.reviewer_a.review_case_id,
                    "reviewer_a_review_unit_id": row.reviewer_a.review_unit_id,
                    "reviewer_b_review_case_id": row.reviewer_b.review_case_id,
                    "reviewer_b_review_unit_id": row.reviewer_b.review_unit_id,
                }
                for row in agreement.aligned_rows
            ],
        }
        private_agreement_mapping_sha = write_artifact(
            staging / "private_agreement_mapping.json",
            json_bytes(private_agreement_mapping),
        )

        adjudication_packet_sha = None
        private_adjudication_mapping_sha = None
        if disagreement_count:
            packet = build_adjudication_packet(agreement.aligned_rows)
            packet_dir = staging / "adjudication_packet"
            packet_dir.mkdir()
            instructions = ADJUDICATOR_INSTRUCTIONS.encode("utf-8")
            template = csv_bytes(ADJUDICATION_COLUMNS, packet.rows)
            manifest = adjudication_manifest(packet, template)
            (packet_dir / "README_ADJUDICATOR.md").write_bytes(instructions)
            (packet_dir / "ADJUDICATION_MANIFEST.json").write_bytes(
                json_bytes(manifest)
            )
            (packet_dir / "adjudication_template.csv").write_bytes(template)
            packet_payload = b"".join(
                hashlib_payload
                for hashlib_payload in (
                    instructions,
                    json_bytes(manifest),
                    template,
                )
            )
            # Composite packet identity is private provenance only; public files
            # remain exactly the three specified packet files.
            adjudication_packet_sha = sha256_bytes(packet_payload)
            private_adjudication_mapping = {
                "status": "PRIVATE_ADJUDICATION_MAPPING_FROZEN",
                "implementation_revision": IMPLEMENTATION_REVISION,
                "adjudication_salt": "step2.6r-3b2-adjudication-v1",
                "rows": list(packet.mapping),
            }
            private_adjudication_mapping_sha = write_artifact(
                staging / "private_adjudication_mapping.json",
                json_bytes(private_adjudication_mapping),
            )

        freeze_report = {
            "status": "INDEPENDENT_REVIEW_FREEZE_AND_AGREEMENT_PASS",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "frozen_case_count": EXPECTED_CASE_COUNT,
            "frozen_unit_count": EXPECTED_UNIT_COUNT,
            "reviewer_a_frozen_csv_sha256": reviewer_a_csv_sha,
            "reviewer_a_provenance_sha256": reviewer_a_provenance_sha,
            "reviewer_b_frozen_csv_sha256": reviewer_b_csv_sha,
            "reviewer_b_provenance_sha256": reviewer_b_provenance_sha,
            "review_source_lock_sha256": source_lock_sha,
            "agreement_report_sha256": agreement_report_sha,
            "agreement_by_case_sha256": agreement_by_case_sha,
            "pre_adjudication_ledger_sha256": pre_adjudication_sha,
            "private_agreement_mapping_sha256": private_agreement_mapping_sha,
            "disagreement_count": disagreement_count,
            "adjudication_required": bool(disagreement_count),
            "adjudication_packet_composite_sha256": adjudication_packet_sha,
            "private_adjudication_mapping_sha256": private_adjudication_mapping_sha,
            "selector_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
            "step_3b3_executed": False,
        }
        write_artifact(
            staging / "review_freeze_report.json", json_bytes(freeze_report)
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze two independent reviews and audit agreement."
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--reviewer-a-completed", type=Path, required=True)
    parser.add_argument("--reviewer-a-provenance", type=Path, required=True)
    parser.add_argument("--reviewer-b-completed", type=Path, required=True)
    parser.add_argument("--reviewer-b-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = freeze_reviews(
            cohort_dir=args.cohort_dir,
            reviewer_a_completed=args.reviewer_a_completed,
            reviewer_a_provenance=args.reviewer_a_provenance,
            reviewer_b_completed=args.reviewer_b_completed,
            reviewer_b_provenance=args.reviewer_b_provenance,
            output_dir=args.output_dir,
        )
    except ReviewFreezeError as exc:
        print(f"review freeze failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
