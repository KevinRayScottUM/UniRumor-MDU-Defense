"""CLI for the Step 2.6R-3B1 cohort build only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .cohort_builder import IndependentAuditBuildError, build_cohort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the score-blind Train-derived relevance audit cohort."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase3a-train-lock-report", type=Path, required=True)
    parser.add_argument("--phase3a-train-lock-report-sha256", required=True)
    parser.add_argument("--phase4a-config", type=Path, required=True)
    parser.add_argument("--phase4a-config-sha256", required=True)
    parser.add_argument("--neutral-dir", type=Path, required=True)
    parser.add_argument("--stage-a-report", type=Path, required=True)
    parser.add_argument("--stage-a-report-sha256", required=True)
    parser.add_argument("--stage-a-replay", type=Path, required=True)
    parser.add_argument("--stage-a-replay-sha256", required=True)
    parser.add_argument("--stage-a-replay-manifest", type=Path, required=True)
    parser.add_argument("--stage-a-replay-manifest-sha256", required=True)
    parser.add_argument(
        "--additional-exclusion-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_cohort(
            project_root=args.project_root,
            phase3a_train_lock_report=args.phase3a_train_lock_report,
            phase3a_train_lock_report_sha256=args.phase3a_train_lock_report_sha256,
            phase4a_config_path=args.phase4a_config,
            phase4a_config_sha256=args.phase4a_config_sha256,
            neutral_dir=args.neutral_dir,
            stage_a_report_path=args.stage_a_report,
            stage_a_report_sha256=args.stage_a_report_sha256,
            stage_a_replay_path=args.stage_a_replay,
            stage_a_replay_sha256=args.stage_a_replay_sha256,
            stage_a_replay_manifest_path=args.stage_a_replay_manifest,
            stage_a_replay_manifest_sha256=args.stage_a_replay_manifest_sha256,
            additional_exclusion_manifests=args.additional_exclusion_manifest,
            output_dir=args.output_dir,
        )
    except IndependentAuditBuildError as exc:
        print(f"independent audit build failed: {exc}", file=sys.stderr)
        return 2
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
