"""CLI for the DICC-only direct-relevance dataset build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .dataset_builder import DatasetBuildError, build_calibration_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build score-blind Train-only direct-relevance calibration data "
            "using the actual Frozen G1 request-normalization policy."
        )
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--phase3a-train-lock-report", required=True, type=Path)
    parser.add_argument("--phase4a-config", required=True, type=Path)
    parser.add_argument("--step25b-selected-manifest", required=True, type=Path)
    parser.add_argument("--heldout-case", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_calibration_dataset(
            project_root=args.project_root,
            phase3a_train_lock_report=args.phase3a_train_lock_report,
            phase4a_config_path=args.phase4a_config,
            step25b_selected_manifest=args.step25b_selected_manifest,
            heldout_cases=args.heldout_case,
            output_dir=args.output_dir,
        )
    except DatasetBuildError as exc:
        print(f"dataset build failed: {exc}", file=sys.stderr)
        return 2
    print(result.output_dir / "build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
