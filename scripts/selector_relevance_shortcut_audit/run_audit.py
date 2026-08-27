"""CLI for the Step 2.6R-1C read-only shortcut audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .audit import AuditInputError, run_shortcut_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the locked calibration claim templates and actual Frozen G1 "
            "source-level pair-encoding contract without model execution."
        )
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_shortcut_audit(
            project_root=args.project_root,
            calibration_dir=args.calibration_dir,
            output_dir=args.output_dir,
        )
    except AuditInputError as exc:
        print(f"shortcut audit failed: {exc}", file=sys.stderr)
        return 2
    print(report["shortcut_risk_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
