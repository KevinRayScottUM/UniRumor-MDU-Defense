"""CLI for Step 2.6R-1D modality-neutral calibration revision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .build_neutral import NeutralBuildError, build_neutral_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the immutable modality-neutral selector calibration artifact "
            "from the closed Step 2.6R-1A v2 artifact."
        )
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_neutral_calibration(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
    except NeutralBuildError as exc:
        print(f"neutral calibration build failed: {exc}", file=sys.stderr)
        return 2
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
