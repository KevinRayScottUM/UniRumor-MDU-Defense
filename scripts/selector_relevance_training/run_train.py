"""DICC CLI for Step 2.6R-2 selector-only calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .trainer import SelectorTrainingError, run_selector_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the Frozen-G1 direct-relevance selection head on the "
            "closed modality-neutral calibration artifact."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--phase4a-config", required=True, type=Path)
    parser.add_argument("--neutral-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--approved-smoke-report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_mode = "full" if args.full else "smoke"
    if run_mode == "full" and args.approved_smoke_report is None:
        print("selector training failed: --full requires --approved-smoke-report", file=sys.stderr)
        return 2
    if run_mode == "smoke" and args.approved_smoke_report is not None:
        print("selector training failed: smoke must not consume an approval report", file=sys.stderr)
        return 2
    try:
        from .dicc_backend import DICCTorchBackend

        report = run_selector_calibration(
            source_dir=args.neutral_dir,
            output_dir=args.output_dir,
            run_mode=run_mode,
            backend_factory=lambda: DICCTorchBackend(
                project_root=args.project_root,
                phase4a_config_path=args.phase4a_config,
                device=args.device,
            ),
            approved_smoke_report=args.approved_smoke_report,
        )
    except SelectorTrainingError as exc:
        print(f"selector training failed: {exc}", file=sys.stderr)
        return 2
    report_name = "smoke_report.json" if run_mode == "smoke" else "training_report.json"
    print(args.output_dir.expanduser().resolve() / report_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
