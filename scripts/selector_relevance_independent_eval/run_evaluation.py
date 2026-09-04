"""CLI for score-free preflight and the separate 3B3 one-shot evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .evaluator import run_one_shot_evaluation, run_preflight
from .schemas import IndependentEvaluationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step 2.6R-3B3 preflight or one-shot selector evaluation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--one-shot-evaluate", action="store_true")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--final-gold-dir", type=Path, required=True)
    parser.add_argument("--stage-a-invariance-report", type=Path, required=True)
    parser.add_argument("--phase4a-config", type=Path, required=True)
    parser.add_argument("--neutral-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--approved-preflight-report", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        common = {
            "cohort_dir": args.cohort_dir,
            "final_gold_dir": args.final_gold_dir,
            "stage_a_invariance_report": args.stage_a_invariance_report,
            "project_root": args.project_root,
            "phase4a_config": args.phase4a_config,
            "neutral_dir": args.neutral_dir,
            "training_dir": args.training_dir,
            "output_dir": args.output_dir,
        }
        if args.preflight:
            if args.approved_preflight_report is not None or args.device is not None:
                raise IndependentEvaluationError(
                    "preflight must not receive evaluation-only arguments"
                )
            report = run_preflight(**common)
            report_path = args.output_dir.expanduser().resolve() / "one_shot_preflight_report.json"
            print(report_path)
            return 0

        if args.approved_preflight_report is None:
            raise IndependentEvaluationError(
                "--approved-preflight-report is required for one-shot evaluation"
            )
        if not isinstance(args.device, str) or not args.device.strip():
            raise IndependentEvaluationError(
                "--device is required for one-shot evaluation"
            )
        report = run_one_shot_evaluation(
            **common,
            approved_preflight_report=args.approved_preflight_report,
            device=args.device,
        )
        report_path = args.output_dir.expanduser().resolve() / "one_shot_evaluation_report.json"
        print(report_path)
        return 0 if report["repair_verification_pass"] is True else 1
    except (IndependentEvaluationError, OSError) as exc:
        print(f"independent selector evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
