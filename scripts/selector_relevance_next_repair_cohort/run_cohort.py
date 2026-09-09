"""Explicit preflight/build CLI. No model, scoring, tuning or override flags."""

import argparse
import json
import sys
from pathlib import Path

from .cohort_builder import BLOCKED_STATUS, build, preflight
from .schemas import Inputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--build-cohort", action="store_true")
    for name in ("project-root", "phase4a-config", "source-3b1-cohort-dir", "step3b3-closure-dir",
                 "neutral-dir", "stage-a-invariance-report", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--approved-preflight-report", type=Path)
    parser.add_argument("--future-exclusion-manifest", type=Path, action="append", default=[])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.build_cohort and args.approved_preflight_report is None:
        parser.error("--build-cohort requires --approved-preflight-report")
    if args.preflight and args.approved_preflight_report is not None:
        parser.error("--approved-preflight-report is build-only")
    inputs = Inputs(args.project_root, args.phase4a_config, args.source_3b1_cohort_dir,
                    args.step3b3_closure_dir, args.neutral_dir, args.stage_a_invariance_report,
                    tuple(args.future_exclusion_manifest))
    try:
        report = (preflight(inputs, args.output_dir) if args.preflight else
                  build(inputs, args.output_dir, args.approved_preflight_report))
    except Exception as exc:
        print(json.dumps({"status": "NEXT_REPAIR_COHORT_INVALID", "error_type": type(exc).__name__,
                          "detail": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))
    return 1 if report["status"] == BLOCKED_STATUS else 0


if __name__ == "__main__":
    raise SystemExit(main())
