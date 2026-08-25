"""Command-line entry point for the controlled Frozen G1 selector audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audit import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen selector-fidelity probe suite on DICC.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        required=True,
        help="Public CPAC production result JSON containing g1_exposure_units.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Existing production runtime configuration for the external Frozen G1 CLI.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("probe_definitions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selector_fidelity_audit"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_audit(
        candidate_pool_path=args.candidate_pool,
        runtime_config_path=args.runtime_config,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(metrics["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
