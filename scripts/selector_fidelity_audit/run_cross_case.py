"""CLI for the DICC-only cross-case Frozen G1 selector audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .cross_case import DiscoveryRoot, run_cross_case_audit


def _discovery_root(value: str) -> DiscoveryRoot:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "discovery roots must use DATASET=/absolute/non-Test/path"
        )
    dataset, raw_path = value.split("=", 1)
    try:
        return DiscoveryRoot(dataset=dataset, path=Path(raw_path))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover non-Test public/training-derived pools score-blind and run "
            "the original Frozen G1 selector cross-case audit on DICC."
        )
    )
    parser.add_argument(
        "--discovery-root",
        action="append",
        type=_discovery_root,
        required=True,
        help="Repeatable DATASET=/absolute/path root; Validation/Test paths are rejected.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Existing DICC production runtime configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selector_fidelity_audit/cross_case"),
    )
    parser.add_argument("--target-case-count", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots: List[DiscoveryRoot] = args.discovery_root
    metrics = run_cross_case_audit(
        discovery_roots=roots,
        runtime_config_path=args.runtime_config,
        output_dir=args.output_dir,
        target_case_count=args.target_case_count,
    )
    print(metrics["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
