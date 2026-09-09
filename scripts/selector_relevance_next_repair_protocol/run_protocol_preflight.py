"""Read the static protocol only and print an honest protocol-only result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .protocol import PREREGISTRATION_PATH, protocol_preflight
from .schemas import ProtocolError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Step 3C1 preregistration only.")
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = protocol_preflight(args.preregistration)
    except ProtocolError as exc:
        print(f"next-repair protocol preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
