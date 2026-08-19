"""Invoke the external Phase4A bridge with a prepared RuntimeUnit payload."""

import argparse
import json
from pathlib import Path

from schemas import RuntimeUnit
from services.frozen_g1_runner import FrozenG1Runner, FrozenG1RunnerConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON containing session_id, claim, and units")
    parser.add_argument("--unirumor-root", required=True, type=Path)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--phase4a-infer", required=True, type=Path)
    parser.add_argument("--phase4a-config", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    config = FrozenG1RunnerConfig(
        unirumor_root=args.unirumor_root,
        python_executable=args.python_executable,
        phase4a_infer=args.phase4a_infer,
        phase4a_config=args.phase4a_config,
        device=args.device,
        timeout_seconds=args.timeout,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    result = FrozenG1Runner(config).run(
        session_id=str(payload["session_id"]),
        claim=str(payload["claim"]),
        units=[RuntimeUnit.from_dict(item) for item in payload.get("units", [])],
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
