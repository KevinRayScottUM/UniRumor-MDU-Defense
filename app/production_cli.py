"""Command-line entry point for the public-safe production runtime."""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TextIO

from services.production_execution import (
    ProductionExecutionOutcome,
    ProductionExecutionService,
    ProductionExecutionStatus,
)


EXIT_SUCCESS = 0
EXIT_EXECUTION_FAILURE = 1
EXIT_CLI_ERROR = 2
INITIALIZATION_FAILURE_MESSAGE = "Production CLI initialization failed."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the UniRumor production video-verification runtime."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="production runtime JSON config path",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="caller-supplied runtime session ID",
    )
    parser.add_argument(
        "--claim",
        required=True,
        help="exact focal claim string",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="source video path",
    )
    return parser


def _write_cli_error(stderr: TextIO) -> int:
    stderr.write(f"{INITIALIZATION_FAILURE_MESSAGE}\n")
    return EXIT_CLI_ERROR


def run_cli(
    args: Optional[Sequence[str]] = None,
    *,
    service_factory: Optional[Callable[[Path], Any]] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    parsed = build_parser().parse_args(args)
    output = sys.stdout if stdout is None else stdout
    error_output = sys.stderr if stderr is None else stderr
    factory = (
        ProductionExecutionService.from_json
        if service_factory is None
        else service_factory
    )

    try:
        service = factory(Path(parsed.config))
    except Exception:
        return _write_cli_error(error_output)

    try:
        outcome = service.execute(
            parsed.session_id,
            parsed.claim,
            parsed.video,
        )
    except Exception:
        return _write_cli_error(error_output)

    if not isinstance(outcome, ProductionExecutionOutcome):
        return _write_cli_error(error_output)

    try:
        outcome_json = outcome.to_json()
    except Exception:
        return _write_cli_error(error_output)

    output.write(f"{outcome_json}\n")
    if outcome.status is ProductionExecutionStatus.SUCCESS:
        return EXIT_SUCCESS
    return EXIT_EXECUTION_FAILURE


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
