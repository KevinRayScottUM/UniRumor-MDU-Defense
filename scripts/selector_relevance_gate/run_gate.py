"""DICC CLI for Step 2.6R-3 normalization and two-stage evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .evaluator import (
    EvaluationError,
    run_heldout_gate,
    run_invariance_smoke,
    verify_approved_invariance_report,
)
from .heldout_loader import (
    ReferenceInputError,
    load_heldout_references,
    load_phase4a_replay_requests,
    sha256_file,
)
from .phase4a_normalizer import (
    AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
    Phase4ANormalizationError,
    prepare_invariance_requests,
)
from .runtime import (
    DICCEvaluationRuntime,
    RuntimeIntegrationError,
    validate_training_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction invariance and held-out direct relevance without "
            "training or modifying Frozen G1."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-invariance-requests", action="store_true")
    mode.add_argument("--invariance-smoke", action="store_true")
    mode.add_argument("--heldout-gate", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--phase4a-config", type=Path)
    parser.add_argument("--neutral-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--historical-phase4a-artifact", type=Path)
    parser.add_argument("--historical-phase4a-sha256")
    parser.add_argument("--phase4a-replay-artifact", type=Path)
    parser.add_argument("--phase4a-replay-sha256")
    parser.add_argument("--phase4a-replay-manifest", type=Path)
    parser.add_argument("--phase4a-replay-manifest-sha256")
    parser.add_argument("--approved-invariance-smoke-report", type=Path)
    parser.add_argument("--heldout-reference-artifact", type=Path)
    parser.add_argument("--heldout-reference-sha256")
    return parser


def _require(value: object, option: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise EvaluationError(f"{option} is required for the selected stage")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require(args.output_dir, "--output-dir")
        if args.prepare_invariance_requests:
            _require(
                args.historical_phase4a_artifact,
                "--historical-phase4a-artifact",
            )
            _require(
                args.historical_phase4a_sha256,
                "--historical-phase4a-sha256",
            )
            forbidden = (
                args.project_root,
                args.phase4a_config,
                args.neutral_dir,
                args.training_dir,
                args.phase4a_replay_artifact,
                args.phase4a_replay_sha256,
                args.phase4a_replay_manifest,
                args.phase4a_replay_manifest_sha256,
                args.approved_invariance_smoke_report,
                args.heldout_reference_artifact,
                args.heldout_reference_sha256,
            )
            if any(value is not None for value in forbidden):
                raise EvaluationError(
                    "normalization mode must not receive model or evaluation inputs"
                )
            prepare_invariance_requests(
                source_artifact=args.historical_phase4a_artifact,
                source_sha256=args.historical_phase4a_sha256,
                output_dir=args.output_dir,
            )
            print(
                args.output_dir.expanduser().resolve()
                / "phase4a_invariance_request_manifest.json"
            )
            return 0

        for value, option in (
            (args.project_root, "--project-root"),
            (args.phase4a_config, "--phase4a-config"),
            (args.neutral_dir, "--neutral-dir"),
            (args.training_dir, "--training-dir"),
        ):
            _require(value, option)
        if (
            args.historical_phase4a_artifact is not None
            or args.historical_phase4a_sha256 is not None
        ):
            raise EvaluationError(
                "evaluation modes must consume normalized requests, not historical input"
            )
        training = validate_training_artifacts(args.training_dir, args.neutral_dir)
        if args.invariance_smoke:
            _require(args.phase4a_replay_artifact, "--phase4a-replay-artifact")
            _require(args.phase4a_replay_sha256, "--phase4a-replay-sha256")
            _require(args.phase4a_replay_manifest, "--phase4a-replay-manifest")
            _require(
                args.phase4a_replay_manifest_sha256,
                "--phase4a-replay-manifest-sha256",
            )
            if any(
                value is not None
                for value in (
                    args.approved_invariance_smoke_report,
                    args.heldout_reference_artifact,
                    args.heldout_reference_sha256,
                )
            ):
                raise EvaluationError("Stage A must not receive held-out inputs")
            replay_sha, manifest_sha, historical_source, requests = (
                load_phase4a_replay_requests(
                    args.phase4a_replay_artifact,
                    expected_sha256=args.phase4a_replay_sha256,
                    manifest_path=args.phase4a_replay_manifest,
                    manifest_expected_sha256=args.phase4a_replay_manifest_sha256,
                )
            )
            runtime = DICCEvaluationRuntime(
                project_root=args.project_root,
                phase4a_config_path=args.phase4a_config,
                training_artifacts=training,
                device=args.device,
            )
            report = run_invariance_smoke(
                requests=requests,
                phase4a_replay_sha256=replay_sha,
                phase4a_replay_manifest_sha256=manifest_sha,
                training_artifacts=training,
                runtime=runtime,
                output_dir=args.output_dir,
                immutable_input_hashes={
                    **training.immutable_file_hashes,
                    args.phase4a_replay_artifact.expanduser().resolve(): replay_sha,
                    args.phase4a_replay_manifest.expanduser().resolve(): manifest_sha,
                    historical_source: AUTHORITATIVE_HISTORICAL_PHASE4A_SHA256,
                },
            )
            report_name = "prediction_invariance_smoke_report.json"
        else:
            _require(
                args.approved_invariance_smoke_report,
                "--approved-invariance-smoke-report",
            )
            _require(args.heldout_reference_artifact, "--heldout-reference-artifact")
            _require(args.heldout_reference_sha256, "--heldout-reference-sha256")
            if any(
                value is not None
                for value in (
                    args.phase4a_replay_artifact,
                    args.phase4a_replay_sha256,
                    args.phase4a_replay_manifest,
                    args.phase4a_replay_manifest_sha256,
                )
            ):
                raise EvaluationError("Stage B must not replay the Stage-A artifact")
            # Approval is verified before the held-out reference artifact is opened.
            verify_approved_invariance_report(
                args.approved_invariance_smoke_report, training
            )
            heldout_sha, references = load_heldout_references(
                args.heldout_reference_artifact,
                expected_sha256=args.heldout_reference_sha256,
            )
            runtime = DICCEvaluationRuntime(
                project_root=args.project_root,
                phase4a_config_path=args.phase4a_config,
                training_artifacts=training,
                device=args.device,
            )
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256=heldout_sha,
                approved_invariance_smoke_path=args.approved_invariance_smoke_report,
                training_artifacts=training,
                runtime=runtime,
                output_dir=args.output_dir,
                immutable_input_hashes={
                    **training.immutable_file_hashes,
                    args.approved_invariance_smoke_report.expanduser().resolve(): sha256_file(
                        args.approved_invariance_smoke_report.expanduser().resolve()
                    ),
                    args.heldout_reference_artifact.expanduser().resolve(): heldout_sha,
                    **{
                        Path(item.source_audit_artifact_path): str(
                            item.source_audit_artifact_sha256
                        )
                        for item in references
                    },
                },
            )
            report_name = "heldout_relevance_gate_report.json"
    except (
        EvaluationError,
        ReferenceInputError,
        RuntimeIntegrationError,
        Phase4ANormalizationError,
    ) as exc:
        print(f"selector relevance gate failed: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir.expanduser().resolve() / report_name)
    return (
        0
        if report["status"]
        in {
            "PREDICTION_INVARIANCE_SMOKE_PASS",
            "HELDOUT_RELEVANCE_AND_INVARIANCE_PASS",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
