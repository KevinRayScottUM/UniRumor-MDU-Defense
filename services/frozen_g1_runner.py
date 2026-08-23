"""Thin subprocess bridge to the external Frozen G1 Phase4A CLI."""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from adapters import build_phase4a_request, parse_phase4a_prediction
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    PipelineStage,
    RuntimeUnit,
    SourceType,
    VerificationResult,
)
from services.cache_manager import safe_target


@dataclass(frozen=True)
class FrozenG1RunnerConfig:
    unirumor_root: Path
    python_executable: str
    phase4a_infer: Path
    phase4a_config: Path
    device: str
    timeout_seconds: float
    cache_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "unirumor_root", Path(self.unirumor_root))
        object.__setattr__(self, "phase4a_infer", Path(self.phase4a_infer))
        object.__setattr__(self, "phase4a_config", Path(self.phase4a_config))
        object.__setattr__(self, "cache_root", Path(self.cache_root).resolve())
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())
        if self.cache_root == self.output_root:
            raise ValueError("Frozen G1 cache_root and output_root must be distinct")
        if not self.python_executable:
            raise ValueError("Frozen G1 python_executable is required")
        if not self.device:
            raise ValueError("Frozen G1 device is required")
        if self.timeout_seconds <= 0:
            raise ValueError("Frozen G1 timeout_seconds must be positive")


class FrozenG1Runner:
    def __init__(
        self,
        config: FrozenG1RunnerConfig,
        subprocess_run: Callable[..., object] = subprocess.run,
    ) -> None:
        self.config = config
        self._subprocess_run = subprocess_run

    def _paths(self, session_id: str) -> Sequence[Path]:
        return (
            safe_target(self.config.cache_root, "phase4a_requests", f"{session_id}.jsonl"),
            safe_target(self.config.output_root, "phase4a_predictions", f"{session_id}.jsonl"),
        )

    def _command(self, request_path: Path, prediction_path: Path) -> List[str]:
        return [
            self.config.python_executable,
            "-u",
            str(self.config.phase4a_infer),
            "--config",
            str(self.config.phase4a_config),
            "--project-root",
            str(self.config.unirumor_root),
            "--input",
            str(request_path),
            "--output",
            str(prediction_path),
            "--device",
            self.config.device,
        ]

    def run(
        self,
        session_id: str,
        claim: str,
        units: Iterable[RuntimeUnit],
        pipeline_stages: Optional[List[PipelineStage]] = None,
    ) -> VerificationResult:
        all_units = list(units)
        unit_ids = set()
        for unit in all_units:
            if unit.unit_id in unit_ids:
                raise ValueError(f"duplicate RuntimeUnit ID: {unit.unit_id!r}")
            unit_ids.add(unit.unit_id)
        request_payload = build_phase4a_request(session_id, claim, all_units)
        candidate_units = request_payload["candidate_units"]
        if not candidate_units:
            for unit in all_units:
                unit.selection_score = None
                unit.logits = None
            return VerificationResult(
                session_id=session_id,
                claim=claim,
                model_verdict=ModelVerdict.NOT_RUN,
                display_verdict=DisplayVerdict.NEI,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                sample_logits={},
                probabilities={},
                all_units=all_units,
                top_k_units=[],
                class_winners={},
                pipeline_stages=list(pipeline_stages or []),
            )

        request_path, prediction_path = self._paths(session_id)
        request_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(request_payload, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        prediction_path.unlink(missing_ok=True)
        started = time.perf_counter()
        try:
            completed = self._subprocess_run(
                self._command(request_path, prediction_path),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("external Phase4A inference timed out") from exc
        runtime_ms = (time.perf_counter() - started) * 1000.0
        if getattr(completed, "returncode", None) != 0:
            stderr = str(getattr(completed, "stderr", "")).strip()
            raise RuntimeError(f"external Phase4A inference failed: {stderr or 'no stderr'}")

        try:
            lines = [
                line
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except FileNotFoundError as exc:
            raise ValueError("external Phase4A prediction file was not created") from exc
        if len(lines) != 1:
            raise ValueError("external Phase4A prediction JSONL must contain exactly one record")
        try:
            prediction_payload = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError("external Phase4A prediction is not valid JSON") from exc

        candidate_ids = [item["unit_id"] for item in candidate_units]
        prediction = parse_phase4a_prediction(
            prediction_payload,
            candidate_ids,
            expected_case_id=session_id,
            expected_claim=claim,
        )

        units_by_id = {unit.unit_id: unit for unit in all_units}
        for unit in all_units:
            unit.selection_score = None
            unit.logits = None
        for output in prediction.unit_outputs:
            unit = units_by_id[output.unit_id]
            unit.selection_score = output.selection_score
            unit.logits = dict(output.veracity_logits)

        warnings = []
        if any(unit.source_type is SourceType.VISUAL_OBSERVATION for unit in all_units):
            warnings.append("supplemental visual units excluded from Frozen G1")
        if len(prediction.unit_outputs) < len(candidate_units):
            warnings.append("official Phase4A truncation occurred")

        model_verdict = ModelVerdict(prediction.prediction)
        return VerificationResult(
            session_id=session_id,
            claim=claim,
            model_verdict=model_verdict,
            display_verdict={
                ModelVerdict.FAKE: DisplayVerdict.FAKE,
                ModelVerdict.REAL: DisplayVerdict.REAL,
                ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
            }[model_verdict],
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits=dict(prediction.sample_logits),
            probabilities=dict(prediction.probabilities),
            all_units=all_units,
            top_k_units=[units_by_id[unit_id] for unit_id in prediction.top_k_unit_ids],
            class_winners=dict(prediction.class_winners),
            pipeline_stages=list(pipeline_stages or []),
            warnings=warnings,
            checkpoint_sha256=prediction.checkpoint_sha256,
            runtime_ms=runtime_ms,
        )
