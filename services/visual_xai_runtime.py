"""Lazy, persistent, bounded runtime coordination for supplemental Visual XAI.

This module controls when attribution runs.  It does not create or modify any
verification input, prediction, score, verdict, or evidence-sufficiency state.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from schemas import GroundedFrameReference, VisualAttributionArtifact
from schemas.visual_observation_snapshot import VisualObservationSnapshot
from services.cache_manager import safe_target, write_json
from services.siglip_visual_retriever import VisualFrame
from services.visual_xai_attributor import (
    VISUAL_XAI_MAX_CONCURRENCY_ENV,
    VisualXAIAttributor,
)


_LOGGER = logging.getLogger(__name__)
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_SAFE_UNIT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_TERMINAL_STATES = frozenset(("ready", "unavailable", "failed"))
VISUAL_XAI_POLL_AFTER_MS = 1500
VISUAL_XAI_FAILED_REASON = "attribution_failed"


class VisualXAIState(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VisualXAIStatusSnapshot:
    job_id: str
    unit_id: str
    state: VisualXAIState
    profile: str
    grid_rows: int
    grid_columns: int
    attribution_batch_size: int
    configuration_fingerprint: str
    source_frame_count: int
    observer_runtime_ms: float
    cache_hit: bool
    queue_wait_ms: Optional[float]
    compute_time_ms: Optional[float]
    heavy_scorer_batches: int
    unavailable_reason: Optional[str]
    artifacts: Tuple[VisualAttributionArtifact, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.state.value in _TERMINAL_STATES

    def to_public_dict(self) -> dict:
        return {
            "status": self.state.value,
            "profile": self.profile,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "attribution_batch_size": self.attribution_batch_size,
            "configuration_fingerprint": self.configuration_fingerprint,
            "source_frame_count": self.source_frame_count,
            "cache_hit": self.cache_hit,
            "queue_wait_ms": self.queue_wait_ms,
            "compute_time_ms": self.compute_time_ms,
            "heavy_scorer_batches": self.heavy_scorer_batches,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class _VisualXAIRequestRecord:
    job_id: str
    snapshot: VisualObservationSnapshot
    frames: Tuple[VisualFrame, ...]
    raw_generation: str
    request_fingerprint: str
    observer_runtime_ms: float
    state: VisualXAIState = VisualXAIState.NOT_REQUESTED
    artifacts: Tuple[VisualAttributionArtifact, ...] = ()
    cache_hit: bool = False
    enqueued_at: Optional[float] = None
    queue_wait_ms: Optional[float] = None
    compute_time_ms: Optional[float] = None
    heavy_scorer_batches: int = 0
    unavailable_reason: Optional[str] = None
    future: Optional[Future] = field(default=None, repr=False)


class PriorityGPUExecutionGate:
    """Serialize GPU work and give waiting authoritative jobs priority."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._active = False
        self._authoritative_waiters = 0

    @contextmanager
    def authoritative(self):
        with self._condition:
            self._authoritative_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._active)
                self._active = True
            finally:
                self._authoritative_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    @contextmanager
    def supplemental(self):
        with self._condition:
            self._condition.wait_for(
                lambda: not self._active and self._authoritative_waiters == 0
            )
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


def visual_xai_max_concurrency(
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    values = __import__("os").environ if environ is None else environ
    raw_value = values.get(VISUAL_XAI_MAX_CONCURRENCY_ENV, "1")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{VISUAL_XAI_MAX_CONCURRENCY_ENV} must be an integer from 1 to 4"
        ) from None
    if not 1 <= value <= 4:
        raise ValueError(
            f"{VISUAL_XAI_MAX_CONCURRENCY_ENV} must be an integer from 1 to 4"
        )
    return value


class VisualXAIRuntimeService:
    """Register cheap inputs now and compute one requested observation later."""

    def __init__(
        self,
        attributor: VisualXAIAttributor,
        runtime_cache_root: Path,
        *,
        max_concurrency: int = 1,
        clock=time.monotonic,
        execution_gate: Optional[PriorityGPUExecutionGate] = None,
    ) -> None:
        if not isinstance(attributor, VisualXAIAttributor):
            raise TypeError("attributor must be a VisualXAIAttributor")
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 4:
            raise ValueError("max_concurrency must be an integer from 1 to 4")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.attributor = attributor
        self.runtime_cache_root = Path(runtime_cache_root).resolve()
        self.state_root = (self.runtime_cache_root / "visual_xai_requests").resolve()
        if self.runtime_cache_root not in self.state_root.parents:
            raise ValueError("visual XAI state root escapes runtime cache")
        self.max_concurrency = max_concurrency
        self.execution_gate = execution_gate or PriorityGPUExecutionGate()
        self._clock = clock
        self._guard = threading.RLock()
        self._records: Dict[Tuple[str, str], _VisualXAIRequestRecord] = {}
        self._futures = set()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="visual-xai-worker",
        )
        self._shutdown = False

    @contextmanager
    def authoritative_execution(self):
        with self.execution_gate.authoritative():
            yield

    @staticmethod
    def _validate_identifiers(job_id: str, unit_id: str) -> None:
        if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
            raise ValueError("invalid visual XAI job identifier")
        if not isinstance(unit_id, str) or _SAFE_UNIT_ID.fullmatch(unit_id) is None:
            raise ValueError("invalid visual XAI unit identifier")

    @staticmethod
    def _snapshot_payload(snapshot: VisualObservationSnapshot) -> dict:
        return {
            "unit_id": snapshot.unit_id,
            "observation_text": snapshot.observation_text,
            "start_timestamp_seconds": snapshot.start_timestamp_seconds,
            "end_timestamp_seconds": snapshot.end_timestamp_seconds,
            "primary_frame_id": snapshot.primary_frame_id,
            "frame_references": [
                {
                    "frame_id": item.frame_id,
                    "frame_index": item.frame_index,
                    "timestamp_seconds": item.timestamp_seconds,
                    "frame_rank": item.frame_rank,
                    "retrieval_rank": item.retrieval_rank,
                    "image_sha256": item.image_sha256,
                }
                for item in snapshot.frame_references
            ],
            "source_index": snapshot.source_index,
            "extraction_method": snapshot.extraction_method,
            "observation_type": snapshot.observation_type,
            "frame_ids": list(snapshot.frame_ids),
            "evidence_refs": list(snapshot.evidence_refs),
            "siglip_model_id": snapshot.siglip_model_id,
            "siglip_revision": snapshot.siglip_revision,
            "qwen_model_id": snapshot.qwen_model_id,
            "qwen_revision": snapshot.qwen_revision,
            "prompt_policy": snapshot.prompt_policy,
            "raw_generation_sha256": snapshot.raw_generation_sha256,
            "recovery_mode": snapshot.recovery_mode,
            "retrieval_policy_id": snapshot.retrieval_policy_id,
            "observer_policy_id": snapshot.observer_policy_id,
        }

    @staticmethod
    def _snapshot_from_payload(payload: Mapping[str, object]) -> VisualObservationSnapshot:
        values = dict(payload)
        values["frame_references"] = tuple(
            GroundedFrameReference(**dict(item))
            for item in values["frame_references"]
        )
        values["frame_ids"] = tuple(values["frame_ids"])
        values["evidence_refs"] = tuple(values["evidence_refs"])
        return VisualObservationSnapshot(**values)

    def _relative_frame_path(self, path: Path) -> str:
        candidate = Path(path)
        if candidate.is_symlink():
            raise ValueError("visual XAI source frame must not be a symlink")
        resolved = candidate.resolve(strict=True)
        if self.runtime_cache_root not in resolved.parents or not resolved.is_file():
            raise ValueError("visual XAI source frame is outside runtime cache")
        return resolved.relative_to(self.runtime_cache_root).as_posix()

    def _frame_payload(self, frame: VisualFrame) -> dict:
        return {
            "frame_id": frame.frame_id,
            "relative_path": self._relative_frame_path(frame.frame_path),
            "frame_index": frame.frame_index,
            "timestamp_sec": frame.timestamp_sec,
            "frame_rank": frame.frame_rank,
            "image_sha256": frame.image_sha256,
            "retrieval_score": frame.retrieval_score,
            "retrieval_rank": frame.retrieval_rank,
        }

    def _frame_from_payload(self, payload: Mapping[str, object]) -> VisualFrame:
        values = dict(payload)
        relative_path = values.pop("relative_path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError("invalid persisted visual XAI frame path")
        candidate = (self.runtime_cache_root / relative_path).resolve(strict=True)
        if self.runtime_cache_root not in candidate.parents or not candidate.is_file():
            raise ValueError("persisted visual XAI frame escapes runtime cache")
        values["frame_path"] = candidate
        return VisualFrame(**values)

    def _request_fingerprint(
        self,
        snapshot: VisualObservationSnapshot,
        frames: Tuple[VisualFrame, ...],
        raw_generation: str,
    ) -> str:
        payload = {
            "snapshot": self._snapshot_payload(snapshot),
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "image_sha256": frame.image_sha256,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "retrieval_rank": frame.retrieval_rank,
                }
                for frame in frames
            ],
            "raw_generation_sha256": hashlib.sha256(
                raw_generation.encode("utf-8")
            ).hexdigest(),
            "configuration_fingerprint": (
                self.attributor.config.configuration_fingerprint
            ),
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def _manifest_path(self, job_id: str, unit_id: str) -> Path:
        self._validate_identifiers(job_id, unit_id)
        key = hashlib.sha256(f"{job_id}\0{unit_id}".encode("utf-8")).hexdigest()
        return safe_target(self.state_root, f"request_{key}", "manifest.json")

    def _record_payload(self, record: _VisualXAIRequestRecord) -> dict:
        return {
            "job_id": record.job_id,
            "unit_id": record.snapshot.unit_id,
            "request_fingerprint": record.request_fingerprint,
            "snapshot": self._snapshot_payload(record.snapshot),
            "frames": [self._frame_payload(frame) for frame in record.frames],
            "raw_generation": record.raw_generation,
            "observer_runtime_ms": record.observer_runtime_ms,
            "state": record.state.value,
            "cache_hit": record.cache_hit,
            "queue_wait_ms": record.queue_wait_ms,
            "compute_time_ms": record.compute_time_ms,
            "heavy_scorer_batches": record.heavy_scorer_batches,
            "unavailable_reason": record.unavailable_reason,
            "artifact_cache_keys": [item.cache_key for item in record.artifacts],
        }

    def _persist(self, record: _VisualXAIRequestRecord) -> None:
        write_json(
            self._manifest_path(record.job_id, record.snapshot.unit_id),
            self._record_payload(record),
        )

    def _load_record(
        self, job_id: str, unit_id: str
    ) -> Optional[_VisualXAIRequestRecord]:
        path = self._manifest_path(job_id, unit_id)
        if not path.is_file() or path.is_symlink():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["job_id"] != job_id or payload["unit_id"] != unit_id:
                return None
            snapshot = self._snapshot_from_payload(payload["snapshot"])
            frames = tuple(self._frame_from_payload(item) for item in payload["frames"])
            raw_generation = payload["raw_generation"]
            fingerprint = self._request_fingerprint(
                snapshot, frames, raw_generation
            )
            if payload["request_fingerprint"] != fingerprint:
                return None
            state = VisualXAIState(payload["state"])
            if state is VisualXAIState.PENDING:
                state = VisualXAIState.NOT_REQUESTED
            artifacts = ()
            cache_hit = False
            if state in {VisualXAIState.READY, VisualXAIState.UNAVAILABLE}:
                cached = self.attributor.cached_artifacts(snapshot, frames)
                if cached is None:
                    state = VisualXAIState.NOT_REQUESTED
                else:
                    artifacts = cached
                    cache_hit = True
            return _VisualXAIRequestRecord(
                job_id=job_id,
                snapshot=snapshot,
                frames=frames,
                raw_generation=raw_generation,
                request_fingerprint=fingerprint,
                observer_runtime_ms=float(payload["observer_runtime_ms"]),
                state=state,
                artifacts=artifacts,
                cache_hit=cache_hit,
                queue_wait_ms=payload.get("queue_wait_ms"),
                compute_time_ms=payload.get("compute_time_ms"),
                heavy_scorer_batches=int(payload.get("heavy_scorer_batches", 0)),
                unavailable_reason=payload.get("unavailable_reason"),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ):
            return None

    def register(
        self,
        job_id: str,
        visual_snapshots: Iterable[VisualObservationSnapshot],
        selected_frames: Iterable[VisualFrame],
        raw_generation: str,
        *,
        observer_runtime_ms: float = 0.0,
    ) -> Tuple[VisualXAIStatusSnapshot, ...]:
        snapshots = tuple(visual_snapshots)
        frames = tuple(selected_frames)
        if not all(isinstance(item, VisualObservationSnapshot) for item in snapshots):
            raise TypeError("visual_snapshots must contain VisualObservationSnapshot only")
        if not all(isinstance(item, VisualFrame) for item in frames):
            raise TypeError("selected_frames must contain VisualFrame only")
        if not isinstance(raw_generation, str) or not raw_generation:
            raise ValueError("raw_generation is required")
        if not isinstance(observer_runtime_ms, (int, float)) or not math.isfinite(
            float(observer_runtime_ms)
        ):
            raise ValueError("observer_runtime_ms must be finite")
        output = []
        with self._guard:
            for snapshot in snapshots:
                self._validate_identifiers(job_id, snapshot.unit_id)
                fingerprint = self._request_fingerprint(
                    snapshot, frames, raw_generation
                )
                key = (job_id, snapshot.unit_id)
                existing = self._records.get(key) or self._load_record(*key)
                if existing is not None and existing.request_fingerprint == fingerprint:
                    record = existing
                else:
                    record = _VisualXAIRequestRecord(
                        job_id=job_id,
                        snapshot=snapshot,
                        frames=frames,
                        raw_generation=raw_generation,
                        request_fingerprint=fingerprint,
                        observer_runtime_ms=float(observer_runtime_ms),
                    )
                self._records[key] = record
                self._persist(record)
                output.append(self._snapshot(record))
        return tuple(output)

    def _get_record(self, job_id: str, unit_id: str) -> Optional[_VisualXAIRequestRecord]:
        self._validate_identifiers(job_id, unit_id)
        key = (job_id, unit_id)
        record = self._records.get(key)
        if record is None:
            record = self._load_record(job_id, unit_id)
            if record is not None:
                self._records[key] = record
        return record

    def _snapshot(self, record: _VisualXAIRequestRecord) -> VisualXAIStatusSnapshot:
        config = self.attributor.config
        return VisualXAIStatusSnapshot(
            job_id=record.job_id,
            unit_id=record.snapshot.unit_id,
            state=record.state,
            profile=config.profile,
            grid_rows=config.grid_rows,
            grid_columns=config.grid_columns,
            attribution_batch_size=config.attribution_batch_size,
            configuration_fingerprint=config.configuration_fingerprint,
            source_frame_count=len(record.snapshot.frame_references),
            observer_runtime_ms=record.observer_runtime_ms,
            cache_hit=record.cache_hit,
            queue_wait_ms=record.queue_wait_ms,
            compute_time_ms=record.compute_time_ms,
            heavy_scorer_batches=record.heavy_scorer_batches,
            unavailable_reason=record.unavailable_reason,
            artifacts=record.artifacts,
        )

    def get_status(
        self, job_id: str, unit_id: str
    ) -> Optional[VisualXAIStatusSnapshot]:
        with self._guard:
            record = self._get_record(job_id, unit_id)
            return None if record is None else self._snapshot(record)

    def request(self, job_id: str, unit_id: str) -> VisualXAIStatusSnapshot:
        with self._guard:
            if self._shutdown:
                raise RuntimeError("visual XAI runtime is shutting down")
            record = self._get_record(job_id, unit_id)
            if record is None:
                raise KeyError(unit_id)
            _LOGGER.info("VISUAL_XAI_REQUEST unit=%s", unit_id)
            if record.state is not VisualXAIState.NOT_REQUESTED:
                return self._snapshot(record)
            cached = self.attributor.cached_artifacts(
                record.snapshot, record.frames
            )
            if cached is not None:
                record.artifacts = cached
                record.cache_hit = True
                record.state = (
                    VisualXAIState.READY
                    if all(item.status == "available" for item in cached)
                    else VisualXAIState.UNAVAILABLE
                )
                record.unavailable_reason = next(
                    (
                        item.unavailable_reason
                        for item in cached
                        if item.unavailable_reason is not None
                    ),
                    None,
                )
                record.heavy_scorer_batches = 0
                self._persist(record)
                _LOGGER.info("VISUAL_XAI_CACHE_HIT unit=%s", unit_id)
                return self._snapshot(record)

            _LOGGER.info("VISUAL_XAI_CACHE_MISS unit=%s", unit_id)
            record.state = VisualXAIState.PENDING
            record.enqueued_at = self._clock()
            record.future = self._executor.submit(self._compute, job_id, unit_id)
            self._futures.add(record.future)
            record.future.add_done_callback(self._future_done)
            self._persist(record)
            _LOGGER.info("VISUAL_XAI_QUEUED unit=%s", unit_id)
            return self._snapshot(record)

    def _future_done(self, future: Future) -> None:
        with self._guard:
            self._futures.discard(future)

    def _compute(self, job_id: str, unit_id: str) -> None:
        with self._guard:
            record = self._records[(job_id, unit_id)]
            enqueued_at = record.enqueued_at or self._clock()
        try:
            with self.execution_gate.supplemental():
                started = self._clock()
                queue_wait_ms = max(0.0, (started - enqueued_at) * 1000.0)
                _LOGGER.info("VISUAL_XAI_STARTED unit=%s", unit_id)
                artifacts = tuple(
                    self.attributor.attribute(
                        (record.snapshot,),
                        record.frames,
                        record.raw_generation,
                    )
                )
                finished = self._clock()
            state = (
                VisualXAIState.READY
                if artifacts and all(item.status == "available" for item in artifacts)
                else VisualXAIState.UNAVAILABLE
            )
            reason = next(
                (
                    item.unavailable_reason
                    for item in artifacts
                    if item.unavailable_reason is not None
                ),
                None,
            )
            with self._guard:
                record.state = state
                record.artifacts = artifacts
                record.cache_hit = False
                record.queue_wait_ms = queue_wait_ms
                record.compute_time_ms = max(0.0, (finished - started) * 1000.0)
                record.heavy_scorer_batches = sum(
                    item.heavy_scorer_batches for item in artifacts
                )
                record.unavailable_reason = reason
                self._persist(record)
                _LOGGER.info(
                    "VISUAL_XAI_%s unit=%s seconds=%.3f",
                    state.value.upper(),
                    unit_id,
                    record.compute_time_ms / 1000.0,
                )
        except Exception:
            with self._guard:
                record.state = VisualXAIState.FAILED
                record.artifacts = ()
                record.cache_hit = False
                record.unavailable_reason = VISUAL_XAI_FAILED_REASON
                record.compute_time_ms = None
                self._persist(record)

    def wait_for_terminal(
        self, job_id: str, unit_id: str, timeout: float = 5.0
    ) -> Optional[VisualXAIStatusSnapshot]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status(job_id, unit_id)
            if status is None or status.terminal:
                return status
            time.sleep(0.01)
        return self.get_status(job_id, unit_id)

    def shutdown(self, timeout: float = 30.0) -> bool:
        if not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._guard:
            self._shutdown = True
            futures = tuple(self._futures)
            for future in futures:
                future.cancel()
        _, unfinished = wait(futures, timeout=float(timeout))
        self._executor.shutdown(wait=False, cancel_futures=True)
        return not unfinished


__all__ = [
    "PriorityGPUExecutionGate",
    "VISUAL_XAI_POLL_AFTER_MS",
    "VisualXAIRuntimeService",
    "VisualXAIState",
    "VisualXAIStatusSnapshot",
    "visual_xai_max_concurrency",
]
