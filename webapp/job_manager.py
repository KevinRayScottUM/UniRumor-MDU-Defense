"""Thread-safe bounded in-memory job manager with one execution lane."""

from __future__ import annotations

import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple, Union

from services.production_execution import (
    ProductionExecutionOutcome,
    ProductionExecutionStatus,
)

from .job_types import JobFailureSnapshot, JobSnapshot, JobState


DEFAULT_MAX_QUEUED_JOBS = 3
DEFAULT_TERMINAL_RETENTION = timedelta(minutes=60)
DEFAULT_EXPIRED_TOMBSTONE_DURATION = timedelta(minutes=10)
WEB_WORKER_FAILURE_CODE = "web_worker_failed"
WEB_WORKER_FAILURE_MESSAGE = "Verification could not be completed."


class JobManagerError(RuntimeError):
    """Base class for stable internal job-manager failures."""


class JobManagerNotAcceptingError(JobManagerError):
    """Raised when lifecycle state forbids new admission."""


class QueueFullError(JobManagerError):
    """Raised when all bounded waiting slots are reserved or occupied."""


class ReservationError(JobManagerError):
    """Raised when a reservation is foreign, released, or already consumed."""


class IllegalJobTransitionError(JobManagerError):
    """Raised when an internal caller attempts a forbidden state transition."""


_ALLOWED_TRANSITIONS = {
    JobState.ACCEPTED: frozenset((JobState.QUEUED,)),
    JobState.QUEUED: frozenset((JobState.RUNNING,)),
    JobState.RUNNING: frozenset((JobState.COMPLETED, JobState.FAILED)),
    JobState.COMPLETED: frozenset((JobState.EXPIRED,)),
    JobState.FAILED: frozenset((JobState.EXPIRED,)),
    JobState.EXPIRED: frozenset(),
}


def _duration_seconds(value: Union[timedelta, int, float], name: str) -> float:
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a timedelta or number of seconds")
    else:
        seconds = float(value)
    if seconds < 0:
        raise ValueError(f"{name} must be nonnegative")
    return seconds


def _utc_text(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _elapsed_ms(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() * 1000))


@dataclass
class _JobRecord:
    job_id: str
    claim: Optional[str]
    video_path: Optional[Path]
    state: JobState
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tombstone_remove_at: Optional[datetime] = None
    outcome: Optional[ProductionExecutionOutcome] = None
    failure: Optional[JobFailureSnapshot] = None
    state_history: List[JobState] = field(default_factory=list)


@dataclass(frozen=True)
class _ExecutionDisposition:
    outcome: Optional[ProductionExecutionOutcome]
    failure: Optional[JobFailureSnapshot]


class CapacityReservation:
    """A private, single-use claim on one future waiting-queue slot."""

    def __init__(self, manager: "JobManager") -> None:
        self._manager = manager
        self._state = "active"

    @property
    def active(self) -> bool:
        with self._manager._condition:
            return self._state == "active"

    def release(self) -> None:
        self._manager._release_reservation(self)

    def __enter__(self) -> "CapacityReservation":
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.release()


class JobManager:
    """Own bounded admission, exact state transitions, and one worker thread."""

    def __init__(
        self,
        execution_service,
        *,
        max_queued_jobs: int = DEFAULT_MAX_QUEUED_JOBS,
        terminal_retention: Union[
            timedelta, int, float
        ] = DEFAULT_TERMINAL_RETENTION,
        expired_tombstone_duration: Union[
            timedelta, int, float
        ] = DEFAULT_EXPIRED_TOMBSTONE_DURATION,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not callable(getattr(execution_service, "execute", None)):
            raise TypeError("execution_service must provide a callable execute method")
        if isinstance(max_queued_jobs, bool) or not isinstance(max_queued_jobs, int):
            raise TypeError("max_queued_jobs must be an integer")
        if max_queued_jobs <= 0:
            raise ValueError("max_queued_jobs must be positive")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")

        self._execution_service = execution_service
        self._max_queued_jobs = max_queued_jobs
        self._terminal_retention_seconds = _duration_seconds(
            terminal_retention,
            "terminal_retention",
        )
        self._tombstone_seconds = _duration_seconds(
            expired_tombstone_duration,
            "expired_tombstone_duration",
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        self._condition = threading.Condition(threading.RLock())
        self._jobs: Dict[str, _JobRecord] = {}
        self._queue: Deque[str] = deque()
        self._reservations: Set[CapacityReservation] = set()
        self._worker: Optional[threading.Thread] = None
        self._worker_stopped = True
        self._started = False
        self._draining = False
        self._running_job_id: Optional[str] = None

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    @property
    def started(self) -> bool:
        with self._condition:
            return self._started

    @property
    def accepting_jobs(self) -> bool:
        with self._condition:
            return self._can_accept_locked() and self._capacity_available_locked()

    @property
    def worker_alive(self) -> bool:
        with self._condition:
            return self._worker_alive_locked()

    @property
    def queued_count(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def reservation_count(self) -> int:
        with self._condition:
            return len(self._reservations)

    @property
    def running_job_id(self) -> Optional[str]:
        with self._condition:
            return self._running_job_id

    @property
    def max_queued_jobs(self) -> int:
        return self._max_queued_jobs

    def _worker_alive_locked(self) -> bool:
        return (
            self._worker is not None
            and not self._worker_stopped
            and self._worker.is_alive()
        )

    def _can_accept_locked(self) -> bool:
        return self._started and not self._draining and self._worker_alive_locked()

    def _capacity_available_locked(self) -> bool:
        return len(self._queue) + len(self._reservations) < self._max_queued_jobs

    def start(self) -> bool:
        """Start the sole worker, returning whether a worker was newly created."""

        with self._condition:
            if self._started:
                return False
            if self._draining:
                raise JobManagerNotAcceptingError("job manager is draining")
            self._started = True
            self._worker_stopped = False
            self._worker = threading.Thread(
                target=self._worker_main,
                name="web-job-worker",
                daemon=True,
            )
            self._worker.start()
            return True

    def reserve_capacity(self) -> CapacityReservation:
        with self._condition:
            if not self._can_accept_locked():
                raise JobManagerNotAcceptingError("job manager is not accepting jobs")
            if not self._capacity_available_locked():
                raise QueueFullError("waiting job capacity is full")
            reservation = CapacityReservation(self)
            self._reservations.add(reservation)
            return reservation

    def _release_reservation(self, reservation: CapacityReservation) -> None:
        with self._condition:
            if reservation._manager is not self:
                raise ReservationError("reservation belongs to a different manager")
            if reservation._state != "active":
                return
            reservation._state = "released"
            self._reservations.discard(reservation)
            self._condition.notify_all()

    def _consume_reservation_locked(
        self,
        reservation: CapacityReservation,
    ) -> None:
        if not isinstance(reservation, CapacityReservation):
            raise ReservationError("a capacity reservation is required")
        if reservation._manager is not self:
            raise ReservationError("reservation belongs to a different manager")
        if reservation._state != "active" or reservation not in self._reservations:
            raise ReservationError("reservation is no longer available")
        if not self._can_accept_locked():
            self._release_reservation(reservation)
            raise JobManagerNotAcceptingError("job manager is not accepting jobs")
        reservation._state = "submitted"
        self._reservations.remove(reservation)

    def _new_job_id_locked(self) -> str:
        while True:
            job_id = "job_" + secrets.token_hex(16)
            if job_id not in self._jobs:
                return job_id

    def submit_reserved(
        self,
        reservation: CapacityReservation,
        *,
        claim: str,
        video_path: Union[str, Path],
    ) -> str:
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path")
        stored_path = Path(video_path)

        with self._condition:
            self._consume_reservation_locked(reservation)
            now = self._now()
            job_id = self._new_job_id_locked()
            record = _JobRecord(
                job_id=job_id,
                claim=claim,
                video_path=stored_path,
                state=JobState.ACCEPTED,
                created_at=now,
                state_history=[JobState.ACCEPTED],
            )
            self._jobs[job_id] = record
            self._transition_locked(record, JobState.QUEUED, now)
            self._queue.append(job_id)
            self._condition.notify_all()
            return job_id

    def _transition_locked(
        self,
        record: _JobRecord,
        new_state: JobState,
        now: datetime,
    ) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[record.state]:
            raise IllegalJobTransitionError(
                f"illegal job transition: {record.state.value} -> {new_state.value}"
            )

        record.state = new_state
        record.state_history.append(new_state)
        if new_state is JobState.RUNNING:
            record.started_at = now
        elif new_state in (JobState.COMPLETED, JobState.FAILED):
            record.finished_at = now
            record.expires_at = now + timedelta(
                seconds=self._terminal_retention_seconds
            )
        elif new_state is JobState.EXPIRED:
            record.outcome = None
            record.failure = None
            record.claim = None
            record.video_path = None
            tombstone_start = record.expires_at or now
            record.tombstone_remove_at = tombstone_start + timedelta(
                seconds=self._tombstone_seconds
            )

    @staticmethod
    def _safe_worker_failure() -> JobFailureSnapshot:
        return JobFailureSnapshot(
            code=WEB_WORKER_FAILURE_CODE,
            message=WEB_WORKER_FAILURE_MESSAGE,
            incident_id="incident_" + secrets.token_hex(16),
        )

    @staticmethod
    def _task06_failure(outcome: ProductionExecutionOutcome) -> JobFailureSnapshot:
        failure = outcome.failure
        if failure is None:
            return JobManager._safe_worker_failure()
        return JobFailureSnapshot(
            code=failure.code.value,
            message=failure.public_message,
            incident_id="incident_" + secrets.token_hex(16),
        )

    def _invoke_service(
        self,
        job_id: str,
        claim: str,
        video_path: Path,
    ) -> _ExecutionDisposition:
        """Invoke the closed Task06 boundary, catching ordinary failures only."""

        try:
            outcome = self._execution_service.execute(job_id, claim, video_path)
        except Exception:
            return _ExecutionDisposition(
                outcome=None,
                failure=self._safe_worker_failure(),
            )

        if not isinstance(outcome, ProductionExecutionOutcome):
            return _ExecutionDisposition(
                outcome=None,
                failure=self._safe_worker_failure(),
            )
        if outcome.status is ProductionExecutionStatus.SUCCESS:
            return _ExecutionDisposition(outcome=outcome, failure=None)
        if outcome.status is ProductionExecutionStatus.FAILURE:
            return _ExecutionDisposition(
                outcome=outcome,
                failure=self._task06_failure(outcome),
            )
        return _ExecutionDisposition(
            outcome=None,
            failure=self._safe_worker_failure(),
        )

    def _worker_main(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._draining:
                        self._condition.wait()
                    if self._draining:
                        return

                    job_id = self._queue.popleft()
                    record = self._jobs[job_id]
                    now = self._now()
                    self._transition_locked(record, JobState.RUNNING, now)
                    self._running_job_id = job_id
                    claim = record.claim
                    video_path = record.video_path
                    self._condition.notify_all()

                if claim is None or video_path is None:
                    disposition = _ExecutionDisposition(
                        outcome=None,
                        failure=self._safe_worker_failure(),
                    )
                else:
                    disposition = self._invoke_service(job_id, claim, video_path)

                with self._condition:
                    record = self._jobs[job_id]
                    now = self._now()
                    if disposition.failure is None:
                        record.outcome = disposition.outcome
                        self._transition_locked(record, JobState.COMPLETED, now)
                    else:
                        record.outcome = disposition.outcome
                        record.failure = disposition.failure
                        self._transition_locked(record, JobState.FAILED, now)
                    self._running_job_id = None
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._worker_stopped = True
                self._condition.notify_all()

    def _snapshot_locked(self, record: _JobRecord, now: datetime) -> JobSnapshot:
        queue_position = None
        if record.state is JobState.QUEUED:
            try:
                queue_position = self._queue.index(record.job_id) + 1
            except ValueError:
                raise JobManagerError("queued job is absent from the queue") from None

        queue_end = record.started_at or now
        execution_elapsed = 0
        if record.started_at is not None:
            execution_end = record.finished_at or now
            execution_elapsed = _elapsed_ms(record.started_at, execution_end)

        return JobSnapshot(
            job_id=record.job_id,
            state=record.state,
            queue_position=queue_position,
            created_at=_utc_text(record.created_at) or "",
            started_at=_utc_text(record.started_at),
            finished_at=_utc_text(record.finished_at),
            expires_at=_utc_text(record.expires_at),
            queue_elapsed_ms=_elapsed_ms(record.created_at, queue_end),
            execution_elapsed_ms=execution_elapsed,
            failure=record.failure if record.state is JobState.FAILED else None,
        )

    def get_snapshot(self, job_id: str) -> Optional[JobSnapshot]:
        with self._condition:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return self._snapshot_locked(record, self._now())

    def wait_for_state(
        self,
        job_id: str,
        states: Union[JobState, Set[JobState], Tuple[JobState, ...]],
        timeout: Optional[float] = None,
    ) -> Optional[JobSnapshot]:
        """Wait without polling until a job reaches one of the requested states."""

        desired = {states} if isinstance(states, JobState) else set(states)
        if not desired or any(not isinstance(state, JobState) for state in desired):
            raise TypeError("states must contain JobState values")

        with self._condition:
            matched = self._condition.wait_for(
                lambda: (
                    job_id not in self._jobs
                    or self._jobs[job_id].state in desired
                ),
                timeout=timeout,
            )
            if not matched or job_id not in self._jobs:
                return None
            return self._snapshot_locked(self._jobs[job_id], self._now())

    def _get_completed_outcome(
        self,
        job_id: str,
    ) -> Optional[ProductionExecutionOutcome]:
        """Internal result-boundary access; never returns failed job outcomes."""

        with self._condition:
            record = self._jobs.get(job_id)
            if record is None or record.state is not JobState.COMPLETED:
                return None
            return record.outcome

    def _get_state_history(self, job_id: str) -> Optional[Tuple[JobState, ...]]:
        with self._condition:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return tuple(record.state_history)

    def sweep_expired(self) -> int:
        """Expire terminal records and remove elapsed tombstones in memory only."""

        changed = 0
        with self._condition:
            now = self._now()
            for record in tuple(self._jobs.values()):
                if (
                    record.state in (JobState.COMPLETED, JobState.FAILED)
                    and record.expires_at is not None
                    and now >= record.expires_at
                ):
                    self._transition_locked(record, JobState.EXPIRED, now)
                    changed += 1

            for job_id, record in tuple(self._jobs.items()):
                if (
                    record.state is JobState.EXPIRED
                    and record.tombstone_remove_at is not None
                    and now >= record.tombstone_remove_at
                ):
                    del self._jobs[job_id]
                    changed += 1
            if changed:
                self._condition.notify_all()
        return changed

    def sweep(self) -> int:
        return self.sweep_expired()

    def shutdown(self, timeout: Optional[float] = 30.0) -> bool:
        """Begin draining and report whether the worker actually stopped."""

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be None or a number of seconds")
            if timeout < 0:
                raise ValueError("timeout must be nonnegative")

        with self._condition:
            self._draining = True
            for reservation in tuple(self._reservations):
                reservation._state = "released"
            self._reservations.clear()
            worker = self._worker
            self._condition.notify_all()

        if worker is None:
            return True
        if worker is threading.current_thread():
            return False
        worker.join(timeout=timeout)
        return not worker.is_alive()
