"""Public-safe, JSON-friendly job state and snapshot contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class JobState(str, Enum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class JobFailureSnapshot:
    """The complete public failure surface for a failed web job."""

    code: str
    message: str
    incident_id: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "incident_id": self.incident_id,
        }


@dataclass(frozen=True)
class JobSnapshot:
    """An immutable snapshot that is safe to serialize at the HTTP boundary."""

    job_id: str
    state: JobState
    queue_position: Optional[int]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    expires_at: Optional[str]
    queue_elapsed_ms: int
    execution_elapsed_ms: int
    failure: Optional[JobFailureSnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "queue_position": self.queue_position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "expires_at": self.expires_at,
            "queue_elapsed_ms": self.queue_elapsed_ms,
            "execution_elapsed_ms": self.execution_elapsed_ms,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }
