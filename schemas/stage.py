"""Pipeline stage names, states, and transition rules."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class StageName(str, Enum):
    REQUEST = "request"
    SESSION = "session"
    ASR = "asr"
    OCR = "ocr"
    VISUAL_RETRIEVAL = "visual_retrieval"
    VLM = "vlm"
    UNIT_POOL = "runtime_unit_pool"
    MOCK_G1 = "mock_g1"
    RESULT = "result"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PipelineStage:
    name: StageName
    sequence: int
    status: StageStatus = StageStatus.PENDING
    runtime_ms: float = 0.0
    detail: str = ""

    def transition(self, new_status: StageStatus) -> None:
        allowed = {
            StageStatus.PENDING: {StageStatus.RUNNING, StageStatus.SKIPPED},
            StageStatus.RUNNING: {StageStatus.COMPLETED, StageStatus.FAILED},
        }
        if new_status not in allowed.get(self.status, set()):
            raise ValueError(f"invalid stage transition: {self.status.value} -> {new_status.value}")
        self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.value,
            "sequence": self.sequence,
            "status": self.status.value,
            "runtime_ms": self.runtime_ms,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStage":
        return cls(
            name=StageName(data["name"]),
            sequence=int(data["sequence"]),
            status=StageStatus(data["status"]),
            runtime_ms=float(data.get("runtime_ms", 0.0)),
            detail=str(data.get("detail", "")),
        )
