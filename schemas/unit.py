"""Runtime evidence-unit contract."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .provenance import UnitProvenance


class SourceType(str, Enum):
    TEXT = "text"
    TRANSCRIPT = "transcript"
    OCR = "ocr"
    VISUAL_OBSERVATION = "visual_observation"


@dataclass
class RuntimeUnit:
    unit_id: str
    source_type: SourceType
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    frame_id: Optional[str] = None
    frame_path: Optional[str] = None
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    producer: str = "unknown"
    provenance: UnitProvenance = field(default_factory=UnitProvenance)
    eligible_for_frozen_g1: bool = False
    selection_score: Optional[float] = None
    logits: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.source_type == SourceType.VISUAL_OBSERVATION and self.eligible_for_frozen_g1:
            raise ValueError("visual_observation units cannot be eligible for frozen G1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_type": self.source_type.value,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frame_id": self.frame_id,
            "frame_path": self.frame_path,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "confidence": self.confidence,
            "producer": self.producer,
            "provenance": self.provenance.to_dict(),
            "eligible_for_frozen_g1": self.eligible_for_frozen_g1,
            "selection_score": self.selection_score,
            "logits": dict(self.logits) if self.logits is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeUnit":
        return cls(
            unit_id=str(data["unit_id"]),
            source_type=SourceType(data["source_type"]),
            text=str(data.get("text", "")),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            frame_id=data.get("frame_id"),
            frame_path=data.get("frame_path"),
            bbox=list(data["bbox"]) if data.get("bbox") is not None else None,
            confidence=data.get("confidence"),
            producer=str(data.get("producer", "unknown")),
            provenance=UnitProvenance.from_dict(data.get("provenance") or {}),
            eligible_for_frozen_g1=bool(data.get("eligible_for_frozen_g1", False)),
            selection_score=data.get("selection_score"),
            logits=dict(data["logits"]) if data.get("logits") is not None else None,
        )
