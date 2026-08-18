"""Provenance carried by every runtime unit."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UnitProvenance:
    source_uri: Optional[str] = None
    source_index: Optional[int] = None
    extraction_method: str = "provided"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "source_index": self.source_index,
            "extraction_method": self.extraction_method,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnitProvenance":
        return cls(
            source_uri=data.get("source_uri"),
            source_index=data.get("source_index"),
            extraction_method=str(data.get("extraction_method", "provided")),
            details=dict(data.get("details") or {}),
        )
