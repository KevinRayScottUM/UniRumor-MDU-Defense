"""Input contract for one verification request."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationRequest:
    claim: str
    request_id: Optional[str] = None
    media_path: Optional[str] = None
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    ocr_observations: List[Dict[str, Any]] = field(default_factory=list)
    visual_inputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "request_id": self.request_id,
            "media_path": self.media_path,
            "transcript_segments": [dict(item) for item in self.transcript_segments],
            "ocr_observations": [dict(item) for item in self.ocr_observations],
            "visual_inputs": [dict(item) for item in self.visual_inputs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationRequest":
        return cls(
            claim=str(data["claim"]),
            request_id=data.get("request_id"),
            media_path=data.get("media_path"),
            transcript_segments=[dict(item) for item in data.get("transcript_segments", [])],
            ocr_observations=[dict(item) for item in data.get("ocr_observations", [])],
            visual_inputs=[dict(item) for item in data.get("visual_inputs", [])],
            metadata=dict(data.get("metadata") or {}),
        )
