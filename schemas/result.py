"""Final verification-result contract."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .stage import PipelineStage
from .unit import RuntimeUnit


class ModelVerdict(str, Enum):
    FAKE = "fake"
    REAL = "real"
    NOT_RUN = "not_run"


class DisplayVerdict(str, Enum):
    FAKE = "Fake"
    REAL = "Real"
    NEI = "NEI"


class EvidenceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass
class VerificationResult:
    session_id: str
    claim: str
    model_verdict: ModelVerdict
    display_verdict: DisplayVerdict
    evidence_status: EvidenceStatus
    sample_logits: Dict[str, float]
    probabilities: Dict[str, float]
    all_units: List[RuntimeUnit]
    top_k_units: List[RuntimeUnit]
    class_winners: Dict[str, str]
    pipeline_stages: List[PipelineStage]
    warnings: List[str] = field(default_factory=list)
    checkpoint_sha256: Optional[str] = None
    runtime_ms: float = 0.0

    def __post_init__(self) -> None:
        binary_classes = {"fake", "real"}
        logit_classes = set(self.sample_logits)
        probability_classes = set(self.probabilities)
        if not logit_classes <= binary_classes or not probability_classes <= binary_classes:
            raise ValueError("model logits and probabilities may contain only fake and real")

        expected_display = {
            ModelVerdict.FAKE: DisplayVerdict.FAKE,
            ModelVerdict.REAL: DisplayVerdict.REAL,
            ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
        }[self.model_verdict]
        if self.model_verdict in {ModelVerdict.FAKE, ModelVerdict.REAL}:
            if logit_classes != binary_classes or probability_classes != binary_classes:
                raise ValueError("FAKE/REAL results require exactly fake and real logits and probabilities")
            if self.display_verdict != expected_display:
                raise ValueError("display verdict must match the FAKE/REAL model verdict")
        elif self.model_verdict == ModelVerdict.NOT_RUN:
            if self.sample_logits or self.probabilities:
                raise ValueError("NOT_RUN results cannot contain logits or probabilities")
            if self.evidence_status != EvidenceStatus.INSUFFICIENT:
                raise ValueError("NOT_RUN results require insufficient evidence")
            if self.display_verdict != expected_display:
                raise ValueError("NOT_RUN results require the NEI display verdict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claim": self.claim,
            "model_verdict": self.model_verdict.value,
            "display_verdict": self.display_verdict.value,
            "evidence_status": self.evidence_status.value,
            "sample_logits": dict(self.sample_logits),
            "probabilities": dict(self.probabilities),
            "all_units": [unit.to_dict() for unit in self.all_units],
            "top_k_units": [unit.to_dict() for unit in self.top_k_units],
            "class_winners": dict(self.class_winners),
            "pipeline_stages": [stage.to_dict() for stage in self.pipeline_stages],
            "warnings": list(self.warnings),
            "checkpoint_sha256": self.checkpoint_sha256,
            "runtime_ms": self.runtime_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(
            session_id=str(data["session_id"]),
            claim=str(data["claim"]),
            model_verdict=ModelVerdict(data["model_verdict"]),
            display_verdict=DisplayVerdict(data["display_verdict"]),
            evidence_status=EvidenceStatus(data["evidence_status"]),
            sample_logits={key: float(value) for key, value in data.get("sample_logits", {}).items()},
            probabilities={key: float(value) for key, value in data.get("probabilities", {}).items()},
            all_units=[RuntimeUnit.from_dict(item) for item in data.get("all_units", [])],
            top_k_units=[RuntimeUnit.from_dict(item) for item in data.get("top_k_units", [])],
            class_winners={str(key): str(value) for key, value in data.get("class_winners", {}).items()},
            pipeline_stages=[PipelineStage.from_dict(item) for item in data.get("pipeline_stages", [])],
            warnings=[str(item) for item in data.get("warnings", [])],
            checkpoint_sha256=data.get("checkpoint_sha256"),
            runtime_ms=float(data.get("runtime_ms", 0.0)),
        )
