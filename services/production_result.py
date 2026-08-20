"""Path-safe API presentation contract for completed production results."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
)
from services.evidence_sufficiency_policy import (
    EvidenceSufficiencyAssessment,
    EvidenceSufficiencyPolicy,
)
from services.video_multimodal_runner import VideoMultimodalResult


SCHEMA_VERSION = 1


def _optional_string_tuple(
    unit: RuntimeUnit,
    field_name: str,
) -> Tuple[str, ...]:
    details = unit.provenance.details
    if field_name not in details:
        return ()
    value = details[field_name]
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(
            f"provenance.details[{field_name!r}] must be a list or tuple of strings"
        )
    return tuple(value)


@dataclass(frozen=True)
class ProductionEvidenceUnit:
    unit_id: str
    source_type: SourceType
    text: str
    start_time: Optional[float]
    end_time: Optional[float]
    frame_id: Optional[str]
    bbox: Optional[Tuple[float, ...]]
    confidence: Optional[float]
    producer: str
    eligible_for_frozen_g1: bool
    selection_score: Optional[float]
    logits: Optional[Tuple[Tuple[str, float], ...]]
    extraction_method: str
    source_index: Optional[int]
    frame_ids: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    source_unit_ids: Tuple[str, ...]
    observation_type: Optional[str]

    @classmethod
    def from_runtime_unit(cls, unit: RuntimeUnit) -> "ProductionEvidenceUnit":
        details = unit.provenance.details
        observation_type = details.get("observation_type")
        if observation_type is not None and not isinstance(observation_type, str):
            raise ValueError(
                "provenance.details['observation_type'] must be a string"
            )
        return cls(
            unit_id=unit.unit_id,
            source_type=unit.source_type,
            text=unit.text,
            start_time=unit.start_time,
            end_time=unit.end_time,
            frame_id=unit.frame_id,
            bbox=None if unit.bbox is None else tuple(unit.bbox),
            confidence=unit.confidence,
            producer=unit.producer,
            eligible_for_frozen_g1=unit.eligible_for_frozen_g1,
            selection_score=unit.selection_score,
            logits=(
                None
                if unit.logits is None
                else tuple(sorted(unit.logits.items()))
            ),
            extraction_method=unit.provenance.extraction_method,
            source_index=unit.provenance.source_index,
            frame_ids=_optional_string_tuple(unit, "frame_ids"),
            evidence_refs=_optional_string_tuple(unit, "evidence_refs"),
            source_unit_ids=_optional_string_tuple(unit, "source_unit_ids"),
            observation_type=observation_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_type": self.source_type.value,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frame_id": self.frame_id,
            "bbox": None if self.bbox is None else list(self.bbox),
            "confidence": self.confidence,
            "producer": self.producer,
            "eligible_for_frozen_g1": self.eligible_for_frozen_g1,
            "selection_score": self.selection_score,
            "logits": None if self.logits is None else dict(self.logits),
            "extraction_method": self.extraction_method,
            "source_index": self.source_index,
            "frame_ids": list(self.frame_ids),
            "evidence_refs": list(self.evidence_refs),
            "source_unit_ids": list(self.source_unit_ids),
            "observation_type": self.observation_type,
        }


@dataclass(frozen=True)
class ProductionResult:
    schema_version: int
    session_id: str
    claim: str
    model_verdict: ModelVerdict
    display_verdict: DisplayVerdict
    evidence_status: EvidenceStatus
    sample_logits: Tuple[Tuple[str, float], ...]
    probabilities: Tuple[Tuple[str, float], ...]
    class_winners: Tuple[Tuple[str, str], ...]
    checkpoint_sha256: Optional[str]
    sufficiency: EvidenceSufficiencyAssessment
    g1_exposure_units: Tuple[ProductionEvidenceUnit, ...]
    g1_top_k_explanation_unit_ids: Tuple[str, ...]
    visual_supplemental_units: Tuple[ProductionEvidenceUnit, ...]
    runtime_ms: float

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("ProductionResult schema_version must equal 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "claim": self.claim,
            "verdict": {
                "model_verdict": self.model_verdict.value,
                "display_verdict": self.display_verdict.value,
                "evidence_status": self.evidence_status.value,
                "sample_logits": dict(self.sample_logits),
                "probabilities": dict(self.probabilities),
                "class_winners": dict(self.class_winners),
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "sufficiency": self.sufficiency.to_dict(),
            "evidence": {
                "g1_exposure_units": [
                    unit.to_dict() for unit in self.g1_exposure_units
                ],
                "g1_top_k_explanation_unit_ids": list(
                    self.g1_top_k_explanation_unit_ids
                ),
                "visual_supplemental_units": [
                    unit.to_dict() for unit in self.visual_supplemental_units
                ],
            },
            "runtime_ms": self.runtime_ms,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ProductionResultBuilder:
    def build(self, result: VideoMultimodalResult) -> ProductionResult:
        if not isinstance(result, VideoMultimodalResult):
            raise TypeError("result must be a VideoMultimodalResult")

        sufficiency = EvidenceSufficiencyPolicy().assess(result)
        verification = result.verification_result
        return ProductionResult(
            schema_version=SCHEMA_VERSION,
            session_id=result.session_id,
            claim=result.claim,
            model_verdict=verification.model_verdict,
            display_verdict=verification.display_verdict,
            evidence_status=verification.evidence_status,
            sample_logits=tuple(sorted(verification.sample_logits.items())),
            probabilities=tuple(sorted(verification.probabilities.items())),
            class_winners=tuple(sorted(verification.class_winners.items())),
            checkpoint_sha256=verification.checkpoint_sha256,
            sufficiency=sufficiency,
            g1_exposure_units=tuple(
                ProductionEvidenceUnit.from_runtime_unit(unit)
                for unit in result.g1_exposure_units
            ),
            g1_top_k_explanation_unit_ids=tuple(
                unit.unit_id for unit in verification.top_k_units
            ),
            visual_supplemental_units=tuple(
                ProductionEvidenceUnit.from_runtime_unit(unit)
                for unit in result.visual_units
            ),
            runtime_ms=result.runtime_ms,
        )
