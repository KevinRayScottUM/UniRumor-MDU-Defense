"""Stable JSON-serializable runtime contracts."""

from .grounded_visual_unit import (
    GROUNDED_VISUAL_ARTIFACT_TYPE,
    GROUNDED_VISUAL_SCHEMA_VERSION,
    GroundedFrameReference,
    GroundedVisualUnit,
    GroundingLineage,
    GroundingModelIdentity,
)
from .provenance import UnitProvenance
from .production_runtime_config import (
    FrozenG1RuntimeConfig,
    OCRRuntimeConfig,
    ProductionRuntimeConfig,
    QwenRuntimeConfig,
    SigLIPRuntimeConfig,
    WhisperRuntimeConfig,
)
from .request import VerificationRequest
from .result import EvidenceStatus, DisplayVerdict, ModelVerdict, VerificationResult
from .stage import PipelineStage, StageName, StageStatus
from .unit import RuntimeUnit, SourceType
from .visual_observation_snapshot import VisualObservationSnapshot
from .visual_xai import (
    QWEN_OCCLUSION_BASELINE,
    QWEN_OCCLUSION_METHOD,
    VISUAL_XAI_ARTIFACT_TYPE,
    VISUAL_XAI_BOUNDARY,
    VISUAL_XAI_DISCLAIMER,
    VISUAL_XAI_SCHEMA_VERSION,
    VisualAttributionArtifact,
    VisualAttributionMap,
    VisualTargetScore,
    VisualTargetSpan,
)

__all__ = [
    "DisplayVerdict",
    "EvidenceStatus",
    "FrozenG1RuntimeConfig",
    "GROUNDED_VISUAL_ARTIFACT_TYPE",
    "GROUNDED_VISUAL_SCHEMA_VERSION",
    "GroundedFrameReference",
    "GroundedVisualUnit",
    "GroundingLineage",
    "GroundingModelIdentity",
    "ModelVerdict",
    "OCRRuntimeConfig",
    "PipelineStage",
    "ProductionRuntimeConfig",
    "QwenRuntimeConfig",
    "RuntimeUnit",
    "SourceType",
    "SigLIPRuntimeConfig",
    "StageName",
    "StageStatus",
    "UnitProvenance",
    "VerificationRequest",
    "VerificationResult",
    "VisualObservationSnapshot",
    "QWEN_OCCLUSION_BASELINE",
    "QWEN_OCCLUSION_METHOD",
    "VISUAL_XAI_ARTIFACT_TYPE",
    "VISUAL_XAI_BOUNDARY",
    "VISUAL_XAI_DISCLAIMER",
    "VISUAL_XAI_SCHEMA_VERSION",
    "VisualAttributionArtifact",
    "VisualAttributionMap",
    "VisualTargetScore",
    "VisualTargetSpan",
    "WhisperRuntimeConfig",
]
