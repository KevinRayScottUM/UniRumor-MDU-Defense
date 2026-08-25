"""Path-safe API presentation contract for completed production results."""

import base64
import copy
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    QWEN_OCCLUSION_BASELINE,
    QWEN_OCCLUSION_METHOD,
    RuntimeUnit,
    SourceType,
    VISUAL_XAI_BOUNDARY,
    VISUAL_XAI_DISCLAIMER,
    VisualAttributionArtifact,
)
from services.evidence_sufficiency_policy import (
    EvidenceSufficiencyAssessment,
    EvidenceSufficiencyPolicy,
)
from services.video_multimodal_runner import VideoMultimodalResult


SCHEMA_VERSION = 1
MAX_EVIDENCE_FRAMES_PER_UNIT = 4
MAX_EVIDENCE_IMAGE_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_PAYLOAD_BYTES = 12 * 1024 * 1024


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _optional_index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    normalized = tuple(_optional_number(item) for item in value)
    if any(item is None for item in normalized):
        return None
    x1, y1, x2, y2 = normalized
    if x2 < x1 or y2 < y1:
        return None
    return (x1, y1, x2, y2)


def _optional_text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


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
class ProductionEvidenceRegion:
    text: Optional[str]
    bbox: Tuple[float, float, float, float]
    confidence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ProductionVisualXAIMap:
    map_id: str
    scope: str
    label: str
    heatmap_image: Optional[str]
    target_token_count: int
    baseline_target_log_probability: float
    raw_importance: Tuple[Tuple[float, ...], ...]
    normalized_importance: Tuple[Tuple[float, ...], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_id": self.map_id,
            "scope": self.scope,
            "label": self.label,
            "heatmap_image": self.heatmap_image,
            "target_token_count": self.target_token_count,
            "baseline_target_log_probability": self.baseline_target_log_probability,
            "raw_importance": [list(row) for row in self.raw_importance],
            "normalized_importance": [
                list(row) for row in self.normalized_importance
            ],
        }


@dataclass(frozen=True)
class ProductionVisualXAI:
    status: str
    unavailable_reason: Optional[str]
    method: str
    model_id: str
    model_revision: str
    model_fingerprint: str
    source_frame_sha256: str
    observation_unit_id: str
    observation_text_sha256: str
    raw_generation_sha256: str
    profile: str
    grid_rows: int
    grid_columns: int
    attribution_batch_size: int
    requested_batch_size: int
    effective_batch_size: int
    adaptive_batching: bool
    oom_retry_count: int
    occlusion_baseline: str
    configuration_version: str
    configuration_fingerprint: str
    phrase_policy: str
    heavy_scorer_batches: int
    disclaimer: str
    scientific_boundary: str
    attribution_maps: Tuple[ProductionVisualXAIMap, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "unavailable_reason": self.unavailable_reason,
            "method": self.method,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "source_frame_sha256": self.source_frame_sha256,
            "observation_unit_id": self.observation_unit_id,
            "observation_text_sha256": self.observation_text_sha256,
            "raw_generation_sha256": self.raw_generation_sha256,
            "profile": self.profile,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "attribution_batch_size": self.attribution_batch_size,
            "requested_batch_size": self.requested_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "adaptive_batching": self.adaptive_batching,
            "oom_retry_count": self.oom_retry_count,
            "occlusion_baseline": self.occlusion_baseline,
            "configuration_version": self.configuration_version,
            "configuration_fingerprint": self.configuration_fingerprint,
            "phrase_policy": self.phrase_policy,
            "heavy_scorer_batches": self.heavy_scorer_batches,
            "disclaimer": self.disclaimer,
            "scientific_boundary": self.scientific_boundary,
            "attribution_maps": [
                item.to_dict() for item in self.attribution_maps
            ],
        }


@dataclass(frozen=True)
class ProductionEvidenceFrame:
    frame_id: Optional[str]
    frame_index: Optional[int]
    timestamp: Optional[float]
    original_image: Optional[str]
    annotated_image: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]
    regions: Tuple[ProductionEvidenceRegion, ...]
    explanation: str
    xai: Optional[ProductionVisualXAI] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "original_image": self.original_image,
            "annotated_image": self.annotated_image,
            "bbox": None if self.bbox is None else list(self.bbox),
            "regions": [region.to_dict() for region in self.regions],
            "explanation": self.explanation,
            "xai": None if self.xai is None else self.xai.to_dict(),
        }


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
    evidence_frames: Tuple[ProductionEvidenceFrame, ...] = ()

    @classmethod
    def from_runtime_unit(
        cls,
        unit: RuntimeUnit,
        evidence_frames: Tuple[ProductionEvidenceFrame, ...] = (),
    ) -> "ProductionEvidenceUnit":
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
            evidence_frames=evidence_frames,
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
            "evidence_frames": [
                frame.to_dict() for frame in self.evidence_frames
            ],
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
    def __init__(self, *, evidence_root: Optional[Path] = None) -> None:
        if evidence_root is not None and not isinstance(evidence_root, Path):
            raise TypeError("evidence_root must be a Path or None")
        self.evidence_root = (
            None if evidence_root is None else evidence_root.expanduser().resolve()
        )

    @staticmethod
    def _image_mime_type(data: bytes) -> Optional[str]:
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    def _image_data_url(
        self,
        path_value: Any,
        remaining_bytes: list,
    ) -> Optional[str]:
        if self.evidence_root is None or not isinstance(path_value, (str, Path)):
            return None
        try:
            root = self.evidence_root.resolve(strict=True)
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_symlink():
                return None
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if root not in resolved.parents or not resolved.is_file():
            return None
        try:
            with resolved.open("rb") as handle:
                data = handle.read(MAX_EVIDENCE_IMAGE_BYTES + 1)
        except OSError:
            return None
        if (
            not data
            or len(data) > MAX_EVIDENCE_IMAGE_BYTES
            or len(data) > remaining_bytes[0]
        ):
            return None
        mime_type = self._image_mime_type(data)
        if mime_type is None:
            return None
        remaining_bytes[0] -= len(data)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _ocr_regions(unit: RuntimeUnit) -> Tuple[ProductionEvidenceRegion, ...]:
        regions = []
        raw_detections = unit.provenance.details.get("accepted_detections")
        if isinstance(raw_detections, (list, tuple)):
            for detection in raw_detections:
                if not isinstance(detection, dict):
                    continue
                bbox = _optional_bbox(detection.get("runtime_bbox"))
                if bbox is None:
                    continue
                confidence = _optional_number(detection.get("confidence"))
                if confidence is not None and not 0.0 <= confidence <= 1.0:
                    confidence = None
                regions.append(
                    ProductionEvidenceRegion(
                        text=_optional_text(detection.get("text")),
                        bbox=bbox,
                        confidence=confidence,
                    )
                )
        if not regions:
            bbox = _optional_bbox(unit.bbox)
            if bbox is not None:
                regions.append(
                    ProductionEvidenceRegion(
                        text=_optional_text(unit.text),
                        bbox=bbox,
                        confidence=_optional_number(unit.confidence),
                    )
                )
        return tuple(regions)

    def _ocr_frames(
        self,
        unit: RuntimeUnit,
        remaining_bytes: list,
    ) -> Tuple[ProductionEvidenceFrame, ...]:
        if unit.frame_id is None and unit.frame_path is None:
            return ()
        regions = self._ocr_regions(unit)
        image = self._image_data_url(unit.frame_path, remaining_bytes)
        explanation = (
            f"OCR text is grounded in {len(regions)} recorded region"
            f"{'s' if len(regions) != 1 else ''} on this frame."
            if regions
            else "OCR region unavailable; this unit is grounded at frame level."
        )
        return (
            ProductionEvidenceFrame(
                frame_id=_optional_text(unit.frame_id),
                frame_index=_optional_index(unit.provenance.source_index),
                timestamp=_optional_number(unit.start_time),
                original_image=image,
                annotated_image=None,
                bbox=_optional_bbox(unit.bbox),
                regions=regions,
                explanation=explanation,
            ),
        )

    def _visual_frames(
        self,
        unit: RuntimeUnit,
        remaining_bytes: list,
        xai_by_frame: Mapping[Tuple[str, str], VisualAttributionArtifact],
    ) -> Tuple[ProductionEvidenceFrame, ...]:
        referenced = unit.provenance.details.get("referenced_frames")
        metadata = (
            list(referenced[:MAX_EVIDENCE_FRAMES_PER_UNIT])
            if isinstance(referenced, (list, tuple))
            else []
        )
        if not metadata and (unit.frame_id is not None or unit.frame_path is not None):
            metadata = [
                {
                    "frame_id": unit.frame_id,
                    "frame_path": unit.frame_path,
                    "frame_index": unit.provenance.source_index,
                    "timestamp_sec": unit.start_time,
                }
            ]

        frames = []
        for item in metadata:
            if not isinstance(item, dict):
                continue
            image = self._image_data_url(item.get("frame_path"), remaining_bytes)
            frame_id = _optional_text(item.get("frame_id"))
            artifact = (
                None
                if frame_id is None
                else xai_by_frame.get((unit.unit_id, frame_id))
            )
            frames.append(
                ProductionEvidenceFrame(
                    frame_id=frame_id,
                    frame_index=_optional_index(item.get("frame_index")),
                    timestamp=_optional_number(item.get("timestamp_sec")),
                    original_image=image,
                    annotated_image=None,
                    bbox=None,
                    regions=(),
                    explanation=(
                        "This is an actual observer source frame. Visual localization "
                        "is available only when a model-derived attribution artifact "
                        "is attached below."
                    ),
                    xai=self._public_xai(artifact, remaining_bytes),
                )
            )
        return tuple(frames)

    def _evidence_frames(
        self,
        unit: RuntimeUnit,
        remaining_bytes: list,
        xai_by_frame: Mapping[Tuple[str, str], VisualAttributionArtifact],
    ) -> Tuple[ProductionEvidenceFrame, ...]:
        if unit.source_type is SourceType.OCR:
            return self._ocr_frames(unit, remaining_bytes)
        if unit.source_type is SourceType.VISUAL_OBSERVATION:
            return self._visual_frames(unit, remaining_bytes, xai_by_frame)
        return ()

    def _public_xai(
        self,
        artifact: Optional[VisualAttributionArtifact],
        remaining_bytes: list,
    ) -> Optional[ProductionVisualXAI]:
        if artifact is None:
            return None
        maps = tuple(
            ProductionVisualXAIMap(
                map_id=item.map_id,
                scope=item.scope,
                label=item.label,
                heatmap_image=self._image_data_url(
                    item.overlay_image_path, remaining_bytes
                ),
                target_token_count=item.target_token_count,
                baseline_target_log_probability=(
                    item.baseline_target_log_probability
                ),
                raw_importance=item.raw_importance,
                normalized_importance=item.normalized_importance,
            )
            for item in artifact.maps
        )
        return ProductionVisualXAI(
            status=artifact.status,
            unavailable_reason=artifact.unavailable_reason,
            method=artifact.method,
            model_id=artifact.model_id,
            model_revision=artifact.model_revision,
            model_fingerprint=artifact.model_fingerprint,
            source_frame_sha256=artifact.source_frame_sha256,
            observation_unit_id=artifact.observation_unit_id,
            observation_text_sha256=artifact.observation_text_sha256,
            raw_generation_sha256=artifact.raw_generation_sha256,
            profile=artifact.profile,
            grid_rows=artifact.grid_rows,
            grid_columns=artifact.grid_columns,
            attribution_batch_size=artifact.attribution_batch_size,
            requested_batch_size=artifact.requested_batch_size,
            effective_batch_size=artifact.effective_batch_size,
            adaptive_batching=artifact.adaptive_batching,
            oom_retry_count=artifact.oom_retry_count,
            occlusion_baseline=artifact.occlusion_baseline,
            configuration_version=artifact.configuration_version,
            configuration_fingerprint=artifact.configuration_fingerprint,
            phrase_policy=artifact.phrase_policy,
            heavy_scorer_batches=artifact.heavy_scorer_batches,
            disclaimer=VISUAL_XAI_DISCLAIMER,
            scientific_boundary=VISUAL_XAI_BOUNDARY,
            attribution_maps=maps,
        )

    @staticmethod
    def _embedded_image_bytes(value: Any) -> int:
        if isinstance(value, dict):
            return sum(
                ProductionResultBuilder._embedded_image_bytes(item)
                for item in value.values()
            )
        if isinstance(value, list):
            return sum(
                ProductionResultBuilder._embedded_image_bytes(item)
                for item in value
            )
        if not isinstance(value, str) or not value.startswith("data:image/"):
            return 0
        marker = ";base64,"
        if marker not in value:
            return 0
        encoded = value.split(marker, 1)[1]
        padding = len(encoded) - len(encoded.rstrip("="))
        return max(0, (len(encoded) * 3) // 4 - padding)

    def _public_xai_state(
        self,
        status: Any,
        artifact: Optional[VisualAttributionArtifact],
        remaining_bytes: list,
    ) -> Dict[str, Any]:
        public_artifact = (
            None
            if artifact is None
            else self._public_xai(artifact, remaining_bytes).to_dict()
        )
        if public_artifact is None:
            public_artifact = {
                "status": status.state.value,
                "unavailable_reason": status.unavailable_reason,
                "method": QWEN_OCCLUSION_METHOD,
                "model_id": None,
                "model_revision": None,
                "model_fingerprint": None,
                "source_frame_sha256": None,
                "observation_unit_id": status.unit_id,
                "observation_text_sha256": None,
                "raw_generation_sha256": None,
                "profile": status.profile,
                "grid_rows": status.grid_rows,
                "grid_columns": status.grid_columns,
                "attribution_batch_size": status.attribution_batch_size,
                "requested_batch_size": getattr(
                    status, "requested_batch_size", status.attribution_batch_size
                ),
                "effective_batch_size": getattr(
                    status, "effective_batch_size", None
                ),
                "adaptive_batching": getattr(
                    status, "adaptive_batching", False
                ),
                "oom_retry_count": getattr(status, "oom_retry_count", 0),
                "occlusion_baseline": QWEN_OCCLUSION_BASELINE,
                "configuration_version": None,
                "configuration_fingerprint": status.configuration_fingerprint,
                "phrase_policy": None,
                "heavy_scorer_batches": status.heavy_scorer_batches,
                "disclaimer": VISUAL_XAI_DISCLAIMER,
                "scientific_boundary": VISUAL_XAI_BOUNDARY,
                "attribution_maps": [],
            }
        else:
            public_artifact["status"] = status.state.value
            public_artifact["unavailable_reason"] = status.unavailable_reason
        public_artifact.update(
            {
                "cache_hit": status.cache_hit,
                "queue_wait_ms": status.queue_wait_ms,
                "compute_time_ms": status.compute_time_ms,
                "source_frame_count": status.source_frame_count,
                "heavy_scorer_batches": status.heavy_scorer_batches,
                "requested_batch_size": getattr(
                    status, "requested_batch_size", status.attribution_batch_size
                ),
                "effective_batch_size": getattr(
                    status, "effective_batch_size", None
                ),
                "adaptive_batching": getattr(
                    status, "adaptive_batching", False
                ),
                "oom_retry_count": getattr(status, "oom_retry_count", 0),
            }
        )
        return public_artifact

    def augment_visual_xai(
        self,
        result_payload: Mapping[str, Any],
        states: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Copy and augment supplemental presentation metadata only."""

        output = copy.deepcopy(dict(result_payload))
        remaining_bytes = [
            max(
                0,
                MAX_EVIDENCE_PAYLOAD_BYTES
                - self._embedded_image_bytes(output),
            )
        ]
        evidence = output.get("evidence")
        if not isinstance(evidence, dict):
            return output
        units = evidence.get("visual_supplemental_units")
        if not isinstance(units, list):
            return output
        for unit in units:
            if not isinstance(unit, dict):
                continue
            unit_id = unit.get("unit_id")
            status = states.get(unit_id)
            frames = unit.get("evidence_frames")
            if status is None or not isinstance(frames, list):
                continue
            artifacts = {
                item.source_frame_id: item for item in status.artifacts
            }
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                artifact = artifacts.get(frame.get("frame_id"))
                frame["xai"] = self._public_xai_state(
                    status, artifact, remaining_bytes
                )
        return output

    def _public_unit(
        self,
        unit: RuntimeUnit,
        remaining_bytes: list,
        xai_by_frame: Mapping[Tuple[str, str], VisualAttributionArtifact],
    ) -> ProductionEvidenceUnit:
        return ProductionEvidenceUnit.from_runtime_unit(
            unit,
            evidence_frames=self._evidence_frames(
                unit, remaining_bytes, xai_by_frame
            ),
        )

    def build(self, result: VideoMultimodalResult) -> ProductionResult:
        if not isinstance(result, VideoMultimodalResult):
            raise TypeError("result must be a VideoMultimodalResult")

        sufficiency = EvidenceSufficiencyPolicy().assess(result)
        verification = result.verification_result
        remaining_bytes = [MAX_EVIDENCE_PAYLOAD_BYTES]
        xai_by_frame = {
            (artifact.observation_unit_id, artifact.source_frame_id): artifact
            for artifact in result.visual_xai_artifacts
        }
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
                self._public_unit(unit, remaining_bytes, xai_by_frame)
                for unit in result.g1_exposure_units
            ),
            g1_top_k_explanation_unit_ids=tuple(
                unit.unit_id for unit in verification.top_k_units
            ),
            visual_supplemental_units=tuple(
                self._public_unit(unit, remaining_bytes, xai_by_frame)
                for unit in result.visual_units
            ),
            runtime_ms=result.runtime_ms,
        )
