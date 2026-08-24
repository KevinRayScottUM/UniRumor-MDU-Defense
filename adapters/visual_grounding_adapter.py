"""Deterministically convert isolated visual snapshots into audit artifacts."""

import hashlib
import json
from typing import Any, List, Mapping

from schemas import (
    GroundedVisualUnit,
    GroundingLineage,
    GroundingModelIdentity,
    VisualObservationSnapshot,
)


ADAPTER_ID = "visual_grounding_shadow_adapter"
ADAPTER_VERSION = "1.0.0"


class VisualGroundingAdapter:
    """Create model-free, shadow-only artifacts from immutable snapshots."""

    @staticmethod
    def _invalid(index: int, message: str) -> ValueError:
        return ValueError(f"visual_snapshots[{index}] {message}")

    @staticmethod
    def _source_observation_sha256(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def _ground_one(
        cls, snapshot: VisualObservationSnapshot, index: int
    ) -> GroundedVisualUnit:
        if not isinstance(snapshot, VisualObservationSnapshot):
            raise TypeError(
                f"visual_snapshots[{index}] must be a VisualObservationSnapshot"
            )

        model_identity = GroundingModelIdentity(
            siglip_model_id=snapshot.siglip_model_id,
            siglip_revision=snapshot.siglip_revision,
            qwen_model_id=snapshot.qwen_model_id,
            qwen_revision=snapshot.qwen_revision,
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )
        source_payload = snapshot.source_identity_payload()
        lineage = GroundingLineage(
            source_observation_id=snapshot.unit_id,
            source_observation_sha256=cls._source_observation_sha256(source_payload),
            source_index=snapshot.source_index,
            extraction_method=snapshot.extraction_method,
            observation_type=snapshot.observation_type,
            frame_ids=snapshot.frame_ids,
            evidence_refs=snapshot.evidence_refs,
            raw_generation_sha256=snapshot.raw_generation_sha256,
            recovery_mode=snapshot.recovery_mode,
            retrieval_policy_id=snapshot.retrieval_policy_id,
            observer_policy_id=snapshot.observer_policy_id,
        )
        try:
            return GroundedVisualUnit.create(
                source_observation_id=snapshot.unit_id,
                text_observation=snapshot.observation_text,
                start_timestamp_seconds=snapshot.start_timestamp_seconds,
                end_timestamp_seconds=snapshot.end_timestamp_seconds,
                frame_references=snapshot.frame_references,
                model_identity=model_identity,
                prompt_policy=snapshot.prompt_policy,
                lineage=lineage,
            )
        except (TypeError, ValueError) as exc:
            raise cls._invalid(index, f"cannot be grounded: {exc}") from exc

    def ground(
        self, visual_snapshots: List[VisualObservationSnapshot]
    ) -> List[GroundedVisualUnit]:
        if not isinstance(visual_snapshots, list):
            raise TypeError("visual_snapshots must be a list")
        grounded_units = []
        seen_source_ids = set()
        for index, snapshot in enumerate(visual_snapshots):
            grounded = self._ground_one(snapshot, index)
            if grounded.source_observation_id in seen_source_ids:
                raise self._invalid(index, "has a duplicate source observation id")
            seen_source_ids.add(grounded.source_observation_id)
            grounded_units.append(grounded)
        return grounded_units
