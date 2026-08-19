"""Map grounded claim-blind visual observations to supplemental RuntimeUnits."""

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from schemas import RuntimeUnit, SourceType, UnitProvenance
from services.qwen_visual_observer import (
    PROMPT_POLICY,
    QWEN_FROZEN_REVISION,
    QWEN_MODEL_ID,
)
from services.siglip_visual_retriever import (
    SIGLIP_FROZEN_REVISION,
    SIGLIP_MODEL_ID,
    VisualFrame,
)


class VisualObservationAdapter:
    @staticmethod
    def _unit_id(index: int, observation: Dict[str, Any]) -> str:
        identity = {
            "index": index,
            "observation_type": observation["observation_type"],
            "observation": observation["observation"],
            "frame_ids": observation["frame_ids"],
            "evidence_refs": observation["evidence_refs"],
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"visual_{hashlib.sha256(encoded).hexdigest()[:20]}"

    @staticmethod
    def _chronology(frame: VisualFrame):
        return (
            frame.timestamp_sec is None,
            frame.timestamp_sec if frame.timestamp_sec is not None else 0.0,
            frame.frame_index,
            str(frame.frame_path),
        )

    def convert(
        self,
        observations: Iterable[Dict[str, Any]],
        selected_frames: Iterable[VisualFrame],
        recovery_mode: str,
        raw_generation_sha256: str,
        source_uri: Optional[str] = None,
    ) -> List[RuntimeUnit]:
        frames_by_id = {frame.frame_id: frame for frame in selected_frames}
        units = []
        seen_ids = set()
        for index, observation in enumerate(observations):
            frame_ids = list(observation["frame_ids"])
            evidence_refs = list(observation["evidence_refs"])
            referenced_ids = list(dict.fromkeys(frame_ids + evidence_refs))
            try:
                referenced_frames = [frames_by_id[item] for item in referenced_ids]
            except KeyError as exc:
                raise ValueError(f"unknown visual frame reference: {exc.args[0]}") from exc
            referenced_frames.sort(key=self._chronology)
            if not referenced_frames:
                raise ValueError("visual observation has no referenced frames")
            unit_id = self._unit_id(index, observation)
            if unit_id in seen_ids:
                raise ValueError(f"duplicate visual RuntimeUnit ID: {unit_id}")
            seen_ids.add(unit_id)
            primary = referenced_frames[0]
            timestamps = [
                frame.timestamp_sec
                for frame in referenced_frames
                if frame.timestamp_sec is not None
            ]
            units.append(
                RuntimeUnit(
                    unit_id=unit_id,
                    source_type=SourceType.VISUAL_OBSERVATION,
                    text=str(observation["observation"]),
                    start_time=min(timestamps) if timestamps else None,
                    end_time=max(timestamps) if timestamps else None,
                    frame_id=primary.frame_id,
                    frame_path=str(primary.frame_path),
                    bbox=None,
                    confidence=None,
                    producer=QWEN_MODEL_ID,
                    provenance=UnitProvenance(
                        source_uri=source_uri,
                        source_index=index,
                        extraction_method="qwen_claim_blind_visual_observer",
                        details={
                            "observation_type": observation["observation_type"],
                            "frame_ids": frame_ids,
                            "evidence_refs": evidence_refs,
                            "referenced_frames": [
                                frame.to_dict() for frame in referenced_frames
                            ],
                            "siglip_model_id": SIGLIP_MODEL_ID,
                            "siglip_revision": SIGLIP_FROZEN_REVISION,
                            "qwen_model_id": QWEN_MODEL_ID,
                            "qwen_revision": QWEN_FROZEN_REVISION,
                            "prompt_policy": PROMPT_POLICY,
                            "recovery_mode": recovery_mode,
                            "raw_generation_sha256": raw_generation_sha256,
                        },
                    ),
                    eligible_for_frozen_g1=False,
                    selection_score=None,
                    logits=None,
                )
            )
        return units
