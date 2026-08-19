"""Map normalized Whisper segments to Frozen G1 transcript units."""

from typing import Any, Dict, Iterable, List, Optional

from schemas import RuntimeUnit, SourceType, UnitProvenance
from services.whisper_asr_service import WHISPER_FROZEN_REVISION, WHISPER_MODEL_ID


class TranscriptUnitAdapter:
    def __init__(
        self,
        model_id: str = WHISPER_MODEL_ID,
        frozen_revision: str = WHISPER_FROZEN_REVISION,
    ) -> None:
        self.model_id = model_id
        self.frozen_revision = frozen_revision

    def convert(
        self,
        segments: Iterable[Dict[str, Any]],
        source_uri: Optional[str] = None,
    ) -> List[RuntimeUnit]:
        units = []
        seen_ids = set()
        for position, segment in enumerate(segments):
            text = str(segment.get("text", "")).strip()
            if not text:
                raise ValueError("ASR transcript segment has blank text")
            unit_id = f"asr_{position:04d}"
            if unit_id in seen_ids:
                raise ValueError(f"duplicate transcript RuntimeUnit ID: {unit_id}")
            seen_ids.add(unit_id)
            units.append(
                RuntimeUnit(
                    unit_id=unit_id,
                    source_type=SourceType.TRANSCRIPT,
                    text=text,
                    start_time=segment.get("start_time"),
                    end_time=segment.get("end_time"),
                    confidence=None,
                    producer=self.model_id,
                    provenance=UnitProvenance(
                        source_uri=source_uri,
                        source_index=int(segment.get("segment_index", position)),
                        extraction_method="whisper_asr",
                        details={
                            "model_id": self.model_id,
                            "frozen_revision": self.frozen_revision,
                        },
                    ),
                    eligible_for_frozen_g1=True,
                    selection_score=None,
                    logits=None,
                )
            )
        return units
