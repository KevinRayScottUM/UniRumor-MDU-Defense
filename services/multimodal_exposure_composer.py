"""Compose bounded transcript-first OCR exposure for the unchanged Frozen G1."""

from typing import Iterable, List

from schemas import RuntimeUnit, SourceType, UnitProvenance


MAX_TRANSCRIPT_EXPOSURE_UNITS = 12
MAX_OCR_EXPOSURE_UNITS = 6
NORMAL_COMBINED_EXPOSURE_UNITS = 18
PHASE4A_HARD_MAX_UNITS = 24


def _clone(unit: RuntimeUnit) -> RuntimeUnit:
    cloned = RuntimeUnit.from_dict(unit.to_dict())
    cloned.selection_score = None
    cloned.logits = None
    return cloned


class MultimodalExposureComposer:
    def __init__(
        self,
        max_transcript_units: int = MAX_TRANSCRIPT_EXPOSURE_UNITS,
        max_ocr_units: int = MAX_OCR_EXPOSURE_UNITS,
    ) -> None:
        if max_transcript_units <= 0 or max_ocr_units <= 0:
            raise ValueError("exposure limits must be positive")
        self.max_transcript_units = max_transcript_units
        self.max_ocr_units = max_ocr_units

    def compose_transcripts(
        self, raw_units: Iterable[RuntimeUnit]
    ) -> List[RuntimeUnit]:
        units = list(raw_units)
        for unit in units:
            if unit.source_type is not SourceType.TRANSCRIPT:
                raise ValueError("transcript exposure accepts only transcript units")
            if not unit.eligible_for_frozen_g1:
                raise ValueError("transcript exposure units must be Frozen G1 eligible")
        if len(units) <= self.max_transcript_units:
            return [_clone(unit) for unit in units]

        group_count = self.max_transcript_units
        grouped = []
        for group_index in range(group_count):
            start = group_index * len(units) // group_count
            end = (group_index + 1) * len(units) // group_count
            source_units = units[start:end]
            if not source_units:
                continue
            grouped.append(
                RuntimeUnit(
                    unit_id=f"transcript_exposure_{group_index:04d}",
                    source_type=SourceType.TRANSCRIPT,
                    text=" ".join(unit.text for unit in source_units),
                    start_time=source_units[0].start_time,
                    end_time=source_units[-1].end_time,
                    confidence=None,
                    producer="multimodal_exposure_composer",
                    provenance=UnitProvenance(
                        source_uri=source_units[0].provenance.source_uri,
                        source_index=group_index,
                        extraction_method="balanced_transcript_exposure",
                        details={
                            "source_unit_ids": [
                                unit.unit_id for unit in source_units
                            ],
                            "source_producers": [
                                unit.producer for unit in source_units
                            ],
                        },
                    ),
                    eligible_for_frozen_g1=True,
                    selection_score=None,
                    logits=None,
                )
            )
        return grouped

    @staticmethod
    def _ocr_quality(unit: RuntimeUnit):
        raw_detection_count = unit.provenance.details.get("raw_detection_count")
        if raw_detection_count is None:
            raw_detection_count = len(
                unit.provenance.details.get("accepted_detections", [])
            )
        frame_rank = unit.provenance.details.get("frame_rank")
        if frame_rank is None:
            frame_rank = unit.provenance.source_index or 0
        confidence = unit.confidence if unit.confidence is not None else -1.0
        return (-confidence, -int(raw_detection_count), int(frame_rank))

    def compose_ocr(self, raw_units: Iterable[RuntimeUnit]) -> List[RuntimeUnit]:
        units = list(raw_units)
        for unit in units:
            if unit.source_type is not SourceType.OCR:
                raise ValueError("OCR exposure accepts only OCR units")
            if not unit.eligible_for_frozen_g1:
                raise ValueError("OCR exposure units must be Frozen G1 eligible")
        selected = sorted(units, key=self._ocr_quality)[: self.max_ocr_units]
        selected.sort(
            key=lambda unit: (
                float("inf") if unit.start_time is None else unit.start_time,
                unit.unit_id,
            )
        )
        return [_clone(unit) for unit in selected]

    def compose(
        self,
        raw_transcript_units: Iterable[RuntimeUnit],
        raw_ocr_units: Iterable[RuntimeUnit],
    ) -> List[RuntimeUnit]:
        transcript_units = self.compose_transcripts(raw_transcript_units)
        ocr_units = self.compose_ocr(raw_ocr_units)
        combined = transcript_units + ocr_units
        if any(unit.source_type is SourceType.VISUAL_OBSERVATION for unit in combined):
            raise ValueError("visual observations cannot enter Frozen G1 exposure")
        if len(combined) > self.max_transcript_units + self.max_ocr_units:
            raise AssertionError("combined exposure exceeded configured engineering policy")
        return combined
