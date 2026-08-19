"""Filter raw PP-OCRv5 detections into frame-level OCR RuntimeUnits."""

import re
import unicodedata
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, List, Optional, Sequence

from schemas import RuntimeUnit, SourceType, UnitProvenance
from services.paddle_ocr_service import (
    DETECTOR_MODEL_ID,
    DETECTOR_REVISION,
    OCRDetection,
    OCRFrameResult,
    RECOGNIZER_MODEL_ID,
    RECOGNIZER_REVISION,
)


PP_OCRV5_PRODUCER = (
    "PaddlePaddle/PP-OCRv5_server_det+PaddlePaddle/PP-OCRv5_server_rec"
)


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_text_key(text: str) -> str:
    return normalize_ocr_text(text).casefold()


def union_bbox(boxes: Iterable[Sequence[float]]) -> List[float]:
    materialized = [list(box) for box in boxes]
    if not materialized:
        raise ValueError("at least one OCR bbox is required")
    if any(len(box) != 4 for box in materialized):
        raise ValueError("OCR bbox must contain xmin, ymin, xmax, ymax")
    return [
        min(box[0] for box in materialized),
        min(box[1] for box in materialized),
        max(box[2] for box in materialized),
        max(box[3] for box in materialized),
    ]


@dataclass(frozen=True)
class OCRFilterConfig:
    confidence_threshold: float = 0.5
    min_normalized_length: int = 3
    max_ocr_units: int = 6

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("OCR confidence threshold must be within [0, 1]")
        if self.min_normalized_length <= 0:
            raise ValueError("OCR minimum normalized length must be positive")
        if self.max_ocr_units <= 0:
            raise ValueError("max_ocr_units must be positive")


@dataclass
class _FrameCandidate:
    frame: OCRFrameResult
    detections: List[OCRDetection]
    accepted_raw_detections: List[OCRDetection]
    raw_detection_count: int
    text: str
    confidence_mean: float
    bbox: List[float]


class OCRUnitAdapter:
    def __init__(self, config: Optional[OCRFilterConfig] = None) -> None:
        self.config = config or OCRFilterConfig()

    def _accepted(self, detection: OCRDetection) -> Optional[OCRDetection]:
        if detection.confidence < self.config.confidence_threshold:
            return None
        text = normalize_ocr_text(detection.text)
        information = "".join(character for character in text if character.isalnum())
        if len(information) < self.config.min_normalized_length:
            return None
        return OCRDetection(
            text=text,
            confidence=detection.confidence,
            polygon=detection.polygon,
            runtime_bbox=detection.runtime_bbox,
        )

    def _candidate(self, frame: OCRFrameResult) -> Optional[_FrameCandidate]:
        accepted_pairs = []
        for raw_detection in frame.detections:
            normalized_detection = self._accepted(raw_detection)
            if normalized_detection is not None:
                accepted_pairs.append((normalized_detection, raw_detection))
        accepted_pairs.sort(
            key=lambda pair: (
                pair[0].runtime_bbox[1],
                pair[0].runtime_bbox[0],
                pair[0].runtime_bbox[3],
                pair[0].runtime_bbox[2],
            )
        )
        detections = [pair[0] for pair in accepted_pairs]
        if not detections:
            return None
        return _FrameCandidate(
            frame=frame,
            detections=detections,
            accepted_raw_detections=[pair[1] for pair in accepted_pairs],
            raw_detection_count=len(frame.detections),
            text=" ".join(detection.text for detection in detections),
            confidence_mean=mean(detection.confidence for detection in detections),
            bbox=union_bbox(detection.runtime_bbox for detection in detections),
        )

    @staticmethod
    def _quality(candidate: _FrameCandidate):
        return (
            -candidate.confidence_mean,
            -candidate.raw_detection_count,
            candidate.frame.frame_rank,
        )

    def convert(
        self,
        frame_results: Iterable[OCRFrameResult],
        source_uri: Optional[str] = None,
    ) -> List[RuntimeUnit]:
        candidates = []
        for frame in frame_results:
            candidate = self._candidate(frame)
            if candidate is not None:
                candidates.append(candidate)

        unique = {}
        for candidate in candidates:
            key = normalized_text_key(candidate.text)
            current = unique.get(key)
            if current is None or self._quality(candidate) < self._quality(current):
                unique[key] = candidate
        selected = sorted(unique.values(), key=self._quality)[: self.config.max_ocr_units]
        selected.sort(key=lambda candidate: candidate.frame.frame_rank)

        units = []
        seen_ids = set()
        for position, candidate in enumerate(selected):
            unit_id = f"ocr_{position:04d}"
            if unit_id in seen_ids:
                raise ValueError(f"duplicate OCR RuntimeUnit ID: {unit_id}")
            seen_ids.add(unit_id)
            frame = candidate.frame
            units.append(
                RuntimeUnit(
                    unit_id=unit_id,
                    source_type=SourceType.OCR,
                    text=candidate.text,
                    start_time=frame.timestamp_sec,
                    end_time=frame.timestamp_sec,
                    frame_id=frame.frame_id,
                    frame_path=str(frame.frame_path),
                    bbox=candidate.bbox,
                    confidence=candidate.confidence_mean,
                    producer=PP_OCRV5_PRODUCER,
                    provenance=UnitProvenance(
                        source_uri=source_uri,
                        source_index=frame.frame_index,
                        extraction_method="paddleocr_v5_frame",
                        details={
                            "frame_rank": frame.frame_rank,
                            "raw_detection_count": candidate.raw_detection_count,
                            "detector_model_id": DETECTOR_MODEL_ID,
                            "detector_revision": DETECTOR_REVISION,
                            "recognizer_model_id": RECOGNIZER_MODEL_ID,
                            "recognizer_revision": RECOGNIZER_REVISION,
                            "accepted_detections": [
                                detection.to_dict()
                                for detection in candidate.accepted_raw_detections
                            ],
                            "normalized_detection_texts": [
                                detection.text for detection in candidate.detections
                            ],
                        },
                    ),
                    eligible_for_frozen_g1=True,
                    selection_score=None,
                    logits=None,
                )
            )
        return units
