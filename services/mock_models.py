"""Deterministic, non-scientific stand-ins for every runtime model."""

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List

from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
    VerificationRequest,
)


class MockASR:
    def transcribe(self, request: VerificationRequest) -> List[RuntimeUnit]:
        units = []
        for index, segment in enumerate(request.transcript_segments):
            text = str(segment.get("text", "")).strip()
            units.append(
                RuntimeUnit(
                    unit_id=f"transcript-{index:03d}",
                    source_type=SourceType.TRANSCRIPT,
                    text=text,
                    start_time=segment.get("start_time"),
                    end_time=segment.get("end_time"),
                    confidence=segment.get("confidence", 1.0),
                    producer="mock_asr",
                    provenance=UnitProvenance(
                        source_uri=request.media_path,
                        source_index=index,
                        extraction_method="mock_asr",
                    ),
                    eligible_for_frozen_g1=bool(text),
                )
            )
        return units


class MockOCR:
    def extract(self, request: VerificationRequest) -> List[RuntimeUnit]:
        units = []
        for index, observation in enumerate(request.ocr_observations):
            text = str(observation.get("text", "")).strip()
            units.append(
                RuntimeUnit(
                    unit_id=f"ocr-{index:03d}",
                    source_type=SourceType.OCR,
                    text=text,
                    frame_id=observation.get("frame_id"),
                    frame_path=observation.get("frame_path"),
                    bbox=list(observation["bbox"]) if observation.get("bbox") is not None else None,
                    confidence=observation.get("confidence"),
                    producer="mock_ocr",
                    provenance=UnitProvenance(
                        source_uri=observation.get("frame_path") or request.media_path,
                        source_index=index,
                        extraction_method="mock_ocr",
                    ),
                    eligible_for_frozen_g1=bool(text),
                )
            )
        return units


class MockVisualRetriever:
    def retrieve(self, request: VerificationRequest) -> List[Dict[str, object]]:
        return [dict(item, retrieval_rank=index) for index, item in enumerate(request.visual_inputs)]


class MockVLM:
    def observe(self, candidates: List[Dict[str, object]]) -> List[RuntimeUnit]:
        units = []
        for index, candidate in enumerate(candidates):
            units.append(
                RuntimeUnit(
                    unit_id=f"visual-{index:03d}",
                    source_type=SourceType.VISUAL_OBSERVATION,
                    text=str(candidate.get("observation", "")).strip(),
                    frame_id=candidate.get("frame_id"),
                    frame_path=candidate.get("frame_path"),
                    confidence=candidate.get("confidence"),
                    producer="mock_vlm",
                    provenance=UnitProvenance(
                        source_uri=candidate.get("frame_path"),
                        source_index=index,
                        extraction_method="mock_vlm",
                        details={"retrieval_rank": candidate.get("retrieval_rank", index)},
                    ),
                    eligible_for_frozen_g1=False,
                )
            )
        return units


def _unit_digest(claim: str, unit: RuntimeUnit) -> bytes:
    canonical = "\x1f".join((claim, unit.unit_id, unit.source_type.value, unit.text))
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _unit_scores(claim: str, unit: RuntimeUnit) -> None:
    digest = _unit_digest(claim, unit)
    denominator = float((1 << 64) - 1)
    selection = int.from_bytes(digest[0:8], "big") / denominator
    fake = (int.from_bytes(digest[8:16], "big") / denominator) * 4.0 - 2.0
    real = (int.from_bytes(digest[16:24], "big") / denominator) * 4.0 - 2.0
    unit.selection_score = round(selection, 12)
    unit.logits = {"fake": round(fake, 12), "real": round(real, 12)}


@dataclass
class MockG1Output:
    model_verdict: ModelVerdict
    display_verdict: DisplayVerdict
    evidence_status: EvidenceStatus
    sample_logits: Dict[str, float]
    probabilities: Dict[str, float]
    top_k_units: List[RuntimeUnit]
    class_winners: Dict[str, str]


def aggregate_all_evaluated(evaluated: List[RuntimeUnit], top_k: int) -> MockG1Output:
    if not evaluated:
        return MockG1Output(
            ModelVerdict.NOT_RUN,
            DisplayVerdict.NEI,
            EvidenceStatus.INSUFFICIENT,
            {},
            {},
            [],
            {},
        )
    sample_logits: Dict[str, float] = {}
    class_winners: Dict[str, str] = {}
    for label in ("fake", "real"):
        winner = max(evaluated, key=lambda unit: (unit.logits[label], unit.unit_id))
        sample_logits[label] = winner.logits[label]
        class_winners[label] = winner.unit_id
    maximum = max(sample_logits.values())
    exponentials = {label: math.exp(value - maximum) for label, value in sample_logits.items()}
    total = sum(exponentials.values())
    probabilities = {label: round(value / total, 12) for label, value in exponentials.items()}
    predicted = "fake" if sample_logits["fake"] >= sample_logits["real"] else "real"
    model_verdict = ModelVerdict(predicted)
    top_units = sorted(evaluated, key=lambda unit: (-unit.selection_score, unit.unit_id))[:top_k]
    return MockG1Output(
        model_verdict,
        {
            ModelVerdict.FAKE: DisplayVerdict.FAKE,
            ModelVerdict.REAL: DisplayVerdict.REAL,
            ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
        }[model_verdict],
        EvidenceStatus.SUFFICIENT,
        sample_logits,
        probabilities,
        top_units,
        class_winners,
    )


class MockG1:
    def __init__(self, max_units: int = 24, top_k: int = 5):
        self.max_units = max_units
        self.top_k = top_k

    def evaluate(self, claim: str, units: List[RuntimeUnit]) -> MockG1Output:
        evaluated = [unit for unit in units if unit.eligible_for_frozen_g1][: self.max_units]
        for unit in evaluated:
            _unit_scores(claim, unit)
        return aggregate_all_evaluated(evaluated, self.top_k)
