"""Map runtime evidence units to the external Phase4A request contract."""

from typing import Any, Dict, Iterable, List

from schemas import RuntimeUnit, SourceType


_UNIT_MAPPING = {
    SourceType.TEXT: ("text", "text"),
    SourceType.TRANSCRIPT: ("transcript", "text"),
    SourceType.OCR: ("ocr", "ocr"),
}


def build_phase4a_request(
    case_id: str,
    claim: str,
    units: Iterable[RuntimeUnit],
    dataset: str = "external",
) -> Dict[str, Any]:
    """Build one Phase4A JSONL record without applying the 24-unit limit."""

    candidate_units: List[Dict[str, str]] = []
    candidate_ids = set()
    for unit in units:
        if unit.source_type is SourceType.VISUAL_OBSERVATION or not unit.eligible_for_frozen_g1:
            continue
        if not unit.text.strip():
            raise ValueError(f"eligible frozen G1 unit has blank text: {unit.unit_id!r}")
        if unit.unit_id in candidate_ids:
            raise ValueError(f"duplicate Phase4A candidate unit ID: {unit.unit_id!r}")
        candidate_ids.add(unit.unit_id)
        unit_type, modality = _UNIT_MAPPING[unit.source_type]
        candidate_units.append(
            {
                "unit_id": unit.unit_id,
                "unit_type": unit_type,
                "modality": modality,
                "text": unit.text,
            }
        )

    return {
        "case_id": case_id,
        "dataset": dataset,
        "claim": claim,
        "candidate_units": candidate_units,
    }
