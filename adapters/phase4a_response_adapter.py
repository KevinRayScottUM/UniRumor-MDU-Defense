"""Validate and normalize external Frozen G1 Phase4A predictions."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping


FROZEN_CHECKPOINT_SHA256 = "b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9"
FROZEN_VARIANT = "G1_text_ocr"
PHASE4A_CONTRACT_VERSION = "1.0.0"
NUMERIC_REL_TOL = 1e-6
NUMERIC_ABS_TOL = 1e-6
_CLASSES = {"fake", "real"}


@dataclass(frozen=True)
class Phase4AUnitOutput:
    unit_id: str
    modality: str
    selection_score: float
    veracity_logits: Dict[str, float]


@dataclass(frozen=True)
class Phase4APrediction:
    prediction: str
    prediction_id: int
    sample_logits: Dict[str, float]
    probabilities: Dict[str, float]
    unit_outputs: List[Phase4AUnitOutput]
    top_k_unit_ids: List[str]
    class_winners: Dict[str, str]
    checkpoint_sha256: str


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Phase4A {field} must be an object")
    return value


def _require_list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Phase4A {field} must be a list")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Phase4A {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Phase4A {field} must be finite")
    return number


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Phase4A {field} must be a nonnegative integer")
    return value


def _binary_scores(value: Any, field: str) -> Dict[str, float]:
    scores = _require_mapping(value, field)
    if set(scores) != _CLASSES:
        raise ValueError(f"Phase4A {field} keys must be exactly fake and real")
    return {label: _finite_float(scores[label], f"{field}.{label}") for label in ("fake", "real")}


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)


def _unit_id(item: Any, field: str) -> str:
    record = _require_mapping(item, field)
    unit_id = record.get("unit_id")
    if not isinstance(unit_id, str):
        raise ValueError(f"Phase4A {field}.unit_id must be a string")
    return unit_id


def parse_phase4a_prediction(
    payload: Mapping[str, Any],
    submitted_candidate_ids: Iterable[str],
    expected_case_id: str,
    expected_claim: str,
) -> Phase4APrediction:
    """Reject output that does not match the verified Frozen G1 contract."""

    payload = _require_mapping(payload, "prediction")
    expected_scalars = {
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "variant": FROZEN_VARIANT,
        "contract_version": PHASE4A_CONTRACT_VERSION,
        "pooling": "max",
        "maximum_units_per_sample": 24,
        "max_length": 256,
        "selection_head_affects_sample_pooling": False,
        "topk_is_only_prediction_basis": False,
        "dropped_visual_unit_count": 0,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected or type(payload.get(field)) is not type(expected):
            raise ValueError(f"Phase4A {field} does not match the frozen contract")
    if payload.get("case_id") != expected_case_id:
        raise ValueError("Phase4A case_id does not match the submitted session ID")
    if payload.get("claim") != expected_claim:
        raise ValueError("Phase4A claim does not match the submitted claim")

    prediction = payload.get("prediction")
    if prediction not in _CLASSES:
        raise ValueError("Phase4A prediction must be fake or real")
    prediction_id = payload.get("prediction_id")
    if type(prediction_id) is not int or prediction_id != {"fake": 0, "real": 1}[prediction]:
        raise ValueError("Phase4A prediction_id is inconsistent with fake=0 and real=1")

    sample_logits = _binary_scores(payload.get("sample_logits"), "sample_logits")
    probabilities = _binary_scores(payload.get("probabilities"), "probabilities")
    submitted_candidate_ids = list(submitted_candidate_ids)
    candidate_ids = set(submitted_candidate_ids)
    submitted_count = len(submitted_candidate_ids)
    input_count = _nonnegative_int(
        payload.get("input_unit_count_before_contract"), "input_unit_count_before_contract"
    )
    if input_count != submitted_count:
        raise ValueError("Phase4A input_unit_count_before_contract does not match submitted candidates")

    raw_unit_outputs = _require_list(payload.get("unit_outputs"), "unit_outputs")
    unit_outputs: List[Phase4AUnitOutput] = []
    exposed_ids = set()
    for index, item in enumerate(raw_unit_outputs):
        record = _require_mapping(item, f"unit_outputs[{index}]")
        unit_id = _unit_id(record, f"unit_outputs[{index}]")
        if unit_id not in candidate_ids:
            raise ValueError(f"Phase4A exposed unknown unit ID: {unit_id!r}")
        if unit_id in exposed_ids:
            raise ValueError(f"Phase4A exposed duplicate unit ID: {unit_id!r}")
        exposed_ids.add(unit_id)
        modality = record.get("modality")
        if modality not in {"text", "ocr"}:
            raise ValueError("Phase4A unit output modality must be text or ocr")
        unit_outputs.append(
            Phase4AUnitOutput(
                unit_id=unit_id,
                modality=modality,
                selection_score=_finite_float(
                    record.get("selection_score"), f"unit_outputs[{index}].selection_score"
                ),
                veracity_logits=_binary_scores(
                    record.get("veracity_logits"), f"unit_outputs[{index}].veracity_logits"
                ),
            )
        )

    exposed_count = _nonnegative_int(
        payload.get("model_exposed_unit_count"), "model_exposed_unit_count"
    )
    if exposed_count != len(unit_outputs):
        raise ValueError("Phase4A model_exposed_unit_count does not match unit_outputs")
    truncated_count = _nonnegative_int(payload.get("truncated_unit_count"), "truncated_unit_count")
    if truncated_count != submitted_count - len(unit_outputs):
        raise ValueError("Phase4A truncated_unit_count does not match submitted and exposed units")
    if not unit_outputs:
        raise ValueError("Phase4A successful prediction must expose at least one unit")

    raw_top_k = _require_list(payload.get("top_k_selection_units"), "top_k_selection_units")
    if len(raw_top_k) > 5:
        raise ValueError("Phase4A top_k_selection_units exceeds frozen top_k=5")
    top_k_unit_ids = [
        _unit_id(item, f"top_k_selection_units[{index}]") for index, item in enumerate(raw_top_k)
    ]
    if len(set(top_k_unit_ids)) != len(top_k_unit_ids):
        raise ValueError("Phase4A top_k_selection_units contains duplicate IDs")
    if not set(top_k_unit_ids) <= exposed_ids:
        raise ValueError("Phase4A top-k IDs must refer to exposed units")

    raw_winners = _require_mapping(payload.get("max_pool_winner_by_class"), "max_pool_winner_by_class")
    if set(raw_winners) != _CLASSES:
        raise ValueError("Phase4A max_pool_winner_by_class keys must be exactly fake and real")
    outputs_by_id = {unit.unit_id: unit for unit in unit_outputs}
    class_winners: Dict[str, str] = {}
    for label in ("fake", "real"):
        winner = _require_mapping(raw_winners[label], f"max_pool_winner_by_class.{label}")
        unit_id = _unit_id(winner, f"max_pool_winner_by_class.{label}")
        if unit_id not in exposed_ids:
            raise ValueError("Phase4A class-winner IDs must refer to exposed units")
        winner_logits = _binary_scores(
            winner.get("veracity_logits"), f"max_pool_winner_by_class.{label}.veracity_logits"
        )
        if not _same_number(winner_logits[label], sample_logits[label]):
            raise ValueError("Phase4A class-winner veracity logit does not match sample logit")
        if not _same_number(outputs_by_id[unit_id].veracity_logits[label], sample_logits[label]):
            raise ValueError("Phase4A class-winner unit does not carry the sample-max logit")
        class_winners[label] = unit_id

    for label in ("fake", "real"):
        recomputed = max(unit.veracity_logits[label] for unit in unit_outputs)
        if not _same_number(recomputed, sample_logits[label]):
            raise ValueError("Phase4A sample logits are not class-wise maxima over all unit outputs")

    return Phase4APrediction(
        prediction=prediction,
        prediction_id=prediction_id,
        sample_logits=sample_logits,
        probabilities=probabilities,
        unit_outputs=unit_outputs,
        top_k_unit_ids=top_k_unit_ids,
        class_winners=class_winners,
        checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
    )
