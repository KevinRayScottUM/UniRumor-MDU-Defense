"""Independent-audit ranking metrics and the sole preregistered 3B3 gate."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from scripts.selector_relevance_gate.metrics import (
    HeldoutRanking,
    ranked_unit_ids,
    reference_metrics,
)

from .schemas import (
    COVERAGE_GATE_MINIMUM,
    EXPECTED_DATASET_COUNTS,
    METRIC_NAMES,
    REQUIRED_SEED,
    IndependentCase,
    IndependentEvaluationError,
)


_FLOAT_EPSILON = 1e-12


def _ranking(
    case: IndependentCase, scores: Sequence[float]
) -> HeldoutRanking:
    try:
        return HeldoutRanking(
            reference_id=case.audit_case_id,
            case_id=case.canonical_case_id,
            dataset=case.dataset,
            reference_modality="independent_audit",
            candidate_unit_ids=case.candidate_unit_ids,
            positive_unit_ids=case.positive_unit_ids,
            selection_scores=tuple(float(value) for value in scores),
        )
    except (TypeError, ValueError) as exc:
        raise IndependentEvaluationError("selection-score ranking is invalid") from exc


def _nullable_metrics(
    case: IndependentCase, scores: Sequence[float]
) -> Tuple[Tuple[str, ...], Mapping[str, float | int | None]]:
    if len(scores) != len(case.candidate_units):
        raise IndependentEvaluationError("selection score count differs from candidates")
    if not case.evaluable:
        # Validate score finiteness/tie behavior using a temporary structural positive,
        # but never treat it as gold or include it in a metric denominator.
        structural = IndependentCase(
            audit_case_id=case.audit_case_id,
            dataset=case.dataset,
            canonical_case_id=case.canonical_case_id,
            claim=case.claim,
            candidate_units=case.candidate_units,
            positive_unit_ids=(case.candidate_unit_ids[0],),
        )
        order = ranked_unit_ids(_ranking(structural, scores))
        return order, {
            "best_direct_rank": None,
            "mrr": None,
            "ndcg_at_5": None,
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
        }
    ranking = _ranking(case, scores)
    values = reference_metrics(ranking)
    return ranked_unit_ids(ranking), {
        "best_direct_rank": int(values["best_positive_rank"]),
        **{name: float(values[name]) for name in METRIC_NAMES},
    }


def evaluate_case_scores(
    case: IndependentCase,
    *,
    original_scores: Sequence[float],
    calibrated_scores: Sequence[float],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return a score-only row and all-30-cases metric ledger row."""

    original_order, original = _nullable_metrics(case, original_scores)
    calibrated_order, calibrated = _nullable_metrics(case, calibrated_scores)
    score_row = {
        "dataset": case.dataset,
        "canonical_case_id": case.canonical_case_id,
        "candidate_unit_ids_in_original_order": list(case.candidate_unit_ids),
        "original_selection_scores": [float(value) for value in original_scores],
        "calibrated_selection_scores": [float(value) for value in calibrated_scores],
    }
    ledger = {
        "dataset": case.dataset,
        "canonical_case_id": case.canonical_case_id,
        "candidate_count": len(case.candidate_units),
        "positive_unit_ids": list(case.positive_unit_ids),
        "direct_positive_count": len(case.positive_unit_ids),
        "evaluable": case.evaluable,
        "original_ranked_unit_ids": list(original_order),
        "calibrated_ranked_unit_ids": list(calibrated_order),
        "original_top5_unit_ids": list(original_order[:5]),
        "calibrated_top5_unit_ids": list(calibrated_order[:5]),
        "original_best_direct_rank": original["best_direct_rank"],
        "calibrated_best_direct_rank": calibrated["best_direct_rank"],
        "best_direct_rank_delta": (
            int(calibrated["best_direct_rank"]) - int(original["best_direct_rank"])
            if case.evaluable
            else None
        ),
    }
    for name in METRIC_NAMES:
        ledger[f"original_{name}"] = original[name]
        ledger[f"calibrated_{name}"] = calibrated[name]
    return score_row, ledger


def _macro(rows: Sequence[Mapping[str, Any]], prefix: str) -> Mapping[str, float]:
    if not rows:
        raise IndependentEvaluationError("metric group has no evaluable cases")
    return {
        name: sum(float(row[f"{prefix}_{name}"]) for row in rows) / len(rows)
        for name in METRIC_NAMES
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    evaluable = [row for row in rows if row.get("evaluable") is True]
    groups = {
        "overall": evaluable,
        "GroundLie360": [row for row in evaluable if row.get("dataset") == "GroundLie360"],
        "TRUE-3MFact": [row for row in evaluable if row.get("dataset") == "TRUE-3MFact"],
    }
    expected_sizes = {
        "overall": sum(
            values["evaluable_case_count"] for values in EXPECTED_DATASET_COUNTS.values()
        ),
        **{
            dataset: values["evaluable_case_count"]
            for dataset, values in EXPECTED_DATASET_COUNTS.items()
        },
    }
    result: Dict[str, Any] = {}
    for group, selected in groups.items():
        if len(selected) != expected_sizes[group]:
            raise IndependentEvaluationError(f"{group} evaluable denominator changed")
        original = _macro(selected, "original")
        calibrated = _macro(selected, "calibrated")
        result[group] = {
            "evaluable_case_count": len(selected),
            "original": original,
            "calibrated": calibrated,
            "delta": {
                name: calibrated[name] - original[name] for name in METRIC_NAMES
            },
        }
    regression_count = sum(
        int(row["calibrated_best_direct_rank"])
        > int(row["original_best_direct_rank"])
        for row in evaluable
    )
    improvement_count = sum(
        int(row["calibrated_best_direct_rank"])
        < int(row["original_best_direct_rank"])
        for row in evaluable
    )
    return {
        "groups": result,
        "descriptive_case_counts": {
            "best_direct_rank_regression_count": regression_count,
            "best_direct_rank_improvement_count": improvement_count,
            "best_direct_rank_unchanged_count": len(evaluable)
            - regression_count
            - improvement_count,
            "used_as_acceptance_gate": False,
        },
    }


def calculate_repair_gate(
    metrics: Mapping[str, Any],
    *,
    evaluable_case_count: int,
    seed: int,
    architecture_condition_pass: bool,
) -> Mapping[str, Any]:
    """Apply exactly the frozen 3B1 gate; Recall@1/3 remain descriptive."""

    groups = metrics.get("groups")
    if not isinstance(groups, Mapping):
        raise IndependentEvaluationError("selector metrics groups are missing")
    try:
        overall = groups["overall"]
        groundlie = groups["GroundLie360"]
        true3m = groups["TRUE-3MFact"]
        original = overall["original"]
        calibrated = overall["calibrated"]
        mrr_delta = float(calibrated["mrr"]) - float(original["mrr"])
        ndcg_delta = float(calibrated["ndcg_at_5"]) - float(
            original["ndcg_at_5"]
        )
        recall5_delta = float(calibrated["recall_at_5"]) - float(
            original["recall_at_5"]
        )
        groundlie_delta = float(groundlie["calibrated"]["mrr"]) - float(
            groundlie["original"]["mrr"]
        )
        true3m_delta = float(true3m["calibrated"]["mrr"]) - float(
            true3m["original"]["mrr"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IndependentEvaluationError("selector metrics cannot drive the frozen gate") from exc

    conditions = {
        "coverage_condition_pass": evaluable_case_count >= COVERAGE_GATE_MINIMUM,
        "seed_condition_pass": seed == REQUIRED_SEED,
        "architecture_condition_pass": architecture_condition_pass is True,
        "mrr_strict_improvement_pass": float(calibrated["mrr"])
        > float(original["mrr"]),
        "ndcg_at_5_strict_improvement_pass": float(calibrated["ndcg_at_5"])
        > float(original["ndcg_at_5"]),
        "minimum_effect_size_pass": max(mrr_delta, ndcg_delta)
        + _FLOAT_EPSILON
        >= 0.05,
        "recall_at_5_tolerance_pass": recall5_delta + _FLOAT_EPSILON >= -0.02,
        "groundlie_mrr_tolerance_pass": groundlie_delta + _FLOAT_EPSILON >= -0.05,
        "true3m_mrr_tolerance_pass": true3m_delta + _FLOAT_EPSILON >= -0.05,
    }
    all_pass = all(conditions.values())
    return {
        **conditions,
        "mrr_delta": mrr_delta,
        "ndcg_at_5_delta": ndcg_delta,
        "recall_at_5_delta": recall5_delta,
        "groundlie_mrr_delta": groundlie_delta,
        "true3m_mrr_delta": true3m_delta,
        "all_preregistered_conditions_pass": all_pass,
        "repair_verification_pass": all_pass,
        "status": (
            "INDEPENDENT_SELECTOR_REPAIR_VERIFICATION_PASS"
            if all_pass
            else "INDEPENDENT_SELECTOR_REPAIR_VERIFICATION_FAIL"
        ),
        "deployment_remains_blocked": not all_pass,
        "recall_at_1_is_descriptive_only": True,
        "recall_at_3_is_descriptive_only": True,
        "case_level_regression_count_is_descriptive_only": True,
        "cpac_gate_present": False,
        "modality_specific_gate_present": False,
        "positive_original_top5_preservation_gate_present": False,
        "numerical_epsilon": _FLOAT_EPSILON,
    }
