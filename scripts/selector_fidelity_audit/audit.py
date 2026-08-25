"""Frozen G1 selector-fidelity audit utilities.

This module is diagnostic-only. It reuses the existing external FrozenG1Runner,
preserves candidate order and scores, and never changes the production path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from adapters.phase4a_response_adapter import FROZEN_CHECKPOINT_SHA256
from schemas import ProductionRuntimeConfig, RuntimeUnit, SourceType, UnitProvenance
from services.frozen_g1_runner import FrozenG1Runner, FrozenG1RunnerConfig


ALLOWED_MODALITIES = {"OCR", "TRANSCRIPT", "VISUAL_SUPPLEMENTAL", "NONE"}
AUDITED_RELEVANCE_STATES = {"audited"}
CLASSIFICATIONS = {
    "NO_CLEAR_SELECTOR_FAILURE",
    "MODALITY_SPECIFIC_RANKING_BIAS",
    "GENERAL_CLAIM_RELEVANCE_FAILURE",
    "INSUFFICIENT_EVIDENCE_TO_CONCLUDE",
}
RANKING_COLUMNS = (
    "probe_id",
    "claim",
    "expected_modality",
    "candidate_exposure_index",
    "unit_id",
    "source_type",
    "unit_text",
    "raw_selection_score",
    "selection_rank",
    "top_k_member",
    "fake_logit",
    "real_logit",
)


class AuditInputError(ValueError):
    """Raised when an audit input cannot preserve the controlled contract."""


@dataclass(frozen=True)
class ProbeDefinition:
    probe_id: str
    claim: str
    expected_modality: str
    expected_relevant_unit_ids: Tuple[str, ...]
    direct_grounding_unit_ids: Tuple[str, ...]
    annotation_status: str
    annotation_basis: str

    @property
    def has_audited_relevance(self) -> bool:
        return (
            self.annotation_status in AUDITED_RELEVANCE_STATES
            and bool(self.expected_relevant_unit_ids)
        )


@dataclass(frozen=True)
class RankedUnit:
    exposure_index: int
    unit_id: str
    source_type: str
    text: str
    raw_selection_score: float
    selection_rank: int
    top_k_member: bool
    fake_logit: float
    real_logit: float


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"malformed JSON input: {path.name}") from exc


def _reject_formal_data_path(path: Path) -> None:
    forbidden = {"validation", "test"}
    if any(part.casefold() in forbidden for part in path.resolve().parts):
        raise AuditInputError("formal Validation/Test paths are forbidden")


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditInputError(f"{field} must be a non-blank string")
    return value.strip()


def _string_tuple(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise AuditInputError(f"{field} must be a list")
    items = tuple(_nonblank(item, field) for item in value)
    if len(set(items)) != len(items):
        raise AuditInputError(f"{field} contains duplicate IDs")
    return items


def load_probe_manifest(path: Path) -> Tuple[Mapping[str, Any], List[ProbeDefinition]]:
    payload = _read_json(Path(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AuditInputError("probe manifest schema_version must equal 1")
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise AuditInputError("probe manifest must contain probes")
    probes: List[ProbeDefinition] = []
    seen_ids = set()
    for index, item in enumerate(raw_probes):
        if not isinstance(item, dict):
            raise AuditInputError(f"probes[{index}] must be an object")
        probe_id = _nonblank(item.get("probe_id"), f"probes[{index}].probe_id")
        if probe_id in seen_ids:
            raise AuditInputError(f"duplicate probe_id: {probe_id}")
        seen_ids.add(probe_id)
        expected_modality = _nonblank(
            item.get("expected_modality"),
            f"probes[{index}].expected_modality",
        )
        if expected_modality not in ALLOWED_MODALITIES:
            raise AuditInputError(f"unsupported expected modality: {expected_modality}")
        probes.append(
            ProbeDefinition(
                probe_id=probe_id,
                claim=_nonblank(item.get("claim"), f"probes[{index}].claim"),
                expected_modality=expected_modality,
                expected_relevant_unit_ids=_string_tuple(
                    item.get("expected_relevant_unit_ids"),
                    f"probes[{index}].expected_relevant_unit_ids",
                ),
                direct_grounding_unit_ids=_string_tuple(
                    item.get("direct_grounding_unit_ids"),
                    f"probes[{index}].direct_grounding_unit_ids",
                ),
                annotation_status=_nonblank(
                    item.get("annotation_status"),
                    f"probes[{index}].annotation_status",
                ),
                annotation_basis=_nonblank(
                    item.get("annotation_basis"),
                    f"probes[{index}].annotation_basis",
                ),
            )
        )
    return payload, probes


def _candidate_records(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        current: Any = payload
        if isinstance(current.get("outcome"), dict):
            current = current["outcome"].get("result")
        elif isinstance(current.get("result"), dict):
            current = current["result"]
        if isinstance(current, dict) and isinstance(current.get("evidence"), dict):
            current = current["evidence"]
        if isinstance(current, dict):
            records = current.get("g1_exposure_units")
        else:
            records = None
    else:
        records = None
    if not isinstance(records, list) or not records:
        raise AuditInputError("candidate input must contain non-empty g1_exposure_units")
    if not all(isinstance(item, dict) for item in records):
        raise AuditInputError("every candidate unit must be an object")
    return records


def load_candidate_pool(path: Path) -> List[RuntimeUnit]:
    candidate_path = Path(path).expanduser().resolve()
    _reject_formal_data_path(candidate_path)
    records = _candidate_records(_read_json(candidate_path))
    units: List[RuntimeUnit] = []
    seen_ids = set()
    for index, item in enumerate(records):
        unit_id = _nonblank(item.get("unit_id"), f"candidate[{index}].unit_id")
        if unit_id in seen_ids:
            raise AuditInputError(f"duplicate candidate unit ID: {unit_id}")
        seen_ids.add(unit_id)
        try:
            source_type = SourceType(item.get("source_type"))
        except ValueError as exc:
            raise AuditInputError(
                f"candidate[{index}] has unsupported source_type"
            ) from exc
        if source_type is SourceType.VISUAL_OBSERVATION:
            raise AuditInputError("visual observations cannot enter the Frozen G1 audit pool")
        if item.get("eligible_for_frozen_g1") is False:
            raise AuditInputError("candidate pool contains a Frozen G1-ineligible unit")
        details = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        units.append(
            RuntimeUnit(
                unit_id=unit_id,
                source_type=source_type,
                text=_nonblank(item.get("text"), f"candidate[{index}].text"),
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                frame_id=item.get("frame_id"),
                bbox=list(item["bbox"]) if isinstance(item.get("bbox"), list) else None,
                confidence=item.get("confidence"),
                producer=str(item.get("producer", "selector_fidelity_audit_input")),
                provenance=UnitProvenance.from_dict(details),
                eligible_for_frozen_g1=True,
            )
        )
    return units


def _fresh_units(units: Iterable[RuntimeUnit]) -> List[RuntimeUnit]:
    return [RuntimeUnit.from_dict(unit.to_dict()) for unit in units]


def candidate_pool_sha256(units: Sequence[RuntimeUnit]) -> str:
    canonical = [
        {
            "unit_id": unit.unit_id,
            "source_type": unit.source_type.value,
            "text": unit.text,
        }
        for unit in units
    ]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_ranked_units(
    units: Sequence[RuntimeUnit],
    top_k_unit_ids: Iterable[str],
) -> List[RankedUnit]:
    scored: List[Tuple[int, float]] = []
    for index, unit in enumerate(units):
        if unit.selection_score is None or not math.isfinite(float(unit.selection_score)):
            raise AuditInputError(f"missing finite selection score for {unit.unit_id}")
        if unit.logits is None or set(unit.logits) != {"fake", "real"}:
            raise AuditInputError(f"missing binary veracity logits for {unit.unit_id}")
        scored.append((index, float(unit.selection_score)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    rank_by_index = {unit_index: rank for rank, (unit_index, _) in enumerate(scored, 1)}
    top_k = set(top_k_unit_ids)
    return [
        RankedUnit(
            exposure_index=index,
            unit_id=unit.unit_id,
            source_type=unit.source_type.value,
            text=unit.text,
            raw_selection_score=float(unit.selection_score),
            selection_rank=rank_by_index[index],
            top_k_member=unit.unit_id in top_k,
            fake_logit=float(unit.logits["fake"]),
            real_logit=float(unit.logits["real"]),
        )
        for index, unit in enumerate(units)
    ]


def _source_summary(ranked: Sequence[RankedUnit]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for item in ranked:
        grouped[item.source_type].append(item.raw_selection_score)
    return {
        source: {
            "mean_selection_score": statistics.fmean(values),
            "median_selection_score": statistics.median(values),
        }
        for source, values in sorted(grouped.items())
    }


def _ndcg_at_five(ranks: Sequence[int], relevant_count: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 5)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(relevant_count, 5) + 1)
    )
    return 0.0 if ideal == 0.0 else dcg / ideal


def compute_probe_metrics(
    probe: ProbeDefinition,
    ranked: Sequence[RankedUnit],
) -> Dict[str, Any]:
    ordered = sorted(ranked, key=lambda item: item.selection_rank)
    top_five = ordered[:5]
    top_five_ids = [item.unit_id for item in top_five]
    top_five_composition = dict(Counter(item.source_type for item in top_five))
    expected_source = {
        "OCR": "ocr",
        "TRANSCRIPT": "transcript",
    }.get(probe.expected_modality)
    expected_source_ranks = sorted(
        item.selection_rank
        for item in ranked
        if expected_source is not None and item.source_type == expected_source
    )
    base: Dict[str, Any] = {
        "probe_id": probe.probe_id,
        "claim": probe.claim,
        "expected_modality": probe.expected_modality,
        "annotation_status": probe.annotation_status,
        "top_5_unit_ids": top_five_ids,
        "top_5_modality_composition": top_five_composition,
        "expected_modality_top_5_count": (
            sum(item.source_type == expected_source for item in top_five)
            if expected_source is not None
            else None
        ),
        "best_expected_modality_rank": (
            expected_source_ranks[0] if expected_source_ranks else None
        ),
        "expected_modality_hit_at_5": (
            any(rank <= 5 for rank in expected_source_ranks)
            if expected_source is not None and expected_source_ranks
            else None
        ),
        "selection_score_by_source": _source_summary(ranked),
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "mrr": None,
        "ndcg_at_5": None,
        "highest_relevant_unit_rank": None,
        "direct_grounding_flags": [],
    }
    rank_by_id = {item.unit_id: item.selection_rank for item in ranked}
    missing_labels = sorted(set(probe.expected_relevant_unit_ids) - set(rank_by_id))
    if missing_labels:
        raise AuditInputError(
            f"probe {probe.probe_id} references unknown relevant IDs: {missing_labels}"
        )
    if probe.has_audited_relevance:
        relevant_ranks = sorted(rank_by_id[unit_id] for unit_id in probe.expected_relevant_unit_ids)
        count = len(relevant_ranks)
        base.update(
            {
                "recall_at_1": sum(rank <= 1 for rank in relevant_ranks) / count,
                "recall_at_3": sum(rank <= 3 for rank in relevant_ranks) / count,
                "recall_at_5": sum(rank <= 5 for rank in relevant_ranks) / count,
                "mrr": 1.0 / relevant_ranks[0],
                "ndcg_at_5": _ndcg_at_five(relevant_ranks, count),
                "highest_relevant_unit_rank": relevant_ranks[0],
            }
        )
    for unit_id in probe.direct_grounding_unit_ids:
        rank = rank_by_id.get(unit_id)
        if rank is None:
            raise AuditInputError(
                f"probe {probe.probe_id} references unknown direct-grounding ID: {unit_id}"
            )
        if rank > 5:
            base["direct_grounding_flags"].append(
                {
                    "flag": "DIRECT_GROUNDING_TOP5_MISS",
                    "unit_id": unit_id,
                    "selection_rank": rank,
                }
            )
    return base


def _mean_available(records: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def _modality_summary(
    probe_metrics: Sequence[Mapping[str, Any]],
    modality: str,
) -> Dict[str, Any]:
    records = [
        item
        for item in probe_metrics
        if item.get("expected_modality") == modality
    ]
    relevance_records = [
        item for item in records if item.get("highest_relevant_unit_rank") is not None
    ]
    return {
        "probe_count": len(records),
        "audited_relevance_probe_count": len(relevance_records),
        "mean_best_modality_rank": _mean_available(
            records,
            "best_expected_modality_rank",
        ),
        "modality_hit_at_5": _mean_available(
            records,
            "expected_modality_hit_at_5",
        ),
        "mean_best_relevant_rank": _mean_available(
            relevance_records,
            "highest_relevant_unit_rank",
        ),
        "hit_at_5": _mean_available(relevance_records, "recall_at_5"),
        "mean_expected_modality_top_5_count": _mean_available(
            records,
            "expected_modality_top_5_count",
        ),
    }


def classify_audit(
    aggregate: Mapping[str, Any],
    *,
    annotations_complete: bool,
) -> str:
    if not annotations_complete:
        return "INSUFFICIENT_EVIDENCE_TO_CONCLUDE"
    ocr = aggregate.get("ocr_summary", {})
    transcript = aggregate.get("transcript_summary", {})
    ocr_hit = ocr.get("hit_at_5")
    transcript_hit = transcript.get("hit_at_5")
    direct_misses = aggregate.get("direct_grounding_misses", [])
    if (
        isinstance(ocr_hit, (int, float))
        and isinstance(transcript_hit, (int, float))
        and direct_misses
        and ocr_hit <= 0.5
        and transcript_hit - ocr_hit >= 0.34
    ):
        return "MODALITY_SPECIFIC_RANKING_BIAS"
    overall_recall = aggregate.get("macro_recall_at_5")
    if isinstance(overall_recall, (int, float)) and overall_recall <= 0.5:
        return "GENERAL_CLAIM_RELEVANCE_FAILURE"
    return "NO_CLEAR_SELECTOR_FAILURE"


def _pair_rank_shifts(
    manifest: Mapping[str, Any],
    rankings: Mapping[str, Sequence[RankedUnit]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for pair in manifest.get("paired_claims", []):
        old_id = pair["old_probe_id"]
        new_id = pair["new_probe_id"]
        old_ranks = {item.unit_id: item.selection_rank for item in rankings[old_id]}
        new_ranks = {item.unit_id: item.selection_rank for item in rankings[new_id]}
        movements = [
            {
                "unit_id": unit_id,
                "old_rank": old_ranks[unit_id],
                "new_rank": new_ranks[unit_id],
                "delta_rank": new_ranks[unit_id] - old_ranks[unit_id],
            }
            for unit_id in old_ranks
        ]
        output.append(
            {
                "pair_id": pair["pair_id"],
                "old_probe_id": old_id,
                "new_probe_id": new_id,
                "unit_rank_movements": movements,
            }
        )
    return output


def _build_runner(config_path: Path) -> FrozenG1Runner:
    runtime_config_path = Path(config_path).expanduser().resolve()
    _reject_formal_data_path(runtime_config_path)
    config = ProductionRuntimeConfig.from_json(runtime_config_path)
    frozen = config.frozen_g1
    return FrozenG1Runner(
        FrozenG1RunnerConfig(
            unirumor_root=frozen.unirumor_root,
            python_executable=frozen.python_executable,
            phase4a_infer=frozen.phase4a_infer,
            phase4a_config=frozen.phase4a_config,
            device=frozen.device,
            timeout_seconds=frozen.timeout_seconds,
            cache_root=config.cache_root / "selector_fidelity_audit",
            output_root=config.output_root / "selector_fidelity_audit",
        )
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_rankings(
    path: Path,
    probes: Sequence[ProbeDefinition],
    rankings: Mapping[str, Sequence[RankedUnit]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RANKING_COLUMNS)
        writer.writeheader()
        for probe in probes:
            for item in rankings[probe.probe_id]:
                writer.writerow(
                    {
                        "probe_id": probe.probe_id,
                        "claim": probe.claim,
                        "expected_modality": probe.expected_modality,
                        "candidate_exposure_index": item.exposure_index,
                        "unit_id": item.unit_id,
                        "source_type": item.source_type,
                        "unit_text": item.text,
                        "raw_selection_score": repr(item.raw_selection_score),
                        "selection_rank": item.selection_rank,
                        "top_k_member": str(item.top_k_member).lower(),
                        "fake_logit": repr(item.fake_logit),
                        "real_logit": repr(item.real_logit),
                    }
                )


def _summary_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen G1 Selector Fidelity Audit",
        "",
        "## OBSERVED",
        "",
        f"- Audit status: `{metrics['audit_status']}`",
        f"- Candidate pool SHA-256: `{metrics['candidate_pool_sha256']}`",
        f"- Candidate count: {metrics['candidate_count']}",
        f"- Frozen checkpoint SHA-256: `{metrics['checkpoint_sha256']}`",
        "- Candidate order was preserved for every probe.",
        "- Raw selection scores and veracity logits were recorded without transformation.",
        "- Top-k membership came from the Frozen G1 response and remains explanation-only.",
        "",
        "### Per-probe Top-5",
        "",
    ]
    for item in metrics["per_probe"]:
        lines.append(
            f"- `{item['probe_id']}` ({item['expected_modality']}): "
            + ", ".join(item["top_5_unit_ids"])
            + f"; composition={item['top_5_modality_composition']}"
        )
    lines.extend(
        [
            "",
            "### Aggregate metrics",
            "",
            f"- Macro Recall@1: {metrics['macro_recall_at_1']}",
            f"- Macro Recall@3: {metrics['macro_recall_at_3']}",
            f"- Macro Recall@5: {metrics['macro_recall_at_5']}",
            f"- MRR: {metrics['mrr']}",
            f"- Mean NDCG@5: {metrics['mean_ndcg_at_5']}",
            f"- OCR Hit@5: {metrics['ocr_summary']['hit_at_5']}",
            f"- Transcript Hit@5: {metrics['transcript_summary']['hit_at_5']}",
            f"- Direct-grounding misses: {len(metrics['direct_grounding_misses'])}",
            f"- Paired claim comparisons: {len(metrics['paired_rank_shifts'])}",
            "",
            "## INTERPRETATION",
            "",
            f"Final classification: **{metrics['classification']}**",
            "",
            metrics["interpretation_note"],
            "",
            "This controlled diagnostic does not claim statistical significance and does not alter prediction, ranking, or Top-k behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    candidate_pool_path: Path,
    runtime_config_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    manifest, probes = load_probe_manifest(manifest_path)
    units = load_candidate_pool(candidate_pool_path)
    expected_count = manifest.get("candidate_pool", {}).get("expected_unit_count")
    if type(expected_count) is not int or len(units) != expected_count:
        raise AuditInputError(
            f"candidate pool must contain exactly {expected_count} units; received {len(units)}"
        )
    known_ids = {unit.unit_id for unit in units}
    for probe in probes:
        unknown_ids = (
            set(probe.expected_relevant_unit_ids)
            | set(probe.direct_grounding_unit_ids)
        ) - known_ids
        if unknown_ids:
            raise AuditInputError(
                f"probe {probe.probe_id} references unknown candidate IDs: {sorted(unknown_ids)}"
            )

    pool_hash = candidate_pool_sha256(units)
    runner = _build_runner(runtime_config_path)
    rankings: Dict[str, List[RankedUnit]] = {}
    per_probe: List[Dict[str, Any]] = []
    for probe in probes:
        probe_units = _fresh_units(units)
        session_id = f"selector_audit_{probe.probe_id}_{pool_hash[:12]}"
        result = runner.run(session_id, probe.claim, probe_units)
        ranked = derive_ranked_units(result.all_units, [unit.unit_id for unit in result.top_k_units])
        rankings[probe.probe_id] = ranked
        per_probe.append(compute_probe_metrics(probe, ranked))

    audited_records = [item for item in per_probe if item["mrr"] is not None]
    direct_misses = [
        {"probe_id": item["probe_id"], **flag}
        for item in per_probe
        for flag in item["direct_grounding_flags"]
    ]
    aggregate: Dict[str, Any] = {
        "audit_status": "COMPLETED",
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "candidate_pool_sha256": pool_hash,
        "candidate_count": len(units),
        "candidate_unit_ids_in_exposure_order": [unit.unit_id for unit in units],
        "per_probe": per_probe,
        "macro_recall_at_1": _mean_available(audited_records, "recall_at_1"),
        "macro_recall_at_3": _mean_available(audited_records, "recall_at_3"),
        "macro_recall_at_5": _mean_available(audited_records, "recall_at_5"),
        "mrr": _mean_available(audited_records, "mrr"),
        "mean_ndcg_at_5": _mean_available(audited_records, "ndcg_at_5"),
        "ocr_summary": _modality_summary(per_probe, "OCR"),
        "transcript_summary": _modality_summary(per_probe, "TRANSCRIPT"),
        "direct_grounding_misses": direct_misses,
        "paired_rank_shifts": _pair_rank_shifts(manifest, rankings),
    }
    annotations_complete = all(
        probe.annotation_status != "pending_exact_candidate_pool"
        for probe in probes
        if probe.expected_modality in {"OCR", "TRANSCRIPT"}
    )
    classification = classify_audit(
        aggregate,
        annotations_complete=annotations_complete,
    )
    if classification not in CLASSIFICATIONS:
        raise AssertionError("unexpected audit classification")
    aggregate["classification"] = classification
    aggregate["observed_cpac_ocr_failure_reproduced"] = any(
        item["probe_id"] == "ocr_01_direct_full_banner"
        for item in direct_misses
    )
    aggregate["interpretation_note"] = (
        "Transcript relevance annotations are incomplete, so the controlled evidence is insufficient for a selector-failure classification."
        if not annotations_complete
        else "Classification follows the declared controlled diagnostic rules; no inferential significance test was performed."
    )

    report_root = Path(output_dir).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    output_manifest = dict(manifest)
    output_manifest["audit_status"] = "COMPLETED"
    output_manifest["checkpoint_sha256"] = FROZEN_CHECKPOINT_SHA256
    output_manifest["candidate_pool_sha256"] = pool_hash
    output_manifest["candidate_unit_ids_in_exposure_order"] = [
        unit.unit_id for unit in units
    ]
    _write_json(report_root / "selector_probe_manifest.json", output_manifest)
    _write_rankings(report_root / "selector_probe_unit_rankings.csv", probes, rankings)
    _write_json(report_root / "selector_probe_metrics.json", aggregate)
    (report_root / "selector_probe_summary.md").write_text(
        _summary_markdown(aggregate),
        encoding="utf-8",
    )
    return aggregate
