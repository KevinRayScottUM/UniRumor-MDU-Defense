"""Score-blind cross-case robustness audit for the frozen G1 selector.

The module extends the single-case audit without changing its candidate loader,
runner, rank derivation, Top-k authority, or relevance metrics.  Discovery and
probe construction deliberately do not read selector outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from schemas import RuntimeUnit, SourceType

from .audit import (
    AuditInputError,
    ProbeDefinition,
    RankedUnit,
    _build_runner,
    _candidate_records,
    _fresh_units,
    _reject_formal_data_path,
    candidate_pool_sha256,
    compute_probe_metrics,
    derive_ranked_units,
    load_candidate_pool_payload,
)


TARGET_CASE_COUNT = 5
MIN_CONCLUSIVE_CASE_COUNT = 4
PROBES_PER_MODALITY_PER_CASE = 2
PUBLIC_RESULT = "PUBLIC_RESULT"
NATIVE_PHASE4A = "NATIVE_PHASE4A"
ARTIFACT_CONTRACTS = {PUBLIC_RESULT, NATIVE_PHASE4A}
_SPLIT_IDENTITY_TOKENS = {"train", "test", "validation", "val"}
CROSS_CASE_CLASSIFICATIONS = {
    "CROSS_CASE_MODALITY_BIAS_CONFIRMED",
    "CASE_LOCAL_OR_MIXED_EFFECT",
    "GENERAL_SELECTOR_RELEVANCE_FAILURE",
    "INCONCLUSIVE",
}
_FORBIDDEN_RESULT_KEYS = {
    "selection_score",
    "top_k",
    "top_k_selection_units",
    "top_k_membership",
    "veracity_logits",
    "logits",
    "sample_logits",
    "probabilities",
    "prediction",
    "prediction_id",
}
_RANKING_COLUMNS = (
    "case_id",
    "dataset",
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


@dataclass(frozen=True)
class DiscoveryRoot:
    """A caller-labelled non-Test artifact root."""

    dataset: str
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise AuditInputError("discovery-root dataset label must be non-blank")
        if self.dataset.strip().casefold() in {"validation", "test"}:
            raise AuditInputError("formal Validation/Test dataset labels are forbidden")
        resolved = Path(self.path).expanduser().resolve()
        _reject_formal_data_path(resolved)
        object.__setattr__(self, "dataset", self.dataset.strip())
        object.__setattr__(self, "path", resolved)


@dataclass(frozen=True)
class CandidateCase:
    dataset: str
    case_id: str
    canonical_underlying_case_id: str
    artifact_contract: str
    artifact_key: str
    artifact_path: Path
    record_index: int
    claim: str
    units: Tuple[RuntimeUnit, ...]
    candidate_pool_sha256: str
    payload: Any

    @property
    def candidate_ids(self) -> Tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.units)


@dataclass(frozen=True)
class ReferenceSnapshot:
    claim: str
    candidate_ids: Tuple[str, ...]
    selection_scores: Mapping[str, float]
    fake_logits: Mapping[str, float]
    real_logits: Mapping[str, float]
    top_k_unit_ids: Tuple[str, ...]


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditInputError(f"{field} must be a non-blank string")
    return value.strip()


def _unwrap_result(payload: Any) -> Any:
    current = payload
    if isinstance(current, dict) and isinstance(current.get("outcome"), dict):
        outcome = current["outcome"]
        if isinstance(outcome.get("result"), dict):
            current = outcome["result"]
    elif isinstance(current, dict) and isinstance(current.get("result"), dict):
        current = current["result"]
    return current


def _case_scalar(payload: Any, names: Sequence[str]) -> Any:
    current = _unwrap_result(payload)
    mappings = [item for item in (current, payload) if isinstance(item, dict)]
    for mapping in mappings:
        for name in names:
            if mapping.get(name) is not None:
                return mapping[name]
    return None


def detect_artifact_contract(payload: Any) -> str:
    """Detect structure without consulting any selector-output value."""

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("claim"), str)
        and isinstance(payload.get("unit_outputs"), list)
        and isinstance(payload.get("top_k_selection_units"), list)
    ):
        return NATIVE_PHASE4A
    try:
        _candidate_records(payload)
    except AuditInputError as exc:
        raise AuditInputError("unsupported selector audit artifact contract") from exc
    return PUBLIC_RESULT


def _native_source_type(unit_type: Any, modality: Any, field: str) -> SourceType:
    normalized_type = _nonblank(unit_type, f"{field}.unit_type").casefold()
    normalized_modality = _nonblank(modality, f"{field}.modality").casefold()
    mapping = {
        "text": SourceType.TEXT,
        "transcript": SourceType.TRANSCRIPT,
        "ocr": SourceType.OCR,
    }
    if normalized_type not in mapping:
        raise AuditInputError(f"{field}.unit_type is not text/transcript/OCR")
    expected_modality = "ocr" if normalized_type == "ocr" else "text"
    if normalized_modality != expected_modality:
        raise AuditInputError(f"{field} has inconsistent unit_type/modality")
    return mapping[normalized_type]


def reconstruct_native_candidate_pool(payload: Any) -> List[RuntimeUnit]:
    """Stage-A reconstruction from native metadata fields only."""

    if detect_artifact_contract(payload) != NATIVE_PHASE4A:
        raise AuditInputError("native candidate reconstruction requires NATIVE_PHASE4A")
    outputs = payload["unit_outputs"]
    if not outputs:
        raise AuditInputError("native unit_outputs must be non-empty")
    units: List[RuntimeUnit] = []
    seen_ids = set()
    for index, item in enumerate(outputs):
        field = f"unit_outputs[{index}]"
        if not isinstance(item, dict):
            raise AuditInputError(f"{field} must be an object")
        unit_id = _nonblank(item.get("unit_id"), f"{field}.unit_id")
        if unit_id in seen_ids:
            raise AuditInputError(f"duplicate native candidate unit ID: {unit_id}")
        seen_ids.add(unit_id)
        units.append(
            RuntimeUnit(
                unit_id=unit_id,
                source_type=_native_source_type(
                    item.get("unit_type"), item.get("modality"), field
                ),
                text=_nonblank(item.get("text"), f"{field}.text"),
                producer="selector_fidelity_native_phase4a_reference",
                eligible_for_frozen_g1=True,
            )
        )
    return units


def reconstruct_candidate_pool(payload: Any, artifact_contract: str) -> List[RuntimeUnit]:
    if artifact_contract == NATIVE_PHASE4A:
        return reconstruct_native_candidate_pool(payload)
    if artifact_contract == PUBLIC_RESULT:
        return load_candidate_pool_payload(payload)
    raise AuditInputError("unsupported selector audit artifact contract")


def canonicalize_underlying_case_id(dataset: str, case_id: str) -> str:
    """Remove only explicit split tokens while retaining semantic identity."""

    dataset_value = _nonblank(dataset, "dataset")
    case_value = _nonblank(case_id, "case_id")

    def retained_parts(value: str) -> List[str]:
        return [
            part.strip()
            for part in value.split(":")
            if part.strip() and part.strip().casefold() not in _SPLIT_IDENTITY_TOKENS
        ]

    dataset_parts = retained_parts(dataset_value)
    case_parts = retained_parts(case_value)
    if not dataset_parts or not case_parts:
        raise AuditInputError("dataset/case identity cannot consist only of split tokens")
    prefix = [part.casefold() for part in dataset_parts]
    current = [part.casefold() for part in case_parts[: len(dataset_parts)]]
    canonical_parts = case_parts if current == prefix else dataset_parts + case_parts
    return ":".join(canonical_parts)


def _is_candidate_list(payload: Any) -> bool:
    return (
        isinstance(payload, list)
        and bool(payload)
        and all(
            isinstance(item, dict)
            and "unit_id" in item
            and "source_type" in item
            for item in payload
        )
    )


def _read_artifact_records(path: Path) -> Iterator[Tuple[int, Any]]:
    _reject_formal_data_path(path)
    if path.suffix.casefold() == ".jsonl":
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                yield index, json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditInputError(f"malformed JSONL record in {path.name}") from exc
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"malformed JSON artifact: {path.name}") from exc
    if isinstance(payload, list) and not _is_candidate_list(payload):
        for index, item in enumerate(payload):
            yield index, item
    else:
        yield 0, payload


def _path_is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _iter_artifact_paths(
    root: DiscoveryRoot,
    excluded_paths: Sequence[Path] = (),
) -> Iterator[Path]:
    if not root.path.is_dir():
        raise AuditInputError(f"discovery root is not a directory: {root.path.name}")
    excluded = tuple(Path(path).expanduser().resolve() for path in excluded_paths)
    if (
        "selector_fidelity_audit" in {part.casefold() for part in root.path.parts}
        or any(_path_is_within(root.path, path) for path in excluded)
    ):
        return
    for directory, directories, filenames in os.walk(root.path):
        base = Path(directory).resolve()
        directories[:] = sorted(
            name
            for name in directories
            if name.casefold()
            not in {"validation", "test", "selector_fidelity_audit"}
            and not any(_path_is_within((base / name).resolve(), path) for path in excluded)
        )
        for filename in sorted(filenames):
            path = base / filename
            if (
                path.suffix.casefold() in {".json", ".jsonl"}
                and not any(_path_is_within(path.resolve(), item) for item in excluded)
            ):
                _reject_formal_data_path(path)
                yield path


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _is_clean_probe_text(text: str) -> bool:
    normalized = _normalized_text(text)
    return len(normalized) >= 4 and any(character.isalnum() for character in normalized)


def _pool_eligibility(units: Sequence[RuntimeUnit]) -> Optional[str]:
    if not 6 <= len(units) <= 24:
        return "candidate_count_outside_6_to_24"
    counts = Counter(unit.source_type for unit in units)
    if counts[SourceType.TRANSCRIPT] < 2:
        return "fewer_than_two_transcript_units"
    if counts[SourceType.OCR] < 2:
        return "fewer_than_two_ocr_units"
    for source_type in (SourceType.TRANSCRIPT, SourceType.OCR):
        clean_count = sum(
            unit.source_type is source_type and _is_clean_probe_text(unit.text)
            for unit in units
        )
        if clean_count < PROBES_PER_MODALITY_PER_CASE:
            return f"fewer_than_two_clean_{source_type.value}_units"
    return None


def _safe_case_id(value: Any, artifact_key: str) -> str:
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return "artifact-" + hashlib.sha256(artifact_key.encode("utf-8")).hexdigest()[:12]


def discover_eligible_cases(
    roots: Sequence[DiscoveryRoot],
    excluded_paths: Sequence[Path] = (),
) -> Tuple[Dict[str, Any], List[CandidateCase]]:
    """Discover cases without consulting scores, logits, Top-k, or predictions."""

    if not roots:
        raise AuditInputError("at least one non-Test discovery root is required")
    inspected_records = 0
    unreadable_records = 0
    excluded = Counter()
    eligible: List[CandidateCase] = []
    for root in sorted(roots, key=lambda item: (item.dataset.casefold(), str(item.path))):
        for path in _iter_artifact_paths(root, excluded_paths):
            relative = path.relative_to(root.path).as_posix()
            try:
                records = list(_read_artifact_records(path))
            except (AuditInputError, OSError, UnicodeError):
                unreadable_records += 1
                continue
            for record_index, payload in records:
                inspected_records += 1
                artifact_key = f"{root.dataset}:{relative}#{record_index}"
                try:
                    artifact_contract = detect_artifact_contract(payload)
                    units = reconstruct_candidate_pool(payload, artifact_contract)
                    claim = _nonblank(_case_scalar(payload, ("claim",)), "source claim")
                except (AuditInputError, KeyError, TypeError, ValueError):
                    excluded["not_reconstructable"] += 1
                    continue
                reason = _pool_eligibility(units)
                if reason is not None:
                    excluded[reason] += 1
                    continue
                pool_hash = candidate_pool_sha256(units)
                dataset_value = _case_scalar(payload, ("dataset", "source_dataset"))
                dataset = (
                    str(dataset_value).strip()
                    if isinstance(dataset_value, str) and dataset_value.strip()
                    else root.dataset
                )
                case_value = _case_scalar(
                    payload,
                    ("session_id", "case_id", "job_id"),
                )
                case_id = _safe_case_id(case_value, artifact_key)
                eligible.append(
                    CandidateCase(
                        dataset=dataset,
                        case_id=case_id,
                        canonical_underlying_case_id=canonicalize_underlying_case_id(
                            dataset, case_id
                        ),
                        artifact_contract=artifact_contract,
                        artifact_key=artifact_key,
                        artifact_path=path,
                        record_index=record_index,
                        claim=claim,
                        units=tuple(units),
                        candidate_pool_sha256=pool_hash,
                        payload=payload,
                    )
                )
    eligible.sort(key=_case_sort_key)
    inventory = {
        "schema_version": 1,
        "discovery_method": "score_blind_dual_contract_artifact_scan",
        "formal_test_accessed": False,
        "selector_outputs_inspected_for_case_selection": False,
        "inspected_record_count": inspected_records,
        "unreadable_artifact_count": unreadable_records,
        "self_discovery_excluded": True,
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "eligible_case_count": len(eligible),
        "eligible_cases": [_public_case_record(case) for case in eligible],
    }
    return inventory, eligible


def _case_sort_key(case: CandidateCase) -> Tuple[str, str, str, str, str]:
    return (
        case.dataset.casefold(),
        case.canonical_underlying_case_id.casefold(),
        case.case_id.casefold(),
        case.candidate_pool_sha256,
        case.artifact_key,
    )


def select_cases_score_blind(
    cases: Sequence[CandidateCase],
    target_count: int = TARGET_CASE_COUNT,
) -> List[CandidateCase]:
    """Prefer dataset diversity, then fill by a stable metadata-only order."""

    if type(target_count) is not int or target_count <= 0:
        raise AuditInputError("target case count must be a positive integer")
    ordered = sorted(cases, key=_case_sort_key)
    independent: List[CandidateCase] = []
    seen_underlying_ids = set()
    seen_hashes = set()
    for case in ordered:
        underlying_key = case.canonical_underlying_case_id.casefold()
        if underlying_key in seen_underlying_ids:
            continue
        if case.candidate_pool_sha256 in seen_hashes:
            continue
        independent.append(case)
        seen_underlying_ids.add(underlying_key)
        seen_hashes.add(case.candidate_pool_sha256)
    first_by_dataset: Dict[str, CandidateCase] = {}
    for case in independent:
        first_by_dataset.setdefault(case.dataset.casefold(), case)
    selected = [first_by_dataset[key] for key in sorted(first_by_dataset)][:target_count]
    selected_hashes = {case.candidate_pool_sha256 for case in selected}
    selected_underlying_ids = {
        case.canonical_underlying_case_id.casefold() for case in selected
    }
    for case in independent:
        if len(selected) >= target_count:
            break
        if (
            case.candidate_pool_sha256 not in selected_hashes
            and case.canonical_underlying_case_id.casefold()
            not in selected_underlying_ids
        ):
            selected.append(case)
            selected_hashes.add(case.candidate_pool_sha256)
            selected_underlying_ids.add(case.canonical_underlying_case_id.casefold())
    return selected


def _public_case_record(case: CandidateCase) -> Dict[str, Any]:
    counts = Counter(unit.source_type.value for unit in case.units)
    return {
        "case_id": case.case_id,
        "canonical_underlying_case_id": case.canonical_underlying_case_id,
        "dataset": case.dataset,
        "artifact_contract": case.artifact_contract,
        "artifact_key": case.artifact_key,
        "candidate_pool_sha256": case.candidate_pool_sha256,
        "candidate_count": len(case.units),
        "candidate_unit_ids_in_exposure_order": list(case.candidate_ids),
        "source_type_counts": dict(sorted(counts.items())),
    }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditInputError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AuditInputError(f"{field} must be finite")
    return number


def _numeric_difference(actual: Any, expected: float, field: str) -> float:
    actual_value = _finite_number(actual, field)
    return float(abs(Decimal(str(actual_value)) - Decimal(str(expected))))


def _public_top_k_ids(payload: Any) -> Tuple[str, ...]:
    current = _unwrap_result(payload)
    candidates: List[Any] = []
    if isinstance(current, dict):
        evidence = current.get("evidence")
        if isinstance(evidence, dict):
            candidates.append(evidence.get("g1_top_k_explanation_unit_ids"))
        candidates.extend(
            [
                current.get("g1_top_k_explanation_unit_ids"),
                current.get("top_k_units"),
            ]
        )
    if isinstance(payload, dict):
        candidates.append(payload.get("top_k_units"))
    for value in candidates:
        if not isinstance(value, list):
            continue
        ids: List[str] = []
        for item in value:
            unit_id = item.get("unit_id") if isinstance(item, dict) else item
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise AuditInputError("reference Top-k contains an invalid unit ID")
            ids.append(unit_id.strip())
        if ids:
            if len(set(ids)) != len(ids):
                raise AuditInputError("reference Top-k contains duplicate unit IDs")
            return tuple(ids)
    raise AuditInputError("reference Top-k unit IDs are unavailable")


def _native_top_k_ids(payload: Any) -> Tuple[str, ...]:
    if not isinstance(payload, dict):
        raise AuditInputError("native Phase4A reference must be an object")
    value = payload.get("top_k_selection_units")
    if not isinstance(value, list) or not value:
        raise AuditInputError("native top_k_selection_units must be non-empty")
    ids: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise AuditInputError(f"top_k_selection_units[{index}] must be an object")
        ids.append(
            _nonblank(
                item.get("unit_id"),
                f"top_k_selection_units[{index}].unit_id",
            )
        )
    if len(set(ids)) != len(ids):
        raise AuditInputError("native top_k_selection_units contains duplicate IDs")
    return tuple(ids)


def extract_reference_snapshot(case: CandidateCase) -> ReferenceSnapshot:
    if case.artifact_contract == NATIVE_PHASE4A:
        if not isinstance(case.payload, dict):
            raise AuditInputError("native Phase4A reference must be an object")
        records = case.payload.get("unit_outputs")
        if not isinstance(records, list):
            raise AuditInputError("native Phase4A unit_outputs must be a list")
        logits_field = "veracity_logits"
        top_k_unit_ids = _native_top_k_ids(case.payload)
    elif case.artifact_contract == PUBLIC_RESULT:
        records = _candidate_records(case.payload)
        logits_field = "logits"
        top_k_unit_ids = _public_top_k_ids(case.payload)
    else:
        raise AuditInputError("unsupported selector audit artifact contract")
    selection_scores: Dict[str, float] = {}
    fake_logits: Dict[str, float] = {}
    real_logits: Dict[str, float] = {}
    candidate_ids: List[str] = []
    for index, record in enumerate(records):
        unit_id = _nonblank(record.get("unit_id"), f"reference[{index}].unit_id")
        candidate_ids.append(unit_id)
        selection_scores[unit_id] = _finite_number(
            record.get("selection_score"),
            f"reference[{index}].selection_score",
        )
        logits = record.get(logits_field)
        if not isinstance(logits, dict) or set(logits) != {"fake", "real"}:
            raise AuditInputError(f"reference[{index}] requires fake/real logits")
        fake_logits[unit_id] = _finite_number(
            logits["fake"], f"reference[{index}].logits.fake"
        )
        real_logits[unit_id] = _finite_number(
            logits["real"], f"reference[{index}].logits.real"
        )
    return ReferenceSnapshot(
        claim=case.claim,
        candidate_ids=tuple(candidate_ids),
        selection_scores=selection_scores,
        fake_logits=fake_logits,
        real_logits=real_logits,
        top_k_unit_ids=top_k_unit_ids,
    )


def _session_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized[:48] or "case"


def run_reproduction_gate(case: CandidateCase, runner: Any) -> Dict[str, Any]:
    base = {
        "case_id": case.case_id,
        "canonical_underlying_case_id": case.canonical_underlying_case_id,
        "dataset": case.dataset,
        "artifact_contract": case.artifact_contract,
        "candidate_pool_sha256": case.candidate_pool_sha256,
        "tolerance": 1e-6,
        "passed": False,
        "candidate_ids_identical": False,
        "candidate_order_identical": False,
        "max_selection_score_difference": None,
        "max_fake_logit_difference": None,
        "max_real_logit_difference": None,
        "top_k_unit_ids_identical": False,
        "failure_reason": None,
    }
    try:
        reference = extract_reference_snapshot(case)
        if reference.candidate_ids != case.candidate_ids:
            raise AuditInputError("reference candidate IDs/order differ from discovered pool")
        units = _fresh_units(case.units)
        result = runner.run(
            "selector_cross_repro_"
            + _session_component(case.case_id)
            + "_"
            + case.candidate_pool_sha256[:12],
            reference.claim,
            units,
        )
        actual_ids = tuple(unit.unit_id for unit in result.all_units)
        base["candidate_ids_identical"] = set(actual_ids) == set(reference.candidate_ids)
        base["candidate_order_identical"] = actual_ids == reference.candidate_ids
        actual_by_id = {unit.unit_id: unit for unit in result.all_units}
        if not base["candidate_ids_identical"]:
            raise AuditInputError("reproduction returned different candidate IDs")
        score_differences = []
        fake_differences = []
        real_differences = []
        for unit_id in reference.candidate_ids:
            unit = actual_by_id[unit_id]
            if unit.selection_score is None or unit.logits is None:
                raise AuditInputError("reproduction omitted selector outputs")
            score_differences.append(
                _numeric_difference(
                    unit.selection_score,
                    reference.selection_scores[unit_id],
                    f"reproduction[{unit_id}].selection_score",
                )
            )
            fake_differences.append(
                _numeric_difference(
                    unit.logits["fake"],
                    reference.fake_logits[unit_id],
                    f"reproduction[{unit_id}].logits.fake",
                )
            )
            real_differences.append(
                _numeric_difference(
                    unit.logits["real"],
                    reference.real_logits[unit_id],
                    f"reproduction[{unit_id}].logits.real",
                )
            )
        base["max_selection_score_difference"] = max(score_differences, default=0.0)
        base["max_fake_logit_difference"] = max(fake_differences, default=0.0)
        base["max_real_logit_difference"] = max(real_differences, default=0.0)
        actual_top_k = tuple(unit.unit_id for unit in result.top_k_units)
        base["top_k_unit_ids_identical"] = actual_top_k == reference.top_k_unit_ids
        base["passed"] = bool(
            base["candidate_order_identical"]
            and base["top_k_unit_ids_identical"]
            and base["max_selection_score_difference"] <= 1e-6
            and base["max_fake_logit_difference"] <= 1e-6
            and base["max_real_logit_difference"] <= 1e-6
        )
        if not base["passed"]:
            base["failure_reason"] = "reproduction_tolerance_or_top_k_mismatch"
    except (AuditInputError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        base["failure_reason"] = str(exc)
    return base


def _direct_phrase(text: str) -> str:
    normalized = _normalized_text(text)
    if len(normalized) <= 120:
        return normalized
    words = normalized.split()
    phrase = " ".join(words[:16])
    return phrase[:120].rstrip()


def generate_direct_grounding_probes(case: CandidateCase) -> List[ProbeDefinition]:
    """Generate balanced direct-grounding probes without selector outputs."""

    probes: List[ProbeDefinition] = []
    for source_type, expected_modality in (
        (SourceType.OCR, "OCR"),
        (SourceType.TRANSCRIPT, "TRANSCRIPT"),
    ):
        choices: List[Tuple[RuntimeUnit, str]] = []
        seen_phrases = set()
        for unit in case.units:
            if unit.source_type is not source_type or not _is_clean_probe_text(unit.text):
                continue
            phrase = _direct_phrase(unit.text)
            key = phrase.casefold()
            if key in seen_phrases:
                continue
            seen_phrases.add(key)
            choices.append((unit, phrase))
            if len(choices) == PROBES_PER_MODALITY_PER_CASE:
                break
        if len(choices) != PROBES_PER_MODALITY_PER_CASE:
            raise AuditInputError(
                f"case {case.case_id} lacks two deterministic {expected_modality} probes"
            )
        for probe_index, (_, phrase) in enumerate(choices, 1):
            relevant_ids = tuple(
                unit.unit_id
                for unit in case.units
                if unit.source_type is source_type
                and phrase.casefold() in _normalized_text(unit.text).casefold()
            )
            quoted = phrase.replace('"', "'")
            claim = (
                f'The on-screen text reads "{quoted}".'
                if expected_modality == "OCR"
                else f'The speaker says "{quoted}".'
            )
            probes.append(
                ProbeDefinition(
                    probe_id=(
                        _session_component(case.case_id).casefold()
                        + f"_{expected_modality.casefold()}_{probe_index:02d}"
                        + "_"
                        + case.candidate_pool_sha256[:8]
                    ),
                    claim=claim,
                    expected_modality=expected_modality,
                    expected_relevant_unit_ids=relevant_ids,
                    direct_grounding_unit_ids=relevant_ids,
                    annotation_status="audited",
                    annotation_basis=(
                        "Score-blind deterministic exact-phrase grounding over the "
                        f"ordered {expected_modality} candidate units."
                    ),
                )
            )
    return probes


def build_pre_scoring_manifest(
    cases: Sequence[CandidateCase],
    probes_by_hash: Mapping[str, Sequence[ProbeDefinition]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_name": "cross_case_frozen_g1_selector_robustness",
        "generation_stage": "FROZEN_BEFORE_MODEL_SCORING",
        "formal_test_accessed": False,
        "probe_generation_used_selector_outputs": False,
        "case_count": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "canonical_underlying_case_id": case.canonical_underlying_case_id,
                "dataset": case.dataset,
                "artifact_contract": case.artifact_contract,
                "candidate_pool_sha256": case.candidate_pool_sha256,
                "candidate_unit_ids_in_exposure_order": list(case.candidate_ids),
                "probes": [
                    {
                        "probe_id": probe.probe_id,
                        "claim": probe.claim,
                        "expected_modality": probe.expected_modality,
                        "relevant_unit_ids": list(probe.expected_relevant_unit_ids),
                        "direct_grounding_unit_ids": list(probe.direct_grounding_unit_ids),
                        "annotation_generation_basis": probe.annotation_basis,
                    }
                    for probe in probes_by_hash[case.candidate_pool_sha256]
                ],
            }
            for case in cases
        ],
    }


def _manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_manifest_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_manifest_bytes(payload))


def _write_candidate_exports(root: Path, cases: Sequence[CandidateCase]) -> None:
    target = root / "candidate_pools"
    target.mkdir(parents=True, exist_ok=True)
    for case in cases:
        name = _session_component(case.case_id).casefold()
        payload = {
            "schema_version": 1,
            "case_id": case.case_id,
            "canonical_underlying_case_id": case.canonical_underlying_case_id,
            "dataset": case.dataset,
            "artifact_contract": case.artifact_contract,
            "candidate_pool_sha256": case.candidate_pool_sha256,
            "g1_exposure_units": [
                {
                    "unit_id": unit.unit_id,
                    "source_type": unit.source_type.value,
                    "text": unit.text,
                    "eligible_for_frozen_g1": True,
                }
                for unit in case.units
            ],
        }
        _write_json(target / f"{name}-{case.candidate_pool_sha256[:12]}.json", payload)


def _write_review(
    path: Path,
    cases: Sequence[CandidateCase],
    probes_by_hash: Mapping[str, Sequence[ProbeDefinition]],
) -> None:
    columns = (
        "case_id",
        "probe_id",
        "claim",
        "unit_id",
        "source_type",
        "unit_text",
        "expected_relevance",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            for probe in probes_by_hash[case.candidate_pool_sha256]:
                relevant = set(probe.expected_relevant_unit_ids)
                for unit in case.units:
                    writer.writerow(
                        {
                            "case_id": case.case_id,
                            "probe_id": probe.probe_id,
                            "claim": probe.claim,
                            "unit_id": unit.unit_id,
                            "source_type": unit.source_type.value,
                            "unit_text": unit.text,
                            "expected_relevance": str(unit.unit_id in relevant).lower(),
                        }
                    )


def _write_rankings(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RANKING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return statistics.fmean(available) if available else None


def _modality_case_metrics(
    records: Sequence[Mapping[str, Any]], modality: str
) -> Dict[str, Any]:
    selected = [item for item in records if item["expected_modality"] == modality]
    source = {"OCR": "ocr", "TRANSCRIPT": "transcript"}[modality]
    hit_values = [
        1.0 if int(item["highest_relevant_unit_rank"]) <= 5 else 0.0
        for item in selected
        if item.get("highest_relevant_unit_rank") is not None
    ]
    top_five_total = sum(len(item["top_5_unit_ids"]) for item in selected)
    source_top_five = sum(
        int(item["top_5_modality_composition"].get(source, 0)) for item in selected
    )
    return {
        "probe_count": len(selected),
        "hit_at_5": _mean(hit_values),
        "mean_best_relevant_rank": _mean(
            item.get("highest_relevant_unit_rank") for item in selected
        ),
        "mrr": _mean(item.get("mrr") for item in selected),
        "ndcg_at_5": _mean(item.get("ndcg_at_5") for item in selected),
        "top_5_share": (
            source_top_five / top_five_total if top_five_total else None
        ),
    }


def compute_cross_case_metrics(
    per_probe: Sequence[Mapping[str, Any]],
    reproduced_case_count: int,
) -> Dict[str, Any]:
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in per_probe:
        by_case[str(item["case_id"])].append(item)
    per_case = []
    for case_id in sorted(by_case):
        records = by_case[case_id]
        ocr = _modality_case_metrics(records, "OCR")
        transcript = _modality_case_metrics(records, "TRANSCRIPT")
        miss_count = sum(bool(item["direct_grounding_flags"]) for item in records)
        ocr_records = [item for item in records if item["expected_modality"] == "OCR"]
        all_transcript_count = sum(
            bool(item["top_5_unit_ids"])
            and set(item["top_5_modality_composition"]) == {"transcript"}
            for item in ocr_records
        )
        per_case.append(
            {
                "case_id": case_id,
                "dataset": records[0]["dataset"],
                "ocr": ocr,
                "transcript": transcript,
                "direct_grounding_miss_rate": miss_count / len(records),
                "delta_hit5": (
                    transcript["hit_at_5"] - ocr["hit_at_5"]
                    if transcript["hit_at_5"] is not None and ocr["hit_at_5"] is not None
                    else None
                ),
                "delta_best_rank": (
                    ocr["mean_best_relevant_rank"]
                    - transcript["mean_best_relevant_rank"]
                    if ocr["mean_best_relevant_rank"] is not None
                    and transcript["mean_best_relevant_rank"] is not None
                    else None
                ),
                "ocr_probe_top5_all_transcript_count": all_transcript_count,
            }
        )
    modality_records = {
        modality: [
            item for item in per_probe if item["expected_modality"] == modality
        ]
        for modality in ("OCR", "TRANSCRIPT")
    }
    micro: Dict[str, Optional[float]] = {}
    for modality, records in modality_records.items():
        micro[modality] = _mean(
            1.0 if int(item["highest_relevant_unit_rank"]) <= 5 else 0.0
            for item in records
            if item.get("highest_relevant_unit_rank") is not None
        )
    macro_ocr = _mean(item["ocr"]["hit_at_5"] for item in per_case)
    macro_transcript = _mean(item["transcript"]["hit_at_5"] for item in per_case)
    top_five_composition = Counter()
    for item in per_probe:
        top_five_composition.update(item["top_5_modality_composition"])
    top_five_count = sum(top_five_composition.values())
    miss_count = sum(bool(item["direct_grounding_flags"]) for item in per_probe)
    ocr_miss_count = sum(
        bool(item["direct_grounding_flags"])
        for item in modality_records["OCR"]
    )
    aggregate: Dict[str, Any] = {
        "reproduced_case_count": reproduced_case_count,
        "probe_count": len(per_probe),
        "ocr_probe_count": len(modality_records["OCR"]),
        "transcript_probe_count": len(modality_records["TRANSCRIPT"]),
        "per_case": per_case,
        "micro_ocr_hit_at_5": micro["OCR"],
        "macro_ocr_hit_at_5": macro_ocr,
        "micro_transcript_hit_at_5": micro["TRANSCRIPT"],
        "macro_transcript_hit_at_5": macro_transcript,
        "micro_hit_at_5_gap": (
            micro["TRANSCRIPT"] - micro["OCR"]
            if micro["TRANSCRIPT"] is not None and micro["OCR"] is not None
            else None
        ),
        "macro_hit_at_5_gap": (
            macro_transcript - macro_ocr
            if macro_transcript is not None and macro_ocr is not None
            else None
        ),
        "mean_best_relevant_rank_by_modality": {
            modality: _mean(
                item.get("highest_relevant_unit_rank") for item in records
            )
            for modality, records in modality_records.items()
        },
        "mrr_by_modality": {
            modality: _mean(item.get("mrr") for item in records)
            for modality, records in modality_records.items()
        },
        "ndcg_at_5_by_modality": {
            modality: _mean(item.get("ndcg_at_5") for item in records)
            for modality, records in modality_records.items()
        },
        "top_5_modality_composition": dict(sorted(top_five_composition.items())),
        "top_5_modality_share": {
            source: count / top_five_count
            for source, count in sorted(top_five_composition.items())
        },
        "direct_grounding_miss_rate": (
            miss_count / len(per_probe) if per_probe else None
        ),
        "ocr_direct_grounding_miss_rate": (
            ocr_miss_count / len(modality_records["OCR"])
            if modality_records["OCR"]
            else None
        ),
        "pool_fraction_with_ocr_hit_below_transcript": (
            sum(
                item["ocr"]["hit_at_5"] < item["transcript"]["hit_at_5"]
                for item in per_case
                if item["ocr"]["hit_at_5"] is not None
                and item["transcript"]["hit_at_5"] is not None
            )
            / len(per_case)
            if per_case
            else None
        ),
    }
    aggregate["classification"] = classify_cross_case(aggregate)
    return aggregate


def classify_cross_case(aggregate: Mapping[str, Any]) -> str:
    reproduced = int(aggregate.get("reproduced_case_count", 0))
    probe_count = int(aggregate.get("probe_count", 0))
    if reproduced < MIN_CONCLUSIVE_CASE_COUNT or probe_count < 16:
        return "INCONCLUSIVE"
    pool_fraction = aggregate.get("pool_fraction_with_ocr_hit_below_transcript")
    ocr_hit = aggregate.get("micro_ocr_hit_at_5")
    transcript_hit = aggregate.get("micro_transcript_hit_at_5")
    ocr_miss_rate = aggregate.get("ocr_direct_grounding_miss_rate")
    if (
        isinstance(pool_fraction, (int, float))
        and isinstance(ocr_hit, (int, float))
        and isinstance(transcript_hit, (int, float))
        and isinstance(ocr_miss_rate, (int, float))
        and pool_fraction >= 0.75
        and ocr_hit <= 0.50
        and transcript_hit - ocr_hit >= 0.25
        and ocr_miss_rate >= 0.50
    ):
        return "CROSS_CASE_MODALITY_BIAS_CONFIRMED"
    if (
        isinstance(ocr_hit, (int, float))
        and isinstance(transcript_hit, (int, float))
        and ocr_hit <= 0.50
        and transcript_hit <= 0.50
    ):
        return "GENERAL_SELECTOR_RELEVANCE_FAILURE"
    return "CASE_LOCAL_OR_MIXED_EFFECT"


def _score_probes(
    cases: Sequence[CandidateCase],
    probes_by_hash: Mapping[str, Sequence[ProbeDefinition]],
    runner: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_probe: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for case in cases:
        for probe in probes_by_hash[case.candidate_pool_sha256]:
            units = _fresh_units(case.units)
            result = runner.run(
                "selector_cross_probe_"
                + _session_component(probe.probe_id)
                + "_"
                + case.candidate_pool_sha256[:12],
                probe.claim,
                units,
            )
            ranked = derive_ranked_units(
                result.all_units,
                [unit.unit_id for unit in result.top_k_units],
            )
            metrics = compute_probe_metrics(probe, ranked)
            metrics["case_id"] = case.case_id
            metrics["dataset"] = case.dataset
            per_probe.append(metrics)
            for item in ranked:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "dataset": case.dataset,
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
    return per_probe, rows


def _summary_markdown(
    metrics: Mapping[str, Any],
    inventory: Mapping[str, Any],
    selected: Sequence[CandidateCase],
    reproduction: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Cross-Case Frozen G1 Selector Robustness Audit",
        "",
        f"- Audit status: `{metrics['audit_status']}`",
        f"- Classification: **{metrics['classification']}**",
        f"- Eligible cases discovered: {inventory['eligible_case_count']}",
        f"- Cases selected score-blind: {len(selected)}",
        f"- Cases passing reproduction: {metrics['reproduced_case_count']}",
        f"- Formal Test accessed: `{str(metrics['formal_test_accessed']).lower()}`",
        "- Selection rule: require unique canonical underlying identity and candidate hash; prefer one lexical case per dataset, then stable metadata order.",
        "",
        "## Reproduction gates",
        "",
    ]
    if reproduction:
        for item in reproduction:
            lines.append(
                f"- `{item['dataset']} / {item['case_id']}`: "
                + ("PASS" if item["passed"] else f"FAIL ({item['failure_reason']})")
            )
    else:
        lines.append("- No eligible local case reached the DICC reproduction boundary.")
    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            f"- OCR probes: {metrics['ocr_probe_count']}",
            f"- Transcript probes: {metrics['transcript_probe_count']}",
            f"- Micro OCR Hit@5: {metrics['micro_ocr_hit_at_5']}",
            f"- Micro Transcript Hit@5: {metrics['micro_transcript_hit_at_5']}",
            f"- Macro OCR Hit@5: {metrics['macro_ocr_hit_at_5']}",
            f"- Macro Transcript Hit@5: {metrics['macro_transcript_hit_at_5']}",
            f"- MRR by modality: {metrics['mrr_by_modality']}",
            f"- NDCG@5 by modality: {metrics['ndcg_at_5_by_modality']}",
            f"- Direct-grounding miss rate: {metrics['direct_grounding_miss_rate']}",
            "",
            "The audit is descriptive only. Top-k remains explanation-only, and all-unit frozen class-wise max pooling remains the prediction rule.",
            "",
        ]
    )
    return "\n".join(lines)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "reproduced_case_count": 0,
        "probe_count": 0,
        "ocr_probe_count": 0,
        "transcript_probe_count": 0,
        "per_case": [],
        "micro_ocr_hit_at_5": None,
        "macro_ocr_hit_at_5": None,
        "micro_transcript_hit_at_5": None,
        "macro_transcript_hit_at_5": None,
        "micro_hit_at_5_gap": None,
        "macro_hit_at_5_gap": None,
        "mean_best_relevant_rank_by_modality": {"OCR": None, "TRANSCRIPT": None},
        "mrr_by_modality": {"OCR": None, "TRANSCRIPT": None},
        "ndcg_at_5_by_modality": {"OCR": None, "TRANSCRIPT": None},
        "top_5_modality_composition": {},
        "top_5_modality_share": {},
        "direct_grounding_miss_rate": None,
        "ocr_direct_grounding_miss_rate": None,
        "pool_fraction_with_ocr_hit_below_transcript": None,
        "classification": "INCONCLUSIVE",
    }


def run_cross_case_audit(
    *,
    discovery_roots: Sequence[DiscoveryRoot],
    runtime_config_path: Path,
    output_dir: Path,
    target_case_count: int = TARGET_CASE_COUNT,
    runner: Any = None,
) -> Dict[str, Any]:
    _reject_formal_data_path(Path(runtime_config_path).expanduser().resolve())
    report_root = Path(output_dir).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    inventory, eligible = discover_eligible_cases(
        discovery_roots,
        excluded_paths=(report_root,),
    )
    selected = select_cases_score_blind(eligible, target_case_count)
    selection_rule = (
        "Require unique canonical_underlying_case_id and model-input candidate hash; "
        "take the lexically first independent case from each dataset, then fill by "
        "(dataset, canonical underlying case ID, case ID, candidate hash, artifact key)."
    )
    selected_manifest = {
        "schema_version": 1,
        "selection_stage": "FROZEN_BEFORE_REPRODUCTION_OR_PROBE_SCORING",
        "selection_rule": selection_rule,
        "target_case_count": target_case_count,
        "selected_case_count": len(selected),
        "formal_test_accessed": False,
        "selector_outputs_inspected_for_case_selection": False,
        "selected_cases": [_public_case_record(case) for case in selected],
    }
    _write_json(report_root / "eligible_case_inventory.json", inventory)
    _write_json(report_root / "selected_case_manifest.json", selected_manifest)
    _write_candidate_exports(report_root, selected)

    effective_runner = runner
    if selected and effective_runner is None:
        effective_runner = _build_runner(runtime_config_path)
    reproduction = [
        run_reproduction_gate(case, effective_runner) for case in selected
    ]
    _write_json(
        report_root / "reproduction_gates.json",
        {
            "schema_version": 1,
            "formal_test_accessed": False,
            "selected_case_count": len(selected),
            "passed_case_count": sum(item["passed"] for item in reproduction),
            "cases": reproduction,
        },
    )
    passed_hashes = {
        item["candidate_pool_sha256"] for item in reproduction if item["passed"]
    }
    reproduced_cases = [
        case for case in selected if case.candidate_pool_sha256 in passed_hashes
    ]
    probes_by_hash = {
        case.candidate_pool_sha256: generate_direct_grounding_probes(case)
        for case in reproduced_cases
    }
    pre_scoring_manifest = build_pre_scoring_manifest(
        reproduced_cases, probes_by_hash
    )
    manifest_path = report_root / "cross_case_probe_manifest_pre_scoring.json"
    manifest_path.write_bytes(_manifest_bytes(pre_scoring_manifest))
    frozen_manifest_hash = manifest_sha256(pre_scoring_manifest)
    (report_root / "cross_case_probe_manifest_pre_scoring.sha256").write_text(
        frozen_manifest_hash
        + "  cross_case_probe_manifest_pre_scoring.json\n",
        encoding="utf-8",
    )
    _write_review(
        report_root / "cross_case_probe_review.csv",
        reproduced_cases,
        probes_by_hash,
    )

    if reproduced_cases:
        per_probe, ranking_rows = _score_probes(
            reproduced_cases, probes_by_hash, effective_runner
        )
        metrics = compute_cross_case_metrics(per_probe, len(reproduced_cases))
    else:
        per_probe, ranking_rows = [], []
        metrics = _empty_metrics()
    if manifest_sha256(json.loads(manifest_path.read_text(encoding="utf-8"))) != frozen_manifest_hash:
        raise RuntimeError("pre-scoring probe manifest changed during scoring")
    _write_rankings(report_root / "cross_case_unit_rankings.csv", ranking_rows)
    metrics.update(
        {
            "schema_version": 1,
            "audit_status": "COMPLETED" if reproduced_cases else "BLOCKED",
            "formal_test_accessed": False,
            "eligible_case_count": inventory["eligible_case_count"],
            "selected_case_count": len(selected),
            "reproduction_gates": reproduction,
            "pre_scoring_manifest_sha256": frozen_manifest_hash,
            "per_probe": per_probe,
            "scientific_boundary": {
                "top_k_is_explanation_only": True,
                "prediction_uses_all_valid_unit_veracity_logits": True,
                "candidate_order_preserved": True,
                "production_or_model_code_changed": False,
            },
        }
    )
    if metrics["classification"] not in CROSS_CASE_CLASSIFICATIONS:
        raise AssertionError("unexpected cross-case classification")
    _write_json(report_root / "cross_case_metrics.json", metrics)
    (report_root / "cross_case_summary.md").write_text(
        _summary_markdown(metrics, inventory, selected, reproduction),
        encoding="utf-8",
    )
    return metrics


def manifest_contains_selector_output_keys(payload: Any) -> bool:
    """Test helper proving the frozen probe manifest contains no score leakage."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.casefold() in _FORBIDDEN_RESULT_KEYS:
                return True
            if manifest_contains_selector_output_keys(value):
                return True
    elif isinstance(payload, list):
        return any(manifest_contains_selector_output_keys(item) for item in payload)
    return False
