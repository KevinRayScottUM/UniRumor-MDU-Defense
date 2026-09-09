"""Frozen preregistration and pure manifest validation; never build or score data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exclusion_contract import ExclusionLock
from .schemas import (
    DATASETS, IMPLEMENTATION_REVISION, LABELS,
    DevelopmentCase, FrozenTrainCase, ProtocolError,
)


PREREGISTRATION_FILENAME = "step2_6r_3c1_preregistration.json"
PREREGISTRATION_PATH = Path(__file__).with_name(PREREGISTRATION_FILENAME)
PREREGISTRATION_SHA256 = "81caac242f486eee630cb34c9009482065bcfab35ac920a799f097d9128bceff"


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("preregistration must contain strict JSON values") from exc


def validate_preregistration(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict) or hashlib.sha256(_canonical_json(payload)).hexdigest() != PREREGISTRATION_SHA256:
        raise ProtocolError("preregistration differs from the exact frozen Step 3C1 contract")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate preregistration JSON key")
        result[key] = value
    return result


def load_preregistration(path: Path = PREREGISTRATION_PATH) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    parts = {re.sub(r"[^a-z0-9]", "", part.casefold()) for part in path.parts}
    if parts & {"validation", "test", "formalvalidation", "formaltest"}:
        raise ProtocolError("Formal Validation/Test protocol path forbidden")
    if path.name != PREREGISTRATION_FILENAME:
        raise ProtocolError("only the static preregistration JSON filename may be read")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("preregistration is missing or malformed") from exc
    validate_preregistration(payload)
    return payload


def protocol_preflight(path: Path = PREREGISTRATION_PATH) -> Mapping[str, Any]:
    payload = load_preregistration(path)
    return {
        "status": "NEXT_REPAIR_PROTOCOL_PREFLIGHT_PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "repair_name": payload["repair_name"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "scope": "PROTOCOL_ONLY",
        "real_exclusion_lock_status": "REQUIRED_NOT_SUPPLIED",
        "real_cohort_ready": False,
        "real_dataset_built": False,
        "reviewer_packets_created": False,
        "model_loaded": False,
        "training_performed": False,
        "selector_scoring_performed": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "sealed_challenge_content_accessed": False,
        "deployment_remains_blocked": True,
    }


def _identity_order(identity: str, dataset: str, purpose: str, salt: str) -> tuple[str, str]:
    # No claim, unit content, relevance/veracity label, or selector score enters sampling.
    digest = hashlib.sha256(_canonical_json([salt, purpose, dataset, identity])).hexdigest()
    return digest, identity


def validate_development_cohort(
    source_inventory: Sequence[FrozenTrainCase],
    proposed_manifest: Sequence[DevelopmentCase],
    exclusions: ExclusionLock,
) -> Mapping[str, Any]:
    """Validate supplied in-memory fixtures/manifests, never construct a dataset."""

    protocol = load_preregistration()
    if not isinstance(exclusions, ExclusionLock):
        raise ProtocolError("complete exclusion lock required; no override")
    if any(not isinstance(case, FrozenTrainCase) for case in source_inventory):
        raise ProtocolError("authoritative Train inventory contract required")
    source = {case.canonical_case_id: case for case in source_inventory}
    if len(source) != len(source_inventory):
        raise ProtocolError("source inventory has duplicate canonical identities")
    if len(proposed_manifest) != 120:
        raise ProtocolError("exactly 120 development cases required; no resampling")
    if any(not isinstance(row, DevelopmentCase) for row in proposed_manifest):
        raise ProtocolError("development manifest contract required")
    identities = [row.case.canonical_case_id for row in proposed_manifest]
    if len(identities) != len(set(identities)):
        raise ProtocolError("duplicate identity or Train/Dev overlap")
    exclusions.reject_overlap(identities)
    for row in proposed_manifest:
        if source.get(row.case.canonical_case_id) != row.case:
            raise ProtocolError("original claim/candidate metadata/content/order differs from source")
    dataset_counts = Counter(row.case.dataset for row in proposed_manifest)
    split_counts = Counter(row.repair_split for row in proposed_manifest)
    group_counts = Counter((row.repair_split, row.case.dataset) for row in proposed_manifest)
    if dataset_counts != {dataset: 60 for dataset in DATASETS}:
        raise ProtocolError("exact 60/60 dataset balance required")
    if split_counts != {"repair_train": 96, "repair_dev": 24}:
        raise ProtocolError("exact 96/24 split required")
    if group_counts != {(split, dataset): count for split, count in
                        (("repair_train", 48), ("repair_dev", 12)) for dataset in DATASETS}:
        raise ProtocolError("exact 48/48 Train and 12/12 Dev balance required")
    sampling = protocol["development_cohort"]["sampling"]
    expected_splits = {}
    allowed = set(source) - exclusions.all_identities
    for dataset in DATASETS:
        pool = [identity for identity in allowed if source[identity].dataset == dataset]
        if len(pool) < 60:
            raise ProtocolError("insufficient unexcluded source cases; no fallback")
        selected = sorted(pool, key=lambda identity: _identity_order(
            identity, dataset, sampling["selection_purpose"], sampling["salt"]
        ))[:60]
        split_order = sorted(selected, key=lambda identity: _identity_order(
            identity, dataset, sampling["split_purpose"], sampling["salt"]
        ))
        expected_splits.update({identity: "repair_train" if index < 48 else "repair_dev"
                                for index, identity in enumerate(split_order)})
    actual_splits = {row.case.canonical_case_id: row.repair_split for row in proposed_manifest}
    if actual_splits != expected_splits:
        raise ProtocolError("manifest violates frozen deterministic hash membership/split")
    return {
        "status": "DEVELOPMENT_MANIFEST_CONTRACT_PASS",
        "case_count": 120,
        "dataset_counts": dict(dataset_counts),
        "split_counts": dict(split_counts),
        "exclusion_accounting": exclusions.accounting(tuple(source)),
        "original_claims_and_candidate_fields_unchanged": True,
        "real_dataset_built": False,
    }


def validate_coverage(
    source_inventory: Sequence[FrozenTrainCase],
    proposed_manifest: Sequence[DevelopmentCase],
    exclusions: ExclusionLock,
    final_labels: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Any]:
    """Check supplied synthetic final labels; do not generate or adjudicate labels."""

    validate_development_cohort(source_inventory, proposed_manifest, exclusions)
    protocol = load_preregistration()
    if set(final_labels) != {row.case.canonical_case_id for row in proposed_manifest}:
        raise ProtocolError("final labels must retain every frozen development case")
    counts = Counter({"overall": 0, "repair_train": 0, "repair_dev": 0})
    zero_direct_ids = []
    for row in proposed_manifest:
        labels = final_labels[row.case.canonical_case_id]
        if not isinstance(labels, Mapping) or set(labels) != {
            candidate.unit_id for candidate in row.case.candidate_units
        }:
            raise ProtocolError("final labels must preserve every frozen candidate ID")
        if any(not isinstance(label, str) or label not in LABELS for label in labels.values()):
            raise ProtocolError("final labels require the four-class human relevance contract")
        if "DIRECT" in labels.values():
            counts["overall"] += 1
            counts[row.repair_split] += 1
        else:
            zero_direct_ids.append(row.case.canonical_case_id)
    checks = {name: counts[name] >= minimum for name, minimum in
              protocol["coverage"]["minimum_evaluable"].items()}
    return {
        "coverage_pass": all(checks.values()),
        "repair_training_blocked": not all(checks.values()),
        "checks": checks,
        "evaluable_counts": dict(counts),
        "retained_case_count": 120,
        "zero_direct_case_ids": sorted(zero_direct_ids),
        "zero_direct_in_ranking_denominator": False,
        "resampling_performed": False,
    }
