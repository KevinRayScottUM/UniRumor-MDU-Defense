"""Preflight, frozen identity sampling and atomically published cohort artifacts."""

import hashlib
import sys
from collections import Counter
from pathlib import Path

from scripts.selector_relevance_next_repair_protocol.protocol import validate_development_cohort
from scripts.selector_relevance_next_repair_protocol.schemas import Candidate, DevelopmentCase, FrozenTrainCase

from . import schemas
from .artifacts import (assert_new_output, compact, digest_bytes, freeze,
                        json_bytes, safe_path)
from .blinding import packets
from .exclusion_loader import exclusion_payload, load_exclusions, old_provenance
from .schemas import CohortError, IMPLEMENTATION_REVISION, Selection
from .source_loader import Phase4ANormalizationExposureAdapter, closure, inventory, resolve_source
from .source_loader import TrainSourceLedger as Ledger

PREFLIGHT_STATUS = "NEXT_REPAIR_COHORT_SOURCE_PREFLIGHT_PASS"
PASS_STATUS = "NEXT_REPAIR_DEVELOPMENT_COHORT_BUILD_PASS"
BLOCKED_STATUS = "NEXT_REPAIR_DEVELOPMENT_COHORT_BUILD_BLOCKED"


def boundary_flags():
    return {"label_count": 0, "selector_score_count": 0,
            "model_loaded": False, "training_started": False,
            "selector_scoring_performed": False, "formal_validation_accessed": False,
            "formal_test_accessed": False, "sealed_challenge_content_accessed": False,
            "revealed_audit_ranking_content_accessed": False, "production_or_ui_changed": False,
            "deployment_remains_blocked": True, "resampling_performed": False}


def lock_implementation(ledger):
    package = Path(__file__).resolve().parent
    scripts = package.parent
    for path in sorted(package.glob("*.py")):
        ledger.add(path, "cohort builder implementation")
    for relative in (
        "selector_relevance_next_repair_protocol/__init__.py",
        "selector_relevance_next_repair_protocol/protocol.py",
        "selector_relevance_next_repair_protocol/schemas.py",
        "selector_relevance_next_repair_protocol/exclusion_contract.py",
        "selector_relevance_calibration/__init__.py",
        "selector_relevance_calibration/dataset_builder.py",
        "selector_fidelity_audit/__init__.py", "selector_fidelity_audit/cross_case.py",
        "selector_fidelity_audit/audit.py", "selector_relevance_independent_audit/__init__.py",
        "selector_relevance_independent_audit/schemas.py",
    ):
        ledger.add(scripts / relative, "reused implementation source")


def prepare(inputs):
    # Validate EVERY supplied path before any input is opened; directory
    # arguments cannot be repurposed to sneak in an old ranking artifact.
    for path in (inputs.project_root, inputs.phase4a_config, inputs.source_3b1_cohort_dir,
                 inputs.step3b3_closure_dir, inputs.neutral_dir, inputs.stage_a_invariance_report,
                 *inputs.future_exclusion_manifests):
        safe_path(path)
    ledger = Ledger()
    ledger.add(schemas.PREREGISTRATION_PATH, "tracked frozen 3C1 preregistration")
    protocol = schemas.load_preregistration()
    # This stage extracts only construction rules, never prior_result metrics.
    lock_implementation(ledger)
    closure(ledger, inputs.step3b3_closure_dir)
    old_lock, audit_ids, audit_sha = old_provenance(inputs, ledger)
    source = resolve_source(inputs, ledger, old_lock, protocol)
    exclusions = load_exclusions(inputs, ledger, protocol, old_lock, audit_ids, audit_sha)
    rows, _, counts = inventory(source, exclusions, protocol)
    excluded = exclusion_payload(exclusions, [r["canonical_case_id"] for r in rows])
    source_lock = {"implementation_revision": IMPLEMENTATION_REVISION,
                   "preregistration_sha256": schemas.PREREGISTRATION_SHA256,
                   **ledger.payload()}
    source_bytes, exclusion_bytes = json_bytes(source_lock), json_bytes(excluded)
    report = {
        "status": PREFLIGHT_STATUS, "implementation_revision": IMPLEMENTATION_REVISION,
        "repair_name": protocol["repair_name"],
        "preregistration_sha256": schemas.PREREGISTRATION_SHA256,
        "authoritative_train_sha256": schemas.AUTHORITATIVE_TRAIN_SHA256,
        "source_case_count": counts["source_case_count"],
        "source_dataset_counts": counts["source_dataset_counts"],
        "cohort_source_lock_sha256": digest_bytes(source_bytes),
        "exclusion_source_lock_sha256": digest_bytes(exclusion_bytes),
        "phase4a_normalization_imported": False, "phase4a_exposure_performed": False,
        "real_cohort_written": False, "reviewer_packets_written": False,
        "exclusion_accounting": {k: v for k, v in excluded.items() if k != "manifests"},
        **boundary_flags(),
    }
    ledger.revalidate()
    return protocol, source, exclusions, ledger, source_lock, excluded, report


def preflight(inputs, output_dir):
    output = assert_new_output(output_dir)
    _, _, _, ledger, source_lock, excluded, report = prepare(inputs)
    freeze(output, {
        "cohort_source_preflight_report.json": json_bytes(report),
        "cohort_source_lock.json": json_bytes(source_lock),
        "exclusion_source_lock.json": json_bytes(excluded),
    }, ledger)
    return report


def identity_hash(protocol, purpose, dataset, canonical):
    sampling = protocol["development_cohort"]["sampling"]
    return hashlib.sha256(compact([sampling["salt"], purpose, dataset, canonical])).hexdigest()


def select(eligible, exclusions, protocol):
    if len({c.canonical_case_id for c in eligible}) != len(eligible):
        raise CohortError("duplicate identity in eligible inventory")
    exclusions.reject_overlap([c.canonical_case_id for c in eligible])
    config = protocol["development_cohort"]
    sampling = config["sampling"]
    chosen = []
    for dataset, target in config["dataset_counts"].items():
        pool = [case for case in eligible if case.dataset == dataset]
        if len(pool) < target:
            return None
        ranked = sorted(pool, key=lambda c: (
            identity_hash(protocol, sampling["selection_purpose"], dataset, c.canonical_case_id),
            c.canonical_case_id))[:target]
        split_ranked = sorted(ranked, key=lambda c: (
            identity_hash(protocol, sampling["split_purpose"], dataset, c.canonical_case_id),
            c.canonical_case_id))
        train_count = config["split_dataset_counts"]["repair_train"][dataset]
        for index, case in enumerate(split_ranked):
            chosen.append(Selection(
                case, "repair_train" if index < train_count else "repair_dev",
                identity_hash(protocol, sampling["selection_purpose"], dataset, case.canonical_case_id),
                identity_hash(protocol, sampling["split_purpose"], dataset, case.canonical_case_id)))
    validate_selection(eligible, chosen, exclusions, protocol)
    return tuple(chosen)


def validate_selection(eligible, chosen, exclusions, protocol):
    # Reuse the independent frozen 3C1 validator for exact membership, split,
    # overlap, original claim and candidate metadata/content/order preservation.
    def frozen(case):
        return FrozenTrainCase(case.canonical_case_id, case.dataset, case.claim,
                               tuple(Candidate(*unit) for unit in case.candidates))
    source = {case.canonical_case_id: case for case in eligible}
    for row in chosen:
        if source.get(row.case.canonical_case_id) != row.case:
            raise CohortError("selected source metadata/content changed")
    validate_development_cohort([frozen(c) for c in eligible],
                               [DevelopmentCase(frozen(r.case), r.repair_split) for r in chosen], exclusions)


def approve(path, ledger, source_lock, excluded, report):
    if path is None:
        raise CohortError("build requires --approved-preflight-report")
    path = safe_path(path)
    if path.name != "cohort_source_preflight_report.json":
        raise CohortError("approved preflight report filename mismatch")
    for name, expected in ((path.name, report), ("cohort_source_lock.json", source_lock),
                           ("exclusion_source_lock.json", excluded)):
        artifact = ledger.add(path.parent / name, "approved preflight " + name, with_sidecar=True)
        if artifact.read_bytes() != json_bytes(expected):
            raise CohortError("approved preflight is not byte-consistent with current inputs")


def build(inputs, output_dir, approved_preflight_report):
    output = assert_new_output(output_dir)
    if approved_preflight_report is None:
        raise CohortError("build requires --approved-preflight-report")
    safe_path(approved_preflight_report)
    protocol, source, exclusions, ledger, source_lock, excluded, report = prepare(inputs)
    approve(approved_preflight_report, ledger, source_lock, excluded, report)
    # Exactly the existing adapter: no alternate normalizer or CLI fixture mode.
    original_bytecode_policy = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        adapter = Phase4ANormalizationExposureAdapter.from_project_root(inputs.project_root, inputs.phase4a_config)
        rows, eligible, counts = inventory(source, exclusions, protocol, adapter=adapter)
    finally:
        sys.dont_write_bytecode = original_bytecode_policy
    selected = select(eligible, exclusions, protocol)
    source_lock = {"implementation_revision": IMPLEMENTATION_REVISION,
                   "preregistration_sha256": schemas.PREREGISTRATION_SHA256, **ledger.payload()}
    artifacts = {
        "development_cohort_source_lock.json": json_bytes(source_lock),
        "exclusion_lock.json": json_bytes(excluded),
        "eligibility_inventory.json": json_bytes({"cases": rows, **counts}),
    }
    report = {
        "status": BLOCKED_STATUS if selected is None else PASS_STATUS,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "repair_name": protocol["repair_name"],
        "preregistration_sha256": schemas.PREREGISTRATION_SHA256,
        "authoritative_train_sha256": schemas.AUTHORITATIVE_TRAIN_SHA256,
        "exclusion_accounting": {k: v for k, v in excluded.items() if k != "manifests"},
        **counts, **boundary_flags(),
    }
    if selected is None:
        report.update(block_reason="INSUFFICIENT_ELIGIBLE_UNEXCLUDED_CASES",
                      required_dataset_counts=protocol["development_cohort"]["dataset_counts"],
                      selected_case_count=0, reviewer_A_row_count=0, reviewer_B_row_count=0,
                      cohort_feasibility_block=True, selector_scientific_result_produced=False)
    else:
        requests = [row.request() for row in selected]
        csvs, mapping = packets(selected)
        artifacts.update(csvs)
        artifacts.update({
            "repair_development_manifest.json": json_bytes({
                "status": "FROZEN", "implementation_revision": IMPLEMENTATION_REVISION,
                "selected_cases": [row.manifest() for row in selected]}),
            "repair_development_requests.jsonl": b"".join(compact(row) + b"\n" for row in requests),
            "review_mapping_private.json": json_bytes(mapping),
        })
        total = sum(len(r.case.candidates) for r in selected)
        pair_counts = Counter((u[1], u[2]) for r in selected for u in r.case.candidates)
        report.update(
            selected_case_count=len(selected),
            selected_dataset_counts=dict(Counter(r.case.dataset for r in selected)),
            split_counts=dict(Counter(r.repair_split for r in selected)),
            split_dataset_counts={split: dict(Counter(r.case.dataset for r in selected if r.repair_split == split))
                                  for split in protocol["development_cohort"]["split_counts"]},
            train_dev_case_overlap_count=0, total_frozen_candidate_unit_count=total,
            unit_type_modality_pair_counts=[{"unit_type": pair[0], "modality": pair[1], "count": count}
                                            for pair, count in sorted(pair_counts.items())],
            reviewer_A_row_count=total, reviewer_B_row_count=total,
            reviewer_C_created=False, cohort_feasibility_block=False,
            selector_scientific_result_produced=False,
        )
    report["artifact_sha256"] = {name: digest_bytes(data) for name, data in artifacts.items()}
    artifacts["development_cohort_build_report.json"] = json_bytes(report)
    freeze(output, artifacts, ledger)
    return report
