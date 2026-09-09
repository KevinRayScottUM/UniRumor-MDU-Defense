"""Identity-only exclusions in the exact frozen sequential accounting order."""

import ast
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from scripts.selector_relevance_next_repair_protocol.exclusion_contract import (
    ExclusionLock, ExclusionManifest, identity_sha256,
)
from scripts.selector_relevance_next_repair_protocol.schemas import canonical_identity

from . import schemas
from .artifacts import read_json, safe_path, sha_file
from .schemas import CohortError
from .source_loader import bound_input, identity, read_identity_rows


def frozen_code_value(ledger, path, name):
    """Reuse an existing literal provenance constant without importing training."""
    path = ledger.add(path, "frozen provenance constant source")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                               for t in node.targets):
            return ast.literal_eval(node.value)
    raise CohortError("required frozen provenance constant missing")


def make_manifest(name, ids, artifact_sha, protocol):
    ids = tuple(sorted(ids))
    expected = protocol["exclusions"]["required_identity_counts"].get(name)
    if expected is not None and len(ids) != expected:
        raise CohortError(f"{name} requires exactly {expected} unique identities")
    return ExclusionManifest(name, ids, identity_sha256(ids), artifact_sha)


def old_provenance(inputs, ledger):
    """This is the complete old 3B1 allowlist; no directory walking."""
    directory = safe_path(inputs.source_3b1_cohort_dir)
    report_path = ledger.add(directory / "build_report.json", "3B1 build provenance", with_sidecar=True)
    report = read_json(report_path)
    if report.get("status") != "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS":
        raise CohortError("3B1 build provenance is not PASS")
    paths = {}
    for name, field in (("cohort_source_lock.json", "cohort_source_lock_sha256"),
                        ("selected_case_manifest.json", "selected_case_manifest_sha256")):
        expected = report.get(field)
        if not isinstance(expected, str):
            raise CohortError("3B1 report hash binding missing")
        paths[name] = ledger.add(directory / name, "3B1 " + name,
                                 expected=expected, with_sidecar=True)
    lock = read_json(paths["cohort_source_lock.json"])
    if lock.get("status") != "PASS":
        raise CohortError("3B1 source lock is not PASS")
    manifest = read_json(paths["selected_case_manifest.json"])
    allowed_top = {"status", "implementation_revision", "sampling_salt", "selected_cases"}
    allowed_row = {"dataset", "canonical_case_id", "original_case_id", "sampling_hash",
                   "model_exposed_unit_count", "candidate_unit_ids_in_original_order",
                   "candidate_unit_types_in_original_order", "candidate_modalities_in_original_order"}
    if set(manifest) != allowed_top or manifest.get("status") != "FROZEN":
        raise CohortError("3B1 identity-only manifest schema drift")
    rows = manifest["selected_cases"]
    if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != allowed_row for r in rows):
        raise CohortError("3B1 identity-only row schema drift")
    ids = [identity(row)[2] for row in rows]
    if len(set(ids)) != len(ids):
        raise CohortError("duplicate revealed audit identities")
    if Counter(i.split(":", 1)[0] for i in ids) != schemas.REVEALED_AUDIT_DATASET_COUNTS:
        raise CohortError("revealed audit requires exactly 15/15 identities")
    return lock, ids, sha_file(paths["selected_case_manifest.json"])


def calibration(inputs, ledger, old_lock):
    base = Path(__file__).resolve().parents[1]
    hashes = frozen_code_value(ledger, base / "selector_relevance_training/trainer.py",
                               "AUTHORITATIVE_SOURCE_HASHES")
    expected_counts = frozen_code_value(ledger, base / "selector_relevance_independent_audit/cohort_builder.py",
                                        "EXPECTED_CALIBRATION_COUNTS")
    directory = safe_path(inputs.neutral_dir)
    # Only the two neutral source JSONLs and status/provenance report are used.
    # Do not open the neutral revision manifest (contains old/new claim text).
    # The closed 1D writer did not emit a report sidecar, and 3B1 did not
    # record this report. Bind its declared data hashes to the frozen code
    # constants, then lock the report bytes in the new preflight.
    report_path = ledger.add(directory / "neutral_build_report.json", "closed neutral report")
    report = read_json(report_path)
    if report.get("status") != "PASS" or report.get("implementation_revision") != "step2.6r-1d-v1":
        raise CohortError("neutral calibration is not the closed revision")
    for field, name in (("neutral_train_sha256", "neutral_calibration_train.jsonl"),
                        ("neutral_dev_sha256", "neutral_calibration_dev.jsonl")):
        if report.get(field) != hashes[name]:
            raise CohortError("closed neutral report/data hash mismatch")
    groups = []
    for name in ("neutral_calibration_train.jsonl", "neutral_calibration_dev.jsonl"):
        path = bound_input(ledger, old_lock, name, "prior calibration identities " + name,
                           supplied=directory / name, project_root=inputs.project_root)
        ledger.add(path, "prior calibration identities " + name,
                   expected=hashes[name], with_sidecar=True)
        ids = {identity(row)[2] for _, _, row in read_identity_rows(path)}
        groups.append(ids)
    if groups[0] & groups[1]:
        raise CohortError("closed neutral calibration Train/Dev overlap")
    ids = groups[0] | groups[1]
    if Counter(i.split(":", 1)[0] for i in ids) != expected_counts:
        raise CohortError("closed neutral calibration identity counts mismatch")
    # An actual immutable artifact binds both neutral source JSONL hashes.
    return ids, sha_file(inputs.source_3b1_cohort_dir / "cohort_source_lock.json")


def stage_a(inputs, ledger, old_lock):
    report_path = bound_input(ledger, old_lock, "stage_a_prediction_invariance_report", "Stage-A provenance report",
                              supplied=inputs.stage_a_invariance_report, project_root=inputs.project_root)
    report = read_json(report_path)
    if report.get("status") != "PREDICTION_INVARIANCE_SMOKE_PASS":
        raise CohortError("Stage-A provenance report is not PASS")
    path = bound_input(ledger, old_lock, "stage_a_normalized_replay_manifest", "Stage-A identity manifest",
                       project_root=inputs.project_root)
    actual = sha_file(path)
    if report.get("phase4a_replay_manifest_sha256") != actual:
        raise CohortError("Stage-A report/identity manifest binding mismatch")
    manifest = read_json(path)
    if (manifest.get("status") != "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS"
            or manifest.get("implementation_revision") != "step2.6r-3a0-r1-v1"):
        raise CohortError("Stage-A identity manifest status/revision mismatch")
    rows = manifest.get("retained_requests")
    if not isinstance(rows, list):
        raise CohortError("Stage-A retained identity records missing")
    ids = []
    for row in rows:
        allowed = {"historical_case_id", "source_case_id", "canonical_underlying_case_id",
                   "request_content_sha256", "row_index"}
        if not isinstance(row, dict) or set(row) != allowed:
            raise CohortError("Stage-A retained record is not identity/provenance-only")
        canonical = row["canonical_underlying_case_id"]
        canonical_identity(canonical)
        original = "GroundLie360:train:" + canonical.split(":", 1)[1]
        if (row["source_case_id"] != original or row["historical_case_id"] != "smoke::" + original):
            raise CohortError("Stage-A identity mapping inconsistent")
        ids.append(canonical)
    expected = schemas.STAGE_A_IDS
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise CohortError("exact seven frozen Stage-A retained identities required")
    count = len(expected)
    if report.get("request_count") != count or manifest.get("retained_request_count") != count:
        raise CohortError("Stage-A retained count mismatch")
    # We deliberately do not open or hash stage_a_normalized_replay content.
    return ids, actual


def load_exclusions(inputs, ledger, protocol, old_lock, audit_ids, audit_sha):
    prior_ids, prior_sha = calibration(inputs, ledger, old_lock)
    stage_ids, stage_sha = stage_a(inputs, ledger, old_lock)
    constant_path = Path(__file__).resolve().parents[1] / "selector_relevance_independent_audit/schemas.py"
    ledger.add(constant_path, "sealed-six and Stage-A identity constants")
    manifests = [
        make_manifest("prior_calibration", prior_ids, prior_sha, protocol),
        make_manifest("revealed_audit", audit_ids, audit_sha, protocol),
        make_manifest("sealed_challenge", schemas.SEALED_CHALLENGE_IDS, sha_file(constant_path), protocol),
        make_manifest("stage_a_replay", stage_ids, stage_sha, protocol),
    ]
    future_names = []
    for path in sorted(inputs.future_exclusion_manifests, key=str):
        path = ledger.add(path, "future identity-only exclusion", with_sidecar=True)
        payload = read_json(path)
        if set(payload) != set(protocol["exclusions"]["identity_manifest_contract"]):
            raise CohortError("future exclusion must use exact identity-only schema")
        if not isinstance(payload["name"], str) or not payload["name"].startswith("future:"):
            raise CohortError("future manifest name must start with future:")
        if not isinstance(payload["canonical_case_ids"], list):
            raise CohortError("future canonical identities must be an array")
        manifest = ExclusionManifest(payload["name"], tuple(payload["canonical_case_ids"]),
                                     payload["identity_sha256"], payload["source_artifact_sha256"])
        manifests.append(manifest)
        future_names.append(manifest.name)
    lock = ExclusionLock(tuple(manifests), tuple(sorted(future_names)))
    return lock


def exclusion_payload(lock, universe):
    """Augment existing Train-universe accounting with whole-set/outside counts."""
    result = dict(lock.accounting(universe))
    pool = set(universe)
    seen = set()

    def counts(ids):
        return {d: sum(i.startswith(d + ":") for i in ids) for d in schemas.EXPECTED_SOURCE_COUNTS}

    for item, group in zip(lock.ordered, result["categories"]):
        members = set(item.canonical_case_ids)
        effective = (members & pool) - seen
        group.update(membership_total_count=len(members),
                     inside_authoritative_train_count=len(members & pool),
                     outside_authoritative_train_count=len(members - pool),
                     membership_dataset_counts=counts(members),
                     inside_train_dataset_counts=counts(members & pool),
                     outside_train_dataset_counts=counts(members - pool),
                     effective_sequential_dataset_counts=counts(effective))
        seen.update(members & pool)
    by_name = {m.name: set(m.canonical_case_ids) for m in lock.ordered}
    for pair in result["pairwise_overlap_counts"]:
        pair["dataset_counts"] = counts(by_name[pair["first"]] & by_name[pair["second"]] & pool)
    result["pairwise_overlap_counts_all_identities"] = [
        {"first": a.name, "second": b.name, "overlap_count": len(set(a.canonical_case_ids) & set(b.canonical_case_ids)),
         "dataset_counts": counts(set(a.canonical_case_ids) & set(b.canonical_case_ids))}
        for a, b in combinations(lock.ordered, 2)
    ]
    result.update(unique_excluded_union_total_count=len(lock.all_identities),
                  unique_excluded_union_outside_train_count=len(lock.all_identities - pool),
                  unique_excluded_union_dataset_counts=counts(lock.all_identities),
                  manifests=[asdict(m) for m in lock.ordered])
    return result
