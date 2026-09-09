"""Read-only provenance resolution and the existing real Phase4A exposure."""

import ast
import copy
import json
import re
from collections import Counter
from pathlib import Path

from scripts.selector_fidelity_audit.cross_case import canonicalize_underlying_case_id
from scripts.selector_relevance_calibration.dataset_builder import (
    DatasetBuildError, ExposureResult, FrozenExposureUnavailableError,
    Phase4ANormalizationExposureAdapter, SourceCase,
    _PROVENANCE_FIELDS, assess_source_case_provenance, verify_train_lock,
)
from scripts.selector_relevance_next_repair_protocol.schemas import canonical_identity

from . import schemas
from .artifacts import read_json, safe_path, unique_object
from .schemas import CANDIDATE_FIELDS, CohortError, IDENTITY_FIELDS, Case

# Syntax-only skipping: excluded claims/candidates and neutral pseudo-labels are
# never deserialized. Hashing the locked source necessarily reads its raw bytes.
_TOKEN = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"|true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[{}\[\]:,]')


def project_line(line, fields):
    tokens = iter(_TOKEN.finditer(line))
    end = 0

    def take():
        nonlocal end
        token = next(tokens, None)
        if token is None or line[end:token.start()].strip():
            raise CohortError("malformed source JSON")
        end = token.end()
        return token

    def value(token):
        start = token.start()
        if token.group() == "{":
            token = take()
            if token.group() != "}":
                seen = set()
                while True:
                    if not token.group().startswith('"') or token.group() in seen:
                        raise CohortError("malformed nested JSON object")
                    seen.add(token.group())
                    if take().group() != ":":
                        raise CohortError("malformed nested JSON separator")
                    value(take())
                    token = take()
                    if token.group() == "}":
                        break
                    if token.group() != ",":
                        raise CohortError("malformed nested JSON object")
                    token = take()
        elif token.group() == "[":
            token = take()
            if token.group() != "]":
                while True:
                    value(token)
                    token = take()
                    if token.group() == "]":
                        break
                    if token.group() != ",":
                        raise CohortError("malformed JSON array")
                    token = take()
        elif token.group() in {"}", "]", ":", ","}:
            raise CohortError("malformed JSON value")
        return start, end

    if take().group() != "{":
        raise CohortError("source row must be an object")
    result, seen = {}, set()
    token = take()
    if token.group() != "}":
        while True:
            if not token.group().startswith('"'):
                raise CohortError("JSON object key required")
            key = json.loads(token.group())
            if key in seen:
                raise CohortError("duplicate source key")
            seen.add(key)
            if take().group() != ":":
                raise CohortError("JSON key separator required")
            start, stop = value(take())
            if key in fields:
                result[key] = json.loads(line[start:stop], object_pairs_hook=unique_object)
            token = take()
            if token.group() == "}":
                break
            if token.group() != ",":
                raise CohortError("JSON object separator required")
            token = take()
    if line[end:].strip():
        raise CohortError("trailing source JSON data")
    return result


def identity(row, require_train=False):
    dataset_values = [row[k] for k in ("dataset", "source_dataset") if k in row]
    if not dataset_values or len(set(dataset_values)) != 1:
        raise CohortError("missing or inconsistent dataset")
    dataset = dataset_values[0]
    if dataset not in schemas.EXPECTED_SOURCE_COUNTS:
        raise CohortError("unsupported dataset")
    originals = [str(row[k]) for k in ("case_id", "original_case_id", "source_case_id", "sample_id", "id")
                 if k in row]
    canonicals = [row[k] for k in ("canonical_case_id", "canonical_underlying_case_id") if k in row]
    if not originals and not canonicals:
        raise CohortError("missing canonical identity")
    for raw in originals:
        if not raw.strip() or re.search(r"(?:^|[:/])(?:test|val|validation)(?:[:/]|$)", raw, re.I):
            raise CohortError("ambiguous case identity provenance")
        canonicals.append(canonicalize_underlying_case_id(dataset, raw))
    if len(set(canonicals)) != 1:
        raise CohortError("inconsistent canonical identity")
    canonical = canonicals[0]
    canonical_identity(canonical)
    if not canonical.startswith(dataset + ":"):
        raise CohortError("dataset identity mismatch")
    if require_train and (not isinstance(row.get("split"), str)
                          or row["split"].strip().casefold() != "train"):
        raise CohortError("explicit top-level Train split required")
    return dataset, originals[0] if originals else canonical.split(":", 1)[1], canonical


def read_identity_rows(path):
    with safe_path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield index, line, project_line(line, IDENTITY_FIELDS)


def bound_input(ledger, lock, key, role, supplied=None, project_root=None):
    record = lock.get("artifacts", {}).get(key)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise CohortError(f"missing exact 3B1 source-lock entry: {key}")
    path = Path(record["path"])
    if not path.is_absolute():
        if project_root is None:
            raise CohortError("relative provenance path without project root")
        path = project_root / path
    path = safe_path(path)
    if supplied is not None and safe_path(supplied) != path:
        raise CohortError(f"supplied {role} differs from frozen 3B1 path")
    return ledger.add(path, role, expected=record["sha256"])


def closure(ledger, directory):
    documents = {}
    for name, expected in schemas.CLOSURE_HASHES.items():
        path = ledger.add(directory / name, "3B3 quarantine " + name,
                          expected=expected, with_sidecar=True)
        documents[name] = read_json(path)
    summary = documents["step3b3_scientific_summary.json"]
    manifest = documents["step3b3_closure_manifest.json"]
    if summary.get("status") != schemas.CLOSURE_STATUS:
        raise CohortError("3B3 closure status does not authorize quarantine")
    expected_flags = {
        "scientific_result_valid": True, "repair_verification_pass": False,
        "deployment_remains_blocked": True, "audit_is_now_revealed": True,
        "audit_may_be_used_for_training": False,
        "audit_may_be_reused_as_future_acceptance_gate": False,
    }
    # Closure files can store these authorization flags in nested sections.
    # Traverse only to locate named boolean flags; never extract metric values.
    def flags(document, field):
        found = []
        if isinstance(document, dict):
            for key, value in document.items():
                if key == field:
                    found.append(value)
                elif isinstance(value, (dict, list)):
                    found.extend(flags(value, field))
        elif isinstance(document, list):
            for value in document:
                found.extend(flags(value, field))
        return found
    for field, expected in expected_flags.items():
        found = flags(summary, field) + flags(manifest, field)
        if not found or any(value is not expected for value in found):
            raise CohortError(f"3B3 quarantine authorization mismatch: {field}")


def resolve_source(inputs, ledger, old_lock, protocol):
    root = safe_path(inputs.project_root)
    report_path = bound_input(ledger, old_lock, "phase3a_train_lock_report",
                              "Train-lock provenance", project_root=root)
    report = read_json(report_path)
    section = report.get("train_lock", report)
    source = section.get("source", report.get("source"))
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise CohortError("Train source provenance missing")
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    # Check path policy BEFORE the reused helper can open or hash this source.
    safe_path(source_path)
    train_lock = verify_train_lock(root, report_path,
                                   expected_sha256=schemas.AUTHORITATIVE_TRAIN_SHA256)
    bound_input(ledger, old_lock, "authoritative_g1_train", "authoritative Train",
                supplied=train_lock.source_path, project_root=root)
    config_path = bound_input(ledger, old_lock, "phase4a_configuration", "Phase4A configuration",
                             supplied=inputs.phase4a_config, project_root=root)
    config = read_json(config_path)
    # The existing Phase4A config is authoritative. Verify frozen constants if
    # present, without resolving or opening any referenced model/data paths.
    expected = {"maximum_units_per_sample": protocol["development_cohort"]["maximum_candidates"],
                "max_length": protocol["model_boundary"]["maximum_sequence_length"]}
    for key, value in expected.items():
        if key in config and (type(config[key]) is not int or config[key] != value):
            raise CohortError("Phase4A configuration violates frozen constants")
    phase3 = root / "MDU/scripts/clip12_phase3_common"
    phase4 = root / "MDU/scripts/clip12_phase4a_inference_handoff"
    for path in (phase3 / "clip12p3_common.py", phase3 / "clip12p3_model.py",
                 phase4 / "clip12p4a_common.py", phase4 / "clip12p4a_engine.py"):
        path = ledger.add(path, "authoritative Phase4A normalization source")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if path.name == "clip12p4a_engine.py":
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "normalize_request"]
            if len(functions) != 1:
                raise CohortError("actual Phase4A normalize_request is unavailable")
    return train_lock.source_path


def check_top_provenance(row):
    # Apply the same candidate provenance policy to top-level provenance fields,
    # with no inherited unit-ID exception at this level.
    from scripts.selector_relevance_calibration.dataset_builder import row_has_ambiguous_provenance
    if row_has_ambiguous_provenance((row,)):
        raise CohortError("ambiguous non-unit-id source provenance")


def exposure_request(row, row_index, protocol, train_sha):
    dataset, original, canonical = identity(row, require_train=True)
    check_top_provenance(row)
    claim = row.get("claim")
    raw = row.get("candidate_units")
    if not isinstance(claim, str) or not claim.strip():
        raise CohortError("original natural claim missing")
    if not isinstance(raw, list) or not all(isinstance(u, dict) for u in raw):
        raise CohortError("source candidate contract invalid")
    assessment = assess_source_case_provenance(SourceCase(
        dataset, original, canonical, row["split"], row_index, {}, tuple(raw)), train_sha)
    if assessment.ambiguous:
        raise CohortError("ambiguous candidate provenance")
    request = {"dataset": dataset, "case_id": original, "claim": claim,
               "candidate_units": [{k: u.get(k) for k in CANDIDATE_FIELDS} for u in raw]}
    pairs = {tuple(pair) for pair in protocol["development_cohort"]["allowed_pairs"]}
    # Unsupported source units must not be silently dropped or hidden by errors.
    for unit in request["candidate_units"]:
        if (unit["unit_type"], unit["modality"]) not in pairs:
            raise CohortError("unsupported/visual source candidate pair")
        if not all(isinstance(v, str) and v.strip() for v in unit.values()):
            raise CohortError("candidate fields must be nonblank strings")
    return dataset, original, canonical, request


def expose(row, row_index, adapter, protocol, train_sha):
    dataset, original, canonical, request = exposure_request(row, row_index, protocol, train_sha)
    saved = copy.deepcopy(request)
    raw = saved["candidate_units"]
    maximum = protocol["development_cohort"]["maximum_candidates"]
    pairs = {tuple(pair) for pair in protocol["development_cohort"]["allowed_pairs"]}
    try:
        result = adapter.normalize(request)
    except (DatasetBuildError, FrozenExposureUnavailableError, TypeError, ImportError, OSError) as exc:
        raise CohortError("Phase4A runtime/interface failure") from exc
    except ValueError:
        if request != saved:
            raise CohortError("Phase4A mutated the input before failure")
        return None
    if request != saved:
        raise CohortError("Phase4A mutated the exposure request")
    if not isinstance(result, ExposureResult):
        raise CohortError("invalid Phase4A exposure result")
    expected_units = saved["candidate_units"][:maximum]
    if (result.source_candidate_count != len(raw)
            or result.truncated_count != max(0, len(raw) - maximum)
            or result.dropped_unsupported_count != 0):
        raise CohortError("Phase4A candidate deletion/accounting drift")
    units = []
    for unit in result.candidate_units:
        if not isinstance(unit, dict) or set(unit) != set(CANDIDATE_FIELDS):
            raise CohortError("forbidden field or projected candidate schema drift")
        if (unit["unit_type"], unit["modality"]) not in pairs:
            raise CohortError("unsupported/visual exposed pair")
        units.append(unit)
    if units != expected_units:
        raise CohortError("Phase4A candidate deletion/mutation/order drift")
    if len({u["unit_id"] for u in units}) != len(units):
        raise CohortError("duplicate candidate IDs")
    return Case(dataset, canonical, original, row_index, saved["claim"],
                tuple(tuple(u[k] for k in CANDIDATE_FIELDS) for u in units))


def inventory(source, exclusions, protocol, adapter=None):
    rows, eligible, seen = [], [], set()
    counts = Counter()
    attempts = failures = below = valid = 0
    by_name = exclusions.ordered
    for index, line, metadata in read_identity_rows(source):
        dataset, original, canonical = identity(metadata, require_train=True)
        if canonical in seen:
            raise CohortError("duplicate authoritative Train identity")
        seen.add(canonical)
        counts[dataset] += 1
        reasons = [item.name for item in by_name if canonical in item.canonical_case_ids]
        item = {"dataset": dataset, "canonical_case_id": canonical,
                "original_case_id": original, "source_row_index": index,
                "excluded": bool(reasons), "exclusion_reason": reasons[0] if reasons else None,
                "exposure_status": "SKIPPED_EXCLUDED" if reasons else "NOT_RUN_PREFLIGHT",
                "exposed_candidate_count": None, "eligible": False}
        # Do not deserialize excluded content, especially the sealed six.
        if not reasons:
            row = project_line(line, IDENTITY_FIELDS | {"claim", "candidate_units"}
                               | set(_PROVENANCE_FIELDS))
            check_top_provenance(row)
            if adapter is None:
                exposure_request(row, index, protocol, schemas.AUTHORITATIVE_TRAIN_SHA256)
            if adapter is not None:
                attempts += 1
                case = expose(row, index, adapter, protocol, schemas.AUTHORITATIVE_TRAIN_SHA256)
                if case is None:
                    failures += 1
                    item["exposure_status"] = "FAILED"
                else:
                    count = len(case.candidates)
                    item.update(exposure_status="PASS", exposed_candidate_count=count)
                    if count < protocol["development_cohort"]["minimum_candidates"]:
                        below += 1
                    else:
                        valid += 1
                        eligible.append(case)
                        item["eligible"] = True
        rows.append(item)
    if dict(counts) != dict(schemas.EXPECTED_SOURCE_COUNTS):
        raise CohortError("authoritative Train source composition mismatch")
    accounting = {"source_case_count": len(seen), "source_dataset_counts": dict(counts),
                  "phase4a_exposure_attempt_count": attempts,
                  "phase4a_exposure_failure_count": failures,
                  "phase4a_exposure_skipped_excluded_count": sum(r["excluded"] for r in rows),
                  "candidate_count_below_6_count": below, "candidate_count_valid_count": valid,
                  "eligible_unexcluded_counts": {d: sum(c.dataset == d for c in eligible)
                                                  for d in counts}}
    return rows, eligible, accounting
