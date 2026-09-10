"""Synthetic-only Step 3C2-A contracts; never resolve DICC or real audit data."""

import builtins
import copy
import csv
import errno
import io
import json
import multiprocessing
import os
import stat
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.selector_relevance_calibration.dataset_builder import ExposureResult, verify_train_lock
from scripts.selector_relevance_next_repair_protocol import protocol as frozen
from scripts.selector_relevance_next_repair_protocol.exclusion_contract import (
    ExclusionLock, ExclusionManifest, identity_sha256,
)
from scripts.selector_relevance_next_repair_protocol.schemas import ProtocolError
from scripts.selector_relevance_next_repair_cohort import (
    artifacts as ar, blinding as blind, cohort_builder as cb,
    exclusion_loader as el, schemas as sc, source_loader as sl,
)
from scripts.selector_relevance_next_repair_cohort.run_cohort import build_parser, main

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = REPO / "cache" / "step2_6r_3c2a_synthetic"


def write_json(path, payload, sidecar=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ar.json_bytes(payload))
    if sidecar:
        ar.sidecar(path).write_text(ar.sha_file(path) + "\n", encoding="ascii")
    return ar.sha_file(path)


def write_lines(path, rows, sidecar=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(ar.compact(row) + b"\n" for row in rows))
    if sidecar:
        ar.sidecar(path).write_text(ar.sha_file(path) + "\n", encoding="ascii")
    return ar.sha_file(path)


def source_row(dataset="GroundLie360", ident="synthetic-0", count=6):
    return {"dataset": dataset, "case_id": ident, "split": "Train",
            "claim": '  Synthetic original claim: 雨 "quoted"\nsecond line  ',
            "label": "UNUSED_SYNTHETIC_VERACITY",
            "candidate_units": [{"unit_id": f"{ident}-u{i}", "unit_type": "evidence",
                                 "modality": "text", "text": f' Exact synthetic text {i}\nwith,comma "quote"  '}
                                for i in range(count)]}


class FixtureAdapter:
    def normalize(self, request):
        candidates = request["candidate_units"]
        return ExposureResult(tuple(copy.deepcopy(candidates[:24])), len(candidates),
                              max(0, len(candidates) - 24), 0)


def _publication_race_worker(source, destination, start, contender_failed, results):
    """Independent processes exercise the OS lock on a synthetic filesystem."""
    def unsupported(*args):
        # Hold the winner's lock until the other process has actually tried
        # and failed. This exercises contention without timing-based sleeps.
        if not contender_failed.wait(10):
            raise RuntimeError("other publisher did not fail closed on the lock")
        raise OSError(errno.EINVAL, "synthetic unsupported filesystem")
    try:
        start.wait(timeout=10)
        with patch.object(ar, "_native_rename_exclusive", side_effect=unsupported):
            ar.rename_exclusive(Path(source), Path(destination))
        results.put((Path(source).name, "PASS"))
    except sc.CohortError:
        contender_failed.set()
        results.put((Path(source).name, "REJECTED"))
    except Exception as exc:
        results.put((Path(source).name, "ERROR:" + repr(exc)))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.outputs = root / "outputs"
        self.project = root / "project"
        self.old = root / "old_identity_only"
        self.neutral = root / "neutral"
        self.stage = root / "stage"
        self.closure = root / "closure"
        self.source = self.project / "source" / "g1_train.jsonl"
        self.phase3 = root / "provenance" / "train_lock.json"
        self.config = root / "provenance" / "phase4a_config.json"
        self.preflight = self.outputs / "preflight"
        self.build_dir = self.outputs / "build"
        self.old_lock = {"status": "PASS", "artifacts": {}}
        self.old_report = {"status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS"}
        self.rows = [source_row(dataset, f"synthetic-{i:04d}") for dataset in sc.EXPECTED_SOURCE_COUNTS
                     for i in range(70)]
        self.refresh_source()
        write_json(self.config, {"maximum_units_per_sample": 24, "max_length": 256})
        self.bind("phase4a_configuration", self.config)
        self.audit_ids = [f"{dataset}:synthetic-audit-{i:02d}"
                          for dataset in sc.EXPECTED_SOURCE_COUNTS for i in range(15)]
        selected = [{"dataset": ident.split(":")[0], "canonical_case_id": ident,
                     "original_case_id": ident.split(":")[1], "sampling_hash": "a" * 64,
                     "model_exposed_unit_count": 6, "candidate_unit_ids_in_original_order": [],
                     "candidate_unit_types_in_original_order": [], "candidate_modalities_in_original_order": []}
                    for ident in self.audit_ids]
        self.audit_manifest = {"status": "FROZEN", "implementation_revision": "step2.6r-3b1-r2-v1",
                               "sampling_salt": "synthetic-fixture", "selected_cases": selected}
        self.old_report["selected_case_manifest_sha256"] = write_json(
            self.old / "selected_case_manifest.json", self.audit_manifest, True)
        self.neutral_hashes = {}
        for name, dataset, count in (("neutral_calibration_train.jsonl", "GroundLie360", 570),
                                     ("neutral_calibration_dev.jsonl", "TRUE-3MFact", 736)):
            rows = [{"source_dataset": dataset, "source_case_id": f"synthetic-cal-{i:04d}",
                     "canonical_underlying_case_id": f"{dataset}:synthetic-cal-{i:04d}",
                     "relevance_targets": ["DO_NOT_DESERIALIZE_SYNTHETIC_SENTINEL"]}
                    for i in range(count)]
            self.neutral_hashes[name] = write_lines(self.neutral / name, rows, True)
            self.bind(name, self.neutral / name)
        write_json(self.neutral / "neutral_build_report.json", {
            "status": "PASS", "implementation_revision": "step2.6r-1d-v1",
            "neutral_train_sha256": self.neutral_hashes["neutral_calibration_train.jsonl"],
            "neutral_dev_sha256": self.neutral_hashes["neutral_calibration_dev.jsonl"]})
        retained = [{"historical_case_id": "smoke::GroundLie360:train:" + ident.split(":")[1],
                     "source_case_id": "GroundLie360:train:" + ident.split(":")[1],
                     "canonical_underlying_case_id": ident, "request_content_sha256": "b" * 64,
                     "row_index": i} for i, ident in enumerate(sorted(sc.STAGE_A_IDS))]
        self.stage_manifest = self.stage / "phase4a_manifest.json"
        manifest_sha = write_json(self.stage_manifest, {
            "status": "PHASE4A_INVARIANCE_REQUEST_NORMALIZATION_PASS",
            "implementation_revision": "step2.6r-3a0-r1-v1", "retained_request_count": 7,
            "retained_requests": retained})
        self.stage_report = self.stage / "prediction_invariance_smoke_report.json"
        write_json(self.stage_report, {"status": "PREDICTION_INVARIANCE_SMOKE_PASS", "request_count": 7,
                                       "phase4a_replay_manifest_sha256": manifest_sha})
        self.bind("stage_a_normalized_replay_manifest", self.stage_manifest)
        self.bind("stage_a_prediction_invariance_report", self.stage_report)
        self.closure_summary = {"status": sc.CLOSURE_STATUS, "scientific_result_valid": True,
                                "repair_verification_pass": False, "deployment_remains_blocked": True,
                                "audit_is_now_revealed": True, "audit_may_be_used_for_training": False,
                                "audit_may_be_reused_as_future_acceptance_gate": False}
        self.closure_hashes = {
            "step3b3_scientific_summary.json": write_json(self.closure / "step3b3_scientific_summary.json",
                                                         self.closure_summary, True),
            "step3b3_closure_manifest.json": write_json(self.closure / "step3b3_closure_manifest.json",
                                                        {"synthetic_identity_provenance_only": True}, True),
        }
        self.phase4_source = self.project / "MDU/scripts/clip12_phase4a_inference_handoff"
        phase3 = self.project / "MDU/scripts/clip12_phase3_common"
        for path in (phase3 / "clip12p3_common.py", phase3 / "clip12p3_model.py",
                     self.phase4_source / "clip12p4a_common.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('def forbidden_model_load():\n    raise AssertionError("NO_MODEL")\n', encoding="utf-8")
        self.engine = self.phase4_source / "clip12p4a_engine.py"
        self.engine.write_text(
            'def normalize_request(request, config, drop_unsupported_visual=False):\n'
            '    assert drop_unsupported_visual is False\n'
            '    return {"candidate_units": [dict(u) for u in request["candidate_units"][:config["maximum_units_per_sample"]]]}\n'
            'def forward(*args):\n    raise AssertionError("NO_FORWARD")\n', encoding="utf-8")
        self.save_old()
        self.inputs = sc.Inputs(self.project, self.config, self.old, self.closure, self.neutral, self.stage_report)

    def bind(self, name, path):
        self.old_lock["artifacts"][name] = {"path": str(path), "sha256": ar.sha_file(path)}

    def refresh_source(self):
        self.train_sha = write_lines(self.source, self.rows)
        write_json(self.phase3, {"train_lock": {"status": "PASS", "source": {
            "path": str(self.source), "sha256": self.train_sha}}})
        self.bind("authoritative_g1_train", self.source)
        self.bind("phase3a_train_lock_report", self.phase3)
        self.source_counts = dict(Counter(row["dataset"] for row in self.rows))

    def save_old(self):
        self.old_report["cohort_source_lock_sha256"] = write_json(self.old / "cohort_source_lock.json",
                                                                 self.old_lock, True)
        write_json(self.old / "build_report.json", self.old_report, True)

    def future(self, name, ids):
        path = self.root / "future" / (name + ".json")
        write_json(path, {"name": "future:" + name, "canonical_case_ids": list(ids),
                          "identity_sha256": identity_sha256(ids), "source_artifact_sha256": "e" * 64}, True)
        return path

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(sc, "AUTHORITATIVE_TRAIN_SHA256", self.train_sha))
        stack.enter_context(patch.object(sc, "EXPECTED_SOURCE_COUNTS", self.source_counts))
        stack.enter_context(patch.object(sc, "CLOSURE_HASHES", self.closure_hashes))
        original = el.frozen_code_value
        stack.enter_context(patch.object(el, "frozen_code_value", side_effect=lambda ledger, path, name:
                                        self.neutral_hashes if name == "AUTHORITATIVE_SOURCE_HASHES"
                                        else original(ledger, path, name)))
        return stack


class PureContracts(unittest.TestCase):
    def setUp(self):
        self.protocol = frozen.load_preregistration()

    def test_exact_frozen_canonical_hash(self):
        self.assertEqual("81caac242f486eee630cb34c9009482065bcfab35ac920a799f097d9128bceff",
                         frozen.PREREGISTRATION_SHA256)
        frozen.validate_preregistration(self.protocol)

    def test_modified_protocol_rejected(self):
        self.protocol["development_cohort"]["case_count"] = 119
        with self.assertRaises(ProtocolError):
            frozen.validate_preregistration(self.protocol)

    def test_wrong_protocol_hash_rejected(self):
        with patch.object(frozen, "PREREGISTRATION_SHA256", "0" * 64), self.assertRaises(ProtocolError):
            frozen.load_preregistration()

    def test_pinned_authoritative_hashes(self):
        self.assertEqual("e807535556441434df0ef53a37921c0bdac5e27215ed045104ac08f38275e406",
                         sc.AUTHORITATIVE_TRAIN_SHA256)
        self.assertEqual({"GroundLie360": 1636, "TRUE-3MFact": 2242}, sc.EXPECTED_SOURCE_COUNTS)
        self.assertEqual("6ed3401614f68e58ae0efb3a1671f9f2f7b6b8b653f2100660199ee72938fdf0",
                         sc.CLOSURE_HASHES["step3b3_scientific_summary.json"])
        self.assertEqual("71732d7c2b84856ed735cc6d81e7da8af3495fdfba49d9b71cd57df7ed92f81f",
                         sc.CLOSURE_HASHES["step3b3_closure_manifest.json"])

    def test_json_projection_skips_content_deserialization(self):
        raw = json.dumps({"dataset": "GroundLie360", "claim": "SEALED_SYNTHETIC_SENTINEL",
                          "candidate_units": [{"text": "SEALED_SYNTHETIC_SENTINEL"}],
                          "label": "SEALED_SYNTHETIC_SENTINEL"})
        original = json.loads
        def guard(value, *args, **kwargs):
            self.assertNotIn("SEALED_SYNTHETIC_SENTINEL", value)
            return original(value, *args, **kwargs)
        with patch.object(sl.json, "loads", side_effect=guard):
            self.assertEqual({"dataset": "GroundLie360"}, sl.project_line(raw, sc.IDENTITY_FIELDS))

    def test_projection_escape_and_nested_content(self):
        obj = {"dataset": "GroundLie360", "candidate_units": [{"text": 'escaped\\" } ] \n'}],
               "claim": 'claim "\\中'}
        self.assertEqual(obj, sl.project_line(json.dumps(obj), set(obj)))

    def test_projection_malformed_and_duplicate_rejected(self):
        for raw in ('{"dataset":"a","dataset":"b"}', '{"a":[1,]}', '{"a":NaN}',
                    '{"a":01}', '{"a":}', '{"a":1} extra', '{"a" 1}'):
            with self.subTest(raw=raw), self.assertRaises(sc.CohortError):
                sl.project_line(raw, sc.IDENTITY_FIELDS)

    def test_formal_and_old_ranking_paths_rejected_before_open(self):
        paths = [REPO / "cache" / name for name in ("Validation/data.json", "Formal-Test/data.json",
                 "test.jsonl", "g1_validation.jsonl", *sc.FORBIDDEN_ARTIFACTS)]
        with patch.object(Path, "open", side_effect=AssertionError("must not open")):
            for path in paths:
                with self.subTest(path=path), self.assertRaises(sc.CohortError):
                    ar.safe_path(path)

    def test_non_train_and_ambiguous_identity_rejected(self):
        for split in ("Test", "Validation", None, 1, ""):
            with self.subTest(split=split), self.assertRaises(sc.CohortError):
                sl.identity({**source_row(), "split": split}, require_train=True)
        with self.assertRaises(sc.CohortError):
            sl.identity({**source_row(), "case_id": "GroundLie360:test:abc"})

    def test_inconsistent_identity_rejected(self):
        with self.assertRaises(sc.CohortError):
            sl.identity({**source_row(), "canonical_case_id": "GroundLie360:wrong"})

    def test_allowed_pairs_exact_claim_text_order(self):
        for unit_type, modality in self.protocol["development_cohort"]["allowed_pairs"]:
            row = source_row()
            for u in row["candidate_units"]:
                u.update(unit_type=unit_type, modality=modality)
            case = sl.expose(row, 5, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)
            self.assertEqual(row["claim"], case.claim)
            self.assertEqual([{**u, "original_candidate_position": i} for i, u in enumerate(row["candidate_units"])], case.units())
            self.assertEqual(5, case.source_row_index)

    def test_visual_image_and_unsupported_pairs_rejected(self):
        for pair in (("visual", "visual"), ("image", "image"), ("evidence", "audio"), ("ocr", "text")):
            row = source_row()
            row["candidate_units"][0].update(unit_type=pair[0], modality=pair[1])
            with self.subTest(pair=pair), self.assertRaises(sc.CohortError):
                sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_groundlie_inherited_exception_exact(self):
        row = source_row(ident="123456")
        row["candidate_units"][0]["unit_id"] = "GroundLie360:test:123456:evidence:0"
        self.assertIsNotNone(sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256))
        with self.assertRaises(sc.CohortError):
            sl.expose(row, 0, FixtureAdapter(), self.protocol, "0" * 64)

    def test_groundlie_wrong_case_and_other_dataset_rejected(self):
        for dataset, uid in (("GroundLie360", "GroundLie360:test:wrong:evidence:0"),
                              ("TRUE-3MFact", "TRUE-3MFact:test:123456:evidence:0")):
            row = source_row(dataset, "123456")
            row["candidate_units"][0]["unit_id"] = uid
            with self.subTest(dataset=dataset), self.assertRaises(sc.CohortError):
                sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_non_unit_id_test_provenance_rejected(self):
        for top in (False, True):
            row = source_row()
            target = row if top else row["candidate_units"][0]
            target["source_path"] = "/locked/test/content"
            with self.subTest(top=top), self.assertRaises(sc.CohortError):
                sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_candidate_deletion_mutation_reorder_and_fields_rejected(self):
        for mode in ("delete", "text", "order", "id", "score", "label", "input", "count"):
            class BadAdapter(FixtureAdapter):
                def normalize(self, request):
                    result = super().normalize(request)
                    units = list(result.candidate_units)
                    if mode == "delete":
                        units.pop()
                    elif mode == "order":
                        units.reverse()
                    elif mode == "input":
                        request["claim"] = "mutated"
                    elif mode == "count":
                        return replace(result, dropped_unsupported_count=1)
                    else:
                        field = {"text": "text", "id": "unit_id", "score": "selector_score", "label": "review_label"}[mode]
                        units[0][field] = "changed"
                    return replace(result, candidate_units=tuple(units))
            with self.subTest(mode=mode), self.assertRaises(sc.CohortError):
                sl.expose(source_row(), 0, BadAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_truncation_only_at_frozen_24(self):
        row = source_row(count=28)
        case = sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)
        self.assertEqual(24, len(case.candidates))
        self.assertEqual([u["unit_id"] for u in row["candidate_units"][:24]], [u[0] for u in case.candidates])

    def test_duplicate_candidate_rejected(self):
        row = source_row()
        row["candidate_units"][1]["unit_id"] = row["candidate_units"][0]["unit_id"]
        with self.assertRaises(sc.CohortError):
            sl.expose(row, 0, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_exposure_failure_vs_runtime_contract(self):
        for exc, expected in ((ValueError("bad request"), None), (RuntimeError("runtime"), RuntimeError),
                              (TypeError("interface"), sc.CohortError)):
            adapter = FixtureAdapter()
            with patch.object(adapter, "normalize", side_effect=exc):
                if expected is None:
                    self.assertIsNone(sl.expose(source_row(), 0, adapter, self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256))
                else:
                    with self.assertRaises(expected):
                        sl.expose(source_row(), 0, adapter, self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256)

    def test_exact_identity_hash_encoding(self):
        import hashlib
        dataset, ident = "GroundLie360", "GroundLie360:synthetic-0"
        for purpose in ("repair-development", "repair-split"):
            expected = hashlib.sha256(json.dumps(["step2.6r-3c1-natural-pairwise-v1", purpose, dataset, ident],
                                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertEqual(expected, cb.identity_hash(self.protocol, purpose, dataset, ident))

    def test_no_override_or_prohibited_cli_flags(self):
        parser = build_parser()
        flags = {flag for action in parser._actions for flag in action.option_strings}
        for name in ("validation", "test", "selector-checkpoint", "model-score-file", "ranking-scores",
                     "relevance-gold", "seed", "threshold", "resample", "overwrite", "fixture", "expected-train-sha256"):
            self.assertNotIn("--" + name, flags)
        self.assertFalse(parser.allow_abbrev)

    def test_overlap_accounting_exact_frozen_sequence(self):
        prior = [f"GroundLie360:p{i}" for i in range(1306)]
        audit = [prior[0], *[f"GroundLie360:a{i}" for i in range(29)]]
        sealed = [prior[0], audit[1], *[f"GroundLie360:s{i}" for i in range(4)]]
        stage = [prior[0], sealed[2], *[f"GroundLie360:t{i}" for i in range(5)]]
        future = [prior[0], audit[1], sealed[2], stage[2], "GroundLie360:future"]
        names = ("prior_calibration", "revealed_audit", "sealed_challenge", "stage_a_replay", "future:a")
        manifests = tuple(el.make_manifest(name, ids, "e" * 64, self.protocol)
                          for name, ids in zip(names, (prior, audit, sealed, stage, future)))
        lock = ExclusionLock(manifests)
        result = el.exclusion_payload(lock, sorted(lock.all_identities))
        self.assertEqual([1306, 29, 4, 5, 1], [g["effective_sequential_count"] for g in result["categories"]])
        self.assertEqual(1345, result["unique_excluded_union_count"])
        pair = next(p for p in result["pairwise_overlap_counts"] if p["first"] == "revealed_audit" and p["second"] == "sealed_challenge")
        self.assertEqual(2, pair["overlap_count"])


class ConstructionContracts(unittest.TestCase):
    def setUp(self):
        SYNTHETIC_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fixture-", dir=SYNTHETIC_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))
        self.protocol = frozen.load_preregistration()

    def prepared(self):
        with self.f.patches():
            return cb.prepare(self.f.inputs)

    def run_preflight(self):
        with self.f.patches():
            return cb.preflight(self.f.inputs, self.f.preflight)

    def run_build(self):
        with self.f.patches():
            return cb.build(self.f.inputs, self.f.build_dir,
                            self.f.preflight / "cohort_source_preflight_report.json")

    def inventory(self):
        with self.f.patches():
            protocol, source, exclusions, *_ = cb.prepare(self.f.inputs)
            rows, eligible, counts = sl.inventory(source, exclusions, protocol, FixtureAdapter())
        return exclusions, rows, eligible, counts

    def assert_sidecars(self, output):
        for path in output.iterdir():
            if path.suffix != ".sha256":
                self.assertEqual(ar.sha_file(path), ar.sidecar(path).read_text().strip())

    def rebind_source(self):
        self.f.refresh_source()
        self.f.save_old()

    def test_synthetic_preflight_exact_closure_hashes_accepted(self):
        report = self.run_preflight()
        self.assertEqual(cb.PREFLIGHT_STATUS, report["status"])
        self.assert_sidecars(self.f.preflight)
        self.assertEqual({"cohort_source_preflight_report.json", "cohort_source_lock.json",
                          "exclusion_source_lock.json"}, {p.name for p in self.f.preflight.iterdir() if p.suffix != ".sha256"})

    def test_wrong_summary_hash_rejected(self):
        (self.f.closure / "step3b3_scientific_summary.json").write_text("{}")
        with self.assertRaisesRegex(sc.CohortError, "SHA-256 mismatch"):
            self.run_preflight()
        self.assertFalse(self.f.preflight.exists())

    def test_wrong_closure_manifest_hash_rejected(self):
        (self.f.closure / "step3b3_closure_manifest.json").write_text("{}")
        with self.assertRaisesRegex(sc.CohortError, "SHA-256 mismatch"):
            self.run_preflight()

    def test_closure_wrong_status_and_flags_rejected(self):
        original = copy.deepcopy(self.f.closure_summary)
        for field in ("status", "scientific_result_valid", "repair_verification_pass", "deployment_remains_blocked",
                      "audit_is_now_revealed", "audit_may_be_used_for_training",
                      "audit_may_be_reused_as_future_acceptance_gate"):
            value = {**original, field: "WRONG" if field == "status" else not original[field]}
            name = "step3b3_scientific_summary.json"
            self.f.closure_hashes[name] = write_json(self.f.closure / name, value, True)
            with self.subTest(field=field), self.assertRaises(sc.CohortError):
                self.run_preflight()

    def test_closure_bool_is_not_integer(self):
        name = "step3b3_scientific_summary.json"
        self.f.closure_hashes[name] = write_json(self.f.closure / name,
                                               {**self.f.closure_summary, "scientific_result_valid": 1}, True)
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_missing_each_required_source_lock_rejected(self):
        for key in list(self.f.old_lock["artifacts"]):
            record = self.f.old_lock["artifacts"].pop(key)
            self.f.save_old()
            with self.subTest(key=key), self.assertRaises(sc.CohortError):
                self.run_preflight()
            self.f.old_lock["artifacts"][key] = record
        self.f.save_old()

    def test_all_exclusion_counts_required(self):
        for name, count in self.protocol["exclusions"]["required_identity_counts"].items():
            ids = [f"GroundLie360:synthetic-ex-{i}" for i in range(count - 1)]
            with self.subTest(name=name), self.assertRaises(ProtocolError):
                el.make_manifest(name, ids, "a" * 64, self.protocol)

    def test_missing_required_exclusion_no_override(self):
        exclusions = self.prepared()[2]
        with self.assertRaises(ProtocolError):
            ExclusionLock(exclusions.manifests[:-1])

    def test_calibration_1306_and_570_736(self):
        exclusions = self.prepared()[2]
        prior = exclusions.ordered[0]
        self.assertEqual(1306, len(prior.canonical_case_ids))
        self.assertEqual({"GroundLie360": 570, "TRUE-3MFact": 736},
                         Counter(i.split(":")[0] for i in prior.canonical_case_ids))

    def test_revealed_exact_30_and_15_15(self):
        exclusions = self.prepared()[2]
        audit = exclusions.ordered[1]
        self.assertEqual(set(self.f.audit_ids), set(audit.canonical_case_ids))
        self.assertEqual({"GroundLie360": 15, "TRUE-3MFact": 15}, Counter(i.split(":")[0] for i in audit.canonical_case_ids))

    def test_audit_manifest_with_scores_or_content_rejected(self):
        self.f.audit_manifest["selected_cases"][0]["claim"] = "disallowed"
        self.f.old_report["selected_case_manifest_sha256"] = write_json(
            self.f.old / "selected_case_manifest.json", self.f.audit_manifest, True)
        self.f.save_old()
        with self.assertRaisesRegex(sc.CohortError, "identity-only"):
            self.run_preflight()

    def test_wrong_audit_count_and_duplicates_rejected(self):
        rows = self.f.audit_manifest["selected_cases"]
        rows[-1] = copy.deepcopy(rows[0])
        self.f.old_report["selected_case_manifest_sha256"] = write_json(
            self.f.old / "selected_case_manifest.json", self.f.audit_manifest, True)
        self.f.save_old()
        with self.assertRaisesRegex(sc.CohortError, "duplicate"):
            self.run_preflight()

    def test_sealed_and_stage_constants_identity_only(self):
        exclusions = self.prepared()[2]
        self.assertEqual(sc.SEALED_CHALLENGE_IDS, frozenset(exclusions.ordered[2].canonical_case_ids))
        self.assertEqual(sc.STAGE_A_IDS, frozenset(exclusions.ordered[3].canonical_case_ids))

    def test_stage_wrong_identity_rejected(self):
        manifest = ar.read_json(self.f.stage_manifest)
        manifest["retained_requests"][0]["canonical_underlying_case_id"] = "GroundLie360:synthetic-wrong"
        sha = write_json(self.f.stage_manifest, manifest)
        report = ar.read_json(self.f.stage_report)
        report["phase4a_replay_manifest_sha256"] = sha
        write_json(self.f.stage_report, report)
        self.f.bind("stage_a_normalized_replay_manifest", self.f.stage_manifest)
        self.f.bind("stage_a_prediction_invariance_report", self.f.stage_report)
        self.f.save_old()
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_future_union_order_overlap_and_outside_accounting(self):
        ident = "GroundLie360:synthetic-0000"
        a = self.f.future("a", [ident, self.f.audit_ids[0]])
        z = self.f.future("z", [ident, "TRUE-3MFact:outside"])
        self.f.inputs = replace(self.f.inputs, future_exclusion_manifests=(z, a))
        result = self.prepared()
        exclusions, payload = result[2], result[5]
        self.assertEqual(["prior_calibration", "revealed_audit", "sealed_challenge", "stage_a_replay", "future:a", "future:z"],
                         [m.name for m in exclusions.ordered])
        self.assertEqual([0, 0, 0, 0, 1, 0], [g["effective_sequential_count"] for g in payload["categories"]])
        self.assertEqual(1, payload["unique_excluded_union_count"])
        pair = next(p for p in payload["pairwise_overlap_counts_all_identities"] if p["first"] == "revealed_audit" and p["second"] == "future:a")
        self.assertEqual(1, pair["overlap_count"])
        self.assertEqual(1, payload["categories"][-1]["outside_authoritative_train_count"])
        second = replace(self.f.inputs, future_exclusion_manifests=(a, z))
        with self.f.patches():
            self.assertEqual(payload, cb.prepare(second)[5])

    def test_future_requires_sidecar_and_identity_hash(self):
        path = self.f.future("a", ["GroundLie360:synthetic-0000"])
        self.f.inputs = replace(self.f.inputs, future_exclusion_manifests=(path,))
        ar.sidecar(path).unlink()
        with self.assertRaises(OSError):
            self.run_preflight()
        payload = ar.read_json(path)
        payload["identity_sha256"] = "0" * 64
        write_json(path, payload, True)
        with self.assertRaises(ProtocolError):
            self.run_preflight()

    def test_future_missing_duplicate_names_and_content_rejected(self):
        a = self.f.future("a", ["GroundLie360:synthetic-0000"])
        b = self.f.future("b", ["GroundLie360:synthetic-0001"])
        payload = ar.read_json(b)
        payload["name"] = "future:a"
        write_json(b, payload, True)
        self.f.inputs = replace(self.f.inputs, future_exclusion_manifests=(a, b))
        with self.assertRaises(ProtocolError):
            self.run_preflight()
        payload["candidate_text"] = "forbidden"
        write_json(b, payload, True)
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_wrong_authoritative_train_hash_rejected(self):
        with self.f.patches(), patch.object(sc, "AUTHORITATIVE_TRAIN_SHA256", "0" * 64), self.assertRaises(ValueError):
            cb.prepare(self.f.inputs)

    def test_train_source_byte_change_rejected(self):
        with self.f.source.open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            self.run_preflight()

    def test_source_counts_mismatch_rejected(self):
        with self.f.patches(), patch.object(sc, "EXPECTED_SOURCE_COUNTS", {"GroundLie360": 71, "TRUE-3MFact": 70}), self.assertRaises(sc.CohortError):
            cb.prepare(self.f.inputs)

    def test_non_train_row_rejected_preflight(self):
        self.f.rows[0]["split"] = "Validation"
        self.rebind_source()
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_duplicate_source_identity_rejected(self):
        self.f.rows[1] = copy.deepcopy(self.f.rows[0])
        self.rebind_source()
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_config_drift_rejected(self):
        write_json(self.f.config, {"maximum_units_per_sample": 23})
        self.f.bind("phase4a_configuration", self.f.config)
        self.f.save_old()
        with self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_source_path_validation_before_reused_train_loader(self):
        self.f.old_lock["artifacts"]["phase3a_train_lock_report"]["path"] = str(self.f.root / "FormalTest" / "lock.json")
        self.f.save_old()
        with patch.object(sl, "verify_train_lock", side_effect=AssertionError("must not resolve")), self.assertRaises(sc.CohortError):
            self.run_preflight()

    def test_phase4a_source_missing_rejected_preflight(self):
        self.f.engine.unlink()
        with self.assertRaises(OSError):
            self.run_preflight()

    def test_preflight_does_not_import_phase4a_or_models(self):
        self.f.engine.write_text('raise AssertionError("PREVENT_IMPORT")\ndef normalize_request(request):\n    pass\n')
        with patch.object(sl.Phase4ANormalizationExposureAdapter, "from_project_root", side_effect=AssertionError("no import")):
            self.assertEqual(cb.PREFLIGHT_STATUS, self.run_preflight()["status"])

    def test_first60_and_independent_first48_split(self):
        exclusions, _, eligible, _ = self.inventory()
        selected = cb.select(eligible, exclusions, self.protocol)
        self.assertEqual(120, len(selected))
        self.assertEqual({"repair_train": 96, "repair_dev": 24}, Counter(r.repair_split for r in selected))
        for dataset in ("GroundLie360", "TRUE-3MFact"):
            pool = [c for c in eligible if c.dataset == dataset]
            expected = sorted(pool, key=lambda c: (cb.identity_hash(self.protocol, "repair-development", dataset, c.canonical_case_id), c.canonical_case_id))[:60]
            split = sorted(expected, key=lambda c: (cb.identity_hash(self.protocol, "repair-split", dataset, c.canonical_case_id), c.canonical_case_id))
            actual = {r.case.canonical_case_id: r.repair_split for r in selected if r.case.dataset == dataset}
            self.assertEqual({c.canonical_case_id: "repair_train" if i < 48 else "repair_dev" for i, c in enumerate(split)}, actual)
            self.assertEqual(60, len(actual))

    def test_sampling_independent_of_every_content_and_label_field(self):
        exclusions, _, eligible, _ = self.inventory()
        def assignment(cases):
            return {r.case.canonical_case_id: r.repair_split for r in cb.select(cases, exclusions, self.protocol)}
        original = assignment(eligible)
        for change in ("claim", "text", "unit_type", "modality", "order", "veracity_label", "selector_score", "review_label"):
            cases = []
            for index, row in enumerate(copy.deepcopy(self.f.rows)):
                if change == "claim":
                    row["claim"] = "Changed original synthetic claim"
                elif change == "text":
                    for u in row["candidate_units"]:
                        u["text"] = "Changed exact text"
                elif change in {"unit_type", "modality"}:
                    for u in row["candidate_units"]:
                        u.update(unit_type="ocr" if change == "modality" else "transcript",
                                 modality="ocr" if change == "modality" else "text")
                elif change == "order":
                    row["candidate_units"].reverse()
                elif change == "veracity_label":
                    row["label"] = "CHANGED_UNUSED_LABEL"
                else:
                    row[change] = "UNUSED"
                    for u in row["candidate_units"]:
                        u[change] = "UNUSED"
                cases.append(sl.expose(row, index, FixtureAdapter(), self.protocol, sc.AUTHORITATIVE_TRAIN_SHA256))
            with self.subTest(change=change):
                self.assertEqual(original, assignment(cases))

    def test_selection_validation_rejects_content_and_membership_mutation(self):
        exclusions, _, eligible, _ = self.inventory()
        selected = list(cb.select(eligible, exclusions, self.protocol))
        selected[0] = replace(selected[0], case=replace(selected[0].case, claim="changed"))
        with self.assertRaises(sc.CohortError):
            cb.validate_selection(eligible, selected, exclusions, self.protocol)

    def test_any_exclusion_overlap_rejected(self):
        exclusions, _, eligible, _ = self.inventory()
        for manifest in exclusions.ordered:
            ident = manifest.canonical_case_ids[0]
            bad = replace(eligible[0], canonical_case_id=ident, dataset=ident.split(":")[0])
            with self.subTest(name=manifest.name), self.assertRaises(ProtocolError):
                cb.select([bad, *eligible[1:]], exclusions, self.protocol)

    def test_59_groundlie_freezes_block_without_packets(self):
        self.assert_blocked("GroundLie360")

    def test_59_true3m_freezes_block_without_packets(self):
        self.assert_blocked("TRUE-3MFact")

    def assert_blocked(self, dataset):
        changed = 0
        for row in self.f.rows:
            if row["dataset"] == dataset and changed < 11:
                row["candidate_units"] = row["candidate_units"][:5]
                changed += 1
        self.rebind_source()
        self.run_preflight()
        report = self.run_build()
        self.assertEqual(cb.BLOCKED_STATUS, report["status"])
        self.assertEqual(59, report["eligible_unexcluded_counts"][dataset])
        self.assertEqual(0, report["selected_case_count"])
        self.assertFalse((self.f.build_dir / "repair_development_manifest.json").exists())
        self.assertFalse(any(self.f.build_dir.glob("reviewer*")))
        self.assert_sidecars(self.f.build_dir)

    def test_candidate_count_and_exposure_accounting(self):
        self.f.rows[0]["candidate_units"] = self.f.rows[0]["candidate_units"][:5]
        self.rebind_source()
        exclusions, rows, eligible, counts = self.inventory()
        self.assertEqual(140, counts["phase4a_exposure_attempt_count"])
        self.assertEqual(1, counts["candidate_count_below_6_count"])
        self.assertEqual(139, counts["candidate_count_valid_count"])
        self.assertEqual(139, len(eligible))
        self.assertFalse(rows[0]["eligible"])
        self.assertNotIn("claim", rows[0])

    def test_build_requires_approved_preflight(self):
        with self.assertRaises(sc.CohortError):
            cb.build(self.f.inputs, self.f.build_dir, None)
        self.assertFalse(self.f.build_dir.exists())

    def test_changed_source_after_preflight_rejected(self):
        self.run_preflight()
        self.f.rows[0]["claim"] += "changed"
        self.rebind_source()
        with self.assertRaisesRegex(sc.CohortError, "byte-consistent"):
            self.run_build()
        self.assertFalse(self.f.build_dir.exists())

    def test_changed_normalizer_after_preflight_rejected(self):
        self.run_preflight()
        self.f.engine.write_text(self.f.engine.read_text() + "\n# changed source\n")
        with self.assertRaisesRegex(sc.CohortError, "byte-consistent"):
            self.run_build()

    def test_approved_report_sidecar_and_bytes_rejected(self):
        self.run_preflight()
        path = self.f.preflight / "cohort_source_preflight_report.json"
        payload = ar.read_json(path)
        payload["source_case_count"] += 1
        write_json(path, payload, True)
        with self.assertRaisesRegex(sc.CohortError, "byte-consistent"):
            self.run_build()

    def test_pass_complete_artifacts_and_flags(self):
        self.run_preflight()
        report = self.run_build()
        self.assertEqual(cb.PASS_STATUS, report["status"])
        self.assertEqual(120, report["selected_case_count"])
        self.assertEqual({"GroundLie360": 60, "TRUE-3MFact": 60}, report["selected_dataset_counts"])
        self.assertEqual({"repair_train": {"GroundLie360": 48, "TRUE-3MFact": 48},
                          "repair_dev": {"GroundLie360": 12, "TRUE-3MFact": 12}}, report["split_dataset_counts"])
        self.assertEqual(720, report["total_frozen_candidate_unit_count"])
        for key, expected in cb.boundary_flags().items():
            self.assertEqual(expected, report[key])
        names = {p.name for p in self.f.build_dir.iterdir() if p.suffix != ".sha256"}
        self.assertEqual({"development_cohort_build_report.json", "development_cohort_source_lock.json",
                          "exclusion_lock.json", "eligibility_inventory.json", "repair_development_manifest.json",
                          "repair_development_requests.jsonl", "reviewer_A_template.csv", "reviewer_B_template.csv",
                          "review_mapping_private.json"}, names)
        self.assert_sidecars(self.f.build_dir)

    def test_review_packets_blinding_and_mapping_reconstruct_exactly(self):
        self.run_preflight()
        self.run_build()
        requests = [json.loads(line) for line in (self.f.build_dir / "repair_development_requests.jsonl").read_text().splitlines()]
        underlying = {(r["canonical_case_id"], u["unit_id"]): (r, u) for r in requests for u in r["candidate_units"]}
        mapping = ar.read_json(self.f.build_dir / "review_mapping_private.json")
        orders, public_ids = {}, {}
        for reviewer in ("A", "B"):
            with (self.f.build_dir / f"reviewer_{reviewer}_template.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(list(sc.PUBLIC_COLUMNS), reader.fieldnames)
                public = list(reader)
            private = mapping[f"reviewer_{reviewer}"]
            self.assertEqual(len(underlying), len(public))
            orders[reviewer], public_ids[reviewer] = [], set()
            for exposed, hidden in zip(public, private):
                self.assertEqual({"reviewer", "review_case_id", "review_unit_id", "dataset", "canonical_case_id",
                                  "original_case_id", "repair_split", "unit_id", "original_candidate_position"}, set(hidden))
                ident = (hidden["canonical_case_id"], hidden["unit_id"])
                original, unit = underlying[ident]
                self.assertEqual(original["claim"], exposed["claim"])
                self.assertEqual(unit["text"], exposed["candidate_text"])
                self.assertEqual(unit["original_candidate_position"], hidden["original_candidate_position"])
                self.assertEqual(hidden["review_case_id"], exposed["review_case_id"])
                self.assertEqual(hidden["review_unit_id"], exposed["review_unit_id"])
                self.assertEqual(("", "", ""), tuple(exposed[k] for k in sc.PUBLIC_COLUMNS[-3:]))
                orders[reviewer].append(ident)
                public_ids[reviewer].add(exposed["review_unit_id"])
            self.assertEqual(set(underlying), set(orders[reviewer]))
        self.assertNotEqual(orders["A"], orders["B"])
        self.assertFalse(public_ids["A"] & public_ids["B"])
        self.assertFalse(any(self.f.build_dir.glob("*reviewer_C*")))

    def test_deterministic_packets_and_manifest_bytes(self):
        exclusions, _, eligible, _ = self.inventory()
        a = cb.select(eligible, exclusions, self.protocol)
        b = cb.select(list(reversed(eligible)), exclusions, self.protocol)
        self.assertEqual(a, b)
        self.assertEqual(blind.packets(a), blind.packets(b))

    def test_source_hashes_unchanged_after_build(self):
        self.run_preflight()
        ledger = ar.read_json(self.f.preflight / "cohort_source_lock.json")["artifacts"]
        before = {r["path"]: ar.sha_file(r["path"]) for r in ledger}
        source_files = sorted(str(p) for p in self.f.project.rglob("*") if p.is_file())
        self.run_build()
        self.assertEqual(before, {p: ar.sha_file(p) for p in before})
        self.assertEqual(source_files, sorted(str(p) for p in self.f.project.rglob("*") if p.is_file()))
        lock = ar.read_json(self.f.build_dir / "development_cohort_source_lock.json")
        self.assertTrue(all(set(row) == {"path", "sha256", "role"} for row in lock["artifacts"]))
        self.assertTrue(any("approved preflight" in row["role"] for row in lock["artifacts"]))

    def test_no_forbidden_files_or_model_imports(self):
        original_open, original_import = Path.open, builtins.__import__
        accessed = []
        def guarded_open(path, *args, **kwargs):
            self.assertNotIn(path.name, sc.FORBIDDEN_ARTIFACTS)
            self.assertNotIn(path.name, {"final_relevance_gold.jsonl", "relevance_gold.jsonl",
                                         "replay_requests.jsonl", "neutral_revision_manifest.json"})
            accessed.append(path)
            return original_open(path, *args, **kwargs)
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "paddle"}:
                raise AssertionError("NO_MODEL_IMPORT")
            return original_import(name, *args, **kwargs)
        with patch.object(Path, "open", guarded_open), patch.object(builtins, "__import__", guarded_import):
            self.run_preflight()
            self.run_build()
        allowed_old = {"build_report.json", "build_report.sha256", "cohort_source_lock.json",
                       "cohort_source_lock.sha256", "selected_case_manifest.json", "selected_case_manifest.sha256"}
        self.assertTrue(all(p.name in allowed_old for p in accessed if p.parent == self.f.old))

    def test_sealed_source_content_never_deserialized_or_exposed(self):
        dataset, ident = sorted(sc.SEALED_CHALLENGE_IDS)[0].split(":")
        self.f.rows.append({"dataset": dataset, "case_id": ident, "split": "Train",
                            "claim": "SEALED_SYNTHETIC_SENTINEL", "candidate_units": "SEALED_SYNTHETIC_SENTINEL"})
        self.rebind_source()
        original = json.loads
        def guard(value, *args, **kwargs):
            self.assertNotIn("SEALED_SYNTHETIC_SENTINEL", value)
            return original(value, *args, **kwargs)
        with patch.object(sl.json, "loads", side_effect=guard):
            self.run_preflight()
            report = self.run_build()
        self.assertEqual(1, report["phase4a_exposure_skipped_excluded_count"])

    def test_preflight_existing_or_partial_outputs_rejected(self):
        self.f.preflight.mkdir(parents=True)
        for partial in (False, True):
            if partial:
                (self.f.preflight / "partial").write_text("do not delete")
            with self.subTest(partial=partial), self.assertRaises(sc.CohortError):
                self.run_preflight()
        self.assertEqual("do not delete", (self.f.preflight / "partial").read_text())

    def test_build_existing_output_rejected(self):
        self.run_preflight()
        self.f.build_dir.mkdir()
        with self.assertRaises(sc.CohortError):
            self.run_build()
        self.assertTrue(self.f.build_dir.exists())

    def test_atomic_failure_leaves_no_partial_preflight(self):
        with patch.object(ar, "rename_exclusive", side_effect=OSError("synthetic publication error")), self.assertRaises(OSError):
            self.run_preflight()
        self.assertFalse(self.f.preflight.exists())
        self.assertFalse(list(self.f.outputs.glob(".cohort-freeze-*")))

    def test_atomic_build_revalidates_before_publication(self):
        self.run_preflight()
        original = blind.packets
        def mutate(selected):
            result = original(selected)
            with self.f.source.open("a") as stream:
                stream.write("\n")
            return result
        with patch.object(cb, "packets", side_effect=mutate), self.assertRaisesRegex(sc.CohortError, "immutable input changed"):
            self.run_build()
        self.assertFalse(self.f.build_dir.exists())
        self.assertFalse(list(self.f.outputs.glob(".cohort-freeze-*")))

    def test_exclusive_rename_preserves_racing_empty_destination(self):
        self.f.outputs.mkdir(parents=True)
        source = self.f.outputs / "stage"
        destination = self.f.outputs / "destination"
        source.mkdir()
        (source / "content").write_text("staged")
        destination.mkdir()
        with self.assertRaises(sc.CohortError):
            ar.rename_exclusive(source, destination)
        self.assertEqual([], list(destination.iterdir()))
        self.assertTrue((source / "content").exists())

    def test_symlink_and_source_directory_output_rejected(self):
        link = self.f.root / "source_alias"
        link.symlink_to(self.f.source)
        with self.assertRaises(sc.CohortError):
            ar.safe_path(link)
        ledger = ar.Ledger()
        ledger.add(self.f.source, "immutable Train source")
        with self.assertRaises(sc.CohortError):
            ar.freeze(self.f.source.parent / "new", {"report.json": b"{}"}, ledger)

    def test_cli_pass_block_and_invalid_exit_codes(self):
        f = self.f
        args = ["--project-root", str(f.project), "--phase4a-config", str(f.config),
                "--source-3b1-cohort-dir", str(f.old), "--step3b3-closure-dir", str(f.closure),
                "--neutral-dir", str(f.neutral), "--stage-a-invariance-report", str(f.stage_report)]
        with f.patches(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(0, main(["--preflight", *args, "--output-dir", str(f.preflight)]))
            self.assertEqual(2, main(["--preflight", *args, "--output-dir", str(f.preflight)]))
            with patch.object(cb, "select", return_value=None):
                self.assertEqual(1, main(["--build-cohort", *args, "--output-dir", str(f.build_dir),
                                          "--approved-preflight-report", str(f.preflight / "cohort_source_preflight_report.json")]))
        self.assertEqual(cb.BLOCKED_STATUS, ar.read_json(f.build_dir / "development_cohort_build_report.json")["status"])


class PublicationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        SYNTHETIC_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="publication-", dir=SYNTHETIC_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "staging"
        self.source.mkdir()
        self.payload = {"report.json": b'{"synthetic":true}\n', "units.jsonl": b'{"unit":"synthetic"}\n'}
        for name, data in self.payload.items():
            (self.source / name).write_bytes(data)
        self.destination = self.root / "published"
        self.lock = self.root / ".published.publish.lock"

    def fallback(self):
        return patch.object(ar, "_native_rename_exclusive",
                            side_effect=OSError(errno.EINVAL, "synthetic unsupported filesystem"))

    def assert_source_preserved(self):
        self.assertEqual(self.payload, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def assert_complete(self):
        self.assertEqual(self.payload, {p.name: p.read_bytes() for p in self.destination.iterdir()})
        self.assertFalse(self.source.exists())
        self.assertFalse(self.lock.exists())

    def test_native_linux_success_never_falls_back(self):
        def native(*args):
            self.assertTrue(self.lock.is_file())
            self.assertEqual((-100, os.fsencode(self.source), -100, os.fsencode(self.destination), 1), args)
            os.rename(self.source, self.destination)
            return 0
        function = Mock(side_effect=native)
        with patch.object(ar.sys, "platform", "linux"), patch.object(ar.ctypes, "CDLL", return_value=SimpleNamespace(renameat2=function)), patch.object(ar, "_portable_rename_exclusive", side_effect=AssertionError("native success must not fallback")):
            ar.rename_exclusive(self.source, self.destination)
        function.assert_called_once()
        self.assert_complete()

    def test_native_linux_einval_return_reaches_portable_fallback(self):
        function = Mock(return_value=-1)
        with patch.object(ar.sys, "platform", "linux"), patch.object(ar.ctypes, "CDLL", return_value=SimpleNamespace(renameat2=function)), patch.object(ar.ctypes, "get_errno", return_value=errno.EINVAL):
            ar.rename_exclusive(self.source, self.destination)
        function.assert_called_once()
        self.assert_complete()

    def test_enosys_falls_back(self):
        with patch.object(ar, "_native_rename_exclusive", side_effect=OSError(errno.ENOSYS, "unsupported")):
            ar.rename_exclusive(self.source, self.destination)
        self.assert_complete()

    def test_eopnotsupp_and_enotsup_each_distinct_value_fall_back(self):
        for error in sorted({errno.EOPNOTSUPP, errno.ENOTSUP}):
            source = self.root / f"source-{error}"
            source.mkdir()
            (source / "complete").write_text(str(error))
            destination = self.root / f"destination-{error}"
            with self.subTest(errno=error), patch.object(ar, "_native_rename_exclusive", side_effect=OSError(error, "unsupported")):
                ar.rename_exclusive(source, destination)
            self.assertEqual(str(error), (destination / "complete").read_text())
            self.assertFalse(source.exists())
            self.assertFalse(destination.with_name("." + destination.name + ".publish.lock").exists())

    def test_missing_native_symbol_falls_back(self):
        with patch.object(ar.sys, "platform", "linux"), patch.object(ar.ctypes, "CDLL", return_value=SimpleNamespace()):
            ar.rename_exclusive(self.source, self.destination)
        self.assert_complete()

    def test_exact_capability_errno_set(self):
        self.assertEqual({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}, ar.UNSUPPORTED_NATIVE_ERRNOS)

    def assert_no_fallback(self, error):
        with patch.object(ar, "_native_rename_exclusive", side_effect=OSError(error, "real failure")), patch.object(ar, "_portable_rename_exclusive", side_effect=AssertionError("unrelated error must not fallback")), self.assertRaises(OSError) as caught:
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(error, caught.exception.errno)
        self.assert_source_preserved()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.lock.exists())

    def test_eacces_does_not_fallback(self):
        self.assert_no_fallback(errno.EACCES)

    def test_eperm_does_not_fallback(self):
        self.assert_no_fallback(errno.EPERM)

    def test_eio_does_not_fallback(self):
        self.assert_no_fallback(errno.EIO)

    def test_erofs_enospc_exdev_do_not_fallback(self):
        for error in (errno.EROFS, errno.ENOSPC, errno.EXDEV):
            with self.subTest(errno=error):
                self.assert_no_fallback(error)

    def test_native_existing_output_errors_do_not_fallback(self):
        for error in (errno.EEXIST, errno.ENOTEMPTY):
            with self.subTest(errno=error), patch.object(ar.sys, "platform", "linux"), patch.object(ar.ctypes, "CDLL", return_value=SimpleNamespace(renameat2=Mock(return_value=-1))), patch.object(ar.ctypes, "get_errno", return_value=error), patch.object(ar, "_portable_rename_exclusive", side_effect=AssertionError("must not fallback")), self.assertRaises(sc.CohortError):
                ar.rename_exclusive(self.source, self.destination)
            self.assertFalse(self.lock.exists())
            self.assert_source_preserved()

    def test_fallback_complete_directory_and_no_staging_left(self):
        with self.fallback():
            ar.rename_exclusive(self.source, self.destination)
        self.assert_complete()

    def test_fallback_preserves_existing_empty_destination(self):
        self.destination.mkdir()
        original_inode = self.destination.stat().st_ino
        with self.fallback(), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(original_inode, self.destination.stat().st_ino)
        self.assertEqual([], list(self.destination.iterdir()))
        self.assert_source_preserved()
        self.assertFalse(self.lock.exists())

    def test_fallback_preserves_existing_nonempty_destination(self):
        self.destination.mkdir()
        (self.destination / "foreign").write_bytes(b"do not modify")
        with self.fallback(), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual({"foreign": b"do not modify"}, {p.name: p.read_bytes() for p in self.destination.iterdir()})
        self.assert_source_preserved()
        self.assertFalse(self.lock.exists())

    def test_existing_destination_symlink_including_dangling_rejected(self):
        for target in (self.source, self.root / "missing"):
            self.destination.symlink_to(target, target_is_directory=True)
            with self.subTest(target=target), self.fallback(), self.assertRaises(sc.CohortError):
                ar.rename_exclusive(self.source, self.destination)
            self.assertEqual(target, self.destination.readlink())
            self.assert_source_preserved()
            self.assertFalse(self.lock.exists())
            self.destination.unlink()

    def test_existing_or_stale_lock_blocks_even_native_attempt(self):
        self.lock.write_bytes(b"foreign or stale lock: operator inspection required")
        original_inode = self.lock.stat().st_ino
        with patch.object(ar, "_native_rename_exclusive", side_effect=AssertionError("must not bypass lock")), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(original_inode, self.lock.stat().st_ino)
        self.assertEqual(b"foreign or stale lock: operator inspection required", self.lock.read_bytes())
        self.assert_source_preserved()

    def test_lock_symlink_never_followed_or_removed(self):
        foreign = self.root / "foreign-lock"
        foreign.write_bytes(b"foreign")
        self.lock.symlink_to(foreign)
        with self.fallback(), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(foreign, self.lock.readlink())
        self.assertEqual(b"foreign", foreign.read_bytes())
        self.assert_source_preserved()

    def test_own_lock_released_after_handled_rename_failure(self):
        def reject(*args):
            self.assertTrue(self.lock.exists())
            self.assertEqual(0o600, stat.S_IMODE(self.lock.stat().st_mode))
            raise OSError(errno.EIO, "synthetic rename failure")
        with self.fallback(), patch.object(ar.os, "rename", side_effect=reject), self.assertRaises(OSError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.lock.exists())
        self.assert_source_preserved()

    def test_replaced_foreign_lock_never_cleaned(self):
        def replace_lock(*args):
            self.lock.unlink()
            self.lock.write_bytes(b"replacement foreign lock")
            raise OSError(errno.EIO, "synthetic external interference")
        with self.fallback(), patch.object(ar.os, "rename", side_effect=replace_lock), self.assertRaises(OSError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(b"replacement foreign lock", self.lock.read_bytes())
        self.assert_source_preserved()

    def test_cross_device_staging_rejected_without_rename(self):
        original = Path.lstat
        def different_device(path):
            result = original(path)
            return SimpleNamespace(st_mode=result.st_mode, st_dev=result.st_dev + 1) if path == self.source else result
        with self.fallback(), patch.object(Path, "lstat", different_device), patch.object(ar.os, "rename", side_effect=AssertionError("EXDEV must not rename")), self.assertRaises(OSError) as caught:
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(errno.EXDEV, caught.exception.errno)
        self.assert_source_preserved()
        self.assertFalse(self.lock.exists())

    def test_staging_symlink_rejected_in_fallback(self):
        alias = self.root / "staging-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.fallback(), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(alias, self.destination)
        self.assertTrue(alias.is_symlink())
        self.assert_source_preserved()
        self.assertFalse(self.lock.exists())

    def test_empty_destination_created_during_native_attempt_is_preserved(self):
        def unsupported(*args):
            self.destination.mkdir()
            raise OSError(errno.EINVAL, "synthetic unsupported filesystem")
        with patch.object(ar, "_native_rename_exclusive", side_effect=unsupported), self.assertRaises(sc.CohortError):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual([], list(self.destination.iterdir()))
        self.assert_source_preserved()
        self.assertFalse(self.lock.exists())

    def test_parent_fsync_occurs_before_lock_release(self):
        seen = []
        def sync(descriptor):
            self.assertTrue(self.lock.is_file())
            self.assertEqual(self.root.stat().st_ino, os.fstat(descriptor).st_ino)
            self.assertFalse(self.source.exists())
            seen.append(descriptor)
        with self.fallback(), patch.object(ar.os, "fsync", side_effect=sync):
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(1, len(seen))
        self.assert_complete()

    def test_directory_fsync_unsupported_tolerated(self):
        with self.fallback(), patch.object(ar.os, "fsync", side_effect=OSError(errno.EINVAL, "directory fsync unsupported")):
            ar.rename_exclusive(self.source, self.destination)
        self.assert_complete()

    def test_directory_fsync_real_error_preserves_complete_output(self):
        with self.fallback(), patch.object(ar.os, "fsync", side_effect=OSError(errno.EIO, "sync failure")), self.assertRaises(OSError) as caught:
            ar.rename_exclusive(self.source, self.destination)
        self.assertEqual(errno.EIO, caught.exception.errno)
        self.assert_complete()

    def test_freeze_revalidates_inputs_before_fallback_publication(self):
        ledger = ar.Ledger()
        before = self.root / "inputs" / "source.json"
        before.parent.mkdir()
        before.write_bytes(b"original")
        ledger.add(before, "synthetic immutable source")
        before.write_bytes(b"modified")
        with self.fallback(), patch.object(ar, "rename_exclusive", side_effect=AssertionError("must revalidate first")), self.assertRaises(sc.CohortError):
            ar.freeze(self.destination, self.payload, ledger)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob(".cohort-freeze-*")))

    def test_freeze_rename_failure_cleans_only_own_staging(self):
        foreign = self.root / ".cohort-freeze-foreign"
        foreign.mkdir()
        (foreign / "foreign").write_bytes(b"keep")
        with self.fallback(), patch.object(ar.os, "rename", side_effect=OSError(errno.EIO, "rename failed")), self.assertRaises(OSError):
            ar.freeze(self.destination, self.payload, ar.Ledger())
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.lock.exists())
        self.assertEqual([foreign], list(self.root.glob(".cohort-freeze-*")))
        self.assertEqual(b"keep", (foreign / "foreign").read_bytes())
        self.assert_source_preserved()

    def test_two_process_publishers_exactly_one_complete_winner(self):
        other = self.root / "other-staging"
        other.mkdir()
        for name in self.payload:
            (other / name).write_bytes(b"other-complete-artifact-set")
        context = multiprocessing.get_context("spawn")
        start = context.Barrier(2)
        failed = context.Event()
        results = context.Queue()
        workers = [context.Process(target=_publication_race_worker,
                                   args=(str(source), str(self.destination), start, failed, results))
                   for source in (self.source, other)]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=15)
                self.assertEqual(0, worker.exitcode)
            outcomes = dict(results.get(timeout=5) for _ in workers)
            self.assertEqual({"PASS": 1, "REJECTED": 1}, Counter(outcomes.values()))
            winner = next(name for name, result in outcomes.items() if result == "PASS")
            expected = self.payload if winner == self.source.name else {name: b"other-complete-artifact-set" for name in self.payload}
            self.assertEqual(expected, {p.name: p.read_bytes() for p in self.destination.iterdir()})
            self.assertFalse((self.root / winner).exists())
            loser = self.root / next(name for name, result in outcomes.items() if result == "REJECTED")
            self.assertEqual(set(self.payload), {p.name for p in loser.iterdir()})
            loser_expected = self.payload if loser == self.source else {name: b"other-complete-artifact-set" for name in self.payload}
            self.assertEqual(loser_expected, {p.name: p.read_bytes() for p in loser.iterdir()})
            self.assertFalse(self.lock.exists())
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2)
            results.close()
            results.join_thread()

    def test_revision_only_scientific_constants_and_salts_unchanged(self):
        self.assertEqual("step2.6r-3c2a-r2-v1", sc.IMPLEMENTATION_REVISION)
        self.assertEqual({"A": "step2.6r-3c2a-reviewer-a-v1", "B": "step2.6r-3c2a-reviewer-b-v1"}, sc.REVIEWER_SALTS)
        protocol = frozen.load_preregistration()
        self.assertEqual("step2.6r-3c1-v1", protocol["implementation_revision"])
        self.assertEqual("81caac242f486eee630cb34c9009482065bcfab35ac920a799f097d9128bceff", sc.PREREGISTRATION_SHA256)
        self.assertEqual("e807535556441434df0ef53a37921c0bdac5e27215ed045104ac08f38275e406", sc.AUTHORITATIVE_TRAIN_SHA256)
        cohort = protocol["development_cohort"]
        self.assertEqual({"GroundLie360": 60, "TRUE-3MFact": 60}, cohort["dataset_counts"])
        self.assertEqual({"repair_train": 96, "repair_dev": 24}, cohort["split_counts"])
        self.assertEqual("step2.6r-3c1-natural-pairwise-v1", cohort["sampling"]["salt"])
        self.assertEqual("SHA-256", cohort["sampling"]["algorithm"])
        self.assertEqual(("repair-development", "repair-split"),
                         (cohort["sampling"]["selection_purpose"], cohort["sampling"]["split_purpose"]))
        self.assertEqual({"prior_calibration": 1306, "revealed_audit": 30, "sealed_challenge": 6, "stage_a_replay": 7},
                         protocol["exclusions"]["required_identity_counts"])


class HistoricalTrainPathTests(unittest.TestCase):
    """Synthetic DICC provenance only; never inspect a real Train artifact."""

    def setUp(self):
        SYNTHETIC_ROOT.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="historical-", dir=SYNTHETIC_ROOT)
        self.addCleanup(temp.cleanup)
        self.f = Fixture(Path(temp.name))
        self.tail = Path(
            "clip12_phase1_full_train_multidataset_visual_candidate_expansion_mdu_integration_and_controlled_ablation"
            "/05_mdu_variants/G1_text_ocr.jsonl")
        self.canonical_dir = self.f.project / "MDU/Academic_Research/outputs"
        canonical = self.canonical_dir / self.tail
        canonical.parent.mkdir(parents=True)
        self.f.source.rename(canonical)
        self.f.source = canonical
        self.f.refresh_source()
        self.alias = self.f.project / "MDU/outputs"
        self.alias.symlink_to("Academic_Research/outputs", target_is_directory=True)
        self.historical = self.alias / self.tail
        self.set_report_path(self.historical)

    def set_report_path(self, path, **section_changes):
        report = ar.read_json(self.f.phase3)
        report["train_lock"]["source"]["path"] = str(path)
        report["train_lock"].update(section_changes)
        write_json(self.f.phase3, report)
        self.f.bind("phase3a_train_lock_report", self.f.phase3)
        self.f.save_old()

    def prepared(self):
        with self.f.patches():
            return cb.prepare(self.f.inputs)

    def preflight(self):
        with self.f.patches():
            return cb.preflight(self.f.inputs, self.f.preflight)

    def build(self):
        with self.f.patches():
            return cb.build(self.f.inputs, self.f.build_dir,
                            self.f.preflight / "cohort_source_preflight_report.json")

    def retarget(self, target):
        self.alias.unlink()
        self.alias.symlink_to(target, target_is_directory=True)

    def assert_preflight_rejected(self, error=sc.CohortError):
        with self.assertRaises(error):
            self.preflight()
        self.assertFalse(self.f.preflight.exists())

    def test_exact_dicc_alias_preflight_records_canonical_source_and_provenance(self):
        original_open = Path.open
        source_opens = []

        def guarded_open(path, *args, **kwargs):
            self.assertFalse(path.is_relative_to(self.alias), "content opened through historical alias")
            if path == self.f.source:
                source_opens.append(path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
                patch.object(sl, "read_identity_rows", wraps=sl.read_identity_rows) as inventory_read, \
                patch.object(sl, "verify_train_lock", wraps=verify_train_lock) as helper:
            report = self.preflight()
        self.assertEqual(cb.PREFLIGHT_STATUS, report["status"])
        helper.assert_called_once_with(self.f.project, self.f.phase3, expected_sha256=self.f.train_sha)
        self.assertTrue(source_opens)
        self.assertTrue(inventory_read.call_args_list)
        self.assertTrue(all(call.args[0] == self.f.source for call in inventory_read.call_args_list))
        self.assertEqual(self.f.train_sha, ar.sha_file(self.f.source))
        lock = ar.read_json(self.f.preflight / "cohort_source_lock.json")
        self.assertEqual({
            "historical_source_path": str(self.historical),
            "historical_source_resolved_path": str(self.f.source),
            "canonical_authoritative_train_path": str(self.f.source),
            "historical_alias_used": True,
            "historical_alias_component": str(self.alias),
            "historical_alias_link_target": "Academic_Research/outputs",
        }, lock["authoritative_train_provenance"])
        self.assertIn({"path": str(self.f.source), "sha256": self.f.train_sha,
                       "role": "authoritative Train"}, lock["artifacts"])
        self.assertEqual(ar.sha_file(self.f.preflight / "cohort_source_lock.json"),
                         report["cohort_source_lock_sha256"])
        with self.assertRaises(sc.CohortError):
            ar.safe_path(self.historical)

    def test_canonical_report_source_accepted_without_alias_exception(self):
        self.set_report_path(self.f.source)
        self.alias.unlink()
        prepared = self.prepared()
        self.assertEqual(self.f.source, prepared[1])
        provenance = prepared[4]["authoritative_train_provenance"]
        self.assertFalse(provenance["historical_alias_used"])
        self.assertIsNone(provenance["historical_alias_component"])
        self.assertIsNone(provenance["historical_alias_link_target"])

    def test_relative_historical_report_path_accepted(self):
        self.set_report_path(self.historical.relative_to(self.f.project))
        self.assertEqual(self.f.source, self.prepared()[1])

    def test_relative_canonical_3b1_path_accepted(self):
        self.f.old_lock["artifacts"]["authoritative_g1_train"]["path"] = str(
            self.f.source.relative_to(self.f.project))
        self.f.save_old()
        self.assertEqual(self.f.source, self.prepared()[1])

    def test_wrong_alias_file_rejected_even_with_identical_bytes(self):
        other = self.canonical_dir / "another.jsonl"
        other.write_bytes(self.f.source.read_bytes())
        self.set_report_path(self.alias / other.name)
        self.assert_preflight_rejected()

    def test_retarget_to_other_directory_rejected_even_with_identical_bytes(self):
        other = self.f.project / "MDU/other_outputs"
        target = other / self.tail
        target.parent.mkdir(parents=True)
        target.write_bytes(self.f.source.read_bytes())
        self.retarget(other)
        self.assert_preflight_rejected()

    def test_tail_mismatch_rejected_even_for_same_inode(self):
        other = self.canonical_dir / "hardlink.jsonl"
        os.link(self.f.source, other)
        self.set_report_path(self.alias / other.name)
        self.assertTrue(other.samefile(self.f.source))
        self.assert_preflight_rejected()

    def test_arbitrary_alias_location_rejected(self):
        link = self.f.project / "historical_source"
        link.symlink_to(self.f.source)
        self.set_report_path(link)
        self.assert_preflight_rejected()

    def test_extra_symlink_in_alias_target_rejected(self):
        intermediate = self.f.project / "MDU/intermediate"
        intermediate.symlink_to(self.canonical_dir, target_is_directory=True)
        self.retarget("intermediate")
        self.assert_preflight_rejected()

    def test_nested_symlink_rejected(self):
        parent = self.f.source.parent
        moved = parent.with_name("moved_variants")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        self.assert_preflight_rejected()

    def test_direct_canonical_file_symlink_rejected(self):
        moved = self.f.source.with_name("moved.jsonl")
        self.f.source.rename(moved)
        self.f.source.symlink_to(moved)
        self.assert_preflight_rejected()

    def test_canonical_3b1_record_using_historical_alias_rejected(self):
        self.f.old_lock["artifacts"]["authoritative_g1_train"]["path"] = str(self.historical)
        self.f.save_old()
        self.assert_preflight_rejected()

    def test_wrong_3b1_authoritative_sha_rejected_before_train_helper(self):
        self.f.old_lock["artifacts"]["authoritative_g1_train"]["sha256"] = "0" * 64
        self.f.save_old()
        with patch.object(sl, "verify_train_lock") as helper:
            self.assert_preflight_rejected()
        helper.assert_not_called()

    def test_changed_canonical_source_bytes_rejected(self):
        with self.f.source.open("ab") as stream:
            stream.write(b"\n")
        self.assert_preflight_rejected()

    def test_wrong_report_source_sha_rejected_by_existing_helper(self):
        self.set_report_path(self.historical, source={"path": str(self.historical), "sha256": "0" * 64})
        self.assert_preflight_rejected(sl.DatasetBuildError)

    def test_train_lock_fail_status_rejected_by_existing_helper(self):
        self.set_report_path(self.historical, status="FAIL")
        self.assert_preflight_rejected(sl.DatasetBuildError)

    def test_3b1_canonical_binding_to_different_file_rejected(self):
        other = self.canonical_dir / "another.jsonl"
        other.write_bytes(self.f.source.read_bytes())
        self.f.bind("authoritative_g1_train", other)
        self.f.save_old()
        self.assert_preflight_rejected()

    def test_helper_source_resolving_elsewhere_rejected(self):
        other = self.canonical_dir / "another.jsonl"
        other.write_bytes(self.f.source.read_bytes())
        result = SimpleNamespace(source_path=other, source_sha256=self.f.train_sha)
        with patch.object(sl, "verify_train_lock", return_value=result):
            self.assert_preflight_rejected()

    def test_helper_wrong_sha_rejected(self):
        result = SimpleNamespace(source_path=self.f.source, source_sha256="0" * 64)
        with patch.object(sl, "verify_train_lock", return_value=result):
            self.assert_preflight_rejected()

    def test_helper_alias_result_compared_after_resolution_but_canonical_returned(self):
        result = SimpleNamespace(source_path=self.historical, source_sha256=self.f.train_sha)
        with patch.object(sl, "verify_train_lock", return_value=result):
            self.assertEqual(self.f.source, self.prepared()[1])

    def test_helper_cannot_change_canonical_bytes_before_final_sha_check(self):
        def changed(*args, **kwargs):
            with self.f.source.open("ab") as stream:
                stream.write(b"\n")
            return SimpleNamespace(source_path=self.f.source, source_sha256=self.f.train_sha)
        with patch.object(sl, "verify_train_lock", side_effect=changed):
            self.assert_preflight_rejected()

    def test_missing_authoritative_source_rejected(self):
        self.f.source.unlink()
        self.assert_preflight_rejected(FileNotFoundError)

    def test_canonical_source_outside_project_rejected_before_read(self):
        other = self.f.root / "outside.jsonl"
        other.write_bytes(self.f.source.read_bytes())
        self.f.bind("authoritative_g1_train", other)
        self.f.save_old()
        self.set_report_path(other)
        with patch.object(sl, "verify_train_lock") as helper:
            self.assert_preflight_rejected()
        helper.assert_not_called()

    def test_historical_parent_traversal_rejected(self):
        self.set_report_path(self.alias / ".." / "outputs" / self.tail)
        self.assert_preflight_rejected()

    def test_formal_paths_rejected_before_source_reads(self):
        original = copy.deepcopy(self.f.old_lock)
        for name in ("Formal_Validation", "Formal_Test", "Validation", "Test"):
            with self.subTest(name=name):
                forbidden = self.canonical_dir / name / "G1_text_ocr.jsonl"
                self.f.old_lock = copy.deepcopy(original)
                self.f.old_lock["artifacts"]["authoritative_g1_train"]["path"] = str(forbidden)
                self.f.save_old()
                self.set_report_path(self.alias / name / "G1_text_ocr.jsonl")
                with patch.object(sl, "verify_train_lock") as helper:
                    self.assert_preflight_rejected()
                helper.assert_not_called()

    def test_generic_safe_path_symlink_file_directory_output_and_future_rejected(self):
        directory = self.f.root / "generic_directory"
        directory.mkdir()
        future = self.f.future("held", ["GroundLie360:synthetic-held"])
        for name, target in (("file", self.f.source), ("directory", directory),
                             ("output", self.f.outputs), ("future", future)):
            with self.subTest(name=name):
                link = self.f.root / ("generic_" + name + "_link")
                link.symlink_to(target, target_is_directory=name in {"directory", "output"})
                with self.assertRaises(sc.CohortError):
                    ar.safe_path(link)

    def test_all_top_level_input_symlinks_still_rejected_before_provenance(self):
        fields = ("project_root", "phase4a_config", "source_3b1_cohort_dir",
                  "step3b3_closure_dir", "neutral_dir", "stage_a_invariance_report")
        original_inputs = self.f.inputs
        for field in fields:
            with self.subTest(field=field):
                target = getattr(original_inputs, field)
                link = self.f.root / (field + "_alias")
                link.symlink_to(target, target_is_directory=target.is_dir())
                self.f.inputs = replace(original_inputs, **{field: link})
                with patch.object(cb, "resolve_source") as resolver:
                    self.assert_preflight_rejected()
                resolver.assert_not_called()

    def test_symlinked_future_manifest_rejected_before_provenance(self):
        future = self.f.future("held", ["GroundLie360:synthetic-held"])
        link = self.f.root / "future_alias.json"
        link.symlink_to(future)
        self.f.inputs = replace(self.f.inputs, future_exclusion_manifests=(link,))
        with patch.object(cb, "resolve_source") as resolver:
            self.assert_preflight_rejected()
        resolver.assert_not_called()

    def test_symlinked_output_rejected_before_provenance(self):
        target = self.f.root / "output_target"
        target.mkdir()
        self.f.outputs.mkdir()
        self.f.preflight.symlink_to(target, target_is_directory=True)
        with patch.object(cb, "resolve_source") as resolver, self.assertRaises(sc.CohortError):
            self.preflight()
        resolver.assert_not_called()
        self.assertTrue(self.f.preflight.is_symlink())
        self.assertEqual([], list(target.iterdir()))

    def test_other_source_lock_record_symlink_rejected(self):
        link = self.f.root / "train_report_alias.json"
        link.symlink_to(self.f.phase3)
        self.f.old_lock["artifacts"]["phase3a_train_lock_report"]["path"] = str(link)
        self.f.save_old()
        self.assert_preflight_rejected()

    def test_retarget_after_preflight_rejects_build_before_exposure(self):
        self.preflight()
        self.retarget(self.f.root / "missing_outputs")
        with patch.object(cb.Phase4ANormalizationExposureAdapter, "from_project_root") as exposure:
            with self.assertRaises(sc.CohortError):
                self.build()
        exposure.assert_not_called()
        self.assertFalse(self.f.build_dir.exists())

    def test_changed_raw_link_target_same_resolution_invalidates_approved_preflight(self):
        self.preflight()
        self.retarget(self.canonical_dir)
        self.assertEqual(self.f.source, self.historical.resolve(strict=True))
        with patch.object(cb.Phase4ANormalizationExposureAdapter, "from_project_root") as exposure:
            with self.assertRaisesRegex(sc.CohortError, "not byte-consistent"):
                self.build()
        exposure.assert_not_called()
        self.assertFalse(self.f.build_dir.exists())

    def test_final_preflight_freeze_rechecks_link_and_cleans_staging(self):
        def changed(output, artifacts, ledger):
            self.retarget(self.canonical_dir)
            return ar.freeze(output, artifacts, ledger)
        with patch.object(cb, "freeze", side_effect=changed):
            self.assert_preflight_rejected()
        self.assertEqual([], list(self.f.outputs.iterdir()))

    def test_final_build_freeze_rechecks_link_and_cleans_staging(self):
        self.preflight()
        def changed(output, artifacts, ledger):
            self.retarget(self.canonical_dir)
            return ar.freeze(output, artifacts, ledger)
        with patch.object(cb, "freeze", side_effect=changed), self.assertRaises(sc.CohortError):
            self.build()
        self.assertFalse(self.f.build_dir.exists())
        self.assertEqual([self.f.preflight], list(self.f.outputs.iterdir()))

    def test_ledger_rechecks_alias_after_hash_revalidation(self):
        ledger = self.prepared()[3]
        original = ar.Ledger.revalidate
        def changed(instance):
            original(instance)
            self.retarget(self.canonical_dir)
        with patch.object(ar.Ledger, "revalidate", changed), self.assertRaises(sc.CohortError):
            ledger.revalidate()

    def test_generic_ledger_cannot_omit_train_provenance_revalidation(self):
        with self.f.patches(), self.assertRaises(sc.CohortError):
            sl.resolve_source(self.f.inputs, ar.Ledger(), self.f.old_lock, frozen.load_preregistration())

    def test_synthetic_build_uses_canonical_source_preserves_frozen_inputs(self):
        paths = [self.f.source, self.f.phase3, *self.f.old.iterdir()]
        original = {path: path.read_bytes() for path in paths}
        self.preflight()
        with patch.object(sl, "read_identity_rows", wraps=sl.read_identity_rows) as reader:
            report = self.build()
        self.assertEqual(cb.PASS_STATUS, report["status"])
        self.assertEqual({"GroundLie360": 60, "TRUE-3MFact": 60}, report["selected_dataset_counts"])
        self.assertEqual({"repair_train": 96, "repair_dev": 24}, report["split_counts"])
        self.assertTrue(reader.call_args_list)
        self.assertTrue(all(call.args[0] == self.f.source for call in reader.call_args_list))
        before = ar.read_json(self.f.preflight / "cohort_source_lock.json")
        after = ar.read_json(self.f.build_dir / "development_cohort_source_lock.json")
        self.assertEqual(before["authoritative_train_provenance"], after["authoritative_train_provenance"])
        for path, data in original.items():
            self.assertEqual(data, path.read_bytes())
        self.assertEqual("Academic_Research/outputs", os.readlink(self.alias))


if __name__ == "__main__":
    unittest.main()
