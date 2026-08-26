import csv
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from schemas import SourceType
from scripts.selector_fidelity_audit import cross_case as cross_case_module
from scripts.selector_fidelity_audit.audit import AuditInputError, candidate_pool_sha256
from scripts.selector_fidelity_audit.cross_case import (
    NATIVE_PHASE4A,
    PUBLIC_RESULT,
    CandidateCase,
    DiscoveryRoot,
    build_pre_scoring_manifest,
    canonicalize_underlying_case_id,
    classify_cross_case,
    compute_cross_case_metrics,
    detect_artifact_contract,
    discover_eligible_cases,
    generate_direct_grounding_probes,
    manifest_contains_selector_output_keys,
    reconstruct_native_candidate_pool,
    run_cross_case_audit,
    run_reproduction_gate,
    select_cases_score_blind,
)


def _unit(unit_id, source_type, text, score):
    return {
        "unit_id": unit_id,
        "source_type": source_type,
        "text": text,
        "eligible_for_frozen_g1": True,
        "selection_score": score,
        "logits": {"fake": score - 0.25, "real": score + 0.25},
        "producer": "public-qa-fixture",
    }


def _artifact(case_id, dataset, *, score_offset=0.0, duplicate_ocr=False):
    units = [
        _unit(f"{case_id}-t1", "transcript", "The speaker discusses public safety.", 0.9 + score_offset),
        _unit(f"{case_id}-t2", "transcript", "Experts describe the event timeline.", 0.8 + score_offset),
        _unit(f"{case_id}-t3", "transcript", "A third transcript statement is available.", 0.7 + score_offset),
        _unit(f"{case_id}-o1", "ocr", "PUBLIC SAFETY NOTICE", 0.6 + score_offset),
        _unit(
            f"{case_id}-o2",
            "ocr",
            "PUBLIC SAFETY NOTICE" if duplicate_ocr else "EVENT STARTS AT NOON",
            0.5 + score_offset,
        ),
        _unit(f"{case_id}-o3", "ocr", "OFFICIAL ARCHIVE", 0.4 + score_offset),
    ]
    return {
        "dataset": dataset,
        "outcome": {
            "result": {
                "session_id": case_id,
                "claim": f"Original source claim for {case_id}.",
                "verdict": {"display_verdict": "Real"},
                "evidence": {
                    "g1_exposure_units": units,
                    "g1_top_k_explanation_unit_ids": [
                        item["unit_id"] for item in units[:5]
                    ],
                    "visual_supplemental_units": [],
                },
            }
        },
    }


def _native_artifact(case_id, dataset, *, text_suffix=""):
    public = _artifact(case_id, dataset)
    public_units = public["outcome"]["result"]["evidence"]["g1_exposure_units"]
    outputs = []
    for item in public_units:
        unit_type = item["source_type"]
        outputs.append(
            {
                "unit_id": item["unit_id"],
                "unit_type": unit_type,
                "modality": "ocr" if unit_type == "ocr" else "text",
                "text": item["text"] + text_suffix,
                "selection_score": item["selection_score"],
                "veracity_logits": dict(item["logits"]),
            }
        )
    return {
        "dataset": dataset,
        "case_id": case_id,
        "claim": f"Original source claim for {case_id}.",
        "unit_outputs": outputs,
        "top_k_selection_units": [
            {"unit_id": item["unit_id"], "selection_score": item["selection_score"]}
            for item in outputs[:5]
        ],
        "sample_logits": {"fake": 1.0, "real": 2.0},
        "probabilities": {"fake": 0.25, "real": 0.75},
        "prediction": "real",
    }


def _write_artifact(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class DeterministicRunner:
    def __init__(self, manifest_path=None):
        self.calls = []
        self.manifest_path = manifest_path

    @staticmethod
    def _reference_score(unit_id):
        suffix = unit_id.rsplit("-", 1)[-1]
        return {"t1": 0.9, "t2": 0.8, "t3": 0.7, "o1": 0.6, "o2": 0.5, "o3": 0.4}[suffix]

    def run(self, session_id, claim, units):
        units = list(units)
        is_probe = claim.startswith("The on-screen") or claim.startswith("The speaker")
        if is_probe and self.manifest_path is not None:
            self.assert_manifest_frozen()
        self.calls.append((claim, [unit.unit_id for unit in units]))
        for index, unit in enumerate(units):
            if is_probe:
                score = 1.0 - index / 10.0
            else:
                score = self._reference_score(unit.unit_id)
            unit.selection_score = score
            unit.logits = {"fake": score - 0.25, "real": score + 0.25}
        # Deliberately authoritative: this list, rather than recomputed rank,
        # defines membership in the audit output.
        top_k = units[:5]
        return SimpleNamespace(all_units=units, top_k_units=top_k)

    def assert_manifest_frozen(self):
        if not self.manifest_path.is_file():
            raise AssertionError("probe scoring started before manifest write")
        digest_path = self.manifest_path.with_suffix(".sha256")
        if not digest_path.is_file():
            raise AssertionError("probe scoring started before manifest hash write")


class ForbiddenScoreMapping(dict):
    def get(self, key, default=None):
        if key in {"selection_score", "veracity_logits", "logits"}:
            raise AssertionError(f"score-blind discovery consulted {key}")
        return super().get(key, default)


class ForbiddenTopKContents(list):
    def __iter__(self):
        raise AssertionError("score-blind discovery iterated authoritative Top-k")

    def __getitem__(self, index):
        raise AssertionError("score-blind discovery indexed authoritative Top-k")


class SelectorFidelityCrossCaseTests(unittest.TestCase):
    def test_native_contract_detection_reconstruction_and_score_blind_fields(self):
        payload = _native_artifact("TRUE-3MFact:train:10145403", "TRUE-3MFact")
        payload["unit_outputs"] = [
            ForbiddenScoreMapping(item) for item in payload["unit_outputs"]
        ]
        payload["top_k_selection_units"] = ForbiddenTopKContents(
            payload["top_k_selection_units"]
        )

        contract = detect_artifact_contract(payload)
        units = reconstruct_native_candidate_pool(payload)

        self.assertEqual(NATIVE_PHASE4A, contract)
        self.assertEqual(
            [
                "TRUE-3MFact:train:10145403-t1",
                "TRUE-3MFact:train:10145403-t2",
                "TRUE-3MFact:train:10145403-t3",
                "TRUE-3MFact:train:10145403-o1",
                "TRUE-3MFact:train:10145403-o2",
                "TRUE-3MFact:train:10145403-o3",
            ],
            [unit.unit_id for unit in units],
        )
        self.assertEqual(
            [
                SourceType.TRANSCRIPT,
                SourceType.TRANSCRIPT,
                SourceType.TRANSCRIPT,
                SourceType.OCR,
                SourceType.OCR,
                SourceType.OCR,
            ],
            [unit.source_type for unit in units],
        )
        self.assertTrue(all(unit.selection_score is None for unit in units))
        self.assertTrue(all(unit.logits is None for unit in units))

    def test_native_candidate_hash_ignores_all_selector_outputs(self):
        original = _native_artifact("case-native", "train-derived")
        changed = copy.deepcopy(original)
        for index, item in enumerate(changed["unit_outputs"]):
            item["selection_score"] = 1000.0 + index
            item["veracity_logits"] = {"fake": -500.0, "real": 500.0}
        changed["top_k_selection_units"] = list(
            reversed(changed["top_k_selection_units"])
        )
        changed["sample_logits"] = {"fake": 999.0, "real": -999.0}
        changed["probabilities"] = {"fake": 1.0, "real": 0.0}
        changed["prediction"] = "fake"

        original_hash = candidate_pool_sha256(
            reconstruct_native_candidate_pool(original)
        )
        changed_hash = candidate_pool_sha256(
            reconstruct_native_candidate_pool(changed)
        )

        self.assertEqual(original_hash, changed_hash)

    def test_native_reproduction_uses_native_scores_logits_and_only_native_top_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _native_artifact("case-native", "train-derived")
            # A decoy public-style Top-k must never override native authority.
            payload["top_k_units"] = [{"unit_id": payload["unit_outputs"][-1]["unit_id"]}]
            _write_artifact(root, "native.jsonl", payload)
            inventory, cases = discover_eligible_cases(
                [DiscoveryRoot("train-derived", root)]
            )

            gate = run_reproduction_gate(cases[0], DeterministicRunner())

        self.assertEqual(NATIVE_PHASE4A, inventory["eligible_cases"][0]["artifact_contract"])
        self.assertEqual(NATIVE_PHASE4A, gate["artifact_contract"])
        self.assertTrue(gate["passed"])
        self.assertEqual(0.0, gate["max_selection_score_difference"])
        self.assertEqual(0.0, gate["max_fake_logit_difference"])
        self.assertTrue(gate["top_k_unit_ids_identical"])

    def test_reproduction_tolerance_remains_inclusive_at_one_e_minus_six(self):
        class OffsetRunner(DeterministicRunner):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def run(self, session_id, claim, units):
                result = super().run(session_id, claim, units)
                result.all_units[0].selection_score += self.offset
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(
                root,
                "native.json",
                _native_artifact("native-tol", "train-derived"),
            )
            _, cases = discover_eligible_cases([DiscoveryRoot("train-derived", root)])
            case = cases[0]
            boundary = run_reproduction_gate(case, OffsetRunner(1e-6))
            outside = run_reproduction_gate(case, OffsetRunner(1.01e-6))

        self.assertTrue(boundary["passed"])
        self.assertFalse(outside["passed"])

    def test_canonical_underlying_identity_removes_only_explicit_split_tokens(self):
        self.assertEqual(
            "TRUE-3MFact:10145403",
            canonicalize_underlying_case_id(
                "TRUE-3MFact", "TRUE-3MFact:train:10145403"
            ),
        )
        self.assertEqual(
            "GroundLie360:13025004",
            canonicalize_underlying_case_id(
                "GroundLie360", "GroundLie360:test:13025004"
            ),
        )
        self.assertEqual(
            "TRUE-3MFact:1014540301",
            canonicalize_underlying_case_id("TRUE-3MFact:val", "1014540301"),
        )
        self.assertEqual(
            "TRUE-3MFact:phase2:10145403",
            canonicalize_underlying_case_id(
                "TRUE-3MFact", "phase2:10145403"
            ),
        )

    def test_selection_deduplicates_underlying_identity_and_pool_hash_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(
                root,
                "one.json",
                _native_artifact("TRUE-3MFact:train:10145403", "TRUE-3MFact"),
            )
            _write_artifact(
                root,
                "same-identity-new-pool.json",
                _native_artifact(
                    "TRUE-3MFact:test:10145403",
                    "TRUE-3MFact",
                    text_suffix=" changed preprocessing",
                ),
            )
            _write_artifact(
                root,
                "different-example.json",
                _native_artifact("TRUE-3MFact:train:10145404", "TRUE-3MFact"),
            )
            same_pool_new_identity = _native_artifact(
                "TRUE-3MFact:train:10145403", "TRUE-3MFact"
            )
            same_pool_new_identity["case_id"] = "GroundLie360:train:13025004"
            same_pool_new_identity["dataset"] = "GroundLie360"
            _write_artifact(
                root,
                "different-identity-same-pool.json",
                same_pool_new_identity,
            )
            inventory, cases = discover_eligible_cases(
                [DiscoveryRoot("train-derived", root)]
            )
            selected = select_cases_score_blind(cases, 5)

        self.assertEqual(4, inventory["eligible_case_count"])
        self.assertEqual(3, len(selected))
        self.assertEqual(
            {
                "GroundLie360:13025004",
                "TRUE-3MFact:10145403",
                "TRUE-3MFact:10145404",
            },
            {case.canonical_underlying_case_id for case in selected},
        )
        self.assertEqual(
            len(selected), len({case.candidate_pool_sha256 for case in selected})
        )

    def test_selector_audit_and_current_output_directories_are_not_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_output = root / "current-cross-case-output"
            _write_artifact(root, "eligible.json", _native_artifact("eligible", "qa"))
            _write_artifact(
                root / "selector_fidelity_audit",
                "self.json",
                _native_artifact("self-audit", "qa"),
            )
            _write_artifact(
                current_output,
                "self.json",
                _native_artifact("current-output", "qa"),
            )

            inventory, cases = discover_eligible_cases(
                [DiscoveryRoot("qa", root)], excluded_paths=(current_output,)
            )

        self.assertTrue(inventory["self_discovery_excluded"])
        self.assertEqual(["eligible"], [case.case_id for case in cases])

    def test_selected_manifest_exists_before_native_reference_fields_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            output = Path(directory) / "output"
            root.mkdir()
            _write_artifact(root, "native.json", _native_artifact("native-order", "qa"))
            selected_manifest_path = output / "selected_case_manifest.json"
            original = cross_case_module.extract_reference_snapshot

            def guarded_extract(case):
                self.assertTrue(selected_manifest_path.is_file())
                selected_payload = json.loads(
                    selected_manifest_path.read_text(encoding="utf-8")
                )
                self.assertEqual(NATIVE_PHASE4A, selected_payload["selected_cases"][0]["artifact_contract"])
                return original(case)

            with patch.object(
                cross_case_module,
                "extract_reference_snapshot",
                side_effect=guarded_extract,
            ):
                run_cross_case_audit(
                    discovery_roots=[DiscoveryRoot("qa", root)],
                    runtime_config_path=Path("unused.json"),
                    output_dir=output,
                    runner=DeterministicRunner(),
                )

            inventory = json.loads(
                (output / "eligible_case_inventory.json").read_text(encoding="utf-8")
            )
            reproduction = json.loads(
                (output / "reproduction_gates.json").read_text(encoding="utf-8")
            )

        self.assertEqual(NATIVE_PHASE4A, inventory["eligible_cases"][0]["artifact_contract"])
        self.assertIn(
            "canonical_underlying_case_id", inventory["eligible_cases"][0]
        )
        self.assertEqual(NATIVE_PHASE4A, reproduction["cases"][0]["artifact_contract"])

    def test_score_blind_discovery_selection_is_deterministic_and_diverse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "a"
            b = root / "b"
            a.mkdir()
            b.mkdir()
            _write_artifact(a, "z.json", _artifact("case-z", "dataset-a", score_offset=100.0))
            _write_artifact(a, "a.json", _artifact("case-a", "dataset-a", score_offset=-100.0))
            _write_artifact(b, "b.json", _artifact("case-b", "dataset-b", score_offset=500.0))
            forbidden = a / "Test"
            forbidden.mkdir()
            (forbidden / "must-not-open.json").write_text("not-json", encoding="utf-8")

            inventory, cases = discover_eligible_cases(
                [DiscoveryRoot("fallback-a", a), DiscoveryRoot("fallback-b", b)]
            )
            forward = select_cases_score_blind(cases, 2)
            reverse = select_cases_score_blind(list(reversed(cases)), 2)

        self.assertFalse(inventory["formal_test_accessed"])
        self.assertFalse(inventory["selector_outputs_inspected_for_case_selection"])
        self.assertEqual(["case-a", "case-b"], [case.case_id for case in forward])
        self.assertEqual(
            [case.candidate_pool_sha256 for case in forward],
            [case.candidate_pool_sha256 for case in reverse],
        )
        with self.assertRaisesRegex(AuditInputError, "Validation/Test"):
            DiscoveryRoot("bad", Path(directory) / "Validation")
        with self.assertRaisesRegex(AuditInputError, "Validation/Test"):
            DiscoveryRoot("Test", Path(directory))

    def test_reproduction_gate_requires_scores_logits_order_and_authoritative_top_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_artifact(root, "case.json", _artifact("case-r", "qa"))
            _, cases = discover_eligible_cases([DiscoveryRoot("qa", root)])
            case = cases[0]
            runner = DeterministicRunner()

            gate = run_reproduction_gate(case, runner)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["outcome"]["result"]["evidence"]["g1_top_k_explanation_unit_ids"] = [
                f"case-r-o3"
            ]
            _write_artifact(root, "altered.json", payload)
            altered = replace(case, payload=payload)
            failed = run_reproduction_gate(altered, runner)

        self.assertEqual(PUBLIC_RESULT, case.artifact_contract)
        self.assertEqual(PUBLIC_RESULT, gate["artifact_contract"])
        self.assertTrue(gate["passed"])
        self.assertEqual(0.0, gate["max_selection_score_difference"])
        self.assertEqual(0.0, gate["max_fake_logit_difference"])
        self.assertTrue(gate["top_k_unit_ids_identical"])
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["top_k_unit_ids_identical"])

    def test_probe_generation_is_balanced_score_blind_and_marks_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_artifact(
                root,
                "case.json",
                _artifact("case-d", "qa", score_offset=999.0, duplicate_ocr=True),
            )
            _, cases = discover_eligible_cases([DiscoveryRoot("qa", root)])
            case = cases[0]
            probes = generate_direct_grounding_probes(case)
            manifest = build_pre_scoring_manifest(
                [case], {case.candidate_pool_sha256: probes}
            )

        self.assertEqual(2, sum(probe.expected_modality == "OCR" for probe in probes))
        self.assertEqual(2, sum(probe.expected_modality == "TRANSCRIPT" for probe in probes))
        first_ocr = next(probe for probe in probes if probe.expected_modality == "OCR")
        self.assertEqual(
            ("case-d-o1", "case-d-o2"), first_ocr.expected_relevant_unit_ids
        )
        self.assertFalse(manifest_contains_selector_output_keys(manifest))
        encoded = json.dumps(manifest)
        self.assertNotIn("999", encoded)
        self.assertNotIn("selection_score", encoded)

    def test_full_orchestration_freezes_manifest_before_scoring_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            output = Path(directory) / "output"
            root.mkdir()
            for index in range(4):
                _write_artifact(
                    root,
                    f"case-{index}.json",
                    _artifact(f"case-{index}", f"dataset-{index % 2}"),
                )
            manifest_path = output / "cross_case_probe_manifest_pre_scoring.json"
            runner = DeterministicRunner(manifest_path)

            metrics = run_cross_case_audit(
                discovery_roots=[DiscoveryRoot("qa", root)],
                runtime_config_path=Path("unused.json"),
                output_dir=output,
                target_case_count=5,
                runner=runner,
            )

            manifest_bytes = manifest_path.read_bytes()
            declared_hash = (
                output / "cross_case_probe_manifest_pre_scoring.sha256"
            ).read_text(encoding="utf-8").split()[0]
            with (output / "cross_case_unit_rankings.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            review_header = (output / "cross_case_probe_review.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]

        self.assertEqual(hashlib.sha256(manifest_bytes).hexdigest(), declared_hash)
        self.assertEqual(4, metrics["reproduced_case_count"])
        self.assertEqual(8, metrics["ocr_probe_count"])
        self.assertEqual(8, metrics["transcript_probe_count"])
        self.assertEqual(4 + 16, len(runner.calls))
        first_probe = rows[0]["probe_id"]
        first_probe_rows = [row for row in rows if row["probe_id"] == first_probe]
        self.assertEqual(
            [0, 1, 2, 3, 4, 5],
            [int(row["candidate_exposure_index"]) for row in first_probe_rows],
        )
        self.assertEqual(
            ["true", "true", "true", "true", "true", "false"],
            [row["top_k_member"] for row in first_probe_rows],
        )
        self.assertNotIn("selection_score", review_header)
        self.assertNotIn("top_k", review_header)
        self.assertFalse(metrics["formal_test_accessed"])

    def test_cross_case_aggregation_and_classification_gates(self):
        per_probe = []
        for case_index in range(4):
            for modality, ranks in (("OCR", [6, 7]), ("TRANSCRIPT", [1, 2])):
                for probe_index, rank in enumerate(ranks):
                    per_probe.append(
                        {
                            "case_id": f"case-{case_index}",
                            "dataset": "qa",
                            "probe_id": f"p-{case_index}-{modality}-{probe_index}",
                            "expected_modality": modality,
                            "highest_relevant_unit_rank": rank,
                            "mrr": 1.0 / rank,
                            "ndcg_at_5": 0.0 if rank > 5 else 1.0,
                            "top_5_unit_ids": [f"u-{n}" for n in range(5)],
                            "top_5_modality_composition": (
                                {"transcript": 5}
                                if modality == "OCR"
                                else {"transcript": 4, "ocr": 1}
                            ),
                            "direct_grounding_flags": (
                                [{"flag": "DIRECT_GROUNDING_TOP5_MISS"}]
                                if modality == "OCR"
                                else []
                            ),
                        }
                    )
        metrics = compute_cross_case_metrics(per_probe, 4)

        self.assertEqual("CROSS_CASE_MODALITY_BIAS_CONFIRMED", metrics["classification"])
        self.assertEqual(0.0, metrics["micro_ocr_hit_at_5"])
        self.assertEqual(1.0, metrics["micro_transcript_hit_at_5"])
        self.assertEqual(1.0, metrics["pool_fraction_with_ocr_hit_below_transcript"])
        self.assertEqual(1.0, metrics["ocr_direct_grounding_miss_rate"])
        self.assertEqual(
            "INCONCLUSIVE",
            classify_cross_case({**metrics, "reproduced_case_count": 3}),
        )
        self.assertEqual(
            "GENERAL_SELECTOR_RELEVANCE_FAILURE",
            classify_cross_case(
                {
                    **metrics,
                    "micro_ocr_hit_at_5": 0.25,
                    "micro_transcript_hit_at_5": 0.5,
                    "pool_fraction_with_ocr_hit_below_transcript": 0.5,
                }
            ),
        )

    def test_insufficient_pool_writes_all_required_outputs_without_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            output = Path(directory) / "output"
            root.mkdir()
            _write_artifact(
                root,
                "too-small.json",
                {
                    "claim": "Small pool.",
                    "g1_exposure_units": [
                        _unit("t1", "transcript", "one transcript", 0.2),
                        _unit("o1", "ocr", "ONE OCR", 0.1),
                    ],
                },
            )
            metrics = run_cross_case_audit(
                discovery_roots=[DiscoveryRoot("qa", root)],
                runtime_config_path=Path("missing-runtime-config.json"),
                output_dir=output,
            )
            required = {
                "eligible_case_inventory.json",
                "selected_case_manifest.json",
                "reproduction_gates.json",
                "cross_case_probe_manifest_pre_scoring.json",
                "cross_case_probe_manifest_pre_scoring.sha256",
                "cross_case_probe_review.csv",
                "cross_case_unit_rankings.csv",
                "cross_case_metrics.json",
                "cross_case_summary.md",
            }
            names = {path.name for path in output.iterdir() if path.is_file()}

        self.assertEqual("BLOCKED", metrics["audit_status"])
        self.assertEqual("INCONCLUSIVE", metrics["classification"])
        self.assertEqual(0, metrics["eligible_case_count"])
        self.assertTrue(required <= names)


if __name__ == "__main__":
    unittest.main()
