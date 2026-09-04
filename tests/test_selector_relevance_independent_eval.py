from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.selector_relevance_gate.runtime import (
    DICCEvaluationRuntime,
    RuntimeIntegrationError,
    TrainingArtifacts,
)
from scripts.selector_relevance_gate.schemas import EvaluationUnit, PredictionSnapshot
from scripts.selector_relevance_independent_eval.evaluator import (
    run_one_shot_evaluation,
    run_preflight,
)
from scripts.selector_relevance_independent_eval.metrics_adapter import (
    aggregate_metrics,
    calculate_repair_gate,
    evaluate_case_scores,
)
from scripts.selector_relevance_independent_eval.run_evaluation import build_parser, main
from scripts.selector_relevance_independent_eval.schemas import (
    FINAL_GOLD_FIELDS,
    IndependentCase,
    IndependentEvaluationError,
)
from scripts.selector_relevance_independent_eval.source_loader import (
    _frozen_protocol,
    _validate_phase4a_config,
    _validate_coverage,
    prepare_inputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_locked(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    digest = _sha(path)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


class Synthetic3B3Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "unirumor"
        phase3 = self.project / "MDU" / "scripts" / "clip12_phase3_common"
        phase4a = (
            self.project
            / "MDU"
            / "scripts"
            / "clip12_phase4a_inference_handoff"
        )
        phase3.mkdir(parents=True)
        phase4a.mkdir(parents=True)
        (phase3 / "clip12p3_model.py").write_text("# synthetic authoritative source\n")
        (phase4a / "clip12p4a_engine.py").write_text(
            "# synthetic authoritative source\n"
        )
        self.defense = self.project / "MDU" / "Defense_Engineering"
        self.defense.mkdir()
        self.cohort = root / "01_cohort_build_r2"
        self.cohort.mkdir()
        self.gold = root / "03_final_relevance_gold"
        self.gold.mkdir()
        self.neutral = root / "neutral"
        self.neutral.mkdir()
        self.training = root / "training"
        (self.training / "seed_42").mkdir(parents=True)
        self.checkpoint = self.project / "artifacts" / "frozen_g1.ckpt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"synthetic frozen g1 checkpoint")
        self.checkpoint_sha = _sha(self.checkpoint)
        self.selector = self.training / "seed_42" / "selector_head.pt"
        self.selector.write_bytes(b"synthetic frozen seed-42 selector")
        self.selector_sha = _sha(self.selector)
        self.config = root / "phase4a_config.json"
        self.config.write_bytes(
            _json_bytes(
                {
                    "checkpoint_path": "artifacts/frozen_g1.ckpt",
                    "model_name": "microsoft/deberta-v3-base",
                    "maximum_units_per_sample": 24,
                    "max_length": 256,
                    "pooling": "max",
                }
            )
        )
        self.config_sha = _sha(self.config)
        self.case_rows, self.request_rows, self.gold_rows = self._cases()
        self.cohort_hashes = self._write_cohort()
        self._write_gold()
        self.stage_a = self._write_stage_a()
        self.training_artifacts = self._write_training()

    def _cases(self):
        manifests = []
        requests = []
        gold = []
        for case_index in range(30):
            dataset = "GroundLie360" if case_index < 15 else "TRUE-3MFact"
            canonical = f"{dataset}:synthetic-{case_index:03d}"
            count = 10 if case_index < 19 else 9
            ids = [f"u-{case_index:03d}-{position:02d}" for position in range(count)]
            candidates = [
                {
                    "unit_id": unit_id,
                    "unit_type": "transcript",
                    "modality": "text",
                    "text": f"Synthetic candidate {case_index}-{position}",
                    "original_candidate_position": position,
                }
                for position, unit_id in enumerate(ids)
            ]
            manifests.append(
                {
                    "dataset": dataset,
                    "canonical_case_id": canonical,
                    "original_case_id": f"synthetic-{case_index:03d}",
                    "sampling_hash": hashlib.sha256(canonical.encode()).hexdigest(),
                    "model_exposed_unit_count": count,
                    "candidate_unit_ids_in_original_order": ids,
                    "candidate_unit_types_in_original_order": ["transcript"] * count,
                    "candidate_modalities_in_original_order": ["text"] * count,
                }
            )
            requests.append(
                {
                    "audit_case_id": f"audit-{case_index:03d}",
                    "dataset": dataset,
                    "canonical_case_id": canonical,
                    "original_case_id": f"synthetic-{case_index:03d}",
                    "claim": f"Original frozen claim {case_index}",
                    "candidate_units": candidates,
                }
            )
            evaluable = case_index < 13 or case_index >= 15
            for position, unit_id in enumerate(ids):
                # The first case deliberately has two DIRECT units.
                direct = evaluable and (position == 0 or (case_index == 0 and position == 1))
                label = "DIRECT" if direct else "RELATED"
                gold.append(
                    {
                        "dataset": dataset,
                        "canonical_case_id": canonical,
                        "unit_id": unit_id,
                        "original_candidate_position": position,
                        "final_relevance_label": label,
                        "binary_direct_relevance_target": int(direct),
                        "resolution_source": "REVIEWER_AGREEMENT",
                    }
                )
        if len(gold) != 289:
            raise AssertionError("synthetic unit count changed")
        return manifests, requests, gold

    def _write_cohort(self) -> dict[str, str]:
        preregistration = {
            "status": "PREREGISTERED_BEFORE_REVIEW_AND_SCORING",
            "implementation_revision": "step2.6r-3b1-r2-v1",
            "deployment_candidate_seed": 42,
            "direct_relevance_binary_mapping": {
                "DIRECT": 1,
                "RELATED": 0,
                "IRRELEVANT": 0,
                "UNREADABLE": 0,
            },
            "future_step_2_6r_3b3": _frozen_protocol(),
            "prohibitions": {
                "training": True,
                "calibration": True,
                "seed_42_43_44_selection": True,
                "selector_architecture_change": True,
                "threshold_tuning": True,
                "iterative_reuse_after_scores": True,
            },
        }
        values = {
            "cohort_source_lock.json": {
                "status": "PASS",
                "implementation_revision": "step2.6r-3b1-r2-v1",
                "artifacts": {
                    "phase4a_configuration": {
                        "path": str(self.config),
                        "sha256": self.config_sha,
                    }
                },
                "sealed_historical_reference_artifacts_opened": False,
                "formal_validation_accessed": False,
                "formal_test_accessed": False,
            },
            "eligibility_inventory.json": {
                "authoritative_source_case_count": 3878,
                "authoritative_groundlie_case_count": 1636,
                "authoritative_true3m_case_count": 2242,
                "selected_groundlie_count": 15,
                "selected_true3m_count": 15,
                "selected_total_count": 30,
            },
            "selected_case_manifest.json": {
                "status": "FROZEN",
                "implementation_revision": "step2.6r-3b1-r2-v1",
                "sampling_salt": "step2.6r-3b1-independent-audit-v1",
                "selected_cases": self.case_rows,
            },
            "independent_relevance_audit_requests.jsonl": self.request_rows,
            "independent_audit_preregistration.json": preregistration,
        }
        hashes = {}
        for name, value in values.items():
            payload = _jsonl_bytes(value) if name.endswith(".jsonl") else _json_bytes(value)
            hashes[name] = _write_locked(self.cohort / name, payload)
        build = {
            "status": "INDEPENDENT_SCORE_BLIND_AUDIT_COHORT_BUILD_PASS",
            "implementation_revision": "step2.6r-3b1-r2-v1",
            "authoritative_g1_case_count": 3878,
            "groundlie_source_case_count": 1636,
            "true3m_source_case_count": 2242,
            "selected_groundlie_count": 15,
            "selected_true3m_count": 15,
            "selected_total_count": 30,
            "selected_candidate_unit_count": 289,
            "selection_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_heldout_reference_content_accessed": False,
            "cohort_source_lock_sha256": hashes["cohort_source_lock.json"],
            "eligibility_inventory_sha256": hashes["eligibility_inventory.json"],
            "selected_case_manifest_sha256": hashes["selected_case_manifest.json"],
            "independent_relevance_audit_requests_sha256": hashes[
                "independent_relevance_audit_requests.jsonl"
            ],
            "preregistration_sha256": hashes["independent_audit_preregistration.json"],
        }
        hashes["build_report.json"] = _write_locked(
            self.cohort / "build_report.json", _json_bytes(build)
        )
        return hashes

    def _write_gold(self) -> None:
        by_case: dict[str, list[dict]] = {}
        for row in self.gold_rows:
            by_case.setdefault(row["canonical_case_id"], []).append(row)
        case_coverage = []
        per_dataset = {
            "GroundLie360": {"total_case_count": 15, "evaluable_case_count": 13},
            "TRUE-3MFact": {"total_case_count": 15, "evaluable_case_count": 15},
        }
        for canonical in sorted(by_case):
            rows = by_case[canonical]
            counts = {label: 0 for label in ("DIRECT", "RELATED", "IRRELEVANT", "UNREADABLE")}
            for row in rows:
                counts[row["final_relevance_label"]] += 1
            case_coverage.append(
                {
                    "dataset": rows[0]["dataset"],
                    "canonical_case_id": canonical,
                    "candidate_count": len(rows),
                    "DIRECT_count": counts["DIRECT"],
                    "RELATED_count": counts["RELATED"],
                    "IRRELEVANT_count": counts["IRRELEVANT"],
                    "UNREADABLE_count": counts["UNREADABLE"],
                    "has_DIRECT": counts["DIRECT"] >= 1,
                }
            )
        source_lock = {
            "status": "PASS",
            "implementation_revision": "step2.6r-3b2-v1",
            "cohort_public_artifacts": {
                name: {"path": str(self.cohort / name), "sha256": digest}
                for name, digest in self.cohort_hashes.items()
            },
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
        }
        coverage = {
            "status": "INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_PASS",
            "frozen_case_count": 30,
            "frozen_unit_count": 289,
            "evaluable_case_count": 28,
            "zero_direct_positive_case_count": 2,
            "coverage_rate": 28 / 30,
            "coverage_gate_minimum": 24,
            "coverage_gate_pass": True,
            "per_dataset": per_dataset,
            "case_coverage": case_coverage,
            "resampling_performed": False,
        }
        values = {
            "final_gold_source_lock.json": source_lock,
            "final_relevance_gold.jsonl": self.gold_rows,
            "review_resolution_ledger.jsonl": [
                {
                    "canonical_case_id": row["canonical_case_id"],
                    "unit_id": row["unit_id"],
                    "resolution_source": row["resolution_source"],
                }
                for row in self.gold_rows
            ],
            "coverage_report.json": coverage,
            "adjudication_frozen.csv": b"adjudication_case_id,final_relevance_label\n",
            "adjudication_provenance.json": {"stage": "step2.6r-3b2"},
        }
        hashes = {}
        for name, value in values.items():
            if isinstance(value, bytes):
                payload = value
            elif name.endswith(".jsonl"):
                payload = _jsonl_bytes(value)
            else:
                payload = _json_bytes(value)
            hashes[name] = _write_locked(self.gold / name, payload)
        manifest = {
            "status": "FINAL_RELEVANCE_GOLD_FROZEN",
            "implementation_revision": "step2.6r-3b2-v1",
            "frozen_case_count": 30,
            "frozen_unit_count": 289,
            "final_gold_fields": list(FINAL_GOLD_FIELDS),
            "final_relevance_gold_sha256": hashes["final_relevance_gold.jsonl"],
            "review_resolution_ledger_sha256": hashes["review_resolution_ledger.jsonl"],
            "coverage_report_sha256": hashes["coverage_report.json"],
            "source_lock_sha256": hashes["final_gold_source_lock.json"],
            "adjudication_used": True,
            "adjudication_frozen_csv_sha256": hashes["adjudication_frozen.csv"],
            "adjudication_provenance_sha256": hashes["adjudication_provenance.json"],
            "coverage_gate_pass": True,
            "resampling_performed": False,
        }
        manifest_sha = _write_locked(
            self.gold / "final_gold_manifest.json", _json_bytes(manifest)
        )
        freeze = {
            "status": "FINAL_RELEVANCE_GOLD_FREEZE_PASS",
            "implementation_revision": "step2.6r-3b2-v1",
            "frozen_case_count": 30,
            "frozen_unit_count": 289,
            "evaluable_case_count": 28,
            "coverage_gate_minimum": 24,
            "coverage_gate_pass": True,
            "final_gold_manifest_sha256": manifest_sha,
            "selector_scores_accessed": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "sealed_historical_reference_content_accessed": False,
            "resampling_performed": False,
            "step_3b3_executed": False,
        }
        _write_locked(self.gold / "final_gold_freeze_report.json", _json_bytes(freeze))

    def _write_stage_a(self) -> Path:
        path = self.root / "prediction_invariance_smoke_report.json"
        report = {
            "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
            "implementation_revision": "step2.6r-3-v1",
            "deployment_candidate_seed": 42,
            "deployment_candidate_selector_sha256": self.selector_sha,
            "base_frozen_g1_checkpoint_sha256": self.checkpoint_sha,
            "prediction_invariance_gate": True,
            "selection_scores_changed": True,
            "encoder_hash_unchanged": True,
            "veracity_head_hash_unchanged": True,
            "selection_head_hash_changed": True,
            "state_difference_names": [
                "selection_head.bias",
                "selection_head.weight",
            ],
            "heldout_relevance_cases_accessed": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "veracity_labels_inspected": False,
            "training_started": False,
            "optimizer_created": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
        }
        _write_locked(path, _json_bytes(report))
        return path

    def _write_training(self) -> TrainingArtifacts:
        summary = {
            "seeds": [42, 43, 44],
            "future_deployment_candidate_seed": 42,
        }
        summary_path = self.training / "multi_seed_summary.json"
        summary_path.write_bytes(_json_bytes(summary))
        report = {
            "status": "PASS",
            "run_mode": "full",
            "base_frozen_g1_checkpoint_sha256": self.checkpoint_sha,
            "multi_seed_summary": summary,
        }
        report_path = self.training / "training_report.json"
        report_path.write_bytes(_json_bytes(report))
        (self.training / "training_report.json.sha256").write_text(
            _sha(report_path) + "\n", encoding="utf-8"
        )
        neutral_primary = []
        for name in (
            "neutral_calibration_train.jsonl",
            "neutral_calibration_dev.jsonl",
            "neutral_revision_manifest.json",
        ):
            path = self.neutral / name
            path.write_text("{}\n", encoding="utf-8")
            neutral_primary.append(path)
            path.with_suffix(".sha256").write_text(_sha(path) + "\n", encoding="utf-8")
        neutral_report = self.neutral / "neutral_build_report.json"
        neutral_report.write_bytes(_json_bytes({"status": "PASS"}))
        immutable = {
            report_path: _sha(report_path),
            summary_path: _sha(summary_path),
            self.selector: self.selector_sha,
            **{path: _sha(path) for path in neutral_primary},
        }
        return TrainingArtifacts(
            training_dir=self.training,
            selector_path=self.selector,
            selector_sha256=self.selector_sha,
            training_report=report,
            neutral_source_hashes={path.name: _sha(path) for path in neutral_primary},
            calibration_case_ids=("Synthetic:neutral",),
            immutable_file_hashes=immutable,
        )

    def validator(self, training: Path, neutral: Path) -> TrainingArtifacts:
        if training.resolve() != self.training.resolve() or neutral.resolve() != self.neutral.resolve():
            raise AssertionError("fixture validator received unexpected paths")
        return self.training_artifacts

    def kwargs(self, output: Path) -> dict:
        return {
            "cohort_dir": self.cohort,
            "final_gold_dir": self.gold,
            "stage_a_invariance_report": self.stage_a,
            "project_root": self.project,
            "phase4a_config": self.config,
            "neutral_dir": self.neutral,
            "training_dir": self.training,
            "output_dir": output,
            "training_validator": self.validator,
        }


class FakeRuntime:
    instances: list["FakeRuntime"] = []
    calibrated_same = False
    bad_differences = False
    immutable_failure = False

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.encoder_hash = "encoder-frozen"
        self.veracity_head_hash = "veracity-frozen"
        self.original_selection_head_hash = "selector-original"
        self.calibrated_selection_head_hash = "selector-calibrated"
        self.state_difference_names = (
            ("encoder.weight",)
            if self.bad_differences
            else ("selection_head.bias", "selection_head.weight")
        )
        self.calls = []
        self.immutable_checked = False
        self.instances.append(self)

    def evaluate(self, request, *, state: str) -> PredictionSnapshot:
        self.calls.append((request.request_id, state))
        size = len(request.candidate_units)
        original = tuple(float(index) for index in range(size))
        calibrated = original if self.calibrated_same else tuple(
            float(size - index) for index in range(size)
        )
        scores = original if state == "original" else calibrated
        order = sorted(range(size), key=lambda index: (-scores[index], index))
        return PredictionSnapshot(
            candidate_unit_ids=request.candidate_unit_ids,
            selection_scores=scores,
            unit_veracity_logits=tuple((0.0, 0.0) for _ in range(size)),
            sample_logits=(0.0, 0.0),
            probabilities=(0.5, 0.5),
            prediction="fake",
            top_k_unit_ids=tuple(request.candidate_unit_ids[index] for index in order[:5]),
        )

    def assert_immutable(self) -> None:
        self.immutable_checked = True
        if self.immutable_failure:
            raise RuntimeError("synthetic immutability failure")


class IndependentEvalFixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Synthetic3B3Fixture(Path(self.temporary.name))
        patches = (
            mock.patch(
                "scripts.selector_relevance_independent_eval.source_loader.CALIBRATED_SELECTOR_SHA256",
                self.fixture.selector_sha,
            ),
            mock.patch(
                "scripts.selector_relevance_independent_eval.source_loader.BASE_G1_SHA256",
                self.fixture.checkpoint_sha,
            ),
            mock.patch(
                "scripts.selector_relevance_independent_eval.evaluator.CALIBRATED_SELECTOR_SHA256",
                self.fixture.selector_sha,
            ),
            mock.patch(
                "scripts.selector_relevance_independent_eval.evaluator.BASE_G1_SHA256",
                self.fixture.checkpoint_sha,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        FakeRuntime.instances.clear()
        FakeRuntime.calibrated_same = False
        FakeRuntime.bad_differences = False
        FakeRuntime.immutable_failure = False

    def preflight(self) -> Path:
        output = self.fixture.root / "04_one_shot_evaluation_preflight"
        run_preflight(**self.fixture.kwargs(output))
        return output

    def evaluate(self, *, output_name="05_one_shot_selector_evaluation"):
        preflight = self.preflight()
        output = self.fixture.root / output_name
        report = run_one_shot_evaluation(
            **self.fixture.kwargs(output),
            approved_preflight_report=preflight / "one_shot_preflight_report.json",
            device="cpu",
            runtime_factory=FakeRuntime,
        )
        return output, report


class PreflightAndSourceTests(IndependentEvalFixtureTestCase):
    def _prepared(self, *, project_root: Path | None = None):
        kwargs = self.fixture.kwargs(self.fixture.root / "unused-output")
        kwargs.pop("output_dir")
        if project_root is not None:
            kwargs["project_root"] = project_root
        return prepare_inputs(**kwargs)

    def test_unirumor_project_root_authoritative_layout_is_accepted_and_locked(self):
        prepared = self._prepared()
        self.assertEqual(prepared.project_root, self.fixture.project.resolve())
        locks = prepared.source_lock["authoritative_runtime_sources"]
        self.assertEqual({"phase3_model", "phase4a_engine"}, set(locks))

    def test_defense_engineering_as_project_root_is_rejected(self):
        with self.assertRaisesRegex(
            IndependentEvaluationError, "authoritative UniRumor runtime layout"
        ):
            self._prepared(project_root=self.fixture.defense)

    def test_relative_checkpoint_resolves_against_unirumor_root(self):
        prepared = self._prepared()
        self.assertEqual(prepared.checkpoint_path, self.fixture.checkpoint.resolve())

    def test_phase4a_config_must_contain_the_frozen_g1_contract(self):
        config = self.fixture.root / "phase4a_without_contract.json"
        config.write_bytes(
            _json_bytes({"checkpoint_path": "artifacts/frozen_g1.ckpt"})
        )
        with self.assertRaisesRegex(
            IndependentEvaluationError, "authoritative G1 contract"
        ):
            _validate_phase4a_config(config, self.fixture.project.resolve())

    def test_phase4a_checkpoint_cannot_resolve_into_formal_test(self):
        protected = self.fixture.project / "FormalTest"
        protected.mkdir()
        (protected / "frozen_g1.ckpt").write_bytes(self.fixture.checkpoint.read_bytes())
        config = self.fixture.root / "phase4a_protected_checkpoint.json"
        payload = json.loads(self.fixture.config.read_text())
        payload["checkpoint_path"] = "FormalTest/frozen_g1.ckpt"
        config.write_bytes(_json_bytes(payload))
        with self.assertRaises(IndependentEvaluationError):
            _validate_phase4a_config(config, self.fixture.project.resolve())

    def test_valid_preflight_is_atomic_score_free_and_exact(self):
        output = self.preflight()
        self.assertFalse(FakeRuntime.instances)
        expected = {
            name
            for artifact in (
                "evaluation_preflight_source_lock.json",
                "evaluation_case_manifest.json",
                "preregistration_lock.json",
                "selector_artifact_lock.json",
                "one_shot_preflight_report.json",
            )
            for name in (artifact, str(Path(artifact).with_suffix(".sha256")))
        }
        self.assertEqual(expected, {path.name for path in output.iterdir()})
        report = json.loads((output / "one_shot_preflight_report.json").read_text())
        self.assertEqual(report["status"], "INDEPENDENT_SELECTOR_ONE_SHOT_PREFLIGHT_PASS")
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["checkpoint_loaded_for_execution"])
        self.assertFalse(report["selector_scoring_performed"])

    def test_source_sha_revision_preregistration_and_pointer_fail_closed(self):
        mutations = {
            "wrong-sha": lambda: (self.fixture.cohort / "build_report.json").write_text("{}"),
            "wrong-revision": lambda: self._mutate_json(
                self.fixture.cohort / "build_report.json", "implementation_revision", "wrong"
            ),
            "altered-preregistration": lambda: self._mutate_json(
                self.fixture.cohort / "independent_audit_preregistration.json",
                "deployment_candidate_seed",
                43,
            ),
            "pointer-mismatch": lambda: self._mutate_json(
                self.fixture.cohort / "build_report.json", "preregistration_sha256", "0" * 64
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                root = Path(self.temporary.name) / f"repeat-{name}"
                fixture = Synthetic3B3Fixture(root)
                self.fixture = fixture
                mutation = {
                    "wrong-sha": lambda: (fixture.cohort / "build_report.json").write_text("{}"),
                    "wrong-revision": lambda: self._mutate_json(
                        fixture.cohort / "build_report.json", "implementation_revision", "wrong"
                    ),
                    "altered-preregistration": lambda: self._mutate_json(
                        fixture.cohort / "independent_audit_preregistration.json",
                        "deployment_candidate_seed",
                        43,
                    ),
                    "pointer-mismatch": lambda: self._mutate_json(
                        fixture.cohort / "build_report.json", "preregistration_sha256", "0" * 64
                    ),
                }[name]
                mutation()
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(**{key: value for key, value in fixture.kwargs(root / "out").items() if key != "output_dir"})

    @staticmethod
    def _mutate_json(path: Path, field: str, value: object) -> None:
        payload = json.loads(path.read_text())
        payload[field] = value
        _write_locked(path, _json_bytes(payload))

    def test_historical_and_formal_paths_are_rejected_before_open(self):
        base = {key: value for key, value in self.fixture.kwargs(self.fixture.root / "x").items() if key != "output_dir"}
        for field, path in (
            ("cohort_dir", self.fixture.root / "heldout" / "old-six.json"),
            ("cohort_dir", self.fixture.root / "CPAC" / "old"),
            ("final_gold_dir", self.fixture.root / "FormalValidation" / "gold"),
            ("training_dir", self.fixture.root / "FormalTest" / "training"),
        ):
            with self.subTest(field=field, path=path):
                values = dict(base)
                values[field] = path
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(**values)

    def test_wrong_stage_a_selector_or_base_sha_rejected(self):
        for field in (
            "deployment_candidate_selector_sha256",
            "base_frozen_g1_checkpoint_sha256",
        ):
            with self.subTest(field=field):
                payload = json.loads(self.fixture.stage_a.read_text())
                payload[field] = "0" * 64
                _write_locked(self.fixture.stage_a, _json_bytes(payload))
                with self.assertRaises(IndependentEvaluationError):
                    run_preflight(**self.fixture.kwargs(self.fixture.root / "preflight"))
                self.fixture.stage_a = self.fixture._write_stage_a()

    def test_selector_checkpoint_and_phase4a_hash_locks_fail_closed(self):
        mutations = (
            ("selector", lambda fixture: fixture.selector.write_bytes(b"changed")),
            ("checkpoint", lambda fixture: fixture.checkpoint.write_bytes(b"changed")),
            ("phase4a", lambda fixture: fixture.config.write_bytes(b"{}\n")),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                root = self.fixture.root / f"lock-{name}"
                fixture = Synthetic3B3Fixture(root)
                mutation(fixture)
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(
                        **{
                            key: value
                            for key, value in fixture.kwargs(root / "out").items()
                            if key != "output_dir"
                        }
                    )

    def test_invalid_preflight_does_not_create_authoritative_output(self):
        output = self.fixture.root / "invalid-preflight"
        (self.fixture.cohort / "build_report.json").write_text("{}")
        with self.assertRaises(IndependentEvaluationError):
            run_preflight(**self.fixture.kwargs(output))
        self.assertFalse(output.exists())

    def test_existing_preflight_output_is_never_overwritten(self):
        output = self.fixture.root / "existing-preflight"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(IndependentEvaluationError):
            run_preflight(**self.fixture.kwargs(output))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class GoldAndJoinTests(IndependentEvalFixtureTestCase):
    def test_valid_gold_contract_and_coverage(self):
        prepared = prepare_inputs(
            **{key: value for key, value in self.fixture.kwargs(self.fixture.root / "x").items() if key != "output_dir"}
        )
        self.assertEqual(len(prepared.cases), 30)
        self.assertEqual(sum(len(case.candidate_units) for case in prepared.cases), 289)
        self.assertEqual(sum(case.evaluable for case in prepared.cases), 28)
        self.assertEqual(sum(not case.evaluable for case in prepared.cases), 2)
        self.assertEqual(sum(case.evaluable and case.dataset == "GroundLie360" for case in prepared.cases), 13)
        self.assertEqual(sum(case.evaluable and case.dataset == "TRUE-3MFact" for case in prepared.cases), 15)
        self.assertEqual(len(prepared.cases[0].positive_unit_ids), 2)

    def test_gold_sha_revision_status_counts_mapping_resolution_and_duplicates_rejected(self):
        cases = (
            ("wrong-sha", "final_relevance_gold.jsonl", lambda rows: rows),
            ("wrong-revision", "final_gold_freeze_report.json", None),
            ("wrong-status", "final_gold_freeze_report.json", None),
            ("missing-unit", "final_relevance_gold.jsonl", lambda rows: rows[:-1]),
            ("duplicate-unit", "final_relevance_gold.jsonl", lambda rows: rows[:-1] + [dict(rows[0])]),
            ("bad-mapping", "final_relevance_gold.jsonl", lambda rows: [{**rows[0], "binary_direct_relevance_target": 0}] + rows[1:]),
            ("bad-resolution", "final_relevance_gold.jsonl", lambda rows: [{**rows[0], "resolution_source": "MAJORITY"}] + rows[1:]),
        )
        for name, artifact, mutation in cases:
            with self.subTest(name=name):
                root = self.fixture.root / f"gold-{name}"
                fixture = Synthetic3B3Fixture(root)
                path = fixture.gold / artifact
                if name == "wrong-sha":
                    path.write_bytes(path.read_bytes() + b" ")
                elif artifact.endswith(".jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    _write_locked(path, _jsonl_bytes(mutation(rows)))
                else:
                    payload = json.loads(path.read_text())
                    payload["implementation_revision" if name == "wrong-revision" else "status"] = "wrong"
                    _write_locked(path, _json_bytes(payload))
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(**{key: value for key, value in fixture.kwargs(root / "out").items() if key != "output_dir"})

    def test_request_join_mutations_are_rejected_by_frozen_hash_chain(self):
        mutation_names = (
            "claim",
            "candidate-text",
            "unit-type",
            "modality",
            "position",
            "reorder",
            "addition",
            "deletion",
        )
        for name in mutation_names:
            with self.subTest(name=name):
                root = self.fixture.root / f"join-{name}"
                fixture = Synthetic3B3Fixture(root)
                path = fixture.cohort / "independent_relevance_audit_requests.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                candidates = rows[0]["candidate_units"]
                if name == "claim":
                    rows[0]["claim"] = "mutated claim"
                elif name == "candidate-text":
                    candidates[0]["text"] = "mutated text"
                elif name == "unit-type":
                    candidates[0]["unit_type"] = "ocr"
                elif name == "modality":
                    candidates[0]["modality"] = "ocr"
                elif name == "position":
                    candidates[0]["original_candidate_position"] = 1
                elif name == "reorder":
                    candidates[0], candidates[1] = candidates[1], candidates[0]
                elif name == "addition":
                    candidates.append(dict(candidates[-1], unit_id="added"))
                else:
                    candidates.pop()
                request_sha = _write_locked(path, _jsonl_bytes(rows))
                build_path = fixture.cohort / "build_report.json"
                build = json.loads(build_path.read_text())
                build["independent_relevance_audit_requests_sha256"] = request_sha
                _write_locked(build_path, _json_bytes(build))
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(**{key: value for key, value in fixture.kwargs(root / "out").items() if key != "output_dir"})

    def test_coverage_exact_frozen_counts_fail_closed_when_changed(self):
        coverage = json.loads((self.fixture.gold / "coverage_report.json").read_text())
        mutations = (
            ("case-count", "frozen_case_count", 29),
            ("unit-count", "frozen_unit_count", 288),
            ("evaluable-count", "evaluable_case_count", 27),
            ("zero-direct-count", "zero_direct_positive_case_count", 3),
            ("coverage-gate", "coverage_gate_pass", False),
            ("resampling", "resampling_performed", True),
        )
        for name, field, value in mutations:
            with self.subTest(name=name):
                changed = dict(coverage)
                changed[field] = value
                with self.assertRaises(IndependentEvaluationError):
                    _validate_coverage(changed, self.fixture.gold_rows)
        changed = json.loads(json.dumps(coverage))
        changed["per_dataset"]["GroundLie360"]["evaluable_case_count"] = 14
        with self.assertRaises(IndependentEvaluationError):
            _validate_coverage(changed, self.fixture.gold_rows)

    def test_final_gold_forbidden_extra_fields_rejected(self):
        for forbidden in ("selection_score", "prediction", "label"):
            with self.subTest(forbidden=forbidden):
                root = self.fixture.root / f"gold-field-{forbidden}"
                fixture = Synthetic3B3Fixture(root)
                path = fixture.gold / "final_relevance_gold.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0][forbidden] = 0
                _write_locked(path, _jsonl_bytes(rows))
                with self.assertRaises(IndependentEvaluationError):
                    prepare_inputs(
                        **{
                            key: value
                            for key, value in fixture.kwargs(root / "out").items()
                            if key != "output_dir"
                        }
                    )


class RankingMetricAndGateTests(unittest.TestCase):
    @staticmethod
    def case(*, positives=("z",), dataset="GroundLie360") -> IndependentCase:
        units = (
            EvaluationUnit("z", "transcript", "text", "first"),
            EvaluationUnit("a", "transcript", "text", "second"),
            EvaluationUnit("m", "ocr", "ocr", "third"),
            EvaluationUnit("b", "transcript", "text", "fourth"),
            EvaluationUnit("y", "ocr", "ocr", "fifth"),
            EvaluationUnit("c", "transcript", "text", "sixth"),
        )
        return IndependentCase("audit", dataset, f"{dataset}:case", "claim", units, positives)

    def test_ties_preserve_original_order_not_unit_id_dataset_or_modality(self):
        case = self.case()
        _, row = evaluate_case_scores(
            case, original_scores=[1] * 6, calibrated_scores=[1] * 6
        )
        self.assertEqual(row["original_ranked_unit_ids"], ["z", "a", "m", "b", "y", "c"])

    def test_nonfinite_scores_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(IndependentEvaluationError):
                evaluate_case_scores(
                    self.case(),
                    original_scores=[value, 1, 1, 1, 1, 1],
                    calibrated_scores=[1] * 6,
                )

    def test_zero_direct_retained_null_and_multiple_direct_all_positive(self):
        zero = self.case(positives=())
        _, row = evaluate_case_scores(
            zero, original_scores=[6, 5, 4, 3, 2, 1], calibrated_scores=[1] * 6
        )
        self.assertFalse(row["evaluable"])
        for field in (
            "original_mrr",
            "original_ndcg_at_5",
            "original_recall_at_1",
            "original_recall_at_3",
            "original_recall_at_5",
        ):
            self.assertIsNone(row[field])
        multi = self.case(positives=("z", "c"))
        _, row = evaluate_case_scores(
            multi,
            original_scores=[1, 6, 5, 4, 3, 2],
            calibrated_scores=[1] * 6,
        )
        self.assertEqual(row["direct_positive_count"], 2)
        self.assertAlmostEqual(row["original_mrr"], 1 / 5)
        expected_dcg = (1 / math.log2(6)) / (
            1 / math.log2(2) + 1 / math.log2(3)
        )
        self.assertAlmostEqual(row["original_ndcg_at_5"], expected_dcg)
        self.assertEqual(row["original_recall_at_1"], 0.0)
        self.assertEqual(row["original_recall_at_3"], 0.0)
        self.assertEqual(row["original_recall_at_5"], 1.0)

    @staticmethod
    def gate_metrics(
        *,
        original_mrr=0.20,
        calibrated_mrr=0.26,
        original_ndcg=0.20,
        calibrated_ndcg=0.26,
        original_r5=0.80,
        calibrated_r5=0.80,
        gl_original=0.30,
        gl_calibrated=0.30,
        t_original=0.30,
        t_calibrated=0.30,
    ):
        def group(original_mrr, calibrated_mrr, original_ndcg=0.2, calibrated_ndcg=0.2):
            return {
                "original": {
                    "mrr": original_mrr,
                    "ndcg_at_5": original_ndcg,
                    "recall_at_5": original_r5,
                    "recall_at_1": 0.9,
                    "recall_at_3": 0.9,
                },
                "calibrated": {
                    "mrr": calibrated_mrr,
                    "ndcg_at_5": calibrated_ndcg,
                    "recall_at_5": calibrated_r5,
                    "recall_at_1": 0.1,
                    "recall_at_3": 0.1,
                },
            }
        return {
            "groups": {
                "overall": group(original_mrr, calibrated_mrr, original_ndcg, calibrated_ndcg),
                "GroundLie360": group(gl_original, gl_calibrated),
                "TRUE-3MFact": group(t_original, t_calibrated),
            }
        }

    def test_gate_pass_and_recall_1_3_regressions_are_descriptive(self):
        gate = calculate_repair_gate(
            self.gate_metrics(),
            evaluable_case_count=24,
            seed=42,
            architecture_condition_pass=True,
        )
        self.assertTrue(gate["all_preregistered_conditions_pass"])
        self.assertTrue(gate["recall_at_1_is_descriptive_only"])
        self.assertTrue(gate["recall_at_3_is_descriptive_only"])
        self.assertFalse(gate["cpac_gate_present"])
        self.assertTrue(gate["case_level_regression_count_is_descriptive_only"])
        self.assertFalse(gate["positive_original_top5_preservation_gate_present"])
        self.assertFalse(gate["modality_specific_gate_present"])

    def test_coverage_seed_strict_improvement_and_effect_size_conditions(self):
        cases = (
            (23, 42, self.gate_metrics()),
            (24, 43, self.gate_metrics()),
            (24, 42, self.gate_metrics(calibrated_mrr=0.20)),
            (24, 42, self.gate_metrics(calibrated_mrr=0.19)),
            (24, 42, self.gate_metrics(calibrated_ndcg=0.20)),
            (24, 42, self.gate_metrics(calibrated_ndcg=0.19)),
            (
                24,
                42,
                self.gate_metrics(calibrated_mrr=0.249, calibrated_ndcg=0.249),
            ),
        )
        for evaluable, seed, metrics in cases:
            with self.subTest(evaluable=evaluable, seed=seed, metrics=metrics):
                gate = calculate_repair_gate(
                    metrics,
                    evaluable_case_count=evaluable,
                    seed=seed,
                    architecture_condition_pass=True,
                )
                self.assertFalse(gate["all_preregistered_conditions_pass"])

    def test_architecture_condition_is_required(self):
        gate = calculate_repair_gate(
            self.gate_metrics(),
            evaluable_case_count=24,
            seed=42,
            architecture_condition_pass=False,
        )
        self.assertFalse(gate["architecture_condition_pass"])
        self.assertFalse(gate["all_preregistered_conditions_pass"])

    def test_no_improvement_count_or_rank_regression_gate_is_applied(self):
        metrics = self.gate_metrics()
        metrics["descriptive_case_counts"] = {
            "best_direct_rank_regression_count": 28,
            "best_direct_rank_improvement_count": 0,
            "used_as_acceptance_gate": False,
        }
        gate = calculate_repair_gate(
            metrics,
            evaluable_case_count=24,
            seed=42,
            architecture_condition_pass=True,
        )
        self.assertTrue(gate["all_preregistered_conditions_pass"])

    def test_inclusive_frozen_boundaries_pass_and_just_below_fail(self):
        passing = (
            self.gate_metrics(calibrated_mrr=0.25, calibrated_ndcg=0.21),
            self.gate_metrics(calibrated_mrr=0.21, calibrated_ndcg=0.25),
            self.gate_metrics(original_r5=0.80, calibrated_r5=0.78),
            self.gate_metrics(gl_original=0.30, gl_calibrated=0.25),
            self.gate_metrics(t_original=0.30, t_calibrated=0.25),
        )
        for metrics in passing:
            with self.subTest(metrics=metrics):
                self.assertTrue(
                    calculate_repair_gate(
                        metrics,
                        evaluable_case_count=24,
                        seed=42,
                        architecture_condition_pass=True,
                    )["all_preregistered_conditions_pass"]
                )
        failing = (
            self.gate_metrics(original_r5=0.80, calibrated_r5=0.779),
            self.gate_metrics(gl_original=0.30, gl_calibrated=0.249),
            self.gate_metrics(t_original=0.30, t_calibrated=0.249),
        )
        for metrics in failing:
            with self.subTest(metrics=metrics):
                self.assertFalse(
                    calculate_repair_gate(
                        metrics,
                        evaluable_case_count=24,
                        seed=42,
                        architecture_condition_pass=True,
                    )["all_preregistered_conditions_pass"]
                )


class OneShotOutputAndBoundaryTests(IndependentEvalFixtureTestCase):
    def test_cuda_one_shot_inherits_existing_cublas_workspace_guard(self):
        preflight = self.preflight()
        output = self.fixture.root / "cuda-without-determinism-contract"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaises(IndependentEvaluationError) as caught:
                run_one_shot_evaluation(
                    **self.fixture.kwargs(output),
                    approved_preflight_report=(
                        preflight / "one_shot_preflight_report.json"
                    ),
                    device="cuda:0",
                    runtime_factory=DICCEvaluationRuntime,
                )
        self.assertIsInstance(caught.exception.__cause__, RuntimeIntegrationError)
        self.assertIn(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            str(caught.exception.__cause__),
        )
        self.assertFalse(output.exists())

    def test_valid_pass_freezes_exact_outputs_and_runtime_is_immutable(self):
        output, report = self.evaluate()
        self.assertEqual(report["status"], "INDEPENDENT_SELECTOR_REPAIR_VERIFICATION_PASS")
        expected = {
            name
            for artifact in (
                "evaluation_source_lock.json",
                "selector_state_lock.json",
                "ranking_scores.jsonl",
                "per_case_ranking_metrics.jsonl",
                "selector_metrics.json",
                "repair_verification_gate_report.json",
                "one_shot_evaluation_report.json",
            )
            for name in (artifact, str(Path(artifact).with_suffix(".sha256")))
        }
        self.assertEqual(expected, {path.name for path in output.iterdir()})
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(
            FakeRuntime.instances[0].kwargs["project_root"],
            self.fixture.project.resolve(),
        )
        self.assertEqual(len(FakeRuntime.instances[0].calls), 60)
        self.assertTrue(FakeRuntime.instances[0].immutable_checked)
        rows = [json.loads(line) for line in (output / "per_case_ranking_metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(row["evaluable"] for row in rows), 28)
        zero = [row for row in rows if not row["evaluable"]]
        self.assertEqual(len(zero), 2)
        self.assertTrue(all(row["original_mrr"] is None for row in zero))
        metrics = json.loads((output / "selector_metrics.json").read_text())
        self.assertEqual(metrics["groups"]["overall"]["evaluable_case_count"], 28)
        self.assertEqual(metrics["groups"]["GroundLie360"]["evaluable_case_count"], 13)
        self.assertEqual(metrics["groups"]["TRUE-3MFact"]["evaluable_case_count"], 15)
        score_rows = [
            json.loads(line)
            for line in (output / "ranking_scores.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(score_rows), 30)
        self.assertTrue(
            all(
                set(row)
                == {
                    "dataset",
                    "canonical_case_id",
                    "candidate_unit_ids_in_original_order",
                    "original_selection_scores",
                    "calibrated_selection_scores",
                }
                for row in score_rows
            )
        )
        output_bytes = b"".join(path.read_bytes() for path in output.iterdir())
        for forbidden in (b"veracity_logits", b"reviewer_a_label", b"review_confidence"):
            self.assertNotIn(forbidden, output_bytes)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["optimizer_created"])
        self.assertFalse(report["seed_43_or_44_scored"])
        self.assertFalse(report["production_switch_performed"])
        self.assertFalse(report["public_demo_changed"])

    def test_scientifically_valid_fail_is_frozen(self):
        FakeRuntime.calibrated_same = True
        output, report = self.evaluate(output_name="valid-fail")
        self.assertEqual(report["status"], "INDEPENDENT_SELECTOR_REPAIR_VERIFICATION_FAIL")
        self.assertFalse(report["repair_verification_pass"])
        self.assertTrue(report["deployment_remains_blocked"])
        self.assertTrue(output.is_dir())

    def test_existing_output_rejected_before_runtime_creation(self):
        preflight = self.preflight()
        output = self.fixture.root / "existing"
        output.mkdir()
        with self.assertRaises(IndependentEvaluationError):
            run_one_shot_evaluation(
                **self.fixture.kwargs(output),
                approved_preflight_report=preflight / "one_shot_preflight_report.json",
                device="cpu",
                runtime_factory=FakeRuntime,
            )
        self.assertFalse(FakeRuntime.instances)

    def test_invalid_architecture_or_immutability_does_not_freeze_result(self):
        preflight = self.preflight()
        for field in ("bad_differences", "immutable_failure"):
            with self.subTest(field=field):
                FakeRuntime.instances.clear()
                setattr(FakeRuntime, field, True)
                output = self.fixture.root / f"invalid-{field}"
                with self.assertRaises(IndependentEvaluationError):
                    run_one_shot_evaluation(
                        **self.fixture.kwargs(output),
                        approved_preflight_report=preflight / "one_shot_preflight_report.json",
                        device="cpu",
                        runtime_factory=FakeRuntime,
                    )
                self.assertFalse(output.exists())
                setattr(FakeRuntime, field, False)

    def test_preflight_tamper_rejected_before_runtime(self):
        preflight = self.preflight()
        report = preflight / "one_shot_preflight_report.json"
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaises(IndependentEvaluationError):
            run_one_shot_evaluation(
                **self.fixture.kwargs(self.fixture.root / "tampered-out"),
                approved_preflight_report=report,
                device="cpu",
                runtime_factory=FakeRuntime,
            )
        self.assertFalse(FakeRuntime.instances)

    def test_source_change_after_approved_preflight_rejected_before_runtime(self):
        preflight = self.preflight()
        self.fixture.stage_a.write_bytes(self.fixture.stage_a.read_bytes() + b" ")
        with self.assertRaises(IndependentEvaluationError):
            run_one_shot_evaluation(
                **self.fixture.kwargs(self.fixture.root / "changed-source-out"),
                approved_preflight_report=preflight / "one_shot_preflight_report.json",
                device="cpu",
                runtime_factory=FakeRuntime,
            )
        self.assertFalse(FakeRuntime.instances)

    def test_preflight_mode_never_constructs_imports_torch_or_scores(self):
        real_import = builtins.__import__

        def reject_torch_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("preflight must not import Torch")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(
            DICCEvaluationRuntime,
            "__init__",
            side_effect=AssertionError("runtime must not be constructed"),
        ), mock.patch(
            "scripts.selector_relevance_gate.runtime._safe_torch_load",
            side_effect=AssertionError("selector tensors must not be loaded"),
        ), mock.patch(
            "scripts.selector_relevance_independent_eval.evaluator.evaluate_case_scores",
            side_effect=AssertionError("selectors must not be scored"),
        ), mock.patch("builtins.__import__", side_effect=reject_torch_import):
            output = self.fixture.root / "score-free-preflight"
            report = run_preflight(**self.fixture.kwargs(output))
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["selector_scoring_performed"])

    def test_no_historical_gate_import_or_forbidden_cli_options(self):
        package = Path(__file__).parents[1] / "scripts" / "selector_relevance_independent_eval"
        source = "\n".join(path.read_text() for path in package.glob("*.py"))
        self.assertNotIn("selector_relevance_gate.heldout_loader", source)
        self.assertNotIn("run_heldout_gate(", source)
        self.assertNotIn("load_heldout_references(", source)
        options = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--seed",
            "--threshold",
            "--overwrite",
            "--retry",
            "--model-select",
            "--use-validation",
            "--use-test",
        ):
            self.assertNotIn(forbidden, options)

    def test_cli_modes_and_exit_codes_are_exact(self):
        common = [
            "--project-root",
            str(self.fixture.project),
            "--cohort-dir",
            str(self.fixture.cohort),
            "--final-gold-dir",
            str(self.fixture.gold),
            "--stage-a-invariance-report",
            str(self.fixture.stage_a),
            "--phase4a-config",
            str(self.fixture.config),
            "--neutral-dir",
            str(self.fixture.neutral),
            "--training-dir",
            str(self.fixture.training),
            "--output-dir",
            str(self.fixture.root / "cli-output"),
        ]
        with mock.patch(
            "scripts.selector_relevance_independent_eval.run_evaluation.run_preflight",
            return_value={"status": "INDEPENDENT_SELECTOR_ONE_SHOT_PREFLIGHT_PASS"},
        ):
            self.assertEqual(main(["--preflight", *common]), 0)
        with mock.patch(
            "scripts.selector_relevance_independent_eval.run_evaluation.run_one_shot_evaluation",
            return_value={"repair_verification_pass": True},
        ):
            self.assertEqual(
                main(
                    [
                        "--one-shot-evaluate",
                        *common,
                        "--approved-preflight-report",
                        str(self.fixture.root / "approved.json"),
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
        with mock.patch(
            "scripts.selector_relevance_independent_eval.run_evaluation.run_one_shot_evaluation",
            return_value={"repair_verification_pass": False},
        ):
            self.assertEqual(
                main(
                    [
                        "--one-shot-evaluate",
                        *common,
                        "--approved-preflight-report",
                        str(self.fixture.root / "approved.json"),
                        "--device",
                        "cpu",
                    ]
                ),
                1,
            )
        self.assertEqual(main(["--one-shot-evaluate", *common]), 2)

    def test_cli_modes_are_mutually_exclusive(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--preflight", "--one-shot-evaluate"])


if __name__ == "__main__":
    unittest.main()
