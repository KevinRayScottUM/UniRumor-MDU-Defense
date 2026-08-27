import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.selector_relevance_gate.evaluator import (
    EvaluationError,
    compare_prediction_pair,
    run_heldout_gate,
    run_invariance_smoke,
    summarize_invariance,
    verify_approved_invariance_report,
)
from scripts.selector_relevance_gate.heldout_loader import (
    EXPECTED_HELDOUT_CASE_IDS,
    ReferenceInputError,
    calibration_overlap_count,
    load_heldout_references,
    load_phase4a_replay_requests,
    sha256_file,
)
from scripts.selector_relevance_gate.metrics import (
    HeldoutRanking,
    grouped_metrics,
    ranked_unit_ids,
    reference_metrics,
)
from scripts.selector_relevance_gate.runtime import (
    EXPECTED_SELECTOR_SHA256,
    RuntimeIntegrationError,
    TrainingArtifacts,
    _safe_torch_load,
    _validate_state_difference,
    _validated_selector_state,
    validate_training_artifacts,
)
from scripts.selector_relevance_gate.schemas import (
    EvaluationRequest,
    EvaluationUnit,
    PredictionSnapshot,
)
from scripts.selector_relevance_training.metrics import METRIC_NAMES
from scripts.selector_relevance_training.trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    IMPLEMENTATION_REVISION as TRAINING_REVISION,
    SELECTOR_ID,
)


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _unit(prefix: str, index: int) -> EvaluationUnit:
    return EvaluationUnit(
        unit_id=f"{prefix}-u{index}",
        unit_type="ocr" if index % 2 else "transcript",
        modality="ocr" if index % 2 else "text",
        text=f"audited evidence {prefix} {index}",
    )


def _request(prefix: str, *, case_id: str = "Engineering:case") -> EvaluationRequest:
    return EvaluationRequest(
        request_id=prefix,
        case_id=case_id,
        dataset=case_id.split(":", 1)[0],
        claim=f"Original challenge claim for {prefix}",
        candidate_units=tuple(_unit(prefix, index) for index in range(6)),
    )


def _snapshot(
    unit_ids,
    scores,
    *,
    unit_logit_delta: float = 0.0,
    sample_delta: float = 0.0,
    probability_delta: float = 0.0,
    prediction: str = "real",
) -> PredictionSnapshot:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return PredictionSnapshot(
        candidate_unit_ids=tuple(unit_ids),
        selection_scores=tuple(scores),
        unit_veracity_logits=tuple(
            (0.1 + unit_logit_delta, 0.9) for _ in unit_ids
        ),
        sample_logits=(0.1 + sample_delta, 0.9),
        probabilities=(0.31 + probability_delta, 0.69),
        prediction=prediction,
        top_k_unit_ids=tuple(unit_ids[index] for index in order[:5]),
    )


def _training_artifacts(root: Path) -> TrainingArtifacts:
    selector = root / "selector_head.pt"
    selector.write_bytes(b"controlled-selector-fixture")
    return TrainingArtifacts(
        training_dir=root,
        selector_path=selector,
        selector_sha256=EXPECTED_SELECTOR_SHA256,
        training_report={},
        neutral_source_hashes={
            "neutral_calibration_train.jsonl": "a" * 64,
            "neutral_calibration_dev.jsonl": "b" * 64,
            "neutral_revision_manifest.json": "c" * 64,
        },
        calibration_case_ids=("GroundLie360:other", "TRUE-3MFact:other"),
        immutable_file_hashes={},
    )


class FakeRuntime:
    encoder_hash = "encoder-hash"
    veracity_head_hash = "veracity-hash"
    original_selection_head_hash = "original-selector"
    calibrated_selection_head_hash = "calibrated-selector"
    state_difference_names = ("selection_head.bias", "selection_head.weight")

    def __init__(self, originals, calibrated):
        self.originals = originals
        self.calibrated = calibrated
        self.calls = []
        self.immutable_checked = False

    def evaluate(self, request, *, state):
        self.calls.append((state, request.request_id, request.candidate_unit_ids))
        return (self.originals if state == "original" else self.calibrated)[
            request.request_id
        ]

    def assert_immutable(self):
        self.immutable_checked = True


class RankingMetricTests(unittest.TestCase):
    def _ranking(self, scores, positives=("b",)):
        return HeldoutRanking(
            reference_id="ref",
            case_id="Dataset:case",
            dataset="Dataset",
            reference_modality="OCR",
            candidate_unit_ids=("a", "b", "c", "d", "e", "f"),
            positive_unit_ids=tuple(positives),
            selection_scores=tuple(scores),
        )

    def test_stable_candidate_order_breaks_score_ties(self):
        self.assertEqual(
            ("a", "b", "c", "d", "e", "f"),
            ranked_unit_ids(self._ranking((1, 1, 1, 1, 1, 1))),
        )

    def test_mrr_and_binary_recall_at_1_3_5(self):
        values = reference_metrics(self._ranking((6, 3, 5, 4, 2, 1)))
        self.assertEqual(4, values["best_positive_rank"])
        self.assertEqual(0.0, values["recall_at_1"])
        self.assertEqual(0.0, values["recall_at_3"])
        self.assertEqual(1.0, values["recall_at_5"])
        self.assertEqual(0.25, values["mrr"])

    def test_ndcg_at_5_uses_binary_positive_set(self):
        values = reference_metrics(
            self._ranking((6, 5, 4, 3, 2, 1), positives=("b", "d"))
        )
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (
            1 + 1 / math.log2(3)
        )
        self.assertAlmostEqual(expected, values["ndcg_at_5"])

    def test_multi_positive_uses_best_rank(self):
        values = reference_metrics(
            self._ranking((1, 2, 3, 4, 5, 6), positives=("a", "f"))
        )
        self.assertEqual(1, values["best_positive_rank"])
        self.assertEqual(1.0, values["mrr"])

    def test_metrics_are_macro_grouped_by_case_dataset_and_modality(self):
        one = self._ranking((6, 5, 4, 3, 2, 1))
        two = HeldoutRanking(
            reference_id="ref-2",
            case_id="Other:case",
            dataset="Other",
            reference_modality="TRANSCRIPT",
            candidate_unit_ids=one.candidate_unit_ids,
            positive_unit_ids=one.positive_unit_ids,
            selection_scores=(1, 6, 5, 4, 3, 2),
        )
        metrics = grouped_metrics((one, two))
        self.assertEqual(0.5, metrics["overall"]["recall_at_1"])
        self.assertEqual(2, len(metrics["by_underlying_case"]))


class InputLoaderTests(unittest.TestCase):
    def _phase4a_payload(self):
        return {
            "schema_version": 1,
            "artifact_type": "phase4a_label_free_replay_requests",
            "requests": [
                {
                    "request_id": f"replay-{index}",
                    "case_id": f"Replay:{index}",
                    "dataset": "EngineeringReplay",
                    "claim": f"Existing replay claim {index}",
                    "candidate_units": [
                        dict(_unit(f"replay-{index}", unit_index).to_dict())
                        for unit_index in range(2)
                    ],
                }
                for index in range(8)
            ],
        }

    def _heldout_payload(self, source: Path):
        source_sha = sha256_file(source)
        references = []
        for index, case_id in enumerate(EXPECTED_HELDOUT_CASE_IDS):
            prefix = f"heldout-{index}"
            units = [_unit(prefix, unit_index) for unit_index in range(6)]
            references.append(
                {
                    "reference_id": f"reference-{index}",
                    "case_id": case_id,
                    "dataset": case_id.split(":", 1)[0],
                    "claim": f"Original challenge claim {index}",
                    "candidate_units": [dict(unit.to_dict()) for unit in units],
                    "positive_unit_ids": [units[-1].unit_id],
                    "reference_modality": "OCR",
                    "source_audit_artifact_path": str(source),
                    "source_audit_artifact_sha256": source_sha,
                    "prior_original_best_positive_rank": 6,
                    "prior_original_top5_unit_ids": [unit.unit_id for unit in units[:5]],
                    "prior_candidate_unit_ids": [unit.unit_id for unit in units],
                }
            )
        references[0]["reference_id"] = "ocr_01_direct_full_banner"
        return {
            "schema_version": 1,
            "artifact_type": "preexisting_heldout_relevance_challenge_references",
            "references": references,
        }

    def test_phase4a_loader_requires_exact_hash_and_eight_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.json"
            digest = _write_json(path, self._phase4a_payload())
            actual, requests = load_phase4a_replay_requests(
                path, expected_sha256=digest
            )
            self.assertEqual(digest, actual)
            self.assertEqual(8, len(requests))
            with self.assertRaisesRegex(ReferenceInputError, "SHA-256 mismatch"):
                load_phase4a_replay_requests(path, expected_sha256="0" * 64)

    def test_native_phase4a_jsonl_contract_is_accepted_without_reencoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            rows = []
            for row in self._phase4a_payload()["requests"]:
                rows.append({key: value for key, value in row.items() if key != "request_id"})
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = sha256_file(path)
            _, requests = load_phase4a_replay_requests(
                path, expected_sha256=digest
            )
            self.assertEqual(tuple(row["case_id"] for row in rows), tuple(
                request.request_id for request in requests
            ))

    def test_phase4a_labels_or_model_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.json"
            payload = self._phase4a_payload()
            payload["requests"][0]["label"] = 0
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "forbidden labels"):
                load_phase4a_replay_requests(path, expected_sha256=digest)

    def test_stage_a_heldout_case_access_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.json"
            payload = self._phase4a_payload()
            payload["requests"][0]["case_id"] = EXPECTED_HELDOUT_CASE_IDS[0]
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "held-out"):
                load_phase4a_replay_requests(path, expected_sha256=digest)

    def test_formal_dataset_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dataset in ("Validation", "Test"):
                payload = self._phase4a_payload()
                payload["requests"][0]["dataset"] = dataset
                path = Path(temporary) / f"{dataset}.json"
                digest = _write_json(path, payload)
                with self.assertRaisesRegex(ReferenceInputError, "dataset labels"):
                    load_phase4a_replay_requests(path, expected_sha256=digest)

    def test_formal_validation_and_test_paths_fail_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("Validation", "Test"):
                directory = Path(temporary) / name
                directory.mkdir()
                path = directory / "artifact.json"
                digest = _write_json(path, self._phase4a_payload())
                with self.assertRaisesRegex(ReferenceInputError, "Formal"):
                    load_phase4a_replay_requests(path, expected_sha256=digest)

    def test_heldout_loader_preserves_original_claims_and_exact_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "immutable-audit.json"
            source.write_text("{}\n", encoding="utf-8")
            path = root / "references.json"
            digest = _write_json(path, self._heldout_payload(source))
            _, references = load_heldout_references(path, expected_sha256=digest)
            self.assertEqual(set(EXPECTED_HELDOUT_CASE_IDS), {item.case_id for item in references})
            self.assertTrue(all(item.claim.startswith("Original challenge") for item in references))

    def test_heldout_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["case_id"] = "GroundLie360:wrong"
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "identity mismatch"):
                load_heldout_references(path, expected_sha256=digest)

    def test_neutral_synthetic_claim_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["claim"] = 'The relevant content states "x".'
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "neutral synthetic"):
                load_heldout_references(path, expected_sha256=digest)

    def test_missing_positive_or_positive_outside_pool_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            for positives in ([], ["missing-unit"]):
                payload = self._heldout_payload(source)
                payload["references"][0]["positive_unit_ids"] = positives
                path = root / f"references-{len(positives)}.json"
                digest = _write_json(path, payload)
                with self.assertRaises((ReferenceInputError, ValueError)):
                    load_heldout_references(path, expected_sha256=digest)

    def test_source_audit_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audit.json"
            source.write_text("{}", encoding="utf-8")
            payload = self._heldout_payload(source)
            payload["references"][0]["source_audit_artifact_sha256"] = "0" * 64
            path = root / "references.json"
            digest = _write_json(path, payload)
            with self.assertRaisesRegex(ReferenceInputError, "source_audit_artifact SHA"):
                load_heldout_references(path, expected_sha256=digest)

    def test_calibration_overlap_count_is_exact(self):
        self.assertEqual(
            1,
            calibration_overlap_count(
                EXPECTED_HELDOUT_CASE_IDS,
                ("Other:case", EXPECTED_HELDOUT_CASE_IDS[2]),
            ),
        )


class InvarianceTests(unittest.TestCase):
    def setUp(self):
        self.request = _request("invariance")
        ids = self.request.candidate_unit_ids
        self.original = _snapshot(ids, (6, 5, 4, 3, 2, 1))
        self.calibrated = _snapshot(ids, (1, 2, 3, 4, 5, 6))

    def test_selection_changes_are_allowed(self):
        comparison = compare_prediction_pair(
            self.request, self.original, self.calibrated
        )
        self.assertTrue(comparison["selection_score_changed"])
        self.assertTrue(comparison["prediction_invariant"])

    def test_candidate_id_and_order_mismatches_fail(self):
        ids = list(self.request.candidate_unit_ids)
        reordered = tuple(reversed(ids))
        comparison = compare_prediction_pair(
            self.request,
            self.original,
            _snapshot(reordered, (1, 2, 3, 4, 5, 6)),
        )
        self.assertFalse(comparison["candidate_order_identical"])
        self.assertFalse(comparison["prediction_invariant"])
        changed = tuple(ids[:-1] + ["different"])
        comparison = compare_prediction_pair(
            self.request,
            self.original,
            _snapshot(changed, (1, 2, 3, 4, 5, 6)),
        )
        self.assertFalse(comparison["candidate_ids_identical"])

    def test_each_prediction_delta_above_tolerance_fails(self):
        ids = self.request.candidate_unit_ids
        variants = (
            _snapshot(ids, (1, 2, 3, 4, 5, 6), unit_logit_delta=2e-6),
            _snapshot(ids, (1, 2, 3, 4, 5, 6), sample_delta=2e-6),
            _snapshot(ids, (1, 2, 3, 4, 5, 6), probability_delta=2e-6),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(
                    compare_prediction_pair(
                        self.request, self.original, variant
                    )["prediction_invariant"]
                )

    def test_prediction_mismatch_fails(self):
        changed = _snapshot(
            self.request.candidate_unit_ids,
            (1, 2, 3, 4, 5, 6),
            prediction="fake",
        )
        self.assertFalse(
            compare_prediction_pair(
                self.request, self.original, changed
            )["prediction_invariant"]
        )

    def test_completely_unchanged_selection_fails_stage_a_gate(self):
        comparisons = (
            compare_prediction_pair(self.request, self.original, self.original),
        )
        summary = summarize_invariance(comparisons)
        self.assertTrue(summary["prediction_invariance_gate"])
        self.assertFalse(summary["selection_scores_changed"])

    def test_stage_a_writes_exact_outputs_and_does_not_touch_heldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = tuple(_request(f"replay-{index}") for index in range(8))
            originals = {
                item.request_id: _snapshot(
                    item.candidate_unit_ids, (6, 5, 4, 3, 2, 1)
                )
                for item in requests
            }
            calibrated = {
                item.request_id: _snapshot(
                    item.candidate_unit_ids, (1, 2, 3, 4, 5, 6)
                )
                for item in requests
            }
            runtime = FakeRuntime(originals, calibrated)
            report = run_invariance_smoke(
                requests=requests,
                phase4a_replay_sha256="d" * 64,
                training_artifacts=_training_artifacts(root),
                runtime=runtime,
                output_dir=root / "stage-a",
            )
            self.assertEqual("PREDICTION_INVARIANCE_SMOKE_PASS", report["status"])
            self.assertFalse(report["heldout_relevance_cases_accessed"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["optimizer_created"])
            self.assertTrue(runtime.immutable_checked)
            self.assertTrue(
                (root / "stage-a/prediction_invariance_smoke_report.sha256").is_file()
            )

    def test_stage_a_rejects_heldout_or_wrong_request_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request("heldout", case_id=EXPECTED_HELDOUT_CASE_IDS[0])
            runtime = FakeRuntime({}, {})
            with self.assertRaises(EvaluationError):
                run_invariance_smoke(
                    requests=(request,),
                    phase4a_replay_sha256="d" * 64,
                    training_artifacts=_training_artifacts(root),
                    runtime=runtime,
                    output_dir=root / "out",
                )


class Tensor:
    def __init__(self, shape, *, finite=True):
        self.shape = shape
        self.finite = finite

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self


class FakeTorch:
    @staticmethod
    def isfinite(tensor):
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: tensor.finite))


class ArtifactBoundaryTests(unittest.TestCase):
    def _payload(self, artifacts, state=None):
        return {
            "selection_head_state_dict": state
            or {"weight": Tensor((1, 4)), "bias": Tensor((1,))},
            "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
            "neutral_train_sha256": artifacts.neutral_source_hashes[
                "neutral_calibration_train.jsonl"
            ],
            "neutral_dev_sha256": artifacts.neutral_source_hashes[
                "neutral_calibration_dev.jsonl"
            ],
            "neutral_manifest_sha256": artifacts.neutral_source_hashes[
                "neutral_revision_manifest.json"
            ],
            "seed": 42,
            "selected_epoch": 1,
            "training_protocol": {},
            "optimizer_protocol": {},
            "train_class_counts": {},
            "dev_metrics": {},
            "implementation_revision": TRAINING_REVISION,
        }

    def test_selector_artifact_unexpected_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            payload = self._payload(
                artifacts,
                {
                    "weight": Tensor((1, 4)),
                    "bias": Tensor((1,)),
                    "veracity_head.weight": Tensor((2, 4)),
                },
            )
            with self.assertRaisesRegex(RuntimeIntegrationError, "only weight and bias"):
                _validated_selector_state(
                    payload,
                    original_selector_state={
                        "weight": Tensor((1, 4)),
                        "bias": Tensor((1,)),
                    },
                    torch=FakeTorch(),
                    training_artifacts=artifacts,
                )

    def test_safe_torch_loading_uses_weights_only_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            payload = self._payload(artifacts)

            class Torch:
                called = None

                @staticmethod
                def load(path, map_location="cpu", weights_only=False):
                    Torch.called = (path, map_location, weights_only)
                    return payload

            path = Path(temporary) / "selector.pt"
            path.write_bytes(b"fixture")
            self.assertIs(payload, _safe_torch_load(Torch, path))
            self.assertEqual((path, "cpu", True), Torch.called)

    def test_selector_shape_or_nonfinite_value_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            for state, message in (
                ({"weight": Tensor((2, 4)), "bias": Tensor((1,))}, "shape"),
                (
                    {"weight": Tensor((1, 4), finite=False), "bias": Tensor((1,))},
                    "non-finite",
                ),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(RuntimeIntegrationError, message):
                        _validated_selector_state(
                            self._payload(artifacts, state),
                            original_selector_state={
                                "weight": Tensor((1, 4)),
                                "bias": Tensor((1,)),
                            },
                            torch=FakeTorch(),
                            training_artifacts=artifacts,
                        )

    def test_selector_base_sha_or_seed_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _training_artifacts(Path(temporary))
            for field, value in (
                ("base_frozen_g1_checkpoint_sha256", "0" * 64),
                ("seed", 43),
            ):
                payload = self._payload(artifacts)
                payload[field] = value
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeIntegrationError, field):
                        _validated_selector_state(
                            payload,
                            original_selector_state={
                                "weight": Tensor((1, 4)),
                                "bias": Tensor((1,)),
                            },
                            torch=FakeTorch(),
                            training_artifacts=artifacts,
                        )

    def test_nonselection_state_difference_fails_closed(self):
        original = {
            "encoder.weight": "a",
            "veracity_head.weight": "b",
            "selection_head.weight": "c",
            "selection_head.bias": "d",
        }
        with self.assertRaisesRegex(RuntimeIntegrationError, "non-selection"):
            _validate_state_difference(original, {**original, "encoder.weight": "x"})
        self.assertEqual(
            ("selection_head.weight",),
            _validate_state_difference(
                original, {**original, "selection_head.weight": "x"}
            ),
        )

    def _training_tree(self, root: Path, *, status="PASS", seed=42, base_sha=None):
        training = root / "training"
        (training / "seed_42").mkdir(parents=True)
        selector = training / "seed_42/selector_head.pt"
        selector.write_bytes(b"selector")
        summary = {
            "seeds": [42, 43, 44],
            "metrics": {
                name: {"mean": 1.0, "std": 0.0} for name in METRIC_NAMES
            },
            "future_deployment_candidate_seed": seed,
            "selection_rule": "frozen internal Dev rule",
            "candidate_dev_metrics": {},
        }
        summary_sha = _write_json(training / "multi_seed_summary.json", summary)
        neutral_hashes = {
            "neutral_calibration_train.jsonl": "a" * 64,
            "neutral_calibration_dev.jsonl": "b" * 64,
            "neutral_revision_manifest.json": "c" * 64,
        }
        report = {
            "status": status,
            "run_mode": "full",
            "implementation_revision": TRAINING_REVISION,
            "selector_id": SELECTOR_ID,
            "base_frozen_g1_checkpoint_sha256": base_sha
            or AUTHORITATIVE_CHECKPOINT_SHA256,
            "source_artifact_sha256": neutral_hashes,
            "multi_seed_summary": summary,
            "artifact_sha256": {
                "multi_seed_summary.json": summary_sha,
                "seed_42/selector_head.pt": sha256_file(selector),
            },
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "step25b_heldout_accessed": False,
            "cpac_heldout_accessed": False,
            "veracity_labels_inspected": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
        }
        report_path = training / "training_report.json"
        report_sha = _write_json(report_path, report)
        (training / "training_report.json.sha256").write_text(
            report_sha + "\n", encoding="utf-8"
        )
        neutral = SimpleNamespace(
            source_dir=root / "neutral",
            source_hashes=neutral_hashes,
            train_examples=(SimpleNamespace(canonical_underlying_case_id="Dataset:train"),),
            dev_examples=(SimpleNamespace(canonical_underlying_case_id="Dataset:dev"),),
        )
        return training, selector, neutral

    def test_training_report_status_seed_base_and_selector_sha_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kwargs, expected in (
                ("status", {"status": "FAIL"}, "training report mismatch"),
                ("seed", {"seed": 43}, "seed is not 42"),
                ("base", {"base_sha": "0" * 64}, "training report mismatch"),
            ):
                case_root = root / name
                training, selector, neutral = self._training_tree(case_root, **kwargs)
                with mock.patch(
                    "scripts.selector_relevance_gate.runtime.load_neutral_data",
                    return_value=neutral,
                ), mock.patch(
                    "scripts.selector_relevance_gate.runtime.EXPECTED_SELECTOR_SHA256",
                    sha256_file(selector),
                ):
                    with self.assertRaisesRegex(RuntimeIntegrationError, expected):
                        validate_training_artifacts(training, case_root / "neutral")
            training, selector, neutral = self._training_tree(root / "selector-sha")
            with mock.patch(
                "scripts.selector_relevance_gate.runtime.load_neutral_data",
                return_value=neutral,
            ), mock.patch(
                "scripts.selector_relevance_gate.runtime.EXPECTED_SELECTOR_SHA256",
                "0" * 64,
            ):
                with self.assertRaisesRegex(RuntimeIntegrationError, "selector SHA mismatch"):
                    validate_training_artifacts(training, root / "selector-sha/neutral")


class HeldoutGateTests(unittest.TestCase):
    def _approved(self, root: Path, artifacts: TrainingArtifacts) -> Path:
        path = root / "prediction_invariance_smoke_report.json"
        report = {
            "status": "PREDICTION_INVARIANCE_SMOKE_PASS",
            "implementation_revision": "step2.6r-3-v1",
            "training_implementation_revision": TRAINING_REVISION,
            "selector_id": SELECTOR_ID,
            "deployment_candidate_seed": 42,
            "deployment_candidate_selector_sha256": artifacts.selector_sha256,
            "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
            "prediction_invariance_gate": True,
            "selection_scores_changed": True,
            "heldout_relevance_cases_accessed": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "veracity_labels_inspected": False,
            "training_started": False,
            "optimizer_created": False,
        }
        digest = _write_json(path, report)
        (root / "prediction_invariance_smoke_report.sha256").write_text(
            digest + "\n", encoding="utf-8"
        )
        return path

    def _references_and_runtime(self, root: Path, *, calibrated_scores=None, delta=0.0):
        source = root / "immutable-source-audit.json"
        source.write_text("{}\n", encoding="utf-8")
        source_sha = sha256_file(source)
        references = []
        originals = {}
        calibrated = {}
        for index, case_id in enumerate(EXPECTED_HELDOUT_CASE_IDS):
            prefix = f"heldout-{index}"
            base = _request(prefix, case_id=case_id)
            ids = base.candidate_unit_ids
            original_scores = (6, 5, 4, 3, 2, 1)
            current_scores = (
                calibrated_scores[index]
                if calibrated_scores is not None
                else (1, 2, 3, 4, 5, 10)
            )
            original = _snapshot(ids, original_scores)
            current = _snapshot(ids, current_scores, sample_delta=delta)
            references.append(
                EvaluationRequest(
                    request_id=base.request_id,
                    reference_id=(
                        "ocr_01_direct_full_banner" if index == 0 else base.request_id
                    ),
                    case_id=base.case_id,
                    dataset=base.dataset,
                    claim=base.claim,
                    candidate_units=base.candidate_units,
                    positive_unit_ids=(ids[-1],),
                    reference_modality="OCR",
                    source_audit_artifact_path=str(source),
                    source_audit_artifact_sha256=source_sha,
                    prior_original_best_positive_rank=6,
                    prior_original_top5_unit_ids=original.top_k_unit_ids,
                    prior_candidate_unit_ids=ids,
                )
            )
            originals[prefix] = original
            calibrated[prefix] = current
        return tuple(references), FakeRuntime(originals, calibrated)

    def test_stage_b_requires_approved_stage_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            missing = root / "missing.json"
            with self.assertRaisesRegex(EvaluationError, "missing"):
                verify_approved_invariance_report(missing, artifacts)

    def test_approved_stage_a_must_match_selector_and_frozen_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            path = self._approved(root, artifacts)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["deployment_candidate_selector_sha256"] = "0" * 64
            digest = _write_json(path, payload)
            path.with_suffix(".sha256").write_text(digest + "\n")
            with self.assertRaisesRegex(EvaluationError, "selector_sha256"):
                verify_approved_invariance_report(path, artifacts)

    def test_full_heldout_gate_passes_preregistered_rules_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "stage-b",
            )
            self.assertEqual(
                "HELDOUT_RELEVANCE_AND_INVARIANCE_PASS", report["status"]
            )
            self.assertTrue(report["deployment_eligible"])
            self.assertEqual(6, report["reference_rank_improvement_count"])
            self.assertTrue(report["cpac_top5_after_calibration"])
            expected = {
                "heldout_reference_manifest.json",
                "heldout_reference_manifest.sha256",
                "heldout_case_results.jsonl",
                "heldout_case_results.sha256",
                "original_selector_metrics.json",
                "calibrated_selector_metrics.json",
                "prediction_invariance_report.json",
                "heldout_relevance_gate_report.json",
                "heldout_relevance_gate_report.sha256",
                "dataset_card.md",
            }
            self.assertEqual(expected, {path.name for path in (root / "stage-b").iterdir()})

    def test_rank_regression_or_fewer_than_two_improvements_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            score_sets = [
                (1, 2, 3, 4, 5, 10),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
                (6, 5, 4, 3, 2, 1),
            ]
            references, runtime = self._references_and_runtime(
                root,
                calibrated_scores=score_sets
            )
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "too-few",
            )
            self.assertEqual("HELDOUT_RELEVANCE_FAIL", report["status"])
            self.assertFalse(report["deployment_eligible"])
            self.assertEqual(1, report["reference_rank_improvement_count"])

    def test_cpac_top5_gate_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            score_sets = [(6, 5, 4, 3, 2, 1)] + [
                (1, 2, 3, 4, 5, 10)
            ] * 5
            references, runtime = self._references_and_runtime(
                root,
                calibrated_scores=score_sets
            )
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "cpac-fail",
            )
            self.assertFalse(report["cpac_top5_after_calibration"])
            self.assertFalse(report["deployment_eligible"])

    def test_prediction_invariance_failure_blocks_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root, delta=2e-6)
            report = run_heldout_gate(
                references=references,
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "prediction-fail",
            )
            self.assertEqual("PREDICTION_INVARIANCE_FAIL", report["status"])
            self.assertFalse(report["deployment_eligible"])

    def test_positive_original_top5_cannot_be_pushed_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            first = references[0]
            original = _snapshot(
                first.candidate_unit_ids, (6, 5, 4, 3, 1, 2)
            )
            calibrated = _snapshot(
                first.candidate_unit_ids, (6, 5, 4, 3, 2, 1)
            )
            runtime.originals[first.request_id] = original
            runtime.calibrated[first.request_id] = calibrated
            changed = list(references)
            changed[0] = EvaluationRequest(
                **{
                    **first.__dict__,
                    "prior_original_best_positive_rank": 5,
                    "prior_original_top5_unit_ids": original.top_k_unit_ids,
                }
            )
            report = run_heldout_gate(
                references=tuple(changed),
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "top5-regression",
            )
            rows = [
                json.loads(line)
                for line in (root / "top5-regression/heldout_case_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(rows[0]["positive_original_top5_preserved"])
            self.assertFalse(report["deployment_eligible"])

    def test_baseline_protocol_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            bad = list(references)
            item = bad[0]
            bad[0] = EvaluationRequest(
                **{
                    **item.__dict__,
                    "prior_original_best_positive_rank": 5,
                }
            )
            report = run_heldout_gate(
                references=tuple(bad),
                heldout_reference_sha256="f" * 64,
                approved_invariance_smoke_path=approved,
                training_artifacts=artifacts,
                runtime=runtime,
                output_dir=root / "baseline-fail",
            )
            self.assertEqual("BASELINE_REPLAY_PROTOCOL_MISMATCH", report["status"])
            self.assertFalse(report["deployment_eligible"])

    def test_calibration_overlap_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = _training_artifacts(root)
            artifacts = TrainingArtifacts(
                **{
                    **artifacts.__dict__,
                    "calibration_case_ids": (EXPECTED_HELDOUT_CASE_IDS[0],),
                }
            )
            approved = self._approved(root, artifacts)
            references, runtime = self._references_and_runtime(root)
            with self.assertRaisesRegex(EvaluationError, "overlap"):
                run_heldout_gate(
                    references=references,
                    heldout_reference_sha256="f" * 64,
                    approved_invariance_smoke_path=approved,
                    training_artifacts=artifacts,
                    runtime=runtime,
                    output_dir=root / "overlap",
                )
            self.assertEqual([], runtime.calls)


class StaticBoundaryTests(unittest.TestCase):
    def test_evaluation_request_dummy_label_is_structural_only(self):
        request = _request("dummy")
        item = request.collator_item()
        self.assertEqual(0, item["label"])
        self.assertNotIn("label", request.__dict__)
        self.assertFalse(hasattr(PredictionSnapshot, "label"))

    def test_package_contains_no_training_or_production_dependency(self):
        package = Path(__file__).resolve().parents[1] / "scripts/selector_relevance_gate"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package.glob("*.py")
        )
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("services.frozen_g1_runner", source)
        self.assertNotIn("webapp", source)
        self.assertNotIn("frontend", source)

    def test_cli_has_two_explicit_nonautomatic_modes(self):
        from scripts.selector_relevance_gate.run_gate import build_parser

        help_text = build_parser().format_help()
        self.assertIn("--invariance-smoke", help_text)
        self.assertIn("--heldout-gate", help_text)
        self.assertIn("--approved-invariance-smoke-report", help_text)

    def test_cli_verifies_stage_a_approval_before_opening_heldout(self):
        from scripts.selector_relevance_gate import run_gate

        training = SimpleNamespace()
        with mock.patch.object(
            run_gate, "validate_training_artifacts", return_value=training
        ), mock.patch.object(
            run_gate,
            "verify_approved_invariance_report",
            side_effect=EvaluationError("approval rejected"),
        ), mock.patch.object(run_gate, "load_heldout_references") as heldout_loader:
            code = run_gate.main(
                [
                    "--heldout-gate",
                    "--project-root",
                    "/project",
                    "--phase4a-config",
                    "/config.json",
                    "--neutral-dir",
                    "/neutral",
                    "--training-dir",
                    "/training",
                    "--output-dir",
                    "/output",
                    "--approved-invariance-smoke-report",
                    "/approved.json",
                    "--heldout-reference-artifact",
                    "/heldout.json",
                    "--heldout-reference-sha256",
                    "0" * 64,
                ]
            )
        self.assertEqual(2, code)
        heldout_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
