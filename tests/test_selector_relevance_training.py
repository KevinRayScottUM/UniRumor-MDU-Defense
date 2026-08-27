"""Focused tests for Step 2.6R-2 selector-only calibration."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.selector_relevance_training import metrics
from scripts.selector_relevance_training.dicc_backend import (
    DICCTorchBackend,
    _authoritative_batch_inputs,
    _move_model_to_training_device,
    _load_authoritative_runtime,
    configure_parameter_boundary,
    validate_optimizer_boundary,
)
from scripts.selector_relevance_training.metrics import (
    RankingExample,
    evaluate_ranking,
    grouped_ranking_metrics,
)
from scripts.selector_relevance_training.run_train import build_parser, main
from scripts.selector_relevance_training.trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    CalibrationExample,
    ExpectedDataCounts,
    IMPLEMENTATION_REVISION,
    SeedBackendResult,
    SelectorTrainingError,
    TRAINABLE_PARAMETER_NAMES,
    load_neutral_data,
    run_selector_calibration,
    sha256_file,
)


FIXTURE_COUNTS = ExpectedDataCounts(
    total_cases=3,
    total_examples=12,
    train_cases=2,
    train_examples=8,
    dev_cases=1,
    dev_examples=4,
    ocr_examples=6,
    transcript_examples=6,
    dataset_case_counts=(("GroundLie360", 2), ("TRUE-3MFact", 1)),
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _example(
    *,
    dataset: str,
    case_id: str,
    split: str,
    modality: str,
    index: int,
) -> Mapping[str, Any]:
    anchor = f"grounded content {case_id} {modality.lower()} {index}"
    positive_id = f"{case_id}-{modality.lower()}-{index}-positive"
    negative_id = f"{case_id}-{modality.lower()}-{index}-negative"
    return {
        "schema_version": 1,
        "calibration_example_id": f"example-{case_id}-{modality}-{index}",
        "source_dataset": dataset,
        "source_case_id": case_id,
        "canonical_underlying_case_id": f"{dataset}:{case_id}",
        "calibration_split": split,
        "expected_modality": modality,
        "claim": f'The relevant content states "{anchor}".',
        "anchor_unit_id": positive_id,
        "anchor_text": anchor,
        "positive_unit_ids": [positive_id],
        "model_exposed_candidate_count": 2,
        "candidate_units": [
            {
                "unit_id": positive_id,
                "unit_type": modality.lower(),
                "modality": "ocr" if modality == "OCR" else "text",
                "text": anchor,
                "relevance_target": 1,
            },
            {
                "unit_id": negative_id,
                "unit_type": "transcript" if modality == "OCR" else "ocr",
                "modality": "text" if modality == "OCR" else "ocr",
                "text": "different same-case content",
                "relevance_target": 0,
            },
        ],
        "source_provenance": {
            "train_variant_sha256": "a" * 64,
            "phase4a_config_sha256": "b" * 64,
            "source_row_index": index,
        },
    }


def _write_fixture(root: Path) -> tuple[Path, Mapping[str, str]]:
    source = root / "neutral-source"
    source.mkdir()
    train = []
    for case_id, dataset in (("train-a", "GroundLie360"), ("train-b", "TRUE-3MFact")):
        for modality in ("OCR", "TRANSCRIPT"):
            for index in range(2):
                train.append(
                    _example(
                        dataset=dataset,
                        case_id=case_id,
                        split="train",
                        modality=modality,
                        index=index,
                    )
                )
    dev = []
    for modality in ("OCR", "TRANSCRIPT"):
        for index in range(2):
            dev.append(
                _example(
                    dataset="GroundLie360",
                    case_id="dev-a",
                    split="dev",
                    modality=modality,
                    index=index,
                )
            )
    train_path = source / "neutral_calibration_train.jsonl"
    dev_path = source / "neutral_calibration_dev.jsonl"
    train_path.write_bytes(_jsonl_bytes(train))
    dev_path.write_bytes(_jsonl_bytes(dev))
    manifest_rows = []
    for item in train + dev:
        manifest_rows.append(
            {
                "calibration_example_id": item["calibration_example_id"],
                "expected_modality": item["expected_modality"],
                "old_claim": "old",
                "new_claim": item["claim"],
                "anchor_text": item["anchor_text"],
                "anchor_text_unchanged": True,
                "candidate_ids_unchanged": True,
                "candidate_order_unchanged": True,
                "candidate_content_unchanged": True,
                "positive_ids_unchanged": True,
                "relevance_targets_unchanged": True,
                "underlying_case_unchanged": True,
                "split_unchanged": True,
                "all_non_claim_content_unchanged": True,
            }
        )
    manifest_path = source / "neutral_revision_manifest.json"
    manifest_path.write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "implementation_revision": "step2.6r-1d-v1",
                "example_count": 12,
                "examples": manifest_rows,
            }
        )
    )
    hashes = {
        train_path.name: sha256_file(train_path),
        dev_path.name: sha256_file(dev_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    for name, digest in hashes.items():
        (source / (name.rsplit(".", 1)[0] + ".sha256")).write_text(
            digest + "\n", encoding="utf-8"
        )
    report = {
        "status": "PASS",
        "implementation_revision": "step2.6r-1d-v1",
        "neutral_train_sha256": hashes["neutral_calibration_train.jsonl"],
        "neutral_dev_sha256": hashes["neutral_calibration_dev.jsonl"],
        "artifact_sha256": dict(hashes),
        "heldout_overlap": 0,
        "train_dev_overlap": 0,
        "selection_outputs_inspected": False,
        "veracity_labels_inspected": False,
        "formal_validation_accessed": False,
        "formal_test_accessed": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "training_started": False,
        "production_or_model_code_changed": False,
    }
    (source / "neutral_build_report.json").write_bytes(_json_bytes(report))
    return source, hashes


def _ranking(item: Any, *, trained: bool) -> RankingExample:
    targets = item.relevance_targets
    scores = tuple((2.0 if target else -1.0) if trained else 0.0 for target in targets)
    return RankingExample(
        calibration_example_id=item.calibration_example_id,
        source_dataset=item.source_dataset,
        expected_modality=item.expected_modality,
        candidate_unit_ids=item.candidate_unit_ids,
        relevance_targets=targets,
        selection_scores=scores,
    )


class FakeBackend:
    checkpoint_sha256 = AUTHORITATIVE_CHECKPOINT_SHA256

    def __init__(self) -> None:
        self.saved_payloads = []

    def baseline_rankings(self, dev_examples: Sequence[Any]) -> Sequence[RankingExample]:
        return tuple(_ranking(item, trained=False) for item in dev_examples)

    def train_seed(self, **kwargs: Any) -> SeedBackendResult:
        seed = kwargs["seed"]
        dev_examples = kwargs["dev_examples"]
        return SeedBackendResult(
            seed=seed,
            selected_epoch=1,
            history=(
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "dev_metrics": grouped_ranking_metrics(
                        tuple(_ranking(item, trained=True) for item in dev_examples)
                    ),
                },
            ),
            dev_rankings=tuple(_ranking(item, trained=True) for item in dev_examples),
            selection_head_state_dict={"weight": [float(seed)], "bias": [0.1]},
            encoder_parameter_hash_before="encoder",
            encoder_parameter_hash_after="encoder",
            veracity_head_parameter_hash_before="veracity",
            veracity_head_parameter_hash_after="veracity",
            selection_head_parameter_hash_before="selection-before",
            selection_head_parameter_hash_after=f"selection-after-{seed}",
            trainable_parameter_names=TRAINABLE_PARAMETER_NAMES,
            optimizer_parameter_names=TRAINABLE_PARAMETER_NAMES,
            loss_finite=True,
            selection_scores_finite=True,
        )

    def save_selector_artifact(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.saved_payloads.append(payload)
        path.write_bytes(_json_bytes(payload))

    def current_checkpoint_sha256(self) -> str:
        return AUTHORITATIVE_CHECKPOINT_SHA256


class RankingMetricTests(unittest.TestCase):
    def test_exact_metrics_for_perfect_ranking(self) -> None:
        example = RankingExample(
            calibration_example_id="example",
            source_dataset="GroundLie360",
            expected_modality="OCR",
            candidate_unit_ids=("a", "b", "c"),
            relevance_targets=(1, 0, 0),
            selection_scores=(0.8, 0.2, -0.1),
        )
        self.assertEqual(
            {
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "ndcg_at_5": 1.0,
            },
            evaluate_ranking((example,)),
        )

    def test_ties_use_original_candidate_order(self) -> None:
        example = RankingExample(
            calibration_example_id="example",
            source_dataset="TRUE-3MFact",
            expected_modality="TRANSCRIPT",
            candidate_unit_ids=("negative-first", "positive-second"),
            relevance_targets=(0, 1),
            selection_scores=(0.0, 0.0),
        )
        result = evaluate_ranking((example,))
        self.assertEqual(0.0, result["recall_at_1"])
        self.assertEqual(0.5, result["mrr"])

    def test_grouped_metrics_report_dataset_and_anchor_modality(self) -> None:
        examples = (
            RankingExample("a", "GroundLie360", "OCR", ("u",), (1,), (1.0,)),
            RankingExample("b", "TRUE-3MFact", "TRANSCRIPT", ("v",), (1,), (1.0,)),
        )
        result = grouped_ranking_metrics(examples)
        self.assertEqual({"GroundLie360", "TRUE-3MFact"}, set(result["by_dataset"]))
        self.assertEqual({"OCR", "TRANSCRIPT"}, set(result["by_anchor_modality"]))


class SourceGateTests(unittest.TestCase):
    def test_closed_neutral_artifact_loads_and_train_only_class_weight_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, hashes = _write_fixture(Path(temporary))
            data = load_neutral_data(
                source, expected_hashes=hashes, expected_counts=FIXTURE_COUNTS
            )
            self.assertEqual(8, data.train_positive_pairs)
            self.assertEqual(8, data.train_negative_pairs)
            self.assertEqual(0.5, data.positive_prevalence)
            self.assertEqual(1.0, data.pos_weight)

    def test_source_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, hashes = _write_fixture(Path(temporary))
            bad_hashes = dict(hashes)
            bad_hashes["neutral_calibration_train.jsonl"] = "0" * 64
            with self.assertRaisesRegex(SelectorTrainingError, "SHA mismatch"):
                load_neutral_data(
                    source,
                    expected_hashes=bad_hashes,
                    expected_counts=FIXTURE_COUNTS,
                )

    def test_source_gate_runs_before_neural_backend_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = _write_fixture(root)
            bad_hashes = dict(hashes)
            bad_hashes["neutral_calibration_dev.jsonl"] = "0" * 64
            calls = []
            with self.assertRaisesRegex(SelectorTrainingError, "SHA mismatch"):
                run_selector_calibration(
                    source_dir=source,
                    output_dir=root / "output",
                    run_mode="smoke",
                    backend_factory=lambda: calls.append(True),
                    expected_hashes=bad_hashes,
                    expected_counts=FIXTURE_COUNTS,
                )
            self.assertEqual([], calls)

    def test_veracity_or_prediction_fields_are_rejected_without_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, hashes = _write_fixture(Path(temporary))
            path = source / "neutral_calibration_train.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["veracity_label"] = "real"
            path.write_bytes(_jsonl_bytes(records))
            digest = sha256_file(path)
            hashes = dict(hashes)
            hashes[path.name] = digest
            (source / "neutral_calibration_train.sha256").write_text(
                digest + "\n", encoding="utf-8"
            )
            report_path = source / "neutral_build_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["neutral_train_sha256"] = digest
            report["artifact_sha256"][path.name] = digest
            report_path.write_bytes(_json_bytes(report))
            with self.assertRaisesRegex(SelectorTrainingError, "forbidden scientific"):
                load_neutral_data(
                    source, expected_hashes=hashes, expected_counts=FIXTURE_COUNTS
                )

    def test_formal_validation_and_test_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("Formal Validation", "Formal_Test"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(SelectorTrainingError, "must not reference"):
                        load_neutral_data(root / name)


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModule:
    def __init__(self) -> None:
        self.mode = None

    def eval(self) -> None:
        self.mode = "eval"

    def train(self) -> None:
        self.mode = "train"


class _FakeModel:
    def __init__(self, *, extra: bool = False) -> None:
        self.encoder = _FakeModule()
        self.veracity_head = _FakeModule()
        self.selection_head = _FakeModule()
        names = [
            "encoder.layer.weight",
            "veracity_head.weight",
            "veracity_head.bias",
            "selection_head.weight",
            "selection_head.bias",
        ]
        if extra:
            names.append("other.weight")
        self.named = [(name, _FakeParameter()) for name in names]

    def named_parameters(self) -> Sequence[tuple[str, _FakeParameter]]:
        return tuple(self.named)


class _FakeOptimizer:
    def __init__(self, parameters: Sequence[_FakeParameter]) -> None:
        self.param_groups = [{"params": list(parameters)}]


class ParameterBoundaryTests(unittest.TestCase):
    def test_only_selection_head_is_trainable_and_optimizer_owned(self) -> None:
        model = _FakeModel()
        trainable = configure_parameter_boundary(model)
        self.assertEqual(TRAINABLE_PARAMETER_NAMES, trainable)
        states = {name: parameter.requires_grad for name, parameter in model.named}
        self.assertFalse(states["encoder.layer.weight"])
        self.assertFalse(states["veracity_head.weight"])
        self.assertTrue(states["selection_head.weight"])
        self.assertEqual("eval", model.encoder.mode)
        self.assertEqual("eval", model.veracity_head.mode)
        self.assertEqual("train", model.selection_head.mode)
        optimizer = _FakeOptimizer([model.named[-2][1], model.named[-1][1]])
        self.assertEqual(TRAINABLE_PARAMETER_NAMES, validate_optimizer_boundary(model, optimizer))

    def test_unknown_model_or_optimizer_parameter_fails_closed(self) -> None:
        with self.assertRaisesRegex(SelectorTrainingError, "outside frozen G1"):
            configure_parameter_boundary(_FakeModel(extra=True))
        model = _FakeModel()
        configure_parameter_boundary(model)
        optimizer = _FakeOptimizer([model.named[0][1]])
        with self.assertRaisesRegex(SelectorTrainingError, "only selection"):
            validate_optimizer_boundary(model, optimizer)

    def test_authoritative_phase3a_collator_is_loaded_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            phase3 = project / "MDU/scripts/clip12_phase3_common"
            phase4 = project / "MDU/scripts/clip12_phase4a_inference_handoff"
            phase3.mkdir(parents=True)
            phase4.mkdir(parents=True)
            (phase3 / "clip12p3_model.py").write_text(
                "def collator(tokenizer, max_length):\n"
                "    def apply(items):\n"
                "        return (tokenizer, max_length, items)\n"
                "    return apply\n",
                encoding="utf-8",
            )
            (phase4 / "clip12p4a_engine.py").write_text(
                "class FrozenG1Engine:\n"
                "    def __init__(self, config, project_root, *, device_name='auto'):\n"
                "        if device_name != 'cpu':\n"
                "            raise AssertionError('engine must initialize on CPU')\n"
                "        self.model = object()\n"
                "        self.tokenizer = 'authoritative-tokenizer'\n",
                encoding="utf-8",
            )
            config_path = project / "phase4a.json"
            config_path.write_text("{}\n", encoding="utf-8")
            model, tokenizer, collate = _load_authoritative_runtime(
                project,
                config_path,
                {},
            )
            self.assertIsNotNone(model)
            self.assertEqual("authoritative-tokenizer", tokenizer)
            self.assertEqual(
                ("authoritative-tokenizer", 256, ["item"]), collate(["item"])
            )

    def test_legacy_phase4a_engine_name_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            phase3 = project / "MDU/scripts/clip12_phase3_common"
            phase4 = project / "MDU/scripts/clip12_phase4a_inference_handoff"
            phase3.mkdir(parents=True)
            phase4.mkdir(parents=True)
            (phase3 / "clip12p3_model.py").write_text(
                "def collator(tokenizer, max_length):\n"
                "    return lambda items: items\n",
                encoding="utf-8",
            )
            (phase4 / "clip12p4a_engine.py").write_text(
                "class Phase4AEngine:\n"
                "    pass\n",
                encoding="utf-8",
            )
            config_path = project / "phase4a.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SelectorTrainingError, "FrozenG1Engine is unavailable"
            ):
                _load_authoritative_runtime(project, config_path, {})

    def test_requested_indexed_cuda_device_is_preserved_after_cpu_engine_load(self) -> None:
        class DeviceModel:
            def __init__(self) -> None:
                self.devices = []

            def to(self, device: str) -> None:
                self.devices.append(device)

        model = DeviceModel()
        _move_model_to_training_device(model, "cuda:1")
        self.assertEqual(["cuda:1"], model.devices)


class AuthoritativeBatchTests(unittest.TestCase):
    class Tensor:
        def __init__(self) -> None:
            self.devices = []

        def to(self, device: str) -> "AuthoritativeBatchTests.Tensor":
            self.devices.append(device)
            return self

        def __getitem__(self, key: Any) -> "AuthoritativeBatchTests.Tensor":
            return self

        def detach(self) -> "AuthoritativeBatchTests.Tensor":
            return self

        def cpu(self) -> "AuthoritativeBatchTests.Tensor":
            return self

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    def setUp(self) -> None:
        self.example = CalibrationExample(
            calibration_example_id="cal-example",
            source_dataset="GroundLie360",
            source_case_id="source-case",
            canonical_underlying_case_id="GroundLie360:source-case",
            calibration_split="train",
            expected_modality="OCR",
            claim='The relevant content states "content".',
            candidate_units=(
                {
                    "unit_id": "unit-a",
                    "unit_type": "ocr",
                    "modality": "ocr",
                    "text": "content",
                    "relevance_target": 1,
                },
                {
                    "unit_id": "unit-b",
                    "unit_type": "transcript",
                    "modality": "text",
                    "text": "other",
                    "relevance_target": 0,
                },
            ),
            relevance_targets=(1, 0),
        )

    def _batch(self, **overrides: Any) -> Any:
        values = {
            "input_ids": self.Tensor(),
            "attention_mask": self.Tensor(),
            "sample_index": [0, 0],
            "labels": [0],
            "case_ids": ["cal-example"],
            "unit_ids": ["unit-a", "unit-b"],
        }
        values.update(overrides)
        return type("Batch", (), values)()

    def test_authoritative_batch_without_encoded_prepares_representations(self) -> None:
        captured_items = []
        batch = self._batch()

        class Encoder:
            def __init__(self, tensor: Any) -> None:
                self.tensor = tensor
                self.calls = []

            def eval(self) -> None:
                return None

            def __call__(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                hidden = self.tensor
                hidden.shape = (2, 4, 8)
                return type("Output", (), {"last_hidden_state": hidden})()

        tensor = self.Tensor()
        encoder = Encoder(tensor)
        backend = object.__new__(DICCTorchBackend)
        backend.device = "cuda:1"
        backend.collate = lambda items: captured_items.extend(items) or batch
        backend.model = type("Model", (), {"encoder": encoder})()
        backend.torch = type("Torch", (), {"no_grad": lambda self: AuthoritativeBatchTests.NoGrad()})()
        backend._cache = {}
        prepared = backend._prepare((self.example,))
        self.assertEqual(1, len(prepared))
        self.assertFalse(hasattr(batch, "encoded"))
        self.assertNotIn("case_id", self.example.collator_item())
        self.assertNotIn("label", self.example.collator_item())
        self.assertEqual("cal-example", captured_items[0]["case_id"])
        self.assertEqual(0, captured_items[0]["label"])
        self.assertEqual({"input_ids", "attention_mask"}, set(encoder.calls[0]))
        self.assertEqual(["cuda:1"], batch.input_ids.devices)
        self.assertEqual(["cuda:1"], batch.attention_mask.devices)

    def test_missing_authoritative_tensors_fail_closed(self) -> None:
        for field in ("input_ids", "attention_mask"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(SelectorTrainingError, field):
                    _authoritative_batch_inputs(
                        self._batch(**{field: None}),
                        example=self.example,
                        device="cuda:0",
                    )

    def test_candidate_order_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SelectorTrainingError, "candidate order"):
            _authoritative_batch_inputs(
                self._batch(unit_ids=["unit-b", "unit-a"]),
                example=self.example,
                device="cuda:0",
            )


class OrchestrationTests(unittest.TestCase):
    def test_smoke_is_small_passes_all_gates_and_preserves_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = _write_fixture(root)
            source_before = {
                path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
            }
            backend = FakeBackend()
            output = root / "smoke-output"
            report = run_selector_calibration(
                source_dir=source,
                output_dir=output,
                run_mode="smoke",
                backend=backend,
                expected_hashes=hashes,
                expected_counts=FIXTURE_COUNTS,
            )
            source_after = {
                path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
            }
            self.assertEqual(source_before, source_after)
            self.assertEqual("PASS", report["status"])
            self.assertEqual("smoke", report["run_mode"])
            self.assertEqual("step2.6r-2-r1-v1", IMPLEMENTATION_REVISION)
            self.assertEqual(IMPLEMENTATION_REVISION, report["implementation_revision"])
            self.assertEqual(8, report["run_train_example_count"])
            self.assertEqual(4, report["run_dev_example_count"])
            self.assertEqual(
                [42], report["training_protocol"]["effective_seeds"]
            )
            self.assertEqual(
                1, report["training_protocol"]["effective_maximum_epochs"]
            )
            self.assertFalse(report["full_training_automatically_triggered"])
            self.assertTrue(report["collator_dummy_label_used"])
            self.assertEqual(0, report["collator_dummy_label_value"])
            self.assertFalse(report["veracity_labels_inspected"])
            self.assertTrue((output / "smoke_report.json").is_file())
            self.assertEqual(1, len(backend.saved_payloads))
            self.assertEqual(
                {"weight", "bias"},
                set(backend.saved_payloads[0]["selection_head_state_dict"]),
            )

    def test_full_requires_approved_pass_smoke_and_runs_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = _write_fixture(root)
            with self.assertRaisesRegex(SelectorTrainingError, "approved-smoke"):
                run_selector_calibration(
                    source_dir=source,
                    output_dir=root / "blocked-full",
                    run_mode="full",
                    backend=FakeBackend(),
                    expected_hashes=hashes,
                    expected_counts=FIXTURE_COUNTS,
                )
            smoke_output = root / "smoke"
            run_selector_calibration(
                source_dir=source,
                output_dir=smoke_output,
                run_mode="smoke",
                backend=FakeBackend(),
                expected_hashes=hashes,
                expected_counts=FIXTURE_COUNTS,
            )
            backend = FakeBackend()
            full_output = root / "full"
            report = run_selector_calibration(
                source_dir=source,
                output_dir=full_output,
                run_mode="full",
                backend=backend,
                approved_smoke_report=smoke_output / "smoke_report.json",
                expected_hashes=hashes,
                expected_counts=FIXTURE_COUNTS,
            )
            self.assertEqual([42, 43, 44], [item["seed"] for item in report["seed_results"]])
            self.assertEqual(3, len(backend.saved_payloads))
            self.assertTrue((full_output / "multi_seed_summary.json").is_file())

    def test_mutated_frozen_hash_or_unchanged_selection_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = _write_fixture(root)

            class UnsafeBackend(FakeBackend):
                def train_seed(self, **kwargs: Any) -> SeedBackendResult:
                    result = super().train_seed(**kwargs)
                    return SeedBackendResult(
                        **{
                            **result.__dict__,
                            "encoder_parameter_hash_after": "changed",
                            "selection_head_parameter_hash_after": "selection-before",
                        }
                    )

            with self.assertRaisesRegex(SelectorTrainingError, "encoder parameters changed"):
                run_selector_calibration(
                    source_dir=source,
                    output_dir=root / "unsafe",
                    run_mode="smoke",
                    backend=UnsafeBackend(),
                    expected_hashes=hashes,
                    expected_counts=FIXTURE_COUNTS,
                )


class CLIBoundaryTests(unittest.TestCase):
    def test_cli_modes_and_full_approval_contract(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--smoke",
                "--project-root",
                "/project",
                "--phase4a-config",
                "/config.json",
                "--neutral-dir",
                "/neutral",
                "--output-dir",
                "/output",
            ]
        )
        self.assertTrue(args.smoke)
        self.assertFalse(args.full)
        self.assertEqual("cuda:0", args.device)
        self.assertEqual(
            2,
            main(
                [
                    "--full",
                    "--project-root",
                    "/project",
                    "--phase4a-config",
                    "/config.json",
                    "--neutral-dir",
                    "/neutral",
                    "--output-dir",
                    "/output",
                ]
            ),
        )

    def test_imports_are_lazy_and_production_scientific_modules_are_untouched(self) -> None:
        package = Path("scripts/selector_relevance_training")
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            top_imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top_imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    top_imports.append(node.module or "")
            self.assertNotIn("torch", top_imports)
            self.assertNotIn("transformers", top_imports)
            self.assertFalse(any(name.startswith("services") for name in top_imports))
            self.assertFalse(any(name.startswith("webapp") for name in top_imports))
        backend_source = (package / "dicc_backend.py").read_text(encoding="utf-8")
        self.assertIn("collator_factory(tokenizer, MAX_LENGTH)", backend_source)
        self.assertIn("BCEWithLogitsLoss", backend_source)
        self.assertIn("for target in item.example.relevance_targets", backend_source)
        self.assertNotIn("batch.labels", backend_source)
        self.assertNotIn("model.veracity_head(", backend_source)


if __name__ == "__main__":
    unittest.main()
