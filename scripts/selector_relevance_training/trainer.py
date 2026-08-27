"""Scientific gates and orchestration for selector-only calibration training.

The module is standard-library-only.  The DICC neural backend is imported only
after an explicit ``--smoke`` or approved ``--full`` CLI invocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from .metrics import (
    METRIC_NAMES,
    RankingExample,
    finite_metrics,
    grouped_ranking_metrics,
    mean_and_population_std,
)


IMPLEMENTATION_REVISION = "step2.6r-2-r1-v1"
SELECTOR_ID = "G1-RelevanceSelector-Cal-v1"
AUTHORITATIVE_CHECKPOINT_SHA256 = (
    "b694f2d4bb5ba6f72dd8a001bd984d46853546f2a85858a812f2496af1f1a0b9"
)
AUTHORITATIVE_SOURCE_HASHES = {
    "neutral_calibration_train.jsonl": (
        "e6920ac7b903cb9a4cd305b57e6019010b2a5684a4699f083f00a46329eccfa2"
    ),
    "neutral_calibration_dev.jsonl": (
        "e88b571e4d7061ee4190c7edfd6a963220330182946bcb2012b553101b125cca"
    ),
    "neutral_revision_manifest.json": (
        "800d192f52c7787d6c2979b814ed87f60065b5e1187b1df9f2f89e1ed197fee7"
    ),
}
NEUTRAL_TEMPLATE_PREFIX = 'The relevant content states "'
NEUTRAL_TEMPLATE_SUFFIX = '".'
TRAINABLE_PARAMETER_NAMES = (
    "selection_head.weight",
    "selection_head.bias",
)
FROZEN_PARAMETER_PREFIXES = ("encoder.", "veracity_head.")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESTRICTED_PATH_PARTS = {"validation", "test", "formalvalidation", "formaltest"}
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "label",
        "veracity_label",
        "veracity_labels",
        "prediction",
        "prediction_id",
        "sample_logits",
        "veracity_logits",
        "probabilities",
        "correct_class_logits",
        "veracity_margin",
        "selection_distillation_target",
        "selection_score",
        "selection_scores",
    }
)
_FALSE_BOUNDARY_FIELDS = (
    "selection_outputs_inspected",
    "veracity_labels_inspected",
    "formal_validation_accessed",
    "formal_test_accessed",
    "model_loaded",
    "checkpoint_loaded",
    "training_started",
    "production_or_model_code_changed",
)
_MANIFEST_TRUE_FIELDS = (
    "anchor_text_unchanged",
    "candidate_ids_unchanged",
    "candidate_order_unchanged",
    "candidate_content_unchanged",
    "positive_ids_unchanged",
    "relevance_targets_unchanged",
    "underlying_case_unchanged",
    "split_unchanged",
    "all_non_claim_content_unchanged",
)


class SelectorTrainingError(RuntimeError):
    """Raised when a source, training, or scientific gate fails closed."""


@dataclass(frozen=True)
class ExpectedDataCounts:
    total_cases: int
    total_examples: int
    train_cases: int
    train_examples: int
    dev_cases: int
    dev_examples: int
    ocr_examples: int
    transcript_examples: int
    dataset_case_counts: Tuple[Tuple[str, int], ...]

    @property
    def datasets(self) -> Mapping[str, int]:
        return dict(self.dataset_case_counts)


AUTHORITATIVE_COUNTS = ExpectedDataCounts(
    total_cases=1306,
    total_examples=5224,
    train_cases=1045,
    train_examples=4180,
    dev_cases=261,
    dev_examples=1044,
    ocr_examples=2612,
    transcript_examples=2612,
    dataset_case_counts=(("GroundLie360", 570), ("TRUE-3MFact", 736)),
)


@dataclass(frozen=True)
class TrainingProtocol:
    seeds: Tuple[int, ...] = (42, 43, 44)
    maximum_epochs: int = 10
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size_examples: int = 32
    early_stopping_patience: int = 2
    primary_metric: str = "mrr"
    objective: str = "BCEWithLogitsLoss"
    pos_weight_source: str = "negative_train_pairs / positive_train_pairs"

    def __post_init__(self) -> None:
        if self.seeds != (42, 43, 44):
            raise SelectorTrainingError("full protocol seeds must be exactly 42, 43, 44")
        if self.maximum_epochs != 10:
            raise SelectorTrainingError("maximum_epochs must equal 10")
        if self.optimizer != "AdamW" or self.learning_rate != 1e-3:
            raise SelectorTrainingError("optimizer protocol must be AdamW at 1e-3")
        if self.weight_decay != 0.0 or self.batch_size_examples != 32:
            raise SelectorTrainingError("weight decay and batch size protocol changed")
        if self.early_stopping_patience != 2 or self.primary_metric != "mrr":
            raise SelectorTrainingError("model-selection protocol changed")
        if self.objective != "BCEWithLogitsLoss":
            raise SelectorTrainingError("selector objective must be BCEWithLogitsLoss")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "seeds": list(self.seeds),
            "maximum_epochs": self.maximum_epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size_examples": self.batch_size_examples,
            "early_stopping_patience": self.early_stopping_patience,
            "primary_metric": self.primary_metric,
            "secondary_metrics": [
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "ndcg_at_5",
            ],
            "objective": self.objective,
            "target": "relevance_target",
            "pos_weight_source": self.pos_weight_source,
            "stable_tie_break": "original candidate order",
        }


@dataclass(frozen=True)
class CalibrationExample:
    calibration_example_id: str
    source_dataset: str
    source_case_id: str
    canonical_underlying_case_id: str
    calibration_split: str
    expected_modality: str
    claim: str
    candidate_units: Tuple[Mapping[str, Any], ...]
    relevance_targets: Tuple[int, ...]

    @property
    def candidate_unit_ids(self) -> Tuple[str, ...]:
        return tuple(str(item["unit_id"]) for item in self.candidate_units)

    def collator_item(self) -> Mapping[str, Any]:
        return {
            "claim": self.claim,
            "dataset": self.source_dataset,
            "units": [dict(item) for item in self.candidate_units],
        }


@dataclass(frozen=True)
class NeutralDataBundle:
    source_dir: Path
    train_examples: Tuple[CalibrationExample, ...]
    dev_examples: Tuple[CalibrationExample, ...]
    source_hashes: Mapping[str, str]
    train_positive_pairs: int
    train_negative_pairs: int

    @property
    def positive_prevalence(self) -> float:
        total = self.train_positive_pairs + self.train_negative_pairs
        return self.train_positive_pairs / total

    @property
    def pos_weight(self) -> float:
        return self.train_negative_pairs / self.train_positive_pairs


@dataclass(frozen=True)
class SeedBackendResult:
    seed: int
    selected_epoch: int
    history: Tuple[Mapping[str, Any], ...]
    dev_rankings: Tuple[RankingExample, ...]
    selection_head_state_dict: Any
    encoder_parameter_hash_before: str
    encoder_parameter_hash_after: str
    veracity_head_parameter_hash_before: str
    veracity_head_parameter_hash_after: str
    selection_head_parameter_hash_before: str
    selection_head_parameter_hash_after: str
    trainable_parameter_names: Tuple[str, ...]
    optimizer_parameter_names: Tuple[str, ...]
    loss_finite: bool
    selection_scores_finite: bool


class SelectorTrainingBackend(Protocol):
    checkpoint_sha256: str

    def baseline_rankings(
        self, dev_examples: Sequence[CalibrationExample]
    ) -> Sequence[RankingExample]: ...

    def train_seed(
        self,
        *,
        seed: int,
        train_examples: Sequence[CalibrationExample],
        dev_examples: Sequence[CalibrationExample],
        pos_weight: float,
        maximum_epochs: int,
        protocol: TrainingProtocol,
    ) -> SeedBackendResult: ...

    def save_selector_artifact(self, path: Path, payload: Mapping[str, Any]) -> None: ...

    def current_checkpoint_sha256(self) -> str: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _resolve_safe_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized = {
        re.sub(r"[^a-z0-9]+", "", part.casefold()) for part in resolved.parts
    }
    if normalized & _RESTRICTED_PATH_PARTS:
        raise SelectorTrainingError(f"{field} must not reference Formal Validation/Test")
    return resolved


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SelectorTrainingError(f"{field} is missing") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectorTrainingError(f"{field} is malformed") from exc
    if not isinstance(value, dict):
        raise SelectorTrainingError(f"{field} must be a JSON object")
    return value


def _read_jsonl(path: Path, field: str) -> Tuple[Mapping[str, Any], ...]:
    records = []
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SelectorTrainingError(f"{field} is missing") from exc
    try:
        with stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SelectorTrainingError(f"{field} row {index} is malformed") from exc
                if not isinstance(value, dict):
                    raise SelectorTrainingError(f"{field} row {index} must be an object")
                records.append(value)
    except UnicodeDecodeError as exc:
        raise SelectorTrainingError(f"{field} is not UTF-8") from exc
    return tuple(records)


def _assert_no_forbidden_inputs(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_INPUT_KEYS:
                raise SelectorTrainingError(f"forbidden scientific input field: {key}")
            _assert_no_forbidden_inputs(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_inputs(item)


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelectorTrainingError(f"{field} must be a nonblank string")
    return value


def _parse_example(
    record: Mapping[str, Any], *, split: str, index: int
) -> CalibrationExample:
    _assert_no_forbidden_inputs(record)
    field = f"neutral {split} row {index}"
    actual_split = _nonblank(record.get("calibration_split"), f"{field}.split")
    if actual_split != split:
        raise SelectorTrainingError(f"{field} is stored under the wrong split")
    modality = _nonblank(record.get("expected_modality"), f"{field}.modality")
    if modality not in {"OCR", "TRANSCRIPT"}:
        raise SelectorTrainingError(f"{field} has an invalid expected modality")
    claim = _nonblank(record.get("claim"), f"{field}.claim")
    anchor_text = _nonblank(record.get("anchor_text"), f"{field}.anchor_text")
    if claim != NEUTRAL_TEMPLATE_PREFIX + anchor_text + NEUTRAL_TEMPLATE_SUFFIX:
        raise SelectorTrainingError(f"{field} does not use the exact neutral template")
    candidates = record.get("candidate_units")
    if not isinstance(candidates, list) or not candidates:
        raise SelectorTrainingError(f"{field}.candidate_units must be nonempty")
    candidate_records = []
    candidate_ids = []
    targets = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise SelectorTrainingError(f"{field}.candidate {candidate_index} is invalid")
        unit_id = _nonblank(candidate.get("unit_id"), f"{field}.candidate.unit_id")
        _nonblank(candidate.get("unit_type"), f"{field}.candidate.unit_type")
        _nonblank(candidate.get("modality"), f"{field}.candidate.modality")
        _nonblank(candidate.get("text"), f"{field}.candidate.text")
        target = candidate.get("relevance_target")
        if type(target) is not int or target not in {0, 1}:
            raise SelectorTrainingError(f"{field}.relevance_target must be 0 or 1")
        candidate_ids.append(unit_id)
        targets.append(target)
        candidate_records.append(dict(candidate))
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SelectorTrainingError(f"{field} contains duplicate candidate IDs")
    if not any(targets):
        raise SelectorTrainingError(f"{field} contains no direct-relevance positive")
    positive_ids = record.get("positive_unit_ids")
    if not isinstance(positive_ids, list) or tuple(
        candidate_id
        for candidate_id, target in zip(candidate_ids, targets)
        if target == 1
    ) != tuple(positive_ids):
        raise SelectorTrainingError(f"{field} positive IDs disagree with targets")
    return CalibrationExample(
        calibration_example_id=_nonblank(
            record.get("calibration_example_id"), f"{field}.id"
        ),
        source_dataset=_nonblank(record.get("source_dataset"), f"{field}.dataset"),
        source_case_id=_nonblank(record.get("source_case_id"), f"{field}.case"),
        canonical_underlying_case_id=_nonblank(
            record.get("canonical_underlying_case_id"), f"{field}.underlying_case"
        ),
        calibration_split=actual_split,
        expected_modality=modality,
        claim=claim,
        candidate_units=tuple(candidate_records),
        relevance_targets=tuple(targets),
    )


def load_neutral_data(
    source_dir: Path,
    *,
    expected_hashes: Mapping[str, str] = AUTHORITATIVE_SOURCE_HASHES,
    expected_counts: ExpectedDataCounts = AUTHORITATIVE_COUNTS,
) -> NeutralDataBundle:
    source_dir = _resolve_safe_path(source_dir, "neutral calibration source")
    if not source_dir.is_dir():
        raise SelectorTrainingError("neutral calibration source is missing")
    required = tuple(expected_hashes) + (
        "neutral_calibration_train.sha256",
        "neutral_calibration_dev.sha256",
        "neutral_revision_manifest.sha256",
        "neutral_build_report.json",
    )
    for name in required:
        if not (source_dir / name).is_file():
            raise SelectorTrainingError(f"required neutral artifact is missing: {name}")
    actual_hashes = {name: sha256_file(source_dir / name) for name in expected_hashes}
    if actual_hashes != dict(expected_hashes):
        raise SelectorTrainingError("authoritative neutral artifact SHA mismatch")
    for name in expected_hashes:
        sidecar = source_dir / (name.rsplit(".", 1)[0] + ".sha256")
        value = sidecar.read_text(encoding="utf-8").strip().casefold()
        if not _SHA256_RE.fullmatch(value) or value != actual_hashes[name]:
            raise SelectorTrainingError(f"neutral SHA sidecar mismatch for {name}")
    report = _read_json(source_dir / "neutral_build_report.json", "neutral build report")
    if report.get("status") != "PASS" or report.get("implementation_revision") != "step2.6r-1d-v1":
        raise SelectorTrainingError("neutral build report is not the closed Step 2.6R-1D artifact")
    report_hashes = report.get("artifact_sha256")
    if not isinstance(report_hashes, Mapping):
        raise SelectorTrainingError("neutral build report artifact hashes are missing")
    declared_hashes = {
        "neutral_calibration_train.jsonl": report.get("neutral_train_sha256"),
        "neutral_calibration_dev.jsonl": report.get("neutral_dev_sha256"),
        "neutral_revision_manifest.json": report_hashes.get(
            "neutral_revision_manifest.json"
        ),
    }
    if declared_hashes != actual_hashes:
        raise SelectorTrainingError("neutral build report SHA declarations mismatch")
    for field in _FALSE_BOUNDARY_FIELDS:
        if report.get(field) is not False:
            raise SelectorTrainingError(f"neutral build report boundary failed: {field}")
    if report.get("heldout_overlap") != 0 or report.get("train_dev_overlap") != 0:
        raise SelectorTrainingError("neutral build report contains held-out or split overlap")
    train_records = _read_jsonl(
        source_dir / "neutral_calibration_train.jsonl", "neutral Train"
    )
    dev_records = _read_jsonl(
        source_dir / "neutral_calibration_dev.jsonl", "neutral Dev"
    )
    train = tuple(
        _parse_example(record, split="train", index=index)
        for index, record in enumerate(train_records)
    )
    dev = tuple(
        _parse_example(record, split="dev", index=index)
        for index, record in enumerate(dev_records)
    )
    all_examples = train + dev
    example_ids = [item.calibration_example_id for item in all_examples]
    if len(example_ids) != len(set(example_ids)):
        raise SelectorTrainingError("neutral calibration example IDs are not unique")
    train_cases = {item.canonical_underlying_case_id for item in train}
    dev_cases = {item.canonical_underlying_case_id for item in dev}
    if train_cases & dev_cases:
        raise SelectorTrainingError("neutral calibration Train/Dev overlap")
    case_datasets = {}
    for item in all_examples:
        prior = case_datasets.setdefault(
            item.canonical_underlying_case_id, item.source_dataset
        )
        if prior != item.source_dataset:
            raise SelectorTrainingError("underlying case dataset identity changed")
    dataset_counts = Counter(case_datasets.values())
    modality_counts = Counter(item.expected_modality for item in all_examples)
    observed = {
        "total_cases": len(case_datasets),
        "total_examples": len(all_examples),
        "train_cases": len(train_cases),
        "train_examples": len(train),
        "dev_cases": len(dev_cases),
        "dev_examples": len(dev),
        "ocr_examples": modality_counts["OCR"],
        "transcript_examples": modality_counts["TRANSCRIPT"],
    }
    for field, expected in (
        ("total_cases", expected_counts.total_cases),
        ("total_examples", expected_counts.total_examples),
        ("train_cases", expected_counts.train_cases),
        ("train_examples", expected_counts.train_examples),
        ("dev_cases", expected_counts.dev_cases),
        ("dev_examples", expected_counts.dev_examples),
        ("ocr_examples", expected_counts.ocr_examples),
        ("transcript_examples", expected_counts.transcript_examples),
    ):
        if observed[field] != expected:
            raise SelectorTrainingError(f"neutral source count mismatch: {field}")
    if dict(dataset_counts) != expected_counts.datasets:
        raise SelectorTrainingError("neutral source dataset case counts mismatch")
    manifest = _read_json(
        source_dir / "neutral_revision_manifest.json", "neutral revision manifest"
    )
    rows = manifest.get("examples")
    if not isinstance(rows, list) or len(rows) != len(all_examples):
        raise SelectorTrainingError("neutral revision manifest example count mismatch")
    manifest_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SelectorTrainingError(f"neutral manifest row {index} is invalid")
        manifest_ids.append(row.get("calibration_example_id"))
        if any(row.get(field) is not True for field in _MANIFEST_TRUE_FIELDS):
            raise SelectorTrainingError("neutral revision invariance manifest failed")
    if set(manifest_ids) != set(example_ids) or len(manifest_ids) != len(set(manifest_ids)):
        raise SelectorTrainingError("neutral revision manifest identities mismatch")
    positive_pairs = sum(sum(item.relevance_targets) for item in train)
    total_train_pairs = sum(len(item.relevance_targets) for item in train)
    negative_pairs = total_train_pairs - positive_pairs
    if positive_pairs <= 0 or negative_pairs <= 0:
        raise SelectorTrainingError("Train class counts must contain both classes")
    return NeutralDataBundle(
        source_dir=source_dir,
        train_examples=train,
        dev_examples=dev,
        source_hashes=actual_hashes,
        train_positive_pairs=positive_pairs,
        train_negative_pairs=negative_pairs,
    )


def _metric_selection_key(metrics: Mapping[str, Any], epoch: int) -> Tuple[float, ...]:
    overall = metrics["overall"]
    return (
        float(overall["mrr"]),
        float(overall["ndcg_at_5"]),
        float(overall["recall_at_5"]),
        -float(epoch),
    )


def _select_smoke_subset(
    examples: Sequence[CalibrationExample], size: int
) -> Tuple[CalibrationExample, ...]:
    ordered = sorted(
        examples,
        key=lambda item: (
            item.expected_modality,
            item.source_dataset,
            item.calibration_example_id,
        ),
    )
    if len(ordered) < size:
        raise SelectorTrainingError("neutral artifact is too small for smoke subset")
    return tuple(ordered[:size])


def _validate_seed_result(result: SeedBackendResult, expected_seed: int) -> None:
    if result.seed != expected_seed or result.selected_epoch < 1:
        raise SelectorTrainingError("backend returned invalid seed or selected epoch")
    if tuple(result.trainable_parameter_names) != TRAINABLE_PARAMETER_NAMES:
        raise SelectorTrainingError("trainable parameters are not selection-head-only")
    if tuple(result.optimizer_parameter_names) != TRAINABLE_PARAMETER_NAMES:
        raise SelectorTrainingError("optimizer contains non-selection parameters")
    if result.encoder_parameter_hash_before != result.encoder_parameter_hash_after:
        raise SelectorTrainingError("encoder parameters changed")
    if result.veracity_head_parameter_hash_before != result.veracity_head_parameter_hash_after:
        raise SelectorTrainingError("veracity head parameters changed")
    if result.selection_head_parameter_hash_before == result.selection_head_parameter_hash_after:
        raise SelectorTrainingError("selection head parameters did not change")
    if not result.loss_finite or not result.selection_scores_finite:
        raise SelectorTrainingError("training produced non-finite loss or scores")
    if not result.history or not result.dev_rankings:
        raise SelectorTrainingError("training history and Dev rankings are required")


def _verify_approved_smoke(
    path: Optional[Path], source_hashes: Mapping[str, str]
) -> Mapping[str, Any]:
    if path is None:
        raise SelectorTrainingError("--full requires --approved-smoke-report")
    resolved = _resolve_safe_path(path, "approved smoke report")
    report = _read_json(resolved, "approved smoke report")
    required = {
        "status": "PASS",
        "run_mode": "smoke",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "source_artifact_sha256": dict(source_hashes),
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise SelectorTrainingError(f"approved smoke report mismatch: {field}")
    for field in (
        "formal_validation_accessed",
        "formal_test_accessed",
        "step25b_heldout_accessed",
        "cpac_heldout_accessed",
        "production_or_model_code_changed",
        "public_demo_changed",
        "frozen_g1_checkpoint_modified",
        "sample_pooling_changed",
        "candidate_exposure_changed",
        "tokenizer_contract_changed",
        "veracity_labels_inspected",
    ):
        if report.get(field) is not False:
            raise SelectorTrainingError(f"approved smoke boundary failed: {field}")
    return report


def _write_json(path: Path, value: Any) -> str:
    content = _json_bytes(value)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def run_selector_calibration(
    *,
    source_dir: Path,
    output_dir: Path,
    run_mode: str,
    backend: Optional[SelectorTrainingBackend] = None,
    backend_factory: Optional[Callable[[], SelectorTrainingBackend]] = None,
    approved_smoke_report: Optional[Path] = None,
    protocol: TrainingProtocol = TrainingProtocol(),
    expected_hashes: Mapping[str, str] = AUTHORITATIVE_SOURCE_HASHES,
    expected_counts: ExpectedDataCounts = AUTHORITATIVE_COUNTS,
) -> Mapping[str, Any]:
    if run_mode not in {"smoke", "full"}:
        raise SelectorTrainingError("run_mode must be smoke or full")
    if (backend is None) == (backend_factory is None):
        raise SelectorTrainingError("provide exactly one training backend or factory")
    output_dir = _resolve_safe_path(output_dir, "selector training output")
    if output_dir.exists():
        raise SelectorTrainingError("selector training output must not already exist")
    data = load_neutral_data(
        source_dir,
        expected_hashes=expected_hashes,
        expected_counts=expected_counts,
    )
    if output_dir == data.source_dir or data.source_dir in output_dir.parents:
        raise SelectorTrainingError("selector output must be isolated from neutral source")
    if run_mode == "full":
        _verify_approved_smoke(approved_smoke_report, data.source_hashes)
        seeds = protocol.seeds
        maximum_epochs = protocol.maximum_epochs
        train_examples = data.train_examples
        dev_examples = data.dev_examples
    else:
        seeds = (42,)
        maximum_epochs = 1
        train_examples = _select_smoke_subset(data.train_examples, 8)
        dev_examples = _select_smoke_subset(data.dev_examples, 4)
    if backend is None:
        if backend_factory is None:  # pragma: no cover - guarded above
            raise SelectorTrainingError("training backend factory is missing")
        backend = backend_factory()
    if backend.checkpoint_sha256 != AUTHORITATIVE_CHECKPOINT_SHA256:
        raise SelectorTrainingError("Frozen G1 checkpoint SHA mismatch")
    effective_training_protocol = {
        "preregistered_full_protocol": dict(protocol.to_dict()),
        "run_mode": run_mode,
        "effective_seeds": list(seeds),
        "effective_maximum_epochs": maximum_epochs,
        "encoder_representation_mode": "eval_no_grad_cached",
        "encoder_precompute_batching": "one calibration example at a time",
        "selection_head_batch_size_examples": protocol.batch_size_examples,
        "collator_dummy_label_used": True,
        "collator_dummy_label_value": 0,
    }
    baseline_rankings = tuple(backend.baseline_rankings(dev_examples))
    baseline_metrics = grouped_ranking_metrics(baseline_rankings)
    if not finite_metrics(baseline_metrics):
        raise SelectorTrainingError("baseline ranking metrics are not finite")
    results = []
    for seed in seeds:
        result = backend.train_seed(
            seed=seed,
            train_examples=train_examples,
            dev_examples=dev_examples,
            pos_weight=data.pos_weight,
            maximum_epochs=maximum_epochs,
            protocol=protocol,
        )
        _validate_seed_result(result, seed)
        dev_metrics = grouped_ranking_metrics(result.dev_rankings)
        if not finite_metrics(dev_metrics):
            raise SelectorTrainingError("Dev ranking metrics are not finite")
        results.append((result, dev_metrics))
    if backend.current_checkpoint_sha256() != AUTHORITATIVE_CHECKPOINT_SHA256:
        raise SelectorTrainingError("Frozen G1 checkpoint changed during training")
    current_source_hashes = {
        name: sha256_file(data.source_dir / name) for name in data.source_hashes
    }
    if current_source_hashes != dict(data.source_hashes):
        raise SelectorTrainingError("neutral source changed during training")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".selector-training-", dir=str(output_dir.parent))
    )
    try:
        seed_reports = []
        artifact_hashes = {}
        for result, dev_metrics in results:
            seed_dir = staging / f"seed_{result.seed}"
            seed_dir.mkdir()
            history_name = f"seed_{result.seed}/training_history.json"
            metrics_name = f"seed_{result.seed}/dev_metrics.json"
            artifact_name = f"seed_{result.seed}/selector_head.pt"
            history_sha = _write_json(seed_dir / "training_history.json", list(result.history))
            metrics_sha = _write_json(seed_dir / "dev_metrics.json", dev_metrics)
            artifact_payload = {
                "selection_head_state_dict": result.selection_head_state_dict,
                "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
                "neutral_train_sha256": data.source_hashes[
                    "neutral_calibration_train.jsonl"
                ],
                "neutral_dev_sha256": data.source_hashes[
                    "neutral_calibration_dev.jsonl"
                ],
                "neutral_manifest_sha256": data.source_hashes[
                    "neutral_revision_manifest.json"
                ],
                "seed": result.seed,
                "selected_epoch": result.selected_epoch,
                "training_protocol": effective_training_protocol,
                "optimizer_protocol": {
                    "optimizer": protocol.optimizer,
                    "learning_rate": protocol.learning_rate,
                    "weight_decay": protocol.weight_decay,
                    "parameter_names": list(TRAINABLE_PARAMETER_NAMES),
                },
                "train_class_counts": {
                    "positive_pairs": data.train_positive_pairs,
                    "negative_pairs": data.train_negative_pairs,
                    "positive_prevalence": data.positive_prevalence,
                    "pos_weight": data.pos_weight,
                },
                "dev_metrics": dev_metrics,
                "implementation_revision": IMPLEMENTATION_REVISION,
            }
            backend.save_selector_artifact(seed_dir / "selector_head.pt", artifact_payload)
            artifact_hashes[history_name] = history_sha
            artifact_hashes[metrics_name] = metrics_sha
            artifact_hashes[artifact_name] = sha256_file(seed_dir / "selector_head.pt")
            seed_reports.append(
                {
                    "seed": result.seed,
                    "selected_epoch": result.selected_epoch,
                    "dev_metrics": dev_metrics,
                    "encoder_parameter_hash_before": result.encoder_parameter_hash_before,
                    "encoder_parameter_hash_after": result.encoder_parameter_hash_after,
                    "veracity_head_parameter_hash_before": result.veracity_head_parameter_hash_before,
                    "veracity_head_parameter_hash_after": result.veracity_head_parameter_hash_after,
                    "selection_head_parameter_hash_before": result.selection_head_parameter_hash_before,
                    "selection_head_parameter_hash_after": result.selection_head_parameter_hash_after,
                    "trainable_parameter_names": list(result.trainable_parameter_names),
                    "optimizer_parameter_names": list(result.optimizer_parameter_names),
                    "loss_finite": result.loss_finite,
                    "selection_scores_finite": result.selection_scores_finite,
                }
            )
        report = {
            "status": "PASS",
            "run_mode": run_mode,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "selector_id": SELECTOR_ID,
            "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
            "source_artifact_sha256": dict(data.source_hashes),
            "source_case_count": expected_counts.total_cases,
            "source_example_count": expected_counts.total_examples,
            "run_train_example_count": len(train_examples),
            "run_dev_example_count": len(dev_examples),
            "train_positive_pair_count": data.train_positive_pairs,
            "train_negative_pair_count": data.train_negative_pairs,
            "train_positive_prevalence": data.positive_prevalence,
            "train_pos_weight": data.pos_weight,
            "training_protocol": effective_training_protocol,
            "baseline_dev_metrics": baseline_metrics,
            "seed_results": seed_reports,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "encoder_weights_unchanged": True,
            "veracity_head_weights_unchanged": True,
            "only_selection_head_trainable": True,
            "optimizer_selection_head_only": True,
            "loss_finite": True,
            "selection_scores_finite": True,
            "ranking_metrics_finite": True,
            "neutral_train_dev_only": True,
            "collator_dummy_label_used": True,
            "collator_dummy_label_value": 0,
            "veracity_labels_inspected": False,
            "formal_validation_accessed": False,
            "formal_test_accessed": False,
            "step25b_heldout_accessed": False,
            "cpac_heldout_accessed": False,
            "production_or_model_code_changed": False,
            "public_demo_changed": False,
            "frozen_g1_checkpoint_modified": False,
            "sample_pooling_changed": False,
            "candidate_exposure_changed": False,
            "tokenizer_contract_changed": False,
            "full_training_automatically_triggered": False,
        }
        if run_mode == "full":
            summary = {
                name: mean_and_population_std(
                    metrics["overall"][name] for _, metrics in results
                )
                for name in METRIC_NAMES
            }
            chosen_result, chosen_metrics = max(
                results,
                key=lambda item: _metric_selection_key(
                    item[1], item[0].selected_epoch
                ),
            )
            multi_seed = {
                "seeds": list(seeds),
                "metrics": summary,
                "future_deployment_candidate_seed": chosen_result.seed,
                "selection_rule": (
                    "highest neutral-Dev MRR, then NDCG@5, Recall@5, earlier epoch"
                ),
                "candidate_dev_metrics": chosen_metrics,
            }
            summary_sha = _write_json(staging / "multi_seed_summary.json", multi_seed)
            report["artifact_sha256"]["multi_seed_summary.json"] = summary_sha
            report["multi_seed_summary"] = multi_seed
        report_name = "smoke_report.json" if run_mode == "smoke" else "training_report.json"
        report_sha = _write_json(staging / report_name, report)
        (staging / f"{report_name}.sha256").write_text(report_sha + "\n", encoding="utf-8")
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
