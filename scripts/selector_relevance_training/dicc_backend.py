"""Lazy DICC backend for Frozen-G1 selector-head-only calibration.

Importing this module does not import Torch, Transformers, or Frozen G1.  Those
dependencies are loaded only when ``DICCTorchBackend`` is explicitly created by
the training CLI on DICC.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .metrics import RankingExample, grouped_ranking_metrics
from .trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    CalibrationExample,
    SeedBackendResult,
    SelectorTrainingError,
    TRAINABLE_PARAMETER_NAMES,
    TrainingProtocol,
    _metric_selection_key,
    _resolve_safe_path,
    sha256_file,
)


MODEL_NAME = "microsoft/deberta-v3-base"
MAXIMUM_UNITS_PER_SAMPLE = 24
MAX_LENGTH = 256
POOLING = "max"


def configure_parameter_boundary(model: Any) -> Tuple[str, ...]:
    """Freeze every parameter except the exact selection-head weight and bias."""

    named = tuple(model.named_parameters())
    names = tuple(name for name, _ in named)
    if set(names) != {
        *TRAINABLE_PARAMETER_NAMES,
        *(name for name in names if name.startswith("encoder.")),
        *(name for name in names if name.startswith("veracity_head.")),
    }:
        raise SelectorTrainingError("model contains parameters outside frozen G1 heads")
    if not any(name.startswith("encoder.") for name in names):
        raise SelectorTrainingError("Frozen G1 encoder parameters are missing")
    if set(name for name in names if name.startswith("veracity_head.")) != {
        "veracity_head.weight",
        "veracity_head.bias",
    }:
        raise SelectorTrainingError("Frozen G1 veracity head structure changed")
    if not all(name in names for name in TRAINABLE_PARAMETER_NAMES):
        raise SelectorTrainingError("Frozen G1 selection head structure changed")
    for name, parameter in named:
        parameter.requires_grad = name in TRAINABLE_PARAMETER_NAMES
    trainable = tuple(name for name, parameter in named if parameter.requires_grad)
    if trainable != TRAINABLE_PARAMETER_NAMES:
        raise SelectorTrainingError("only selection_head.weight/bias may be trainable")
    model.encoder.eval()
    model.veracity_head.eval()
    model.selection_head.train()
    return trainable


def validate_optimizer_boundary(
    model: Any, optimizer: Any
) -> Tuple[str, ...]:
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_names = []
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            name = names_by_identity.get(id(parameter))
            if name is None:
                raise SelectorTrainingError("optimizer contains an unknown parameter")
            optimizer_names.append(name)
    if tuple(optimizer_names) != TRAINABLE_PARAMETER_NAMES:
        raise SelectorTrainingError("optimizer must contain only selection head parameters")
    return tuple(optimizer_names)


def _recursive_values(payload: Any, aliases: set[str]) -> list[Any]:
    values = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).casefold() in aliases:
                values.append(value)
            values.extend(_recursive_values(value, aliases))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_recursive_values(item, aliases))
    return values


def _require_config_value(
    config: Mapping[str, Any], aliases: set[str], expected: Any, field: str
) -> None:
    values = _recursive_values(config, aliases)
    if expected not in values:
        raise SelectorTrainingError(f"Phase4A config does not freeze {field}={expected!r}")


def _checkpoint_path(config: Mapping[str, Any], project_root: Path) -> Path:
    candidates = []

    def visit(value: Any, checkpoint_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).casefold()
                is_checkpoint = "checkpoint" in normalized
                if isinstance(item, str) and (
                    is_checkpoint and "sha" not in normalized
                ):
                    candidates.append(item)
                elif checkpoint_context and normalized in {"path", "file"} and isinstance(item, str):
                    candidates.append(item)
                visit(item, checkpoint_context or is_checkpoint)
        elif isinstance(value, list):
            for item in value:
                visit(item, checkpoint_context)

    visit(config)
    unique = []
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved = path.resolve()
        if resolved not in unique:
            unique.append(resolved)
    existing = [path for path in unique if path.is_file()]
    if len(existing) != 1:
        raise SelectorTrainingError(
            "Phase4A config must resolve exactly one existing Frozen G1 checkpoint"
        )
    return existing[0]


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectorTrainingError("Phase4A config is unavailable or malformed") from exc
    if not isinstance(value, dict):
        raise SelectorTrainingError("Phase4A config must be a JSON object")
    return value


def _construct_engine(
    engine_type: Any,
    *,
    config: Mapping[str, Any],
    project_root: Path,
) -> Any:
    try:
        signature = inspect.signature(engine_type)
    except (TypeError, ValueError) as exc:
        raise SelectorTrainingError("FrozenG1Engine signature cannot be inspected") from exc
    args = (config, project_root)
    kwargs = {"device_name": "cpu"}
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise SelectorTrainingError(
            "FrozenG1Engine constructor must accept "
            "(config, project_root, *, device_name)"
        ) from exc
    try:
        return engine_type(*args, **kwargs)
    except Exception as exc:
        raise SelectorTrainingError("FrozenG1Engine initialization failed") from exc


def _load_authoritative_runtime(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> Tuple[Any, Any, Any]:
    phase3_dir = project_root / "MDU" / "scripts" / "clip12_phase3_common"
    phase4a_dir = project_root / "MDU" / "scripts" / "clip12_phase4a_inference_handoff"
    model_path = phase3_dir / "clip12p3_model.py"
    engine_path = phase4a_dir / "clip12p4a_engine.py"
    for path in (model_path, engine_path):
        if not path.is_file():
            raise SelectorTrainingError(f"authoritative Frozen G1 source is missing: {path.name}")
    original_path = list(sys.path)
    isolated_names = (
        "clip12p3_common",
        "clip12p3_model",
        "clip12p4a_common",
        "_selector_training_phase4a_engine",
    )
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in isolated_names}
    try:
        sys.path[:] = [str(phase3_dir), str(phase4a_dir)] + [
            entry
            for entry in original_path
            if entry not in {str(phase3_dir), str(phase4a_dir)}
        ]
        for name in isolated_names:
            sys.modules.pop(name, None)
        model_spec = importlib.util.spec_from_file_location("clip12p3_model", model_path)
        if model_spec is None or model_spec.loader is None:
            raise SelectorTrainingError("cannot load authoritative Phase3A model module")
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules["clip12p3_model"] = model_module
        model_spec.loader.exec_module(model_module)
        collator_factory = getattr(model_module, "collator", None)
        if not callable(collator_factory):
            raise SelectorTrainingError("authoritative Phase3A collator is unavailable")
        engine_spec = importlib.util.spec_from_file_location(
            "_selector_training_phase4a_engine", engine_path
        )
        if engine_spec is None or engine_spec.loader is None:
            raise SelectorTrainingError("cannot load authoritative Phase4A engine module")
        engine_module = importlib.util.module_from_spec(engine_spec)
        sys.modules["_selector_training_phase4a_engine"] = engine_module
        engine_spec.loader.exec_module(engine_module)
        engine_type = getattr(engine_module, "FrozenG1Engine", None)
        if engine_type is None:
            raise SelectorTrainingError("authoritative FrozenG1Engine is unavailable")
        engine = _construct_engine(
            engine_type,
            config=config,
            project_root=project_root,
        )
        model = getattr(engine, "model", getattr(engine, "_model", None))
        tokenizer = getattr(engine, "tokenizer", getattr(engine, "_tokenizer", None))
        if model is None or tokenizer is None:
            raise SelectorTrainingError("Phase4A engine does not expose model/tokenizer")
        collate = collator_factory(tokenizer, MAX_LENGTH)
        if not callable(collate):
            raise SelectorTrainingError("authoritative collator factory returned non-callable")
        return model, tokenizer, collate
    except SelectorTrainingError:
        raise
    except Exception as exc:
        raise SelectorTrainingError("cannot load authoritative Frozen G1 runtime") from exc
    finally:
        sys.path[:] = original_path
        for name, prior in original_modules.items():
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _module_parameter_hash(module: Any, torch: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    named = tuple(sorted(module.named_parameters(), key=lambda item: item[0]))
    if not named:
        raise SelectorTrainingError("parameter hash requires at least one tensor")
    for name, parameter in named:
        tensor = parameter.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class _CachedExample:
    example: CalibrationExample
    representations: Any


def _authoritative_collator_item(
    example: CalibrationExample,
) -> Mapping[str, Any]:
    item = dict(example.collator_item())
    item["case_id"] = example.calibration_example_id
    item["label"] = 0
    return item


def _authoritative_batch_inputs(
    batch: Any,
    *,
    example: CalibrationExample,
    device: str,
) -> Mapping[str, Any]:
    input_ids = getattr(batch, "input_ids", None)
    attention_mask = getattr(batch, "attention_mask", None)
    if input_ids is None:
        raise SelectorTrainingError("authoritative collator Batch.input_ids is missing")
    if attention_mask is None:
        raise SelectorTrainingError(
            "authoritative collator Batch.attention_mask is missing"
        )
    unit_ids = tuple(getattr(batch, "unit_ids", ()))
    if unit_ids != example.candidate_unit_ids:
        raise SelectorTrainingError("authoritative collator changed candidate order")
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _move_model_to_training_device(model: Any, device: str) -> None:
    model.to(device)


class DICCTorchBackend:
    """Train only the real Frozen-G1 selection head over cached representations."""

    def __init__(
        self,
        *,
        project_root: Path,
        phase4a_config_path: Path,
        device: str,
    ) -> None:
        self.project_root = _resolve_safe_path(project_root, "DICC project root")
        self.phase4a_config_path = _resolve_safe_path(
            phase4a_config_path, "Phase4A config"
        )
        if not self.project_root.is_dir() or not self.phase4a_config_path.is_file():
            raise SelectorTrainingError("DICC project root or Phase4A config is missing")
        if not isinstance(device, str) or not device.strip():
            raise SelectorTrainingError("device must be nonblank")
        self.device = device.strip()
        try:
            import torch
        except ImportError as exc:
            raise SelectorTrainingError("DICC selector training requires installed Torch") from exc
        self.torch = torch
        self.config = _load_json(self.phase4a_config_path)
        _require_config_value(
            self.config, {"model_name", "backbone"}, MODEL_NAME, "model_name"
        )
        _require_config_value(
            self.config,
            {"maximum_units_per_sample", "max_units"},
            MAXIMUM_UNITS_PER_SAMPLE,
            "maximum_units_per_sample",
        )
        _require_config_value(self.config, {"max_length"}, MAX_LENGTH, "max_length")
        _require_config_value(self.config, {"pooling"}, POOLING, "pooling")
        self.checkpoint_path = _checkpoint_path(self.config, self.project_root)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if self.checkpoint_sha256 != AUTHORITATIVE_CHECKPOINT_SHA256:
            raise SelectorTrainingError("Frozen G1 checkpoint SHA mismatch")
        model, self.tokenizer, self.collate = _load_authoritative_runtime(
            self.project_root,
            self.phase4a_config_path,
            self.config,
        )
        self.model = getattr(model, "module", model)
        for attribute in ("encoder", "veracity_head", "selection_head"):
            if not hasattr(self.model, attribute):
                raise SelectorTrainingError(f"Frozen G1 model is missing {attribute}")
        _move_model_to_training_device(self.model, self.device)
        configure_parameter_boundary(self.model)
        self._initial_selection_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.selection_head.state_dict().items()
        }
        if set(self._initial_selection_state) != {"weight", "bias"}:
            raise SelectorTrainingError("selection head state must contain only weight/bias")
        self._encoder_hash = _module_parameter_hash(self.model.encoder, self.torch)
        self._veracity_hash = _module_parameter_hash(self.model.veracity_head, self.torch)
        self._selection_hash = _module_parameter_hash(self.model.selection_head, self.torch)
        self._cache: Dict[str, _CachedExample] = {}

    def _restore_selection_head(self) -> None:
        self.model.selection_head.load_state_dict(self._initial_selection_state, strict=True)
        configure_parameter_boundary(self.model)

    def _prepare(self, examples: Sequence[CalibrationExample]) -> Tuple[_CachedExample, ...]:
        prepared = []
        self.model.encoder.eval()
        with self.torch.no_grad():
            for example in examples:
                cached = self._cache.get(example.calibration_example_id)
                if cached is None:
                    batch = self.collate([_authoritative_collator_item(example)])
                    device_inputs = _authoritative_batch_inputs(
                        batch,
                        example=example,
                        device=self.device,
                    )
                    output = self.model.encoder(**device_inputs)
                    hidden = getattr(output, "last_hidden_state", None)
                    if hidden is None or hidden.shape[0] != len(example.candidate_units):
                        raise SelectorTrainingError("Frozen encoder representation shape mismatch")
                    representations = hidden[:, 0].detach().cpu()
                    cached = _CachedExample(example=example, representations=representations)
                    self._cache[example.calibration_example_id] = cached
                prepared.append(cached)
        return tuple(prepared)

    def _rankings(self, prepared: Sequence[_CachedExample]) -> Tuple[RankingExample, ...]:
        rankings = []
        self.model.selection_head.eval()
        with self.torch.no_grad():
            for item in prepared:
                scores = self.model.selection_head(
                    item.representations.to(self.device)
                ).squeeze(-1)
                if not bool(self.torch.isfinite(scores).all().item()):
                    raise SelectorTrainingError("selection scores are not finite")
                rankings.append(
                    RankingExample(
                        calibration_example_id=item.example.calibration_example_id,
                        source_dataset=item.example.source_dataset,
                        expected_modality=item.example.expected_modality,
                        candidate_unit_ids=item.example.candidate_unit_ids,
                        relevance_targets=item.example.relevance_targets,
                        selection_scores=tuple(float(value) for value in scores.cpu().tolist()),
                    )
                )
        return tuple(rankings)

    def baseline_rankings(
        self, dev_examples: Sequence[CalibrationExample]
    ) -> Sequence[RankingExample]:
        self._restore_selection_head()
        return self._rankings(self._prepare(dev_examples))

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.torch.use_deterministic_algorithms(True)
        if hasattr(self.torch.backends, "cudnn"):
            self.torch.backends.cudnn.benchmark = False
            self.torch.backends.cudnn.deterministic = True

    def train_seed(
        self,
        *,
        seed: int,
        train_examples: Sequence[CalibrationExample],
        dev_examples: Sequence[CalibrationExample],
        pos_weight: float,
        maximum_epochs: int,
        protocol: TrainingProtocol,
    ) -> SeedBackendResult:
        self._seed_everything(seed)
        self._restore_selection_head()
        trainable_names = configure_parameter_boundary(self.model)
        encoder_before = _module_parameter_hash(self.model.encoder, self.torch)
        veracity_before = _module_parameter_hash(self.model.veracity_head, self.torch)
        selection_before = _module_parameter_hash(self.model.selection_head, self.torch)
        train_cache = self._prepare(train_examples)
        dev_cache = self._prepare(dev_examples)
        optimizer = self.torch.optim.AdamW(
            self.model.selection_head.parameters(),
            lr=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
        )
        optimizer_names = validate_optimizer_boundary(self.model, optimizer)
        loss_function = self.torch.nn.BCEWithLogitsLoss(
            pos_weight=self.torch.tensor([pos_weight], device=self.device)
        )
        best_key = None
        best_epoch = 0
        best_state = None
        best_rankings = None
        history = []
        epochs_without_improvement = 0
        loss_finite = True
        for epoch in range(1, maximum_epochs + 1):
            order = list(range(len(train_cache)))
            random.Random(seed * 1000 + epoch).shuffle(order)
            losses = []
            self.model.selection_head.train()
            for start in range(0, len(order), protocol.batch_size_examples):
                items = [train_cache[index] for index in order[start : start + protocol.batch_size_examples]]
                representations = self.torch.cat(
                    [item.representations for item in items], dim=0
                ).to(self.device)
                targets = self.torch.tensor(
                    [
                        target
                        for item in items
                        for target in item.example.relevance_targets
                    ],
                    dtype=self.torch.float32,
                    device=self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                scores = self.model.selection_head(representations).squeeze(-1)
                loss = loss_function(scores, targets)
                if not bool(self.torch.isfinite(loss).item()) or not bool(
                    self.torch.isfinite(scores).all().item()
                ):
                    loss_finite = False
                    raise SelectorTrainingError("selector training produced non-finite values")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            rankings = self._rankings(dev_cache)
            metrics = grouped_ranking_metrics(rankings)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "dev_metrics": metrics,
                }
            )
            key = _metric_selection_key(metrics, epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_rankings = rankings
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.selection_head.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= protocol.early_stopping_patience:
                    break
        if best_state is None or best_rankings is None:
            raise SelectorTrainingError("selector training did not select an epoch")
        self.model.selection_head.load_state_dict(best_state, strict=True)
        final_rankings = self._rankings(dev_cache)
        if final_rankings != best_rankings:
            raise SelectorTrainingError("selected selector state does not reproduce Dev rankings")
        encoder_after = _module_parameter_hash(self.model.encoder, self.torch)
        veracity_after = _module_parameter_hash(self.model.veracity_head, self.torch)
        selection_after = _module_parameter_hash(self.model.selection_head, self.torch)
        return SeedBackendResult(
            seed=seed,
            selected_epoch=best_epoch,
            history=tuple(history),
            dev_rankings=tuple(final_rankings),
            selection_head_state_dict=best_state,
            encoder_parameter_hash_before=encoder_before,
            encoder_parameter_hash_after=encoder_after,
            veracity_head_parameter_hash_before=veracity_before,
            veracity_head_parameter_hash_after=veracity_after,
            selection_head_parameter_hash_before=selection_before,
            selection_head_parameter_hash_after=selection_after,
            trainable_parameter_names=trainable_names,
            optimizer_parameter_names=optimizer_names,
            loss_finite=loss_finite and all(
                math.isfinite(float(item["train_loss"])) for item in history
            ),
            selection_scores_finite=True,
        )

    def save_selector_artifact(self, path: Path, payload: Mapping[str, Any]) -> None:
        expected_fields = {
            "selection_head_state_dict",
            "base_frozen_g1_checkpoint_sha256",
            "neutral_train_sha256",
            "neutral_dev_sha256",
            "neutral_manifest_sha256",
            "seed",
            "selected_epoch",
            "training_protocol",
            "optimizer_protocol",
            "train_class_counts",
            "dev_metrics",
            "implementation_revision",
        }
        if set(payload) != expected_fields:
            raise SelectorTrainingError("selector artifact contains unexpected fields")
        state = payload["selection_head_state_dict"]
        if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
            raise SelectorTrainingError("selector artifact state must contain only weight/bias")
        self.torch.save(dict(payload), path)

    def current_checkpoint_sha256(self) -> str:
        return sha256_file(self.checkpoint_path)
