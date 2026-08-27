"""Lazy, read-only authoritative Frozen G1 evaluation runtime for DICC."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from scripts.selector_relevance_training.dicc_backend import (
    _checkpoint_path,
    _load_authoritative_runtime,
    _load_json,
)
from scripts.selector_relevance_training.metrics import METRIC_NAMES
from scripts.selector_relevance_training.trainer import (
    AUTHORITATIVE_CHECKPOINT_SHA256,
    IMPLEMENTATION_REVISION as TRAINING_IMPLEMENTATION_REVISION,
    SELECTOR_ID,
    load_neutral_data,
    sha256_file,
)

from .schemas import EvaluationRequest, PredictionSnapshot


DEPLOYMENT_CANDIDATE_SEED = 42
EXPECTED_SELECTOR_SHA256 = (
    "10cd426a97b61f14097145efcc3e67ca4eb381b7d4c6588a3c733c5955cb7687"
)
_SELECTOR_ARTIFACT_FIELDS = {
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
_ALLOWED_STATE_DIFFERENCES = {
    "selection_head.weight",
    "selection_head.bias",
}


class RuntimeIntegrationError(RuntimeError):
    """Raised when authoritative evaluation cannot preserve its frozen contract."""


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrationError(f"{field} is unavailable or malformed") from exc
    if not isinstance(value, Mapping):
        raise RuntimeIntegrationError(f"{field} must be an object")
    return value


def _reject_formal_path(path: Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() in {"validation", "test"} for part in resolved.parts):
        raise RuntimeIntegrationError(f"{field} cannot access Formal Validation/Test")
    return resolved


def _verify_json_sidecar(path: Path) -> str:
    actual = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != actual:
        raise RuntimeIntegrationError(f"SHA sidecar mismatch for {path.name}")
    return actual


@dataclass(frozen=True)
class TrainingArtifacts:
    training_dir: Path
    selector_path: Path
    selector_sha256: str
    training_report: Mapping[str, Any]
    neutral_source_hashes: Mapping[str, str]
    calibration_case_ids: Tuple[str, ...]
    immutable_file_hashes: Mapping[Path, str]


def validate_training_artifacts(
    training_dir: Path,
    neutral_dir: Path,
) -> TrainingArtifacts:
    training_root = _reject_formal_path(training_dir, "training artifact root")
    neutral_root = _reject_formal_path(neutral_dir, "neutral calibration root")
    if not training_root.is_dir():
        raise RuntimeIntegrationError("full training artifact root is missing")
    try:
        neutral = load_neutral_data(neutral_root)
    except Exception as exc:
        raise RuntimeIntegrationError("closed neutral calibration artifact is invalid") from exc
    report_path = training_root / "training_report.json"
    summary_path = training_root / "multi_seed_summary.json"
    selector_path = training_root / "seed_42" / "selector_head.pt"
    for path in (report_path, summary_path, selector_path):
        if not path.is_file():
            raise RuntimeIntegrationError(f"required training artifact is missing: {path.name}")
    _verify_json_sidecar(report_path)
    report = _read_json(report_path, "training report")
    summary = _read_json(summary_path, "multi-seed summary")
    required_report = {
        "status": "PASS",
        "run_mode": "full",
        "implementation_revision": TRAINING_IMPLEMENTATION_REVISION,
        "selector_id": SELECTOR_ID,
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "source_artifact_sha256": dict(neutral.source_hashes),
    }
    for field, expected in required_report.items():
        if report.get(field) != expected:
            raise RuntimeIntegrationError(f"training report mismatch: {field}")
    for field in (
        "formal_validation_accessed",
        "formal_test_accessed",
        "step25b_heldout_accessed",
        "cpac_heldout_accessed",
        "veracity_labels_inspected",
        "production_or_model_code_changed",
        "public_demo_changed",
    ):
        if report.get(field) is not False:
            raise RuntimeIntegrationError(f"training report boundary failed: {field}")
    if summary != report.get("multi_seed_summary"):
        raise RuntimeIntegrationError("multi-seed summary disagrees with training report")
    if summary.get("seeds") != [42, 43, 44]:
        raise RuntimeIntegrationError("multi-seed summary does not contain preregistered seeds")
    if summary.get("future_deployment_candidate_seed") != DEPLOYMENT_CANDIDATE_SEED:
        raise RuntimeIntegrationError("deployment candidate seed is not 42")
    metric_summary = summary.get("metrics")
    if not isinstance(metric_summary, Mapping):
        raise RuntimeIntegrationError("multi-seed metric summary is missing")
    for name in METRIC_NAMES:
        values = metric_summary.get(name)
        if not isinstance(values, Mapping) or float(values.get("std", -1.0)) != 0.0:
            raise RuntimeIntegrationError("multi-seed internal metrics are not identical")
    declared = report.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        raise RuntimeIntegrationError("training artifact hashes are missing")
    summary_sha = sha256_file(summary_path)
    selector_sha = sha256_file(selector_path)
    if declared.get("multi_seed_summary.json") != summary_sha:
        raise RuntimeIntegrationError("multi-seed summary SHA mismatch")
    if declared.get("seed_42/selector_head.pt") != selector_sha:
        raise RuntimeIntegrationError("selector SHA disagrees with training report")
    if selector_sha != EXPECTED_SELECTOR_SHA256:
        raise RuntimeIntegrationError("deployment selector SHA mismatch")
    calibration_case_ids = tuple(
        sorted(
            {
                item.canonical_underlying_case_id
                for item in neutral.train_examples + neutral.dev_examples
            }
        )
    )
    return TrainingArtifacts(
        training_dir=training_root,
        selector_path=selector_path,
        selector_sha256=selector_sha,
        training_report=report,
        neutral_source_hashes=dict(neutral.source_hashes),
        calibration_case_ids=calibration_case_ids,
        immutable_file_hashes={
            report_path: sha256_file(report_path),
            summary_path: summary_sha,
            selector_path: selector_sha,
            **{
                neutral.source_dir / name: digest
                for name, digest in neutral.source_hashes.items()
            },
        },
    )


def _tensor_hash(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_hashes(model: Any) -> Mapping[str, str]:
    return {name: _tensor_hash(tensor) for name, tensor in model.state_dict().items()}


def _module_hash(state_hashes: Mapping[str, str], prefix: str) -> str:
    selected = tuple(
        (name, value)
        for name, value in sorted(state_hashes.items())
        if name.startswith(prefix)
    )
    if not selected:
        raise RuntimeIntegrationError(f"model state has no {prefix} tensors")
    digest = hashlib.sha256()
    for name, value in selected:
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _safe_torch_load(torch: Any, path: Path) -> Mapping[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError) as exc:
        raise RuntimeIntegrationError("Torch load signature cannot be inspected") from exc
    if "weights_only" in signature.parameters:
        kwargs["weights_only"] = True
    try:
        payload = torch.load(path, **kwargs)
    except Exception as exc:
        raise RuntimeIntegrationError("selector artifact cannot be safely loaded") from exc
    if not isinstance(payload, Mapping) or set(payload) != _SELECTOR_ARTIFACT_FIELDS:
        raise RuntimeIntegrationError("selector artifact contains unexpected fields")
    return payload


def _validated_selector_state(
    payload: Mapping[str, Any],
    *,
    original_selector_state: Mapping[str, Any],
    torch: Any,
    training_artifacts: TrainingArtifacts,
) -> Mapping[str, Any]:
    required_metadata = {
        "base_frozen_g1_checkpoint_sha256": AUTHORITATIVE_CHECKPOINT_SHA256,
        "seed": DEPLOYMENT_CANDIDATE_SEED,
        "implementation_revision": TRAINING_IMPLEMENTATION_REVISION,
    }
    for field, expected in required_metadata.items():
        if payload.get(field) != expected:
            raise RuntimeIntegrationError(f"selector artifact mismatch: {field}")
    source_hashes = training_artifacts.neutral_source_hashes
    if (
        payload.get("neutral_train_sha256")
        != source_hashes["neutral_calibration_train.jsonl"]
        or payload.get("neutral_dev_sha256")
        != source_hashes["neutral_calibration_dev.jsonl"]
        or payload.get("neutral_manifest_sha256")
        != source_hashes["neutral_revision_manifest.json"]
    ):
        raise RuntimeIntegrationError("selector neutral-source provenance mismatch")
    selector_state = payload.get("selection_head_state_dict")
    if not isinstance(selector_state, Mapping) or set(selector_state) != {
        "weight",
        "bias",
    }:
        raise RuntimeIntegrationError("selector state must contain only weight and bias")
    for name in ("weight", "bias"):
        candidate = selector_state[name]
        original = original_selector_state[name]
        if tuple(candidate.shape) != tuple(original.shape):
            raise RuntimeIntegrationError(f"selector tensor shape mismatch: {name}")
        if not bool(torch.isfinite(candidate).all().item()):
            raise RuntimeIntegrationError(f"selector tensor is non-finite: {name}")
    return {
        name: selector_state[name].detach().cpu().clone()
        for name in ("weight", "bias")
    }


def _validate_state_difference(
    original_hashes: Mapping[str, str], calibrated_hashes: Mapping[str, str]
) -> Tuple[str, ...]:
    if set(calibrated_hashes) != set(original_hashes):
        raise RuntimeIntegrationError("calibrated model state keys changed")
    differences = {
        name
        for name in original_hashes
        if calibrated_hashes[name] != original_hashes[name]
    }
    if not differences or not differences <= _ALLOWED_STATE_DIFFERENCES:
        raise RuntimeIntegrationError("non-selection model tensor difference detected")
    return tuple(sorted(differences))


class DICCEvaluationRuntime:
    """Evaluate original/calibrated selector states without gradients or training."""

    def __init__(
        self,
        *,
        project_root: Path,
        phase4a_config_path: Path,
        training_artifacts: TrainingArtifacts,
        device: str,
    ) -> None:
        self.project_root = _reject_formal_path(project_root, "project root")
        self.phase4a_config_path = _reject_formal_path(
            phase4a_config_path, "Phase4A config"
        )
        if not self.project_root.is_dir() or not self.phase4a_config_path.is_file():
            raise RuntimeIntegrationError("project root or Phase4A config is missing")
        if not isinstance(device, str) or not device.strip():
            raise RuntimeIntegrationError("device must be nonblank")
        self.device = device.strip()
        if self.device.startswith("cuda") and os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ) != ":4096:8":
            raise RuntimeIntegrationError(
                "CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        self.training_artifacts = training_artifacts
        try:
            import torch
        except ImportError as exc:
            raise RuntimeIntegrationError("DICC evaluation requires installed Torch") from exc
        self.torch = torch
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        config = _load_json(self.phase4a_config_path)
        self.checkpoint_path = _checkpoint_path(config, self.project_root)
        self._checkpoint_sha_before = sha256_file(self.checkpoint_path)
        if self._checkpoint_sha_before != AUTHORITATIVE_CHECKPOINT_SHA256:
            raise RuntimeIntegrationError("Frozen G1 checkpoint SHA mismatch")
        model, self.tokenizer, self.collate = _load_authoritative_runtime(
            self.project_root, self.phase4a_config_path, config
        )
        self.model = getattr(model, "module", model)
        for attribute in ("encoder", "veracity_head", "selection_head"):
            if not hasattr(self.model, attribute):
                raise RuntimeIntegrationError(f"Frozen G1 model is missing {attribute}")
        self.model.to(self.device)
        self.model.eval()
        self._selector_sha_before = sha256_file(training_artifacts.selector_path)
        payload = _safe_torch_load(torch, training_artifacts.selector_path)
        original_selector = self.model.selection_head.state_dict()
        self._original_selector_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in original_selector.items()
        }
        self._calibrated_selector_state = _validated_selector_state(
            payload,
            original_selector_state=original_selector,
            torch=torch,
            training_artifacts=training_artifacts,
        )
        self._original_state_hashes = _state_hashes(self.model)
        self.model.selection_head.load_state_dict(
            self._calibrated_selector_state, strict=True
        )
        calibrated_hashes = _state_hashes(self.model)
        self.state_difference_names = _validate_state_difference(
            self._original_state_hashes, calibrated_hashes
        )
        self.encoder_hash = _module_hash(self._original_state_hashes, "encoder.")
        self.veracity_head_hash = _module_hash(
            self._original_state_hashes, "veracity_head."
        )
        self.original_selection_head_hash = _module_hash(
            self._original_state_hashes, "selection_head."
        )
        self.calibrated_selection_head_hash = _module_hash(
            calibrated_hashes, "selection_head."
        )
        self.model.selection_head.load_state_dict(self._original_selector_state, strict=True)

    def _activate(self, state: str) -> None:
        if state == "original":
            selected = self._original_selector_state
        elif state == "calibrated":
            selected = self._calibrated_selector_state
        else:
            raise RuntimeIntegrationError("evaluation state must be original or calibrated")
        self.model.selection_head.load_state_dict(selected, strict=True)
        self.model.eval()

    def evaluate(self, request: EvaluationRequest, *, state: str) -> PredictionSnapshot:
        self._activate(state)
        with self.torch.no_grad():
            batch = self.collate([request.collator_item()])
            input_ids = getattr(batch, "input_ids", None)
            attention_mask = getattr(batch, "attention_mask", None)
            if input_ids is None or attention_mask is None:
                raise RuntimeIntegrationError("authoritative Batch tensors are missing")
            unit_ids = tuple(getattr(batch, "unit_ids", ()))
            if unit_ids != request.candidate_unit_ids:
                raise RuntimeIntegrationError("authoritative collator changed candidate order")
            encoded = self.model.encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            hidden = getattr(encoded, "last_hidden_state", None)
            if hidden is None or hidden.shape[0] != len(request.candidate_units):
                raise RuntimeIntegrationError("Frozen encoder representation shape mismatch")
            representations = hidden[:, 0]
            selection = self.model.selection_head(representations).squeeze(-1)
            unit_logits = self.model.veracity_head(representations)
            if tuple(unit_logits.shape) != (len(request.candidate_units), 2):
                raise RuntimeIntegrationError("Frozen veracity output shape mismatch")
            if not bool(self.torch.isfinite(selection).all().item()) or not bool(
                self.torch.isfinite(unit_logits).all().item()
            ):
                raise RuntimeIntegrationError("evaluation produced non-finite unit outputs")
            sample_logits = unit_logits.max(dim=0).values
            probabilities = self.torch.softmax(sample_logits, dim=0)
            prediction_index = int(self.torch.argmax(sample_logits).item())
            scores = tuple(float(value) for value in selection.detach().cpu().tolist())
            logits = tuple(
                (float(pair[0]), float(pair[1]))
                for pair in unit_logits.detach().cpu().tolist()
            )
            sample = tuple(
                float(value) for value in sample_logits.detach().cpu().tolist()
            )
            probs = tuple(float(value) for value in probabilities.detach().cpu().tolist())
        ranking = sorted(
            range(len(scores)), key=lambda index: (-scores[index], index)
        )
        return PredictionSnapshot(
            candidate_unit_ids=request.candidate_unit_ids,
            selection_scores=scores,
            unit_veracity_logits=logits,
            sample_logits=(sample[0], sample[1]),
            probabilities=(probs[0], probs[1]),
            prediction=("fake", "real")[prediction_index],
            top_k_unit_ids=tuple(
                request.candidate_unit_ids[index] for index in ranking[:5]
            ),
        )

    def assert_immutable(self) -> None:
        if sha256_file(self.checkpoint_path) != self._checkpoint_sha_before:
            raise RuntimeIntegrationError("Frozen G1 checkpoint changed during evaluation")
        if sha256_file(self.training_artifacts.selector_path) != self._selector_sha_before:
            raise RuntimeIntegrationError("selector artifact changed during evaluation")
        current = _state_hashes(self.model)
        if _module_hash(current, "encoder.") != self.encoder_hash:
            raise RuntimeIntegrationError("encoder state changed during evaluation")
        if _module_hash(current, "veracity_head.") != self.veracity_head_hash:
            raise RuntimeIntegrationError("veracity-head state changed during evaluation")
