"""Claim-blind local Qwen2.5-VL visual observation with grounded filtering."""

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.siglip_visual_retriever import (
    VisualFrame,
    runtime_tree_sha256,
)


QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_FROZEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN_RUNTIME_TREE_SHA256 = (
    "d9ba83a668c72098d9e952eac7da926164c6a87c2c46bbc9a8ffe7306945ee87"
)
QWEN_RUNTIME_TREE_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
ALLOWED_OBSERVATION_TYPES = {
    "entity",
    "action",
    "scene",
    "object_state",
    "spatial_relation",
    "temporal_change",
}
RISKY_PHRASES = (
    "probably",
    "likely",
    "seems",
    "appears",
    "appears to",
    "suggests",
    "because",
    "causes",
    "intends",
    "wants",
    "the text reads",
    "the text says",
    "subtitle",
    "logo",
    "watermark",
    "misinformation",
    "fake",
    "false",
    "true",
    "veracity",
    "claim",
    "proves",
    "disproves",
    "contradicts",
    "supports",
)
PROMPT_POLICY = "claim_blind_visible_atomic_facts_no_ocr_no_inference"


@dataclass(frozen=True)
class QwenVisualObserverConfig:
    model_path: Path
    device: str = "cuda:0"
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if not self.device:
            raise ValueError("Qwen visual observer device is required")
        if self.max_new_tokens != 512:
            raise ValueError("frozen Qwen max_new_tokens must equal 512")


@dataclass
class QwenVisualObservationResult:
    observations: List[Dict[str, Any]]
    recovery_mode: str
    raw_generation_sha256: str
    rejected_observation_count: int
    raw_generation: str = ""
    rejected_observations: List[Dict[str, Any]] = field(default_factory=list)
    model_id: str = QWEN_MODEL_ID
    frozen_revision: str = QWEN_FROZEN_REVISION
    runtime_tree_sha256: str = QWEN_RUNTIME_TREE_SHA256
    prompt_policy: str = PROMPT_POLICY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": [dict(item) for item in self.observations],
            "recovery_mode": self.recovery_mode,
            "raw_generation": self.raw_generation,
            "raw_generation_sha256": self.raw_generation_sha256,
            "rejected_observation_count": self.rejected_observation_count,
            "rejected_observations": [
                dict(item) for item in self.rejected_observations
            ],
            "model_id": self.model_id,
            "frozen_revision": self.frozen_revision,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "prompt_policy": self.prompt_policy,
        }


class QwenVisualObserver:
    def __init__(
        self,
        config: QwenVisualObserverConfig,
        transformers_module: Any = None,
        torch_module: Any = None,
        process_vision_info: Any = None,
        asset_verifier: Any = None,
    ) -> None:
        self.config = config
        self._transformers = transformers_module
        self._torch = torch_module
        self._process_vision_info = process_vision_info
        self._asset_verifier = asset_verifier or runtime_tree_sha256
        self._asset_verified = False
        self._processor = None
        self._model = None
        self._device = None

    @staticmethod
    def build_prompt(frames: Sequence[VisualFrame]) -> str:
        mapping = "\n".join(
            f"Image {index} = {frame.frame_id}"
            for index, frame in enumerate(frames, start=1)
        )
        return f"""You are a claim-blind visual observer. Use only the supplied frames.
You have no hidden claim, truth label, or dataset label. Do not make a veracity
decision. Report only directly visible atomic facts. Do not infer an exact
location, identity, intent, cause, or event beyond what is directly visible.
Do not transcribe OCR text, subtitles, logos, or watermarks. Do not report dates,
names, or numbers. Do not use probably, likely, appears, seems, suggests,
because, causes, intends, or wants. Use a generic identity or category when
uncertain. Return a maximum of eight observations; an empty observations list
is valid. Return JSON only.

Frame mapping (use these exact IDs in frame_ids and evidence_refs):
{mapping}

Allowed observation_type values are: entity, action, scene, object_state,
spatial_relation, temporal_change. Each observation must use exactly one of
these values.

Return JSON in this shape:
{{"observations":[{{"observation_type":"scene","observation":"A directly visible atomic fact.","frame_ids":["{frames[0].frame_id if frames else 'F000'}"],"evidence_refs":["{frames[0].frame_id if frames else 'F000'}"]}}]}}
"""

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[-1].strip() == "```":
                lines = lines[1:-1]
                text = "\n".join(lines).strip()
        return text

    @classmethod
    def recover_json(cls, raw: str) -> Tuple[List[Dict[str, Any]], str]:
        try:
            payload = json.loads(cls._strip_code_fence(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen visual output is not valid JSON") from exc
        if isinstance(payload, dict) and "observations" in payload:
            observations = payload["observations"]
            mode = "canonical_object"
        elif isinstance(payload, list):
            observations = payload
            mode = "top_level_array_wrapped"
        elif isinstance(payload, dict) and {
            "observation_type",
            "observation",
            "frame_ids",
            "evidence_refs",
        } <= set(payload):
            observations = [payload]
            mode = "single_observation_object_wrapped"
        else:
            raise ValueError("Qwen visual output has no recoverable observation form")
        if not isinstance(observations, list) or not all(
            isinstance(item, dict) for item in observations
        ):
            raise ValueError("Qwen observations must be a list of objects")
        return observations, mode

    @staticmethod
    def _string_list(value: Any) -> Optional[List[str]]:
        if not isinstance(value, list) or not value:
            return None
        if not all(isinstance(item, str) and item for item in value):
            return None
        return list(value)

    @classmethod
    def filter_observations(
        cls, observations: Iterable[Dict[str, Any]], selected_frame_ids: Iterable[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        materialized = list(observations)
        allowed_frames = set(selected_frame_ids)
        accepted = []
        rejected = []
        for index, observation in enumerate(materialized):
            if not isinstance(observation, dict):
                rejected.append(
                    {
                        "observation_index": index,
                        "reasons": ["observation_not_object"],
                        "observation": observation,
                    }
                )
                continue
            reasons = []
            observation_type = observation.get("observation_type")
            raw_text = observation.get("observation")
            text = raw_text.strip() if isinstance(raw_text, str) else ""
            frame_ids = cls._string_list(observation.get("frame_ids"))
            evidence_refs = cls._string_list(observation.get("evidence_refs"))
            if observation_type not in ALLOWED_OBSERVATION_TYPES:
                reasons.append("invalid_observation_type")
            if not 5 <= len(text) <= 280:
                reasons.append("invalid_observation_length")
            if frame_ids is None:
                reasons.append("invalid_frame_ids")
            elif not set(frame_ids) <= allowed_frames:
                reasons.append("unknown_frame_ids")
            if evidence_refs is None:
                reasons.append("invalid_evidence_refs")
            elif not set(evidence_refs) <= allowed_frames:
                reasons.append("unknown_evidence_refs")
            lowered = text.casefold()
            if any(phrase in lowered for phrase in RISKY_PHRASES):
                reasons.append("risky_inference_or_ocr_language")
            if observation_type == "temporal_change" and len(set(frame_ids or [])) < 2:
                reasons.append("temporal_change_requires_two_frame_ids")
            if not reasons and len(accepted) >= 8:
                reasons.append("maximum_observations_exceeded")
            if reasons:
                rejected.append(
                    {
                        "observation_index": index,
                        "reasons": reasons,
                        "observation": dict(observation),
                    }
                )
                continue
            accepted.append(
                {
                    "observation_type": observation_type,
                    "observation": text,
                    "frame_ids": frame_ids,
                    "evidence_refs": evidence_refs,
                }
            )
        return accepted, rejected

    def _dependencies(self):
        transformers_module = self._transformers or importlib.import_module(
            "transformers"
        )
        torch_module = self._torch or importlib.import_module("torch")
        process_vision_info = self._process_vision_info
        if process_vision_info is None:
            process_vision_info = importlib.import_module(
                "qwen_vl_utils"
            ).process_vision_info
        return transformers_module, torch_module, process_vision_info

    def _verify_assets(self) -> None:
        if self._asset_verified:
            return
        actual = self._asset_verifier(self.config.model_path, QWEN_RUNTIME_TREE_FILES)
        if actual != QWEN_RUNTIME_TREE_SHA256:
            raise ValueError(
                "Qwen frozen runtime tree SHA256 mismatch: "
                f"expected {QWEN_RUNTIME_TREE_SHA256}, got {actual}"
            )
        self._asset_verified = True

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.config.model_path.is_dir():
            raise FileNotFoundError(
                f"configured local Qwen directory not found: {self.config.model_path}"
            )
        self._verify_assets()
        transformers_module, torch_module, process_vision_info = self._dependencies()
        cuda_available = bool(torch_module.cuda.is_available())
        device = self.config.device
        if device == "auto":
            device = "cuda:0" if cuda_available else "cpu"
        elif device.startswith("cuda") and not cuda_available:
            raise RuntimeError(f"configured Qwen device is unavailable: {device}")
        bf16_supported = bool(
            device.startswith("cuda")
            and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
        )
        dtype = torch_module.bfloat16 if bf16_supported else torch_module.float32
        local_path = str(self.config.model_path)
        self._processor = transformers_module.AutoProcessor.from_pretrained(
            local_path, local_files_only=True, use_fast=False
        )
        model_class = transformers_module.Qwen2_5_VLForConditionalGeneration
        try:
            self._model = model_class.from_pretrained(
                local_path,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            self._model = model_class.from_pretrained(
                local_path,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._process_vision_info = process_vision_info

    def _generate(self, frames: Sequence[VisualFrame], prompt: str) -> str:
        self.load()
        content = [
            {"type": "image", "image": str(frame.frame_path)} for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device) if hasattr(inputs, "to") else inputs
        generated_ids = self._model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=self.config.max_new_tokens,
        )
        input_ids = inputs["input_ids"]
        trimmed = [
            output_ids[len(source_ids) :]
            for source_ids, output_ids in zip(input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def observe(self, frames: Sequence[VisualFrame]) -> QwenVisualObservationResult:
        selected = list(frames)
        prompt = self.build_prompt(selected)
        raw = self._generate(selected, prompt)
        observations, recovery_mode = self.recover_json(raw)
        accepted, rejected = self.filter_observations(
            observations, [frame.frame_id for frame in selected]
        )
        raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return QwenVisualObservationResult(
            observations=accepted,
            recovery_mode=recovery_mode,
            raw_generation_sha256=raw_sha256,
            rejected_observation_count=len(rejected),
            raw_generation=raw,
            rejected_observations=rejected,
        )
