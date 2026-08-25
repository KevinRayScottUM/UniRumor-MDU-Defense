"""Claim-blind local Qwen2.5-VL visual observation with grounded filtering."""

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from schemas.visual_xai import VisualTargetScore, VisualTargetSpan
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
    "speaking about",
    "talking about",
    "discussing",
    "discusses",
    "mentions",
    "the topic",
    "topic",
    "speaking",
    "talking",
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


@dataclass(frozen=True, slots=True)
class QwenPreparedTargetScoring:
    """Image-independent state for one fixed teacher-forced target."""

    frame_ids: Tuple[str, ...]
    prompt: str
    rendered_text: str
    target_sequence: str
    spans: Tuple[VisualTargetSpan, ...]
    target_token_ids: Tuple[int, ...]
    span_relative_indices: Tuple[Tuple[str, Tuple[int, ...]], ...]


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

    @property
    def runtime_fingerprint(self) -> str:
        """Frozen model-tree identity used by deterministic XAI cache keys."""

        return QWEN_RUNTIME_TREE_SHA256

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
        self._transformers = transformers_module
        self._torch = torch_module
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

    @staticmethod
    def _find_unique_subsequence(
        values: Sequence[int], target: Sequence[int]
    ) -> int:
        if not target or len(target) > len(values):
            raise ValueError("fixed target tokens are unavailable")
        matches = [
            index
            for index in range(len(values) - len(target) + 1)
            if list(values[index : index + len(target)]) == list(target)
        ]
        if len(matches) != 1:
            raise ValueError("fixed target tokens do not map uniquely")
        return matches[0]

    @staticmethod
    def _flat_tokenizer_value(value: Any) -> List[Any]:
        materialized = value.tolist() if hasattr(value, "tolist") else list(value)
        if materialized and isinstance(materialized[0], list):
            if len(materialized) != 1:
                raise ValueError("expected one tokenized text sequence")
            materialized = materialized[0]
        return list(materialized)

    @staticmethod
    def _validate_target_scoring_request(
        frames: Sequence[VisualFrame],
        target_sequence: str,
        spans: Sequence[VisualTargetSpan],
    ) -> Tuple[Tuple[VisualFrame, ...], Tuple[VisualTargetSpan, ...]]:
        prepared_frames = tuple(frames)
        requested_spans = tuple(spans)
        if not prepared_frames:
            raise ValueError("frames must not be empty")
        if not isinstance(target_sequence, str) or not target_sequence:
            raise ValueError("target_sequence is required")
        if not requested_spans or not all(
            isinstance(span, VisualTargetSpan) for span in requested_spans
        ):
            raise TypeError("spans must contain VisualTargetSpan objects")
        if any(span.end_character > len(target_sequence) for span in requested_spans):
            raise ValueError("target span exceeds the fixed target sequence")
        return prepared_frames, requested_spans

    def prepare_target_scoring(
        self,
        frames: Sequence[VisualFrame],
        target_sequence: str,
        spans: Sequence[VisualTargetSpan],
    ) -> QwenPreparedTargetScoring:
        """Prepare fixed text/token/span state once without running the model.

        Image decoding and processor tensor construction remain batch-specific.
        The prompt, rendered target, target token IDs, deterministic spans, and
        phrase-to-token mapping are invariant across Gaussian-blur variants.
        """

        prepared_frames, requested_spans = self._validate_target_scoring_request(
            frames, target_sequence, spans
        )
        self.load()
        prompt = self.build_prompt(prepared_frames)
        content = [
            {"type": "image", "image": str(frame.frame_path)}
            for frame in prepared_frames
        ]
        content.append({"type": "text", "text": prompt})
        user_messages = [{"role": "user", "content": content}]
        messages = [
            *user_messages,
            {"role": "assistant", "content": target_sequence},
        ]
        prompt_rendered = self._processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        target_start = rendered.rfind(target_sequence)
        if target_start < 0:
            raise ValueError("fixed target is absent from the rendered prompt")
        target_end = target_start + len(target_sequence)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if not callable(tokenizer):
            raise RuntimeError("Qwen processor tokenizer offsets are unavailable")

        try:
            tokenized = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            text_ids = self._flat_tokenizer_value(tokenized["input_ids"])
            offsets = self._flat_tokenizer_value(tokenized["offset_mapping"])
            target_text_indices = [
                index
                for index, offset in enumerate(offsets)
                if len(offset) == 2
                and int(offset[1]) > target_start
                and int(offset[0]) < target_end
            ]
            if not target_text_indices:
                raise ValueError("fixed target has no scoreable tokens")
            target_token_ids = tuple(
                int(text_ids[index]) for index in target_text_indices
            )
            span_relative_indices = []
            for span in requested_spans:
                absolute_start = target_start + span.start_character
                absolute_end = target_start + span.end_character
                relative_indices = tuple(
                    target_index
                    for target_index, text_index in enumerate(target_text_indices)
                    if int(offsets[text_index][1]) > absolute_start
                    and int(offsets[text_index][0]) < absolute_end
                )
                if not relative_indices:
                    raise ValueError(f"target span {span.span_id!r} has no tokens")
                span_relative_indices.append((span.span_id, relative_indices))
        except (NotImplementedError, TypeError, ValueError):
            full_tokenized = tokenizer(rendered, add_special_tokens=False)
            prompt_tokenized = tokenizer(prompt_rendered, add_special_tokens=False)
            full_text_ids = self._flat_tokenizer_value(full_tokenized["input_ids"])
            prompt_text_ids = self._flat_tokenizer_value(
                prompt_tokenized["input_ids"]
            )
            if full_text_ids[: len(prompt_text_ids)] != prompt_text_ids:
                raise ValueError(
                    "assistant target does not follow the generation prompt"
                )
            target_token_ids = tuple(
                int(value) for value in full_text_ids[len(prompt_text_ids) :]
            )
            decode = getattr(tokenizer, "decode", None)
            if not callable(decode):
                raise RuntimeError("Qwen tokenizer decoding is unavailable")
            decoded_prefixes = [""]
            for end_index in range(1, len(target_token_ids) + 1):
                decoded_prefixes.append(
                    decode(
                        target_token_ids[:end_index],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            decoded_assistant = decoded_prefixes[-1]
            decoded_target_start = decoded_assistant.find(target_sequence)
            if (
                decoded_target_start < 0
                or decoded_assistant.find(target_sequence, decoded_target_start + 1)
                >= 0
            ):
                raise ValueError("decoded fixed target does not map uniquely")
            span_relative_indices = []
            for span in requested_spans:
                absolute_start = decoded_target_start + span.start_character
                absolute_end = decoded_target_start + span.end_character
                relative_indices = tuple(
                    index
                    for index in range(len(target_token_ids))
                    if len(decoded_prefixes[index + 1]) > absolute_start
                    and len(decoded_prefixes[index]) < absolute_end
                )
                if not relative_indices:
                    raise ValueError(
                        f"target span {span.span_id!r} has no decoded tokens"
                    )
                span_relative_indices.append((span.span_id, relative_indices))

        return QwenPreparedTargetScoring(
            frame_ids=tuple(frame.frame_id for frame in prepared_frames),
            prompt=prompt,
            rendered_text=rendered,
            target_sequence=target_sequence,
            spans=requested_spans,
            target_token_ids=target_token_ids,
            span_relative_indices=tuple(span_relative_indices),
        )

    def score_prepared_target_logprob_batch(
        self,
        frame_batches: Sequence[Sequence[VisualFrame]],
        prepared: QwenPreparedTargetScoring,
    ) -> List[VisualTargetScore]:
        """Score image variants using one immutable fixed-target preparation."""

        if not isinstance(prepared, QwenPreparedTargetScoring):
            raise TypeError("prepared must be QwenPreparedTargetScoring")
        batches = [tuple(frames) for frames in frame_batches]
        if not batches or any(not frames for frames in batches):
            raise ValueError("frame_batches must contain non-empty frame sequences")
        if any(
            tuple(frame.frame_id for frame in frames) != prepared.frame_ids
            for frames in batches
        ):
            raise ValueError("frame batch identity/order differs from prepared target")
        self.load()
        messages_batch = []
        for frames in batches:
            content = [
                {"type": "image", "image": str(frame.frame_path)}
                for frame in frames
            ]
            content.append({"type": "text", "text": prepared.prompt})
            messages_batch.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": prepared.target_sequence},
                ]
            )

        image_inputs, video_inputs = self._process_vision_info(messages_batch)
        inputs = self._processor(
            text=[prepared.rendered_text] * len(batches),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device) if hasattr(inputs, "to") else inputs

        mappings = []
        input_ids_batch = inputs["input_ids"]
        attention_batch = inputs.get("attention_mask")
        relative_by_span = dict(prepared.span_relative_indices)
        for row_index in range(len(batches)):
            raw_ids = input_ids_batch[row_index]
            raw_ids = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
            if attention_batch is None:
                active_positions = list(range(len(raw_ids)))
            else:
                raw_mask = attention_batch[row_index]
                raw_mask = (
                    raw_mask.tolist()
                    if hasattr(raw_mask, "tolist")
                    else list(raw_mask)
                )
                active_positions = [
                    index for index, value in enumerate(raw_mask) if int(value) != 0
                ]
            active_ids = [int(raw_ids[index]) for index in active_positions]
            sequence_start = self._find_unique_subsequence(
                active_ids, prepared.target_token_ids
            )
            span_positions = {}
            for span in prepared.spans:
                relative_indices = relative_by_span[span.span_id]
                model_positions = tuple(
                    active_positions[sequence_start + index]
                    for index in relative_indices
                )
                model_token_ids = tuple(
                    prepared.target_token_ids[index] for index in relative_indices
                )
                if any(position < 1 for position in model_positions):
                    raise ValueError("fixed target begins before a scoreable position")
                span_positions[span.span_id] = (model_positions, model_token_ids)
            mappings.append(span_positions)

        with self._torch.inference_mode():
            outputs = self._model(**inputs, use_cache=False)
        logits = outputs.logits
        results = []
        for row_index, span_positions in enumerate(mappings):
            span_scores = []
            span_counts = []
            for span in prepared.spans:
                positions, token_ids = span_positions[span.span_id]
                total = 0.0
                for position, token_id in zip(positions, token_ids):
                    token_logits = logits[row_index, position - 1]
                    log_probabilities = self._torch.log_softmax(
                        token_logits.float(), dim=-1
                    )
                    total += float(log_probabilities[token_id].item())
                span_scores.append((span.span_id, total))
                span_counts.append((span.span_id, len(positions)))
            results.append(
                VisualTargetScore(
                    span_log_probabilities=tuple(span_scores),
                    span_token_counts=tuple(span_counts),
                )
            )
        return results

    def is_cuda_out_of_memory(self, error: BaseException) -> bool:
        """Recognize only PyTorch's concrete CUDA OOM exception types."""

        torch_module = self._torch
        if (
            torch_module is None
            or not isinstance(self._device, str)
            or not self._device.startswith("cuda")
        ):
            return False
        oom_type = getattr(
            getattr(torch_module, "cuda", None), "OutOfMemoryError", None
        )
        return isinstance(oom_type, type) and isinstance(error, oom_type)

    def clear_cuda_oom_cache(self) -> None:
        """Release cached CUDA blocks only at an adaptive OOM retry boundary."""

        cuda = getattr(self._torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if (
            isinstance(self._device, str)
            and self._device.startswith("cuda")
            and callable(empty_cache)
        ):
            empty_cache()

    def score_target_logprob_batch(
        self,
        frame_batches: Sequence[Sequence[VisualFrame]],
        target_sequence: str,
        spans: Sequence[VisualTargetSpan],
    ) -> List[VisualTargetScore]:
        """Teacher-force the exact prior generation and score requested spans.

        No decoding or free-form regeneration occurs. Each sample uses the same
        claim-blind observer prompt and its supplied source-frame sequence.
        """

        batches = [tuple(frames) for frames in frame_batches]
        if not batches:
            raise ValueError("frame_batches must contain non-empty frame sequences")
        prepared = self.prepare_target_scoring(
            batches[0], target_sequence, spans
        )
        return self.score_prepared_target_logprob_batch(batches, prepared)

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
