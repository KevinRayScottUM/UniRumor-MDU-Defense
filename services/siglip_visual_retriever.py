"""Local-only SigLIP2 claim-to-frame retrieval with frozen semantics."""

import hashlib
import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.cache_manager import safe_target


SIGLIP_MODEL_ID = "google/siglip2-base-patch16-384"
SIGLIP_FROZEN_REVISION = "f775b65a79762255128c981547af89addcfe0f88"
SIGLIP_RUNTIME_TREE_IDENTITY = (
    "5179f3193de143151f8062760999e8af4a1f3aa7885808b9a4aa5e855e2389e8"
)
SIGLIP_RUNTIME_TREE_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
MAX_CANDIDATE_FRAMES = 12
TOP_K_RETRIEVAL_FRAMES = 4
CLAIM_TOKEN_MAX_LENGTH = 64
JPEG_QUALITY = 95


def runtime_tree_sha256(model_path: Path, runtime_files: Sequence[str]) -> str:
    root = Path(model_path)
    rows = []
    for relative_filename in runtime_files:
        path = root / relative_filename
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen runtime file: {path}")
        rows.append(
            {
                "path": relative_filename,
                "size": path.stat().st_size,
                "sha256": SigLIPVisualRetriever._sha256(path),
            }
        )
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def historical_clip12_positions(total_frames: int, count: int = 12) -> List[int]:
    if total_frames <= 0 or count <= 0:
        return []
    return sorted(
        {
            round(i * (total_frames - 1) / max(count - 1, 1))
            for i in range(count)
        }
    )


@dataclass(frozen=True)
class VisualFrame:
    frame_id: str
    frame_path: Path
    frame_index: int
    timestamp_sec: Optional[float]
    frame_rank: int
    image_sha256: str
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_path": str(self.frame_path),
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "frame_rank": self.frame_rank,
            "image_sha256": self.image_sha256,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass(frozen=True)
class SigLIPRetrieverConfig:
    model_path: Path
    cache_root: Path
    device: str = "cuda:0"
    candidate_frame_count: int = MAX_CANDIDATE_FRAMES
    top_k: int = TOP_K_RETRIEVAL_FRAMES
    claim_max_length: int = CLAIM_TOKEN_MAX_LENGTH

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        object.__setattr__(self, "cache_root", Path(self.cache_root).resolve())
        if not self.device:
            raise ValueError("SigLIP device is required")
        if self.candidate_frame_count != MAX_CANDIDATE_FRAMES:
            raise ValueError("frozen SigLIP candidate_frame_count must equal 12")
        if self.top_k != TOP_K_RETRIEVAL_FRAMES:
            raise ValueError("frozen SigLIP retrieval requires top_k=4")
        if self.claim_max_length != CLAIM_TOKEN_MAX_LENGTH:
            raise ValueError("frozen SigLIP claim_max_length must equal 64")


@dataclass
class SigLIPRetrievalResult:
    candidate_frames: List[VisualFrame]
    selected_frames: List[VisualFrame]
    claim_token_audit: Dict[str, Any]
    model_id: str = SIGLIP_MODEL_ID
    frozen_revision: str = SIGLIP_FROZEN_REVISION
    runtime_tree_identity: str = SIGLIP_RUNTIME_TREE_IDENTITY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_frames": [frame.to_dict() for frame in self.candidate_frames],
            "selected_frames": [frame.to_dict() for frame in self.selected_frames],
            "claim_token_audit": dict(self.claim_token_audit),
            "model_id": self.model_id,
            "frozen_revision": self.frozen_revision,
            "runtime_tree_identity": self.runtime_tree_identity,
            "purpose": "retrieval_only",
        }


class SigLIPVisualRetriever:
    def __init__(
        self,
        config: SigLIPRetrieverConfig,
        cv2_module: Any = None,
        torch_module: Any = None,
        transformers_module: Any = None,
        image_module: Any = None,
        asset_verifier: Any = None,
    ) -> None:
        self.config = config
        self._cv2 = cv2_module
        self._torch = torch_module
        self._transformers = transformers_module
        self._image_module = image_module
        self._asset_verifier = asset_verifier or runtime_tree_sha256
        self._asset_verified = False
        self._processor = None
        self._model = None
        self._device = None

    @staticmethod
    def frame_id_for_rank(frame_rank: int) -> str:
        if not 0 <= frame_rank < MAX_CANDIDATE_FRAMES:
            raise ValueError("visual frame rank must be within 0..11")
        return f"F{frame_rank:03d}"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def extract_candidate_frames(
        self, session_id: str, video_path: Path
    ) -> List[VisualFrame]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"video path is not a regular file: {path}")
        cv2_module = self._cv2 or importlib.import_module("cv2")
        capture = cv2_module.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"OpenCV could not open video: {path}")
        try:
            total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)
            positions = historical_clip12_positions(
                total_frames, self.config.candidate_frame_count
            )
            extracted = []
            session_cache = "visual_" + hashlib.sha256(
                str(session_id).encode("utf-8")
            ).hexdigest()[:20]
            for frame_rank, position in enumerate(positions):
                capture.set(cv2_module.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"OpenCV could not read visual frame at position {position}"
                    )
                frame_id = self.frame_id_for_rank(frame_rank)
                filename = f"frame_{frame_rank:02d}_{position:08d}.jpg"
                target = safe_target(
                    self.config.cache_root,
                    session_cache,
                    filename,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                written = cv2_module.imwrite(
                    str(target),
                    frame,
                    [int(cv2_module.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not written or not target.is_file():
                    raise ValueError(f"OpenCV could not write visual frame: {target}")
                extracted.append(
                    VisualFrame(
                        frame_id=frame_id,
                        frame_path=target,
                        frame_index=int(position),
                        timestamp_sec=(float(position) / fps if fps > 0 else None),
                        frame_rank=frame_rank,
                        image_sha256=self._sha256(target),
                    )
                )
            return extracted
        finally:
            capture.release()

    @staticmethod
    def chronology_key(frame: VisualFrame) -> Tuple[bool, float, int, str]:
        return (
            frame.timestamp_sec is None,
            frame.timestamp_sec if frame.timestamp_sec is not None else 0.0,
            frame.frame_index,
            str(frame.frame_path),
        )

    @classmethod
    def rank_frames(
        cls, frames: Iterable[VisualFrame], scores: Sequence[float]
    ) -> List[VisualFrame]:
        materialized = list(frames)
        if len(materialized) != len(scores):
            raise ValueError("SigLIP score count does not match candidate frames")
        scored = [
            replace(frame, retrieval_score=float(score))
            for frame, score in zip(materialized, scores)
        ]
        ranked = sorted(
            scored,
            key=lambda frame: (
                -frame.retrieval_score,
                *cls.chronology_key(frame),
            ),
        )
        return [
            replace(frame, retrieval_rank=rank)
            for rank, frame in enumerate(ranked, start=1)
        ]

    @classmethod
    def rank_and_select(
        cls, frames: Iterable[VisualFrame], scores: Sequence[float], top_k: int = 4
    ) -> List[VisualFrame]:
        ranked = cls.rank_frames(frames, scores)
        selected = ranked[:top_k]
        selected.sort(key=cls.chronology_key)
        return selected

    @staticmethod
    def _valid_limit(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if 0 < value < 1_000_000 else None

    @classmethod
    def effective_text_max_length(
        cls, configured_limit: int, tokenizer: Any, model: Any
    ) -> int:
        limits = [configured_limit]
        tokenizer_limit = cls._valid_limit(
            getattr(tokenizer, "model_max_length", None)
        )
        if tokenizer_limit is not None:
            limits.append(tokenizer_limit)
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        for field in ("max_position_embeddings", "max_sequence_length"):
            model_limit = cls._valid_limit(getattr(text_config, field, None))
            if model_limit is not None:
                limits.append(model_limit)
        return min(limits)

    def _dependencies(self):
        torch_module = self._torch or importlib.import_module("torch")
        transformers_module = self._transformers or importlib.import_module(
            "transformers"
        )
        return torch_module, transformers_module

    def _verify_assets(self) -> None:
        if self._asset_verified:
            return
        actual = self._asset_verifier(
            self.config.model_path, SIGLIP_RUNTIME_TREE_FILES
        )
        if actual != SIGLIP_RUNTIME_TREE_IDENTITY:
            raise ValueError(
                "SigLIP frozen runtime tree SHA256 mismatch: "
                f"expected {SIGLIP_RUNTIME_TREE_IDENTITY}, got {actual}"
            )
        self._asset_verified = True

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.config.model_path.is_dir():
            raise FileNotFoundError(
                f"configured local SigLIP directory not found: {self.config.model_path}"
            )
        self._verify_assets()
        torch_module, transformers_module = self._dependencies()
        cuda_available = bool(torch_module.cuda.is_available())
        device = self.config.device
        if device == "auto":
            device = "cuda:0" if cuda_available else "cpu"
        elif device.startswith("cuda") and not cuda_available:
            raise RuntimeError(f"configured SigLIP device is unavailable: {device}")
        if device.startswith("cuda"):
            bf16_supported = bool(
                getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
            )
            dtype = (
                torch_module.bfloat16 if bf16_supported else torch_module.float16
            )
        else:
            dtype = torch_module.float32
        local_path = str(self.config.model_path)
        self._processor = transformers_module.AutoProcessor.from_pretrained(
            local_path, local_files_only=True, use_fast=False
        )
        try:
            self._model = transformers_module.AutoModel.from_pretrained(
                local_path, local_files_only=True, dtype=dtype
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            self._model = transformers_module.AutoModel.from_pretrained(
                local_path, local_files_only=True, torch_dtype=dtype
            )
        self._model.to(device)
        self._model.eval()
        self._device = device

    def _images(self, frames: List[VisualFrame]) -> List[Any]:
        image_module = self._image_module
        if image_module is None:
            image_module = importlib.import_module("PIL.Image")
        images = []
        for frame in frames:
            with image_module.open(frame.frame_path) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _prepare_image_inputs(self, frames: List[VisualFrame]) -> Any:
        return self._processor(
            images=self._images(frames),
            padding=True,
            return_tensors="pt",
        )

    @staticmethod
    def _to_device(inputs: Any, device: str) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _sequence_length(value: Any) -> int:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        return len(value) if isinstance(value, list) else 0

    @classmethod
    def _model_input_token_count(cls, text_inputs: Any) -> int:
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is not None:
            first_mask = attention_mask[0]
            if hasattr(first_mask, "sum"):
                return int(first_mask.sum().item())
            return int(sum(first_mask))
        input_ids = text_inputs["input_ids"]
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        return cls._sequence_length(input_ids)

    def _prepare_text_inputs(
        self, original_claim: str, effective_max: int
    ) -> Tuple[Any, Dict[str, Any]]:
        tokenizer = self._processor.tokenizer
        tokenizer.truncation_side = "right"
        untruncated = tokenizer(
            original_claim,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        original_token_count = self._sequence_length(untruncated["input_ids"])
        text_inputs = self._processor(
            text=[original_claim],
            padding="max_length",
            truncation=True,
            max_length=effective_max,
            return_tensors="pt",
        )
        model_input_token_count = self._model_input_token_count(text_inputs)
        audit = {
            "policy": "right_truncate_for_siglip_retrieval_only",
            "configured_max_length": self.config.claim_max_length,
            "effective_max_length": effective_max,
            "original_token_count": original_token_count,
            "model_input_token_count": model_input_token_count,
            "truncated": original_token_count > effective_max,
            "padding": "max_length",
            "original_claim_preserved_for_mdu": True,
            "claim_sha256": hashlib.sha256(
                original_claim.encode("utf-8")
            ).hexdigest(),
        }
        return text_inputs, audit

    def score_frames(
        self, claim: str, frames: Iterable[VisualFrame]
    ) -> Tuple[List[float], Dict[str, Any]]:
        original_claim = str(claim)
        materialized = list(frames)
        if not materialized:
            return [], {
                "policy": "right_truncate_for_siglip_retrieval_only",
                "configured_max_length": self.config.claim_max_length,
                "effective_max_length": self.config.claim_max_length,
                "original_token_count": None,
                "model_input_token_count": 0,
                "truncated": False,
                "padding": "max_length",
                "original_claim_preserved_for_mdu": True,
                "claim_sha256": hashlib.sha256(
                    original_claim.encode("utf-8")
                ).hexdigest(),
            }
        self.load()
        tokenizer = self._processor.tokenizer
        effective_max = self.effective_text_max_length(
            self.config.claim_max_length, tokenizer, self._model
        )
        text_inputs, audit = self._prepare_text_inputs(original_claim, effective_max)
        image_inputs = self._prepare_image_inputs(materialized)
        text_inputs = self._to_device(text_inputs, self._device)
        image_inputs = self._to_device(image_inputs, self._device)
        torch_module = self._torch or importlib.import_module("torch")
        with torch_module.inference_mode():
            text_embedding = self._model.get_text_features(**text_inputs).to(
                dtype=torch_module.float32
            )
            image_embeddings = self._model.get_image_features(**image_inputs).to(
                dtype=torch_module.float32
            )
            text_norm = text_embedding / torch_module.linalg.vector_norm(
                text_embedding, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            image_norm = image_embeddings / torch_module.linalg.vector_norm(
                image_embeddings, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            scores = torch_module.matmul(image_norm, text_norm[0]).detach().cpu().tolist()
        return [float(score) for score in scores], audit

    def retrieve(
        self, claim: str, video_path: Path, session_id: str
    ) -> SigLIPRetrievalResult:
        candidates = self.extract_candidate_frames(session_id, video_path)
        scores, audit = self.score_frames(claim, candidates)
        ranked = self.rank_frames(candidates, scores)
        selected = sorted(ranked[: self.config.top_k], key=self.chronology_key)
        scored_by_id = {frame.frame_id: frame for frame in ranked}
        candidate_output = [scored_by_id.get(frame.frame_id, frame) for frame in candidates]
        return SigLIPRetrievalResult(
            candidate_frames=candidate_output,
            selected_frames=selected,
            claim_token_audit=audit,
        )
