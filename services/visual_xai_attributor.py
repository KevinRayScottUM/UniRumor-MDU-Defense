"""Post-verdict Qwen occlusion attribution for supplemental visual evidence.

The service scores the fixed original Qwen generation.  It never generates a
new observation, creates a RuntimeUnit, or contributes to a verification path.
"""

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from schemas import (
    QWEN_OCCLUSION_BASELINE,
    QWEN_OCCLUSION_METHOD,
    VISUAL_XAI_ARTIFACT_TYPE,
    VISUAL_XAI_SCHEMA_VERSION,
    VisualAttributionArtifact,
    VisualAttributionMap,
    VisualObservationSnapshot,
    VisualTargetScore,
    VisualTargetSpan,
)
from services.cache_manager import safe_target, write_json
from services.qwen_visual_observer import QWEN_RUNTIME_TREE_SHA256
from services.siglip_visual_retriever import VisualFrame


VISUAL_XAI_CONFIGURATION_VERSION = "qwen_occlusion_blur_v2"
VISUAL_XAI_PHRASE_POLICY = "deterministic_visible_concept_tokens_v1"
VISUAL_XAI_FAILURE_WARNING = "Visual attribution unavailable."
VISUAL_XAI_PROFILE_ENV = "MDU_VISUAL_XAI_PROFILE"
VISUAL_XAI_GRID_SIZE_ENV = "MDU_VISUAL_XAI_GRID_SIZE"
VISUAL_XAI_MAX_CONCURRENCY_ENV = "MDU_VISUAL_XAI_MAX_CONCURRENCY"
VISUAL_XAI_PUBLIC_GRID_SIZE = 6
VISUAL_XAI_RESEARCH_GRID_SIZE = 8
_SAFE_UNAVAILABLE_REASONS = {
    "attribution_failed",
    "attribution_timeout",
    "source_frame_unavailable",
    "source_frame_fingerprint_mismatch",
    "target_span_unavailable",
    "unsupported_model_scoring",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_PHRASE_STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "also",
    "among",
    "around",
    "because",
    "before",
    "behind",
    "being",
    "below",
    "between",
    "centrally",
    "contains",
    "contain",
    "displayed",
    "during",
    "front",
    "from",
    "her",
    "hers",
    "him",
    "his",
    "have",
    "having",
    "into",
    "near",
    "positioned",
    "several",
    "shows",
    "shown",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "them",
    "this",
    "those",
    "through",
    "under",
    "visible",
    "where",
    "which",
    "while",
    "with",
}
_SAFE_CAPITALIZED_CONCEPTS = {
    "audience",
    "building",
    "camera",
    "court",
    "crowd",
    "microphone",
    "microphones",
    "person",
    "player",
    "players",
    "podium",
    "screen",
    "speaker",
    "stage",
    "vehicle",
}


@dataclass(frozen=True, slots=True)
class VisualXAIConfig:
    cache_root: Path
    profile: str = "research"
    grid_rows: int = 8
    grid_columns: int = 8
    attribution_batch_size: int = 2
    blur_kernel_size: int = 31
    overlay_alpha: float = 0.62
    timeout_seconds: float = 300.0
    maximum_phrase_count: int = 6
    configuration_version: str = VISUAL_XAI_CONFIGURATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", Path(self.cache_root).resolve())
        if self.profile not in {"public", "research"}:
            raise ValueError("profile must be public or research")
        for field_name in (
            "grid_rows",
            "grid_columns",
            "attribution_batch_size",
            "maximum_phrase_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            type(self.blur_kernel_size) is not int
            or self.blur_kernel_size < 3
            or self.blur_kernel_size % 2 == 0
        ):
            raise ValueError("blur_kernel_size must be an odd integer of at least 3")
        if not isinstance(self.overlay_alpha, float) or not 0.0 < self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be within (0, 1]")
        if not isinstance(self.timeout_seconds, float) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.configuration_version:
            raise ValueError("configuration_version is required")

    @property
    def configuration_fingerprint(self) -> str:
        payload = {
            "profile": self.profile,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "attribution_batch_size": self.attribution_batch_size,
            "blur_kernel_size": self.blur_kernel_size,
            "overlay_alpha": self.overlay_alpha,
            "maximum_phrase_count": self.maximum_phrase_count,
            "configuration_version": self.configuration_version,
            "phrase_policy": VISUAL_XAI_PHRASE_POLICY,
            "method": QWEN_OCCLUSION_METHOD,
            "baseline": QWEN_OCCLUSION_BASELINE,
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    @classmethod
    def from_environment(
        cls,
        cache_root: Path,
        *,
        environ: Optional[Mapping[str, str]] = None,
        attribution_batch_size: int = 2,
    ) -> "VisualXAIConfig":
        values = os.environ if environ is None else environ
        profile = values.get(VISUAL_XAI_PROFILE_ENV, "research").strip().lower()
        if profile not in {"public", "research"}:
            raise ValueError(
                f"{VISUAL_XAI_PROFILE_ENV} must be public or research"
            )
        default_grid = (
            VISUAL_XAI_PUBLIC_GRID_SIZE
            if profile == "public"
            else VISUAL_XAI_RESEARCH_GRID_SIZE
        )
        raw_grid = values.get(VISUAL_XAI_GRID_SIZE_ENV)
        if raw_grid is None or not raw_grid.strip():
            grid = default_grid
        else:
            try:
                grid = int(raw_grid)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{VISUAL_XAI_GRID_SIZE_ENV} must be 6 or 8"
                ) from None
            if grid not in {
                VISUAL_XAI_PUBLIC_GRID_SIZE,
                VISUAL_XAI_RESEARCH_GRID_SIZE,
            }:
                raise ValueError(f"{VISUAL_XAI_GRID_SIZE_ENV} must be 6 or 8")
        return cls(
            cache_root=cache_root,
            profile=profile,
            grid_rows=grid,
            grid_columns=grid,
            attribution_batch_size=attribution_batch_size,
        )


class VisualXAIAttributor:
    """Create model-derived XAI artifacts from isolated visual snapshots."""

    def __init__(
        self,
        scorer: Any,
        config: VisualXAIConfig,
        *,
        cv2_module: Any = None,
        numpy_module: Any = None,
        clock: Any = None,
    ) -> None:
        if not callable(getattr(scorer, "score_target_logprob_batch", None)):
            raise TypeError(
                "scorer must provide score_target_logprob_batch(frame_batches, target_sequence, spans)"
            )
        if not isinstance(config, VisualXAIConfig):
            raise TypeError("config must be a VisualXAIConfig")
        self.scorer = scorer
        self.config = config
        self._cv2 = cv2_module
        self._numpy = numpy_module
        self._clock = clock or time.monotonic
        self.cache_hits = 0

    def _image_dependencies(self):
        return (
            self._cv2 or importlib.import_module("cv2"),
            self._numpy or importlib.import_module("numpy"),
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _text_sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _model_fingerprint(snapshot: VisualObservationSnapshot, scorer: Any) -> str:
        fingerprint = getattr(scorer, "runtime_fingerprint", None)
        if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            return fingerprint
        if snapshot.qwen_model_id and snapshot.qwen_revision:
            return hashlib.sha256(
                f"{snapshot.qwen_model_id}\0{snapshot.qwen_revision}".encode("utf-8")
            ).hexdigest()
        return QWEN_RUNTIME_TREE_SHA256

    @staticmethod
    def _source_frame(
        reference_id: str, frames: Sequence[VisualFrame]
    ) -> Optional[VisualFrame]:
        matches = [frame for frame in frames if frame.frame_id == reference_id]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def phrase_spans(text: str, maximum: int = 6) -> Tuple[Tuple[str, int, int], ...]:
        """Return conservative deterministic visual-concept token spans.

        This deliberately copies existing tokens and suppresses likely proper-name
        pairs.  It is not an identity-recognition or language-generation step.
        """

        matches = list(_WORD.finditer(text))
        capitalized = {index for index, item in enumerate(matches) if item.group(0)[0].isupper()}
        likely_name_indices = set()
        for index in capitalized:
            if index + 1 in capitalized:
                likely_name_indices.update({index, index + 1})
        output = []
        seen = set()
        for index, match in enumerate(matches):
            label = match.group(0)
            lowered = label.casefold()
            if (
                lowered in _PHRASE_STOPWORDS
                or index in likely_name_indices
                or (label[0].isupper() and lowered not in _SAFE_CAPITALIZED_CONCEPTS)
                or lowered in seen
            ):
                continue
            seen.add(lowered)
            output.append((label, match.start(), match.end()))
            if len(output) >= maximum:
                break
        return tuple(output)

    @classmethod
    def _target_spans(
        cls,
        snapshot: VisualObservationSnapshot,
        raw_generation: str,
        maximum_phrases: int,
    ) -> Tuple[VisualTargetSpan, ...]:
        text = snapshot.observation_text
        starts = [match.start() for match in re.finditer(re.escape(text), raw_generation)]
        if len(starts) != 1:
            raise ValueError("observation text does not map uniquely to raw generation")
        observation_start = starts[0]
        spans = [
            VisualTargetSpan(
                span_id="observation",
                scope="observation",
                label="Whole observation",
                start_character=observation_start,
                end_character=observation_start + len(text),
            )
        ]
        for phrase_index, (label, start, end) in enumerate(
            cls.phrase_spans(text, maximum_phrases), start=1
        ):
            spans.append(
                VisualTargetSpan(
                    span_id=f"phrase_{phrase_index:02d}",
                    scope="phrase",
                    label=label,
                    start_character=observation_start + start,
                    end_character=observation_start + end,
                )
            )
        return tuple(spans)

    def _cache_key(
        self,
        snapshot: VisualObservationSnapshot,
        source_frame: VisualFrame,
        all_frames: Sequence[VisualFrame],
        model_fingerprint: str,
    ) -> str:
        payload = {
            "method": QWEN_OCCLUSION_METHOD,
            "model_fingerprint": model_fingerprint,
            "source_frame_sha256": source_frame.image_sha256,
            "context_frames": [
                {"frame_id": frame.frame_id, "image_sha256": frame.image_sha256}
                for frame in all_frames
            ],
            "observation_text_sha256": self._text_sha256(snapshot.observation_text),
            "raw_generation_sha256": snapshot.raw_generation_sha256,
            "prompt_policy": snapshot.prompt_policy,
            "profile": self.config.profile,
            "grid_rows": self.config.grid_rows,
            "grid_columns": self.config.grid_columns,
            "attribution_batch_size": self.config.attribution_batch_size,
            "blur_kernel_size": self.config.blur_kernel_size,
            "overlay_alpha": self.config.overlay_alpha,
            "configuration_version": self.config.configuration_version,
            "configuration_fingerprint": self.config.configuration_fingerprint,
            "phrase_policy": VISUAL_XAI_PHRASE_POLICY,
            "phrase_spans": [
                {"label": label, "start": start, "end": end}
                for label, start, end in self.phrase_spans(
                    snapshot.observation_text,
                    self.config.maximum_phrase_count,
                )
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_paths(self, cache_key: str) -> Tuple[Path, Path]:
        category = f"xai_{cache_key}"
        directory = safe_target(self.config.cache_root, category, "manifest.json").parent
        return directory, directory / "manifest.json"

    @staticmethod
    def _map_from_dict(
        payload: Mapping[str, Any], overlay_path: Optional[Path]
    ) -> VisualAttributionMap:
        return VisualAttributionMap(
            map_id=payload["map_id"],
            scope=payload["scope"],
            label=payload["label"],
            target_start_character=payload["target_start_character"],
            target_end_character=payload["target_end_character"],
            target_token_count=payload["target_token_count"],
            baseline_target_log_probability=payload[
                "baseline_target_log_probability"
            ],
            raw_importance=tuple(
                tuple(row) for row in payload["raw_importance"]
            ),
            normalized_importance=tuple(
                tuple(row) for row in payload["normalized_importance"]
            ),
            overlay_image_path=overlay_path,
        )

    def _load_cache(self, cache_key: str) -> Optional[VisualAttributionArtifact]:
        directory, manifest = self._cache_paths(cache_key)
        if not manifest.is_file() or manifest.is_symlink():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            artifact_data = payload["artifact"]
            overlay_files = payload.get("overlay_files", {})
            maps = []
            for map_data in artifact_data["maps"]:
                filename = overlay_files.get(map_data["map_id"])
                overlay_path = None
                if filename is not None:
                    if not isinstance(filename, str) or not re.fullmatch(
                        r"overlay_[A-Za-z0-9_-]+\.png", filename
                    ):
                        return None
                    candidate = (directory / filename).resolve()
                    if directory.resolve() not in candidate.parents or not candidate.is_file():
                        return None
                    overlay_path = candidate
                maps.append(self._map_from_dict(map_data, overlay_path))
            artifact = VisualAttributionArtifact(
                **{
                    key: value
                    for key, value in artifact_data.items()
                    if key not in {"maps"}
                },
                maps=tuple(maps),
            )
            if artifact.cache_key != cache_key:
                return None
            if VisualAttributionArtifact.compute_identity(
                artifact.identity_payload()
            ) != artifact.artifact_id:
                return None
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None
        self.cache_hits += 1
        return artifact

    @staticmethod
    def _rebind_cached_artifact(
        artifact: VisualAttributionArtifact,
        snapshot: VisualObservationSnapshot,
    ) -> VisualAttributionArtifact:
        if artifact.observation_unit_id == snapshot.unit_id:
            return artifact
        rebound = replace(
            artifact,
            artifact_id="0" * 64,
            observation_unit_id=snapshot.unit_id,
        )
        return replace(
            rebound,
            artifact_id=VisualAttributionArtifact.compute_identity(
                rebound.identity_payload()
            ),
        )

    def _write_cache(self, artifact: VisualAttributionArtifact) -> None:
        _, manifest = self._cache_paths(artifact.cache_key)
        overlay_files = {
            item.map_id: item.overlay_image_path.name
            for item in artifact.maps
            if item.overlay_image_path is not None
        }
        write_json(
            manifest,
            {"artifact": artifact.to_dict(), "overlay_files": overlay_files},
        )

    @staticmethod
    def _matrix(values: Sequence[float], rows: int, columns: int):
        return tuple(
            tuple(float(values[row * columns + column]) for column in range(columns))
            for row in range(rows)
        )

    @staticmethod
    def _normalize_positive(values: Sequence[float]) -> Tuple[float, ...]:
        positive = tuple(max(0.0, float(value)) for value in values)
        maximum = max(positive, default=0.0)
        if maximum <= 0.0:
            return tuple(0.0 for _ in positive)
        return tuple(value / maximum for value in positive)

    @staticmethod
    def _cell_bounds(
        row: int, column: int, rows: int, columns: int, height: int, width: int
    ) -> Tuple[int, int, int, int]:
        y1 = (row * height) // rows
        y2 = ((row + 1) * height) // rows
        x1 = (column * width) // columns
        x2 = ((column + 1) * width) // columns
        return x1, y1, x2, y2

    def _write_variant(
        self,
        image: Any,
        blurred: Any,
        row: int,
        column: int,
        target: Path,
        cv2_module: Any,
    ) -> None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._cell_bounds(
            row,
            column,
            self.config.grid_rows,
            self.config.grid_columns,
            height,
            width,
        )
        variant = image.copy()
        variant[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2_module.imwrite(str(target), variant):
            raise ValueError("could not write an occluded frame variant")

    def _write_overlay(
        self,
        image: Any,
        normalized: Sequence[float],
        target: Path,
        cv2_module: Any,
        numpy_module: Any,
    ) -> None:
        overlay = image.astype(numpy_module.float32).copy()
        height, width = image.shape[:2]
        red = numpy_module.array([32.0, 32.0, 235.0], dtype=numpy_module.float32)
        for index, value in enumerate(normalized):
            if value <= 0.0:
                continue
            row, column = divmod(index, self.config.grid_columns)
            x1, y1, x2, y2 = self._cell_bounds(
                row,
                column,
                self.config.grid_rows,
                self.config.grid_columns,
                height,
                width,
            )
            alpha = self.config.overlay_alpha * float(value)
            region = overlay[y1:y2, x1:x2]
            overlay[y1:y2, x1:x2] = region * (1.0 - alpha) + red * alpha
        rendered = numpy_module.clip(overlay, 0, 255).astype(numpy_module.uint8)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        if not cv2_module.imwrite(str(temporary), rendered):
            raise ValueError("could not write attribution overlay")
        temporary.replace(target)

    def _unavailable(
        self,
        snapshot: VisualObservationSnapshot,
        reference: Any,
        model_fingerprint: str,
        cache_key: str,
        reason: str,
    ) -> VisualAttributionArtifact:
        if reason not in _SAFE_UNAVAILABLE_REASONS:
            reason = "attribution_failed"
        payload = {
            "schema_version": VISUAL_XAI_SCHEMA_VERSION,
            "artifact_type": VISUAL_XAI_ARTIFACT_TYPE,
            "status": "unavailable",
            "unavailable_reason": reason,
            "method": QWEN_OCCLUSION_METHOD,
            "model_id": snapshot.qwen_model_id,
            "model_revision": snapshot.qwen_revision,
            "model_fingerprint": model_fingerprint,
            "source_frame_id": reference.frame_id,
            "source_frame_index": reference.frame_index,
            "source_timestamp_seconds": reference.timestamp_seconds,
            "source_frame_sha256": reference.image_sha256,
            "observation_unit_id": snapshot.unit_id,
            "observation_text": snapshot.observation_text,
            "observation_text_sha256": self._text_sha256(snapshot.observation_text),
            "raw_generation_sha256": snapshot.raw_generation_sha256,
            "profile": self.config.profile,
            "grid_rows": self.config.grid_rows,
            "grid_columns": self.config.grid_columns,
            "attribution_batch_size": self.config.attribution_batch_size,
            "occlusion_baseline": QWEN_OCCLUSION_BASELINE,
            "configuration_version": self.config.configuration_version,
            "configuration_fingerprint": self.config.configuration_fingerprint,
            "phrase_policy": VISUAL_XAI_PHRASE_POLICY,
            "heavy_scorer_batches": 0,
            "maps": (),
            "cache_key": cache_key,
        }
        artifact_id = VisualAttributionArtifact.compute_identity(
            {**payload, "maps": []}
        )
        return VisualAttributionArtifact(artifact_id=artifact_id, **payload)

    def _attribute_frame(
        self,
        snapshot: VisualObservationSnapshot,
        reference: Any,
        all_frames: Sequence[VisualFrame],
        raw_generation: str,
    ) -> VisualAttributionArtifact:
        model_fingerprint = self._model_fingerprint(snapshot, self.scorer)
        source_frame = self._source_frame(reference.frame_id, all_frames)
        cache_key = self._cache_key(
            snapshot,
            source_frame
            if source_frame is not None
            else VisualFrame(
                frame_id=reference.frame_id,
                frame_path=Path("unavailable"),
                frame_index=reference.frame_index,
                timestamp_sec=reference.timestamp_seconds,
                frame_rank=reference.frame_rank,
                image_sha256=reference.image_sha256,
                retrieval_rank=reference.retrieval_rank,
            ),
            all_frames,
            model_fingerprint,
        )
        cached = self._load_cache(cache_key)
        if cached is not None:
            return self._rebind_cached_artifact(cached, snapshot)
        if source_frame is None or not source_frame.frame_path.is_file():
            artifact = self._unavailable(
                snapshot,
                reference,
                model_fingerprint,
                cache_key,
                "source_frame_unavailable",
            )
            self._write_cache(artifact)
            return artifact
        if self._sha256(source_frame.frame_path) != reference.image_sha256:
            artifact = self._unavailable(
                snapshot,
                reference,
                model_fingerprint,
                cache_key,
                "source_frame_fingerprint_mismatch",
            )
            self._write_cache(artifact)
            return artifact
        try:
            spans = self._target_spans(
                snapshot, raw_generation, self.config.maximum_phrase_count
            )
        except ValueError:
            artifact = self._unavailable(
                snapshot,
                reference,
                model_fingerprint,
                cache_key,
                "target_span_unavailable",
            )
            self._write_cache(artifact)
            return artifact

        started = self._clock()
        cv2_module, numpy_module = self._image_dependencies()
        image = cv2_module.imread(str(source_frame.frame_path), cv2_module.IMREAD_COLOR)
        if image is None or getattr(image, "size", 0) == 0:
            artifact = self._unavailable(
                snapshot,
                reference,
                model_fingerprint,
                cache_key,
                "source_frame_unavailable",
            )
            self._write_cache(artifact)
            return artifact
        blurred = cv2_module.GaussianBlur(
            image,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            sigmaX=0,
        )
        baseline = self.scorer.score_target_logprob_batch(
            [tuple(all_frames)], raw_generation, spans
        )[0]
        heavy_scorer_batches = 1
        raw_by_span = {span.span_id: [] for span in spans}
        directory, _ = self._cache_paths(cache_key)
        temporary_directory = directory / "working"
        try:
            cell_count = self.config.grid_rows * self.config.grid_columns
            for batch_start in range(0, cell_count, self.config.attribution_batch_size):
                if self._clock() - started > self.config.timeout_seconds:
                    raise TimeoutError("visual XAI attribution timeout")
                batch_frames = []
                temporary_paths = []
                for cell_index in range(
                    batch_start,
                    min(batch_start + self.config.attribution_batch_size, cell_count),
                ):
                    row, column = divmod(cell_index, self.config.grid_columns)
                    path = temporary_directory / f"variant_r{row:02d}_c{column:02d}.png"
                    self._write_variant(
                        image, blurred, row, column, path, cv2_module
                    )
                    temporary_paths.append(path)
                    batch_frames.append(
                        tuple(
                            replace(frame, frame_path=path)
                            if frame.frame_id == source_frame.frame_id
                            else frame
                            for frame in all_frames
                        )
                    )
                scores = self.scorer.score_target_logprob_batch(
                    batch_frames, raw_generation, spans
                )
                heavy_scorer_batches += 1
                if len(scores) != len(batch_frames) or not all(
                    isinstance(score, VisualTargetScore) for score in scores
                ):
                    raise ValueError("visual scorer returned an invalid batch")
                for score in scores:
                    for span in spans:
                        raw_by_span[span.span_id].append(
                            baseline.log_probability(span.span_id)
                            - score.log_probability(span.span_id)
                        )
                for path in temporary_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
        finally:
            if temporary_directory.exists():
                for child in temporary_directory.iterdir():
                    if child.is_file():
                        child.unlink()
                try:
                    temporary_directory.rmdir()
                except OSError:
                    pass

        maps = []
        for span in spans:
            raw_values = raw_by_span[span.span_id]
            normalized_values = self._normalize_positive(raw_values)
            overlay_path = directory / f"overlay_{span.span_id}.png"
            self._write_overlay(
                image,
                normalized_values,
                overlay_path,
                cv2_module,
                numpy_module,
            )
            maps.append(
                VisualAttributionMap(
                    map_id=span.span_id,
                    scope=span.scope,
                    label=span.label,
                    target_start_character=span.start_character,
                    target_end_character=span.end_character,
                    target_token_count=baseline.token_count(span.span_id),
                    baseline_target_log_probability=baseline.log_probability(
                        span.span_id
                    ),
                    raw_importance=self._matrix(
                        raw_values,
                        self.config.grid_rows,
                        self.config.grid_columns,
                    ),
                    normalized_importance=self._matrix(
                        normalized_values,
                        self.config.grid_rows,
                        self.config.grid_columns,
                    ),
                    overlay_image_path=overlay_path,
                )
            )
        payload = {
            "schema_version": VISUAL_XAI_SCHEMA_VERSION,
            "artifact_type": VISUAL_XAI_ARTIFACT_TYPE,
            "status": "available",
            "unavailable_reason": None,
            "method": QWEN_OCCLUSION_METHOD,
            "model_id": snapshot.qwen_model_id,
            "model_revision": snapshot.qwen_revision,
            "model_fingerprint": model_fingerprint,
            "source_frame_id": reference.frame_id,
            "source_frame_index": reference.frame_index,
            "source_timestamp_seconds": reference.timestamp_seconds,
            "source_frame_sha256": reference.image_sha256,
            "observation_unit_id": snapshot.unit_id,
            "observation_text": snapshot.observation_text,
            "observation_text_sha256": self._text_sha256(snapshot.observation_text),
            "raw_generation_sha256": snapshot.raw_generation_sha256,
            "profile": self.config.profile,
            "grid_rows": self.config.grid_rows,
            "grid_columns": self.config.grid_columns,
            "attribution_batch_size": self.config.attribution_batch_size,
            "occlusion_baseline": QWEN_OCCLUSION_BASELINE,
            "configuration_version": self.config.configuration_version,
            "configuration_fingerprint": self.config.configuration_fingerprint,
            "phrase_policy": VISUAL_XAI_PHRASE_POLICY,
            "heavy_scorer_batches": heavy_scorer_batches,
            "maps": tuple(maps),
            "cache_key": cache_key,
        }
        identity_payload = {
            **payload,
            "maps": [item.to_dict() for item in maps],
        }
        artifact = VisualAttributionArtifact(
            artifact_id=VisualAttributionArtifact.compute_identity(identity_payload),
            **payload,
        )
        self._write_cache(artifact)
        return artifact

    def cached_artifacts(
        self,
        visual_snapshot: VisualObservationSnapshot,
        selected_frames: Iterable[VisualFrame],
    ) -> Optional[Tuple[VisualAttributionArtifact, ...]]:
        """Return a complete persisted cache hit without invoking model scoring."""

        if not isinstance(visual_snapshot, VisualObservationSnapshot):
            raise TypeError("visual_snapshot must be a VisualObservationSnapshot")
        frames = tuple(selected_frames)
        if not all(isinstance(item, VisualFrame) for item in frames):
            raise TypeError("selected_frames must contain VisualFrame only")
        model_fingerprint = self._model_fingerprint(visual_snapshot, self.scorer)
        artifacts = []
        for reference in visual_snapshot.frame_references:
            source_frame = self._source_frame(reference.frame_id, frames)
            if source_frame is None:
                return None
            cache_key = self._cache_key(
                visual_snapshot,
                source_frame,
                frames,
                model_fingerprint,
            )
            artifact = self._load_cache(cache_key)
            if artifact is None:
                return None
            artifacts.append(
                self._rebind_cached_artifact(artifact, visual_snapshot)
            )
        return tuple(artifacts)

    def attribute(
        self,
        visual_snapshots: Iterable[VisualObservationSnapshot],
        selected_frames: Iterable[VisualFrame],
        raw_generation: str,
    ) -> List[VisualAttributionArtifact]:
        snapshots = tuple(visual_snapshots)
        frames = tuple(selected_frames)
        if not all(isinstance(item, VisualObservationSnapshot) for item in snapshots):
            raise TypeError("visual_snapshots must contain VisualObservationSnapshot only")
        if not all(isinstance(item, VisualFrame) for item in frames):
            raise TypeError("selected_frames must contain VisualFrame only")
        if not isinstance(raw_generation, str) or not raw_generation:
            raise ValueError("raw_generation is required for fixed-target attribution")
        artifacts = []
        for snapshot in snapshots:
            for reference in snapshot.frame_references:
                try:
                    artifacts.append(
                        self._attribute_frame(
                            snapshot, reference, frames, raw_generation
                        )
                    )
                except TimeoutError:
                    model_fingerprint = self._model_fingerprint(snapshot, self.scorer)
                    source_frame = self._source_frame(reference.frame_id, frames)
                    cache_key = self._cache_key(
                        snapshot,
                        source_frame
                        if source_frame is not None
                        else VisualFrame(
                            frame_id=reference.frame_id,
                            frame_path=Path("unavailable"),
                            frame_index=reference.frame_index,
                            timestamp_sec=reference.timestamp_seconds,
                            frame_rank=reference.frame_rank,
                            image_sha256=reference.image_sha256,
                            retrieval_rank=reference.retrieval_rank,
                        ),
                        frames,
                        model_fingerprint,
                    )
                    artifact = self._unavailable(
                            snapshot,
                            reference,
                            model_fingerprint,
                            cache_key,
                            "attribution_timeout",
                    )
                    self._write_cache(artifact)
                    artifacts.append(artifact)
                except Exception:
                    model_fingerprint = self._model_fingerprint(snapshot, self.scorer)
                    source_frame = self._source_frame(reference.frame_id, frames)
                    cache_key = self._cache_key(
                        snapshot,
                        source_frame
                        if source_frame is not None
                        else VisualFrame(
                            frame_id=reference.frame_id,
                            frame_path=Path("unavailable"),
                            frame_index=reference.frame_index,
                            timestamp_sec=reference.timestamp_seconds,
                            frame_rank=reference.frame_rank,
                            image_sha256=reference.image_sha256,
                            retrieval_rank=reference.retrieval_rank,
                        ),
                        frames,
                        model_fingerprint,
                    )
                    artifact = self._unavailable(
                            snapshot,
                            reference,
                            model_fingerprint,
                            cache_key,
                            "attribution_failed",
                    )
                    self._write_cache(artifact)
                    artifacts.append(artifact)
        return artifacts
