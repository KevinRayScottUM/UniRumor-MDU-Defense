"""Parent-side subprocess bridge for the isolated PaddleOCR worker."""

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from services.cache_manager import safe_target, write_json


OCR_SCHEMA_VERSION = 1
DETECTOR_MODEL_ID = "PaddlePaddle/PP-OCRv5_server_det"
DETECTOR_REVISION = "ca867c897ecbca8873081573a802ad70d499cb94"
DETECTOR_RUNTIME_TREE_SHA256 = (
    "a6e8aae048291ebff5d6b604ccda060ccf516ed82d5f8e5f4f4421e762395983"
)
RECOGNIZER_MODEL_ID = "PaddlePaddle/PP-OCRv5_server_rec"
RECOGNIZER_REVISION = "b26c3587fda8da3c8ec0ce357214b4d661ff1558"
RECOGNIZER_RUNTIME_TREE_SHA256 = (
    "248824aeede7ff94190ff2b82cce0679d89868713c749cc8cd3f6678006be259"
)
DEFAULT_CUDNN8_LIBRARY_PATH = Path(
    "/scr/user/kevin2002/TensorCat/runtime_libs/cudnn8-cu11/nvidia/cudnn/lib"
)


def polygon_to_bbox(polygon: Iterable[Iterable[Any]]) -> List[float]:
    points = []
    for point in polygon:
        values = list(point)
        if len(values) != 2:
            raise ValueError("OCR polygon points must contain exactly x and y")
        normalized = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("OCR polygon coordinates must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("OCR polygon coordinates must be finite")
            normalized.append(number)
        points.append(normalized)
    if len(points) < 3:
        raise ValueError("OCR polygon must contain at least three points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


@dataclass(frozen=True)
class OCRDetection:
    text: str
    confidence: float
    polygon: List[List[float]]
    runtime_bbox: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "polygon": [list(point) for point in self.polygon],
            "runtime_bbox": list(self.runtime_bbox),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRDetection":
        if not isinstance(data, dict):
            raise ValueError("OCR detection must be a mapping")
        confidence = data.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("OCR confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("OCR confidence must be finite and within [0, 1]")
        polygon = [list(point) for point in data.get("polygon", [])]
        computed_bbox = polygon_to_bbox(polygon)
        supplied_bbox = [float(value) for value in data.get("runtime_bbox", [])]
        if supplied_bbox != computed_bbox:
            raise ValueError("OCR runtime_bbox does not match its polygon")
        return cls(
            text=str(data.get("text", "")),
            confidence=confidence,
            polygon=polygon,
            runtime_bbox=computed_bbox,
        )


@dataclass(frozen=True)
class OCRFrameResult:
    frame_rank: int
    frame_index: int
    timestamp_sec: float
    frame_id: str
    frame_path: Path
    detections: List[OCRDetection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_rank": self.frame_rank,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "frame_id": self.frame_id,
            "frame_path": str(self.frame_path),
            "detections": [detection.to_dict() for detection in self.detections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRFrameResult":
        return cls(
            frame_rank=int(data["frame_rank"]),
            frame_index=int(data["frame_index"]),
            timestamp_sec=float(data["timestamp_sec"]),
            frame_id=str(data["frame_id"]),
            frame_path=Path(data["frame_path"]),
            detections=[
                OCRDetection.from_dict(item) for item in data.get("detections", [])
            ],
        )


def frozen_model_metadata() -> Dict[str, Dict[str, str]]:
    return {
        "detector": {
            "model_id": DETECTOR_MODEL_ID,
            "revision": DETECTOR_REVISION,
            "runtime_tree_sha256": DETECTOR_RUNTIME_TREE_SHA256,
        },
        "recognizer": {
            "model_id": RECOGNIZER_MODEL_ID,
            "revision": RECOGNIZER_REVISION,
            "runtime_tree_sha256": RECOGNIZER_RUNTIME_TREE_SHA256,
        },
    }


@dataclass(frozen=True)
class PaddleOCRServiceConfig:
    detector_model_path: Path
    recognizer_model_path: Path
    cache_root: Path
    python_executable: str = sys.executable
    worker_module: str = "services.paddle_ocr_worker"
    device: str = "gpu:0"
    timeout_seconds: float = 300.0
    cudnn8_library_path: Path = DEFAULT_CUDNN8_LIBRARY_PATH

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_model_path", Path(self.detector_model_path))
        object.__setattr__(self, "recognizer_model_path", Path(self.recognizer_model_path))
        object.__setattr__(self, "cache_root", Path(self.cache_root).resolve())
        object.__setattr__(self, "cudnn8_library_path", Path(self.cudnn8_library_path))
        if not self.python_executable:
            raise ValueError("OCR worker Python executable is required")
        if not self.worker_module:
            raise ValueError("OCR worker module is required")
        if not self.device:
            raise ValueError("OCR device is required")
        if self.timeout_seconds <= 0:
            raise ValueError("OCR worker timeout must be positive")


class PaddleOCRService:
    def __init__(
        self,
        config: PaddleOCRServiceConfig,
        subprocess_run: Callable[..., object] = subprocess.run,
    ) -> None:
        self.config = config
        self._subprocess_run = subprocess_run

    def _paths(self, session_id: str) -> Sequence[Path]:
        return (
            safe_target(
                self.config.cache_root, "ocr_worker", f"{session_id}.request.json"
            ),
            safe_target(
                self.config.cache_root, "ocr_worker", f"{session_id}.result.json"
            ),
        )

    def _environment(self) -> Dict[str, str]:
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["DISABLE_MODEL_SOURCE_CHECK"] = "True"
        isolated = str(self.config.cudnn8_library_path)
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{isolated}{os.pathsep}{existing}" if existing else isolated
        )
        return environment

    def _request(self, session_id: str, frames: Iterable[Any]) -> Dict[str, Any]:
        serialized_frames = []
        for frame in frames:
            payload = frame.to_dict() if hasattr(frame, "to_dict") else dict(frame)
            frame_path = Path(payload["frame_path"])
            if not frame_path.is_file():
                raise FileNotFoundError(f"sampled OCR frame not found: {frame_path}")
            serialized_frames.append(payload)
        models = frozen_model_metadata()
        models["detector"]["local_path"] = str(self.config.detector_model_path)
        models["recognizer"]["local_path"] = str(self.config.recognizer_model_path)
        return {
            "schema_version": OCR_SCHEMA_VERSION,
            "session_id": session_id,
            "device": self.config.device,
            "models": models,
            "frames": serialized_frames,
        }

    @staticmethod
    def _validate_output(
        payload: Dict[str, Any], session_id: str, requested_frames: List[Dict[str, Any]]
    ) -> List[OCRFrameResult]:
        if not isinstance(payload, dict):
            raise ValueError("OCR worker output must be a mapping")
        if payload.get("schema_version") != OCR_SCHEMA_VERSION:
            raise ValueError("OCR worker schema_version mismatch")
        if payload.get("status") != "ok":
            raise ValueError("OCR worker did not report successful completion")
        if payload.get("session_id") != session_id:
            raise ValueError("OCR worker session_id mismatch")
        if payload.get("models") != frozen_model_metadata():
            raise ValueError("OCR worker frozen model provenance/hash mismatch")
        raw_frames = payload.get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) != len(requested_frames):
            raise ValueError("OCR worker frame result count mismatch")
        results = [OCRFrameResult.from_dict(item) for item in raw_frames]
        for requested, result in zip(requested_frames, results):
            expected = {
                "frame_rank": int(requested["frame_rank"]),
                "frame_index": int(requested["frame_index"]),
                "timestamp_sec": float(requested["timestamp_sec"]),
                "frame_id": str(requested["frame_id"]),
                "frame_path": str(requested["frame_path"]),
            }
            actual = result.to_dict()
            for field, value in expected.items():
                if actual[field] != value:
                    raise ValueError(f"OCR worker frame provenance mismatch: {field}")
        return results

    def predict(self, session_id: str, frames: Iterable[Any]) -> List[OCRFrameResult]:
        request_payload = self._request(session_id, frames)
        request_path, output_path = self._paths(session_id)
        write_json(request_path, request_payload)
        output_path.unlink(missing_ok=True)
        command = [
            self.config.python_executable,
            "-m",
            self.config.worker_module,
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
        started = time.perf_counter()
        try:
            completed = self._subprocess_run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                shell=False,
                env=self._environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("PaddleOCR worker timed out") from exc
        _ = (time.perf_counter() - started) * 1000.0
        if getattr(completed, "returncode", None) != 0:
            stderr = str(getattr(completed, "stderr", "")).strip()
            raise RuntimeError(f"PaddleOCR worker failed: {stderr or 'no stderr'}")
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("PaddleOCR worker output was not created") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("PaddleOCR worker output is malformed JSON") from exc
        return self._validate_output(
            payload, session_id, request_payload["frames"]
        )
