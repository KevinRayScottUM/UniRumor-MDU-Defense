"""Isolated PP-OCRv5 worker. This is the only module that imports PaddleOCR."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from services.paddle_ocr_service import (
    DETECTOR_MODEL_ID,
    DETECTOR_REVISION,
    DETECTOR_RUNTIME_TREE_SHA256,
    OCRDetection,
    OCR_SCHEMA_VERSION,
    RECOGNIZER_MODEL_ID,
    RECOGNIZER_REVISION,
    RECOGNIZER_RUNTIME_TREE_SHA256,
    frozen_model_metadata,
    polygon_to_bbox,
)


RUNTIME_TREE_FILES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)


def runtime_tree_sha256(model_path: Path) -> str:
    """Hash the canonical manifest for the exact frozen runtime files."""
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"OCR model directory not found: {root}")
    rows = []
    for name in RUNTIME_TREE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"OCR runtime asset missing: {path}")
        file_digest = hashlib.sha256()
        file_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
                file_size += len(chunk)
        rows.append(
            {
                "path": name,
                "size": file_size,
                "sha256": file_digest.hexdigest(),
            }
        )
    canonical_manifest = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_manifest).hexdigest()


def _validate_model(
    model: Dict[str, Any], model_id: str, revision: str, expected_hash: str
) -> Path:
    expected = {
        "model_id": model_id,
        "revision": revision,
        "runtime_tree_sha256": expected_hash,
    }
    for field, value in expected.items():
        if model.get(field) != value:
            raise ValueError(f"OCR worker model {field} mismatch")
    path = Path(model["local_path"])
    actual_hash = runtime_tree_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"OCR runtime tree SHA256 mismatch for {model_id}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return path


def _mapping(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("unsupported PaddleOCR result object")
    nested = payload.get("res", payload)
    if not isinstance(nested, dict):
        raise ValueError("PaddleOCR result payload is malformed")
    return nested


def normalize_paddle_results(results: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for result in results:
        payload = _mapping(result)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        polygons = payload.get("rec_polys")
        if polygons is None:
            polygons = payload.get("dt_polys", [])
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if hasattr(polygons, "tolist"):
            polygons = polygons.tolist()
        if not (len(texts) == len(scores) == len(polygons)):
            raise ValueError("PaddleOCR result arrays have inconsistent lengths")
        for text, confidence, polygon in zip(texts, scores, polygons):
            text = str(text)
            if not text.strip():
                continue
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            normalized_polygon = [
                [float(value) for value in point] for point in points
            ]
            detection = OCRDetection.from_dict(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "polygon": normalized_polygon,
                    "runtime_bbox": polygon_to_bbox(normalized_polygon),
                }
            )
            normalized.append(detection.to_dict())
    return normalized


def _create_engine(detector_path: Path, recognizer_path: Path, device: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_dir=str(detector_path),
        text_recognition_model_dir=str(recognizer_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )


def run_request(request: Dict[str, Any]) -> Dict[str, Any]:
    if request.get("schema_version") != OCR_SCHEMA_VERSION:
        raise ValueError("OCR worker request schema_version mismatch")
    session_id = str(request.get("session_id", ""))
    if not session_id:
        raise ValueError("OCR worker session_id is required")
    models = request.get("models") or {}
    detector_path = _validate_model(
        models.get("detector") or {},
        DETECTOR_MODEL_ID,
        DETECTOR_REVISION,
        DETECTOR_RUNTIME_TREE_SHA256,
    )
    recognizer_path = _validate_model(
        models.get("recognizer") or {},
        RECOGNIZER_MODEL_ID,
        RECOGNIZER_REVISION,
        RECOGNIZER_RUNTIME_TREE_SHA256,
    )
    engine = _create_engine(detector_path, recognizer_path, str(request["device"]))
    frame_results = []
    for frame in request.get("frames", []):
        frame_path = Path(frame["frame_path"])
        if not frame_path.is_file():
            raise FileNotFoundError(f"OCR input frame not found: {frame_path}")
        prediction = engine.predict(input=str(frame_path))
        result = dict(frame)
        result["detections"] = normalize_paddle_results(prediction)
        frame_results.append(result)
    return {
        "schema_version": OCR_SCHEMA_VERSION,
        "status": "ok",
        "session_id": session_id,
        "models": frozen_model_metadata(),
        "frames": frame_results,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_request(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
