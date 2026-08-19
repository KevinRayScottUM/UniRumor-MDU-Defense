"""Run isolated PP-OCRv5 over deterministically sampled video frames."""

import argparse
import json
from pathlib import Path

from adapters.ocr_unit_adapter import OCRFilterConfig, OCRUnitAdapter
from services.paddle_ocr_service import (
    DEFAULT_CUDNN8_LIBRARY_PATH,
    PaddleOCRService,
    PaddleOCRServiceConfig,
)
from services.video_frame_sampler import VideoFrameSampler
from services.video_ocr_runner import VideoOCRRunner


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--detector-model-path", required=True, type=Path)
    parser.add_argument("--recognizer-model-path", required=True, type=Path)
    parser.add_argument("--python-executable", default=__import__("sys").executable)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--cudnn8-library-path", type=Path, default=DEFAULT_CUDNN8_LIBRARY_PATH
    )
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-text-length", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    sampler = VideoFrameSampler(args.cache_root, args.frames_per_video)
    service = PaddleOCRService(
        PaddleOCRServiceConfig(
            detector_model_path=args.detector_model_path,
            recognizer_model_path=args.recognizer_model_path,
            cache_root=args.cache_root,
            python_executable=args.python_executable,
            device=args.device,
            timeout_seconds=args.timeout,
            cudnn8_library_path=args.cudnn8_library_path,
        )
    )
    adapter = OCRUnitAdapter(
        OCRFilterConfig(
            confidence_threshold=args.confidence_threshold,
            min_normalized_length=args.minimum_text_length,
        )
    )
    result = VideoOCRRunner(sampler, service, adapter).run(
        args.session_id, args.video
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
