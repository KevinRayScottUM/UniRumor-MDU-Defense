"""Run local-only video audio decoding and Whisper transcription."""

import argparse
import json
from pathlib import Path

from services.video_asr_runner import VideoASRRunner
from services.video_audio_decoder import VideoAudioDecoder
from services.whisper_asr_service import WhisperASRConfig, WhisperASRService


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--verify-asset-sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    decoder = VideoAudioDecoder(max_duration_seconds=args.max_duration)
    asr_service = WhisperASRService(
        WhisperASRConfig(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            verify_asset_sha256=args.verify_asset_sha256,
        )
    )
    result = VideoASRRunner(decoder, asr_service).run(
        session_id=args.session_id,
        claim=args.claim,
        video_path=args.video,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
