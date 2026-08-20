"""Lazy construction and filesystem preflight for the real service graph."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from schemas import ProductionRuntimeConfig
from services.frozen_g1_runner import FrozenG1Runner, FrozenG1RunnerConfig
from services.multimodal_exposure_composer import MultimodalExposureComposer
from services.paddle_ocr_service import PaddleOCRService, PaddleOCRServiceConfig
from services.qwen_visual_observer import (
    QwenVisualObserver,
    QwenVisualObserverConfig,
)
from services.siglip_visual_retriever import (
    SigLIPRetrieverConfig,
    SigLIPVisualRetriever,
)
from services.video_asr_runner import VideoASRRunner
from services.video_audio_decoder import VideoAudioDecoder
from services.video_frame_sampler import VideoFrameSampler
from services.video_multimodal_runner import VideoMultimodalRunner
from services.video_ocr_runner import VideoOCRRunner
from services.video_text_ocr_runner import VideoTextOCRRunner
from services.video_visual_runner import VideoVisualRunner
from services.whisper_asr_service import WhisperASRConfig, WhisperASRService


@dataclass(frozen=True)
class ProductionRuntimeServices:
    config: ProductionRuntimeConfig
    video_audio_decoder: VideoAudioDecoder
    whisper_asr_service: WhisperASRService
    video_asr_runner: VideoASRRunner
    video_frame_sampler: VideoFrameSampler
    paddle_ocr_service: PaddleOCRService
    video_ocr_runner: VideoOCRRunner
    exposure_composer: MultimodalExposureComposer
    frozen_g1_runner: FrozenG1Runner
    video_text_ocr_runner: VideoTextOCRRunner
    siglip_retriever: SigLIPVisualRetriever
    qwen_observer: QwenVisualObserver
    video_visual_runner: VideoVisualRunner
    video_multimodal_runner: VideoMultimodalRunner


class ProductionRuntimeFactory:
    def __init__(self, config: ProductionRuntimeConfig) -> None:
        if not isinstance(config, ProductionRuntimeConfig):
            raise TypeError("config must be a ProductionRuntimeConfig")
        self.config = config

    @staticmethod
    def _require_directory(path: Path, field_name: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"{field_name} must be a directory: {path}")

    @staticmethod
    def _require_file(path: Path, field_name: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{field_name} must be a regular file: {path}")

    @staticmethod
    def _require_executable(executable: str, field_name: str) -> None:
        expanded = Path(executable).expanduser()
        is_explicit_path = expanded.is_absolute() or os.sep in executable
        if is_explicit_path:
            candidate = expanded.resolve()
            if not candidate.exists():
                raise FileNotFoundError(
                    f"{field_name} does not exist: {candidate}"
                )
            if not candidate.is_file():
                raise ValueError(
                    f"{field_name} must be a regular file: {candidate}"
                )
            if not os.access(candidate, os.X_OK):
                raise ValueError(f"{field_name} is not executable: {candidate}")
            return
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"{field_name} is not resolvable through PATH: {executable}"
            )
        candidate = Path(resolved)
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"{field_name} is not executable: {candidate}")

    def preflight(self) -> None:
        config = self.config
        self._require_directory(config.whisper.model_path, "whisper.model_path")
        self._require_directory(
            config.ocr.detector_model_path, "ocr.detector_model_path"
        )
        self._require_directory(
            config.ocr.recognizer_model_path, "ocr.recognizer_model_path"
        )
        self._require_directory(
            config.ocr.cudnn8_library_path, "ocr.cudnn8_library_path"
        )
        self._require_executable(
            config.ocr.python_executable, "ocr.python_executable"
        )
        self._require_directory(config.siglip.model_path, "siglip.model_path")
        self._require_directory(config.qwen.model_path, "qwen.model_path")
        self._require_directory(
            config.frozen_g1.unirumor_root, "frozen_g1.unirumor_root"
        )
        self._require_file(
            config.frozen_g1.phase4a_infer, "frozen_g1.phase4a_infer"
        )
        self._require_file(
            config.frozen_g1.phase4a_config, "frozen_g1.phase4a_config"
        )
        self._require_executable(
            config.frozen_g1.python_executable,
            "frozen_g1.python_executable",
        )

    def build(self, *, run_preflight: bool = True) -> ProductionRuntimeServices:
        if run_preflight:
            self.preflight()
        config = self.config

        video_audio_decoder = VideoAudioDecoder()
        whisper_asr_service = WhisperASRService(
            WhisperASRConfig(
                model_path=config.whisper.model_path,
                device=config.whisper.device,
                dtype=config.whisper.dtype,
                verify_asset_sha256=True,
            )
        )
        video_asr_runner = VideoASRRunner(
            decoder=video_audio_decoder,
            asr_service=whisper_asr_service,
        )

        ocr_cache_root = config.cache_root / "ocr"
        video_frame_sampler = VideoFrameSampler(cache_root=ocr_cache_root)
        paddle_ocr_service = PaddleOCRService(
            PaddleOCRServiceConfig(
                detector_model_path=config.ocr.detector_model_path,
                recognizer_model_path=config.ocr.recognizer_model_path,
                cache_root=ocr_cache_root,
                python_executable=config.ocr.python_executable,
                device=config.ocr.device,
                timeout_seconds=config.ocr.timeout_seconds,
                cudnn8_library_path=config.ocr.cudnn8_library_path,
            )
        )
        video_ocr_runner = VideoOCRRunner(
            frame_sampler=video_frame_sampler,
            ocr_service=paddle_ocr_service,
        )

        exposure_composer = MultimodalExposureComposer()
        frozen_g1_runner = FrozenG1Runner(
            FrozenG1RunnerConfig(
                unirumor_root=config.frozen_g1.unirumor_root,
                python_executable=config.frozen_g1.python_executable,
                phase4a_infer=config.frozen_g1.phase4a_infer,
                phase4a_config=config.frozen_g1.phase4a_config,
                device=config.frozen_g1.device,
                timeout_seconds=config.frozen_g1.timeout_seconds,
                cache_root=config.cache_root / "g1",
                output_root=config.output_root / "g1",
            )
        )
        video_text_ocr_runner = VideoTextOCRRunner(
            video_asr_runner=video_asr_runner,
            video_ocr_runner=video_ocr_runner,
            exposure_composer=exposure_composer,
            frozen_g1_runner=None,
        )

        siglip_retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(
                model_path=config.siglip.model_path,
                cache_root=config.cache_root / "visual",
                device=config.siglip.device,
            )
        )
        qwen_observer = QwenVisualObserver(
            QwenVisualObserverConfig(
                model_path=config.qwen.model_path,
                device=config.qwen.device,
            )
        )
        video_visual_runner = VideoVisualRunner(
            retriever=siglip_retriever,
            observer=qwen_observer,
        )
        video_multimodal_runner = VideoMultimodalRunner(
            video_text_ocr_runner=video_text_ocr_runner,
            video_visual_runner=video_visual_runner,
            frozen_g1_runner=frozen_g1_runner,
        )

        return ProductionRuntimeServices(
            config=config,
            video_audio_decoder=video_audio_decoder,
            whisper_asr_service=whisper_asr_service,
            video_asr_runner=video_asr_runner,
            video_frame_sampler=video_frame_sampler,
            paddle_ocr_service=paddle_ocr_service,
            video_ocr_runner=video_ocr_runner,
            exposure_composer=exposure_composer,
            frozen_g1_runner=frozen_g1_runner,
            video_text_ocr_runner=video_text_ocr_runner,
            siglip_retriever=siglip_retriever,
            qwen_observer=qwen_observer,
            video_visual_runner=video_visual_runner,
            video_multimodal_runner=video_multimodal_runner,
        )
