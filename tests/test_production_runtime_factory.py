import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from schemas import ProductionRuntimeConfig
from services.frozen_g1_runner import FrozenG1Runner
from services.multimodal_exposure_composer import MultimodalExposureComposer
from services.paddle_ocr_service import PaddleOCRService
from services.production_runtime_factory import (
    ProductionRuntimeFactory,
    ProductionRuntimeServices,
)
from services.qwen_visual_observer import QwenVisualObserver
from services.siglip_visual_retriever import SigLIPVisualRetriever
from services.video_asr_runner import VideoASRRunner
from services.video_audio_decoder import VideoAudioDecoder
from services.video_frame_sampler import VideoFrameSampler
from services.video_multimodal_runner import VideoMultimodalRunner
from services.video_ocr_runner import VideoOCRRunner
from services.video_text_ocr_runner import VideoTextOCRRunner
from services.video_visual_runner import VideoVisualRunner
from services.whisper_asr_service import WhisperASRService


class ProductionRuntimeFactoryTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.payload = {
            "schema_version": 1,
            "profile": "production",
            "cache_root": "cache",
            "output_root": "outputs",
            "whisper": {
                "model_path": "assets/whisper",
                "device": "cuda:1",
                "dtype": "float16",
            },
            "ocr": {
                "detector_model_path": "assets/ocr/detector",
                "recognizer_model_path": "assets/ocr/recognizer",
                "python_executable": sys.executable,
                "device": "gpu:1",
                "cudnn8_library_path": "runtime/cudnn8",
                "timeout_seconds": 123.5,
            },
            "siglip": {
                "model_path": "assets/siglip",
                "device": "cuda:2",
            },
            "qwen": {
                "model_path": "assets/qwen",
                "device": "cuda:3",
            },
            "frozen_g1": {
                "unirumor_root": "external/unirumor",
                "python_executable": sys.executable,
                "phase4a_infer": "external/unirumor/phase4a_infer.py",
                "phase4a_config": "external/unirumor/phase4a_config.json",
                "device": "cuda:4",
                "timeout_seconds": 456.5,
            },
        }

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _config(self, payload=None):
        return ProductionRuntimeConfig.from_dict(
            copy.deepcopy(self.payload if payload is None else payload),
            base_dir=self.root,
        )

    def _create_preflight_inputs(self, payload=None):
        config = self._config(payload)
        directories = (
            config.whisper.model_path,
            config.ocr.detector_model_path,
            config.ocr.recognizer_model_path,
            config.ocr.cudnn8_library_path,
            config.siglip.model_path,
            config.qwen.model_path,
            config.frozen_g1.unirumor_root,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        config.frozen_g1.phase4a_infer.write_text("# fixture\n", encoding="utf-8")
        config.frozen_g1.phase4a_config.write_text("{}\n", encoding="utf-8")
        return config

    def _build(self):
        return ProductionRuntimeFactory(self._config()).build(
            run_preflight=False
        )

    def test_factory_rejects_non_production_runtime_config(self):
        with self.assertRaisesRegex(
            TypeError, "config must be a ProductionRuntimeConfig"
        ):
            ProductionRuntimeFactory({})

    def test_build_without_preflight_accepts_placeholder_paths(self):
        services = self._build()
        self.assertIsInstance(services, ProductionRuntimeServices)

    def test_bundle_exposes_all_expected_service_types(self):
        services = self._build()
        expected = {
            "video_audio_decoder": VideoAudioDecoder,
            "whisper_asr_service": WhisperASRService,
            "video_asr_runner": VideoASRRunner,
            "video_frame_sampler": VideoFrameSampler,
            "paddle_ocr_service": PaddleOCRService,
            "video_ocr_runner": VideoOCRRunner,
            "exposure_composer": MultimodalExposureComposer,
            "frozen_g1_runner": FrozenG1Runner,
            "video_text_ocr_runner": VideoTextOCRRunner,
            "siglip_retriever": SigLIPVisualRetriever,
            "qwen_observer": QwenVisualObserver,
            "video_visual_runner": VideoVisualRunner,
            "video_multimodal_runner": VideoMultimodalRunner,
        }
        for field_name, expected_type in expected.items():
            with self.subTest(field_name=field_name):
                self.assertIsInstance(getattr(services, field_name), expected_type)

    def test_build_is_model_lazy(self):
        services = self._build()
        self.assertIsNone(services.whisper_asr_service._pipeline)
        self.assertIsNone(services.siglip_retriever._model)
        self.assertIsNone(services.qwen_observer._model)

    def test_whisper_configuration_maps_exactly_and_verification_is_forced(self):
        config = self._config()
        services = ProductionRuntimeFactory(config).build(run_preflight=False)
        actual = services.whisper_asr_service.config
        self.assertEqual(actual.model_path, config.whisper.model_path)
        self.assertEqual(actual.device, config.whisper.device)
        self.assertEqual(actual.dtype, config.whisper.dtype)
        self.assertIs(actual.verify_asset_sha256, True)

    def test_ocr_configuration_maps_exactly_and_preserves_worker(self):
        config = self._config()
        services = ProductionRuntimeFactory(config).build(run_preflight=False)
        actual = services.paddle_ocr_service.config
        self.assertEqual(actual.detector_model_path, config.ocr.detector_model_path)
        self.assertEqual(
            actual.recognizer_model_path, config.ocr.recognizer_model_path
        )
        self.assertEqual(actual.python_executable, config.ocr.python_executable)
        self.assertEqual(actual.device, config.ocr.device)
        self.assertEqual(actual.timeout_seconds, config.ocr.timeout_seconds)
        self.assertEqual(
            actual.cudnn8_library_path, config.ocr.cudnn8_library_path
        )
        self.assertEqual(actual.worker_module, "services.paddle_ocr_worker")

    def test_ocr_sampler_preserves_historical_policy_and_cache_namespace(self):
        config = self._config()
        services = ProductionRuntimeFactory(config).build(run_preflight=False)
        expected_cache = config.cache_root / "ocr"
        self.assertEqual(services.video_frame_sampler.frames_per_video, 8)
        self.assertEqual(services.video_frame_sampler.cache_root, expected_cache)
        self.assertEqual(
            services.paddle_ocr_service.config.cache_root, expected_cache
        )

    def test_exposure_composer_preserves_default_limits(self):
        composer = self._build().exposure_composer
        self.assertEqual(composer.max_transcript_units, 12)
        self.assertEqual(composer.max_ocr_units, 6)

    def test_frozen_g1_configuration_maps_exactly(self):
        config = self._config()
        actual = ProductionRuntimeFactory(config).build(
            run_preflight=False
        ).frozen_g1_runner.config
        self.assertEqual(actual.unirumor_root, config.frozen_g1.unirumor_root)
        self.assertEqual(
            actual.python_executable, config.frozen_g1.python_executable
        )
        self.assertEqual(actual.phase4a_infer, config.frozen_g1.phase4a_infer)
        self.assertEqual(actual.phase4a_config, config.frozen_g1.phase4a_config)
        self.assertEqual(actual.device, config.frozen_g1.device)
        self.assertEqual(actual.timeout_seconds, config.frozen_g1.timeout_seconds)
        self.assertEqual(actual.cache_root, config.cache_root / "g1")
        self.assertEqual(actual.output_root, config.output_root / "g1")

    def test_text_ocr_runner_has_no_independent_frozen_g1_runner(self):
        self.assertIsNone(self._build().video_text_ocr_runner.frozen_g1_runner)

    def test_siglip_configuration_preserves_frozen_defaults(self):
        config = self._config()
        actual = ProductionRuntimeFactory(config).build(
            run_preflight=False
        ).siglip_retriever.config
        self.assertEqual(actual.model_path, config.siglip.model_path)
        self.assertEqual(actual.cache_root, config.cache_root / "visual")
        self.assertEqual(actual.device, config.siglip.device)
        self.assertEqual(actual.candidate_frame_count, 12)
        self.assertEqual(actual.top_k, 4)
        self.assertEqual(actual.claim_max_length, 64)

    def test_qwen_configuration_preserves_frozen_default(self):
        config = self._config()
        actual = ProductionRuntimeFactory(config).build(
            run_preflight=False
        ).qwen_observer.config
        self.assertEqual(actual.model_path, config.qwen.model_path)
        self.assertEqual(actual.device, config.qwen.device)
        self.assertEqual(actual.max_new_tokens, 512)

    def test_visual_runner_uses_exact_retriever_and_observer_objects(self):
        services = self._build()
        self.assertIs(
            services.video_visual_runner.retriever, services.siglip_retriever
        )
        self.assertIs(services.video_visual_runner.observer, services.qwen_observer)

    def test_multimodal_runner_uses_exact_subrunner_objects(self):
        services = self._build()
        runner = services.video_multimodal_runner
        self.assertIs(runner.video_text_ocr_runner, services.video_text_ocr_runner)
        self.assertIs(runner.video_visual_runner, services.video_visual_runner)
        self.assertIs(runner.frozen_g1_runner, services.frozen_g1_runner)

    def test_text_and_ocr_subgraph_uses_exact_constructed_objects(self):
        services = self._build()
        self.assertIs(
            services.video_asr_runner.decoder, services.video_audio_decoder
        )
        self.assertIs(
            services.video_asr_runner.asr_service, services.whisper_asr_service
        )
        self.assertIs(
            services.video_ocr_runner.frame_sampler, services.video_frame_sampler
        )
        self.assertIs(
            services.video_ocr_runner.ocr_service, services.paddle_ocr_service
        )
        self.assertIs(
            services.video_text_ocr_runner.video_asr_runner,
            services.video_asr_runner,
        )
        self.assertIs(
            services.video_text_ocr_runner.video_ocr_runner,
            services.video_ocr_runner,
        )
        self.assertIs(
            services.video_text_ocr_runner.exposure_composer,
            services.exposure_composer,
        )

    def test_build_does_not_call_model_load_methods(self):
        with (
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
        ):
            self._build()

    def test_build_does_not_call_inference_or_subprocess(self):
        with (
            patch.object(
                WhisperASRService, "transcribe", side_effect=AssertionError
            ),
            patch.object(
                SigLIPVisualRetriever, "retrieve", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "observe", side_effect=AssertionError),
            patch.object(
                VideoMultimodalRunner, "run", side_effect=AssertionError
            ),
            patch.object(FrozenG1Runner, "run", side_effect=AssertionError),
            patch.object(subprocess, "run", side_effect=AssertionError),
        ):
            self._build()

    def test_build_creates_no_cache_or_output_directories(self):
        config = self._config()
        ProductionRuntimeFactory(config).build(run_preflight=False)
        self.assertFalse(config.cache_root.exists())
        self.assertFalse(config.output_root.exists())

    def test_build_does_not_mutate_configuration_or_parent_ld_library_path(self):
        config = self._config()
        before_config = config.to_dict()
        before_library_path = os.environ.get("LD_LIBRARY_PATH")
        services = ProductionRuntimeFactory(config).build(run_preflight=False)
        self.assertIs(services.config, config)
        self.assertEqual(config.to_dict(), before_config)
        self.assertEqual(os.environ.get("LD_LIBRARY_PATH"), before_library_path)

    def test_preflight_accepts_valid_filesystem_and_sys_executable(self):
        config = self._create_preflight_inputs()
        ProductionRuntimeFactory(config).preflight()

    def test_build_defaults_to_preflight(self):
        with self.assertRaisesRegex(FileNotFoundError, "whisper.model_path"):
            ProductionRuntimeFactory(self._config()).build()

    def test_preflight_rejects_each_missing_model_or_runtime_directory(self):
        field_paths = (
            ("whisper.model_path", lambda config: config.whisper.model_path),
            (
                "ocr.detector_model_path",
                lambda config: config.ocr.detector_model_path,
            ),
            (
                "ocr.recognizer_model_path",
                lambda config: config.ocr.recognizer_model_path,
            ),
            (
                "ocr.cudnn8_library_path",
                lambda config: config.ocr.cudnn8_library_path,
            ),
            ("siglip.model_path", lambda config: config.siglip.model_path),
            ("qwen.model_path", lambda config: config.qwen.model_path),
            (
                "frozen_g1.unirumor_root",
                lambda config: config.frozen_g1.unirumor_root,
            ),
        )
        for field_name, select_path in field_paths:
            with self.subTest(field_name=field_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    original_root = self.root
                    self.root = Path(temporary_directory)
                    try:
                        config = self._create_preflight_inputs()
                        target = select_path(config)
                        target.rename(target.with_name(target.name + "-missing"))
                        with self.assertRaisesRegex(FileNotFoundError, field_name):
                            ProductionRuntimeFactory(config).preflight()
                    finally:
                        self.root = original_root

    def test_preflight_rejects_each_missing_phase4a_file(self):
        fields = (
            ("frozen_g1.phase4a_infer", "phase4a_infer"),
            ("frozen_g1.phase4a_config", "phase4a_config"),
        )
        for field_name, attribute in fields:
            with self.subTest(field_name=field_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    original_root = self.root
                    self.root = Path(temporary_directory)
                    try:
                        config = self._create_preflight_inputs()
                        getattr(config.frozen_g1, attribute).unlink()
                        with self.assertRaisesRegex(FileNotFoundError, field_name):
                            ProductionRuntimeFactory(config).preflight()
                    finally:
                        self.root = original_root

    def test_preflight_rejects_directory_where_phase4a_file_is_required(self):
        config = self._create_preflight_inputs()
        config.frozen_g1.phase4a_infer.unlink()
        config.frozen_g1.phase4a_infer.mkdir()
        with self.assertRaisesRegex(ValueError, "frozen_g1.phase4a_infer"):
            ProductionRuntimeFactory(config).preflight()

    def test_preflight_rejects_missing_ocr_python_executable(self):
        payload = copy.deepcopy(self.payload)
        payload["ocr"]["python_executable"] = str(self.root / "missing-ocr-python")
        config = self._create_preflight_inputs(payload)
        with self.assertRaisesRegex(FileNotFoundError, "ocr.python_executable"):
            ProductionRuntimeFactory(config).preflight()

    def test_preflight_rejects_missing_frozen_g1_python_executable(self):
        payload = copy.deepcopy(self.payload)
        payload["frozen_g1"]["python_executable"] = str(
            self.root / "missing-g1-python"
        )
        config = self._create_preflight_inputs(payload)
        with self.assertRaisesRegex(
            FileNotFoundError, "frozen_g1.python_executable"
        ):
            ProductionRuntimeFactory(config).preflight()

    def test_preflight_rejects_non_executable_explicit_python_path(self):
        executable = self.root / "not-executable"
        executable.write_text("fixture\n", encoding="utf-8")
        executable.chmod(0o644)
        payload = copy.deepcopy(self.payload)
        payload["ocr"]["python_executable"] = str(executable)
        config = self._create_preflight_inputs(payload)
        with self.assertRaisesRegex(ValueError, "ocr.python_executable"):
            ProductionRuntimeFactory(config).preflight()

    def test_preflight_accepts_python_command_resolved_through_path(self):
        payload = copy.deepcopy(self.payload)
        payload["ocr"]["python_executable"] = "ocr-python-command"
        payload["frozen_g1"]["python_executable"] = "g1-python-command"
        config = self._create_preflight_inputs(payload)
        with patch(
            "services.production_runtime_factory.shutil.which",
            return_value=sys.executable,
        ) as which:
            ProductionRuntimeFactory(config).preflight()
        self.assertEqual(
            which.call_args_list[0].args[0], config.ocr.python_executable
        )
        self.assertEqual(
            which.call_args_list[1].args[0],
            config.frozen_g1.python_executable,
        )

    def test_preflight_does_not_load_or_hash_assets(self):
        config = self._create_preflight_inputs()
        with (
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                WhisperASRService, "verify_asset", side_effect=AssertionError
            ),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(
                SigLIPVisualRetriever,
                "_verify_assets",
                side_effect=AssertionError,
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
            patch.object(
                QwenVisualObserver, "_verify_assets", side_effect=AssertionError
            ),
        ):
            ProductionRuntimeFactory(config).preflight()

    def test_preflight_creates_no_roots_and_has_no_parent_environment_side_effect(self):
        config = self._create_preflight_inputs()
        before_library_path = os.environ.get("LD_LIBRARY_PATH")
        ProductionRuntimeFactory(config).preflight()
        self.assertFalse(config.cache_root.exists())
        self.assertFalse(config.output_root.exists())
        self.assertEqual(os.environ.get("LD_LIBRARY_PATH"), before_library_path)


if __name__ == "__main__":
    unittest.main()
