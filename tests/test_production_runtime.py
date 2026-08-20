import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from schemas import ProductionRuntimeConfig
from services.frozen_g1_runner import FrozenG1Runner
from services.production_runtime import ProductionRuntime
from services.production_runtime_factory import ProductionRuntimeFactory
from services.qwen_visual_observer import QwenVisualObserver
from services.siglip_visual_retriever import SigLIPVisualRetriever
from services.video_text_ocr_runner import VideoTextOCRRunner
from services.video_visual_runner import VideoVisualRunner
from services.whisper_asr_service import WhisperASRService


class FakeMultimodalRunner:
    def __init__(self, result=None, error=None):
        self.result = object() if result is None else result
        self.error = error
        self.calls = []

    def run(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        if self.error is not None:
            raise self.error
        return self.result


class FakeFactory:
    def __init__(self, config, services=None, outcomes=None):
        self.config = config
        self.services_to_return = services
        self.outcomes = list(outcomes or [])
        self.build_calls = []
        self.preflight_calls = 0

    def preflight(self):
        self.preflight_calls += 1
        raise AssertionError("ProductionRuntime must not call preflight directly")

    def build(self, *, run_preflight=True):
        self.build_calls.append(run_preflight)
        outcome = self.outcomes.pop(0) if self.outcomes else self.services_to_return
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ProductionRuntimeTests(unittest.TestCase):
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
                "device": "cuda:0",
                "dtype": "float16",
            },
            "ocr": {
                "detector_model_path": "assets/ocr/detector",
                "recognizer_model_path": "assets/ocr/recognizer",
                "python_executable": "/deployment/ocr-python",
                "device": "gpu:0",
                "cudnn8_library_path": "runtime/cudnn8",
                "timeout_seconds": 300,
            },
            "siglip": {
                "model_path": "assets/siglip",
                "device": "cuda:0",
            },
            "qwen": {
                "model_path": "assets/qwen",
                "device": "cuda:0",
            },
            "frozen_g1": {
                "unirumor_root": "external/unirumor",
                "python_executable": "/deployment/g1-python",
                "phase4a_infer": "external/unirumor/phase4a_infer.py",
                "phase4a_config": "external/unirumor/phase4a_config.json",
                "device": "cuda:0",
                "timeout_seconds": 300,
            },
        }
        self.config = ProductionRuntimeConfig.from_dict(
            copy.deepcopy(self.payload), base_dir=self.root
        )
        self.video_path = self.root / "source-video.mp4"
        self.video_path.write_bytes(b"test video fixture")
        self.runner = FakeMultimodalRunner()
        self.services = SimpleNamespace(
            video_multimodal_runner=self.runner,
            whisper_asr_service=SimpleNamespace(_pipeline=None),
            siglip_retriever=SimpleNamespace(_model=None),
            qwen_observer=SimpleNamespace(_model=None),
        )
        self.factory = FakeFactory(self.config, self.services)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _runtime(self):
        return ProductionRuntime(self.config, factory=self.factory)

    def test_constructor_accepts_config_and_stores_exact_object(self):
        runtime = self._runtime()
        self.assertIs(runtime.config, self.config)
        self.assertIs(runtime.factory, self.factory)

    def test_constructor_rejects_non_production_config(self):
        with self.assertRaisesRegex(
            TypeError, "config must be a ProductionRuntimeConfig"
        ):
            ProductionRuntime({})

    def test_constructor_creates_default_factory_without_starting(self):
        runtime = ProductionRuntime(self.config)
        self.assertIsInstance(runtime.factory, ProductionRuntimeFactory)
        self.assertIs(runtime.factory.config, self.config)
        self.assertFalse(runtime.started)

    def test_constructor_does_not_preflight_or_build(self):
        runtime = self._runtime()
        self.assertEqual(self.factory.preflight_calls, 0)
        self.assertEqual(self.factory.build_calls, [])
        self.assertFalse(runtime.started)

    def test_constructor_creates_no_cache_or_output_roots(self):
        self._runtime()
        self.assertFalse(self.config.cache_root.exists())
        self.assertFalse(self.config.output_root.exists())

    def test_injected_factory_requires_identical_config_object(self):
        equal_but_distinct = ProductionRuntimeConfig.from_dict(
            copy.deepcopy(self.payload), base_dir=self.root
        )
        mismatched_factory = FakeFactory(equal_but_distinct, self.services)
        self.assertEqual(equal_but_distinct, self.config)
        self.assertIsNot(equal_but_distinct, self.config)
        with self.assertRaisesRegex(ValueError, "exact ProductionRuntimeConfig"):
            ProductionRuntime(self.config, factory=mismatched_factory)

    def test_from_json_loads_config_without_starting_or_creating_roots(self):
        config_path = self.root / "configs" / "production.json"
        config_path.parent.mkdir()
        config_path.write_text(json.dumps(self.payload), encoding="utf-8")
        runtime = ProductionRuntime.from_json(config_path)
        self.assertIsInstance(runtime.config, ProductionRuntimeConfig)
        self.assertEqual(runtime.config.profile, "production")
        self.assertFalse(runtime.started)
        self.assertFalse(runtime.config.cache_root.exists())
        self.assertFalse(runtime.config.output_root.exists())

    def test_from_json_calls_config_loader_only(self):
        with (
            patch.object(
                ProductionRuntimeConfig,
                "from_json",
                return_value=self.config,
            ) as config_loader,
            patch.object(
                ProductionRuntimeFactory,
                "build",
                side_effect=AssertionError("build must remain lazy"),
            ),
        ):
            runtime = ProductionRuntime.from_json(self.root / "unused.json")
        config_loader.assert_called_once_with(self.root / "unused.json")
        self.assertFalse(runtime.started)

    def test_started_false_and_services_raises_before_start(self):
        runtime = self._runtime()
        self.assertFalse(runtime.started)
        with self.assertRaisesRegex(RuntimeError, "has not been started"):
            _ = runtime.services

    def test_start_builds_with_preflight_and_returns_exact_bundle(self):
        runtime = self._runtime()
        returned = runtime.start()
        self.assertEqual(self.factory.build_calls, [True])
        self.assertIs(returned, self.services)
        self.assertTrue(runtime.started)
        self.assertIs(runtime.services, self.services)

    def test_repeated_start_is_idempotent(self):
        runtime = self._runtime()
        first = runtime.start()
        second = runtime.start()
        third = runtime.start()
        self.assertIs(first, second)
        self.assertIs(second, third)
        self.assertEqual(self.factory.build_calls, [True])

    def test_failed_start_propagates_and_retry_can_succeed(self):
        failure = RuntimeError("preflight failed")
        factory = FakeFactory(
            self.config,
            outcomes=[failure, self.services],
        )
        runtime = ProductionRuntime(self.config, factory=factory)
        with self.assertRaises(RuntimeError) as caught:
            runtime.start()
        self.assertIs(caught.exception, failure)
        self.assertFalse(runtime.started)
        with self.assertRaises(RuntimeError):
            _ = runtime.services
        self.assertIs(runtime.start(), self.services)
        self.assertTrue(runtime.started)
        self.assertEqual(factory.build_calls, [True, True])

    def test_run_rejects_invalid_session_ids_before_start(self):
        invalid_values = (
            None,
            7,
            "",
            " ",
            "has space",
            "a/b",
            "a\\b",
            "../x",
            "x" * 81,
        )
        runtime = self._runtime()
        for session_id in invalid_values:
            with self.subTest(session_id=session_id):
                with self.assertRaises((TypeError, ValueError)):
                    runtime.run(session_id, "claim", self.video_path)
        self.assertEqual(self.factory.build_calls, [])
        self.assertEqual(self.runner.calls, [])

    def test_run_accepts_safe_session_values(self):
        runtime = self._runtime()
        valid_values = ("a", "Session_01.test-value", "x" * 80)
        for session_id in valid_values:
            with self.subTest(session_id=session_id):
                runtime.run(session_id, "claim", self.video_path)
        self.assertEqual(
            [call[0] for call in self.runner.calls], list(valid_values)
        )
        self.assertEqual(self.factory.build_calls, [True])

    def test_run_rejects_invalid_claim_before_start(self):
        runtime = self._runtime()
        for claim in (None, 7, "", "  \t\n"):
            with self.subTest(claim=claim):
                with self.assertRaises((TypeError, ValueError)):
                    runtime.run("session-1", claim, self.video_path)
        self.assertEqual(self.factory.build_calls, [])
        self.assertEqual(self.runner.calls, [])

    def test_run_preserves_original_claim_exactly(self):
        runtime = self._runtime()
        original_claim = "  Original Ｃｌａｉｍ  "
        runtime.run("session-1", original_claim, self.video_path)
        self.assertEqual(self.runner.calls[0][1], original_claim)

    def test_run_accepts_path_and_string_video_inputs(self):
        runtime = self._runtime()
        runtime.run("path-input", "claim", self.video_path)
        runtime.run("string-input", "claim", str(self.video_path))
        resolved = self.video_path.resolve()
        self.assertEqual(self.runner.calls[0][2], resolved)
        self.assertEqual(self.runner.calls[1][2], resolved)

    def test_run_expands_user_home_and_resolves_before_delegation(self):
        home = self.root / "home"
        home.mkdir()
        video = home / "linked-name.mp4"
        video.write_bytes(b"fixture")
        runtime = self._runtime()
        with patch.dict(os.environ, {"HOME": str(home)}):
            runtime.run("session-1", "claim", "~/linked-name.mp4")
        self.assertEqual(self.runner.calls[0][2], video.resolve())

    def test_run_rejects_missing_file_and_directory_before_start(self):
        runtime = self._runtime()
        directory = self.root / "video-directory"
        directory.mkdir()
        invalid_paths = (self.root / "missing.mp4", directory)
        for video_path in invalid_paths:
            with self.subTest(video_path=video_path):
                with self.assertRaises((FileNotFoundError, ValueError)):
                    runtime.run("session-1", "claim", video_path)
        self.assertEqual(self.factory.build_calls, [])
        self.assertEqual(self.runner.calls, [])

    def test_run_rejects_non_path_input_before_start(self):
        runtime = self._runtime()
        with self.assertRaisesRegex(TypeError, "video_path"):
            runtime.run("session-1", "claim", object())
        self.assertEqual(self.factory.build_calls, [])

    def test_first_valid_run_starts_once_and_delegates_exactly_once(self):
        runtime = self._runtime()
        result = runtime.run("session-1", "  claim  ", self.video_path)
        self.assertTrue(runtime.started)
        self.assertEqual(self.factory.build_calls, [True])
        self.assertEqual(
            self.runner.calls,
            [("session-1", "  claim  ", self.video_path.resolve())],
        )
        self.assertIs(result, self.runner.result)

    def test_run_returns_exact_result_without_mutation(self):
        result = {"warnings": ["original"], "probabilities": {"fake": 0.5}}
        runner = FakeMultimodalRunner(result=result)
        services = SimpleNamespace(video_multimodal_runner=runner)
        factory = FakeFactory(self.config, services)
        runtime = ProductionRuntime(self.config, factory=factory)
        before = copy.deepcopy(result)
        returned = runtime.run("session-1", "claim", self.video_path)
        self.assertIs(returned, result)
        self.assertEqual(result, before)

    def test_two_runs_reuse_bundle_and_multimodal_runner(self):
        runtime = self._runtime()
        runtime.run("session-a", "claim A", self.video_path)
        first_services = runtime.services
        first_runner = runtime.services.video_multimodal_runner
        runtime.run("session-b", "claim B", self.video_path)
        self.assertIs(runtime.services, first_services)
        self.assertIs(runtime.services.video_multimodal_runner, first_runner)
        self.assertEqual(self.factory.build_calls, [True])
        self.assertEqual(len(self.runner.calls), 2)

    def test_wrapper_never_calls_subrunners_or_frozen_g1_directly(self):
        runtime = self._runtime()
        with (
            patch.object(FrozenG1Runner, "run", side_effect=AssertionError),
            patch.object(VideoTextOCRRunner, "run", side_effect=AssertionError),
            patch.object(VideoVisualRunner, "run", side_effect=AssertionError),
        ):
            runtime.run("session-1", "claim", self.video_path)
        self.assertEqual(len(self.runner.calls), 1)

    def test_underlying_value_error_propagates_unchanged(self):
        failure = ValueError("asset verification failed")
        runner = FakeMultimodalRunner(error=failure)
        services = SimpleNamespace(video_multimodal_runner=runner)
        runtime = ProductionRuntime(
            self.config,
            factory=FakeFactory(self.config, services),
        )
        with self.assertRaises(ValueError) as caught:
            runtime.run("session-1", "claim", self.video_path)
        self.assertIs(caught.exception, failure)

    def test_underlying_runtime_error_is_not_converted_to_nei(self):
        failure = RuntimeError("model load failed")
        runner = FakeMultimodalRunner(error=failure)
        services = SimpleNamespace(video_multimodal_runner=runner)
        runtime = ProductionRuntime(
            self.config,
            factory=FakeFactory(self.config, services),
        )
        with self.assertRaises(RuntimeError) as caught:
            runtime.run("session-1", "claim", self.video_path)
        self.assertIs(caught.exception, failure)

    def test_construction_and_start_do_not_eagerly_load_models(self):
        runtime = self._runtime()
        with (
            patch.object(WhisperASRService, "load", side_effect=AssertionError),
            patch.object(
                SigLIPVisualRetriever, "load", side_effect=AssertionError
            ),
            patch.object(QwenVisualObserver, "load", side_effect=AssertionError),
        ):
            runtime.start()
        self.assertIsNone(runtime.services.whisper_asr_service._pipeline)
        self.assertIsNone(runtime.services.siglip_retriever._model)
        self.assertIsNone(runtime.services.qwen_observer._model)

    def test_wrapper_does_not_mutate_config_or_parent_environment(self):
        before_config = self.config.to_dict()
        before_library_path = os.environ.get("LD_LIBRARY_PATH")
        runtime = self._runtime()
        runtime.start()
        self.assertEqual(self.config.to_dict(), before_config)
        self.assertEqual(os.environ.get("LD_LIBRARY_PATH"), before_library_path)

    def test_wrapper_writes_nothing_before_delegated_execution(self):
        before = sorted(
            path.relative_to(self.root)
            for path in self.root.rglob("*")
            if path.is_file()
        )
        runtime = self._runtime()
        runtime.start()
        after = sorted(
            path.relative_to(self.root)
            for path in self.root.rglob("*")
            if path.is_file()
        )
        self.assertEqual(after, before)
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(self.config.cache_root.exists())
        self.assertFalse(self.config.output_root.exists())


if __name__ == "__main__":
    unittest.main()
