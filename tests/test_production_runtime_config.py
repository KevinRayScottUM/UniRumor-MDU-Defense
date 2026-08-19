import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from schemas import (
    FrozenG1RuntimeConfig,
    OCRRuntimeConfig,
    ProductionRuntimeConfig,
    QwenRuntimeConfig,
    SigLIPRuntimeConfig,
    WhisperRuntimeConfig,
)


def valid_payload():
    return {
        "schema_version": 1,
        "profile": "production",
        "cache_root": "cache/production",
        "output_root": "outputs/production",
        "whisper": {
            "model_path": "assets/whisper",
            "device": "cuda:0",
            "dtype": "float16",
        },
        "ocr": {
            "detector_model_path": "assets/ocr-detector",
            "recognizer_model_path": "assets/ocr-recognizer",
            "python_executable": "/runtime/bin/python",
            "device": "gpu:0",
            "cudnn8_library_path": "runtime/cudnn8/lib",
            "timeout_seconds": 300.0,
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
            "unirumor_root": "external/uni-rumor",
            "python_executable": "/runtime/bin/python",
            "phase4a_infer": "external/phase4a_infer.py",
            "phase4a_config": "external/phase4a_config.json",
            "device": "auto",
            "timeout_seconds": 300.0,
        },
    }


class ProductionRuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repository_root = Path(self.temporary.name) / "repository"
        self.config_dir = self.repository_root / "configs"
        self.config_dir.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, payload=None, name="production.json"):
        path = self.config_dir / name
        path.write_text(
            json.dumps(valid_payload() if payload is None else payload),
            encoding="utf-8",
        )
        return path

    def test_valid_full_config_from_dict(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        self.assertEqual(1, config.schema_version)
        self.assertEqual("production", config.profile)
        self.assertIsInstance(config.whisper, WhisperRuntimeConfig)
        self.assertIsInstance(config.ocr, OCRRuntimeConfig)
        self.assertIsInstance(config.siglip, SigLIPRuntimeConfig)
        self.assertIsInstance(config.qwen, QwenRuntimeConfig)
        self.assertIsInstance(config.frozen_g1, FrozenG1RuntimeConfig)

    def test_valid_json_config_load(self):
        config = ProductionRuntimeConfig.from_json(self.write_config())
        self.assertEqual("cuda:0", config.whisper.device)
        self.assertEqual(300.0, config.frozen_g1.timeout_seconds)

    def test_relative_paths_resolve_from_repository_root(self):
        config = ProductionRuntimeConfig.from_json(self.write_config())
        self.assertEqual(
            (self.repository_root / "cache/production").resolve(),
            config.cache_root,
        )
        self.assertEqual(
            (self.repository_root / "outputs/production").resolve(),
            config.output_root,
        )
        self.assertEqual(
            (self.repository_root / "assets/whisper").resolve(),
            config.whisper.model_path,
        )

    def test_user_home_expansion(self):
        payload = valid_payload()
        payload["siglip"]["model_path"] = "~/models/siglip"
        config = ProductionRuntimeConfig.from_dict(
            payload, base_dir=self.repository_root
        )
        self.assertEqual(
            (Path.home() / "models/siglip").resolve(),
            config.siglip.model_path,
        )

    def test_to_dict_is_json_serializable(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        rendered = json.dumps(config.to_dict(), sort_keys=True)
        self.assertIsInstance(rendered, str)
        self.assertNotIn("PosixPath", rendered)

    def test_from_dict_to_dict_semantic_round_trip(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        self.assertEqual(
            config,
            ProductionRuntimeConfig.from_dict(config.to_dict()),
        )

    def test_configs_are_frozen(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        with self.assertRaises(FrozenInstanceError):
            config.profile = "mock"
        with self.assertRaises(FrozenInstanceError):
            config.whisper.device = "cpu"

    def test_profile_must_equal_production(self):
        payload = valid_payload()
        payload["profile"] = "mock"
        with self.assertRaisesRegex(ValueError, "profile"):
            ProductionRuntimeConfig.from_dict(payload)

    def test_schema_version_must_equal_one(self):
        for value in (0, 2, True, 1.0):
            with self.subTest(value=value):
                payload = valid_payload()
                payload["schema_version"] = value
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    ProductionRuntimeConfig.from_dict(payload)

    def test_cache_and_output_roots_must_differ(self):
        payload = valid_payload()
        payload["output_root"] = payload["cache_root"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            ProductionRuntimeConfig.from_dict(
                payload, base_dir=self.repository_root
            )

    def test_blank_devices_are_rejected(self):
        for section in ("whisper", "ocr", "siglip", "qwen", "frozen_g1"):
            with self.subTest(section=section):
                payload = valid_payload()
                payload[section]["device"] = "  "
                with self.assertRaisesRegex(ValueError, "device"):
                    ProductionRuntimeConfig.from_dict(payload)

    def test_invalid_whisper_dtype_is_rejected(self):
        payload = valid_payload()
        payload["whisper"]["dtype"] = "int8"
        with self.assertRaisesRegex(ValueError, "dtype"):
            ProductionRuntimeConfig.from_dict(payload)

    def test_all_allowed_whisper_dtypes_are_accepted(self):
        for dtype in ("float16", "float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                payload = valid_payload()
                payload["whisper"]["dtype"] = dtype
                config = ProductionRuntimeConfig.from_dict(payload)
                self.assertEqual(dtype, config.whisper.dtype)

    def test_non_positive_ocr_timeout_is_rejected(self):
        for timeout in (0, -1.0, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                payload = valid_payload()
                payload["ocr"]["timeout_seconds"] = timeout
                with self.assertRaisesRegex(ValueError, "timeout"):
                    ProductionRuntimeConfig.from_dict(payload)

    def test_non_positive_frozen_g1_timeout_is_rejected(self):
        for timeout in (0, -1.0, False, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                payload = valid_payload()
                payload["frozen_g1"]["timeout_seconds"] = timeout
                with self.assertRaisesRegex(ValueError, "timeout"):
                    ProductionRuntimeConfig.from_dict(payload)

    def test_blank_python_executable_is_rejected(self):
        for section in ("ocr", "frozen_g1"):
            with self.subTest(section=section):
                payload = valid_payload()
                payload[section]["python_executable"] = ""
                with self.assertRaisesRegex(ValueError, "python_executable"):
                    ProductionRuntimeConfig.from_dict(payload)

    def test_blank_path_string_is_rejected(self):
        payload = valid_payload()
        payload["qwen"]["model_path"] = "  "
        with self.assertRaisesRegex(ValueError, "model_path"):
            ProductionRuntimeConfig.from_dict(payload)

    def assert_unknown_rejected(self, section, field, value=True):
        payload = valid_payload()
        target = payload if section is None else payload[section]
        target[field] = value
        with self.assertRaisesRegex(ValueError, "unknown"):
            ProductionRuntimeConfig.from_dict(payload)

    def test_unknown_top_level_field_is_rejected(self):
        self.assert_unknown_rejected(None, "unexpected")

    def test_unknown_whisper_field_is_rejected(self):
        self.assert_unknown_rejected("whisper", "revision", "mutable")

    def test_unknown_ocr_field_is_rejected(self):
        self.assert_unknown_rejected("ocr", "worker_module", "override")

    def test_unknown_siglip_field_is_rejected(self):
        self.assert_unknown_rejected("siglip", "runtime_tree_sha256", "override")

    def test_unknown_qwen_field_is_rejected(self):
        self.assert_unknown_rejected("qwen", "prompt_policy", "override")

    def test_unknown_frozen_g1_field_is_rejected(self):
        self.assert_unknown_rejected("frozen_g1", "checkpoint", "override")

    def test_attempt_to_configure_max_units_is_rejected(self):
        self.assert_unknown_rejected(None, "max_units", 99)

    def test_attempt_to_configure_pooling_is_rejected(self):
        self.assert_unknown_rejected(None, "pooling", "mean")

    def test_attempt_to_configure_siglip_top_k_is_rejected(self):
        self.assert_unknown_rejected("siglip", "top_k", 99)

    def test_attempt_to_configure_siglip_candidate_count_is_rejected(self):
        self.assert_unknown_rejected("siglip", "candidate_frame_count", 100)

    def test_attempt_to_configure_qwen_max_tokens_is_rejected(self):
        self.assert_unknown_rejected("qwen", "max_new_tokens", 2048)

    def test_parsing_does_not_require_model_paths_to_exist(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        self.assertFalse(config.whisper.model_path.exists())
        self.assertFalse(config.ocr.detector_model_path.exists())
        self.assertFalse(config.qwen.model_path.exists())

    def test_parsing_creates_no_cache_or_output_directories(self):
        config = ProductionRuntimeConfig.from_dict(
            valid_payload(), base_dir=self.repository_root
        )
        self.assertFalse(config.cache_root.exists())
        self.assertFalse(config.output_root.exists())

    def test_malformed_json_is_rejected(self):
        path = self.config_dir / "malformed.json"
        path.write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "malformed"):
            ProductionRuntimeConfig.from_json(path)

    def test_missing_required_nested_section_is_rejected(self):
        payload = valid_payload()
        del payload["qwen"]
        with self.assertRaisesRegex(ValueError, "missing"):
            ProductionRuntimeConfig.from_dict(payload)

    def test_missing_required_nested_field_is_rejected(self):
        payload = valid_payload()
        del payload["ocr"]["device"]
        with self.assertRaisesRegex(ValueError, "missing"):
            ProductionRuntimeConfig.from_dict(payload)

    def test_example_config_loads_without_asset_access(self):
        example = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "production_runtime.example.json"
        )
        config = ProductionRuntimeConfig.from_json(example)
        self.assertEqual("production", config.profile)
        self.assertTrue(config.whisper.model_path.is_absolute())


if __name__ == "__main__":
    unittest.main()
