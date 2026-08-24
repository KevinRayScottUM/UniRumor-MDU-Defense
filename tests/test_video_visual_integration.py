import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from adapters.phase4a_request_adapter import build_phase4a_request
from adapters.visual_observation_adapter import VisualObservationAdapter
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
)
from services.multimodal_exposure_composer import MultimodalExposureComposer
from services.qwen_visual_observer import (
    QWEN_FROZEN_REVISION,
    QWEN_MODEL_ID,
    QWEN_RUNTIME_TREE_FILES,
    QWEN_RUNTIME_TREE_SHA256,
    QwenVisualObservationResult,
    QwenVisualObserver,
    QwenVisualObserverConfig,
)
from services.siglip_visual_retriever import (
    CLAIM_TOKEN_MAX_LENGTH,
    JPEG_QUALITY,
    SIGLIP_FROZEN_REVISION,
    SIGLIP_MODEL_ID,
    SIGLIP_RUNTIME_TREE_FILES,
    SIGLIP_RUNTIME_TREE_IDENTITY,
    SigLIPRetrievalResult,
    SigLIPRetrieverConfig,
    SigLIPVisualRetriever,
    VisualFrame,
    historical_clip12_positions,
    runtime_tree_sha256,
)
from services.video_multimodal_runner import VideoMultimodalRunner
from services.video_visual_runner import VideoVisualRunner


def visual_frame(index, timestamp=None, path=None, score=None, rank=None):
    return VisualFrame(
        frame_id=f"F{index:03d}",
        frame_path=Path(path or f"/cache/frame-{index}.jpg"),
        frame_index=index * 10,
        timestamp_sec=float(index if timestamp is None else timestamp),
        frame_rank=index,
        image_sha256=f"{index:064x}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def valid_observation(**overrides):
    payload = {
        "observation_type": "scene",
        "observation": "A person stands beside a parked vehicle.",
        "frame_ids": ["F000"],
        "evidence_refs": ["F000"],
    }
    payload.update(overrides)
    return payload


class FakeSigLIPModel:
    calls = []

    def __init__(self):
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )
        self.device = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class FakeSigLIPProcessor:
    calls = []
    input_calls = []

    class Tokenizer:
        model_max_length = 512

        def __init__(self):
            self.calls = []
            self.truncation_side = None

        def __call__(self, claim, **kwargs):
            self.calls.append((claim, kwargs))
            return {"input_ids": list(range(len(claim.split()) + 2))}

    def __init__(self):
        self.tokenizer = self.Tokenizer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return cls()

    def __call__(self, **kwargs):
        type(self).input_calls.append(kwargs)
        if "text" in kwargs:
            token_count = min(
                len(kwargs["text"][0].split()) + 2, kwargs["max_length"]
            )
            return {
                "input_ids": [list(range(kwargs["max_length"]))],
                "attention_mask": [
                    [1] * token_count
                    + [0] * (kwargs["max_length"] - token_count)
                ],
            }
        return {"pixel_values": object()}


class FakeTorch:
    float16 = object()
    float32 = object()
    bfloat16 = object()
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)


class FakeVideoCapture:
    def __init__(self, total_frames=2661, fps=25.0, opened=True):
        self.total_frames = total_frames
        self.fps = fps
        self.opened = opened
        self.current_position = None
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == FakeCV2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        if property_id == FakeCV2.CAP_PROP_FPS:
            return self.fps
        raise AssertionError(f"unexpected OpenCV property: {property_id}")

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        self.current_position = int(value)
        return True

    def read(self):
        return True, f"frame-at-{self.current_position}"

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FPS = 2
    CAP_PROP_POS_FRAMES = 3
    IMWRITE_JPEG_QUALITY = 4

    def __init__(self, capture=None):
        self.capture = capture or FakeVideoCapture()
        self.video_capture_calls = []
        self.imwrite_calls = []

    def VideoCapture(self, path):
        self.video_capture_calls.append(path)
        return self.capture

    def imwrite(self, path, frame, parameters):
        self.imwrite_calls.append((path, frame, parameters))
        Path(path).write_bytes(str(frame).encode("utf-8"))
        return True


class FakeQwenModel(FakeSigLIPModel):
    calls = []


class FakeQwenProcessor(FakeSigLIPProcessor):
    calls = []


class VideoVisualIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.model_path = self.base / "model"
        self.model_path.mkdir()
        FakeSigLIPModel.calls = []
        FakeSigLIPProcessor.calls = []
        FakeSigLIPProcessor.input_calls = []
        FakeQwenModel.calls = []
        FakeQwenProcessor.calls = []
        FakeQwenProcessor.input_calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def test_historical_raw_video_twelve_frame_positions(self):
        self.assertEqual(
            [0, 242, 484, 725, 967, 1209, 1451, 1693, 1935, 2176, 2418, 2660],
            historical_clip12_positions(2661, 12),
        )

    def test_historical_positions_include_first_last_and_jpeg_quality_is_95(self):
        positions = historical_clip12_positions(2661, 12)
        self.assertEqual(0, positions[0])
        self.assertEqual(2660, positions[-1])
        self.assertEqual(95, JPEG_QUALITY)

    def test_historical_frame_ids_are_exactly_three_digits(self):
        self.assertEqual(
            [f"F{index:03d}" for index in range(12)],
            [
                SigLIPVisualRetriever.frame_id_for_rank(index)
                for index in range(12)
            ],
        )

    def test_siglip_extractor_uses_historical_opencv_seek_and_jpeg_contract(self):
        source_dir = self.base / "source"
        source_dir.mkdir()
        video_path = source_dir / "video.mp4"
        video_path.write_bytes(b"source-video-must-not-change")
        source_before = video_path.read_bytes()
        source_mtime_before = video_path.stat().st_mtime_ns
        cache_root = self.base / "cache"
        fake_cv2 = FakeCV2(FakeVideoCapture(total_frames=2661, fps=25.0))
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, cache_root),
            cv2_module=fake_cv2,
        )

        frames = retriever.extract_candidate_frames("session", video_path)

        positions = [
            0,
            242,
            484,
            725,
            967,
            1209,
            1451,
            1693,
            1935,
            2176,
            2418,
            2660,
        ]
        self.assertEqual([f"F{rank:03d}" for rank in range(12)], [item.frame_id for item in frames])
        self.assertEqual(positions, [item.frame_index for item in frames])
        self.assertEqual(
            [
                f"frame_{rank:02d}_{position:08d}.jpg"
                for rank, position in enumerate(positions)
            ],
            [item.frame_path.name for item in frames],
        )
        self.assertEqual(
            [(FakeCV2.CAP_PROP_POS_FRAMES, position) for position in positions],
            fake_cv2.capture.set_calls,
        )
        self.assertTrue(fake_cv2.capture.released)
        self.assertEqual([str(video_path)], fake_cv2.video_capture_calls)
        self.assertTrue(
            all(call[2] == [int(FakeCV2.IMWRITE_JPEG_QUALITY), 95] for call in fake_cv2.imwrite_calls)
        )
        resolved_cache = cache_root.resolve()
        self.assertTrue(
            all(resolved_cache in item.frame_path.resolve().parents for item in frames)
        )
        self.assertEqual([position / 25.0 for position in positions], [item.timestamp_sec for item in frames])
        self.assertEqual(source_before, video_path.read_bytes())
        self.assertEqual(source_mtime_before, video_path.stat().st_mtime_ns)

    def test_short_video_exposes_unique_frames_and_missing_fps_uses_null_time(self):
        video_path = self.base / "short.mp4"
        video_path.write_bytes(b"short-video")
        fake_cv2 = FakeCV2(FakeVideoCapture(total_frames=1, fps=0.0))
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, self.base / "cache"),
            cv2_module=fake_cv2,
        )
        frames = retriever.extract_candidate_frames("short", video_path)
        self.assertEqual(1, len(frames))
        self.assertEqual("F000", frames[0].frame_id)
        self.assertEqual("frame_00_00000000.jpg", frames[0].frame_path.name)
        self.assertIsNone(frames[0].timestamp_sec)

    def test_retrieval_chronology_key_is_timestamp_index_path(self):
        frames = [
            visual_frame(2, timestamp=1.0, path="/z.jpg"),
            visual_frame(1, timestamp=1.0, path="/b.jpg"),
            visual_frame(0, timestamp=0.5, path="/x.jpg"),
        ]
        ordered = sorted(frames, key=SigLIPVisualRetriever.chronology_key)
        self.assertEqual(["F000", "F001", "F002"], [frame.frame_id for frame in ordered])

    def test_retrieval_scores_sort_descending(self):
        frames = [visual_frame(index) for index in range(4)]
        ranked = SigLIPVisualRetriever.rank_frames(frames, [0.2, 0.9, 0.5, 0.7])
        self.assertEqual(["F001", "F003", "F002", "F000"], [frame.frame_id for frame in ranked])
        self.assertEqual([1, 2, 3, 4], [frame.retrieval_rank for frame in ranked])

    def test_equal_score_tie_is_deterministic_chronology(self):
        frames = [
            visual_frame(0, timestamp=3.0),
            visual_frame(1, timestamp=1.0),
            visual_frame(2, timestamp=2.0),
        ]
        first = SigLIPVisualRetriever.rank_frames(frames, [0.5, 0.5, 0.5])
        second = SigLIPVisualRetriever.rank_frames(frames, [0.5, 0.5, 0.5])
        self.assertEqual(["F001", "F002", "F000"], [frame.frame_id for frame in first])
        self.assertEqual([frame.to_dict() for frame in first], [frame.to_dict() for frame in second])

    def test_top_four_relevance_selection_is_restored_chronologically(self):
        frames = [visual_frame(index, timestamp=index) for index in range(12)]
        scores = [0.1] * 12
        scores[1], scores[5], scores[6], scores[8] = 0.9, 0.8, 1.0, 0.7
        selected = SigLIPVisualRetriever.rank_and_select(
            frames, scores, top_k=4
        )
        self.assertEqual(
            ["F001", "F005", "F006", "F008"],
            [frame.frame_id for frame in selected],
        )
        self.assertEqual([2, 3, 1, 4], [frame.retrieval_rank for frame in selected])

    def test_claim_tokenization_configured_maximum_is_64(self):
        tokenizer = SimpleNamespace(model_max_length=512)
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=77)
            )
        )
        self.assertEqual(64, CLAIM_TOKEN_MAX_LENGTH)
        self.assertEqual(
            64,
            SigLIPVisualRetriever.effective_text_max_length(64, tokenizer, model),
        )
        with self.assertRaises(ValueError):
            SigLIPRetrieverConfig(
                self.model_path, self.base / "cache", claim_max_length=63
            )

    def test_frozen_visual_config_limits_reject_nonhistorical_values(self):
        with self.assertRaisesRegex(ValueError, "candidate_frame_count"):
            SigLIPRetrieverConfig(
                self.model_path, self.base / "cache", candidate_frame_count=11
            )
        with self.assertRaisesRegex(ValueError, "top_k=4"):
            SigLIPRetrieverConfig(
                self.model_path, self.base / "cache", top_k=3
            )
        with self.assertRaisesRegex(ValueError, "claim_max_length"):
            SigLIPRetrieverConfig(
                self.model_path, self.base / "cache", claim_max_length=63
            )
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            QwenVisualObserverConfig(self.model_path, max_new_tokens=511)

    def test_claim_token_audit_uses_untruncated_tokenizer_and_processor_input(self):
        processor = FakeSigLIPProcessor()
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, self.base / "cache")
        )
        retriever._processor = processor
        claim = " ".join(f"token-{index}" for index in range(70))
        text_inputs, audit = retriever._prepare_text_inputs(claim, 64)
        self.assertEqual(
            (
                claim,
                {
                    "add_special_tokens": True,
                    "truncation": False,
                    "return_attention_mask": False,
                },
            ),
            processor.tokenizer.calls[0],
        )
        self.assertEqual("right", processor.tokenizer.truncation_side)
        self.assertEqual(
            {
                "text": [claim],
                "padding": "max_length",
                "truncation": True,
                "max_length": 64,
                "return_tensors": "pt",
            },
            FakeSigLIPProcessor.input_calls[0],
        )
        self.assertEqual(64, len(text_inputs["input_ids"][0]))
        self.assertEqual("right_truncate_for_siglip_retrieval_only", audit["policy"])
        self.assertEqual(64, audit["configured_max_length"])
        self.assertEqual(64, audit["effective_max_length"])
        self.assertEqual(72, audit["original_token_count"])
        self.assertEqual(64, audit["model_input_token_count"])
        self.assertTrue(audit["truncated"])
        self.assertEqual("max_length", audit["padding"])
        self.assertTrue(audit["original_claim_preserved_for_mdu"])
        self.assertEqual(hashlib.sha256(claim.encode()).hexdigest(), audit["claim_sha256"])

    def test_model_input_token_count_falls_back_to_input_shape(self):
        input_ids = SimpleNamespace(shape=(1, 37))
        self.assertEqual(
            37,
            SigLIPVisualRetriever._model_input_token_count(
                {"input_ids": input_ids}
            ),
        )

    def test_siglip_image_processor_uses_historical_padding_call(self):
        processor = FakeSigLIPProcessor()
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, self.base / "cache")
        )
        retriever._processor = processor
        retriever._images = mock.Mock(return_value=["image-0"])
        result = retriever._prepare_image_inputs([visual_frame(0)])
        self.assertEqual({"pixel_values": mock.ANY}, result)
        self.assertEqual(
            {
                "images": ["image-0"],
                "padding": True,
                "return_tensors": "pt",
            },
            FakeSigLIPProcessor.input_calls[0],
        )

    def test_siglip_load_is_local_only_and_use_fast_false(self):
        transformers = SimpleNamespace(
            AutoProcessor=FakeSigLIPProcessor,
            AutoModel=FakeSigLIPModel,
        )
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, self.base / "cache"),
            torch_module=FakeTorch,
            transformers_module=transformers,
            asset_verifier=lambda path, files: SIGLIP_RUNTIME_TREE_IDENTITY,
        )
        retriever.load()
        self.assertEqual(
            ((str(self.model_path),), {"local_files_only": True, "use_fast": False}),
            FakeSigLIPProcessor.calls[0],
        )
        self.assertIs(True, FakeSigLIPModel.calls[0][1]["local_files_only"])
        self.assertIs(FakeTorch.bfloat16, FakeSigLIPModel.calls[0][1]["dtype"])

    def test_siglip_device_dtype_policy(self):
        cases = (
            ("cuda:0", True, True, "bfloat16"),
            ("cuda:0", True, False, "float16"),
            ("cpu", False, False, "float32"),
        )
        for device, cuda_available, bf16_supported, expected_name in cases:
            with self.subTest(device=device, bf16_supported=bf16_supported):
                dtype_values = SimpleNamespace(
                    bfloat16=object(), float16=object(), float32=object()
                )
                torch_module = SimpleNamespace(
                    bfloat16=dtype_values.bfloat16,
                    float16=dtype_values.float16,
                    float32=dtype_values.float32,
                    cuda=SimpleNamespace(
                        is_available=lambda value=cuda_available: value,
                        is_bf16_supported=lambda value=bf16_supported: value,
                    ),
                )
                FakeSigLIPModel.calls = []
                retriever = SigLIPVisualRetriever(
                    SigLIPRetrieverConfig(
                        self.model_path, self.base / "cache", device=device
                    ),
                    torch_module=torch_module,
                    transformers_module=SimpleNamespace(
                        AutoProcessor=FakeSigLIPProcessor,
                        AutoModel=FakeSigLIPModel,
                    ),
                    asset_verifier=lambda path, files: SIGLIP_RUNTIME_TREE_IDENTITY,
                )
                retriever.load()
                self.assertIs(
                    getattr(dtype_values, expected_name),
                    FakeSigLIPModel.calls[0][1]["dtype"],
                )

    def test_runtime_tree_hash_uses_exact_ordered_file_scope(self):
        runtime_files = ("config.json", "model.safetensors")
        contents = (b"config", b"weights")
        for filename, content in zip(runtime_files, contents):
            (self.model_path / filename).write_bytes(content)
        (self.model_path / "README.md").write_bytes(b"ignored")
        rows = [
            {
                "path": filename,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for filename, content in zip(runtime_files, contents)
        ]
        canonical = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(expected, runtime_tree_sha256(self.model_path, runtime_files))
        (self.model_path / "README.md").write_bytes(b"changed but ignored")
        self.assertEqual(expected, runtime_tree_sha256(self.model_path, runtime_files))
        with self.assertRaises(FileNotFoundError):
            runtime_tree_sha256(self.model_path, runtime_files + ("missing.json",))

    def test_frozen_runtime_file_lists_are_exact(self):
        self.assertEqual(
            (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ),
            SIGLIP_RUNTIME_TREE_FILES,
        )
        self.assertEqual(
            (
                "chat_template.json",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
            QWEN_RUNTIME_TREE_FILES,
        )

    def test_frozen_asset_mismatch_rejected_before_from_pretrained(self):
        for filename in SIGLIP_RUNTIME_TREE_FILES:
            (self.model_path / filename).write_bytes(filename.encode())
        retriever = SigLIPVisualRetriever(
            SigLIPRetrieverConfig(self.model_path, self.base / "cache"),
            torch_module=FakeTorch,
            transformers_module=SimpleNamespace(
                AutoProcessor=FakeSigLIPProcessor, AutoModel=FakeSigLIPModel
            ),
        )
        with self.assertRaisesRegex(ValueError, "runtime tree SHA256 mismatch"):
            retriever.load()
        self.assertEqual([], FakeSigLIPProcessor.calls)
        self.assertEqual([], FakeSigLIPModel.calls)

        qwen_path = self.base / "qwen"
        qwen_path.mkdir()
        for filename in QWEN_RUNTIME_TREE_FILES:
            (qwen_path / filename).write_bytes(filename.encode())
        observer = QwenVisualObserver(
            QwenVisualObserverConfig(qwen_path),
            transformers_module=SimpleNamespace(
                AutoProcessor=FakeQwenProcessor,
                Qwen2_5_VLForConditionalGeneration=FakeQwenModel,
            ),
            torch_module=FakeTorch,
            process_vision_info=lambda messages: ([], []),
        )
        with self.assertRaisesRegex(ValueError, "runtime tree SHA256 mismatch"):
            observer.load()
        self.assertEqual([], FakeQwenProcessor.calls)
        self.assertEqual([], FakeQwenModel.calls)

    def test_siglip_result_is_retrieval_only_without_veracity(self):
        result = SigLIPRetrievalResult([], [], {"configured_max_length": 64})
        payload = result.to_dict()
        self.assertEqual("retrieval_only", payload["purpose"])
        self.assertNotIn("veracity", payload)
        self.assertNotIn("prediction", payload)

    def test_qwen_observer_public_api_has_no_claim_argument(self):
        parameters = list(inspect.signature(QwenVisualObserver.observe).parameters)
        self.assertEqual(["self", "frames"], parameters)

    def test_qwen_prompt_maps_frame_ids_and_excludes_secret_claim(self):
        frames = [visual_frame(0), visual_frame(1)]
        secret_claim = "SECRET SAMPLE CLAIM 918273"
        prompt = QwenVisualObserver.build_prompt(frames)
        self.assertIn("Image 1 = F000", prompt)
        self.assertIn("Image 2 = F001", prompt)
        self.assertNotIn(secret_claim, prompt)
        self.assertIn('"observation_type":"scene"', prompt)
        self.assertIn('"observation":"A directly visible atomic fact."', prompt)
        self.assertNotIn("parked vehicle", prompt)
        self.assertNotIn("entity|action|scene", prompt)
        for required_policy in (
            "no hidden claim",
            "truth label",
            "dataset label",
            "exact\nlocation",
            "OCR text",
            "subtitles, logos, or watermarks",
            "dates,\nnames, or numbers",
            "empty observations list\nis valid",
            "Return JSON only",
        ):
            with self.subTest(policy=required_policy):
                self.assertIn(required_policy, prompt)

    def test_qwen_load_is_local_only_bf16_sdpa(self):
        transformers = SimpleNamespace(
            AutoProcessor=FakeQwenProcessor,
            Qwen2_5_VLForConditionalGeneration=FakeQwenModel,
        )
        observer = QwenVisualObserver(
            QwenVisualObserverConfig(self.model_path),
            transformers_module=transformers,
            torch_module=FakeTorch,
            process_vision_info=lambda messages: ([], []),
            asset_verifier=lambda path, files: QWEN_RUNTIME_TREE_SHA256,
        )
        observer.load()
        self.assertIs(True, FakeQwenProcessor.calls[0][1]["local_files_only"])
        self.assertIs(False, FakeQwenProcessor.calls[0][1]["use_fast"])
        kwargs = FakeQwenModel.calls[0][1]
        self.assertIs(True, kwargs["local_files_only"])
        self.assertIs(FakeTorch.bfloat16, kwargs["dtype"])
        self.assertEqual("sdpa", kwargs["attn_implementation"])

    def test_canonical_object_parsing(self):
        observations, mode = QwenVisualObserver.recover_json(
            json.dumps({"observations": [valid_observation()]})
        )
        self.assertEqual("canonical_object", mode)
        self.assertEqual(1, len(observations))

    def test_top_level_array_recovery(self):
        observations, mode = QwenVisualObserver.recover_json(
            json.dumps([valid_observation()])
        )
        self.assertEqual("top_level_array_wrapped", mode)
        self.assertEqual(1, len(observations))

    def test_single_observation_object_recovery(self):
        observations, mode = QwenVisualObserver.recover_json(
            json.dumps(valid_observation())
        )
        self.assertEqual("single_observation_object_wrapped", mode)
        self.assertEqual(1, len(observations))
        legacy = valid_observation()
        legacy["text"] = legacy.pop("observation")
        with self.assertRaises(ValueError):
            QwenVisualObserver.recover_json(json.dumps(legacy))

    def assert_rejected(self, observation, frame_ids=("F000", "F001")):
        accepted, rejected = QwenVisualObserver.filter_observations(
            [observation], frame_ids
        )
        self.assertEqual([], accepted)
        self.assertEqual(1, len(rejected))

    def test_invalid_and_literal_union_observation_types_are_rejected(self):
        self.assert_rejected(valid_observation(observation_type="relation"))
        self.assert_rejected(
            valid_observation(
                observation_type="entity|action|scene|object_state|spatial_relation|temporal_change"
            )
        )

    def test_invalid_frame_ids_and_evidence_refs_are_rejected(self):
        self.assert_rejected(valid_observation(frame_ids=["F999"]))
        self.assert_rejected(valid_observation(evidence_refs=["F999"]))

    def test_one_frame_temporal_change_is_rejected(self):
        self.assert_rejected(valid_observation(observation_type="temporal_change"))

    def test_temporal_change_requires_two_distinct_frame_ids_not_reference_union(self):
        observation = valid_observation(
            observation_type="temporal_change",
            frame_ids=["F001"],
            evidence_refs=["F001", "F005"],
        )
        accepted, rejected = QwenVisualObserver.filter_observations(
            [observation], ["F001", "F005"]
        )
        self.assertEqual([], accepted)
        self.assertEqual(
            ["temporal_change_requires_two_frame_ids"],
            rejected[0]["reasons"],
        )

    def test_rejected_observations_retain_deterministic_reasons(self):
        invalid = valid_observation(observation_type="relation")
        accepted, rejected = QwenVisualObserver.filter_observations(
            [invalid], ["F000"]
        )
        self.assertEqual([], accepted)
        self.assertEqual(0, rejected[0]["observation_index"])
        self.assertEqual(["invalid_observation_type"], rejected[0]["reasons"])
        self.assertEqual(invalid, rejected[0]["observation"])

    def test_speech_topic_inference_is_rejected_but_visible_description_is_accepted(self):
        inferred_speech = (
            "The person is speaking about the topic '汉语取代英语' "
            "(Chinese replacing English)."
        )
        generic_speech = "The person is talking about a political topic."
        for text in (inferred_speech, generic_speech):
            with self.subTest(text=text):
                accepted, rejected = QwenVisualObserver.filter_observations(
                    [valid_observation(observation=text)], ["F000"]
                )
                self.assertEqual([], accepted)
                self.assertIn(
                    "risky_inference_or_ocr_language",
                    rejected[0]["reasons"],
                )

        visible = valid_observation(
            observation="The person has long dark hair."
        )
        accepted, rejected = QwenVisualObserver.filter_observations(
            [visible], ["F000"]
        )
        self.assertEqual([visible], accepted)
        self.assertEqual([], rejected)

    def test_qwen_result_retains_raw_generation_once_with_sha_and_rejections(self):
        raw = json.dumps(
            {
                "observations": [
                    valid_observation(),
                    valid_observation(observation_type="invalid"),
                ]
            }
        )
        observer = QwenVisualObserver(QwenVisualObserverConfig(self.model_path))
        observer._generate = mock.Mock(return_value=raw)
        result = observer.observe([visual_frame(0)])
        payload = result.to_dict()
        self.assertEqual(raw, result.raw_generation)
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), result.raw_generation_sha256)
        self.assertEqual(raw, payload["raw_generation"])
        self.assertEqual(1, result.rejected_observation_count)
        self.assertEqual(
            ["invalid_observation_type"],
            result.rejected_observations[0]["reasons"],
        )

    def test_ocr_and_inference_language_is_rejected(self):
        for text in (
            "The text says danger on the sign.",
            "A person probably leaves the room.",
            "The subtitle is visible below the image.",
        ):
            with self.subTest(text=text):
                self.assert_rejected(valid_observation(observation=text))

    def test_veracity_and_claim_language_is_rejected(self):
        for text in (
            "The image supports the claim.",
            "This proves the report is true.",
            "The scene contains misinformation.",
        ):
            with self.subTest(text=text):
                self.assert_rejected(valid_observation(observation=text))

    def test_valid_visual_runtime_unit_is_permanently_g1_ineligible(self):
        frame = visual_frame(0, score=0.8, rank=1)
        unit = VisualObservationAdapter().convert(
            [valid_observation()],
            [frame],
            "canonical_object",
            "a" * 64,
            source_uri="video.mp4",
        )[0]
        self.assertIs(SourceType.VISUAL_OBSERVATION, unit.source_type)
        self.assertFalse(unit.eligible_for_frozen_g1)
        self.assertIsNone(unit.selection_score)
        self.assertIsNone(unit.logits)
        self.assertIsNone(unit.confidence)
        self.assertEqual(QWEN_MODEL_ID, unit.producer)

    def test_multiframe_provenance_preserves_all_references_and_retrieval(self):
        frames = [
            visual_frame(0, timestamp=4.0, score=0.7, rank=2),
            visual_frame(1, timestamp=1.0, score=0.9, rank=1),
        ]
        observation = valid_observation(
            observation_type="temporal_change",
            observation="The vehicle door is closed first and open later.",
            frame_ids=["F001", "F000"],
            evidence_refs=["F001", "F000"],
        )
        unit = VisualObservationAdapter().convert(
            [observation], frames, "top_level_array_wrapped", "b" * 64
        )[0]
        details = unit.provenance.details
        self.assertEqual(["F001", "F000"], details["frame_ids"])
        self.assertEqual(["F001", "F000"], details["evidence_refs"])
        self.assertEqual(["F001", "F000"], [item["frame_id"] for item in details["referenced_frames"]])
        self.assertEqual([1, 2], [item["retrieval_rank"] for item in details["referenced_frames"]])
        self.assertEqual(1.0, unit.start_time)
        self.assertEqual(4.0, unit.end_time)
        self.assertEqual("F001", unit.frame_id)
        self.assertEqual(SIGLIP_MODEL_ID, details["siglip_model_id"])
        self.assertEqual(SIGLIP_FROZEN_REVISION, details["siglip_revision"])
        self.assertEqual(QWEN_MODEL_ID, details["qwen_model_id"])
        self.assertEqual(QWEN_FROZEN_REVISION, details["qwen_revision"])
        self.assertNotIn("raw_generation", details)

    def test_visual_unit_id_is_deterministic_sha256_based(self):
        frame = visual_frame(0)
        adapter = VisualObservationAdapter()
        first = adapter.convert([valid_observation()], [frame], "canonical_object", "c" * 64)[0]
        second = adapter.convert([valid_observation()], [frame], "canonical_object", "c" * 64)[0]
        self.assertEqual(first.unit_id, second.unit_id)
        self.assertRegex(first.unit_id, r"^visual_[0-9a-f]{20}$")

    def test_visual_units_cannot_enter_multimodal_exposure_composer(self):
        unit = VisualObservationAdapter().convert(
            [valid_observation()], [visual_frame(0)], "canonical_object", "d" * 64
        )[0]
        with self.assertRaises(ValueError):
            MultimodalExposureComposer().compose_transcripts([unit])
        with self.assertRaises(ValueError):
            MultimodalExposureComposer().compose_ocr([unit])

    def test_visual_attachment_does_not_change_phase4a_candidates(self):
        text = RuntimeUnit("text", SourceType.TEXT, "claim text", eligible_for_frozen_g1=True)
        ocr = RuntimeUnit("ocr", SourceType.OCR, "frame text", eligible_for_frozen_g1=True)
        visual = VisualObservationAdapter().convert(
            [valid_observation()], [visual_frame(0)], "canonical_object", "e" * 64
        )[0]
        baseline = build_phase4a_request("case", "claim", [text, ocr])
        attached = build_phase4a_request("case", "claim", [text, ocr, visual])
        self.assertEqual(baseline["candidate_units"], attached["candidate_units"])
        self.assertEqual(baseline, attached)

    def test_video_visual_runner_passes_claim_only_to_retriever(self):
        frame = visual_frame(0, score=0.9, rank=1)
        retrieval = SigLIPRetrievalResult(
            [frame], [frame], {"configured_max_length": 64}
        )
        observation_result = QwenVisualObservationResult(
            [valid_observation()], "canonical_object", "f" * 64, 0
        )
        retriever = mock.Mock()
        retriever.retrieve.return_value = retrieval
        observer = mock.Mock()
        observer.observe.return_value = observation_result
        result = VideoVisualRunner(retriever, observer).run(
            "session", "secret claim", Path("video.mp4")
        )
        retriever.retrieve.assert_called_once_with(
            claim="secret claim", video_path=Path("video.mp4"), session_id="session"
        )
        observer.observe.assert_called_once_with([frame])
        self.assertEqual(1, len(result.runtime_units))

    def test_visual_only_runtime_returns_nei_without_frozen_g1(self):
        visual_unit = VisualObservationAdapter().convert(
            [valid_observation()], [visual_frame(0)], "canonical_object", "0" * 64
        )[0]
        text_runner = mock.Mock()
        text_runner.run.return_value = SimpleNamespace(
            g1_exposure_units=[], warnings=[]
        )
        visual_runner = mock.Mock()
        visual_runner.run.return_value = SimpleNamespace(
            runtime_units=[visual_unit], warnings=[]
        )
        frozen = mock.Mock()
        result = VideoMultimodalRunner(text_runner, visual_runner, frozen).run(
            "session", "claim", Path("video.mp4")
        )
        frozen.run.assert_not_called()
        verification = result.verification_result
        self.assertEqual(ModelVerdict.NOT_RUN, verification.model_verdict)
        self.assertEqual(DisplayVerdict.NEI, verification.display_verdict)
        self.assertEqual(EvidenceStatus.INSUFFICIENT, verification.evidence_status)
        self.assertEqual([], verification.all_units)

    def test_multimodal_runner_passes_only_g1_units_to_frozen_g1(self):
        text_unit = RuntimeUnit(
            "text", SourceType.TEXT, "text evidence", eligible_for_frozen_g1=True
        )
        visual_unit = VisualObservationAdapter().convert(
            [valid_observation()], [visual_frame(0)], "canonical_object", "1" * 64
        )[0]
        text_runner = mock.Mock()
        text_runner.run.return_value = SimpleNamespace(
            g1_exposure_units=[text_unit], warnings=[]
        )
        visual_runner = mock.Mock()
        visual_runner.run.return_value = SimpleNamespace(
            runtime_units=[visual_unit], warnings=[]
        )
        frozen = mock.Mock()
        frozen.run.return_value = object()
        result = VideoMultimodalRunner(text_runner, visual_runner, frozen).run(
            "session", "claim", Path("video.mp4")
        )
        passed_units = frozen.run.call_args.args[2]
        self.assertEqual([text_unit], passed_units)
        request = build_phase4a_request("session", "claim", passed_units)
        self.assertEqual(["text"], [item["unit_id"] for item in request["candidate_units"]])
        self.assertEqual([text_unit, visual_unit], result.all_runtime_units)
        self.assertIs(frozen.run.return_value, result.verification_result)


if __name__ == "__main__":
    unittest.main()
