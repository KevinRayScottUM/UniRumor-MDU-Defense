import ast
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from adapters.ocr_unit_adapter import (
    OCRFilterConfig,
    OCRUnitAdapter,
    normalize_ocr_text,
    union_bbox,
)
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
)
from services import paddle_ocr_service, paddle_ocr_worker
from services.multimodal_exposure_composer import (
    PHASE4A_HARD_MAX_UNITS,
    MultimodalExposureComposer,
)
from services.paddle_ocr_service import (
    DEFAULT_CUDNN8_LIBRARY_PATH,
    OCRDetection,
    OCRFrameResult,
    PaddleOCRService,
    PaddleOCRServiceConfig,
    frozen_model_metadata,
    polygon_to_bbox,
)
from services.paddle_ocr_worker import RUNTIME_TREE_FILES, runtime_tree_sha256
from services.video_asr_runner import VideoASRResult
from services.video_frame_sampler import (
    SampledVideoFrame,
    VideoFrameSampler,
    sample_frame_indices,
)
from services.video_ocr_runner import VideoOCRResult
from services.video_text_ocr_runner import VideoTextOCRRunner


def detection(text, confidence=0.9, bbox=(0, 0, 10, 10)):
    xmin, ymin, xmax, ymax = bbox
    polygon = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return OCRDetection(
        text=text,
        confidence=confidence,
        polygon=polygon,
        runtime_bbox=[float(value) for value in bbox],
    )


def frame_result(rank, detections, timestamp=None):
    return OCRFrameResult(
        frame_rank=rank,
        frame_index=rank * 10,
        timestamp_sec=float(rank if timestamp is None else timestamp),
        frame_id=f"frame_{rank:04d}_{rank * 10:06d}",
        frame_path=Path(f"/cache/frame-{rank}.jpg"),
        detections=list(detections),
    )


def transcript_unit(index):
    return RuntimeUnit(
        unit_id=f"asr_{index:04d}",
        source_type=SourceType.TRANSCRIPT,
        text=f"segment {index}",
        start_time=float(index),
        end_time=float(index + 1),
        producer="openai/whisper-large-v3-turbo",
        provenance=UnitProvenance(
            source_uri="video.mp4",
            source_index=index,
            extraction_method="whisper_asr",
        ),
        eligible_for_frozen_g1=True,
    )


class WorkerStub:
    def __init__(self, mode="ok", detections=None):
        self.mode = mode
        self.detections = detections or []
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.mode == "failure":
            return SimpleNamespace(returncode=2, stdout="", stderr="worker boom")
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if self.mode == "malformed":
            output_path.write_text("not-json", encoding="utf-8")
        else:
            frames = []
            for frame in request["frames"]:
                result = dict(frame)
                result["detections"] = list(self.detections)
                frames.append(result)
            models = frozen_model_metadata()
            if self.mode == "wrong-hash":
                models["detector"]["runtime_tree_sha256"] = "0" * 64
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "ok",
                        "session_id": request["session_id"],
                        "models": models,
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class FakeVideoFrame:
    pass


class FakeVideoContainer:
    def __init__(self, frame_count=20, fps=10):
        self.stream = SimpleNamespace(frames=frame_count, average_rate=fps)
        self.streams = SimpleNamespace(video=[self.stream])
        self.frame_count = frame_count
        self.closed = False

    def decode(self, stream):
        return iter(FakeVideoFrame() for _ in range(self.frame_count))

    def close(self):
        self.closed = True


class FakeAV:
    def __init__(self, frame_count=20, fps=10):
        self.frame_count = frame_count
        self.fps = fps
        self.containers = []

    def open(self, path):
        container = FakeVideoContainer(self.frame_count, self.fps)
        self.containers.append(container)
        return container


class VideoOCRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.cache_root = self.base / "cache"
        self.video = self.base / "video.mp4"
        self.video.write_bytes(b"source-video")
        self.frame_path = self.base / "frame.jpg"
        self.frame_path.write_bytes(b"frame")
        self.detector_path = self.base / "detector"
        self.recognizer_path = self.base / "recognizer"

    def tearDown(self):
        self.temporary.cleanup()

    def service(self, stub):
        return PaddleOCRService(
            PaddleOCRServiceConfig(
                detector_model_path=self.detector_path,
                recognizer_model_path=self.recognizer_path,
                cache_root=self.cache_root,
                python_executable="/isolated/python",
                timeout_seconds=12,
            ),
            subprocess_run=stub,
        )

    def sampled_frame(self):
        return SampledVideoFrame(0, 4, 0.4, "frame_0000_000004", self.frame_path)

    def test_exact_historical_eight_frame_sampling(self):
        self.assertEqual([11, 22, 33, 44, 55, 66, 77, 88], sample_frame_indices(100, 8))

    def test_frame_count_at_or_below_n_returns_every_frame(self):
        self.assertEqual(list(range(5)), sample_frame_indices(5, 8))
        self.assertEqual([], sample_frame_indices(0, 8))

    def test_single_frame_request_uses_middle_index(self):
        self.assertEqual([5], sample_frame_indices(10, 1))

    def test_sampling_order_is_deterministic(self):
        first = sample_frame_indices(137, 8)
        self.assertEqual(first, sample_frame_indices(137, 8))
        self.assertEqual(sorted(first), first)
        self.assertEqual(len(first), len(set(first)))

    def test_extracted_frames_stay_under_cache_and_source_is_unchanged(self):
        fake_av = FakeAV(frame_count=20, fps=10)

        def writer(frame, target):
            target.write_bytes(b"jpeg")

        sampler = VideoFrameSampler(
            self.cache_root, av_module=fake_av, frame_writer=writer
        )
        frames = sampler.sample("session", self.video)
        self.assertEqual(8, len(frames))
        self.assertTrue(
            all(self.cache_root.resolve() in frame.frame_path.resolve().parents for frame in frames)
        )
        self.assertEqual(b"source-video", self.video.read_bytes())

    def test_unicode_chinese_normalization_survives(self):
        self.assertEqual("纯中文内容", normalize_ocr_text("  纯中文内容  "))
        units = OCRUnitAdapter().convert(
            [frame_result(0, [detection("  纯中文内容  ")])]
        )
        self.assertEqual("纯中文内容", units[0].text)
        self.assertEqual(
            "  纯中文内容  ",
            units[0].provenance.details["accepted_detections"][0]["text"],
        )

    def test_low_confidence_and_short_noise_are_removed(self):
        adapter = OCRUnitAdapter(OCRFilterConfig(0.5, 3, 6))
        frames = [
            frame_result(0, [detection("valid text", 0.49)]),
            frame_result(1, [detection("AI", 0.99)]),
            frame_result(2, [detection("有效文本", 0.8)]),
        ]
        units = adapter.convert(frames)
        self.assertEqual(["有效文本"], [unit.text for unit in units])

    def test_polygon_is_converted_to_frozen_flat_bbox(self):
        polygon = [[8, 2], [12, 4], [10, 9], [4, 7]]
        self.assertEqual([4.0, 2.0, 12.0, 9.0], polygon_to_bbox(polygon))

    def test_union_bbox_and_reading_order_frame_grouping(self):
        detections = [
            detection("bottom", 0.8, (0, 20, 10, 30)),
            detection("topright", 0.9, (20, 0, 30, 10)),
            detection("topleft", 1.0, (0, 0, 10, 10)),
        ]
        units = OCRUnitAdapter().convert([frame_result(2, detections, timestamp=3.5)])
        unit = units[0]
        self.assertEqual("topleft topright bottom", unit.text)
        self.assertEqual([0, 0, 30, 30], union_bbox(d.runtime_bbox for d in detections))
        self.assertEqual([0, 0, 30, 30], unit.bbox)
        self.assertAlmostEqual(0.9, unit.confidence)

    def test_unicode_normalized_frame_text_dedup_keeps_best(self):
        frames = [
            frame_result(0, [detection("ＡＢＣ 新闻", 0.7)]),
            frame_result(1, [detection("abc 新闻", 0.95)]),
        ]
        units = OCRUnitAdapter().convert(frames)
        self.assertEqual(1, len(units))
        self.assertEqual("abc 新闻", units[0].text)
        self.assertEqual("frame_0001_000010", units[0].frame_id)

    def test_max_six_quality_rank_then_chronological_restore(self):
        frames = [
            frame_result(index, [detection(f"unique text {index}", 0.50 + index / 100)])
            for index in range(8)
        ]
        units = OCRUnitAdapter().convert(frames)
        self.assertEqual(6, len(units))
        self.assertEqual([20, 30, 40, 50, 60, 70], [unit.provenance.source_index for unit in units])

    def test_ocr_quality_tie_uses_raw_detection_count(self):
        adapter = OCRUnitAdapter(OCRFilterConfig(max_ocr_units=1))
        frames = [
            frame_result(0, [detection("first valid", 0.9)]),
            frame_result(
                1,
                [detection("second valid", 0.9), detection("x", 0.1)],
            ),
        ]
        units = adapter.convert(frames)
        self.assertEqual("second valid", units[0].text)
        self.assertEqual(2, units[0].provenance.details["raw_detection_count"])

    def test_ocr_units_are_deterministic_eligible_and_preserve_fields(self):
        frames = [frame_result(3, [detection("valid text", 0.8, (1, 2, 9, 12))], 4.25)]
        first = OCRUnitAdapter().convert(frames, source_uri="video.mp4")
        second = OCRUnitAdapter().convert(frames, source_uri="video.mp4")
        self.assertEqual(["ocr_0000"], [unit.unit_id for unit in first])
        self.assertEqual([unit.to_dict() for unit in first], [unit.to_dict() for unit in second])
        unit = first[0]
        self.assertIs(SourceType.OCR, unit.source_type)
        self.assertTrue(unit.eligible_for_frozen_g1)
        self.assertEqual(4.25, unit.start_time)
        self.assertEqual(4.25, unit.end_time)
        self.assertEqual("frame_0003_000030", unit.frame_id)
        self.assertEqual([1, 2, 9, 12], unit.bbox)
        self.assertEqual(0.8, unit.confidence)
        accepted = unit.provenance.details["accepted_detections"][0]
        self.assertEqual("valid text", accepted["text"])
        self.assertIn("polygon", accepted)

    def test_parent_service_has_no_paddle_import_and_only_worker_imports_it(self):
        services_root = Path(paddle_ocr_service.__file__).parent
        importing_files = []
        for path in services_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name == "paddle" or name.startswith("paddleocr") for name in names):
                    importing_files.append(path.name)
        self.assertEqual(["paddle_ocr_worker.py"], sorted(set(importing_files)))

    def test_subprocess_uses_argument_list_shell_false_and_local_paths(self):
        stub = WorkerStub()
        service = self.service(stub)
        service.predict("ocr-session", [self.sampled_frame()])
        command, kwargs = stub.calls[0]
        self.assertIsInstance(command, list)
        self.assertEqual("/isolated/python", command[0])
        self.assertIn("services.paddle_ocr_worker", command)
        self.assertIs(False, kwargs["shell"])
        request_path = Path(command[command.index("--request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(str(self.detector_path), request["models"]["detector"]["local_path"])
        self.assertEqual(str(self.recognizer_path), request["models"]["recognizer"]["local_path"])

    def test_worker_environment_isolated_without_parent_mutation(self):
        stub = WorkerStub()
        before = dict(os.environ)
        self.service(stub).predict("environment", [self.sampled_frame()])
        environment = stub.calls[0][1]["env"]
        self.assertEqual("1", environment["OMP_NUM_THREADS"])
        self.assertEqual("True", environment["DISABLE_MODEL_SOURCE_CHECK"])
        self.assertEqual(
            str(DEFAULT_CUDNN8_LIBRARY_PATH),
            environment["LD_LIBRARY_PATH"].split(os.pathsep)[0],
        )
        self.assertEqual(before, dict(os.environ))

    def test_wrong_worker_hash_and_malformed_json_are_rejected(self):
        for mode, message in (
            ("wrong-hash", "provenance/hash"),
            ("malformed", "malformed JSON"),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, message):
                self.service(WorkerStub(mode)).predict(mode, [self.sampled_frame()])

    def test_worker_failure_is_distinct_from_valid_empty_ocr(self):
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.service(WorkerStub("failure")).predict("failure", [self.sampled_frame()])
        results = self.service(WorkerStub()).predict("empty", [self.sampled_frame()])
        self.assertEqual(1, len(results))
        self.assertEqual([], results[0].detections)

    def test_runtime_tree_hash_uses_only_exact_runtime_files(self):
        model_path = self.base / "model-tree"
        model_path.mkdir()
        for index, name in enumerate(RUNTIME_TREE_FILES):
            (model_path / name).write_bytes(f"file-{index}".encode())
        metadata = model_path / ".cache"
        metadata.mkdir()
        noise = metadata / "ignored"
        noise.write_text("one", encoding="utf-8")
        first = runtime_tree_sha256(model_path)
        self.assertEqual(
            "cf382228f46cdd0ce36b28ab1f260254f15e353f719a627530e556ccd8857043",
            first,
        )
        noise.write_text("two", encoding="utf-8")
        self.assertEqual(first, runtime_tree_sha256(model_path))
        (model_path / "config.json").write_text("changed", encoding="utf-8")
        self.assertNotEqual(first, runtime_tree_sha256(model_path))

    def test_worker_initializes_explicit_local_models_and_disables_auxiliary_models(self):
        calls = []
        fake_module = ModuleType("paddleocr")

        def fake_ocr(**kwargs):
            calls.append(kwargs)
            return object()

        fake_module.PaddleOCR = fake_ocr
        with mock.patch.dict(sys.modules, {"paddleocr": fake_module}):
            paddle_ocr_worker._create_engine(Path("/det"), Path("/rec"), "gpu:0")
        self.assertEqual("/det", calls[0]["text_detection_model_dir"])
        self.assertEqual("/rec", calls[0]["text_recognition_model_dir"])
        self.assertIs(False, calls[0]["use_doc_orientation_classify"])
        self.assertIs(False, calls[0]["use_doc_unwarping"])
        self.assertIs(False, calls[0]["use_textline_orientation"])

    def test_31_transcripts_group_to_twelve_without_loss_or_mutation(self):
        raw = [transcript_unit(index) for index in range(31)]
        before = [copy.deepcopy(unit.to_dict()) for unit in raw]
        exposed = MultimodalExposureComposer().compose_transcripts(raw)
        self.assertEqual(12, len(exposed))
        source_ids = [
            unit_id
            for unit in exposed
            for unit_id in unit.provenance.details["source_unit_ids"]
        ]
        self.assertEqual([unit.unit_id for unit in raw], source_ids)
        self.assertEqual(0.0, exposed[0].start_time)
        self.assertEqual(31.0, exposed[-1].end_time)
        self.assertEqual(before, [unit.to_dict() for unit in raw])

    def test_combined_exposure_is_transcript_first_bounded_and_not_new_g1_limit(self):
        transcripts = [transcript_unit(index) for index in range(31)]
        ocr = OCRUnitAdapter().convert(
            [frame_result(index, [detection(f"ocr evidence {index}")]) for index in range(8)]
        )
        combined = MultimodalExposureComposer().compose(transcripts, ocr)
        self.assertLessEqual(len(combined), 18)
        source_types = [unit.source_type for unit in combined]
        first_ocr = source_types.index(SourceType.OCR)
        self.assertTrue(all(kind is SourceType.TRANSCRIPT for kind in source_types[:first_ocr]))
        self.assertTrue(all(kind is SourceType.OCR for kind in source_types[first_ocr:]))
        self.assertEqual(24, PHASE4A_HARD_MAX_UNITS)
        self.assertTrue(all(kind is not SourceType.VISUAL_OBSERVATION for kind in source_types))

    def combined_runner(self, transcripts, ocr_units, frozen_g1=None):
        asr_result = VideoASRResult(
            session_id="combined",
            claim="claim",
            video_metadata={},
            asr_text=" ".join(unit.text for unit in transcripts),
            asr_segments=[],
            runtime_units=transcripts,
        )
        ocr_result = VideoOCRResult(
            session_id="combined",
            video_path=str(self.video),
            sampled_frames=[],
            raw_ocr_artifacts=[],
            ocr_units=ocr_units,
        )
        asr_runner = mock.Mock()
        asr_runner.run.return_value = asr_result
        ocr_runner = mock.Mock()
        ocr_runner.run.return_value = ocr_result
        return VideoTextOCRRunner(
            asr_runner, ocr_runner, frozen_g1_runner=frozen_g1
        )

    def test_asr_only_and_ocr_only_paths_reach_existing_g1(self):
        ocr_unit = OCRUnitAdapter().convert(
            [frame_result(0, [detection("ocr evidence")])]
        )[0]
        for transcripts, ocr_units, expected_type in (
            ([transcript_unit(0)], [], SourceType.TRANSCRIPT),
            ([], [ocr_unit], SourceType.OCR),
        ):
            frozen = mock.Mock()
            frozen.run.return_value = object()
            result = self.combined_runner(transcripts, ocr_units, frozen).run_with_frozen_g1(
                "combined", "claim", self.video
            )
            exposed = frozen.run.call_args.args[2]
            self.assertEqual([expected_type], [unit.source_type for unit in exposed])
            self.assertIs(frozen.run.return_value, result.verification_result)

    def test_zero_asr_and_ocr_returns_engineering_nei_without_g1(self):
        frozen = mock.Mock()
        result = self.combined_runner([], [], frozen).run_with_frozen_g1(
            "combined", "claim", self.video
        )
        frozen.run.assert_not_called()
        self.assertEqual(ModelVerdict.NOT_RUN, result.verification_result.model_verdict)
        self.assertEqual(DisplayVerdict.NEI, result.verification_result.display_verdict)
        self.assertEqual(EvidenceStatus.INSUFFICIENT, result.verification_result.evidence_status)


if __name__ == "__main__":
    unittest.main()
