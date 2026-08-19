import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

from adapters.transcript_unit_adapter import TranscriptUnitAdapter
from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict, SourceType
from services import video_asr_runner, video_audio_decoder, whisper_asr_service
from services.video_asr_runner import VideoASRRunner
from services.video_audio_decoder import DecodedAudio, VideoAudioDecoder
from services.whisper_asr_service import (
    WHISPER_FROZEN_REVISION,
    WHISPER_MODEL_ID,
    WhisperASRConfig,
    WhisperASRService,
)


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_output():
    return json.loads((FIXTURES / "whisper_asr_result.json").read_text(encoding="utf-8"))


class FakeFrame:
    def __init__(self, values):
        self.values = numpy.asarray(values)

    def to_ndarray(self):
        return self.values


class FakeContainer:
    def __init__(self, frames, has_audio=True):
        self.frames = list(frames)
        self.streams = SimpleNamespace(audio=[object()] if has_audio else [])
        self.closed = False

    def decode(self, stream):
        return iter(self.frames)

    def close(self):
        self.closed = True


class FakeAV:
    def __init__(self, container):
        self.container = container
        self.resampler_kwargs = None

    def open(self, path):
        return self.container

    def AudioResampler(self, **kwargs):
        self.resampler_kwargs = kwargs

        class Resampler:
            @staticmethod
            def resample(frame):
                return [] if frame is None else [frame]

        return Resampler()


class FakeTorch:
    float16 = object()
    float32 = object()
    bfloat16 = object()
    cuda = SimpleNamespace(is_available=lambda: True)


class FakeTransformers:
    processor_calls = []
    model_calls = []
    pipeline_calls = []

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            FakeTransformers.processor_calls.append((args, kwargs))
            return SimpleNamespace(tokenizer="tokenizer", feature_extractor="features")

    class AutoModelForSpeechSeq2Seq:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            FakeTransformers.model_calls.append((args, kwargs))
            return "model"

    @classmethod
    def pipeline(cls, *args, **kwargs):
        cls.pipeline_calls.append((args, kwargs))

        def infer(*call_args, **call_kwargs):
            return fixture_output()

        return infer

    @classmethod
    def reset(cls):
        cls.processor_calls = []
        cls.model_calls = []
        cls.pipeline_calls = []


class VideoASRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.video_path = self.base / "video.mp4"
        self.video_path.write_bytes(b"fixture-video-placeholder")
        FakeTransformers.reset()

    def tearDown(self):
        self.temporary.cleanup()

    def test_decoder_configuration_requests_16k_mono(self):
        fake_av = FakeAV(FakeContainer([FakeFrame([[0.0, 0.25]])]))
        result = VideoAudioDecoder(
            av_module=fake_av, numpy_module=numpy
        ).decode(self.video_path)
        self.assertEqual(
            {"format": "fltp", "layout": "mono", "rate": 16000},
            fake_av.resampler_kwargs,
        )
        self.assertEqual(16000, result.sample_rate)

    def test_decoder_rejects_video_without_audio(self):
        container = FakeContainer([], has_audio=False)
        decoder = VideoAudioDecoder(av_module=FakeAV(container), numpy_module=numpy)
        with self.assertRaisesRegex(ValueError, "no audio stream"):
            decoder.decode(self.video_path)
        self.assertTrue(container.closed)

    def test_waveform_is_contiguous_float32_and_normalized(self):
        fake_av = FakeAV(
            FakeContainer([FakeFrame([[-2.0, -0.5]]), FakeFrame([[0.5, 2.0]])])
        )
        decoded = VideoAudioDecoder(
            av_module=fake_av, numpy_module=numpy
        ).decode(self.video_path)
        self.assertEqual(numpy.float32, decoded.waveform.dtype)
        self.assertTrue(decoded.waveform.flags["C_CONTIGUOUS"])
        numpy.testing.assert_array_equal(
            numpy.asarray([-1.0, -0.5, 0.5, 1.0], dtype=numpy.float32),
            decoded.waveform,
        )
        self.assertAlmostEqual(4 / 16000, decoded.duration_seconds)

    def test_ordered_asr_segment_normalization(self):
        segments = WhisperASRService.normalize_output(fixture_output())
        self.assertEqual([0, 1], [segment["segment_index"] for segment in segments])
        self.assertEqual(["你好，世界。", "This is a multilingual transcript."], [
            segment["text"] for segment in segments
        ])
        self.assertEqual((0.0, 1.25), (segments[0]["start_time"], segments[0]["end_time"]))
        self.assertEqual((1.5, 3.75), (segments[1]["start_time"], segments[1]["end_time"]))

    def test_blank_asr_segments_are_removed_and_reindexed(self):
        segments = WhisperASRService.normalize_output(
            {
                "chunks": [
                    {"text": " ", "timestamp": [-1.0, -2.0]},
                    {"text": "kept", "timestamp": [2.0, 3.0]},
                ]
            }
        )
        self.assertEqual(
            [{"segment_index": 0, "text": "kept", "start_time": 2.0, "end_time": 3.0}],
            segments,
        )

    def test_invalid_asr_timestamps_are_rejected(self):
        invalid_chunks = [
            [{"text": "negative", "timestamp": [-0.1, 1.0]}],
            [{"text": "reversed", "timestamp": [2.0, 1.0]}],
            [
                {"text": "later", "timestamp": [2.0, 3.0]},
                {"text": "earlier", "timestamp": [1.0, 4.0]},
            ],
            [
                {"text": "long", "timestamp": [0.0, 4.0]},
                {"text": "regressed end", "timestamp": [5.0, 3.0]},
            ],
        ]
        for chunks in invalid_chunks:
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                WhisperASRService.normalize_output({"chunks": chunks})

    def test_transcript_runtime_unit_mapping_preserves_segment_order(self):
        segments = WhisperASRService.normalize_output(fixture_output())
        units = TranscriptUnitAdapter().convert(segments, source_uri="video.mp4")
        self.assertEqual([segment["text"] for segment in segments], [unit.text for unit in units])
        self.assertEqual([0.0, 1.5], [unit.start_time for unit in units])
        self.assertEqual([1.25, 3.75], [unit.end_time for unit in units])
        self.assertTrue(all(unit.source_type is SourceType.TRANSCRIPT for unit in units))

    def test_transcript_unit_ids_are_deterministic(self):
        segments = WhisperASRService.normalize_output(fixture_output())
        adapter = TranscriptUnitAdapter()
        first = adapter.convert(segments)
        second = adapter.convert(segments)
        self.assertEqual(["asr_0000", "asr_0001"], [unit.unit_id for unit in first])
        self.assertEqual([unit.unit_id for unit in first], [unit.unit_id for unit in second])
        self.assertEqual(len(first), len({unit.unit_id for unit in first}))

    def test_transcript_units_are_frozen_g1_eligible(self):
        units = TranscriptUnitAdapter().convert(
            WhisperASRService.normalize_output(fixture_output())
        )
        self.assertTrue(all(unit.eligible_for_frozen_g1 for unit in units))

    def test_transcript_units_do_not_invent_confidence_scores_or_logits(self):
        units = TranscriptUnitAdapter().convert(
            WhisperASRService.normalize_output(fixture_output())
        )
        for unit in units:
            self.assertIsNone(unit.confidence)
            self.assertIsNone(unit.selection_score)
            self.assertIsNone(unit.logits)

    def test_transcript_provenance_contains_model_and_revision(self):
        unit = TranscriptUnitAdapter().convert(
            WhisperASRService.normalize_output(fixture_output()),
            source_uri="video.mp4",
        )[0]
        self.assertEqual("whisper_asr", unit.provenance.extraction_method)
        self.assertEqual("video.mp4", unit.provenance.source_uri)
        self.assertEqual(WHISPER_MODEL_ID, unit.producer)
        self.assertEqual(WHISPER_MODEL_ID, unit.provenance.details["model_id"])
        self.assertEqual(
            WHISPER_FROZEN_REVISION, unit.provenance.details["frozen_revision"]
        )

    def test_transcript_adapter_rejects_blank_text(self):
        with self.assertRaisesRegex(ValueError, "blank text"):
            TranscriptUnitAdapter().convert(
                [{"segment_index": 0, "text": " ", "start_time": 0.0, "end_time": 1.0}]
            )

    def test_model_loading_is_local_only_and_uses_dtype(self):
        model_path = self.base / "configured-model"
        model_path.mkdir()
        service = WhisperASRService(
            WhisperASRConfig(model_path=model_path),
            transformers_module=FakeTransformers,
            torch_module=FakeTorch,
        )
        service.load()
        self.assertEqual(((str(model_path),), {"local_files_only": True}), FakeTransformers.processor_calls[0])
        model_args, model_kwargs = FakeTransformers.model_calls[0]
        self.assertEqual((str(model_path),), model_args)
        self.assertIs(True, model_kwargs["local_files_only"])
        self.assertIs(FakeTorch.float16, model_kwargs["dtype"])
        self.assertNotIn("torch_dtype", model_kwargs)
        pipeline_args, pipeline_kwargs = FakeTransformers.pipeline_calls[0]
        self.assertEqual(("automatic-speech-recognition",), pipeline_args)
        self.assertEqual("cuda:0", pipeline_kwargs["device"])
        self.assertIs(FakeTorch.float16, pipeline_kwargs["dtype"])

    def test_real_call_contract_transcribes_without_forcing_language(self):
        model_path = self.base / "configured-model"
        model_path.mkdir()
        service = WhisperASRService(
            WhisperASRConfig(model_path=model_path),
            transformers_module=FakeTransformers,
            torch_module=FakeTorch,
        )
        service.load()
        inference = mock.Mock(return_value=fixture_output())
        service._pipeline = inference
        segments = service.transcribe(numpy.zeros(16, dtype=numpy.float32))
        self.assertEqual(2, len(segments))
        args, kwargs = inference.call_args
        self.assertEqual(16000, args[0]["sampling_rate"])
        self.assertIs(True, kwargs["return_timestamps"])
        self.assertEqual({"task": "transcribe"}, kwargs["generate_kwargs"])
        self.assertNotIn("language", kwargs["generate_kwargs"])

    def test_strict_sha_mismatch_is_rejected_before_model_loading(self):
        model_path = self.base / "configured-model"
        model_path.mkdir()
        (model_path / "model.safetensors").write_bytes(b"wrong asset")
        service = WhisperASRService(
            WhisperASRConfig(
                model_path=model_path,
                verify_asset_sha256=True,
                expected_safetensors_sha256="0" * 64,
            ),
            transformers_module=FakeTransformers,
            torch_module=FakeTorch,
        )
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            service.load()
        self.assertEqual([], FakeTransformers.processor_calls)
        self.assertEqual([], FakeTransformers.model_calls)

    def test_video_asr_modules_do_not_use_subprocess_or_ffmpeg_cli(self):
        source = "\n".join(
            inspect.getsource(module)
            for module in (video_audio_decoder, whisper_asr_service, video_asr_runner)
        ).lower()
        self.assertNotIn("subprocess", source)
        self.assertNotIn("ffmpeg", source)
        self.assertNotIn("ffprobe", source)

    def test_zero_transcript_returns_insufficient_nei_without_g1(self):
        decoder = mock.Mock()
        decoder.decode.return_value = DecodedAudio(
            numpy.zeros(160, dtype=numpy.float32), 0.01
        )
        asr = mock.Mock()
        asr.transcribe.return_value = []
        frozen_g1 = mock.Mock()
        result = VideoASRRunner(
            decoder, asr, frozen_g1_runner=frozen_g1
        ).run_with_frozen_g1("zero-asr", "claim", self.video_path)
        frozen_g1.run.assert_not_called()
        self.assertEqual(ModelVerdict.NOT_RUN, result.verification_result.model_verdict)
        self.assertEqual(DisplayVerdict.NEI, result.verification_result.display_verdict)
        self.assertEqual(
            EvidenceStatus.INSUFFICIENT, result.verification_result.evidence_status
        )
        self.assertEqual([], result.runtime_units)

    def test_optional_handoff_uses_existing_frozen_g1_runner(self):
        decoder = mock.Mock()
        decoder.decode.return_value = DecodedAudio(
            numpy.zeros(160, dtype=numpy.float32), 0.01
        )
        asr = mock.Mock()
        asr.transcribe.return_value = WhisperASRService.normalize_output(fixture_output())
        frozen_g1 = mock.Mock()
        expected = object()
        frozen_g1.run.return_value = expected
        result = VideoASRRunner(
            decoder, asr, frozen_g1_runner=frozen_g1
        ).run_with_frozen_g1("handoff", "claim", self.video_path)
        args = frozen_g1.run.call_args.args
        self.assertEqual(("handoff", "claim"), args[:2])
        self.assertEqual(["asr_0000", "asr_0001"], [unit.unit_id for unit in args[2]])
        self.assertIs(expected, result.verification_result)


if __name__ == "__main__":
    unittest.main()
