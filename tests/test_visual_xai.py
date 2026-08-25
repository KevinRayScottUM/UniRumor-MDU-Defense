import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters.visual_observation_adapter import VisualObservationAdapter
from schemas import (
    DisplayVerdict,
    EvidenceStatus,
    GroundedFrameReference,
    ModelVerdict,
    RuntimeUnit,
    SourceType,
    UnitProvenance,
    VerificationResult,
    VisualObservationSnapshot,
    VisualTargetScore,
    VisualTargetSpan,
)
from services.claim_consistency_gate import ConsistencyResult
from services.production_result import ProductionResultBuilder
from services.qwen_visual_observer import (
    QwenVisualObservationResult,
    QwenVisualObserver,
    QwenVisualObserverConfig,
)
from services.siglip_visual_retriever import SigLIPRetrievalResult, VisualFrame
from services.video_multimodal_runner import VideoMultimodalResult, VideoMultimodalRunner
from services.video_visual_runner import VideoVisualResult
from services.visual_xai_attributor import (
    VISUAL_XAI_FAILURE_WARNING,
    VisualXAIAttributor,
    VisualXAIConfig,
)


OBSERVATION = (
    "The speaker is positioned centrally on the stage with microphones "
    "in front of him."
)
RAW_GENERATION = json.dumps(
    {
        "observations": [
            {
                "observation_type": "scene",
                "observation": OBSERVATION,
                "frame_ids": ["F001"],
                "evidence_refs": ["F001"],
            }
        ]
    },
    separators=(",", ":"),
)


class MockTargetScorer:
    runtime_fingerprint = "9" * 64

    def __init__(self):
        self.calls = []

    def score_target_logprob_batch(self, frame_batches, target_sequence, spans):
        self.calls.append((tuple(frame_batches), target_sequence, tuple(spans)))
        output = []
        for frames in frame_batches:
            variant_name = next(
                (frame.frame_path.name for frame in frames if "variant_" in frame.frame_path.name),
                "baseline",
            )
            row_column = None
            if variant_name != "baseline":
                row = int(variant_name.split("_r", 1)[1].split("_", 1)[0])
                column = int(variant_name.split("_c", 1)[1].split(".", 1)[0])
                row_column = (row, column)
            scores = []
            counts = []
            for span in spans:
                baseline = -2.0 if span.scope == "observation" else -0.5
                penalty = 0.0
                if span.scope == "observation" and row_column == (0, 0):
                    penalty = 4.0
                if span.label.casefold() == "microphones" and row_column == (1, 1):
                    penalty = 2.5
                scores.append((span.span_id, baseline - penalty))
                counts.append((span.span_id, 5 if span.scope == "observation" else 1))
            output.append(
                VisualTargetScore(
                    span_log_probabilities=tuple(scores),
                    span_token_counts=tuple(counts),
                )
            )
        return output


class SlowTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("slow tokenizer has no offsets")
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeScoreInputs(dict):
    def to(self, device):
        return self


class FakeScoreProcessor:
    def __init__(self):
        self.tokenizer = SlowTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return "PROMPT"
        return "PROMPT" + messages[-1]["content"]

    def __call__(self, *, text, **kwargs):
        rows = [[ord(character) for character in item] for item in text]
        return FakeScoreInputs(
            input_ids=rows,
            attention_mask=[[1] * len(row) for row in rows],
        )


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeVector:
    def float(self):
        return self

    def __getitem__(self, token_id):
        return FakeScalar(-0.25)


class FakeLogits:
    def __getitem__(self, coordinates):
        return FakeVector()


class FakeScoreModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=FakeLogits())


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeScoreTorch:
    @staticmethod
    def inference_mode():
        return FakeInferenceMode()

    @staticmethod
    def log_softmax(vector, dim):
        return vector


class QwenFixedTargetScoringTests(unittest.TestCase):
    def test_slow_tokenizer_fallback_maps_exact_phrase_tokens_without_generation(self):
        processor = FakeScoreProcessor()
        observer = QwenVisualObserver(
            QwenVisualObserverConfig(model_path=Path("/not-loaded")),
            torch_module=FakeScoreTorch(),
            process_vision_info=lambda messages: (None, None),
        )
        observer.load = lambda: None
        observer._processor = processor
        observer._model = FakeScoreModel()
        observer._torch = FakeScoreTorch()
        observer._device = "cpu"
        frame = VisualFrame(
            frame_id="F001",
            frame_path=Path("/not-opened.png"),
            frame_index=1,
            timestamp_sec=1.0,
            frame_rank=1,
            image_sha256="a" * 64,
            retrieval_rank=1,
        )
        target = "speaker stage"
        spans = (
            VisualTargetSpan("observation", "observation", "Whole observation", 0, 13),
            VisualTargetSpan("phrase_01", "phrase", "speaker", 0, 7),
            VisualTargetSpan("phrase_02", "phrase", "stage", 8, 13),
        )

        result = observer.score_target_logprob_batch([[frame]], target, spans)[0]

        self.assertEqual(13, result.token_count("observation"))
        self.assertEqual(7, result.token_count("phrase_01"))
        self.assertEqual(5, result.token_count("phrase_02"))
        self.assertEqual(-1.75, result.log_probability("phrase_01"))
        self.assertEqual(-1.25, result.log_probability("phrase_02"))


class VisualXAIAttributorTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy
        except ImportError as exc:
            self.skipTest(f"existing visual runtime dependencies unavailable: {exc}")
        self.cv2 = cv2
        self.numpy = numpy
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frame_path = self.root / "source.png"
        image = self.numpy.zeros((48, 64, 3), dtype=self.numpy.uint8)
        image[:, :32] = (210, 160, 80)
        image[:, 32:] = (40, 90, 190)
        self.assertTrue(self.cv2.imwrite(str(self.frame_path), image))
        self.frame_sha256 = hashlib.sha256(self.frame_path.read_bytes()).hexdigest()
        self.frame = VisualFrame(
            frame_id="F001",
            frame_path=self.frame_path,
            frame_index=25,
            timestamp_sec=1.0,
            frame_rank=1,
            image_sha256=self.frame_sha256,
            retrieval_score=0.7,
            retrieval_rank=1,
        )
        self.snapshot = VisualObservationSnapshot(
            unit_id="visual_unit_1",
            observation_text=OBSERVATION,
            start_timestamp_seconds=1.0,
            end_timestamp_seconds=1.0,
            primary_frame_id="F001",
            frame_references=(
                GroundedFrameReference(
                    frame_id="F001",
                    frame_index=25,
                    timestamp_seconds=1.0,
                    frame_rank=1,
                    retrieval_rank=1,
                    image_sha256=self.frame_sha256,
                ),
            ),
            source_index=0,
            extraction_method="qwen_claim_blind_visual_observer",
            observation_type="scene",
            frame_ids=("F001",),
            evidence_refs=("F001",),
            siglip_model_id="siglip",
            siglip_revision="siglip-revision",
            qwen_model_id="qwen",
            qwen_revision="qwen-revision",
            prompt_policy="claim-blind",
            raw_generation_sha256=hashlib.sha256(
                RAW_GENERATION.encode("utf-8")
            ).hexdigest(),
            recovery_mode="canonical_object",
            retrieval_policy_id="top4",
            observer_policy_id="claim-blind",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def attributor(self, scorer=None):
        return VisualXAIAttributor(
            scorer or MockTargetScorer(),
            VisualXAIConfig(
                cache_root=self.root / "xai",
                grid_rows=2,
                grid_columns=2,
                attribution_batch_size=2,
                blur_kernel_size=3,
                timeout_seconds=30.0,
            ),
            cv2_module=self.cv2,
            numpy_module=self.numpy,
        )

    def test_occlusion_scores_strong_and_irrelevant_regions_and_phrase_tokens(self):
        original_sha = hashlib.sha256(self.frame_path.read_bytes()).hexdigest()
        scorer = MockTargetScorer()
        artifact = self.attributor(scorer).attribute(
            [self.snapshot], [self.frame], RAW_GENERATION
        )[0]
        self.assertEqual("available", artifact.status)
        whole = next(item for item in artifact.maps if item.scope == "observation")
        microphones = next(
            item for item in artifact.maps if item.label.casefold() == "microphones"
        )
        phrase_labels = {
            item.label.casefold() for item in artifact.maps if item.scope == "phrase"
        }
        self.assertTrue({"speaker", "stage", "microphones"} <= phrase_labels)
        self.assertNotIn("him", phrase_labels)
        self.assertEqual(4.0, whole.raw_importance[0][0])
        self.assertEqual(1.0, whole.normalized_importance[0][0])
        self.assertEqual(0.0, whole.normalized_importance[0][1])
        self.assertEqual(2.5, microphones.raw_importance[1][1])
        self.assertEqual(0.0, microphones.raw_importance[0][0])
        self.assertEqual(1, microphones.target_token_count)
        self.assertTrue(whole.overlay_image_path.is_file())
        self.assertEqual(original_sha, hashlib.sha256(self.frame_path.read_bytes()).hexdigest())
        self.assertEqual(3, len(scorer.calls))
        self.assertTrue(
            all(len(call[2]) == len(artifact.maps) for call in scorer.calls)
        )

    def test_phrase_policy_does_not_turn_named_people_into_visual_labels(self):
        labels = {
            label.casefold()
            for label, _, _ in VisualXAIAttributor.phrase_spans(
                "Donald Trump stands behind microphones on a stage."
            )
        }
        self.assertNotIn("donald", labels)
        self.assertNotIn("trump", labels)
        self.assertTrue({"microphones", "stage"} <= labels)

    def test_artifacts_and_heatmaps_are_deterministic_and_cache_is_reused(self):
        scorer = MockTargetScorer()
        attributor = self.attributor(scorer)
        first = attributor.attribute([self.snapshot], [self.frame], RAW_GENERATION)[0]
        first_overlay_hashes = [
            hashlib.sha256(item.overlay_image_path.read_bytes()).hexdigest()
            for item in first.maps
        ]
        call_count = len(scorer.calls)
        second = attributor.attribute([self.snapshot], [self.frame], RAW_GENERATION)[0]
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(call_count, len(scorer.calls))
        self.assertEqual(1, attributor.cache_hits)
        self.assertEqual(
            first_overlay_hashes,
            [
                hashlib.sha256(item.overlay_image_path.read_bytes()).hexdigest()
                for item in second.maps
            ],
        )

    def test_corrupted_persistent_cache_recomputes_safely(self):
        scorer = MockTargetScorer()
        attributor = self.attributor(scorer)
        artifact = attributor.attribute(
            [self.snapshot], [self.frame], RAW_GENERATION
        )[0]
        initial_calls = len(scorer.calls)
        _, manifest = attributor._cache_paths(artifact.cache_key)
        manifest.write_text("{corrupted", encoding="utf-8")

        repaired = attributor.attribute(
            [self.snapshot], [self.frame], RAW_GENERATION
        )[0]

        self.assertEqual(artifact.artifact_id, repaired.artifact_id)
        self.assertGreater(len(scorer.calls), initial_calls)
        self.assertEqual("available", repaired.status)

    def test_serialization_never_exposes_paths_or_prediction_fields(self):
        artifact = self.attributor().attribute(
            [self.snapshot], [self.frame], RAW_GENERATION
        )[0]
        rendered = json.dumps(artifact.to_dict(), sort_keys=True)
        self.assertNotIn(str(self.root), rendered)
        for forbidden in (
            "verdict",
            "logits",
            "probabilities",
            "selection_score",
            "confidence",
            "eligible_for_frozen_g1",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_missing_frame_returns_safe_unavailable_artifact(self):
        scorer = MockTargetScorer()
        missing = VisualFrame(
            frame_id=self.frame.frame_id,
            frame_path=self.root / "missing.png",
            frame_index=self.frame.frame_index,
            timestamp_sec=self.frame.timestamp_sec,
            frame_rank=self.frame.frame_rank,
            image_sha256=self.frame.image_sha256,
            retrieval_rank=self.frame.retrieval_rank,
        )
        artifact = self.attributor(scorer).attribute(
            [self.snapshot], [missing], RAW_GENERATION
        )[0]
        self.assertEqual("unavailable", artifact.status)
        self.assertEqual("source_frame_unavailable", artifact.unavailable_reason)
        self.assertEqual((), artifact.maps)
        self.assertEqual([], scorer.calls)

    def test_public_payload_embeds_heatmaps_without_paths_and_enforces_limits(self):
        observation = {
            "observation_type": "scene",
            "observation": OBSERVATION,
            "frame_ids": ["F001"],
            "evidence_refs": ["F001"],
        }
        visual_unit = VisualObservationAdapter().convert(
            [observation],
            [self.frame],
            recovery_mode="canonical_object",
            raw_generation_sha256=self.snapshot.raw_generation_sha256,
        )[0]
        public_snapshot = VisualObservationSnapshot.from_runtime_unit(visual_unit)
        artifact = self.attributor().attribute(
            [public_snapshot], [self.frame], RAW_GENERATION
        )[0]
        g1_unit = RuntimeUnit(
            unit_id="text_1",
            source_type=SourceType.TEXT,
            text="Source evidence.",
            producer="fixture",
            provenance=UnitProvenance(extraction_method="fixture"),
            eligible_for_frozen_g1=True,
        )
        verification = VerificationResult(
            session_id="session",
            claim="A visible scene is shown.",
            model_verdict=ModelVerdict.FAKE,
            display_verdict=DisplayVerdict.FAKE,
            evidence_status=EvidenceStatus.SUFFICIENT,
            sample_logits={"fake": 2.0, "real": -1.0},
            probabilities={"fake": 0.95, "real": 0.05},
            all_units=[g1_unit],
            top_k_units=[g1_unit],
            class_winners={"fake": g1_unit.unit_id},
            pipeline_stages=[],
            warnings=[],
        )
        result = VideoMultimodalResult(
            session_id="session",
            claim="A visible scene is shown.",
            text_ocr_result=SimpleNamespace(),
            visual_result=SimpleNamespace(),
            g1_exposure_units=[g1_unit],
            visual_units=[visual_unit],
            all_runtime_units=[g1_unit, visual_unit],
            verification_result=verification,
            visual_xai_artifacts=[artifact],
        )
        public = ProductionResultBuilder(evidence_root=self.root).build(result)
        frame = public.visual_supplemental_units[0].to_dict()["evidence_frames"][0]
        self.assertEqual("available", frame["xai"]["status"])
        self.assertTrue(
            frame["xai"]["attribution_maps"][0]["heatmap_image"].startswith(
                "data:image/png;base64,"
            )
        )
        rendered = json.dumps(public.to_dict(), sort_keys=True)
        self.assertNotIn(str(self.root), rendered)
        self.assertIn("does not affect the authoritative verification verdict", rendered)
        with patch("services.production_result.MAX_EVIDENCE_IMAGE_BYTES", 8):
            limited = ProductionResultBuilder(evidence_root=self.root).build(result)
        limited_frame = limited.visual_supplemental_units[0].to_dict()[
            "evidence_frames"
        ][0]
        self.assertIsNone(limited_frame["original_image"])
        self.assertIsNone(
            limited_frame["xai"]["attribution_maps"][0]["heatmap_image"]
        )

        verdict_before = public.to_dict()["verdict"]
        base_without_xai = VideoMultimodalResult(
            session_id=result.session_id,
            claim=result.claim,
            text_ocr_result=result.text_ocr_result,
            visual_result=result.visual_result,
            g1_exposure_units=result.g1_exposure_units,
            visual_units=result.visual_units,
            all_runtime_units=result.all_runtime_units,
            verification_result=result.verification_result,
        )
        builder = ProductionResultBuilder(evidence_root=self.root)
        base_payload = builder.build(base_without_xai).to_dict()
        state = SimpleNamespace(
            state=SimpleNamespace(value="not_requested"),
            unavailable_reason=None,
            unit_id=visual_unit.unit_id,
            profile="public",
            grid_rows=6,
            grid_columns=6,
            attribution_batch_size=2,
            configuration_fingerprint="a" * 64,
            cache_hit=False,
            queue_wait_ms=None,
            compute_time_ms=None,
            source_frame_count=1,
            heavy_scorer_batches=0,
            artifacts=(),
        )
        augmented = builder.augment_visual_xai(
            base_payload, {visual_unit.unit_id: state}
        )
        self.assertEqual(verdict_before, augmented["verdict"])
        self.assertEqual(
            "not_requested",
            augmented["evidence"]["visual_supplemental_units"][0][
                "evidence_frames"
            ][0]["xai"]["status"],
        )


class VideoXAIIsolationTests(unittest.TestCase):
    def test_xai_failure_does_not_change_verdict_or_frozen_g1_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_path = root / "frame.png"
            frame_path.write_bytes(b"frame fixture")
            frame = VisualFrame(
                frame_id="F001",
                frame_path=frame_path,
                frame_index=1,
                timestamp_sec=1.0,
                frame_rank=1,
                image_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                retrieval_rank=1,
            )
            observation = {
                "observation_type": "scene",
                "observation": "A generic speaker stands on a stage.",
                "frame_ids": ["F001"],
                "evidence_refs": ["F001"],
            }
            raw = json.dumps({"observations": [observation]})
            visual_units = VisualObservationAdapter().convert(
                [observation],
                [frame],
                recovery_mode="canonical_object",
                raw_generation_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            )
            g1_unit = RuntimeUnit(
                unit_id="transcript_1",
                source_type=SourceType.TRANSCRIPT,
                text="A speaker addresses an audience.",
                producer="asr",
                provenance=UnitProvenance(extraction_method="asr"),
                eligible_for_frozen_g1=True,
            )
            authoritative = VerificationResult(
                session_id="session",
                claim="A speaker addresses an audience.",
                model_verdict=ModelVerdict.REAL,
                display_verdict=DisplayVerdict.REAL,
                evidence_status=EvidenceStatus.SUFFICIENT,
                sample_logits={"fake": -1.0, "real": 2.0},
                probabilities={"fake": 0.05, "real": 0.95},
                all_units=[g1_unit],
                top_k_units=[g1_unit],
                class_winners={"real": g1_unit.unit_id},
                pipeline_stages=[],
                warnings=[],
            )

            class FrozenG1:
                def __init__(self):
                    self.units = None

                def run(self, session_id, claim, units):
                    self.units = list(units)
                    return authoritative

            class Raises:
                def register(self, *args, **kwargs):
                    if frozen.units is None:
                        raise AssertionError(
                            "visual XAI registration ran before Frozen G1"
                        )
                    raise RuntimeError("private failure")

            frozen = FrozenG1()
            runner = VideoMultimodalRunner(
                video_text_ocr_runner=SimpleNamespace(
                    run=lambda *args, **kwargs: SimpleNamespace(
                        g1_exposure_units=[g1_unit], warnings=[]
                    )
                ),
                video_visual_runner=SimpleNamespace(
                    run=lambda *args, **kwargs: VideoVisualResult(
                        session_id="session",
                        claim="A speaker addresses an audience.",
                        video_path=str(root / "video.mp4"),
                        retrieval_result=SigLIPRetrievalResult(
                            candidate_frames=[frame],
                            selected_frames=[frame],
                            claim_token_audit={},
                        ),
                        observation_result=QwenVisualObservationResult(
                            observations=[observation],
                            recovery_mode="canonical_object",
                            raw_generation_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                            rejected_observation_count=0,
                            raw_generation=raw,
                        ),
                        runtime_units=visual_units,
                    )
                ),
                frozen_g1_runner=frozen,
                claim_consistency_gate=SimpleNamespace(
                    evaluate=lambda **kwargs: ConsistencyResult.PASS
                ),
                visual_grounding_shadow_runner=SimpleNamespace(run=lambda units: []),
                visual_xai_service=Raises(),
            )
            result = runner.run(
                "session",
                "A speaker addresses an audience.",
                root / "video.mp4",
            )
            self.assertIs(result.verification_result, authoritative)
            self.assertEqual([g1_unit], frozen.units)
            self.assertNotIn(visual_units[0], frozen.units)
            self.assertEqual([], result.visual_xai_artifacts)
            self.assertIn(VISUAL_XAI_FAILURE_WARNING, result.warnings)
            public = ProductionResultBuilder().build(result)
            self.assertIs(DisplayVerdict.REAL, public.display_verdict)
            self.assertIs(EvidenceStatus.SUFFICIENT, public.evidence_status)


if __name__ == "__main__":
    unittest.main()
