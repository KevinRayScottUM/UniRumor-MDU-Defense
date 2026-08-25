import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from schemas import (
    GroundedFrameReference,
    QWEN_OCCLUSION_BASELINE,
    QWEN_OCCLUSION_METHOD,
    VISUAL_XAI_ARTIFACT_TYPE,
    VISUAL_XAI_SCHEMA_VERSION,
    VisualAttributionArtifact,
    VisualAttributionMap,
    VisualObservationSnapshot,
)
from services.siglip_visual_retriever import VisualFrame
from services.visual_xai_attributor import (
    VISUAL_XAI_CONFIGURATION_VERSION,
    VISUAL_XAI_GRID_SIZE_ENV,
    VISUAL_XAI_PROFILE_ENV,
    VisualXAIAttributor,
    VisualXAIConfig,
)
from services.visual_xai_runtime import (
    VisualXAIRuntimeService,
    VisualXAIState,
    visual_xai_max_concurrency,
)


class _UnusedScorer:
    runtime_fingerprint = "f" * 64

    def score_target_logprob_batch(self, *args, **kwargs):
        raise AssertionError("model scoring was not expected")


class _DifferentModelScorer(_UnusedScorer):
    runtime_fingerprint = "e" * 64


def _snapshot(frame_sha256: str, unit_id: str = "visual-unit-1"):
    return VisualObservationSnapshot(
        unit_id=unit_id,
        observation_text="A speaker stands on a stage with microphones.",
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
                image_sha256=frame_sha256,
            ),
        ),
        source_index=0,
        extraction_method="claim_blind_visual_observation",
        observation_type="scene",
        frame_ids=("F001",),
        evidence_refs=("F001",),
        siglip_model_id="siglip",
        siglip_revision="siglip-revision",
        qwen_model_id="qwen",
        qwen_revision="qwen-revision",
        prompt_policy="claim_blind_prompt_v1",
        raw_generation_sha256=hashlib.sha256(b"raw generation").hexdigest(),
        recovery_mode="canonical_object",
        retrieval_policy_id="claim_conditioned_siglip_top4",
        observer_policy_id="claim_blind_prompt_v1",
    )


def _artifact(snapshot, frame, config, *, available=True):
    values = tuple(
        tuple(0.0 for _ in range(config.grid_columns))
        for _ in range(config.grid_rows)
    )
    maps = (
        VisualAttributionMap(
            map_id="observation",
            scope="observation",
            label="Whole observation",
            target_start_character=0,
            target_end_character=len(snapshot.observation_text),
            target_token_count=8,
            baseline_target_log_probability=-4.0,
            raw_importance=values,
            normalized_importance=values,
        ),
    ) if available else ()
    payload = {
        "schema_version": VISUAL_XAI_SCHEMA_VERSION,
        "artifact_type": VISUAL_XAI_ARTIFACT_TYPE,
        "status": "available" if available else "unavailable",
        "unavailable_reason": None if available else "attribution_failed",
        "method": QWEN_OCCLUSION_METHOD,
        "model_id": snapshot.qwen_model_id,
        "model_revision": snapshot.qwen_revision,
        "model_fingerprint": "f" * 64,
        "source_frame_id": frame.frame_id,
        "source_frame_index": frame.frame_index,
        "source_timestamp_seconds": frame.timestamp_sec,
        "source_frame_sha256": frame.image_sha256,
        "observation_unit_id": snapshot.unit_id,
        "observation_text": snapshot.observation_text,
        "observation_text_sha256": hashlib.sha256(
            snapshot.observation_text.encode()
        ).hexdigest(),
        "raw_generation_sha256": snapshot.raw_generation_sha256,
        "profile": config.profile,
        "grid_rows": config.grid_rows,
        "grid_columns": config.grid_columns,
        "attribution_batch_size": config.attribution_batch_size,
        "occlusion_baseline": QWEN_OCCLUSION_BASELINE,
        "configuration_version": VISUAL_XAI_CONFIGURATION_VERSION,
        "configuration_fingerprint": config.configuration_fingerprint,
        "phrase_policy": "deterministic_visible_concept_tokens_v1",
        "heavy_scorer_batches": 1
        + (config.grid_rows * config.grid_columns + config.attribution_batch_size - 1)
        // config.attribution_batch_size,
        "maps": maps,
        "cache_key": "c" * 64,
    }
    identity = {**payload, "maps": [item.to_dict() for item in maps]}
    return VisualAttributionArtifact(
        artifact_id=VisualAttributionArtifact.compute_identity(identity),
        **payload,
    )


class VisualXAIProfileTests(unittest.TestCase):
    def test_public_and_research_profiles_and_override(self):
        root = Path("/tmp/visual-xai-profile-fixture")
        public = VisualXAIConfig.from_environment(
            root, environ={VISUAL_XAI_PROFILE_ENV: "public"}
        )
        research = VisualXAIConfig.from_environment(
            root, environ={VISUAL_XAI_PROFILE_ENV: "research"}
        )
        override = VisualXAIConfig.from_environment(
            root,
            environ={
                VISUAL_XAI_PROFILE_ENV: "public",
                VISUAL_XAI_GRID_SIZE_ENV: "8",
            },
        )
        self.assertEqual((public.grid_rows, public.grid_columns), (6, 6))
        self.assertEqual((research.grid_rows, research.grid_columns), (8, 8))
        self.assertEqual((override.grid_rows, override.grid_columns), (8, 8))
        self.assertEqual(1 + 36 // public.attribution_batch_size, 19)
        self.assertEqual(1 + 64 // research.attribution_batch_size, 33)

    def test_invalid_profile_grid_and_concurrency_are_rejected(self):
        root = Path("/tmp/visual-xai-profile-fixture")
        with self.assertRaises(ValueError):
            VisualXAIConfig.from_environment(
                root, environ={VISUAL_XAI_PROFILE_ENV: "fast"}
            )
        with self.assertRaises(ValueError):
            VisualXAIConfig.from_environment(
                root,
                environ={
                    VISUAL_XAI_PROFILE_ENV: "public",
                    VISUAL_XAI_GRID_SIZE_ENV: "7",
                },
            )
        with self.assertRaises(ValueError):
            visual_xai_max_concurrency({"MDU_VISUAL_XAI_MAX_CONCURRENCY": "0"})


class VisualXAIRuntimeServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        frame_path = self.root / "visual" / "session" / "frame.jpg"
        frame_path.parent.mkdir(parents=True)
        frame_path.write_bytes(b"source frame")
        frame_sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        self.frame = VisualFrame(
            frame_id="F001",
            frame_path=frame_path,
            frame_index=25,
            timestamp_sec=1.0,
            frame_rank=1,
            image_sha256=frame_sha,
            retrieval_rank=1,
        )
        self.snapshot = _snapshot(frame_sha)
        self.config = VisualXAIConfig(
            cache_root=self.root / "visual_xai",
            profile="public",
            grid_rows=6,
            grid_columns=6,
            attribution_batch_size=2,
        )
        self.attributor = VisualXAIAttributor(_UnusedScorer(), self.config)
        self.artifact = _artifact(
            self.snapshot, self.frame, self.config
        )

    def _service(self, max_concurrency=1):
        return VisualXAIRuntimeService(
            self.attributor,
            self.root,
            max_concurrency=max_concurrency,
        )

    def _register(self, service, snapshot=None):
        return service.register(
            "job_0123456789abcdef0123456789abcdef",
            (snapshot or self.snapshot,),
            (self.frame,),
            "raw generation",
            observer_runtime_ms=125.0,
        )[0]

    def test_registration_is_lazy_and_background_request_is_idempotent(self):
        service = self._service()
        self.addCleanup(service.shutdown, 1)
        with (
            patch.object(self.attributor, "cached_artifacts", return_value=None),
            patch.object(
                self.attributor,
                "attribute",
                return_value=[self.artifact],
            ) as compute,
        ):
            registered = self._register(service)
            self.assertIs(registered.state, VisualXAIState.NOT_REQUESTED)
            compute.assert_not_called()
            first = service.request(registered.job_id, registered.unit_id)
            second = service.request(registered.job_id, registered.unit_id)
            self.assertIs(first.state, VisualXAIState.PENDING)
            self.assertIn(second.state, {VisualXAIState.PENDING, VisualXAIState.READY})
            finished = service.wait_for_terminal(
                registered.job_id, registered.unit_id
            )
            self.assertIs(finished.state, VisualXAIState.READY)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(finished.heavy_scorer_batches, 19)

    def test_cache_hit_bypasses_model_computation_and_survives_reload(self):
        service = self._service()
        with (
            patch.object(
                self.attributor,
                "cached_artifacts",
                return_value=(self.artifact,),
            ),
            patch.object(
                self.attributor,
                "attribute",
                side_effect=AssertionError("cache hit must bypass scoring"),
            ),
        ):
            registered = self._register(service)
            ready = service.request(registered.job_id, registered.unit_id)
            self.assertIs(ready.state, VisualXAIState.READY)
            self.assertTrue(ready.cache_hit)
            self.assertEqual(ready.heavy_scorer_batches, 0)
        self.assertTrue(service.shutdown(1))

        reloaded = self._service()
        self.addCleanup(reloaded.shutdown, 1)
        with patch.object(
            self.attributor,
            "cached_artifacts",
            return_value=(self.artifact,),
        ):
            restored = reloaded.get_status(registered.job_id, registered.unit_id)
        self.assertIs(restored.state, VisualXAIState.READY)
        self.assertTrue(restored.cache_hit)

    def test_concurrent_units_remain_gpu_serialized(self):
        service = self._service(max_concurrency=2)
        self.addCleanup(service.shutdown, 2)
        second_snapshot = _snapshot(self.frame.image_sha256, "visual-unit-2")
        second_artifact = _artifact(second_snapshot, self.frame, self.config)
        active = 0
        maximum_active = 0
        guard = threading.Lock()

        def compute(snapshots, frames, raw_generation):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return [
                self.artifact
                if snapshots[0].unit_id == self.snapshot.unit_id
                else second_artifact
            ]

        with (
            patch.object(self.attributor, "cached_artifacts", return_value=None),
            patch.object(self.attributor, "attribute", side_effect=compute),
        ):
            first = self._register(service)
            second = self._register(service, second_snapshot)
            service.request(first.job_id, first.unit_id)
            service.request(second.job_id, second.unit_id)
            self.assertTrue(
                service.wait_for_terminal(first.job_id, first.unit_id, 2).terminal
            )
            self.assertTrue(
                service.wait_for_terminal(second.job_id, second.unit_id, 2).terminal
            )
        self.assertEqual(maximum_active, 1)

    def test_cache_key_changes_with_scientific_inputs_and_configuration(self):
        model = self.attributor._model_fingerprint(self.snapshot, self.attributor.scorer)
        base = self.attributor._cache_key(
            self.snapshot, self.frame, (self.frame,), model
        )
        changed_text = _snapshot(self.frame.image_sha256)
        object.__setattr__(changed_text, "observation_text", "A different observation.")
        changed_text_key = self.attributor._cache_key(
            changed_text, self.frame, (self.frame,), model
        )
        changed_frame = VisualFrame(
            **{**self.frame.__dict__, "image_sha256": "1" * 64}
        )
        changed_frame_key = self.attributor._cache_key(
            self.snapshot, changed_frame, (changed_frame,), model
        )
        research = VisualXAIAttributor(
            _UnusedScorer(),
            VisualXAIConfig(
                cache_root=self.root / "research",
                profile="research",
                grid_rows=8,
                grid_columns=8,
            ),
        )
        research_key = research._cache_key(
            self.snapshot, self.frame, (self.frame,), model
        )
        different_model = VisualXAIAttributor(
            _DifferentModelScorer(), self.config
        )
        model_key = different_model._cache_key(
            self.snapshot,
            self.frame,
            (self.frame,),
            different_model._model_fingerprint(
                self.snapshot, different_model.scorer
            ),
        )
        phrase_config = VisualXAIAttributor(
            _UnusedScorer(),
            VisualXAIConfig(
                cache_root=self.root / "phrases",
                profile="public",
                grid_rows=6,
                grid_columns=6,
                maximum_phrase_count=1,
            ),
        )
        phrase_key = phrase_config._cache_key(
            self.snapshot, self.frame, (self.frame,), model
        )
        self.assertEqual(
            len(
                {
                    base,
                    changed_text_key,
                    changed_frame_key,
                    research_key,
                    model_key,
                    phrase_key,
                }
            ),
            6,
        )

    def test_registration_rejects_source_frame_outside_runtime_cache(self):
        service = self._service()
        self.addCleanup(service.shutdown, 1)
        outside = self.root.parent / "outside-visual-xai-frame.jpg"
        outside.write_bytes(b"outside")
        self.addCleanup(outside.unlink, missing_ok=True)
        outside_frame = VisualFrame(
            **{
                **self.frame.__dict__,
                "frame_path": outside,
                "image_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        )
        outside_snapshot = _snapshot(outside_frame.image_sha256)
        with self.assertRaisesRegex(ValueError, "outside runtime cache"):
            service.register(
                "job_0123456789abcdef0123456789abcdef",
                (outside_snapshot,),
                (outside_frame,),
                "raw generation",
            )


if __name__ == "__main__":
    unittest.main()
