import inspect
import json
import re
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas import DisplayVerdict, EvidenceStatus, ModelVerdict
from services.evidence_sufficiency_policy import EvidenceSufficiencyAssessment
from services.production_execution import (
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    OperationalFailure,
    OperationalFailureCode,
    OperationalFailureStage,
    ProductionExecutionOutcome,
    ProductionExecutionStatus,
)
from services.production_result import ProductionResult
from services.session_manager import SAFE_SESSION
from webapp.execution_adapter import (
    ADAPTER_FAILURE_EXCEPTION_TYPE,
    ProductionExecutionAdapter,
    ProductionExecutionRequest,
)
from webapp.job_manager import (
    WEB_WORKER_FAILURE_CODE,
    WEB_WORKER_FAILURE_MESSAGE,
    JobManager,
    JobManagerNotAcceptingError,
    QueueFullError,
    ReservationError,
)
from webapp.job_types import JobState


class FakeClock:
    def __init__(self):
        self.value = datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


class ConstantService:
    def __init__(self, outcome=None, error=None):
        self.outcome = outcome
        self.error = error
        self.calls = []

    def execute(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        if self.error is not None:
            raise self.error
        return self.outcome


class BlockingSerialService:
    def __init__(self, outcome, call_count=3):
        self.outcome = outcome
        self.started = [threading.Event() for _ in range(call_count)]
        self.release = [threading.Event() for _ in range(call_count)]
        self.calls = []
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()

    def execute(self, session_id, claim, video_path):
        with self._lock:
            index = len(self.calls)
            self.calls.append((session_id, claim, video_path))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        self.started[index].set()
        if not self.release[index].wait(3):
            raise RuntimeError("test synchronization deadline reached")
        with self._lock:
            self.active -= 1
        return self.outcome

    def release_all(self):
        for event in self.release:
            event.set()


class BarrierBaseException(BaseException):
    pass


class WebJobManagerTests(unittest.TestCase):
    PRIVATE_VIDEO = Path("/private/web-runtime/jobs/secret/input.mp4")

    @staticmethod
    def _production_result(model_verdict=ModelVerdict.FAKE):
        model_ran = model_verdict in (ModelVerdict.FAKE, ModelVerdict.REAL)
        display_verdict = {
            ModelVerdict.FAKE: DisplayVerdict.FAKE,
            ModelVerdict.REAL: DisplayVerdict.REAL,
            ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
        }[model_verdict]
        evidence_status = (
            EvidenceStatus.SUFFICIENT
            if model_ran
            else EvidenceStatus.INSUFFICIENT
        )
        return ProductionResult(
            schema_version=1,
            session_id="job_0123456789abcdef0123456789abcdef",
            claim="exact claim",
            model_verdict=model_verdict,
            display_verdict=display_verdict,
            evidence_status=evidence_status,
            sample_logits=(
                (("fake", 1.25), ("real", -0.25)) if model_ran else ()
            ),
            probabilities=(
                (("fake", 0.75), ("real", 0.25)) if model_ran else ()
            ),
            class_winners=(
                (("fake", "unit-1"), ("real", "unit-2"))
                if model_ran
                else ()
            ),
            checkpoint_sha256="checkpoint" if model_ran else None,
            sufficiency=EvidenceSufficiencyAssessment(
                status=evidence_status,
                reason_code=(
                    "frozen_g1_evidence_available_and_model_completed"
                    if model_ran
                    else "no_frozen_g1_eligible_evidence"
                ),
                model_was_run=model_ran,
                g1_exposure_count=1 if model_ran else 0,
                transcript_exposure_count=1 if model_ran else 0,
                ocr_exposure_count=0,
                visual_unit_count=1,
                top_k_count=1 if model_ran else 0,
                supplemental_visual_present=True,
            ),
            g1_exposure_units=(),
            g1_top_k_explanation_unit_ids=("unit-1",) if model_ran else (),
            visual_supplemental_units=(),
            runtime_ms=123.456,
        )

    @classmethod
    def _success(cls, verdict=ModelVerdict.FAKE):
        return ProductionExecutionOutcome(
            schema_version=1,
            status=ProductionExecutionStatus.SUCCESS,
            result=cls._production_result(verdict),
            failure=None,
        )

    @staticmethod
    def _failure():
        return ProductionExecutionOutcome(
            schema_version=1,
            status=ProductionExecutionStatus.FAILURE,
            result=None,
            failure=OperationalFailure(
                stage=OperationalFailureStage.RUNTIME,
                code=OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                exception_type="PrivateRuntimeError",
                public_message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
            ),
        )

    def _started_manager(self, service, **kwargs):
        manager = JobManager(service, **kwargs)
        manager.start()

        def cleanup():
            release_all = getattr(service, "release_all", None)
            if release_all is not None:
                release_all()
            manager.shutdown(timeout=2)

        self.addCleanup(cleanup)
        return manager

    @staticmethod
    def _submit(manager, claim="exact claim", video_path=PRIVATE_VIDEO):
        with manager.reserve_capacity() as reservation:
            return manager.submit_reserved(
                reservation,
                claim=claim,
                video_path=video_path,
            )

    def test_job_state_enum_is_exact(self):
        self.assertEqual(
            [state.value for state in JobState],
            ["accepted", "queued", "running", "completed", "failed", "expired"],
        )

    def test_adapter_forwards_exact_request_to_injected_contract(self):
        outcome = self._success(ModelVerdict.REAL)
        contract = ConstantService(outcome)
        adapter = ProductionExecutionAdapter(contract)
        request = ProductionExecutionRequest(
            session_id="job_0123456789abcdef0123456789abcdef",
            claim="  exact focal claim  ",
            video_path=self.PRIVATE_VIDEO,
        )

        returned = adapter.execute_request(request)

        self.assertIs(adapter.execution_contract, contract)
        self.assertIs(returned, outcome)
        self.assertEqual(
            contract.calls,
            [(request.session_id, request.claim, request.video_path)],
        )

    def test_adapter_runtime_exception_becomes_safe_failure_outcome(self):
        private_detail = "/private/model/cache runtime exploded"
        contract = ConstantService(error=RuntimeError(private_detail))
        adapter = ProductionExecutionAdapter(contract)

        outcome = adapter.execute(
            "job_0123456789abcdef0123456789abcdef",
            "claim",
            self.PRIVATE_VIDEO,
        )

        self.assertIs(outcome.status, ProductionExecutionStatus.FAILURE)
        self.assertIsNone(outcome.result)
        self.assertEqual(
            outcome.failure.code,
            OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
        )
        self.assertEqual(
            outcome.failure.public_message,
            RUNTIME_FAILURE_PUBLIC_MESSAGE,
        )
        self.assertEqual(
            outcome.failure.exception_type,
            ADAPTER_FAILURE_EXCEPTION_TYPE,
        )
        serialized = outcome.to_json()
        self.assertNotIn(private_detail, serialized)
        self.assertNotIn("RuntimeError", serialized)

    def test_adapter_invalid_contract_output_becomes_safe_failure_outcome(self):
        adapter = ProductionExecutionAdapter(
            ConstantService(outcome={"status": "success"})
        )

        outcome = adapter.execute(
            "job_0123456789abcdef0123456789abcdef",
            "claim",
            self.PRIVATE_VIDEO,
        )

        self.assertIs(outcome.status, ProductionExecutionStatus.FAILURE)
        self.assertEqual(
            outcome.failure.exception_type,
            ADAPTER_FAILURE_EXCEPTION_TYPE,
        )

    def test_adapter_source_has_only_closed_execution_contract_dependency(self):
        source = inspect.getsource(
            __import__("webapp.execution_adapter", fromlist=["*"])
        )
        prohibited = (
            "ProductionExecutionService",
            "ProductionRuntime",
            "ProductionResultBuilder",
            "FrozenG1Runner",
            "VideoMultimodalRunner",
            "Whisper",
            "PaddleOCR",
            "SigLIP",
            "Qwen",
            "production_runtime",
            "subprocess",
            "shell=True",
        )
        for name in prohibited:
            with self.subTest(name=name):
                self.assertNotIn(name, source)
        self.assertIn("services.production_execution", source)

    def test_construction_is_side_effect_free(self):
        service = ConstantService(self._success())
        manager = JobManager(service)

        self.assertFalse(manager.started)
        self.assertFalse(manager.worker_alive)
        self.assertFalse(manager.accepting_jobs)
        self.assertEqual(service.calls, [])
        with self.assertRaises(JobManagerNotAcceptingError):
            manager.reserve_capacity()

    def test_start_creates_exactly_one_worker(self):
        service = ConstantService(self._success())
        manager = JobManager(service)
        self.addCleanup(manager.shutdown, 2)

        self.assertTrue(manager.start())
        worker = manager._worker
        self.assertTrue(manager.worker_alive)
        self.assertFalse(manager.start())
        self.assertIs(manager._worker, worker)

    def test_generated_ids_are_exact_unique_and_safe_sessions(self):
        manager = self._started_manager(
            ConstantService(self._success()),
            max_queued_jobs=20,
        )
        job_ids = [self._submit(manager, claim=f"claim {index}") for index in range(10)]

        self.assertEqual(len(set(job_ids)), 10)
        for job_id in job_ids:
            with self.subTest(job_id=job_id):
                self.assertRegex(job_id, r"^job_[0-9a-f]{32}$")
                self.assertIsNotNone(SAFE_SESSION.fullmatch(job_id))
                self.assertNotIn("claim", job_id)

    def test_reservation_succeeds_and_queue_full_creates_no_job(self):
        manager = self._started_manager(
            ConstantService(self._success()),
            max_queued_jobs=1,
        )
        reservation = manager.reserve_capacity()

        self.assertTrue(reservation.active)
        self.assertEqual(manager.reservation_count, 1)
        with self.assertRaises(QueueFullError):
            manager.reserve_capacity()
        self.assertEqual(manager._jobs, {})
        reservation.release()

    def test_abandoned_reservation_context_releases_capacity(self):
        manager = self._started_manager(
            ConstantService(self._success()),
            max_queued_jobs=1,
        )
        with manager.reserve_capacity() as reservation:
            self.assertTrue(reservation.active)
        self.assertFalse(reservation.active)
        self.assertEqual(manager.reservation_count, 0)
        manager.reserve_capacity().release()

    def test_explicit_release_is_idempotent_and_prevents_submit(self):
        manager = self._started_manager(ConstantService(self._success()))
        reservation = manager.reserve_capacity()
        reservation.release()
        reservation.release()

        self.assertEqual(manager.reservation_count, 0)
        with self.assertRaises(ReservationError):
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.PRIVATE_VIDEO,
            )

    def test_shutdown_invalidated_reservation_is_not_accepting(self):
        manager = self._started_manager(ConstantService(self._success()))
        reservation = manager.reserve_capacity()
        self.assertTrue(manager.shutdown(timeout=1))

        with self.assertRaises(JobManagerNotAcceptingError):
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.PRIVATE_VIDEO,
            )
        self.assertEqual(manager.reservation_count, 0)

    def test_reservation_cannot_be_submitted_twice(self):
        manager = self._started_manager(ConstantService(self._success()))
        reservation = manager.reserve_capacity()
        self.assertRegex(
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.PRIVATE_VIDEO,
            ),
            r"^job_",
        )
        with self.assertRaises(ReservationError):
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.PRIVATE_VIDEO,
            )

    def test_reservation_cannot_cross_managers(self):
        first = self._started_manager(ConstantService(self._success()))
        second = self._started_manager(ConstantService(self._success()))
        reservation = first.reserve_capacity()

        with self.assertRaises(ReservationError):
            second.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.PRIVATE_VIDEO,
            )
        self.assertTrue(reservation.active)
        first.submit_reserved(
            reservation,
            claim="claim",
            video_path=self.PRIVATE_VIDEO,
        )

    def test_queued_plus_reservations_never_exceeds_bound(self):
        service = BlockingSerialService(self._success(), call_count=2)
        manager = self._started_manager(service, max_queued_jobs=2)
        self._submit(manager, claim="running")
        self.assertTrue(service.started[0].wait(1))

        self._submit(manager, claim="queued")
        reservation = manager.reserve_capacity()
        self.assertEqual(manager.queued_count + manager.reservation_count, 2)
        with self.assertRaises(QueueFullError):
            manager.reserve_capacity()
        reservation.release()

    def test_concurrent_reservations_cannot_overcommit(self):
        manager = self._started_manager(
            ConstantService(self._success()),
            max_queued_jobs=3,
        )
        barrier = threading.Barrier(21)
        reservations = []
        full_count = []
        collection_lock = threading.Lock()

        def reserve():
            barrier.wait()
            try:
                reservation = manager.reserve_capacity()
            except QueueFullError:
                with collection_lock:
                    full_count.append(1)
            else:
                with collection_lock:
                    reservations.append(reservation)

        threads = [threading.Thread(target=reserve) for _ in range(20)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(reservations), 3)
        self.assertEqual(len(full_count), 17)
        self.assertEqual(manager.reservation_count, 3)
        for reservation in reservations:
            reservation.release()

    def test_success_path_records_only_allowed_state_sequence(self):
        manager = self._started_manager(ConstantService(self._success()))
        job_id = self._submit(manager)
        snapshot = manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)

        self.assertIsNotNone(snapshot)
        self.assertEqual(
            manager._get_state_history(job_id),
            (
                JobState.ACCEPTED,
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.COMPLETED,
            ),
        )

    def test_fifo_positions_single_execution_and_exactly_once(self):
        service = BlockingSerialService(self._success(), call_count=3)
        manager = self._started_manager(service, max_queued_jobs=3)
        first = self._submit(manager, claim="first")
        self.assertTrue(service.started[0].wait(1))
        second = self._submit(manager, claim="second")
        third = self._submit(manager, claim="third")

        self.assertEqual(manager.get_snapshot(second).queue_position, 1)
        self.assertEqual(manager.get_snapshot(third).queue_position, 2)
        self.assertIsNone(manager.get_snapshot(first).queue_position)

        service.release[0].set()
        self.assertTrue(service.started[1].wait(1))
        self.assertEqual(manager.get_snapshot(third).queue_position, 1)
        service.release[1].set()
        self.assertTrue(service.started[2].wait(1))
        service.release[2].set()
        self.assertIsNotNone(
            manager.wait_for_state(third, JobState.COMPLETED, timeout=1)
        )

        self.assertEqual([call[0] for call in service.calls], [first, second, third])
        self.assertEqual(len({call[0] for call in service.calls}), 3)
        self.assertEqual(service.maximum_active, 1)

    def test_running_job_does_not_consume_waiting_capacity(self):
        service = BlockingSerialService(self._success(), call_count=2)
        manager = self._started_manager(service, max_queued_jobs=1)
        first = self._submit(manager, claim="first")
        self.assertTrue(service.started[0].wait(1))
        self.assertEqual(manager.running_job_id, first)

        second = self._submit(manager, claim="second")
        self.assertEqual(manager.get_snapshot(second).queue_position, 1)
        with self.assertRaises(QueueFullError):
            manager.reserve_capacity()

    def test_fake_real_and_successful_nei_all_complete(self):
        for verdict in (ModelVerdict.FAKE, ModelVerdict.REAL, ModelVerdict.NOT_RUN):
            with self.subTest(verdict=verdict):
                outcome = self._success(verdict)
                manager = self._started_manager(ConstantService(outcome))
                job_id = self._submit(manager)
                snapshot = manager.wait_for_state(
                    job_id,
                    JobState.COMPLETED,
                    timeout=1,
                )
                self.assertIsNotNone(snapshot)
                self.assertIs(manager._get_completed_outcome(job_id), outcome)

    def test_task06_failure_becomes_failed_without_exception_type(self):
        outcome = self._failure()
        manager = self._started_manager(ConstantService(outcome))
        job_id = self._submit(manager)
        snapshot = manager.wait_for_state(job_id, JobState.FAILED, timeout=1)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.failure.code, "runtime_execution_failed")
        self.assertEqual(snapshot.failure.message, RUNTIME_FAILURE_PUBLIC_MESSAGE)
        self.assertNotIn("exception_type", snapshot.to_dict()["failure"])
        self.assertIsNone(manager._get_completed_outcome(job_id))
        self.assertEqual(
            manager._get_state_history(job_id)[-2:],
            (JobState.RUNNING, JobState.FAILED),
        )

    def test_unexpected_exception_uses_fixed_redacted_failure(self):
        leaked = "/private/model/cache secret exception"
        manager = self._started_manager(
            ConstantService(error=RuntimeError(leaked))
        )
        job_id = self._submit(manager)
        snapshot = manager.wait_for_state(job_id, JobState.FAILED, timeout=1)
        serialized = json.dumps(snapshot.to_dict())

        self.assertEqual(snapshot.failure.code, WEB_WORKER_FAILURE_CODE)
        self.assertEqual(snapshot.failure.message, WEB_WORKER_FAILURE_MESSAGE)
        self.assertRegex(snapshot.failure.incident_id, r"^incident_[0-9a-f]{32}$")
        self.assertNotIn(leaked, serialized)
        self.assertNotIn("RuntimeError", serialized)

    def test_invalid_service_return_fails_safely(self):
        manager = self._started_manager(ConstantService(outcome={"status": "success"}))
        job_id = self._submit(manager)
        snapshot = manager.wait_for_state(job_id, JobState.FAILED, timeout=1)

        self.assertEqual(snapshot.failure.code, WEB_WORKER_FAILURE_CODE)
        self.assertIsNone(manager._get_completed_outcome(job_id))

    def test_narrow_execution_helper_does_not_catch_baseexception(self):
        manager = JobManager(ConstantService(error=BarrierBaseException()))
        with self.assertRaises(BarrierBaseException):
            manager._invoke_service("job_test", "claim", self.PRIVATE_VIDEO)

    def test_completed_outcome_identity_and_scientific_values_are_preserved(self):
        outcome = self._success(ModelVerdict.FAKE)
        original_logits = outcome.result.sample_logits
        original_probabilities = outcome.result.probabilities
        manager = self._started_manager(ConstantService(outcome))
        job_id = self._submit(manager)
        manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
        stored = manager._get_completed_outcome(job_id)

        self.assertIs(stored, outcome)
        self.assertIs(stored.result, outcome.result)
        self.assertEqual(stored.result.sample_logits, original_logits)
        self.assertEqual(stored.result.probabilities, original_probabilities)

    def test_active_job_never_expires(self):
        clock = FakeClock()
        service = BlockingSerialService(self._success(), call_count=1)
        manager = self._started_manager(
            service,
            clock=clock,
            terminal_retention=60,
            expired_tombstone_duration=10,
        )
        job_id = self._submit(manager)
        self.assertTrue(service.started[0].wait(1))
        clock.advance(hours=24)

        self.assertEqual(manager.sweep_expired(), 0)
        self.assertEqual(manager.get_snapshot(job_id).state, JobState.RUNNING)

    def test_terminal_retention_expiration_drops_outcome_then_tombstone(self):
        clock = FakeClock()
        outcome = self._success()
        manager = self._started_manager(
            ConstantService(outcome),
            clock=clock,
            terminal_retention=60,
            expired_tombstone_duration=10,
        )
        job_id = self._submit(manager)
        manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)

        clock.advance(seconds=59)
        self.assertEqual(manager.sweep_expired(), 0)
        self.assertIs(manager._get_completed_outcome(job_id), outcome)
        clock.advance(seconds=1)
        self.assertEqual(manager.sweep_expired(), 1)
        self.assertEqual(manager.get_snapshot(job_id).state, JobState.EXPIRED)
        self.assertIsNone(manager._get_completed_outcome(job_id))
        clock.advance(seconds=10)
        self.assertEqual(manager.sweep_expired(), 1)
        self.assertIsNone(manager.get_snapshot(job_id))

    def test_expired_failed_job_discards_failure_detail(self):
        clock = FakeClock()
        manager = self._started_manager(
            ConstantService(self._failure()),
            clock=clock,
            terminal_retention=1,
            expired_tombstone_duration=10,
        )
        job_id = self._submit(manager)
        self.assertIsNotNone(
            manager.wait_for_state(job_id, JobState.FAILED, timeout=1).failure
        )
        clock.advance(seconds=1)
        manager.sweep_expired()

        snapshot = manager.get_snapshot(job_id)
        self.assertEqual(snapshot.state, JobState.EXPIRED)
        self.assertIsNone(snapshot.failure)

    def test_sweep_performs_no_filesystem_deletion(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = Path(temporary_directory) / "input.mp4"
            video.write_bytes(b"not-real-video")
            manager = self._started_manager(
                ConstantService(self._success()),
                clock=clock,
                terminal_retention=1,
                expired_tombstone_duration=1,
            )
            job_id = self._submit(manager, video_path=video)
            manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
            clock.advance(seconds=2)
            manager.sweep_expired()

            self.assertTrue(video.exists())

    def test_shutdown_stops_admission_and_is_repeatable(self):
        manager = self._started_manager(ConstantService(self._success()))
        self.assertTrue(manager.shutdown(timeout=1))
        with self.assertRaises(JobManagerNotAcceptingError):
            manager.reserve_capacity()
        self.assertTrue(manager.shutdown(timeout=1))
        self.assertFalse(manager.worker_alive)

    def test_shutdown_timeout_is_honest_and_queued_work_does_not_start(self):
        service = BlockingSerialService(self._success(), call_count=2)
        manager = self._started_manager(service, max_queued_jobs=2)
        first = self._submit(manager, claim="running")
        self.assertTrue(service.started[0].wait(1))
        second = self._submit(manager, claim="must remain queued")

        self.assertFalse(manager.shutdown(timeout=0.01))
        self.assertTrue(manager.worker_alive)
        service.release[0].set()
        self.assertTrue(manager.shutdown(timeout=1))

        self.assertEqual([call[0] for call in service.calls], [first])
        self.assertEqual(manager.get_snapshot(second).state, JobState.QUEUED)
        self.assertTrue(manager.shutdown(timeout=0))

    def test_snapshot_is_json_friendly_rfc3339_and_path_redacted(self):
        clock = FakeClock()
        manager = self._started_manager(
            ConstantService(self._success()),
            clock=clock,
        )
        job_id = self._submit(
            manager,
            claim="private focal claim",
            video_path=self.PRIVATE_VIDEO,
        )
        snapshot = manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
        public = snapshot.to_dict()
        serialized = json.dumps(public)

        self.assertTrue(public["created_at"].endswith("Z"))
        self.assertTrue(public["started_at"].endswith("Z"))
        self.assertTrue(public["finished_at"].endswith("Z"))
        self.assertIsInstance(public["queue_elapsed_ms"], int)
        self.assertGreaterEqual(public["queue_elapsed_ms"], 0)
        self.assertGreaterEqual(public["execution_elapsed_ms"], 0)
        self.assertNotIn(str(self.PRIVATE_VIDEO), serialized)
        self.assertNotIn("private focal claim", serialized)
        self.assertEqual(
            set(public),
            {
                "job_id",
                "state",
                "queue_position",
                "created_at",
                "started_at",
                "finished_at",
                "expires_at",
                "queue_elapsed_ms",
                "execution_elapsed_ms",
                "failure",
            },
        )

    def test_job_manager_source_preserves_closed_task06_boundary(self):
        source = inspect.getsource(__import__("webapp.job_manager", fromlist=["*"]))
        prohibited = (
            "FrozenG1Runner",
            "VideoMultimodalRunner",
            "Whisper",
            "PaddleOCR",
            "SigLIP",
            "Qwen",
            "ProductionResultBuilder",
            "subprocess",
            "shell=True",
        )
        for name in prohibited:
            with self.subTest(name=name):
                self.assertNotIn(name, source)
        self.assertEqual(source.count(".execute("), 1)


if __name__ == "__main__":
    unittest.main()
