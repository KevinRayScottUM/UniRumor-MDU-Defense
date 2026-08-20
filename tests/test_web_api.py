import asyncio
import copy
import inspect
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

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
from webapp.api import _RequestBoundaryMiddleware, create_app
from webapp.api_config import APIConfig
from webapp.job_manager import JobManager
from webapp.job_types import JobFailureSnapshot, JobSnapshot, JobState
from webapp.runtime_lifecycle import APIRuntimeLifecycle, APIRuntimeStartupError
from webapp.server_lock import ServerLock, ServerLockUnavailableError


JOB_A = "job_0123456789abcdef0123456789abcdef"
JOB_B = "job_fedcba9876543210fedcba9876543210"
PRIVATE_PATH = "/private/model/cache/secret"


class BarrierBaseException(BaseException):
    pass


class FakeRuntime:
    def __init__(self, events=None, error=None):
        self.events = events if events is not None else []
        self.error = error
        self.start_calls = 0

    def start(self):
        self.events.append("runtime.start")
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        return object()


class FakeService:
    def __init__(self, outcome=None, *, events=None, runtime_error=None, error=None):
        self.events = events if events is not None else []
        self.runtime = FakeRuntime(self.events, runtime_error)
        self.outcome = outcome
        self.error = error
        self.calls = []

    def execute(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        if self.error is not None:
            raise self.error
        return self.outcome


class RecordingLock:
    def __init__(self, root, events):
        self._lock = ServerLock(root)
        self.events = events

    @property
    def acquired(self):
        return self._lock.acquired

    def acquire(self):
        self.events.append("lock.acquire")
        self._lock.acquire()
        return self

    def release(self):
        self.events.append("lock.release")
        self._lock.release()


class RecordingManager(JobManager):
    def __init__(self, service, events):
        events.append("manager.construct")
        self.events = events
        super().__init__(service)

    def start(self):
        self.events.append("manager.start")
        return super().start()

    def shutdown(self, timeout=30.0):
        self.events.append("manager.shutdown")
        return super().shutdown(timeout)


class ScriptedManager:
    def __init__(self, service, snapshots=None, outcomes=None):
        self.service = service
        self.snapshots = snapshots or {}
        self.outcomes = outcomes or {}
        self.sweep_calls = 0
        self._worker_alive = False
        self._draining = False
        self._full = False

    @property
    def worker_alive(self):
        return self._worker_alive

    @property
    def accepting_jobs(self):
        return self._worker_alive and not self._draining and not self._full

    @property
    def queued_count(self):
        return 1 if self._full else 0

    @property
    def reservation_count(self):
        return 0

    @property
    def max_queued_jobs(self):
        return 1

    def start(self):
        self._worker_alive = True
        return True

    def shutdown(self, timeout=30.0):
        self._draining = True
        self._worker_alive = False
        return True

    def sweep_expired(self):
        self.sweep_calls += 1
        return 0

    def get_snapshot(self, job_id):
        value = self.snapshots.get(job_id)
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, BaseException):
            raise value
        return value

    def _get_completed_outcome(self, job_id):
        return self.outcomes.get(job_id)


def production_result(verdict, job_id=JOB_A):
    model_ran = verdict in (ModelVerdict.FAKE, ModelVerdict.REAL)
    display = {
        ModelVerdict.FAKE: DisplayVerdict.FAKE,
        ModelVerdict.REAL: DisplayVerdict.REAL,
        ModelVerdict.NOT_RUN: DisplayVerdict.NEI,
    }[verdict]
    evidence = (
        EvidenceStatus.SUFFICIENT
        if model_ran
        else EvidenceStatus.INSUFFICIENT
    )
    return ProductionResult(
        schema_version=1,
        session_id=job_id,
        claim="exact focal claim",
        model_verdict=verdict,
        display_verdict=display,
        evidence_status=evidence,
        sample_logits=(
            (("fake", 1.23456789), ("real", -0.33333333)) if model_ran else ()
        ),
        probabilities=(
            (("fake", 0.81234567), ("real", 0.18765433)) if model_ran else ()
        ),
        class_winners=(
            (("fake", "unit-f"), ("real", "unit-r")) if model_ran else ()
        ),
        checkpoint_sha256="checkpoint" if model_ran else None,
        sufficiency=EvidenceSufficiencyAssessment(
            status=evidence,
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
        g1_top_k_explanation_unit_ids=("unit-f",) if model_ran else (),
        visual_supplemental_units=(),
        runtime_ms=123.456789,
    )


def success_outcome(verdict=ModelVerdict.FAKE, job_id=JOB_A):
    return ProductionExecutionOutcome(
        schema_version=1,
        status=ProductionExecutionStatus.SUCCESS,
        result=production_result(verdict, job_id),
        failure=None,
    )


def failure_outcome():
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


def snapshot(state, *, job_id=JOB_A, queue_position=None, failure=None):
    terminal = state in (JobState.COMPLETED, JobState.FAILED)
    started = state in (JobState.RUNNING, JobState.COMPLETED, JobState.FAILED)
    return JobSnapshot(
        job_id=job_id,
        state=state,
        queue_position=queue_position,
        created_at="2026-08-20T08:00:00Z",
        started_at="2026-08-20T08:00:01Z" if started else None,
        finished_at="2026-08-20T08:00:02Z" if terminal else None,
        expires_at="2026-08-20T09:00:02Z" if terminal else None,
        queue_elapsed_ms=1000,
        execution_elapsed_ms=1000 if started else 0,
        failure=failure,
    )


class WebAPIConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_valid_origins_are_explicit_deduplicated_and_ordered(self):
        config = APIConfig(
            self.root,
            (
                "https://example.test",
                "http://localhost:5173",
                "https://example.test",
            ),
        )
        self.assertEqual(
            config.allowed_origins,
            ("https://example.test", "http://localhost:5173"),
        )
        self.assertEqual(config.poll_after_ms, 3000)
        self.assertEqual(config.retry_after_seconds, 3)
        self.assertEqual(config.graceful_shutdown_timeout_seconds, 30.0)

    def test_wildcard_and_blank_origins_are_rejected(self):
        for origins in (("*",), ("https://*.example.test",), ("",), ("  ",)):
            with self.subTest(origins=origins), self.assertRaises(ValueError):
                APIConfig(self.root, origins)

    def test_non_origin_values_are_rejected(self):
        for origin in (
            "example.test",
            "ftp://example.test",
            "https://example.test/path",
            "https://user@example.test",
            "https://example.test?query=1",
        ):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                APIConfig(self.root, (origin,))

    def test_runtime_root_must_exist(self):
        missing = self.root / "missing"
        with self.assertRaises(ValueError):
            APIConfig(missing)
        self.assertFalse(missing.exists())

    def test_runtime_root_must_be_directory(self):
        file_path = self.root / "file"
        file_path.write_text("content", encoding="utf-8")
        with self.assertRaises(ValueError):
            APIConfig(file_path)

    def test_runtime_root_symlink_is_rejected(self):
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            APIConfig(link)

    def test_runtime_root_is_canonical_without_creating_files(self):
        nested = self.root / "nested"
        nested.mkdir()
        before = sorted(self.root.rglob("*"))
        config = APIConfig(nested / ".." / "nested")
        after = sorted(self.root.rglob("*"))
        self.assertEqual(config.web_runtime_root, nested.resolve())
        self.assertEqual(after, before)


class WebAPILifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = APIConfig(
            self.root,
            ("https://frontend.example",),
            graceful_shutdown_timeout_seconds=0.02,
        )

    def test_factory_construction_is_side_effect_free(self):
        events = []
        service = FakeService(events=events)

        def service_factory():
            events.append("service.construct")
            return service

        def lock_factory(root):
            events.append("lock.construct")
            return RecordingLock(root, events)

        def manager_factory(execution_service):
            return RecordingManager(execution_service, events)

        app = create_app(
            self.config,
            execution_service_factory=service_factory,
            server_lock_factory=lock_factory,
            job_manager_factory=manager_factory,
        )
        self.assertEqual(events, [])
        self.assertEqual(service.runtime.start_calls, 0)
        self.assertEqual(service.calls, [])
        self.assertFalse((self.root / ".server.lock").exists())
        self.assertIsNotNone(app)

    def test_lifespan_order_and_single_service_graph_are_exact(self):
        events = []
        service = FakeService(outcome=success_outcome(), events=events)

        def provider():
            events.append("service.obtain")
            return service

        app = create_app(
            self.config,
            execution_service_factory=provider,
            server_lock_factory=lambda root: RecordingLock(root, events),
            job_manager_factory=lambda value: RecordingManager(value, events),
        )
        with TestClient(app):
            self.assertIs(app.state.api_runtime_state.execution_service, service)
            self.assertIs(
                app.state.api_runtime_state.manager._execution_service,
                service,
            )
            self.assertEqual(service.runtime.start_calls, 1)
        self.assertEqual(
            events,
            [
                "lock.acquire",
                "service.obtain",
                "runtime.start",
                "manager.construct",
                "manager.start",
                "manager.shutdown",
                "lock.release",
            ],
        )
        self.assertEqual(service.calls, [])

    def test_real_service_construction_from_server_path_is_deferred(self):
        config_path = self.root / "server-config.json"
        config = APIConfig(
            self.root,
            production_runtime_config_path=config_path,
        )
        service = FakeService()
        with patch(
            "webapp.runtime_lifecycle.ProductionRuntimeConfig.from_json",
            return_value=SimpleNamespace(cache_root=self.root.parent),
        ) as config_loader, patch(
            "webapp.api.ProductionExecutionService.from_json",
            return_value=service,
        ) as constructor:
            app = create_app(config)
            constructor.assert_not_called()
            config_loader.assert_not_called()
            with TestClient(app):
                config_loader.assert_called_once_with(config_path.resolve())
                constructor.assert_called_once_with(config_path.resolve())
        self.assertEqual(service.runtime.start_calls, 1)

    def test_lock_contention_prevents_runtime_and_manager_startup(self):
        owner = ServerLock(self.root).acquire()
        self.addCleanup(owner.release)
        service_calls = []
        manager_calls = []

        def provider():
            service_calls.append(1)
            return FakeService()

        def manager_factory(service):
            manager_calls.append(1)
            return JobManager(service)

        app = create_app(
            self.config,
            execution_service_factory=provider,
            job_manager_factory=manager_factory,
        )
        with self.assertRaises(APIRuntimeStartupError):
            with TestClient(app):
                pass
        self.assertEqual(service_calls, [])
        self.assertEqual(manager_calls, [])
        self.assertTrue(app.state.api_runtime_state.startup_failed)
        self.assertFalse(app.state.api_runtime_state.startup_complete)

    def test_second_app_cannot_create_second_active_runtime(self):
        first_service = FakeService()
        second_service = FakeService()
        first = create_app(self.config, execution_service=first_service)
        second = create_app(self.config, execution_service=second_service)
        with TestClient(first):
            with self.assertRaises(APIRuntimeStartupError):
                with TestClient(second):
                    pass
            self.assertEqual(first_service.runtime.start_calls, 1)
            self.assertEqual(second_service.runtime.start_calls, 0)

    def test_clean_shutdown_releases_lock_for_another_owner(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app):
            with self.assertRaises(ServerLockUnavailableError):
                ServerLock(self.root).acquire()
        replacement = ServerLock(self.root).acquire()
        replacement.release()
        state = app.state.api_runtime_state
        self.assertTrue(state.shutdown_complete)
        self.assertFalse(state.shutdown_incomplete)

    def test_runtime_start_failure_does_not_construct_manager_and_releases_lock(self):
        manager_calls = []

        def manager_factory(service):
            manager_calls.append(service)
            return JobManager(service)

        app = create_app(
            self.config,
            execution_service=FakeService(
                runtime_error=RuntimeError(PRIVATE_PATH)
            ),
            job_manager_factory=manager_factory,
        )
        with self.assertRaises(APIRuntimeStartupError) as caught:
            with TestClient(app):
                pass
        self.assertNotIn(PRIVATE_PATH, str(caught.exception))
        self.assertEqual(manager_calls, [])
        replacement = ServerLock(self.root).acquire()
        replacement.release()

    def test_incomplete_shutdown_retains_lock_until_worker_stops(self):
        release = threading.Event()
        started = threading.Event()

        class BlockingService(FakeService):
            def execute(self, session_id, claim, video_path):
                started.set()
                release.wait(2)
                return success_outcome(job_id=session_id)

        lifecycle = APIRuntimeLifecycle(self.config, BlockingService)
        lifecycle.startup()
        manager = lifecycle.state.manager
        with manager.reserve_capacity() as reservation:
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.root / "input.mp4",
            )
        self.assertTrue(started.wait(1))
        self.assertFalse(lifecycle.shutdown())
        self.assertTrue(lifecycle.state.shutdown_incomplete)
        self.assertTrue(lifecycle.state.singleton_acquired)
        with self.assertRaises(ServerLockUnavailableError):
            ServerLock(self.root).acquire()

        release.set()
        self.assertTrue(lifecycle.shutdown())
        replacement = ServerLock(self.root).acquire()
        replacement.release()


class WebAPIHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = APIConfig(
            self.root,
            ("https://frontend.example",),
            poll_after_ms=4321,
            retry_after_seconds=7,
        )

    def scripted_app(self, snapshots=None, outcomes=None, service=None):
        service = service or FakeService()
        holder = {}

        def manager_factory(execution_service):
            manager = ScriptedManager(execution_service, snapshots, outcomes)
            holder["manager"] = manager
            return manager

        app = create_app(
            self.config,
            execution_service=service,
            job_manager_factory=manager_factory,
        )
        return app, holder, service

    def assert_public_error(self, response, status_code, code):
        self.assertEqual(response.status_code, status_code)
        request_id = response.headers["X-Request-ID"]
        self.assertRegex(request_id, r"^req_[0-9a-f]{32}$")
        self.assertEqual(
            response.json(),
            {
                "api_version": "v1",
                "error": {
                    "code": code,
                    "message": response.json()["error"]["message"],
                    "request_id": request_id,
                },
            },
        )

    def test_health_is_exact_liveness_only_with_server_request_id(self):
        service = FakeService()
        app = create_app(self.config, execution_service=service)
        with TestClient(app) as client:
            before = service.runtime.start_calls
            response = client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "caller-controlled"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"api_version": "v1", "status": "ok"})
            self.assertRegex(
                response.headers["X-Request-ID"],
                r"^req_[0-9a-f]{32}$",
            )
            self.assertNotEqual(
                response.headers["X-Request-ID"],
                "caller-controlled",
            )
            self.assertEqual(service.runtime.start_calls, before)
            self.assertEqual(service.calls, [])

    def test_readiness_ready_is_exact_and_does_not_execute(self):
        service = FakeService()
        app = create_app(self.config, execution_service=service)
        with TestClient(app) as client:
            response = client.get("/api/v1/readiness")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "api_version": "v1",
                    "status": "ready",
                    "accepting_jobs": True,
                    "capacity_state": "available",
                },
            )
            self.assertEqual(service.calls, [])

    def test_readiness_queue_full_is_503_with_retry_hint(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app) as client:
            manager = app.state.api_runtime_state.manager
            reservations = [
                manager.reserve_capacity() for _ in range(manager.max_queued_jobs)
            ]
            try:
                response = client.get("/api/v1/readiness")
            finally:
                for reservation in reservations:
                    reservation.release()
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json(),
                {
                    "api_version": "v1",
                    "status": "not_ready",
                    "accepting_jobs": False,
                    "capacity_state": "full",
                },
            )
            self.assertEqual(response.headers["Retry-After"], "7")

    def test_worker_death_and_draining_make_readiness_false(self):
        service = FakeService(error=BarrierBaseException())
        app = create_app(self.config, execution_service=service)
        with patch("threading.excepthook"):
            with TestClient(app) as client:
                manager = app.state.api_runtime_state.manager
                with manager.reserve_capacity() as reservation:
                    manager.submit_reserved(
                        reservation,
                        claim="claim",
                        video_path=self.root / "input.mp4",
                    )
                deadline = time.monotonic() + 1
                while manager.worker_alive and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertFalse(manager.worker_alive)
                response = client.get("/api/v1/readiness")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["capacity_state"], "unavailable")
                self.assertFalse(response.json()["accepting_jobs"])
                self.assertEqual(service.calls[0][1], "claim")

    def test_shutdown_state_is_public_safe_not_ready(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app) as client:
            self.assertTrue(app.state.api_lifecycle.shutdown())
            response = client.get("/api/v1/readiness")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["capacity_state"], "unavailable")
            encoded = json.dumps(response.json())
            self.assertNotIn(str(self.root), encoded)
            self.assertNotIn("exception", encoded.lower())

    def test_job_status_preserves_every_public_state_and_polling_semantics(self):
        failure = JobFailureSnapshot(
            code="runtime_execution_failed",
            message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
            incident_id="incident_0123456789abcdef0123456789abcdef",
        )
        cases = (
            (JobState.ACCEPTED, None, None),
            (JobState.QUEUED, 2, None),
            (JobState.RUNNING, None, None),
            (JobState.COMPLETED, None, None),
            (JobState.FAILED, None, failure),
        )
        for state, position, failure_value in cases:
            with self.subTest(state=state):
                app, holder, _ = self.scripted_app(
                    {JOB_A: snapshot(
                        state,
                        queue_position=position,
                        failure=failure_value,
                    )}
                )
                with TestClient(app) as client:
                    response = client.get(f"/api/v1/jobs/{JOB_A}")
                self.assertEqual(response.status_code, 200)
                job = response.json()["job"]
                self.assertEqual(job["state"], state.value)
                self.assertEqual(job["queue_position"], position)
                self.assertEqual(
                    job["poll_after_ms"],
                    None
                    if state in (JobState.COMPLETED, JobState.FAILED)
                    else 4321,
                )
                self.assertEqual(
                    job["links"],
                    {
                        "self": f"/api/v1/jobs/{JOB_A}",
                        "result": f"/api/v1/jobs/{JOB_A}/result",
                    },
                )
                self.assertEqual(holder["manager"].sweep_calls, 1)
                encoded = json.dumps(response.json())
                self.assertNotIn("exact focal claim", encoded)
                self.assertNotIn(PRIVATE_PATH, encoded)
                if state is JobState.FAILED:
                    self.assertEqual(
                        set(job["failure"]),
                        {"code", "message", "incident_id"},
                    )

    def test_queued_status_uses_current_state_not_fabricated_accepted(self):
        app, _, _ = self.scripted_app(
            {JOB_A: snapshot(JobState.QUEUED, queue_position=1)}
        )
        with TestClient(app) as client:
            response = client.get(f"/api/v1/jobs/{JOB_A}")
        self.assertEqual(response.json()["job"]["state"], "queued")
        self.assertEqual(response.json()["job"]["queue_position"], 1)

    def test_expired_malformed_and_unknown_status_mappings_are_safe(self):
        app, _, _ = self.scripted_app(
            {JOB_A: snapshot(JobState.EXPIRED)}
        )
        with TestClient(app) as client:
            expired = client.get(f"/api/v1/jobs/{JOB_A}")
            malformed = client.get("/api/v1/jobs/not-a-job")
            unknown = client.get(f"/api/v1/jobs/{JOB_B}")
        self.assert_public_error(expired, 410, "job_expired")
        self.assert_public_error(malformed, 404, "job_not_found")
        self.assert_public_error(unknown, 404, "job_not_found")
        for key in ("code", "message"):
            self.assertEqual(
                malformed.json()["error"][key],
                unknown.json()["error"][key],
            )

    def test_fake_real_and_nei_completed_results_use_authoritative_outcome(self):
        for verdict in (ModelVerdict.FAKE, ModelVerdict.REAL, ModelVerdict.NOT_RUN):
            with self.subTest(verdict=verdict):
                outcome = success_outcome(verdict)
                before = copy.deepcopy(outcome.to_dict())
                app, holder, _ = self.scripted_app(
                    {JOB_A: snapshot(JobState.COMPLETED)},
                    {JOB_A: outcome},
                )
                with TestClient(app) as client:
                    response = client.get(f"/api/v1/jobs/{JOB_A}/result")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {
                        "api_version": "v1",
                        "job_id": JOB_A,
                        "outcome": outcome.to_dict(),
                    },
                )
                self.assertIs(holder["manager"].outcomes[JOB_A], outcome)
                self.assertEqual(outcome.to_dict(), before)
                self.assertEqual(
                    response.json()["outcome"]["result"]["verdict"][
                        "sample_logits"
                    ],
                    dict(outcome.result.sample_logits),
                )

    def test_result_maps_active_failed_expired_and_unknown_without_leaks(self):
        failure = JobFailureSnapshot(
            code="runtime_execution_failed",
            message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
            incident_id="incident_0123456789abcdef0123456789abcdef",
        )
        cases = (
            (JobState.ACCEPTED, 409, "job_not_completed", None),
            (JobState.QUEUED, 409, "job_not_completed", None),
            (JobState.RUNNING, 409, "job_not_completed", None),
            (JobState.FAILED, 409, "job_failed", failure),
            (JobState.EXPIRED, 410, "job_expired", None),
        )
        for state, status_code, code, failure_value in cases:
            with self.subTest(state=state):
                app, _, _ = self.scripted_app(
                    {JOB_A: snapshot(state, failure=failure_value)}
                )
                with TestClient(app) as client:
                    response = client.get(f"/api/v1/jobs/{JOB_A}/result")
                self.assert_public_error(response, status_code, code)
                encoded = json.dumps(response.json())
                self.assertNotIn("PrivateRuntimeError", encoded)
                self.assertNotIn("verdict", encoded)

        app, _, _ = self.scripted_app()
        with TestClient(app) as client:
            response = client.get(f"/api/v1/jobs/{JOB_B}/result")
        self.assert_public_error(response, 404, "job_not_found")

    def test_completed_outcome_expiration_race_is_rechecked(self):
        app, _, _ = self.scripted_app(
            {
                JOB_A: [
                    snapshot(JobState.COMPLETED),
                    snapshot(JobState.EXPIRED),
                ]
            }
        )
        with TestClient(app) as client:
            response = client.get(f"/api/v1/jobs/{JOB_A}/result")
        self.assert_public_error(response, 410, "job_expired")

    def test_unhandled_exception_is_fixed_500_with_matching_request_id(self):
        manager = ScriptedManager(
            FakeService(),
            {JOB_A: RuntimeError(PRIVATE_PATH + " raw failure")},
        )
        app = create_app(
            self.config,
            execution_service=manager.service,
            job_manager_factory=lambda service: manager,
        )
        with TestClient(app) as client:
            response = client.get(f"/api/v1/jobs/{JOB_A}")
        self.assert_public_error(response, 500, "internal_error")
        encoded = json.dumps(response.json())
        self.assertNotIn(PRIVATE_PATH, encoded)
        self.assertNotIn("RuntimeError", encoded)

    def test_baseexception_is_not_converted_to_http_500(self):
        async def inner(scope, receive, send):
            raise BarrierBaseException()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        messages = []

        async def send(message):
            messages.append(message)

        middleware = _RequestBoundaryMiddleware(inner)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/health",
            "headers": [],
            "state": {},
        }
        with self.assertRaises(BarrierBaseException):
            asyncio.run(middleware(scope, receive, send))
        self.assertEqual(messages, [])

    def test_unknown_route_uses_public_safe_envelope(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app) as client:
            response = client.get("/api/v1/unknown")
        self.assert_public_error(response, 404, "not_found")
        self.assertNotIn("detail", response.json())

    def test_service_not_ready_uses_public_safe_envelope(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app) as client:
            manager = app.state.api_runtime_state.manager
            try:
                app.state.api_runtime_state.manager = None
                response = client.get(f"/api/v1/jobs/{JOB_A}")
            finally:
                app.state.api_runtime_state.manager = manager
        self.assert_public_error(response, 503, "service_not_ready")

    def test_cors_exact_allowlist_exposes_request_id_without_credentials(self):
        app = create_app(self.config, execution_service=FakeService())
        with TestClient(app) as client:
            allowed = client.get(
                "/api/v1/health",
                headers={"Origin": "https://frontend.example"},
            )
            unlisted = client.get(
                "/api/v1/health",
                headers={"Origin": "https://attacker.example"},
            )
            preflight = client.options(
                "/api/v1/jobs",
                headers={
                    "Origin": "https://frontend.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type, Accept",
                },
            )
            rejected_preflight = client.options(
                "/api/v1/jobs",
                headers={
                    "Origin": "https://attacker.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
        self.assertEqual(
            allowed.headers["Access-Control-Allow-Origin"],
            "https://frontend.example",
        )
        self.assertIn(
            "X-Request-ID",
            allowed.headers["Access-Control-Expose-Headers"],
        )
        self.assertNotIn("Access-Control-Allow-Credentials", allowed.headers)
        self.assertNotIn("Access-Control-Allow-Origin", unlisted.headers)
        self.assertEqual(preflight.status_code, 200)
        self.assertNotIn("Access-Control-Allow-Credentials", preflight.headers)
        self.assertRegex(
            preflight.headers["X-Request-ID"],
            r"^req_[0-9a-f]{32}$",
        )
        self.assert_public_error(
            rejected_preflight,
            400,
            "malformed_request",
        )
        self.assertNotIn(
            "Access-Control-Allow-Origin",
            rejected_preflight.headers,
        )

    def test_route_table_has_only_task07c_api_routes_and_no_post_jobs(self):
        app = create_app(self.config, execution_service=FakeService())
        routes = {
            (method, route.path)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        for expected in (
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/readiness"),
            ("GET", "/api/v1/jobs/{job_id}"),
            ("GET", "/api/v1/jobs/{job_id}/result"),
        ):
            self.assertIn(expected, routes)
        self.assertNotIn(("POST", "/api/v1/jobs"), routes)

    def test_production_web_source_preserves_closed_execution_boundary(self):
        webapp_root = Path(__file__).resolve().parents[1] / "webapp"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(webapp_root.glob("*.py"))
        )
        prohibited = (
            "FrozenG1Runner",
            "VideoMultimodalRunner",
            "Whisper",
            "PaddleOCR",
            "SigLIP",
            "Qwen",
            "ProductionResultBuilder",
            "production_cli",
            "shell=True",
            "str(exc)",
            "repr(exc)",
            "traceback",
        )
        for value in prohibited:
            with self.subTest(value=value):
                self.assertNotIn(value, source)
        api_source = inspect.getsource(__import__("webapp.api", fromlist=["*"]))
        self.assertNotIn("multipart", api_source.lower())
        self.assertNotIn(".execute(", api_source)


if __name__ == "__main__":
    unittest.main()
