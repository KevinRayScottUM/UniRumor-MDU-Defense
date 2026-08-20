import inspect
import json
import os
import re
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.production_execution import (
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    OperationalFailure,
    OperationalFailureCode,
    OperationalFailureStage,
    ProductionExecutionOutcome,
    ProductionExecutionStatus,
)
from webapp.api import create_app
from webapp.api_config import APIConfig
from webapp.job_manager import (
    JobManager,
    QueueFullError,
    ReservationError,
)
from webapp.job_types import JobState
from webapp.runtime_lifecycle import APIRuntimeLifecycle, APIRuntimeStartupError
from webapp.server_lock import ServerLock, ServerLockUnavailableError
from webapp.workspace import (
    ALLOWED_INPUT_EXTENSIONS,
    WebWorkspaceError,
    WebWorkspaceManager,
    WebWorkspaceSecurityError,
    validate_production_cache_containment,
)


JOB_A = "job_0123456789abcdef0123456789abcdef"
JOB_B = "job_fedcba9876543210fedcba9876543210"


def success_outcome():
    return ProductionExecutionOutcome(
        schema_version=1,
        status=ProductionExecutionStatus.SUCCESS,
        result=object(),
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
            exception_type="PrivateFailure",
            public_message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
        ),
    )


class ConstantService:
    def __init__(self, outcome=None):
        self.outcome = outcome if outcome is not None else success_outcome()
        self.calls = []

    def execute(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        return self.outcome


class FakeRuntime:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.start_calls = 0

    def start(self):
        self.events.append("runtime.start")
        self.start_calls += 1
        return object()


class FakeService(ConstantService):
    def __init__(self, outcome=None, events=None):
        super().__init__(outcome)
        self.runtime = FakeRuntime(events)


class BlockingService(FakeService):
    def __init__(self):
        super().__init__(success_outcome())
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, session_id, claim, video_path):
        self.calls.append((session_id, claim, video_path))
        self.started.set()
        if not self.release.wait(3):
            raise RuntimeError("test synchronization deadline reached")
        return self.outcome


class CallbackBarrier(BaseException):
    pass


class ReservationAndTerminalCallbackTests(unittest.TestCase):
    VIDEO = Path("/private/task07/jobs/input.mp4")

    def started_manager(self, service=None, **kwargs):
        service = service if service is not None else ConstantService()
        manager = JobManager(service, **kwargs)
        manager.start()

        def cleanup():
            release = getattr(service, "release", None)
            if release is not None:
                release.set()
            manager.shutdown(timeout=2)

        self.addCleanup(cleanup)
        return manager

    @staticmethod
    def submit(manager, claim="exact claim"):
        reservation = manager.reserve_capacity()
        job_id = manager.submit_reserved(
            reservation,
            claim=claim,
            video_path=ReservationAndTerminalCallbackTests.VIDEO,
        )
        return reservation, job_id

    def test_reservation_owns_exact_read_only_id_without_public_job(self):
        manager = self.started_manager(max_queued_jobs=2)
        reservation = manager.reserve_capacity()

        self.assertRegex(reservation.job_id, r"^job_[0-9a-f]{32}$")
        self.assertIsNone(manager.get_snapshot(reservation.job_id))
        self.assertEqual(manager._get_state_history(reservation.job_id), None)
        with self.assertRaises(AttributeError):
            reservation.job_id = JOB_A
        reservation.release()

    def test_submit_consumes_the_exact_reserved_id(self):
        manager = self.started_manager()
        reservation = manager.reserve_capacity()
        reserved_id = reservation.job_id

        submitted_id = manager.submit_reserved(
            reservation,
            claim="exact claim",
            video_path=self.VIDEO,
        )

        self.assertEqual(submitted_id, reserved_id)
        self.assertIsNotNone(manager.get_snapshot(submitted_id))

    def test_released_reservation_creates_no_job_or_history(self):
        manager = self.started_manager(max_queued_jobs=1)
        reservation = manager.reserve_capacity()
        reserved_id = reservation.job_id
        reservation.release()
        reservation.release()

        self.assertIsNone(manager.get_snapshot(reserved_id))
        self.assertIsNone(manager._get_state_history(reserved_id))
        self.assertEqual(manager.reservation_count, 0)
        with self.assertRaises(ReservationError):
            manager.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.VIDEO,
            )

    def test_released_future_id_is_no_longer_reserved(self):
        manager = self.started_manager(max_queued_jobs=1)
        value = "a" * 32
        with patch("webapp.job_manager.secrets.token_hex", return_value=value):
            first = manager.reserve_capacity()
            first.release()
            second = manager.reserve_capacity()
        self.assertEqual(first.job_id, second.job_id)
        second.release()

    def test_active_and_existing_ids_are_both_collision_checked(self):
        manager = self.started_manager(max_queued_jobs=3)
        with patch(
            "webapp.job_manager.secrets.token_hex",
            side_effect=["a" * 32, "a" * 32, "b" * 32],
        ):
            first = manager.reserve_capacity()
            second = manager.reserve_capacity()
        self.assertNotEqual(first.job_id, second.job_id)
        manager.submit_reserved(first, claim="claim", video_path=self.VIDEO)
        second.release()

        with patch(
            "webapp.job_manager.secrets.token_hex",
            side_effect=["a" * 32, "c" * 32],
        ):
            third = manager.reserve_capacity()
        self.assertEqual(third.job_id, "job_" + "c" * 32)
        third.release()

    def test_concurrent_reservations_have_unique_cryptographic_ids(self):
        manager = self.started_manager(max_queued_jobs=16)
        barrier = threading.Barrier(17)
        reservations = []
        collection_lock = threading.Lock()

        def reserve():
            barrier.wait()
            reservation = manager.reserve_capacity()
            with collection_lock:
                reservations.append(reservation)

        threads = [threading.Thread(target=reserve) for _ in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        ids = [reservation.job_id for reservation in reservations]
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 16)
        self.assertTrue(all(re.fullmatch(r"job_[0-9a-f]{32}", value) for value in ids))
        for reservation in reservations:
            reservation.release()

    def test_reserved_ids_count_against_bounded_capacity(self):
        manager = self.started_manager(max_queued_jobs=2)
        first = manager.reserve_capacity()
        second = manager.reserve_capacity()
        self.assertEqual(manager.reservation_count, 2)
        with self.assertRaises(QueueFullError):
            manager.reserve_capacity()
        self.assertEqual(manager._jobs, {})
        first.release()
        second.release()

    def test_cross_manager_reservation_behavior_is_unchanged(self):
        first = self.started_manager()
        second = self.started_manager()
        reservation = first.reserve_capacity()
        reserved_id = reservation.job_id

        with self.assertRaises(ReservationError):
            second.submit_reserved(
                reservation,
                claim="claim",
                video_path=self.VIDEO,
            )
        self.assertTrue(reservation.active)
        submitted = first.submit_reserved(
            reservation,
            claim="claim",
            video_path=self.VIDEO,
        )
        self.assertEqual(submitted, reserved_id)

    def test_completed_callback_is_once_post_state_and_outside_lock(self):
        observed = []
        callback_finished = threading.Event()
        holder = {}

        def on_terminal(job_id):
            manager = holder["manager"]
            observed.append(
                (
                    job_id,
                    manager.get_snapshot(job_id).state,
                    manager._get_completed_outcome(job_id),
                    manager._condition._is_owned(),
                )
            )
            callback_finished.set()

        outcome = success_outcome()
        manager = self.started_manager(
            ConstantService(outcome),
            on_terminal=on_terminal,
        )
        holder["manager"] = manager
        _, job_id = self.submit(manager)

        self.assertIsNotNone(
            manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
        )
        self.assertTrue(callback_finished.wait(1))
        self.assertEqual(observed, [(job_id, JobState.COMPLETED, outcome, False)])
        manager.sweep_expired()
        self.assertEqual(len(observed), 1)

    def test_failed_callback_observes_stored_public_failure_once(self):
        observed = []
        callback_finished = threading.Event()
        holder = {}

        def on_terminal(job_id):
            snapshot = holder["manager"].get_snapshot(job_id)
            observed.append((job_id, snapshot.state, snapshot.failure.code))
            callback_finished.set()

        manager = self.started_manager(
            ConstantService(failure_outcome()),
            on_terminal=on_terminal,
        )
        holder["manager"] = manager
        _, job_id = self.submit(manager)
        self.assertIsNotNone(manager.wait_for_state(job_id, JobState.FAILED, 1))
        self.assertTrue(callback_finished.wait(1))
        self.assertEqual(
            observed,
            [(job_id, JobState.FAILED, "runtime_execution_failed")],
        )

    def test_callback_exception_preserves_outcome_and_worker_continues(self):
        calls = []
        callbacks_finished = threading.Event()

        def on_terminal(job_id):
            calls.append(job_id)
            if len(calls) == 2:
                callbacks_finished.set()
            raise RuntimeError("cleanup failed")

        outcome = success_outcome()
        manager = self.started_manager(
            ConstantService(outcome),
            max_queued_jobs=2,
            on_terminal=on_terminal,
        )
        _, first = self.submit(manager, "first")
        _, second = self.submit(manager, "second")

        self.assertIsNotNone(
            manager.wait_for_state(first, JobState.COMPLETED, timeout=1)
        )
        self.assertIsNotNone(
            manager.wait_for_state(second, JobState.COMPLETED, timeout=1)
        )
        self.assertTrue(callbacks_finished.wait(1))
        self.assertEqual(calls, [first, second])
        self.assertIs(manager._get_completed_outcome(first), outcome)
        self.assertIs(manager._get_completed_outcome(second), outcome)
        self.assertTrue(manager.worker_alive)

    def test_callback_exception_does_not_rewrite_failed_outcome(self):
        outcome = failure_outcome()
        manager = self.started_manager(
            ConstantService(outcome),
            on_terminal=lambda job_id: (_ for _ in ()).throw(
                RuntimeError("cleanup failed")
            ),
        )
        _, job_id = self.submit(manager)
        snapshot = manager.wait_for_state(job_id, JobState.FAILED, timeout=1)

        self.assertEqual(snapshot.failure.code, "runtime_execution_failed")
        with manager._condition:
            self.assertIs(manager._jobs[job_id].outcome, outcome)
        self.assertTrue(manager.worker_alive)

    def test_callback_does_not_intentionally_swallow_baseexception(self):
        callback_called = threading.Event()

        def on_terminal(job_id):
            callback_called.set()
            raise CallbackBarrier()

        manager = self.started_manager(on_terminal=on_terminal)
        with patch("threading.excepthook"):
            _, job_id = self.submit(manager)
            self.assertIsNotNone(
                manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
            )
            self.assertTrue(callback_called.wait(1))
            deadline = time.monotonic() + 1
            while manager.worker_alive and time.monotonic() < deadline:
                time.sleep(0.005)
        self.assertFalse(manager.worker_alive)
        self.assertEqual(manager.get_snapshot(job_id).state, JobState.COMPLETED)


class WebWorkspaceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name).resolve()
        self.root = self.base / "web-runtime"
        self.root.mkdir()

    def initialized_manager(self):
        manager = WebWorkspaceManager(self.root)
        manager.initialize()
        return manager

    def test_constructor_is_side_effect_free(self):
        before = sorted(self.base.rglob("*"))
        manager = WebWorkspaceManager(self.root)
        after = sorted(self.base.rglob("*"))

        self.assertFalse(manager.initialized)
        self.assertFalse(manager.jobs_root.exists())
        self.assertEqual(after, before)
        with self.assertRaises(WebWorkspaceError):
            manager.prepare_job_workspace(JOB_A)

    def test_constructor_rejects_symlink_relative_and_noncanonical_roots(self):
        link = self.base / "runtime-link"
        link.symlink_to(self.root, target_is_directory=True)
        for candidate in (
            link,
            Path("relative-web-runtime"),
            self.root / ".." / "web-runtime",
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                WebWorkspaceManager(candidate)

    def test_initialize_creates_owner_only_jobs_root(self):
        manager = WebWorkspaceManager(self.root)
        manager.initialize()
        manager.initialize()

        self.assertTrue(manager.initialized)
        self.assertTrue(manager.jobs_root.is_dir())
        self.assertEqual(stat.S_IMODE(manager.jobs_root.stat().st_mode), 0o700)

    def test_symlink_or_file_jobs_root_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        jobs = self.root / "jobs"
        jobs.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(WebWorkspaceSecurityError):
            WebWorkspaceManager(self.root).initialize()
        jobs.unlink()
        jobs.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(WebWorkspaceSecurityError):
            WebWorkspaceManager(self.root).initialize()

    def test_valid_workspace_and_fixed_input_paths_stay_beneath_jobs(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)

        self.assertEqual(job_path, self.root / "jobs" / JOB_A)
        self.assertEqual(job_path.parent, manager.jobs_root)
        self.assertEqual(stat.S_IMODE(job_path.stat().st_mode), 0o700)
        for extension in sorted(ALLOWED_INPUT_EXTENSIONS):
            with self.subTest(extension=extension):
                self.assertEqual(
                    manager.job_input_path(JOB_A, extension),
                    job_path / ("input" + extension),
                )

    def test_invalid_job_ids_and_caller_like_paths_are_rejected(self):
        manager = self.initialized_manager()
        invalid = (
            "",
            "job_ABCDEF0123456789ABCDEF0123456789",
            "job_0123",
            "../" + JOB_A,
            JOB_A + "/child",
            "/" + JOB_A,
            "job_0123456789abcdef0123456789abcdeg",
        )
        for job_id in invalid:
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                manager.prepare_job_workspace(job_id)
            with self.subTest(cleanup=job_id), self.assertRaises(ValueError):
                manager.cleanup_job(job_id)

    def test_internal_extension_allowlist_is_exact(self):
        manager = self.initialized_manager()
        manager.prepare_job_workspace(JOB_A)
        invalid = ("mp4", ".MP4", ".avi", "../mp4", ".mp4/other", "input.mp4")
        for extension in invalid:
            with self.subTest(extension=extension), self.assertRaises(ValueError):
                manager.job_input_path(JOB_A, extension)

    def test_input_path_rejects_an_existing_symlink(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)
        outside = self.base / "outside.mp4"
        outside.write_bytes(b"outside")
        (job_path / "input.mp4").symlink_to(outside)

        with self.assertRaises(WebWorkspaceSecurityError):
            manager.job_input_path(JOB_A, ".mp4")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_input_creation_is_owner_only_exclusive_and_fixed_name(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)
        with manager.create_job_input(JOB_A, ".mp4") as output:
            output.write(b"video")

        input_path = job_path / "input.mp4"
        self.assertEqual(input_path.read_bytes(), b"video")
        self.assertEqual(stat.S_IMODE(input_path.stat().st_mode), 0o600)
        with self.assertRaises(WebWorkspaceSecurityError):
            manager.create_job_input(JOB_A, ".mp4")

    def test_input_creation_never_follows_an_existing_symlink(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)
        outside = self.base / "outside.mp4"
        outside.write_bytes(b"preserve")
        (job_path / "input.mp4").symlink_to(outside)

        with self.assertRaises(WebWorkspaceSecurityError):
            manager.create_job_input(JOB_A, ".mp4")
        self.assertEqual(outside.read_bytes(), b"preserve")

    def test_cleanup_is_recursive_and_idempotent(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)
        nested = job_path / "nested" / "deeper"
        nested.mkdir(parents=True)
        (nested / "ordinary.bin").write_bytes(b"data")

        manager.cleanup_job(JOB_A)
        manager.cleanup_job(JOB_A)
        self.assertFalse(job_path.exists())
        self.assertTrue(manager.jobs_root.exists())

    def test_symlinks_inside_workspace_are_unlinked_not_traversed(self):
        manager = self.initialized_manager()
        job_path = manager.prepare_job_workspace(JOB_A)
        outside = self.base / "outside"
        outside.mkdir()
        outside_file = outside / "preserve.txt"
        outside_file.write_text("preserve", encoding="utf-8")
        (job_path / "outside-link").symlink_to(outside, target_is_directory=True)
        (job_path / "file-link").symlink_to(outside_file)

        manager.cleanup_job(JOB_A)
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "preserve")
        self.assertTrue(outside.is_dir())

    def test_matching_orphans_removed_and_unknown_siblings_preserved(self):
        manager = self.initialized_manager()
        matching = manager.jobs_root / JOB_A
        matching.mkdir()
        (matching / "partial.bin").write_bytes(b"partial")
        unknown = manager.jobs_root / "keep-me"
        unknown.mkdir()
        task06_names = []
        for name in ("ocr", "visual", "g1"):
            sibling = self.root / name
            sibling.mkdir()
            (sibling / "artifact").write_bytes(b"owned by Task06")
            task06_names.append(sibling)
        web_sibling = self.base / "outside-web-runtime"
        web_sibling.mkdir()

        self.assertEqual(manager.cleanup_orphans(), 1)
        self.assertFalse(matching.exists())
        self.assertTrue(unknown.exists())
        self.assertTrue(web_sibling.exists())
        for sibling in task06_names:
            self.assertTrue((sibling / "artifact").exists())

    def test_matching_orphan_symlink_is_removed_without_target_deletion(self):
        manager = self.initialized_manager()
        outside = self.base / "outside-target"
        outside.mkdir()
        marker = outside / "marker"
        marker.write_text("preserve", encoding="utf-8")
        link = manager.jobs_root / JOB_A
        link.symlink_to(outside, target_is_directory=True)

        manager.cleanup_orphans()
        self.assertFalse(os.path.lexists(link))
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_replaced_jobs_root_symlink_is_rejected_without_target_changes(self):
        manager = self.initialized_manager()
        manager.jobs_root.rmdir()
        outside = self.base / "outside-target"
        outside.mkdir()
        marker = outside / "marker"
        marker.write_text("preserve", encoding="utf-8")
        manager.jobs_root.symlink_to(outside, target_is_directory=True)

        with self.assertRaises(WebWorkspaceSecurityError):
            manager.cleanup_job(JOB_A)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_existing_job_workspace_is_never_reused(self):
        manager = self.initialized_manager()
        manager.prepare_job_workspace(JOB_A)
        with self.assertRaises(WebWorkspaceSecurityError):
            manager.prepare_job_workspace(JOB_A)


class RecordingLock:
    def __init__(self, root, events):
        self._lock = ServerLock(root)
        self.events = events

    def acquire(self):
        self.events.append("lock.acquire")
        self._lock.acquire()
        return self

    def release(self):
        self.events.append("lock.release")
        self._lock.release()


class RecordingWorkspace:
    def __init__(self, root, events):
        events.append("workspace.construct")
        self._workspace = WebWorkspaceManager(root)
        self.events = events

    @property
    def jobs_root(self):
        return self._workspace.jobs_root

    def initialize(self):
        self.events.append("workspace.initialize")
        return self._workspace.initialize()

    def cleanup_orphans(self):
        self.events.append("workspace.cleanup_orphans")
        return self._workspace.cleanup_orphans()

    def prepare_job_workspace(self, job_id):
        return self._workspace.prepare_job_workspace(job_id)

    def job_input_path(self, job_id, extension):
        return self._workspace.job_input_path(job_id, extension)

    def create_job_input(self, job_id, extension):
        return self._workspace.create_job_input(job_id, extension)

    def cleanup_job(self, job_id):
        self.events.append("workspace.cleanup_job")
        return self._workspace.cleanup_job(job_id)

    def cleanup_all_job_workspaces(self):
        self.events.append("workspace.cleanup_all")
        return self._workspace.cleanup_all_job_workspaces()


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


class WorkspaceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name).resolve()

    def write_production_config(self, cache_root):
        config_dir = self.base / "config"
        config_dir.mkdir(exist_ok=True)
        path = config_dir / "production.json"
        payload = {
            "schema_version": 1,
            "profile": "production",
            "cache_root": str(cache_root),
            "output_root": str(self.base / "outputs"),
            "whisper": {
                "model_path": str(self.base / "models" / "whisper"),
                "device": "cuda",
                "dtype": "float16",
            },
            "ocr": {
                "detector_model_path": str(self.base / "models" / "detector"),
                "recognizer_model_path": str(self.base / "models" / "recognizer"),
                "python_executable": "/usr/bin/python3",
                "device": "gpu:0",
                "cudnn8_library_path": str(self.base / "cudnn8"),
                "timeout_seconds": 60,
            },
            "siglip": {
                "model_path": str(self.base / "models" / "siglip"),
                "device": "cuda",
            },
            "qwen": {
                "model_path": str(self.base / "models" / "qwen"),
                "device": "cuda",
            },
            "frozen_g1": {
                "unirumor_root": str(self.base / "unirumor"),
                "python_executable": "/usr/bin/python3",
                "phase4a_infer": str(self.base / "phase4a.py"),
                "phase4a_config": str(self.base / "phase4a.json"),
                "device": "cuda",
                "timeout_seconds": 60,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_cache_containment_accepts_only_strict_descendant(self):
        cache = self.base / "cache"
        web = cache / "web-runtime"
        web.mkdir(parents=True)
        validate_production_cache_containment(web, cache)

    def test_cache_containment_rejects_equal_root(self):
        cache = self.base / "cache"
        cache.mkdir()
        with self.assertRaises(ValueError):
            validate_production_cache_containment(cache, cache)

    def test_cache_containment_rejects_outside_root(self):
        cache = self.base / "cache"
        web = self.base / "outside" / "web-runtime"
        cache.mkdir()
        web.mkdir(parents=True)
        with self.assertRaises(ValueError):
            validate_production_cache_containment(web, cache)

    def test_startup_order_is_lock_workspace_cleanup_service_runtime_manager(self):
        events = []
        root = self.base / "web-runtime"
        orphan = root / "jobs" / JOB_A
        orphan.mkdir(parents=True)
        service = FakeService(events=events)

        def service_provider():
            self.assertFalse(orphan.exists())
            events.append("service.obtain")
            return service

        lifecycle = APIRuntimeLifecycle(
            APIConfig(root),
            service_provider,
            server_lock_factory=lambda value: RecordingLock(value, events),
            workspace_manager_factory=lambda value: RecordingWorkspace(
                value, events
            ),
            job_manager_factory=lambda value: RecordingManager(value, events),
        )
        lifecycle.startup()
        self.addCleanup(lifecycle.shutdown)

        self.assertEqual(
            events[:8],
            [
                "lock.acquire",
                "workspace.construct",
                "workspace.initialize",
                "workspace.cleanup_orphans",
                "service.obtain",
                "runtime.start",
                "manager.construct",
                "manager.start",
            ],
        )

    def test_startup_rejects_missing_workspace_submission_method(self):
        required_methods = (
            "prepare_job_workspace",
            "job_input_path",
            "create_job_input",
        )
        for method_name in required_methods:
            with self.subTest(method_name=method_name):
                root = self.base / ("missing-" + method_name)
                root.mkdir()
                events = []
                service = FakeService(events=events)

                def workspace_factory(value, missing=method_name):
                    workspace = RecordingWorkspace(value, events)
                    methods = {
                        name: getattr(workspace, name)
                        for name in (
                            "initialize",
                            "cleanup_orphans",
                            "cleanup_job",
                            "cleanup_all_job_workspaces",
                            *required_methods,
                        )
                        if name != missing
                    }
                    return SimpleNamespace(**methods)

                lifecycle = APIRuntimeLifecycle(
                    APIConfig(root),
                    lambda: service,
                    workspace_manager_factory=workspace_factory,
                )

                with self.assertRaises(APIRuntimeStartupError):
                    lifecycle.startup()
                readiness = lifecycle.state.readiness()
                self.assertEqual(readiness.status, "not_ready")
                self.assertFalse(readiness.accepting_jobs)
                self.assertFalse(lifecycle.state.singleton_acquired)
                self.assertEqual(service.runtime.start_calls, 0)

    def test_startup_rejects_non_callable_workspace_submission_method(self):
        required_methods = (
            "prepare_job_workspace",
            "job_input_path",
            "create_job_input",
        )
        for method_name in required_methods:
            with self.subTest(method_name=method_name):
                root = self.base / ("non-callable-" + method_name)
                root.mkdir()
                events = []
                service = FakeService(events=events)

                def workspace_factory(value, invalid=method_name):
                    workspace = RecordingWorkspace(value, events)
                    setattr(workspace, invalid, None)
                    return workspace

                lifecycle = APIRuntimeLifecycle(
                    APIConfig(root),
                    lambda: service,
                    workspace_manager_factory=workspace_factory,
                )

                with self.assertRaises(APIRuntimeStartupError):
                    lifecycle.startup()
                self.assertFalse(lifecycle.state.readiness().accepting_jobs)
                self.assertFalse(lifecycle.state.singleton_acquired)
                self.assertEqual(service.runtime.start_calls, 0)

    def test_startup_rejects_incompatible_workspace_submission_signature(self):
        incompatible_methods = {
            "prepare_job_workspace": lambda: None,
            "job_input_path": lambda job_id: None,
            "create_job_input": lambda job_id: None,
        }
        for method_name, incompatible in incompatible_methods.items():
            with self.subTest(method_name=method_name):
                root = self.base / ("incompatible-" + method_name)
                root.mkdir()
                events = []
                service = FakeService(events=events)

                def workspace_factory(
                    value,
                    invalid=method_name,
                    replacement=incompatible,
                ):
                    workspace = RecordingWorkspace(value, events)
                    setattr(workspace, invalid, replacement)
                    return workspace

                lifecycle = APIRuntimeLifecycle(
                    APIConfig(root),
                    lambda: service,
                    workspace_manager_factory=workspace_factory,
                )

                with self.assertRaises(APIRuntimeStartupError):
                    lifecycle.startup()
                self.assertFalse(lifecycle.state.readiness().accepting_jobs)
                self.assertFalse(lifecycle.state.singleton_acquired)
                self.assertEqual(service.runtime.start_calls, 0)

    def test_compatible_workspace_submission_signatures_are_not_executed(self):
        root = self.base / "compatible-submission-workspace"
        root.mkdir()
        events = []
        submission_calls = []
        service = FakeService(events=events)

        def workspace_factory(value):
            workspace = RecordingWorkspace(value, events)
            prepare = workspace.prepare_job_workspace
            input_path = workspace.job_input_path
            create_input = workspace.create_job_input

            def prepare_job_workspace(job_id):
                submission_calls.append("prepare_job_workspace")
                return prepare(job_id)

            def job_input_path(job_id, extension):
                submission_calls.append("job_input_path")
                return input_path(job_id, extension)

            def create_job_input(job_id, extension):
                submission_calls.append("create_job_input")
                return create_input(job_id, extension)

            workspace.prepare_job_workspace = prepare_job_workspace
            workspace.job_input_path = job_input_path
            workspace.create_job_input = create_job_input
            return workspace

        lifecycle = APIRuntimeLifecycle(
            APIConfig(root),
            lambda: service,
            workspace_manager_factory=workspace_factory,
        )
        lifecycle.startup()
        self.addCleanup(lifecycle.shutdown)

        readiness = lifecycle.state.readiness()
        self.assertEqual(readiness.status, "ready")
        self.assertTrue(readiness.accepting_jobs)
        self.assertEqual(submission_calls, [])

    def test_lock_contention_performs_zero_workspace_or_runtime_work(self):
        root = self.base / "web-runtime"
        root.mkdir()
        owner = ServerLock(root).acquire()
        self.addCleanup(owner.release)
        events = []
        lifecycle = APIRuntimeLifecycle(
            APIConfig(root),
            lambda: events.append("service.obtain"),
            workspace_manager_factory=lambda value: events.append(
                "workspace.construct"
            ),
            job_manager_factory=lambda value: events.append("manager.construct"),
        )

        with self.assertRaises(APIRuntimeStartupError):
            lifecycle.startup()
        self.assertEqual(events, [])
        self.assertFalse((root / "jobs").exists())

    def test_real_config_containment_is_checked_before_lock(self):
        cache = self.base / "cache"
        cache.mkdir()
        outside = self.base / "outside-web-runtime"
        outside.mkdir()
        config_path = self.write_production_config(cache)
        events = []
        app = create_app(
            APIConfig(outside, production_runtime_config_path=config_path),
            execution_service_factory=lambda: events.append("service.obtain"),
            server_lock_factory=lambda value: events.append("lock.construct"),
        )

        with self.assertRaises(APIRuntimeStartupError):
            with TestClient(app):
                pass
        self.assertEqual(events, [])
        self.assertFalse((outside / ".server.lock").exists())
        self.assertFalse((outside / "jobs").exists())

    def test_missing_real_config_fails_before_lock_or_workspace(self):
        cache = self.base / "cache"
        root = cache / "web-runtime"
        root.mkdir(parents=True)
        missing_config = self.base / "config" / "missing.json"
        events = []
        app = create_app(
            APIConfig(root, production_runtime_config_path=missing_config),
            execution_service_factory=lambda: events.append("service.obtain"),
            server_lock_factory=lambda value: events.append("lock.construct"),
            workspace_manager_factory=lambda value: events.append(
                "workspace.construct"
            ),
        )

        with self.assertRaises(APIRuntimeStartupError):
            with TestClient(app):
                pass
        self.assertEqual(events, [])
        self.assertFalse((root / ".server.lock").exists())
        self.assertFalse((root / "jobs").exists())

    def test_real_config_containment_accepts_descendant_without_model_load(self):
        cache = self.base / "cache"
        root = cache / "web-runtime"
        root.mkdir(parents=True)
        config_path = self.write_production_config(cache)
        service = FakeService()
        app = create_app(
            APIConfig(root, production_runtime_config_path=config_path),
            execution_service=service,
        )

        with TestClient(app):
            self.assertEqual(service.runtime.start_calls, 1)
            self.assertEqual(service.calls, [])

    def test_default_manager_terminal_callback_cleans_only_job_workspace(self):
        root = self.base / "web-runtime"
        root.mkdir()
        service = FakeService()
        app = create_app(APIConfig(root), execution_service=service)

        with TestClient(app):
            state = app.state.api_runtime_state
            manager = state.manager
            workspace = state.workspace_manager
            reservation = manager.reserve_capacity()
            job_path = workspace.prepare_job_workspace(reservation.job_id)
            input_path = workspace.job_input_path(reservation.job_id, ".mp4")
            input_path.write_bytes(b"test video placeholder")
            job_id = manager.submit_reserved(
                reservation,
                claim="exact claim",
                video_path=input_path,
            )
            self.assertEqual(job_id, reservation.job_id)
            self.assertIsNotNone(
                manager.wait_for_state(job_id, JobState.COMPLETED, timeout=1)
            )
            deadline = time.monotonic() + 1
            while job_path.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertFalse(job_path.exists())

    def test_clean_shutdown_cleans_owned_workspaces_before_lock_release(self):
        root = self.base / "web-runtime"
        root.mkdir()
        events = []
        app = create_app(
            APIConfig(root),
            execution_service=FakeService(events=events),
            server_lock_factory=lambda value: RecordingLock(value, events),
            workspace_manager_factory=lambda value: RecordingWorkspace(
                value, events
            ),
        )

        with TestClient(app):
            workspace = app.state.api_runtime_state.workspace_manager
            job_path = workspace.prepare_job_workspace(JOB_A)
            self.assertTrue(job_path.exists())
        self.assertFalse(job_path.exists())
        self.assertLess(
            events.index("workspace.cleanup_all"),
            events.index("lock.release"),
        )

    def test_incomplete_shutdown_retains_workspace_and_singleton_lock(self):
        root = self.base / "web-runtime"
        root.mkdir()
        service = BlockingService()
        lifecycle = APIRuntimeLifecycle(
            APIConfig(root, graceful_shutdown_timeout_seconds=0.01),
            lambda: service,
        )
        lifecycle.startup()
        self.addCleanup(service.release.set)
        self.addCleanup(lifecycle.shutdown)
        manager = lifecycle.state.manager
        workspace = lifecycle.state.workspace_manager
        reservation = manager.reserve_capacity()
        job_path = workspace.prepare_job_workspace(reservation.job_id)
        input_path = workspace.job_input_path(reservation.job_id, ".mp4")
        input_path.write_bytes(b"test video placeholder")
        manager.submit_reserved(
            reservation,
            claim="exact claim",
            video_path=input_path,
        )
        self.assertTrue(service.started.wait(1))

        self.assertFalse(lifecycle.shutdown())
        self.assertTrue(job_path.exists())
        self.assertTrue(lifecycle.state.singleton_acquired)
        with self.assertRaises(ServerLockUnavailableError):
            ServerLock(root).acquire()

        service.release.set()
        self.assertIsNotNone(
            manager.wait_for_state(reservation.job_id, JobState.COMPLETED, 1)
        )
        self.assertTrue(lifecycle.shutdown())
        self.assertFalse(lifecycle.state.singleton_acquired)


class WorkspaceBoundaryTests(unittest.TestCase):
    def test_post_jobs_uses_streaming_parser_without_uploadfile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                APIConfig(Path(temporary_directory)),
                execution_service=FakeService(),
            )
            routes = {
                (method, route.path)
                for route in app.routes
                for method in getattr(route, "methods", set())
            }
        self.assertIn(("POST", "/api/v1/jobs"), routes)
        webapp_root = Path(__file__).resolve().parents[1] / "webapp"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(webapp_root.glob("*.py"))
        )
        self.assertNotIn("UploadFile", source)
        self.assertIn("request.stream()", source)

    def test_task07d_production_source_has_no_direct_scientific_dependency(self):
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


if __name__ == "__main__":
    unittest.main()
