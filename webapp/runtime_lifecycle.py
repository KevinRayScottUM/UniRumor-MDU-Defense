"""Ordered FastAPI lifespan ownership for one production execution lane."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from .api_config import APIConfig, validate_web_runtime_root
from .api_types import ReadinessPayload
from .job_manager import JobManager
from .server_lock import ServerLock


class APIRuntimeStartupError(RuntimeError):
    """Fixed internal startup failure without deployment details."""


class APIRuntimeState:
    """Small coarse lifecycle state; no exception or deployment detail is stored."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self.singleton_acquired = False
        self.startup_complete = False
        self.startup_failed = False
        self.shutdown_started = False
        self.shutdown_complete = False
        self.shutdown_incomplete = False
        self.execution_service = None
        self.manager = None
        self.server_lock = None

    def readiness(self) -> ReadinessPayload:
        with self._guard:
            manager = self.manager
            available = (
                self.singleton_acquired
                and self.startup_complete
                and not self.startup_failed
                and not self.shutdown_started
                and manager is not None
                and manager.worker_alive
            )
            if not available:
                return ReadinessPayload(
                    status="not_ready",
                    accepting_jobs=False,
                    capacity_state="unavailable",
                )
            if manager.accepting_jobs:
                return ReadinessPayload(
                    status="ready",
                    accepting_jobs=True,
                    capacity_state="available",
                )
            if (
                manager.queued_count + manager.reservation_count
                >= manager.max_queued_jobs
            ):
                return ReadinessPayload(
                    status="not_ready",
                    accepting_jobs=False,
                    capacity_state="full",
                )
            return ReadinessPayload(
                status="not_ready",
                accepting_jobs=False,
                capacity_state="unavailable",
            )


class APIRuntimeLifecycle:
    """Acquire ownership, start the closed runtime graph, then start one worker."""

    def __init__(
        self,
        config: APIConfig,
        execution_service_provider: Callable[[], object],
        *,
        server_lock_factory: Callable[[Path], object] = ServerLock,
        job_manager_factory: Callable[[object], object] = JobManager,
    ) -> None:
        if not isinstance(config, APIConfig):
            raise TypeError("config must be an APIConfig")
        if not callable(execution_service_provider):
            raise TypeError("execution_service_provider must be callable")
        if not callable(server_lock_factory):
            raise TypeError("server_lock_factory must be callable")
        if not callable(job_manager_factory):
            raise TypeError("job_manager_factory must be callable")
        self.config = config
        self._execution_service_provider = execution_service_provider
        self._server_lock_factory = server_lock_factory
        self._job_manager_factory = job_manager_factory
        self.state = APIRuntimeState()

    def _release_lock(self) -> None:
        lock = self.state.server_lock
        if lock is None or not self.state.singleton_acquired:
            return
        lock.release()
        with self.state._guard:
            self.state.singleton_acquired = False

    def _failed_startup_cleanup(self) -> None:
        manager = self.state.manager
        stopped = True
        if manager is not None:
            stopped = manager.shutdown(
                timeout=self.config.graceful_shutdown_timeout_seconds
            )
        if stopped:
            self._release_lock()
        else:
            with self.state._guard:
                self.state.shutdown_incomplete = True

    def startup(self) -> None:
        """Use explicit startup failure instead of serving deceptive liveness."""

        canonical_root = validate_web_runtime_root(self.config.web_runtime_root)
        if canonical_root != self.config.web_runtime_root:
            with self.state._guard:
                self.state.startup_failed = True
            raise APIRuntimeStartupError("API runtime startup failed")

        try:
            lock = self._server_lock_factory(canonical_root)
            lock.acquire()
        except Exception:
            with self.state._guard:
                self.state.startup_failed = True
            raise APIRuntimeStartupError("API runtime startup failed") from None

        with self.state._guard:
            self.state.server_lock = lock
            self.state.singleton_acquired = True

        try:
            service = self._execution_service_provider()
            if not callable(getattr(service, "execute", None)):
                raise TypeError("execution service must provide execute")
            runtime = getattr(service, "runtime", None)
            if not callable(getattr(runtime, "start", None)):
                raise TypeError("execution service runtime must provide start")
            with self.state._guard:
                self.state.execution_service = service
            runtime.start()

            manager = self._job_manager_factory(service)
            if not callable(getattr(manager, "start", None)):
                raise TypeError("job manager must provide start")
            with self.state._guard:
                self.state.manager = manager
            if not manager.start() or not manager.worker_alive:
                raise RuntimeError("job manager worker did not start")
        except Exception:
            with self.state._guard:
                self.state.startup_failed = True
            self._failed_startup_cleanup()
            raise APIRuntimeStartupError("API runtime startup failed") from None

        with self.state._guard:
            self.state.startup_complete = True

    def shutdown(self) -> bool:
        with self.state._guard:
            if self.state.shutdown_complete:
                return True
            self.state.shutdown_started = True
            manager = self.state.manager

        stopped = True
        if manager is not None:
            stopped = manager.shutdown(
                timeout=self.config.graceful_shutdown_timeout_seconds
            )
        if not stopped:
            with self.state._guard:
                self.state.shutdown_incomplete = True
            return False

        self._release_lock()
        with self.state._guard:
            self.state.shutdown_complete = True
            self.state.shutdown_incomplete = False
        return True

    @asynccontextmanager
    async def lifespan(self, app):
        self.startup()
        try:
            yield
        finally:
            self.shutdown()


__all__ = [
    "APIRuntimeLifecycle",
    "APIRuntimeStartupError",
    "APIRuntimeState",
]
