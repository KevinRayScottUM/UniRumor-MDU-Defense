"""Ordered FastAPI lifespan ownership for one production execution lane."""

from __future__ import annotations

import inspect
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from schemas import ProductionRuntimeConfig

from .api_config import APIConfig, validate_web_runtime_root
from .api_types import ReadinessPayload
from .execution_adapter import ProductionExecutionAdapter
from .job_manager import JobManager
from .server_lock import ServerLock
from .workspace import (
    WebWorkspaceManager,
    validate_production_cache_containment,
)


class APIRuntimeStartupError(RuntimeError):
    """Fixed internal startup failure without deployment details."""


def _require_positional_interface(
    owner: object,
    method_name: str,
    positional_argument_count: int,
    *,
    owner_name: str = "workspace manager",
) -> None:
    """Validate a bound callable signature without invoking the dependency."""

    method = getattr(owner, method_name, None)
    if not callable(method):
        raise TypeError(f"{owner_name} must provide {method_name}")
    try:
        signature = inspect.signature(method)
        signature.bind(*(object() for _ in range(positional_argument_count)))
    except (TypeError, ValueError):
        raise TypeError(
            f"{owner_name} method {method_name} has an incompatible signature"
        ) from None


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
        self.execution_adapter = None
        self.execution_service = None
        self.manager = None
        self.server_lock = None
        self.workspace_manager = None

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
        execution_adapter_factory: Callable[[object], object] = (
            ProductionExecutionAdapter
        ),
        server_lock_factory: Callable[[Path], object] = ServerLock,
        job_manager_factory: Callable[[object], object] = JobManager,
        workspace_manager_factory: Callable[[Path], object] = WebWorkspaceManager,
    ) -> None:
        if not isinstance(config, APIConfig):
            raise TypeError("config must be an APIConfig")
        if not callable(execution_service_provider):
            raise TypeError("execution_service_provider must be callable")
        if not callable(execution_adapter_factory):
            raise TypeError("execution_adapter_factory must be callable")
        if not callable(server_lock_factory):
            raise TypeError("server_lock_factory must be callable")
        if not callable(job_manager_factory):
            raise TypeError("job_manager_factory must be callable")
        if not callable(workspace_manager_factory):
            raise TypeError("workspace_manager_factory must be callable")
        self.config = config
        self._execution_service_provider = execution_service_provider
        self._execution_adapter_factory = execution_adapter_factory
        self._server_lock_factory = server_lock_factory
        self._job_manager_factory = job_manager_factory
        self._uses_default_job_manager = job_manager_factory is JobManager
        self._workspace_manager_factory = workspace_manager_factory
        self.state = APIRuntimeState()

    def _validate_production_containment(self, canonical_root: Path) -> None:
        config_path = self.config.production_runtime_config_path
        if config_path is None:
            return
        production_config = ProductionRuntimeConfig.from_json(config_path)
        validate_production_cache_containment(
            canonical_root,
            production_config.cache_root,
        )

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

        try:
            canonical_root = validate_web_runtime_root(self.config.web_runtime_root)
            if canonical_root != self.config.web_runtime_root:
                raise ValueError("web runtime root must remain canonical")
            self._validate_production_containment(canonical_root)
        except Exception:
            with self.state._guard:
                self.state.startup_failed = True
            raise APIRuntimeStartupError("API runtime startup failed") from None

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
            workspace = self._workspace_manager_factory(canonical_root)
            if not callable(getattr(workspace, "initialize", None)):
                raise TypeError("workspace manager must provide initialize")
            if not callable(getattr(workspace, "cleanup_orphans", None)):
                raise TypeError("workspace manager must provide cleanup_orphans")
            if not callable(getattr(workspace, "cleanup_job", None)):
                raise TypeError("workspace manager must provide cleanup_job")
            _require_positional_interface(workspace, "prepare_job_workspace", 1)
            _require_positional_interface(workspace, "job_input_path", 2)
            _require_positional_interface(workspace, "create_job_input", 2)
            if not callable(
                getattr(workspace, "cleanup_all_job_workspaces", None)
            ):
                raise TypeError(
                    "workspace manager must provide cleanup_all_job_workspaces"
                )
            with self.state._guard:
                self.state.workspace_manager = workspace
            workspace.initialize()
            workspace.cleanup_orphans()

            service = self._execution_service_provider()
            if not callable(getattr(service, "execute", None)):
                raise TypeError("execution service must provide execute")
            runtime = getattr(service, "runtime", None)
            if not callable(getattr(runtime, "start", None)):
                raise TypeError("execution service runtime must provide start")
            with self.state._guard:
                self.state.execution_service = service
            runtime.start()

            adapter = self._execution_adapter_factory(service)
            _require_positional_interface(
                adapter,
                "execute",
                3,
                owner_name="execution adapter",
            )
            with self.state._guard:
                self.state.execution_adapter = adapter

            if self._uses_default_job_manager:
                manager = self._job_manager_factory(
                    adapter,
                    on_terminal=workspace.cleanup_job,
                )
            else:
                manager = self._job_manager_factory(adapter)
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

        workspace = self.state.workspace_manager
        if workspace is not None:
            try:
                workspace.cleanup_all_job_workspaces()
            except Exception:
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
