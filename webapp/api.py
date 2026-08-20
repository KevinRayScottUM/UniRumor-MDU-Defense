"""FastAPI application factory for the closed production execution boundary."""

from __future__ import annotations

import secrets
from typing import Callable, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from services.production_execution import ProductionExecutionService

from .api_config import APIConfig
from .api_types import API_VERSION, error_envelope
from .execution_adapter import ProductionExecutionAdapter
from .job_manager import (
    JobManager,
    JobManagerNotAcceptingError,
    QueueFullError,
    ReservationError,
)
from .job_types import JobSnapshot, JobState
from .runtime_lifecycle import APIRuntimeLifecycle
from .server_lock import ServerLock
from .submission import (
    SubmissionValidationError,
    receive_submission,
    validate_submission_headers,
)
from .workspace import JOB_ID_PATTERN, WebWorkspaceManager


REQUEST_ID_HEADER = "X-Request-ID"


def _new_request_id() -> str:
    return "req_" + secrets.token_hex(16)


class _RequestBoundaryMiddleware:
    """Generate correlation IDs and redact ordinary unhandled failures."""

    def __init__(self, app, *, allowed_origins=()) -> None:
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        response_started = False
        request_headers = Headers(scope=scope)
        request_origin = request_headers.get("origin")

        async def send_with_headers(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                if scope.get("path", "").startswith("/api/"):
                    headers["Cache-Control"] = "no-store"
            await send(message)

        is_preflight = (
            scope.get("method") == "OPTIONS"
            and request_origin is not None
            and request_headers.get("access-control-request-method") is not None
        )
        if is_preflight:
            requested_method = request_headers[
                "access-control-request-method"
            ].upper()
            requested_headers = {
                value.strip().lower()
                for value in request_headers.get(
                    "access-control-request-headers", ""
                ).split(",")
                if value.strip()
            }
            if (
                request_origin not in self.allowed_origins
                or requested_method not in {"GET", "POST", "OPTIONS"}
                or not requested_headers.issubset({"content-type", "accept"})
            ):
                response = JSONResponse(
                    error_envelope("malformed_request", request_id),
                    status_code=400,
                )
                await response(scope, receive, send_with_headers)
                return

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            if response_started:
                raise
            headers = None
            if request_origin in self.allowed_origins:
                headers = {
                    "Access-Control-Allow-Origin": request_origin,
                    "Access-Control-Expose-Headers": REQUEST_ID_HEADER,
                    "Vary": "Origin",
                }
            response = JSONResponse(
                error_envelope("internal_error", request_id),
                status_code=500,
                headers=headers,
            )
            await response(scope, receive, send_with_headers)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else _new_request_id()


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    *,
    headers: Optional[dict] = None,
) -> JSONResponse:
    return JSONResponse(
        error_envelope(code, _request_id(request)),
        status_code=status_code,
        headers=headers,
    )


def _snapshot_lookup(
    request: Request,
    lifecycle: APIRuntimeLifecycle,
    job_id: str,
) -> Tuple[Optional[object], Optional[JobSnapshot], Optional[JSONResponse]]:
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        return None, None, _error_response(request, 404, "job_not_found")
    manager = lifecycle.state.manager
    if manager is None:
        return None, None, _error_response(request, 503, "service_not_ready")
    manager.sweep_expired()
    snapshot = manager.get_snapshot(job_id)
    if snapshot is None:
        return manager, None, _error_response(request, 404, "job_not_found")
    if snapshot.state is JobState.EXPIRED:
        return manager, snapshot, _error_response(request, 410, "job_expired")
    return manager, snapshot, None


def _job_payload(snapshot: JobSnapshot, poll_after_ms: int) -> dict:
    payload = snapshot.to_dict()
    payload["links"] = {
        "self": f"/api/v1/jobs/{snapshot.job_id}",
        "result": f"/api/v1/jobs/{snapshot.job_id}/result",
    }
    payload["poll_after_ms"] = (
        None
        if snapshot.state in (JobState.COMPLETED, JobState.FAILED)
        else poll_after_ms
    )
    return {"api_version": API_VERSION, "job": payload}


def create_app(
    config: APIConfig,
    *,
    execution_service=None,
    execution_service_factory: Optional[Callable[[], object]] = None,
    execution_adapter_factory: Callable = ProductionExecutionAdapter,
    server_lock_factory: Callable = ServerLock,
    job_manager_factory: Callable = JobManager,
    workspace_manager_factory: Callable = WebWorkspaceManager,
) -> FastAPI:
    """Construct the HTTP surface without acquiring or starting dependencies."""

    if not isinstance(config, APIConfig):
        raise TypeError("config must be an APIConfig")
    if execution_service is not None and execution_service_factory is not None:
        raise ValueError(
            "provide execution_service or execution_service_factory, not both"
        )
    if execution_service is not None:
        execution_service_provider = lambda: execution_service
    elif execution_service_factory is not None:
        execution_service_provider = execution_service_factory
    else:
        runtime_config_path = config.production_runtime_config_path
        if runtime_config_path is None:
            raise ValueError(
                "production_runtime_config_path or an execution service is required"
            )

        def execution_service_provider():
            return ProductionExecutionService.from_json(runtime_config_path)

    lifecycle = APIRuntimeLifecycle(
        config,
        execution_service_provider,
        execution_adapter_factory=execution_adapter_factory,
        server_lock_factory=server_lock_factory,
        job_manager_factory=job_manager_factory,
        workspace_manager_factory=workspace_manager_factory,
    )
    app = FastAPI(
        title="UniRumor MDU Defense API",
        version=API_VERSION,
        lifespan=lifecycle.lifespan,
    )
    app.state.api_lifecycle = lifecycle
    app.state.api_runtime_state = lifecycle.state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.add_middleware(
        _RequestBoundaryMiddleware,
        allowed_origins=config.allowed_origins,
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        if error.status_code == 404:
            return _error_response(request, 404, "not_found")
        if error.status_code == 405:
            return _error_response(request, 405, "method_not_allowed")
        return _error_response(request, error.status_code, "malformed_request")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return _error_response(request, 422, "malformed_request")

    @app.get("/api/v1/health")
    async def health():
        return JSONResponse({"api_version": API_VERSION, "status": "ok"})

    @app.get("/api/v1/readiness")
    async def readiness():
        payload = lifecycle.state.readiness()
        status_code = 200 if payload.accepting_jobs else 503
        headers = None
        if status_code == 503:
            headers = {"Retry-After": str(config.retry_after_seconds)}
        return JSONResponse(
            payload.to_dict(),
            status_code=status_code,
            headers=headers,
        )

    @app.post("/api/v1/jobs")
    async def submit_job(request: Request):
        try:
            submission_headers = validate_submission_headers(
                request,
                config.max_upload_bytes,
            )
        except SubmissionValidationError as error:
            return _error_response(request, error.status_code, error.code)

        manager = lifecycle.state.manager
        workspace = lifecycle.state.workspace_manager
        if manager is None or workspace is None:
            return _error_response(
                request,
                503,
                "service_not_ready",
                headers={"Retry-After": str(config.retry_after_seconds)},
            )
        try:
            reservation = manager.reserve_capacity()
        except QueueFullError:
            return _error_response(
                request,
                429,
                "queue_full",
                headers={"Retry-After": str(config.retry_after_seconds)},
            )
        except JobManagerNotAcceptingError:
            return _error_response(
                request,
                503,
                "service_not_ready",
                headers={"Retry-After": str(config.retry_after_seconds)},
            )

        submitted = False
        try:
            workspace.prepare_job_workspace(reservation.job_id)
            try:
                parsed = await receive_submission(
                    request,
                    submission_headers,
                    workspace,
                    reservation.job_id,
                    config.max_upload_bytes,
                )
            except SubmissionValidationError as error:
                return _error_response(request, error.status_code, error.code)

            try:
                job_id = manager.submit_reserved(
                    reservation,
                    claim=parsed.claim,
                    video_path=parsed.video_path,
                )
            except (JobManagerNotAcceptingError, ReservationError):
                return _error_response(
                    request,
                    503,
                    "service_not_ready",
                    headers={"Retry-After": str(config.retry_after_seconds)},
                )
            submitted = True
            snapshot = manager.get_snapshot(job_id)
            if snapshot is None:
                return _error_response(request, 500, "internal_error")
            request_id = _request_id(request)
            return JSONResponse(
                {
                    "api_version": API_VERSION,
                    "job_id": job_id,
                    "state": snapshot.state.value,
                    "request_id": request_id,
                },
                status_code=202,
                headers={"Location": f"/api/v1/jobs/{job_id}"},
            )
        finally:
            if not submitted:
                reservation.release()
                try:
                    workspace.cleanup_job(reservation.job_id)
                except Exception:
                    pass

    @app.get("/api/v1/jobs/{job_id}")
    async def job_status(request: Request, job_id: str):
        _, snapshot, error = _snapshot_lookup(request, lifecycle, job_id)
        if error is not None:
            return error
        return JSONResponse(_job_payload(snapshot, config.poll_after_ms))

    @app.get("/api/v1/jobs/{job_id}/result")
    async def job_result(request: Request, job_id: str):
        manager, snapshot, error = _snapshot_lookup(request, lifecycle, job_id)
        if error is not None:
            return error
        if snapshot.state in (
            JobState.ACCEPTED,
            JobState.QUEUED,
            JobState.RUNNING,
        ):
            return _error_response(request, 409, "job_not_completed")
        if snapshot.state is JobState.FAILED:
            return _error_response(request, 409, "job_failed")

        outcome = manager._get_completed_outcome(job_id)
        if outcome is not None:
            return JSONResponse(
                {
                    "api_version": API_VERSION,
                    "job_id": job_id,
                    "outcome": outcome.to_dict(),
                }
            )

        current = manager.get_snapshot(job_id)
        if current is None:
            return _error_response(request, 404, "job_not_found")
        if current.state is JobState.EXPIRED:
            return _error_response(request, 410, "job_expired")
        if current.state is JobState.FAILED:
            return _error_response(request, 409, "job_failed")
        if current.state in (
            JobState.ACCEPTED,
            JobState.QUEUED,
            JobState.RUNNING,
        ):
            return _error_response(request, 409, "job_not_completed")
        return _error_response(request, 500, "internal_error")

    return app


__all__ = [
    "JOB_ID_PATTERN",
    "REQUEST_ID_HEADER",
    "create_app",
]
