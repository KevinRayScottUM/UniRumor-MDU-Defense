"""Public-safe web response types without duplicating scientific schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


API_VERSION = "v1"

PUBLIC_ERROR_MESSAGES = {
    "job_not_found": "Job not found.",
    "job_expired": "Job has expired.",
    "job_not_completed": "Job has not completed.",
    "job_failed": "Job execution failed.",
    "service_not_ready": "Service is not ready.",
    "internal_error": "An internal server error occurred.",
    "not_found": "Resource not found.",
    "method_not_allowed": "Method not allowed.",
    "malformed_request": "Request could not be processed.",
    "upload_too_large": "Uploaded video exceeds the configured size limit.",
    "unsupported_video_type": "Video type is not supported.",
    "invalid_claim": "Claim must contain between 1 and 2,000 characters.",
    "empty_upload": "Uploaded video must not be empty.",
    "invalid_filename": "Video filename is invalid.",
    "queue_full": "Job queue is full.",
    "visual_xai_not_found": "Visual attribution request not found.",
}


@dataclass(frozen=True)
class PublicError:
    code: str
    message: str
    request_id: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class ErrorEnvelope:
    error: PublicError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "error": self.error.to_dict(),
        }


@dataclass(frozen=True)
class ReadinessPayload:
    status: str
    accepting_jobs: bool
    capacity_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "status": self.status,
            "accepting_jobs": self.accepting_jobs,
            "capacity_state": self.capacity_state,
        }


def error_envelope(code: str, request_id: str) -> Dict[str, Any]:
    message = PUBLIC_ERROR_MESSAGES.get(code, PUBLIC_ERROR_MESSAGES["internal_error"])
    return ErrorEnvelope(
        PublicError(code=code, message=message, request_id=request_id)
    ).to_dict()


__all__ = [
    "API_VERSION",
    "ErrorEnvelope",
    "PUBLIC_ERROR_MESSAGES",
    "PublicError",
    "ReadinessPayload",
    "error_envelope",
]
