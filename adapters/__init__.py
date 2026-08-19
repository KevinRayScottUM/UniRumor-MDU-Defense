"""Adapters for external runtime contracts."""

from .phase4a_request_adapter import build_phase4a_request
from .phase4a_response_adapter import parse_phase4a_prediction

__all__ = ["build_phase4a_request", "parse_phase4a_prediction"]
