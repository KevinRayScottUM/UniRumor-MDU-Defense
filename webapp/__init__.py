"""Task07 production web application foundations."""

from .api import create_app
from .api_config import APIConfig, WebAPIConfig
from .api_types import API_VERSION
from .execution_adapter import (
    ProductionExecutionAdapter,
    ProductionExecutionContract,
    ProductionExecutionRequest,
)

from .job_manager import (
    CapacityReservation,
    JobManager,
    JobManagerError,
    JobManagerNotAcceptingError,
    QueueFullError,
    ReservationError,
)
from .job_types import JobFailureSnapshot, JobSnapshot, JobState
from .server_lock import (
    ServerLock,
    ServerLockError,
    ServerLockUnavailableError,
)
from .workspace import (
    ALLOWED_INPUT_EXTENSIONS,
    JOB_ID_PATTERN,
    WebWorkspaceError,
    WebWorkspaceManager,
    WebWorkspaceSecurityError,
    validate_production_cache_containment,
)

__all__ = [
    "APIConfig",
    "API_VERSION",
    "ALLOWED_INPUT_EXTENSIONS",
    "CapacityReservation",
    "JOB_ID_PATTERN",
    "JobFailureSnapshot",
    "JobManager",
    "JobManagerError",
    "JobManagerNotAcceptingError",
    "JobSnapshot",
    "JobState",
    "ProductionExecutionAdapter",
    "ProductionExecutionContract",
    "ProductionExecutionRequest",
    "QueueFullError",
    "ReservationError",
    "ServerLock",
    "ServerLockError",
    "ServerLockUnavailableError",
    "WebAPIConfig",
    "WebWorkspaceError",
    "WebWorkspaceManager",
    "WebWorkspaceSecurityError",
    "create_app",
    "validate_production_cache_containment",
]
