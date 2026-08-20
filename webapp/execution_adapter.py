"""Injected web boundary for the closed Task06 production execution contract."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from services.production_execution import (
    RUNTIME_FAILURE_PUBLIC_MESSAGE,
    SCHEMA_VERSION,
    OperationalFailure,
    OperationalFailureCode,
    OperationalFailureStage,
    ProductionExecutionOutcome,
    ProductionExecutionStatus,
)


ADAPTER_FAILURE_EXCEPTION_TYPE = "ProductionExecutionAdapterFailure"


class ProductionExecutionContract(Protocol):
    """Narrow injected Task06 interface consumed by the web adapter."""

    def execute(
        self,
        session_id: str,
        claim: str,
        video_path: Path,
    ) -> ProductionExecutionOutcome:
        ...


@dataclass(frozen=True)
class ProductionExecutionRequest:
    """Validated server-owned values forwarded without scientific changes."""

    session_id: str
    claim: str
    video_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.claim, str):
            raise TypeError("claim must be a string")
        if not isinstance(self.video_path, Path):
            raise TypeError("video_path must be a Path")


class ProductionExecutionAdapter:
    """Invoke one injected execution contract and normalize operational failure."""

    def __init__(self, execution_contract: ProductionExecutionContract) -> None:
        execute = getattr(execution_contract, "execute", None)
        if not callable(execute):
            raise TypeError("execution_contract must provide execute")
        try:
            inspect.signature(execute).bind(
                "session",
                "claim",
                Path("input.mp4"),
            )
        except (TypeError, ValueError):
            raise TypeError(
                "execution_contract execute signature is incompatible"
            ) from None
        self._execution_contract = execution_contract

    @property
    def execution_contract(self) -> ProductionExecutionContract:
        return self._execution_contract

    @staticmethod
    def _safe_failure() -> ProductionExecutionOutcome:
        return ProductionExecutionOutcome(
            schema_version=SCHEMA_VERSION,
            status=ProductionExecutionStatus.FAILURE,
            result=None,
            failure=OperationalFailure(
                stage=OperationalFailureStage.RUNTIME,
                code=OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                exception_type=ADAPTER_FAILURE_EXCEPTION_TYPE,
                public_message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
            ),
        )

    def execute_request(
        self,
        request: ProductionExecutionRequest,
    ) -> ProductionExecutionOutcome:
        if not isinstance(request, ProductionExecutionRequest):
            raise TypeError("request must be a ProductionExecutionRequest")
        try:
            outcome = self._execution_contract.execute(
                request.session_id,
                request.claim,
                request.video_path,
            )
        except Exception:
            return self._safe_failure()
        if not isinstance(outcome, ProductionExecutionOutcome):
            return self._safe_failure()
        return outcome

    def execute(
        self,
        session_id: str,
        claim: str,
        video_path: Path,
    ) -> ProductionExecutionOutcome:
        return self.execute_request(
            ProductionExecutionRequest(
                session_id=session_id,
                claim=claim,
                video_path=video_path,
            )
        )


__all__ = [
    "ADAPTER_FAILURE_EXCEPTION_TYPE",
    "ProductionExecutionAdapter",
    "ProductionExecutionContract",
    "ProductionExecutionRequest",
]
