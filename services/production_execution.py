"""Public-safe execution outcomes for production runtime and packaging."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from services.production_result import ProductionResult, ProductionResultBuilder
from services.production_runtime import ProductionRuntime


SCHEMA_VERSION = 1
RUNTIME_FAILURE_PUBLIC_MESSAGE = "Production runtime execution failed."
RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE = (
    "Production result packaging failed."
)


class ProductionExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class OperationalFailureStage(str, Enum):
    RUNTIME = "runtime"
    RESULT_PACKAGING = "result_packaging"


class OperationalFailureCode(str, Enum):
    RUNTIME_EXECUTION_FAILED = "runtime_execution_failed"
    RESULT_PACKAGING_FAILED = "result_packaging_failed"


@dataclass(frozen=True)
class OperationalFailure:
    stage: OperationalFailureStage
    code: OperationalFailureCode
    exception_type: str
    public_message: str

    def __post_init__(self) -> None:
        expected = {
            OperationalFailureStage.RUNTIME: (
                OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                RUNTIME_FAILURE_PUBLIC_MESSAGE,
            ),
            OperationalFailureStage.RESULT_PACKAGING: (
                OperationalFailureCode.RESULT_PACKAGING_FAILED,
                RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE,
            ),
        }
        if not isinstance(self.stage, OperationalFailureStage):
            raise TypeError("stage must be an OperationalFailureStage")
        expected_code, expected_message = expected[self.stage]
        if self.code is not expected_code:
            raise ValueError("failure code must match failure stage")
        if self.public_message != expected_message:
            raise ValueError("public message must be the fixed message for its stage")
        if not isinstance(self.exception_type, str) or not self.exception_type:
            raise ValueError("exception_type must be a non-empty class name")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "code": self.code.value,
            "exception_type": self.exception_type,
            "public_message": self.public_message,
        }


@dataclass(frozen=True)
class ProductionExecutionOutcome:
    schema_version: int
    status: ProductionExecutionStatus
    result: Optional[ProductionResult]
    failure: Optional[OperationalFailure]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("ProductionExecutionOutcome schema_version must equal 1")
        if self.status is ProductionExecutionStatus.SUCCESS:
            if self.result is None or self.failure is not None:
                raise ValueError("success requires a result and forbids a failure")
        elif self.status is ProductionExecutionStatus.FAILURE:
            if self.result is not None or self.failure is None:
                raise ValueError("failure requires a failure and forbids a result")
        else:
            raise TypeError("status must be a ProductionExecutionStatus")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result is not None else None,
            "failure": (
                self.failure.to_dict() if self.failure is not None else None
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ProductionExecutionService:
    def __init__(
        self,
        runtime: ProductionRuntime,
        *,
        result_builder: Optional[ProductionResultBuilder] = None,
    ) -> None:
        if not callable(getattr(runtime, "run", None)):
            raise TypeError("runtime must provide a callable run method")
        if result_builder is None:
            runtime_config = getattr(runtime, "config", None)
            cache_root = getattr(runtime_config, "cache_root", None)
            result_builder = ProductionResultBuilder(
                evidence_root=cache_root if isinstance(cache_root, Path) else None
            )
        elif not callable(getattr(result_builder, "build", None)):
            raise TypeError("result_builder must provide a callable build method")
        self.runtime = runtime
        self.result_builder = result_builder

    @classmethod
    def from_json(cls, config_path: Path) -> "ProductionExecutionService":
        return cls(ProductionRuntime.from_json(config_path))

    @staticmethod
    def _failure(
        stage: OperationalFailureStage,
        exc: Exception,
    ) -> OperationalFailure:
        if stage is OperationalFailureStage.RUNTIME:
            return OperationalFailure(
                stage=stage,
                code=OperationalFailureCode.RUNTIME_EXECUTION_FAILED,
                exception_type=type(exc).__name__,
                public_message=RUNTIME_FAILURE_PUBLIC_MESSAGE,
            )
        return OperationalFailure(
            stage=stage,
            code=OperationalFailureCode.RESULT_PACKAGING_FAILED,
            exception_type=type(exc).__name__,
            public_message=RESULT_PACKAGING_FAILURE_PUBLIC_MESSAGE,
        )

    def execute(
        self,
        session_id: str,
        claim: str,
        video_path: Union[str, Path],
    ) -> ProductionExecutionOutcome:
        try:
            internal_result = self.runtime.run(session_id, claim, video_path)
        except Exception as exc:
            return ProductionExecutionOutcome(
                schema_version=SCHEMA_VERSION,
                status=ProductionExecutionStatus.FAILURE,
                result=None,
                failure=self._failure(OperationalFailureStage.RUNTIME, exc),
            )

        try:
            production_result = self.result_builder.build(internal_result)
        except Exception as exc:
            return ProductionExecutionOutcome(
                schema_version=SCHEMA_VERSION,
                status=ProductionExecutionStatus.FAILURE,
                result=None,
                failure=self._failure(
                    OperationalFailureStage.RESULT_PACKAGING,
                    exc,
                ),
            )

        return ProductionExecutionOutcome(
            schema_version=SCHEMA_VERSION,
            status=ProductionExecutionStatus.SUCCESS,
            result=production_result,
            failure=None,
        )
