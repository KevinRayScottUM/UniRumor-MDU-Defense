"""Lifecycle and delegation wrapper for the real production service graph."""

from pathlib import Path
from typing import Optional, Union

from schemas import ProductionRuntimeConfig
from services.production_runtime_factory import (
    ProductionRuntimeFactory,
    ProductionRuntimeServices,
)
from services.session_manager import SAFE_SESSION
from services.video_multimodal_runner import VideoMultimodalResult


class ProductionRuntime:
    def __init__(
        self,
        config: ProductionRuntimeConfig,
        *,
        factory: Optional[ProductionRuntimeFactory] = None,
    ) -> None:
        if not isinstance(config, ProductionRuntimeConfig):
            raise TypeError("config must be a ProductionRuntimeConfig")
        if factory is None:
            factory = ProductionRuntimeFactory(config)
        elif getattr(factory, "config", None) is not config:
            raise ValueError(
                "factory.config must be the exact ProductionRuntimeConfig object"
            )
        self.config = config
        self.factory = factory
        self._services: Optional[ProductionRuntimeServices] = None

    @classmethod
    def from_json(cls, path: Path) -> "ProductionRuntime":
        return cls(ProductionRuntimeConfig.from_json(path))

    @property
    def started(self) -> bool:
        return self._services is not None

    @property
    def services(self) -> ProductionRuntimeServices:
        if self._services is None:
            raise RuntimeError("production runtime has not been started")
        return self._services

    def start(self) -> ProductionRuntimeServices:
        if self._services is None:
            services = self.factory.build(run_preflight=True)
            self._services = services
        return self._services

    def run(
        self,
        session_id: str,
        claim: str,
        video_path: Union[str, Path],
    ) -> VideoMultimodalResult:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        if SAFE_SESSION.fullmatch(session_id) is None:
            raise ValueError(
                "session_id must satisfy the repository SAFE_SESSION contract"
            )
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        if not claim.strip():
            raise ValueError("claim must contain non-whitespace text")
        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a string or Path")

        resolved_video_path = Path(video_path).expanduser().resolve()
        if not resolved_video_path.exists():
            raise FileNotFoundError(
                f"video_path does not exist: {resolved_video_path}"
            )
        if not resolved_video_path.is_file():
            raise ValueError(
                f"video_path must be a regular file: {resolved_video_path}"
            )

        services = self.start()
        return services.video_multimodal_runner.run(
            session_id,
            claim,
            resolved_video_path,
        )
