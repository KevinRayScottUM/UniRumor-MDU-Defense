"""Configuration and state for one deterministic pipeline run."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from schemas import PipelineStage, RuntimeUnit, StageName, StageStatus, VerificationRequest


FROZEN_BACKBONE = "microsoft/deberta-v3-base"


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


@dataclass
class RuntimeConfig:
    cache_root: Path
    output_root: Path
    profile: str = "mock"
    implementation: str = "deterministic_mock"
    backbone: str = FROZEN_BACKBONE
    max_units: int = 24
    max_length: int = 256
    pooling: str = "max"
    fake_label: int = 0
    real_label: int = 1
    top_k: int = 5

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        frozen = (
            self.backbone == FROZEN_BACKBONE
            and self.max_units == 24
            and self.max_length == 256
            and self.pooling == "max"
            and self.fake_label == 0
            and self.real_label == 1
            and self.top_k == 5
        )
        if not frozen:
            raise ValueError("runtime config violates the immutable scientific contract")
        if self.cache_root == self.output_root:
            raise ValueError("cache_root and output_root must be distinct")

    @classmethod
    def from_yaml(cls, path: Path) -> "RuntimeConfig":
        path = Path(path).resolve()
        values: Dict[str, object] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition(":")
            if not separator:
                raise ValueError(f"unsupported config line: {raw_line}")
            values[key.strip()] = _parse_scalar(value)
        base_dir = path.parent.parent
        cache_root = Path(str(values.pop("cache_root")))
        output_root = Path(str(values.pop("output_root")))
        if not cache_root.is_absolute():
            cache_root = base_dir / cache_root
        if not output_root.is_absolute():
            output_root = base_dir / output_root
        return cls(cache_root=cache_root, output_root=output_root, **values)


STAGE_ORDER = list(StageName)


@dataclass
class PipelineContext:
    request: VerificationRequest
    config: RuntimeConfig
    session_id: Optional[str] = None
    units: List[RuntimeUnit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=lambda: ["MOCK_NON_SCIENTIFIC_OUTPUT"])
    stages: List[PipelineStage] = field(
        default_factory=lambda: [PipelineStage(name, index) for index, name in enumerate(STAGE_ORDER)]
    )
    _next_stage: int = 0

    def start_stage(self, name: StageName) -> PipelineStage:
        if self._next_stage >= len(self.stages) or self.stages[self._next_stage].name != name:
            expected = self.stages[self._next_stage].name.value if self._next_stage < len(self.stages) else "none"
            raise ValueError(f"stage order violation: expected {expected}, got {name.value}")
        stage = self.stages[self._next_stage]
        stage.transition(StageStatus.RUNNING)
        return stage

    def complete_stage(self, name: StageName, detail: str = "") -> None:
        stage = self.stages[self._next_stage]
        if stage.name != name:
            raise ValueError(f"cannot complete inactive stage {name.value}")
        stage.detail = detail
        stage.transition(StageStatus.COMPLETED)
        self._next_stage += 1
