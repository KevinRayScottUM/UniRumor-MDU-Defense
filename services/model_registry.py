"""Identity-only registry; never resolves or inspects model assets."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelAsset:
    logical_name: str
    identity: Optional[str]
    implementation: str
    checkpoint_access: str = "forbidden"


class ModelRegistry:
    def __init__(self):
        self._assets: Dict[str, ModelAsset] = {
            "g1": ModelAsset("g1", "microsoft/deberta-v3-base", "deterministic_mock"),
            "asr": ModelAsset("asr", None, "deterministic_mock"),
            "ocr": ModelAsset("ocr", None, "deterministic_mock"),
            "visual_retrieval": ModelAsset("visual_retrieval", None, "deterministic_mock"),
            "vlm": ModelAsset("vlm", None, "deterministic_mock"),
        }

    def get(self, logical_name: str) -> ModelAsset:
        return self._assets[logical_name]

    def describe(self) -> Dict[str, Dict[str, Optional[str]]]:
        return {
            key: {
                "identity": asset.identity,
                "implementation": asset.implementation,
                "checkpoint_access": asset.checkpoint_access,
            }
            for key, asset in self._assets.items()
        }
