"""Constrained JSON writes beneath a configured cache root."""

import json
import re
from pathlib import Path
from typing import Any, Dict


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def safe_target(root: Path, category: str, filename: str) -> Path:
    root = Path(root).resolve()
    if not SAFE_COMPONENT.fullmatch(category) or not SAFE_COMPONENT.fullmatch(filename):
        raise ValueError("unsafe runtime path component")
    target = (root / category / filename).resolve()
    if root not in target.parents:
        raise ValueError("runtime path escapes configured root")
    return target


def write_json(target: Path, payload: Dict[str, Any]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(target)
    return target


class CacheManager:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def put_json(self, category: str, key: str, payload: Dict[str, Any]) -> Path:
        return write_json(safe_target(self.root, category, f"{key}.json"), payload)
