"""Structured JSONL logging beneath the configured output root."""

import json
from pathlib import Path
from typing import Any, Dict

from .cache_manager import safe_target


class RuntimeLogger:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root).resolve()

    def log(self, session_id: str, event: str, payload: Dict[str, Any]) -> Path:
        target = safe_target(self.output_root, "logs", f"{session_id}.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "payload": payload, "session_id": session_id}
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return target
