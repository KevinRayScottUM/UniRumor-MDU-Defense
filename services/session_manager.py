"""Deterministic session creation."""

import hashlib
import json
import re
from typing import Optional

from schemas import VerificationRequest

from .cache_manager import CacheManager


SAFE_SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class SessionManager:
    def __init__(self, cache: CacheManager):
        self.cache = cache

    def create(self, request: VerificationRequest) -> str:
        requested: Optional[str] = request.request_id
        if requested and SAFE_SESSION.fullmatch(requested):
            session_id = requested
        else:
            canonical = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            session_id = "session-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        self.cache.put_json("sessions", session_id, {"session_id": session_id, "request": request.to_dict()})
        return session_id
