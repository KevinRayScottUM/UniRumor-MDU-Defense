"""POSIX process-lifetime singleton lock for one web runtime root."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from typing import Optional, Union


class ServerLockError(RuntimeError):
    """Base class for stable internal singleton-lock failures."""


class ServerLockUnavailableError(ServerLockError):
    """Raised when another process owns the runtime singleton lock."""


class ServerLock:
    """Hold one exclusive advisory lock for the complete acquired lifetime."""

    LOCK_FILENAME = ".server.lock"

    def __init__(self, runtime_root: Union[str, Path]) -> None:
        if not isinstance(runtime_root, (str, Path)):
            raise TypeError("runtime_root must be a string or Path")
        root = Path(runtime_root)
        if not root.exists() or not root.is_dir():
            raise ValueError("runtime_root must be an existing directory")
        self._runtime_root = root
        self._lock_path = root / self.LOCK_FILENAME
        self._fd: Optional[int] = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    @property
    def lock_path(self) -> Path:
        """Internal lifecycle path; callers must not publish it."""

        return self._lock_path

    def acquire(self) -> "ServerLock":
        if self._fd is not None:
            return self

        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = None
        try:
            fd = os.open(self._lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
        except OSError:
            if fd is not None:
                os.close(fd)
            raise ServerLockError("server singleton lock could not be opened") from None

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise ServerLockUnavailableError(
                    "server singleton lock is unavailable"
                ) from None
            raise ServerLockError(
                "server singleton lock could not be acquired"
            ) from None

        self._fd = fd
        return self

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        failed = False
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            failed = True
        try:
            os.close(fd)
        except OSError:
            failed = True
        if failed:
            raise ServerLockError("server singleton lock could not be released") from None

    def __enter__(self) -> "ServerLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.release()
