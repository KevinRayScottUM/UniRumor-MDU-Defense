"""Containment-safe ownership of Task07 upload workspaces only."""

from __future__ import annotations

import errno
import os
import re
import stat
from pathlib import Path
from typing import BinaryIO, Union


JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{32}$")
ALLOWED_INPUT_EXTENSIONS = frozenset((".mp4", ".m4v", ".mov", ".webm"))


class WebWorkspaceError(RuntimeError):
    """Base class for internal Task07 workspace failures."""


class WebWorkspaceSecurityError(WebWorkspaceError):
    """Raised when a path cannot satisfy Task07 ownership constraints."""


def _canonical_existing_directory(value: Union[str, Path], field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a string or Path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ValueError(f"{field_name} must be an existing directory") from None
    if resolved != path or not path.is_dir():
        raise ValueError(f"{field_name} must be a canonical existing directory")
    return path


def validate_production_cache_containment(
    web_runtime_root: Union[str, Path],
    production_cache_root: Union[str, Path],
) -> None:
    """Require the canonical web root to be strictly inside the cache root."""

    web_root = _canonical_existing_directory(
        web_runtime_root,
        "web_runtime_root",
    )
    if not isinstance(production_cache_root, (str, Path)):
        raise TypeError("production_cache_root must be a string or Path")
    try:
        cache_root = Path(production_cache_root).expanduser().resolve(strict=True)
    except OSError:
        raise ValueError("production_cache_root must be an existing directory") from None
    if not cache_root.is_dir():
        raise ValueError("production_cache_root must be an existing directory")
    if web_root == cache_root:
        raise ValueError("web_runtime_root must be a strict cache-root descendant")
    try:
        relative = web_root.relative_to(cache_root)
    except ValueError:
        raise ValueError(
            "web_runtime_root must be a strict cache-root descendant"
        ) from None
    if not relative.parts:
        raise ValueError("web_runtime_root must be a strict cache-root descendant")


class WebWorkspaceManager:
    """Own only ``<WEB_RUNTIME_ROOT>/jobs`` and canonical job children."""

    JOBS_DIRECTORY_NAME = "jobs"

    def __init__(self, web_runtime_root: Union[str, Path]) -> None:
        self._runtime_root = _canonical_existing_directory(
            web_runtime_root,
            "web_runtime_root",
        )
        self._jobs_root = self._runtime_root / self.JOBS_DIRECTORY_NAME
        self._initialized = False

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    @property
    def jobs_root(self) -> Path:
        return self._jobs_root

    @property
    def initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )

    def _validate_runtime_root(self) -> None:
        current = _canonical_existing_directory(
            self._runtime_root,
            "web_runtime_root",
        )
        if current != self._runtime_root:
            raise WebWorkspaceSecurityError("web runtime root identity changed")

    def initialize(self) -> None:
        """Create and secure the managed jobs root after singleton acquisition."""

        self._validate_runtime_root()
        try:
            existing = os.lstat(self._jobs_root)
        except FileNotFoundError:
            try:
                os.mkdir(self._jobs_root, 0o700)
            except FileExistsError:
                existing = os.lstat(self._jobs_root)
            else:
                existing = os.lstat(self._jobs_root)

        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
            raise WebWorkspaceSecurityError(
                "managed jobs root must be a non-symlink directory"
            )

        try:
            jobs_fd = os.open(self._jobs_root, self._directory_flags())
        except OSError:
            raise WebWorkspaceSecurityError(
                "managed jobs root could not be opened safely"
            ) from None
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(jobs_fd, 0o700)
        finally:
            os.close(jobs_fd)
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise WebWorkspaceError("workspace manager has not been initialized")

    def _open_jobs_root(self) -> int:
        self._require_initialized()
        self._validate_runtime_root()
        try:
            current = os.lstat(self._jobs_root)
        except FileNotFoundError:
            raise WebWorkspaceSecurityError("managed jobs root is missing") from None
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise WebWorkspaceSecurityError(
                "managed jobs root must be a non-symlink directory"
            )
        try:
            return os.open(self._jobs_root, self._directory_flags())
        except OSError:
            raise WebWorkspaceSecurityError(
                "managed jobs root could not be opened safely"
            ) from None

    @staticmethod
    def _validate_job_id(job_id: str) -> str:
        if not isinstance(job_id, str):
            raise TypeError("job_id must be a string")
        if JOB_ID_PATTERN.fullmatch(job_id) is None:
            raise ValueError("job_id must use the canonical Task07 format")
        return job_id

    def _job_path(self, job_id: str) -> Path:
        canonical_id = self._validate_job_id(job_id)
        candidate = self._jobs_root / canonical_id
        try:
            relative = candidate.relative_to(self._jobs_root)
        except ValueError:
            raise WebWorkspaceSecurityError("job workspace escaped jobs root") from None
        if relative != Path(canonical_id) or candidate.parent != self._jobs_root:
            raise WebWorkspaceSecurityError("job workspace escaped jobs root")
        return candidate

    @classmethod
    def _remove_entry_at(cls, parent_fd: int, name: str) -> None:
        """Delete one dirfd-relative entry without following any symlink."""

        try:
            entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return

        if not stat.S_ISDIR(entry_stat.st_mode):
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            return

        try:
            child_fd = os.open(name, cls._directory_flags(), dir_fd=parent_fd)
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno not in (errno.ELOOP, errno.ENOTDIR):
                raise
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            return

        try:
            with os.scandir(child_fd) as entries:
                child_names = sorted(entry.name for entry in entries)
            for child_name in child_names:
                cls._remove_entry_at(child_fd, child_name)
        finally:
            os.close(child_fd)

        try:
            os.rmdir(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass

    def prepare_job_workspace(self, job_id: str) -> Path:
        """Exclusively create one owner-only canonical job directory."""

        job_path = self._job_path(job_id)
        jobs_fd = self._open_jobs_root()
        try:
            try:
                os.mkdir(job_id, 0o700, dir_fd=jobs_fd)
            except FileExistsError:
                raise WebWorkspaceSecurityError(
                    "job workspace already exists"
                ) from None
            try:
                job_fd = os.open(job_id, self._directory_flags(), dir_fd=jobs_fd)
            except OSError:
                self._remove_entry_at(jobs_fd, job_id)
                raise WebWorkspaceSecurityError(
                    "job workspace could not be opened safely"
                ) from None
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(job_fd, 0o700)
            finally:
                os.close(job_fd)
        finally:
            os.close(jobs_fd)
        return job_path

    def job_input_path(self, job_id: str, validated_extension: str) -> Path:
        """Return only the fixed internal input path for an existing workspace."""

        job_path = self._job_path(job_id)
        if not isinstance(validated_extension, str):
            raise TypeError("validated_extension must be a string")
        if validated_extension not in ALLOWED_INPUT_EXTENSIONS:
            raise ValueError("validated_extension is not allowed")

        jobs_fd = self._open_jobs_root()
        try:
            try:
                job_fd = os.open(job_id, self._directory_flags(), dir_fd=jobs_fd)
            except OSError:
                raise WebWorkspaceSecurityError(
                    "job workspace must be a non-symlink directory"
                ) from None
            try:
                basename = "input" + validated_extension
                try:
                    input_stat = os.stat(
                        basename,
                        dir_fd=job_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    input_stat = None
                if input_stat is not None and stat.S_ISLNK(input_stat.st_mode):
                    raise WebWorkspaceSecurityError(
                        "job input path must not be a symlink"
                    )
            finally:
                os.close(job_fd)
        finally:
            os.close(jobs_fd)

        candidate = job_path / basename
        if candidate.parent != job_path:
            raise WebWorkspaceSecurityError("job input path escaped workspace")
        return candidate

    def create_job_input(
        self,
        job_id: str,
        validated_extension: str,
    ) -> BinaryIO:
        """Exclusively open one owner-only fixed input file without symlinks."""

        self._job_path(job_id)
        if not isinstance(validated_extension, str):
            raise TypeError("validated_extension must be a string")
        if validated_extension not in ALLOWED_INPUT_EXTENSIONS:
            raise ValueError("validated_extension is not allowed")

        jobs_fd = self._open_jobs_root()
        input_fd = None
        try:
            try:
                job_fd = os.open(job_id, self._directory_flags(), dir_fd=jobs_fd)
            except OSError:
                raise WebWorkspaceSecurityError(
                    "job workspace must be a non-symlink directory"
                ) from None
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    input_fd = os.open(
                        "input" + validated_extension,
                        flags,
                        0o600,
                        dir_fd=job_fd,
                    )
                except OSError:
                    raise WebWorkspaceSecurityError(
                        "job input file could not be created safely"
                    ) from None
                if hasattr(os, "fchmod"):
                    os.fchmod(input_fd, 0o600)
            except Exception:
                if input_fd is not None:
                    os.close(input_fd)
                    input_fd = None
                raise
            finally:
                os.close(job_fd)
        finally:
            os.close(jobs_fd)

        try:
            return os.fdopen(input_fd, "wb")
        except Exception:
            os.close(input_fd)
            raise

    def cleanup_job(self, job_id: str) -> None:
        """Idempotently delete exactly one canonical Task07-owned workspace."""

        self._job_path(job_id)
        jobs_fd = self._open_jobs_root()
        try:
            self._remove_entry_at(jobs_fd, job_id)
        finally:
            os.close(jobs_fd)

    def _cleanup_owned_entries(self) -> int:
        jobs_fd = self._open_jobs_root()
        removed = 0
        try:
            with os.scandir(jobs_fd) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                if JOB_ID_PATTERN.fullmatch(name) is None:
                    continue
                self._job_path(name)
                self._remove_entry_at(jobs_fd, name)
                removed += 1
        finally:
            os.close(jobs_fd)
        return removed

    def cleanup_orphans(self) -> int:
        """Remove only recognized direct Task07 entries from a previous process."""

        return self._cleanup_owned_entries()

    def cleanup_all_job_workspaces(self) -> int:
        """Remove all remaining recognized Task07 workspaces during shutdown."""

        return self._cleanup_owned_entries()


__all__ = [
    "ALLOWED_INPUT_EXTENSIONS",
    "JOB_ID_PATTERN",
    "WebWorkspaceError",
    "WebWorkspaceManager",
    "WebWorkspaceSecurityError",
    "validate_production_cache_containment",
]
