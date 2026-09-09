"""Strict artifact reads, immutable locks and atomic exclusive publication."""

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .schemas import CohortError, FORBIDDEN_ARTIFACTS


def safe_path(value):
    raw = Path(value).expanduser().absolute()
    resolved = raw.resolve()
    for path in (raw, resolved):
        for part in path.parts:
            normalized = re.sub(r"[^a-z0-9]", "", part.casefold())
            tokens = set(re.split(r"[^a-z0-9]+", part.casefold()))
            if (normalized in {"validation", "test", "formalvalidation", "formaltest"}
                    or tokens & {"validation", "test"}):
                raise CohortError("Formal Validation/Test path forbidden")
            if part.casefold() in FORBIDDEN_ARTIFACTS:
                raise CohortError("old ranking artifact forbidden")
    if raw != resolved:
        # Locks must not silently change meaning when a symlink is retargeted.
        raise CohortError("symlink or noncanonical input/output path forbidden")
    return resolved


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    path = safe_path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CohortError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    value = json.loads(safe_path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=unique_object,
                       parse_constant=lambda x: (_ for _ in ()).throw(CohortError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise CohortError("artifact must contain a JSON object")
    return value


def sidecar(path):
    return Path(path).with_suffix(".sha256")


class Ledger:
    def __init__(self):
        self.records = {}

    def add(self, path, role, expected=None, with_sidecar=False):
        path = safe_path(path)
        actual = sha_file(path)
        if expected is not None and actual != expected:
            raise CohortError(f"{role} SHA-256 mismatch")
        record = {"path": str(path), "sha256": actual, "role": role}
        prior = self.records.setdefault(str(path), record)
        if prior["sha256"] != actual:
            raise CohortError("immutable input changed during reading")
        if with_sidecar:
            sha_path = sidecar(path)
            declared = safe_path(sha_path).read_text(encoding="ascii").strip().split()
            if not declared or declared[0] != actual:
                raise CohortError(f"{role} SHA sidecar mismatch")
            if len(declared) > 1 and declared[1].lstrip("*") != path.name:
                raise CohortError("SHA sidecar filename mismatch")
            if len(declared) > 2:
                raise CohortError("malformed SHA sidecar")
            self.add(sha_path, role + " sidecar")
        return path

    def payload(self):
        return {"artifacts": sorted(self.records.values(), key=lambda r: (r["role"], r["path"]))}

    def revalidate(self):
        for record in self.records.values():
            if sha_file(record["path"]) != record["sha256"]:
                raise CohortError("immutable input changed before final freeze")


def assert_new_output(path):
    output = safe_path(path)
    if output.exists() or output.is_symlink():
        raise CohortError("output already exists; no overwrite or partial-output reuse")
    if not any(part in {"outputs", "cache"} for part in output.parts):
        raise CohortError("output must be under a configured outputs/ or cache/ root")
    return output


UNSUPPORTED_NATIVE_ERRNOS = frozenset({
    errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP,
})


def _native_rename_exclusive(source, destination):
    """Prefer the native primitive; symbol availability is not FS support."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        args = (os.fsencode(source), os.fsencode(destination), 0x00000004)
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_uint]
    else:
        raise OSError(errno.ENOSYS, "native no-replace rename unavailable")
    function.restype = ctypes.c_int
    if function(*args):
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CohortError("output already exists; exclusive publication refused")
        raise OSError(error, "exclusive directory rename failed")


@contextmanager
def _publication_lock(destination):
    """Never wait for, steal or clean another publisher's (possibly stale) lock."""
    lock_path = destination.with_name("." + destination.name + ".publish.lock")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise CohortError("publication lock already exists; operator inspection required") from exc
    owner = None
    try:
        owner = os.fstat(descriptor)
        yield
    finally:
        try:
            # Keep the descriptor open until after this comparison/unlink so
            # its inode cannot be recycled. A replaced/foreign lock is left
            # untouched. Cooperating publishers never replace existing locks.
            if owner is not None:
                try:
                    current = lock_path.lstat()
                except FileNotFoundError:
                    current = None
                if current is not None and (current.st_dev, current.st_ino) == (owner.st_dev, owner.st_ino):
                    lock_path.unlink()
        finally:
            os.close(descriptor)


def _portable_rename_exclusive(source, destination, parent_descriptor):
    """Caller MUST hold _publication_lock for this exact destination."""
    # Recheck after the native attempt, while every official publisher is
    # excluded. In particular, an existing empty directory must be preserved.
    if destination.exists() or destination.is_symlink():
        raise CohortError("output already exists; exclusive publication refused")
    source_stat = source.lstat()
    if not stat.S_ISDIR(source_stat.st_mode):
        raise CohortError("publication staging source must be a real directory")
    if source_stat.st_dev != os.fstat(parent_descriptor).st_dev:
        raise OSError(errno.EXDEV, "publication requires a same-filesystem directory rename")
    os.rename(source, destination)


def _fsync_directory(descriptor):
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in UNSUPPORTED_NATIVE_ERRNOS:
            raise


def rename_exclusive(source, destination):
    """Atomic publication for cooperating publishers, with native fast path.

    Both paths take the same lock: a native publisher must not bypass a
    fallback publisher's existence check, or a stale/foreign publication lock.
    Only a capability error permits normal rename under the already-held lock.
    """
    source, destination = Path(source), Path(destination)
    with _publication_lock(destination):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(destination.parent, flags)
        try:
            try:
                _native_rename_exclusive(source, destination)
            except OSError as exc:
                if exc.errno not in UNSUPPORTED_NATIVE_ERRNOS:
                    raise
                _portable_rename_exclusive(source, destination, parent_descriptor)
            # A real sync error is propagated, preserving the complete output
            # already renamed into place; never attempt destructive rollback.
            _fsync_directory(parent_descriptor)
        finally:
            os.close(parent_descriptor)


def freeze(output, artifacts, ledger):
    output = assert_new_output(output)
    for record in ledger.records.values():
        source = Path(record["path"])
        if output == source.parent or output in source.parents or source.parent in output.parents:
            raise CohortError("output must be separate from immutable input directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cohort-freeze-", dir=output.parent))
    try:
        staging.chmod(0o700)
        for name, data in artifacts.items():
            if Path(name).name != name or name.endswith(".sha256"):
                raise CohortError("invalid output artifact name")
            target = staging / name
            with target.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(0o600)
            sha_path = sidecar(target)
            with sha_path.open("xb") as stream:
                stream.write((digest_bytes(data) + "\n").encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            sha_path.chmod(0o600)
        ledger.revalidate()
        rename_exclusive(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
