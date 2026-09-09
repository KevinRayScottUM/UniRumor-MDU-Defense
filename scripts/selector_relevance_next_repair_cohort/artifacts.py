"""Strict artifact reads, immutable locks and atomic exclusive publication."""

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
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


def rename_exclusive(source, destination):
    """Use OS no-replace rename, including the concurrent empty-directory case."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        args = (os.fsencode(source), os.fsencode(destination), 0x00000004)
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_uint]
    else:
        raise CohortError("atomic no-replace rename unavailable; no unsafe fallback")
    function.restype = ctypes.c_int
    if function(*args):
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CohortError("output already exists; exclusive publication refused")
        raise OSError(error, "exclusive directory rename failed")


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
