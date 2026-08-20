import inspect
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from webapp.server_lock import (
    ServerLock,
    ServerLockUnavailableError,
)


class ServerLockTests(unittest.TestCase):
    CHILD_PROGRAM = """
import sys
from pathlib import Path
from webapp.server_lock import ServerLock, ServerLockUnavailableError

lock = ServerLock(Path(sys.argv[1]))
try:
    lock.acquire()
except ServerLockUnavailableError:
    raise SystemExit(23)
else:
    lock.release()
    raise SystemExit(0)
"""

    @classmethod
    def _child_attempt(cls, root):
        return subprocess.run(
            [sys.executable, "-c", cls.CHILD_PROGRAM, os.fspath(root)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_construction_requires_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            with self.assertRaises(ValueError):
                ServerLock(missing)

    def test_acquire_creates_restrictive_file_beneath_supplied_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = ServerLock(root)
            self.assertFalse(lock.acquired)

            self.assertIs(lock.acquire(), lock)
            self.assertTrue(lock.acquired)
            self.assertEqual(lock.lock_path.parent, root)
            self.assertEqual(lock.lock_path.name, ".server.lock")
            self.assertTrue(lock.lock_path.is_file())
            self.assertEqual(stat.S_IMODE(lock.lock_path.stat().st_mode), 0o600)

            lock.release()
            self.assertFalse(lock.acquired)
            self.assertTrue(lock.lock_path.exists())

    def test_context_manager_holds_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = ServerLock(temporary_directory)
            with lock as acquired:
                self.assertIs(acquired, lock)
                self.assertTrue(lock.acquired)
            self.assertFalse(lock.acquired)

    def test_existing_lock_file_alone_is_not_ownership(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock_file = root / ".server.lock"
            lock_file.touch(mode=0o644)

            lock = ServerLock(root).acquire()
            self.assertTrue(lock.acquired)
            self.assertEqual(stat.S_IMODE(lock_file.stat().st_mode), 0o600)
            lock.release()

    def test_second_os_process_cannot_acquire_held_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = ServerLock(root).acquire()
            child = self._child_attempt(root)
            lock.release()

            self.assertEqual(child.returncode, 23, child.stderr)
            self.assertEqual(child.stdout, "")

    def test_other_process_can_acquire_after_release(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock = ServerLock(root).acquire()
            self.assertEqual(self._child_attempt(root).returncode, 23)
            lock.release()

            child = self._child_attempt(root)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_repeated_acquire_and_release_are_safe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = ServerLock(temporary_directory)
            self.assertIs(lock.acquire(), lock)
            self.assertIs(lock.acquire(), lock)
            lock.release()
            lock.release()
            self.assertFalse(lock.acquired)

    def test_lock_source_has_no_task06_or_model_lifecycle_dependency(self):
        source = inspect.getsource(sys.modules[ServerLock.__module__])
        for prohibited in (
            "ProductionExecutionService",
            "ProductionRuntime",
            "orphan",
            "FrozenG1Runner",
            "VideoMultimodalRunner",
        ):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, source)


if __name__ == "__main__":
    unittest.main()
