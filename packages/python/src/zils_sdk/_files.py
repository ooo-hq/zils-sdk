"""Private atomic state and locks released by the operating system on exit."""

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".zils-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
            move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
            move.restype = wintypes.BOOL
            # REPLACE_EXISTING | WRITE_THROUGH commits the rename before token exchange.
            if not move(temporary, str(path), 0x1 | 0x8):
                raise ctypes.WinError()
        else:
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, timeout: float = 10):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                stream.write(b"\0")
                stream.flush()

            def lock():
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

            def unlock():
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock():
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(fd, fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                lock()
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "Another Zils command is updating local state; try again."
                    ) from None
                time.sleep(0.05)
            except OSError:
                if os.name != "nt":
                    raise
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "Another Zils command is updating local state; try again."
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            unlock()
