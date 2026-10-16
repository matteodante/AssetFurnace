"""The one GPU every session on this machine shares, taken only while a job computes.

The rule is in AGENTS.md: ``%TEMP%\\assetfurnace-gpu.lock`` holds ``<pid> <session>``. It is
created exclusively before GPU work and deleted afterwards, also on error; it is never held
across installs, downloads or waits. A lock whose process has died may be taken over.

    with gpu("materials"):
        run_the_model()
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "assetfurnace-gpu.lock"


class GpuBusy(RuntimeError):
    """Another live session kept the GPU for longer than this job would wait."""


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running. Never signals it."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) on Windows terminates the process; ask the kernel instead.
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.GetLastError() == 5                    # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder(path: Path = LOCK) -> tuple[int | None, str] | None:
    """(pid, text) of whoever holds the lock, or None when nobody does."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    first = text.split(maxsplit=1)[0] if text else ""
    return (int(first) if first.isdigit() else None), text


def _take(path: Path, mine: str) -> bool:
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(mine)
    return True


@contextmanager
def gpu(session: str, *, wait: float = 3600.0, poll: float = 10.0, path: Path = LOCK,
        alive: Callable[[int], bool] = pid_alive, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic) -> Iterator[None]:
    """Hold the shared GPU lock for the duration of the block, waiting up to `wait` seconds."""
    path = Path(path)
    mine = f"{os.getpid()} {session}"
    start = clock()
    while not _take(path, mine):
        found = holder(path)
        if found is None:
            continue                                   # released between the two calls
        pid, text = found
        if pid is not None and not alive(pid):
            # Re-read first: never remove a lock someone took over a moment ago.
            if holder(path) == found:
                path.unlink(missing_ok=True)
            continue
        if clock() - start > wait:
            raise GpuBusy(f"the GPU is busy ({text}); try again when it is free")
        sleep(poll)
    try:
        yield
    finally:
        if holder(path) == (os.getpid(), mine):
            path.unlink(missing_ok=True)
