"""The one GPU every session on this machine shares, taken only while a job computes.

The rule is in AGENTS.md: ``%TEMP%\\assetfurnace-gpu.lock`` holds ``<pid> <session>``. It is
created exclusively before GPU work and deleted afterwards, also on error; it is never held
across installs, downloads or waits. A lock whose process has died may be taken over.

    with gpu("materials"):
        run_the_model()

The file appears with its holder already written in it (written aside, then linked into
place), so nobody reads a half-made lock. One that cannot be read anyway, left by a writer
that died or by hand, is taken over after UNREADABLE_AFTER seconds rather than waited on
for an hour; and a live pid that started after its lock was written is a reused number.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "assetfurnace-gpu.lock"

# a lock with no pid in it this long is nobody's: a writer that died before writing it
UNREADABLE_AFTER = 60.0


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


def process_started(pid: int) -> float | None:
    """When the process `pid` started, in seconds since the epoch; None when unknown.
    Windows and Linux only: elsewhere a reused pid is not detected."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.windll.kernel32
            handle = kernel.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                    return None
                ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                return ticks / 1e7 - 11644473600.0               # 100 ns since 1601 -> epoch
            finally:
                kernel.CloseHandle(handle)
        stat = Path(f"/proc/{pid}/stat").read_text()
        since_boot = int(stat.rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
        return boot + since_boot
    except (OSError, ValueError, AttributeError, IndexError, StopIteration):
        return None


def holder(path: Path = LOCK) -> tuple[int | None, str] | None:
    """(pid, text) of whoever holds the lock, or None when nobody does."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    first = text.split(maxsplit=1)[0] if text else ""
    return (int(first) if first.isdigit() else None), text


def _take(path: Path, mine: str) -> bool:
    """Create the lock with `mine` already in it; False when someone holds it."""
    aside = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    aside.write_text(mine, encoding="utf-8")
    try:
        os.link(aside, path)                           # exclusive, and the file is complete
        return True
    except FileExistsError:
        return False
    except OSError:
        pass                                           # no hard links here (FAT, some shares)
    finally:
        aside.unlink(missing_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(mine)
    return True


def breaking(path: Path) -> Path:
    """The file a job holds while it removes a stale lock at `path`: one breaker at a time."""
    return path.with_name(f"{path.name}.breaking")


def _break(path: Path, judged: str, alive: Callable[[int], bool] = pid_alive) -> bool:
    """Remove the stale lock `judged`, and only that one. Two waiters can judge the same dead
    lock; moving it aside and back (as this did before) let a third job take the GPU in
    between. Now one breaker at a time holds `breaking(path)`, and under it the lock is
    removed only while it still says `judged`: a dead holder never rewrites its lock, a taker
    only creates one where there is none, and every other breaker waits on the guard, so
    nothing can replace it between the reading and the removal. False while another
    breaker is at it."""
    guard = breaking(path)
    try:
        handle = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _clear_dead_guard(guard, alive)
        return False                                   # someone else is at it: look again
    except OSError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(str(os.getpid()))
    try:
        try:
            now = path.read_text(encoding="utf-8").strip()
        except OSError:                                # gone already, or open: look again
            return True
        if now == judged:
            try:
                path.unlink()
            except OSError:
                pass
    finally:
        _unlink(guard)
    return True


GUARD_STALE = 60.0  # a breaker holds its guard for milliseconds; older than this, it died


def _clear_dead_guard(guard: Path, alive: Callable[[int], bool]) -> None:
    """Remove the guard of a breaker that died holding it, else it would block every break."""
    try:
        text = guard.read_text(encoding="utf-8").strip()
        age = time.time() - guard.stat().st_mtime
    except OSError:
        return
    if (text.isdigit() and not alive(int(text))) or age > GUARD_STALE:
        _unlink(guard)


def _unlink(path: Path, attempts: int = 500, nap: float = 0.01) -> None:
    """Delete `path`; on Windows a reader that has it open makes that fail for a moment."""
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(nap)
        except OSError:
            return


def _stale(path: Path, pid: int | None, alive: Callable[[int], bool],
           started: Callable[[int], float | None]) -> bool:
    try:
        written = path.stat().st_mtime
    except FileNotFoundError:
        return False
    if pid is None:
        return time.time() - written > UNREADABLE_AFTER
    if not alive(pid):
        return True
    born = started(pid)
    return born is not None and born > written + 2.0  # a newer process got the old number


@contextmanager
def gpu(session: str, *, wait: float = 3600.0, poll: float = 10.0, path: Path | None = None,
        alive: Callable[[int], bool] = pid_alive,
        started: Callable[[int], float | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic) -> Iterator[None]:
    """Hold the shared GPU lock for the duration of the block, waiting up to `wait` seconds.

    `started` tells when a pid's process began (process_started when the real `alive` is
    used; a test that fakes `alive` fakes it too, or no reused pid is seen)."""
    path = Path(path or LOCK)
    if started is None:
        started = process_started if alive is pid_alive else (lambda pid: None)
    mine = f"{os.getpid()} {session}"
    start = clock()
    while not _take(path, mine):
        try:
            found = holder(path)
        except OSError:                                # being moved or written: look again
            sleep(min(poll, 1.0))
            continue
        if found is None:
            continue                                   # released between the two calls
        pid, text = found
        if _stale(path, pid, alive, started):
            if not _break(path, text, alive):
                sleep(min(poll, 0.05))
            continue
        if clock() - start > wait:
            raise GpuBusy(f"the GPU is busy ({text or 'an empty lock file'}); try again when it is free")
        sleep(poll)
    try:
        yield
    finally:
        _release(path, mine)


def _release(path: Path, mine: str, attempts: int = 500, nap: float = 0.01) -> None:
    """Delete this job's lock. Windows refuses to delete a file another process has open,
    and a waiting job opens the lock to read who holds it, so a refusal is tried again
    (for up to five seconds): giving up would leave the lock to block every GPU job."""
    for _ in range(attempts):
        try:
            if holder(path) == (os.getpid(), mine):
                path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(nap)
        except OSError:
            return
