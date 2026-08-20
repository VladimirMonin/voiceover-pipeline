from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

if sys.platform == "win32":
    import msvcrt

    _fcntl: Any = None
else:
    import fcntl

    _fcntl = fcntl

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87

# ctypes.get_last_error() only reports errors for WinDLL handles created with
# use_last_error=True; the tiny indirection keeps the Windows liveness branch
# fakeable in tests that run on any host OS.
_win_last_error: Callable[[], int] = getattr(ctypes, "get_last_error", lambda: 0)

_kernel32: Any
if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
else:
    _kernel32 = None

LockBackend = Callable[[Path], AbstractContextManager[None]]


def _default_lock_backend() -> LockBackend:
    if sys.platform == "win32":
        return _windows_lock_backend
    return _posix_lock_backend


@contextmanager
def _windows_lock_backend(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fd = lock_file.fileno()
        # LK_NBLCK is never blocking and is released when the handle is
        # closed even if this process dies, so ownership never survives
        # the owner. The current file position must cover the locked byte.
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EDEADLK):
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextmanager
def _posix_lock_backend(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)


class GPULeaseError(RuntimeError):
    pass


class GPULeaseCancelledError(GPULeaseError):
    pass


@dataclass(frozen=True)
class GPULease:
    _manager: "GPULeaseManager"
    token: str
    owner: str
    pid: int

    def release(self) -> None:
        self._manager.release(self)

    def __enter__(self) -> "GPULease":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


class GPULeaseManager:
    """FIFO local GPU ownership with minimal, non-content cross-process metadata."""

    def __init__(
        self,
        *,
        metadata_path: Path,
        stale_after_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        owner_alive: Callable[[int], bool] | None = None,
        lock_backend: LockBackend | None = None,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self._metadata_path = metadata_path
        self._lock_path = metadata_path.with_name(f"{metadata_path.name}.lock")
        self._clock = clock
        self._owner_alive = owner_alive or _process_alive
        self._lock_backend = lock_backend or _default_lock_backend()
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active: GPULease | None = None

    def acquire(
        self,
        owner: str,
        *,
        owner_pid: int | None = None,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> GPULease:
        if not owner.strip():
            raise ValueError("GPU lease owner must not be blank")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("GPU lease timeout must be non-negative")
        pid = os.getpid() if owner_pid is None else owner_pid
        ticket = object()
        deadline = None if timeout_seconds is None else self._clock() + timeout_seconds
        with self._condition:
            self._waiters.append(ticket)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise GPULeaseCancelledError("GPU lease acquisition cancelled")
                    self._reap_stale_locked()
                    if self._waiters[0] is ticket and self._active is None:
                        lease = GPULease(self, uuid4().hex, owner, pid)
                        if self._try_claim(lease):
                            self._active = lease
                            self._waiters.popleft()
                            return lease
                    if deadline is not None:
                        remaining = deadline - self._clock()
                        if remaining <= 0:
                            raise GPULeaseError("timed out waiting for GPU lease")
                        self._condition.wait(min(remaining, 0.05))
                    else:
                        self._condition.wait(0.05)
            finally:
                try:
                    self._waiters.remove(ticket)
                except ValueError:
                    pass

    def release(self, lease: GPULease) -> None:
        with self._condition:
            self._release_matching_metadata(lease)
            if self._active is not None and self._active.token == lease.token:
                self._active = None
            self._condition.notify_all()

    def reap_stale(self) -> bool:
        with self._condition:
            changed = self._reap_stale_locked()
            if changed:
                self._condition.notify_all()
            return changed

    def metadata(self) -> dict[str, Any] | None:
        with self._condition:
            metadata = self._read_metadata()
            return dict(metadata) if metadata is not None else None

    def _reap_stale_locked(self) -> bool:
        with self._locked_file():
            metadata = self._read_metadata_unlocked()
            if metadata is None or not self._metadata_is_stale(metadata):
                return False
            self._remove_metadata_unlocked()
        self._active = None
        return True

    def _try_claim(self, lease: GPULease) -> bool:
        with self._locked_file():
            metadata = self._read_metadata_unlocked()
            if metadata is not None and self._metadata_is_stale(metadata):
                self._remove_metadata_unlocked()
                self._active = None
                metadata = None
            if metadata is not None:
                return False
            self._write_metadata_unlocked(_lease_metadata(lease, self._clock()))
            return True

    def _release_matching_metadata(self, lease: GPULease) -> None:
        with self._locked_file():
            metadata = self._read_metadata_unlocked()
            if (
                metadata is not None
                and metadata.get("token") == lease.token
                and metadata.get("pid") == lease.pid
            ):
                self._remove_metadata_unlocked()

    def _metadata_is_stale(self, metadata: dict[str, Any]) -> bool:
        pid = metadata.get("pid")
        acquired_at = metadata.get("acquired_at")
        if not isinstance(pid, int) or not isinstance(acquired_at, (int, float)):
            return True
        # A finite ASR/TTS invocation may legitimately exceed the former age
        # cutoff. Reclaim only a malformed lease or one whose owner has died.
        return not self._owner_alive(pid)

    @contextmanager
    def _locked_file(self) -> Iterator[None]:
        with self._lock_backend(self._lock_path):
            yield

    def _read_metadata(self) -> dict[str, Any] | None:
        with self._locked_file():
            return self._read_metadata_unlocked()

    def _read_metadata_unlocked(self) -> dict[str, Any] | None:
        if not self._metadata_path.exists():
            return None
        try:
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _write_metadata_unlocked(self, metadata: dict[str, Any]) -> None:
        temporary = self._metadata_path.with_name(f"{self._metadata_path.name}.tmp")
        temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._metadata_path)

    def _remove_metadata_unlocked(self) -> None:
        try:
            self._metadata_path.unlink()
        except FileNotFoundError:
            pass


def _lease_metadata(lease: GPULease, acquired_at: float) -> dict[str, Any]:
    return {
        "owner": lease.owner,
        "pid": lease.pid,
        "token": lease.token,
        "acquired_at": acquired_at,
    }


_ProcessLiveness = Literal["alive", "dead", "unknown"]


def _process_alive(pid: int) -> bool:
    """Report whether a process is alive, failing closed when unsure.

    On POSIX the behavior is unchanged: ``os.kill(pid, 0)`` signals the
    process and reports it dead when no such process exists, mirroring the
    pre-existing semantics exactly.

    On Windows ``os.kill(pid, 0)`` cannot be trusted for liveness (it always
    succeeds for any integer handle, including exited processes), so the
    process is queried through ``OpenProcess`` with
    PROCESS_QUERY_LIMITED_INFORMATION and ``GetExitCodeProcess``. A
    successfully queried process whose exit code is STILL_ACTIVE is alive;
    a process that cannot be created or queried is treated as alive unless
    it is positively dead. A stale lease is only reclaimed when this
    function returns False, so an undetermined liveness always keeps the
    lease held (fail-closed) and never allows a look-alike steal.
    """
    return _probe_liveness(pid) != "dead"


def _probe_liveness(pid: int) -> _ProcessLiveness:
    """Tri-state liveness probe: alive, dead, or unknown.

    ``unknown`` means the state could not be determined (an access-denied
    handle, a failed exit-code query, or another unexpected error). Callers
    must treat ``unknown`` as held: a lease is only reclaimed when the owner
    is positively dead.
    """
    if pid <= 0:
        return "dead"
    if sys.platform == "win32":
        return _probe_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "alive"
    return "alive"


def _probe_windows(pid: int) -> _ProcessLiveness:
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = _win_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            # No such process (PID is gone or never existed).
            return "dead"
        if error == _ERROR_ACCESS_DENIED:
            # The process exists but refused the handle: treat as alive.
            return "alive"
        return "unknown"
    try:
        exit_code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return "unknown"
        return "alive" if exit_code.value == _STILL_ACTIVE else "dead"
    finally:
        _kernel32.CloseHandle(handle)
