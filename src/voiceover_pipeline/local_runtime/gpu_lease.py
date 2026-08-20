from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


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
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self._metadata_path = metadata_path
        self._lock_path = metadata_path.with_name(f"{metadata_path.name}.lock")
        self._clock = clock
        self._owner_alive = owner_alive or _process_alive
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
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
