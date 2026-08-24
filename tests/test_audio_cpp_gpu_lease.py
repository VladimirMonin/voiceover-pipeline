from __future__ import annotations

import ctypes
import errno
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import pytest

import voiceover_pipeline.local_runtime.gpu_lease as gpu_lease_mod

_STILL_ACTIVE = gpu_lease_mod._STILL_ACTIVE
_ERROR_ACCESS_DENIED = gpu_lease_mod._ERROR_ACCESS_DENIED
_ERROR_INVALID_PARAMETER = gpu_lease_mod._ERROR_INVALID_PARAMETER


def test_gpu_lease_fifo_and_release_metadata_excludes_request_payload(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager

    manager = GPULeaseManager(metadata_path=tmp_path / "lease.json")
    first = manager.acquire("first")
    order: list[str] = []
    started = [threading.Event(), threading.Event()]

    def contender(owner: str, ready: threading.Event) -> None:
        ready.set()
        with manager.acquire(owner):
            order.append(owner)
            time.sleep(0.01)

    one = threading.Thread(target=contender, args=("second", started[0]))
    two = threading.Thread(target=contender, args=("third", started[1]))
    one.start()
    started[0].wait(timeout=1)
    two.start()
    started[1].wait(timeout=1)
    first.release()
    one.join(timeout=2)
    two.join(timeout=2)

    assert order == ["second", "third"]
    assert manager.metadata() is None


def test_gpu_lease_cancellation_stale_owner_and_process_death_release_waiters(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseCancelledError, GPULeaseManager

    alive = {999: True}
    now = [0.0]
    manager = GPULeaseManager(
        metadata_path=tmp_path / "lease.json",
        clock=lambda: now[0],
        owner_alive=lambda pid: alive.get(pid, False),
        stale_after_seconds=5.0,
    )
    manager.acquire("crashed-owner", owner_pid=999)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(GPULeaseCancelledError):
        manager.acquire("cancelled", cancel_event=cancelled)

    alive[999] = False
    assert manager.reap_stale() is True
    with manager.acquire("replacement", owner_pid=1000) as lease:
        metadata = manager.metadata()
        assert metadata is not None
        assert metadata["owner"] == "replacement"
        assert metadata["pid"] == 1000
        assert "audio" not in metadata
        assert "text" not in metadata
        assert lease.owner == "replacement"


def test_gpu_lease_reaps_dead_owner_without_killing_any_external_work(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager

    now = [10.0]
    manager = GPULeaseManager(
        metadata_path=tmp_path / "lease.json",
        clock=lambda: now[0],
        owner_alive=lambda _pid: False,
        stale_after_seconds=1.0,
    )
    manager.acquire("dead-owner", owner_pid=42)
    now[0] = 12.0

    assert manager.reap_stale() is True
    with manager.acquire("next-owner", owner_pid=43) as lease:
        assert lease.owner == "next-owner"


def test_gpu_lease_never_steals_a_live_long_inference_after_lease_age(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseError, GPULeaseManager

    now = [0.0]
    metadata_path = tmp_path / "lease.json"
    owner = GPULeaseManager(
        metadata_path=metadata_path,
        clock=lambda: now[0],
        owner_alive=lambda _pid: True,
        stale_after_seconds=1.0,
    )
    contender = GPULeaseManager(
        metadata_path=metadata_path,
        clock=lambda: now[0],
        owner_alive=lambda _pid: True,
        stale_after_seconds=1.0,
    )
    lease = owner.acquire("live-long-job", owner_pid=42)
    now[0] = 600.0

    with pytest.raises(GPULeaseError, match="timed out"):
        contender.acquire("competing-job", owner_pid=43, timeout_seconds=0)

    metadata = owner.metadata()
    assert metadata is not None
    assert metadata["owner"] == "live-long-job"
    lease.release()


def test_gpu_lease_serializes_claims_between_manager_instances(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager

    metadata_path = tmp_path / "lease.json"
    first_manager = GPULeaseManager(metadata_path=metadata_path)
    second_manager = GPULeaseManager(metadata_path=metadata_path)
    entered: list[str] = []
    start = threading.Barrier(2)
    release = threading.Event()

    def contender(manager: GPULeaseManager, owner: str) -> None:
        start.wait(timeout=1)
        with manager.acquire(owner, timeout_seconds=1):
            entered.append(owner)
            release.wait(timeout=1)

    first = threading.Thread(target=contender, args=(first_manager, "one"))
    second = threading.Thread(target=contender, args=(second_manager, "two"))
    first.start()
    second.start()
    for _ in range(50):
        if entered:
            break
        time.sleep(0.01)
    time.sleep(0.05)

    assert len(entered) == 1
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert sorted(entered) == ["one", "two"]


class _FakeKernel32:
    """ctypes-shaped fake for the Win32 process liveness branch."""

    def __init__(
        self,
        *,
        open_result: int,
        query_ok: bool = True,
        exit_code: int = _STILL_ACTIVE,
    ) -> None:
        self.open_result = open_result
        self.query_ok = query_ok
        self.exit_code = exit_code
        self.open_calls: list[int] = []
        self.close_calls: list[int] = []

    def OpenProcess(self, desired_access: int, inherit: bool, pid: int) -> int:
        self.open_calls.append(pid)
        return self.open_result

    def GetExitCodeProcess(self, handle: int, exit_code_out: object) -> bool:
        if not self.query_ok:
            return False
        ctypes.cast(exit_code_out, ctypes.POINTER(wintypes.DWORD)).contents.value = self.exit_code
        return True

    def CloseHandle(self, handle: int) -> bool:
        self.close_calls.append(handle)
        return True


def _fake_windows_liveness(
    monkeypatch: pytest.MonkeyPatch, kernel: _FakeKernel32, last_error: int = 0
) -> None:
    monkeypatch.setattr(gpu_lease_mod.sys, "platform", "win32")
    monkeypatch.setattr(gpu_lease_mod, "_kernel32", kernel)
    monkeypatch.setattr(gpu_lease_mod, "_win_last_error", lambda: last_error)


def test_windows_liveness_tri_state_alive_dead_unknown(monkeypatch):
    from voiceover_pipeline.local_runtime.gpu_lease import _probe_liveness, _process_alive

    # alive: handle opens and exit code is STILL_ACTIVE
    kernel = _FakeKernel32(open_result=0x1234, exit_code=_STILL_ACTIVE)
    _fake_windows_liveness(monkeypatch, kernel)
    assert _probe_liveness(4242) == "alive"
    assert _process_alive(4242) is True

    # dead: exit code is a real termination code
    kernel = _FakeKernel32(open_result=0x1234, exit_code=1)
    _fake_windows_liveness(monkeypatch, kernel)
    assert _probe_liveness(4242) == "dead"
    assert _process_alive(4242) is False

    # dead: OpenProcess fails with ERROR_INVALID_PARAMETER (no such process)
    kernel = _FakeKernel32(open_result=0)
    _fake_windows_liveness(monkeypatch, kernel, last_error=_ERROR_INVALID_PARAMETER)
    assert _probe_liveness(4242) == "dead"

    # alive: OpenProcess fails with ERROR_ACCESS_DENIED (exists but protected)
    kernel = _FakeKernel32(open_result=0)
    _fake_windows_liveness(monkeypatch, kernel, last_error=_ERROR_ACCESS_DENIED)
    assert _probe_liveness(4242) == "alive"
    assert _process_alive(4242) is True

    # unknown: OpenProcess fails with an unexpected error
    kernel = _FakeKernel32(open_result=0)
    _fake_windows_liveness(monkeypatch, kernel, last_error=12345)
    assert _probe_liveness(4242) == "unknown"

    # unknown: GetExitCodeProcess fails on a valid handle
    kernel = _FakeKernel32(open_result=0x1234, query_ok=False)
    _fake_windows_liveness(monkeypatch, kernel)
    assert _probe_liveness(4242) == "unknown"

    # unknown must not allow a safe-looking steal: treated as alive
    kernel = _FakeKernel32(open_result=0)
    _fake_windows_liveness(monkeypatch, kernel, last_error=12345)
    assert _process_alive(4242) is True

    # non-positive PIDs are dead on any platform
    assert _probe_liveness(0) == "dead"
    assert _probe_liveness(-7) == "dead"


@pytest.mark.platform_simulated
def test_windows_liveness_unknown_prevents_lease_steal(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import (
        GPULeaseError,
        GPULeaseManager,
        _process_alive,
    )

    # Owner PID cannot be probed (unknown) -> the lease must stay held.
    kernel = _FakeKernel32(open_result=0)
    _fake_windows_liveness(monkeypatch, kernel, last_error=12345)
    metadata_path = tmp_path / "lease.json"
    owner = GPULeaseManager(
        metadata_path=metadata_path,
        clock=lambda: 0.0,
        owner_alive=_process_alive,
        stale_after_seconds=1.0,
    )
    owner.acquire("owner", owner_pid=999)
    now = [100.0]
    contender = GPULeaseManager(
        metadata_path=metadata_path,
        clock=lambda: now[0],
        owner_alive=_process_alive,
        stale_after_seconds=1.0,
    )
    with pytest.raises(GPULeaseError, match="timed out"):
        contender.acquire("thief", owner_pid=1, timeout_seconds=0)
    assert owner.metadata()["owner"] == "owner"


def test_posix_liveness_preserves_os_kill_semantics(monkeypatch):
    from voiceover_pipeline.local_runtime.gpu_lease import _probe_liveness

    monkeypatch.setattr(gpu_lease_mod.sys, "platform", "linux")
    captured: list[int] = []

    def fake_kill(pid: int, _sig: int) -> None:
        captured.append(pid)

    monkeypatch.setattr(gpu_lease_mod.os, "kill", fake_kill)
    assert _probe_liveness(777) == "alive"
    assert captured == [777]

    def fake_kill_lookup(pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(gpu_lease_mod.os, "kill", fake_kill_lookup)
    assert _probe_liveness(777) == "dead"

    def fake_kill_permission(pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(gpu_lease_mod.os, "kill", fake_kill_permission)
    assert _probe_liveness(777) == "alive"


@pytest.mark.platform_simulated
def test_windows_lock_backend_retries_until_free_and_unlocks(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import _windows_lock_backend

    lock_attempts: list[int] = []
    unlock_calls: list[int] = []
    state = {"locked": False}

    class FakeMsvcrt:
        LK_NBLCK = 2
        LK_UNLCK = 3

        def locking(self, fd: int, op: int, nbytes: int) -> None:
            if op == self.LK_UNLCK:
                unlock_calls.append(fd)
                state["locked"] = False
                return
            lock_attempts.append(fd)
            if state["locked"]:
                raise OSError(errno.EACCES, "already locked")
            state["locked"] = True

    monkeypatch.setattr(gpu_lease_mod, "msvcrt", FakeMsvcrt(), raising=False)
    lock_path = tmp_path / "lease.json.lock"
    with _windows_lock_backend(lock_path):
        pass
    with _windows_lock_backend(lock_path):
        pass
    assert len(lock_attempts) == 2
    assert len(unlock_calls) == 2


def test_posix_lock_backend_uses_flock_ex_and_unlock(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import _posix_lock_backend

    calls: list[tuple[int, int]] = []

    class FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 8

        def flock(self, fd: int, op: int) -> None:
            calls.append((fd, op))

    monkeypatch.setattr(gpu_lease_mod, "_fcntl", FakeFcntl(), raising=False)
    lock_path = tmp_path / "lock.json.lock"

    with _posix_lock_backend(lock_path):
        pass

    assert len(calls) == 2
    assert calls[0][1] == 1
    assert calls[1][0] == calls[0][0]
    assert calls[1][1] == 8


@pytest.mark.native_windows
@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows file locking")
def test_windows_lock_backend_exclusive_across_processes(tmp_path):
    if sys.platform != "win32":
        pytest.skip("real Windows cross-process lock")

    from voiceover_pipeline.local_runtime.gpu_lease import _windows_lock_backend

    lock_path = tmp_path / "lease.json.lock"
    package_dir = os.path.dirname(gpu_lease_mod.__file__)
    src_dir = os.path.abspath(os.path.join(package_dir, "..", "..", "..", "src"))
    child_code = (
        "import sys\n"
        f"sys.path.insert(0, {src_dir!r})\n"
        "from pathlib import Path\n"
        "from voiceover_pipeline.local_runtime.gpu_lease import _windows_lock_backend\n"
        f"with _windows_lock_backend(Path({str(lock_path)!r})):\n"
        "    print('HELD')\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    with _windows_lock_backend(lock_path):
        time.sleep(0.4)
    stdout, stderr = child.communicate(timeout=10)
    assert child.returncode == 0, f"child failed: {stdout!r} {stderr!r}"
    assert "HELD" in stdout
