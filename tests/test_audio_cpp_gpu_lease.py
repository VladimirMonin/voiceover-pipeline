from __future__ import annotations

import threading
import time

import pytest


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
