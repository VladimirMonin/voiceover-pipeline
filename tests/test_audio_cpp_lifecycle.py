from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from voiceover_pipeline.local_runtime.contracts import (
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    RuntimeDriverHealth,
)


def _request() -> LocalRuntimeRequest:
    return LocalRuntimeRequest(
        request_id="request-1",
        operation="asr",
        family="qwen3-asr",
        provider_id="qwen-local",
        payload={"model_id": "Qwen/Qwen3-ASR-0.6B"},
    )


class FixtureDriver:
    driver_id = "audio-cpp"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.cancelled: list[str] = []
        self.closed = 0

    def health(self) -> RuntimeDriverHealth:
        return RuntimeDriverHealth(available=True)

    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture driver failure")
        return LocalRuntimeResponse(
            request_id=request.request_id, payload={"transcript": "fixture"}
        )

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)

    def close(self) -> None:
        self.closed += 1


def test_real_local_gpu_probe_collects_only_safety_metrics_and_wvm_state():
    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        commands.append(command)
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="2048, 12, 52\n4096, 44, 63\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="123, python\n456, helper\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="Xid Errors : 31, 45\n")

    snapshot = probe_local_gpu_state(
        runner=runner,
        process_command=lambda pid: "python -m whisper_voice_machine.worker" if pid == 123 else "",
    )

    assert snapshot.free_vram_mb == 2048
    assert snapshot.utilization_percent == 44
    assert snapshot.temperature_c == 63
    assert snapshot.xid_errors == (31, 45)
    assert snapshot.wvm_active is True
    assert snapshot.probe_error is None
    assert commands == [
        (
            "nvidia-smi",
            "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ),
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ),
        ("nvidia-smi", "-q"),
    ]


def test_real_local_gpu_probe_fails_closed_when_the_local_state_is_unavailable(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import (
        GPULifecycleBlockedError,
        GPULifecycleOwner,
        probe_local_gpu_state,
    )

    def unavailable(_command: tuple[str, ...]) -> SimpleNamespace:
        raise OSError("nvidia-smi unavailable")

    snapshot = probe_local_gpu_state(runner=unavailable)
    lifecycle = GPULifecycleOwner(
        GPULeaseManager(metadata_path=tmp_path / "lease.json"),
        probe=lambda: snapshot,
    )

    assert snapshot.probe_error == "nvidia-smi safety probe is unavailable"
    with pytest.raises(GPULifecycleBlockedError, match="cannot verify"):
        lifecycle.execute(FixtureDriver(), _request())


def test_real_local_gpu_probe_fails_closed_on_incomplete_metrics():
    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    snapshot = probe_local_gpu_state(
        runner=lambda _command: SimpleNamespace(stdout="N/A, N/A, N/A\n")
    )

    assert snapshot.probe_error == "nvidia-smi safety probe is unavailable"


def test_lifecycle_blocks_wvm_or_unhealthy_gpu_without_killing_external_owner(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import (
        GPULifecycleBlockedError,
        GPULifecycleOwner,
        GPUSnapshot,
    )

    driver = FixtureDriver()
    lifecycle = GPULifecycleOwner(
        GPULeaseManager(metadata_path=tmp_path / "lease.json"),
        probe=lambda: GPUSnapshot(
            free_vram_mb=4096,
            utilization_percent=0,
            temperature_c=50,
            xid_errors=(),
            wvm_active=True,
        ),
    )

    with pytest.raises(GPULifecycleBlockedError, match="WVM"):
        lifecycle.execute(driver, _request())

    assert driver.calls == 0
    assert driver.cancelled == []
    assert driver.closed == 0


def test_lifecycle_preflight_and_cleanup_release_lease_on_every_driver_exit(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import (
        GPULifecycleBlockedError,
        GPULifecycleOwner,
        GPUSnapshot,
    )

    manager = GPULeaseManager(metadata_path=tmp_path / "lease.json")
    bad = GPULifecycleOwner(
        manager,
        probe=lambda: GPUSnapshot(
            free_vram_mb=512, utilization_percent=10, temperature_c=55, xid_errors=(31,)
        ),
        min_free_vram_mb=1024,
    )
    with pytest.raises(GPULifecycleBlockedError, match="Xid"):
        bad.execute(FixtureDriver(), _request())

    lifecycle = GPULifecycleOwner(
        manager,
        probe=lambda: GPUSnapshot(
            free_vram_mb=4096, utilization_percent=10, temperature_c=55, xid_errors=()
        ),
    )
    with pytest.raises(RuntimeError, match="fixture driver failure"):
        lifecycle.execute(FixtureDriver(fail=True), _request())
    with manager.acquire("after-failure") as lease:
        assert lease.owner == "after-failure"


def test_lifecycle_cancellation_routes_to_active_driver_and_releases_lease(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleOwner, GPUSnapshot

    class BlockingDriver(FixtureDriver):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.cancelled_event = threading.Event()

        def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
            self.started.set()
            assert self.cancelled_event.wait(timeout=2)
            return LocalRuntimeResponse(
                request_id=request.request_id, payload={"transcript": "fixture"}
            )

        def cancel(self, request_id: str) -> None:
            super().cancel(request_id)
            self.cancelled_event.set()

    manager = GPULeaseManager(metadata_path=tmp_path / "lease.json")
    lifecycle = GPULifecycleOwner(
        manager,
        probe=lambda: GPUSnapshot(
            free_vram_mb=4096, utilization_percent=10, temperature_c=55, xid_errors=()
        ),
    )
    driver = BlockingDriver()
    worker = threading.Thread(target=lambda: lifecycle.execute(driver, _request()))
    worker.start()
    assert driver.started.wait(timeout=1)

    lifecycle.cancel("request-1", family="qwen3-asr")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert driver.cancelled == ["request-1"]
    with manager.acquire("after-cancel") as lease:
        assert lease.owner == "after-cancel"
