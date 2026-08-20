from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from ctypes import wintypes
from types import SimpleNamespace

import pytest

import voiceover_pipeline.local_runtime.lifecycle as lifecycle_mod
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


class _FakeKernel32:
    """ctypes-shaped fake for the Win32 process image-name branch."""

    def __init__(
        self,
        *,
        image_name: str,
        query_ok: bool = True,
        last_error: int = 0,
    ) -> None:
        self.image_name = image_name
        self.query_ok = query_ok
        self.last_error = last_error
        self.open_calls: list[int] = []
        self.handles: list[int] = []
        self.query_calls: list[int] = []
        self.close_calls: list[int] = []

    def OpenProcess(self, desired_access: int, inherit: bool, pid: int) -> int:
        self.open_calls.append(pid)
        if self.query_ok:
            handle = 0x1234 + pid
            self.handles.append(handle)
            return handle
        return 0

    def QueryFullProcessImageNameW(
        self, handle: int, flags: int, buffer: object, size: object
    ) -> bool:
        self.query_calls.append(handle)
        if not self.query_ok:
            return False
        contents = ctypes.cast(buffer, ctypes.POINTER(wintypes.WCHAR))
        for index, char in enumerate(self.image_name):
            contents[index] = char
        contents[len(self.image_name)] = "\0"
        size_contents = ctypes.cast(size, ctypes.POINTER(wintypes.DWORD))
        size_contents.contents.value = len(self.image_name) + 1
        return True

    def CloseHandle(self, handle: int) -> bool:
        self.close_calls.append(handle)
        return True


def _fake_windows(
    monkeypatch: pytest.MonkeyPatch, kernel: _FakeKernel32, *, platform: str = "win32"
) -> None:
    monkeypatch.setattr(lifecycle_mod.sys, "platform", platform)
    monkeypatch.setattr(lifecycle_mod, "_kernel32", kernel, raising=False)
    monkeypatch.setattr(lifecycle_mod, "_win_last_error", lambda: kernel.last_error, raising=False)
    monkeypatch.setattr(lifecycle_mod, "_ERROR_ACCESS_DENIED", 5)
    monkeypatch.setattr(lifecycle_mod, "_ERROR_INVALID_PARAMETER", 87)


def _windows_probe(
    monkeypatch: pytest.MonkeyPatch,
    kernel: _FakeKernel32,
    *,
    process_command: Callable[[int], str] | None = None,
) -> object:
    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="4096, 10, 55\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="4242, python.exe\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="")

    _fake_windows(monkeypatch, kernel)
    return probe_local_gpu_state(runner=runner, process_command=process_command)


def test_windows_process_identity_queries_image_and_closes_handle(monkeypatch):
    from voiceover_pipeline.local_runtime.lifecycle import _windows_process_identity

    kernel = _FakeKernel32(image_name="C:\\App\\whisper_voice_machine.exe")
    _fake_windows(monkeypatch, kernel)

    assert _windows_process_identity(4242) == "C:\\App\\whisper_voice_machine.exe"
    assert kernel.open_calls == [4242]
    assert len(kernel.query_calls) == 1
    assert len(kernel.close_calls) == 1
    assert kernel.close_calls == kernel.handles


def test_windows_probe_detects_wvm_image_and_healthy_state(monkeypatch):
    kernel = _FakeKernel32(image_name="C:\\Tools\\whisper-voice-machine.exe")
    snapshot = _windows_probe(monkeypatch, kernel)

    assert snapshot.probe_error is None
    assert snapshot.free_vram_mb == 4096
    assert snapshot.wvm_active is True


def test_windows_probe_unknown_image_blocks_job(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import (
        GPULifecycleBlockedError,
        GPULifecycleOwner,
    )

    snapshot = _windows_probe(
        monkeypatch, _FakeKernel32(image_name="", query_ok=False), process_command=None
    )

    assert snapshot.probe_error is not None
    assert snapshot.wvm_active is False
    lifecycle = GPULifecycleOwner(
        GPULeaseManager(metadata_path=tmp_path / "lease.json"),
        probe=lambda: snapshot,
    )
    with pytest.raises(GPULifecycleBlockedError, match="cannot verify"):
        lifecycle.execute(FixtureDriver(), _request())


def test_windows_unknown_native_identity_fails_closed_even_when_name_looks_like_wvm(
    monkeypatch,
):
    kernel = _FakeKernel32(image_name="", query_ok=False)
    _fake_windows(monkeypatch, kernel)

    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="4096, 10, 55\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="4242, whisper_voice_machine.exe\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="")

    snapshot = probe_local_gpu_state(runner=runner)

    assert snapshot.probe_error is not None
    assert snapshot.wvm_active is False


def test_windows_probe_empty_image_is_not_wvm_and_healthy(monkeypatch):
    snapshot = _windows_probe(monkeypatch, _FakeKernel32(image_name="", query_ok=True))
    assert snapshot.probe_error is None
    assert snapshot.wvm_active is False


def test_windows_unknown_identity_makes_probe_error(monkeypatch):
    snapshot = _windows_probe(monkeypatch, _FakeKernel32(image_name="", query_ok=False))
    assert snapshot.probe_error is not None
    assert snapshot.wvm_active is False


def test_posix_probe_gone_process_is_not_wvm(monkeypatch):
    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="2048, 12, 52\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="4242, python\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="")

    def process_command(pid: int) -> str:
        assert pid == 4242
        return ""

    _fake_windows(monkeypatch, _FakeKernel32(image_name=""), platform="linux")
    snapshot = probe_local_gpu_state(runner=runner, process_command=process_command)

    assert snapshot.probe_error is None
    assert snapshot.wvm_active is False


def test_posix_missing_proc_returns_empty_command(monkeypatch):
    from voiceover_pipeline.local_runtime.lifecycle import _read_process_command

    class MissingProc:
        def read_bytes(self) -> bytes:
            raise FileNotFoundError("no such file")

        def __truediv__(self, _part: str) -> MissingProc:
            return self

    monkeypatch.setattr(lifecycle_mod, "Path", lambda _path: MissingProc())
    assert _read_process_command(1234) == ""


def test_posix_probe_missing_proc_is_not_wvm_and_not_an_error(monkeypatch):
    from voiceover_pipeline.local_runtime.lifecycle import (
        _read_process_command,
        probe_local_gpu_state,
    )

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="2048, 12, 52\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="4242, python\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="")

    _fake_windows(monkeypatch, _FakeKernel32(image_name=""), platform="linux")
    snapshot = probe_local_gpu_state(runner=runner, process_command=_read_process_command)

    assert snapshot.probe_error is None
    assert snapshot.wvm_active is False


def test_unknown_blocks_job_and_releases_lease(tmp_path):
    from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
    from voiceover_pipeline.local_runtime.lifecycle import (
        GPULifecycleBlockedError,
        GPULifecycleOwner,
        GPUSnapshot,
    )

    manager = GPULeaseManager(metadata_path=tmp_path / "lease.json")
    driver = FixtureDriver()
    lifecycle = GPULifecycleOwner(
        manager,
        probe=lambda: GPUSnapshot(
            free_vram_mb=4096,
            utilization_percent=1,
            temperature_c=50,
            xid_errors=(),
            wvm_active=False,
            probe_error="cannot determine WVM ownership state",
        ),
    )
    with pytest.raises(GPULifecycleBlockedError, match="cannot verify"):
        lifecycle.execute(driver, _request())

    assert driver.calls == 0
    with manager.acquire("after-block") as lease:
        assert lease.owner == "after-block"


def test_unknown_probe_state_is_structured_not_traceback(monkeypatch):
    from voiceover_pipeline.local_runtime.lifecycle import probe_local_gpu_state

    def runner(command: tuple[str, ...]) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout="2048, 12, 52\n")
        if command[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout="123, python\n")
        assert command == ("nvidia-smi", "-q")
        return SimpleNamespace(stdout="")

    def process_command(pid: int) -> str | None:
        raise OSError("process disappeared between snapshots")

    _fake_windows(monkeypatch, _FakeKernel32(image_name=""), platform="linux")
    snapshot = probe_local_gpu_state(runner=runner, process_command=process_command)
    assert snapshot.probe_error == "nvidia-smi safety probe is unavailable"
    assert snapshot.wvm_active is False


def test_windows_doctor_json_health_structure(monkeypatch, capsys):
    import json

    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(cli, "read_polza_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_openrouter_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_groq_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_xai_key", lambda: "fixture")
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "/fixture/bin")
    monkeypatch.setattr(
        cli, "omnivoice_local_dependency_probe", lambda: SimpleNamespace(available=True)
    )
    monkeypatch.setattr(
        lifecycle_mod,
        "probe_local_gpu_state",
        lambda: SimpleNamespace(probe_error=None),
    )
    args = cli.build_parser().parse_args(
        "doctor --provider omnivoice-local --timing-device cuda --json".split()
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.doctor_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["status"] == "success"
    assert data["checks"]["cuda"] == {"ok": True, "required": True}
    assert data["checks"]["omnivoice_local"] == {"ok": True, "required": True}
