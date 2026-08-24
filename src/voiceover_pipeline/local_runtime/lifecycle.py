from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceover_pipeline.local_runtime.contracts import (
    LocalAudioRuntimeDriver,
    LocalRuntimeRequest,
    LocalRuntimeResponse,
)
from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager

if sys.platform == "win32":
    _kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
else:
    _kernel32: Any = None

# ctypes.get_last_error() only reports errors for WinDLL handles created with
# use_last_error=True; the tiny indirection keeps the Windows identity branch
# fakeable in tests that run on any host OS.
_win_last_error: Callable[[], int] = getattr(ctypes, "get_last_error", lambda: 0)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MAX_IMAGE_PATH = 4096
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


class GPULifecycleBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUSnapshot:
    free_vram_mb: int | None
    utilization_percent: int | None
    temperature_c: int | None
    xid_errors: tuple[int, ...] = ()
    wvm_active: bool = False
    probe_error: str | None = None


class GPULifecycleOwner:
    """Owns lease, preflight, cancellation, and cleanup without touching WVM."""

    def __init__(
        self,
        leases: GPULeaseManager,
        *,
        probe: Callable[[], GPUSnapshot],
        min_free_vram_mb: int = 0,
        max_utilization_percent: int = 100,
        max_temperature_c: int = 95,
    ) -> None:
        self._leases = leases
        self._probe = probe
        self._min_free_vram_mb = min_free_vram_mb
        self._max_utilization_percent = max_utilization_percent
        self._max_temperature_c = max_temperature_c
        self._active: dict[str, tuple[str, LocalAudioRuntimeDriver]] = {}
        self._lock = threading.Lock()

    def execute(
        self, driver: LocalAudioRuntimeDriver, request: LocalRuntimeRequest
    ) -> LocalRuntimeResponse:
        with self._leases.acquire(request.family) as _lease:
            self._preflight()
            with self._lock:
                self._active[request.request_id] = (request.family, driver)
            try:
                return driver.invoke(request)
            finally:
                with self._lock:
                    self._active.pop(request.request_id, None)

    def cancel(self, request_id: str, *, family: str) -> bool:
        with self._lock:
            active = self._active.get(request_id)
        if active is None or active[0] != family:
            return False
        active[1].cancel(request_id)
        return True

    def restart(self, driver: LocalAudioRuntimeDriver) -> None:
        driver.close()

    def _preflight(self) -> None:
        snapshot = self._probe()
        if snapshot.probe_error is not None:
            raise GPULifecycleBlockedError("GPU preflight cannot verify local GPU/WVM safety state")
        if snapshot.wvm_active:
            raise GPULifecycleBlockedError(
                "WVM owns active GPU work; voiceover job will wait or fail closed"
            )
        if snapshot.xid_errors:
            raise GPULifecycleBlockedError(
                f"GPU Xid errors present: {', '.join(str(code) for code in snapshot.xid_errors)}"
            )
        if snapshot.free_vram_mb is not None and snapshot.free_vram_mb < self._min_free_vram_mb:
            raise GPULifecycleBlockedError("insufficient free GPU memory for voiceover job")
        if (
            snapshot.utilization_percent is not None
            and snapshot.utilization_percent > self._max_utilization_percent
        ):
            raise GPULifecycleBlockedError(
                "GPU utilization is above the voiceover safety threshold"
            )
        if snapshot.temperature_c is not None and snapshot.temperature_c > self._max_temperature_c:
            raise GPULifecycleBlockedError(
                "GPU temperature is above the voiceover safety threshold"
            )


def probe_local_gpu_state(
    *,
    runner: Callable[[tuple[str, ...]], object] | None = None,
    process_command: Callable[[int], str | None] | None = None,
) -> GPUSnapshot:
    """Read only local GPU safety state; never retain process details or command output."""
    run = runner or _run_nvidia_smi
    process_text = process_command or _process_identity
    try:
        metric_rows = _gpu_metric_rows(
            _completed_stdout(
                run(
                    (
                        "nvidia-smi",
                        "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    )
                )
            )
        )
        applications = _completed_stdout(
            run(
                (
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader,nounits",
                )
            )
        )
        errors = _completed_stdout(run(("nvidia-smi", "-q")))
        wvm_state = _wvm_process_state(_gpu_processes(applications), process_text)
    except (OSError, subprocess.SubprocessError, ValueError):
        return GPUSnapshot(
            free_vram_mb=None,
            utilization_percent=None,
            temperature_c=None,
            probe_error="nvidia-smi safety probe is unavailable",
        )

    free_vram = [row[0] for row in metric_rows if row[0] is not None]
    utilization = [row[1] for row in metric_rows if row[1] is not None]
    temperature = [row[2] for row in metric_rows if row[2] is not None]
    return GPUSnapshot(
        free_vram_mb=min(free_vram) if free_vram else None,
        utilization_percent=max(utilization) if utilization else None,
        temperature_c=max(temperature) if temperature else None,
        xid_errors=_xid_error_codes(errors),
        wvm_active=wvm_state is True,
        probe_error=None
        if wvm_state is not None
        else "local process ownership state is unavailable",
    )


def _process_identity(pid: int) -> str | None:
    """Return a stable process identifier, or None when it cannot be determined.

    ``None`` (unknown) means ownership cannot be proven, which must fail
    closed; ``""`` means the process is gone, which is a determinate
    "not running" answer.
    """
    if sys.platform == "win32":
        return _windows_process_identity(pid)
    return _read_process_command(pid)


def _run_nvidia_smi(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=3)


def _completed_stdout(completed: object) -> str:
    stdout = getattr(completed, "stdout", None)
    if not isinstance(stdout, str):
        raise ValueError("GPU safety probe did not return text output")
    return stdout


def _gpu_metric_rows(output: str) -> tuple[tuple[int | None, int | None, int | None], ...]:
    rows: list[tuple[int | None, int | None, int | None]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 3:
            raise ValueError("GPU safety probe returned malformed metrics")
        free_vram, utilization, temperature = (
            _optional_nonnegative_int(column) for column in columns
        )
        if free_vram is None or utilization is None or temperature is None:
            raise ValueError("GPU safety probe returned incomplete metrics")
        rows.append((free_vram, utilization, temperature))
    if not rows:
        raise ValueError("GPU safety probe returned no metrics")
    return tuple(rows)


def _optional_nonnegative_int(value: str) -> int | None:
    if value in {"", "N/A", "[N/A]"}:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("GPU safety probe returned a negative metric")
    return number


def _gpu_processes(output: str) -> tuple[tuple[int, str], ...]:
    processes: list[tuple[int, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid_text, separator, process_name = line.partition(",")
        if not separator:
            raise ValueError("GPU safety probe returned malformed process data")
        processes.append((int(pid_text.strip()), process_name.strip()))
    return tuple(processes)


def _wvm_process_state(
    processes: tuple[tuple[int, str], ...],
    process_command: Callable[[int], str | None],
) -> bool | None:
    """Tri-state WVM ownership: True (active), False (absent), None (unknown)."""
    for pid, process_name in processes:
        identity = process_command(pid)
        if identity is None:
            return None
        if _is_wvm_process(pid, process_name, identity):
            return True
    return False


def _is_wvm_process(pid: int, process_name: str, identity: str) -> bool:
    candidate = f"{process_name} {identity}".casefold()
    return any(
        marker in candidate
        for marker in ("whisper-voice-machine", "whisper_voice_machine", "whisper voice machine")
    )


def _read_process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return ""


def _windows_process_identity(pid: int) -> str | None:
    """Windows process image path via QueryFullProcessImageNameW.

    Returns the image path when it can be read, ``""`` when the process no
    longer exists (ERROR_INVALID_PARAMETER), and ``None`` when the image
    cannot be determined (unknown, must fail closed).
    """
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = _win_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            return ""
        if error == _ERROR_ACCESS_DENIED:
            return None
        return None
    try:
        size = wintypes.DWORD(_MAX_IMAGE_PATH)
        buffer = ctypes.create_unicode_buffer(_MAX_IMAGE_PATH)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return buffer.value
    finally:
        _kernel32.CloseHandle(handle)


def _xid_error_codes(output: str) -> tuple[int, ...]:
    codes: list[int] = []
    for match in re.finditer(r"(?i)\bxid(?:\s+errors)?\s*:\s*([0-9,\s]+)", output):
        codes.extend(int(value) for value in re.findall(r"\d+", match.group(1)) if int(value) > 0)
    return tuple(dict.fromkeys(codes))
