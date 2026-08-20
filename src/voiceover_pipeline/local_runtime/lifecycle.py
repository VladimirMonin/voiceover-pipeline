from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voiceover_pipeline.local_runtime.contracts import (
    LocalAudioRuntimeDriver,
    LocalRuntimeRequest,
    LocalRuntimeResponse,
)
from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager


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
    process_command: Callable[[int], str] | None = None,
) -> GPUSnapshot:
    """Read only local GPU safety state; never retain process details or command output."""
    run = runner or _run_nvidia_smi
    process_text = process_command or _read_process_command
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
        wvm_active=any(
            _is_wvm_process(pid, name, process_text) for pid, name in _gpu_processes(applications)
        ),
    )


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


def _read_process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return ""


def _is_wvm_process(pid: int, process_name: str, process_command: Callable[[int], str]) -> bool:
    candidate = f"{process_name} {process_command(pid)}".casefold()
    return any(
        marker in candidate
        for marker in ("whisper-voice-machine", "whisper_voice_machine", "whisper voice machine")
    )


def _xid_error_codes(output: str) -> tuple[int, ...]:
    codes: list[int] = []
    for match in re.finditer(r"(?i)\bxid(?:\s+errors)?\s*:\s*([0-9,\s]+)", output):
        codes.extend(int(value) for value in re.findall(r"\d+", match.group(1)) if int(value) > 0)
    return tuple(dict.fromkeys(codes))
