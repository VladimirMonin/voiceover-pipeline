from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Final

from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError


_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_CHILD_ENVIRONMENT_KEYS: Final = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)


class SubprocessJSONTransport:
    """Per-request JSON transport with no shell and cancellable process groups."""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 60.0) -> None:
        if not command or not all(command):
            raise ValueError("audio.cpp transport command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("audio.cpp transport timeout must be positive")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds
        self._lock = RLock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()

    def invoke(self, request_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with tempfile.TemporaryDirectory(prefix=_PRIVATE_TMP_PREFIX) as temporary_directory:
            os.chmod(temporary_directory, 0o700)
            environment = {
                name: value
                for name in _CHILD_ENVIRONMENT_KEYS
                if (value := os.environ.get(name)) is not None
            }
            environment["VOICEOVER_PIPELINE_RUNTIME_TMPDIR"] = temporary_directory
            try:
                process = subprocess.Popen(
                    self._command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temporary_directory,
                    env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeTransportError("audio.cpp process could not be started") from exc
            with self._lock:
                self._processes[request_id] = process
            try:
                stdout, _stderr = process.communicate(encoded, timeout=self._timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process(process)
                raise RuntimeTransportError("audio.cpp invocation timed out") from exc
            finally:
                with self._lock:
                    self._processes.pop(request_id, None)
            with self._lock:
                cancelled = request_id in self._cancelled
                self._cancelled.discard(request_id)
            if cancelled:
                raise RuntimeTransportError("audio.cpp invocation cancelled")
            if process.returncode != 0:
                raise RuntimeTransportError(f"audio.cpp process exited with code {process.returncode}")
        try:
            decoded = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeProtocolError("audio.cpp response is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeProtocolError("audio.cpp response must be a JSON object")
        return decoded

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._cancelled.add(request_id)
            process = self._processes.get(request_id)
        if process is not None:
            self._terminate_process(process)

    def close(self) -> None:
        with self._lock:
            request_ids = tuple(self._processes)
        for request_id in request_ids:
            self.cancel(request_id)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
