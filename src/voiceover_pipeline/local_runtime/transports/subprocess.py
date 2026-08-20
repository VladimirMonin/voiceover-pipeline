from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Final

from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError

_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_MAX_TTS_WAV_BYTES: Final = 64 * 1024 * 1024
_CHILD_ENVIRONMENT_KEYS: Final = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)


class SubprocessJSONTransport:
    """Per-request JSON transport with no shell and cancellable process groups."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 60.0,
        host_platform: str | None = None,
    ) -> None:
        if not command or not all(command):
            raise ValueError("audio.cpp transport command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("audio.cpp transport timeout must be positive")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds
        self._host_platform = host_platform or sys.platform
        self._lock = RLock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()

    def invoke(self, request_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with tempfile.TemporaryDirectory(prefix=_PRIVATE_TMP_PREFIX) as temporary_directory:
            if not self._is_windows:
                os.chmod(temporary_directory, 0o700)
            environment = {
                name: value
                for name in _CHILD_ENVIRONMENT_KEYS
                if (value := os.environ.get(name)) is not None
            }
            environment["VOICEOVER_PIPELINE_RUNTIME_TMPDIR"] = temporary_directory
            process = self._start_process(temporary_directory, environment)
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
                raise RuntimeTransportError(
                    f"audio.cpp process exited with code {process.returncode}"
                )
            decoded = self._decode_response(stdout)
            return self._materialize_tts_wav_response(
                decoded, request=payload, temporary_directory=temporary_directory
            )

    @staticmethod
    def _decode_response(stdout: str) -> Mapping[str, object]:
        try:
            decoded = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeProtocolError("audio.cpp response is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeProtocolError("audio.cpp response must be a JSON object")
        return decoded

    @classmethod
    def _materialize_tts_wav_response(
        cls,
        response: Mapping[str, object],
        *,
        request: Mapping[str, object],
        temporary_directory: str,
    ) -> Mapping[str, object]:
        if request.get("operation") != "tts":
            return response
        raw_payload = response.get("response")
        if not isinstance(raw_payload, Mapping):
            return response
        raw_audio_path = raw_payload.get("audio_path")
        if raw_audio_path is None:
            return response
        if not isinstance(raw_audio_path, str) or not raw_audio_path:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio_path must be a non-empty string"
            )
        audio_bytes = cls._read_private_tts_wav(temporary_directory, raw_audio_path)
        materialized_payload = dict(raw_payload)
        materialized_payload.pop("audio_path")
        materialized_payload["audio_bytes"] = audio_bytes
        materialized_payload["audio_format"] = "wav"
        materialized_response = dict(response)
        materialized_response["response"] = materialized_payload
        return materialized_response

    @staticmethod
    def _read_private_tts_wav(temporary_directory: str, raw_audio_path: str) -> bytes:
        workspace = Path(temporary_directory).resolve(strict=True)
        declared_path = Path(raw_audio_path)
        candidate = declared_path if declared_path.is_absolute() else workspace / declared_path
        try:
            resolved_candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact is missing or invalid"
            ) from exc
        if not resolved_candidate.is_relative_to(workspace):
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact escaped the private workspace"
            )
        try:
            artifact_path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact is missing or invalid"
            ) from exc
        if not artifact_path.is_relative_to(workspace):
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact escaped the private workspace"
            )
        if not artifact_path.is_file():
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact must be a non-empty regular file"
            )
        try:
            artifact_size = artifact_path.stat().st_size
        except OSError as exc:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact is missing or invalid"
            ) from exc
        if artifact_size <= 0:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact must be a non-empty regular file"
            )
        if artifact_size > _MAX_TTS_WAV_BYTES:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact exceeds the transport limit"
            )
        try:
            audio_bytes = artifact_path.read_bytes()
        except OSError as exc:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact could not be read"
            ) from exc
        if len(audio_bytes) != artifact_size or not audio_bytes:
            raise RuntimeProtocolError(
                "audio.cpp TTS response audio artifact is missing or invalid"
            )
        return audio_bytes

    @property
    def _is_windows(self) -> bool:
        return self._host_platform.startswith("win")

    def _start_process(
        self, temporary_directory: str, environment: Mapping[str, str]
    ) -> subprocess.Popen[str]:
        try:
            if self._is_windows:
                return subprocess.Popen(
                    self._command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=temporary_directory,
                    env=environment,
                    shell=False,
                    creationflags=int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)),
                )
            return subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=temporary_directory,
                env=environment,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeTransportError("audio.cpp process could not be started") from exc

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

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if self._is_windows:
            control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if control_break is not None:
                try:
                    process.send_signal(control_break)
                except (OSError, ValueError):
                    pass
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    return
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    return
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
