from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Final

from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    build_audio_cpp_family_arguments,
    decode_audio_cpp_cli_request,
    decode_audio_cpp_cli_response,
    decode_audio_cpp_words,
)

PINNED_AUDIO_CPP_CONTAINER_IMAGE: Final = (
    "ghcr.io/0xshug0/audio.cpp@"
    "sha256:b46770ff33321ad187329659eb38ef22a5ae2bc6a8295f00a4f3b785b4211e58"
)
PINNED_AUDIO_CPP_CONTAINER_TAG: Final = "ghcr.io/0xshug0/audio.cpp:full-cuda13-20260816-502b5b7"
_QWEN_ASR_FAMILY: Final = "qwen3-asr"
_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_STAGED_AUDIO_RATE_HZ: Final = 16_000
_CHILD_ENVIRONMENT_KEYS: Final = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)


class AudioCppContainerCLITransport:
    """Convert the runtime-neutral Qwen ASR envelope to the pinned container CLI."""

    def __init__(
        self,
        *,
        asr_model_path: Path,
        forced_aligner_model_path: Path,
        image: str = PINNED_AUDIO_CPP_CONTAINER_IMAGE,
        container_command: Sequence[str] = ("docker",),
        ffmpeg_command: Sequence[str] = ("ffmpeg",),
        timeout_seconds: float = 300.0,
    ) -> None:
        if sys.platform.startswith("win"):
            raise ValueError("audio.cpp container transport is unavailable on Windows")
        if image != PINNED_AUDIO_CPP_CONTAINER_IMAGE:
            raise ValueError("audio.cpp container image must use the verified pinned digest")
        if not container_command or not all(container_command):
            raise ValueError("audio.cpp container command must not be empty")
        if not ffmpeg_command or not all(ffmpeg_command):
            raise ValueError("audio.cpp ffmpeg command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("audio.cpp container timeout must be positive")
        if not asr_model_path.is_file() or not forced_aligner_model_path.is_file():
            raise ValueError("audio.cpp Qwen model and forced aligner files must exist")
        self._asr_model_path = asr_model_path.resolve()
        self._forced_aligner_model_path = forced_aligner_model_path.resolve()
        self._image = image
        self._container_command = tuple(container_command)
        self._ffmpeg_command = tuple(ffmpeg_command)
        self._timeout_seconds = timeout_seconds
        self._lock = RLock()
        self._processes: dict[str, subprocess.Popen[bytes] | None] = {}
        self._cancelled: set[str] = set()

    def invoke(self, request_id: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = self._decode_vop_request(payload)
        with self._lock:
            self._processes[request_id] = None
        try:
            with tempfile.TemporaryDirectory(prefix=_PRIVATE_TMP_PREFIX) as temporary_directory:
                workspace = Path(temporary_directory)
                os.chmod(workspace, 0o700)
                staged_audio_path = workspace / "input.wav"
                output_directory = workspace / "output"
                output_directory.mkdir(mode=0o700)
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("audio.cpp container invocation cancelled")
                audio_path = request["audio_path"]
                assert isinstance(audio_path, str)
                duration_s = self._stage_audio(Path(audio_path), staged_audio_path)
                command = self._build_command(
                    staged_audio_path=staged_audio_path,
                    output_directory=output_directory,
                    request=request,
                )
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("audio.cpp container invocation cancelled")
                    process = self._start_container(command)
                    self._processes[request_id] = process
                try:
                    process.communicate(timeout=self._timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    self._terminate_process(process)
                    raise RuntimeTransportError("audio.cpp container invocation timed out") from exc
                with self._lock:
                    cancelled = request_id in self._cancelled
                if cancelled:
                    raise RuntimeTransportError("audio.cpp container invocation cancelled")
                if process.returncode != 0:
                    raise RuntimeTransportError(
                        f"audio.cpp container process exited with code {process.returncode}"
                    )
                return self._encode_vop_response(
                    request_id=request_id,
                    output_directory=output_directory,
                    timestamp_mode=request["timestamp_mode"],
                    duration_s=duration_s,
                )
        finally:
            with self._lock:
                self._processes.pop(request_id, None)
                self._cancelled.discard(request_id)

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
    def _decode_vop_request(payload: Mapping[str, object]) -> dict[str, object]:
        request = decode_audio_cpp_cli_request(payload)
        if request.get("family") != _QWEN_ASR_FAMILY:
            raise RuntimeProtocolError("audio.cpp container only supports the Qwen ASR route")
        return request

    def _build_command(
        self,
        *,
        staged_audio_path: Path,
        output_directory: Path,
        request: Mapping[str, object],
    ) -> tuple[str, ...]:
        command = [
            *self._container_command,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--gpus",
            "all",
            "--entrypoint",
            "/app/audiocpp_cli",
            "--mount",
            self._readonly_mount(self._asr_model_path, "/models/qwen3-asr.gguf"),
            "--mount",
            self._readonly_mount(staged_audio_path, "/input/audio.wav"),
            "--mount",
            f"type=bind,src={output_directory},dst=/output",
        ]
        if request["timestamp_mode"] == "word":
            command.extend(
                [
                    "--mount",
                    self._readonly_mount(
                        self._forced_aligner_model_path, "/models/qwen3-forced-aligner.gguf"
                    ),
                ]
            )
        command.append(self._image)
        command.extend(
            build_audio_cpp_family_arguments(
                family=_QWEN_ASR_FAMILY,
                payload=request,
                model_argument="/models/qwen3-asr.gguf",
                output_directory=Path("/output"),
                audio_argument="/input/audio.wav",
                forced_aligner_argument="/models/qwen3-forced-aligner.gguf",
            )
        )
        return tuple(command)

    @staticmethod
    def _readonly_mount(source: Path, destination: str) -> str:
        return f"type=bind,src={source},dst={destination},readonly"

    def _stage_audio(self, source: Path, staged_audio_path: Path) -> float:
        if not source.is_file():
            raise RuntimeTransportError("audio.cpp input audio is unavailable")
        try:
            completed = subprocess.run(
                [
                    *self._ffmpeg_command,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ar",
                    str(_STAGED_AUDIO_RATE_HZ),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(staged_audio_path),
                ],
                check=False,
                cwd=staged_audio_path.parent,
                env=self._child_environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeTransportError("audio.cpp input preparation failed") from exc
        if completed.returncode != 0:
            raise RuntimeTransportError("audio.cpp input preparation failed")
        try:
            with wave.open(str(staged_audio_path), "rb") as audio:
                if audio.getnchannels() != 1 or audio.getframerate() != _STAGED_AUDIO_RATE_HZ:
                    raise RuntimeProtocolError(
                        "audio.cpp input staging produced an invalid WAV format"
                    )
                return audio.getnframes() / audio.getframerate()
        except (OSError, wave.Error) as exc:
            raise RuntimeProtocolError(
                "audio.cpp input staging did not produce a readable WAV"
            ) from exc

    def _start_container(self, command: Sequence[str]) -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd="/",
                env=self._child_environment(),
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeTransportError("audio.cpp container process could not be started") from exc

    @staticmethod
    def _encode_vop_response(
        *,
        request_id: str,
        output_directory: Path,
        timestamp_mode: object,
        duration_s: float,
    ) -> Mapping[str, object]:
        encoded = decode_audio_cpp_cli_response(
            request_id=request_id,
            family=_QWEN_ASR_FAMILY,
            payload={"timestamp_mode": timestamp_mode},
            output_directory=output_directory,
        )
        response = encoded["response"]
        assert isinstance(response, Mapping)
        return {
            "schema_version": 1,
            "request_id": request_id,
            "ok": True,
            "response": {**response, "duration_s": duration_s},
        }

    @staticmethod
    def _decode_words(words_path: Path) -> list[dict[str, object]]:
        return decode_audio_cpp_words(words_path)

    @staticmethod
    def _child_environment() -> dict[str, str]:
        return {
            name: value
            for name in _CHILD_ENVIRONMENT_KEYS
            if (value := os.environ.get(name)) is not None
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
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
