from __future__ import annotations

import json
import os
import shutil
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
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    PINNED_AUDIO_CPP_CONTAINER_IMAGE,
)

_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_QWEN_TTS_FAMILY: Final = "qwen3-tts"
_MOUNTED_MODEL_PATH: Final = "/models/qwen3-tts"
_MOUNTED_REFERENCE_PATH: Final = "/input/reference.wav"
_OUTPUT_FILENAME: Final = "qwen3-tts.wav"
_REQUIRED_MODEL_PACKAGE_FILES: Final = (
    "model.safetensors",
    "config.json",
    "tokenizer_config.json",
)
_MODEL_ID_CONFIG_KEY: Final = "_name_or_path"
_MODEL_SIZE_CONFIG_KEY: Final = "tts_model_size"
_MODEL_TYPE_CONFIG_KEY: Final = "tts_model_type"
_QWEN_TTS_MODE_BY_MODEL_ID: Final = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "voice-clone",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "voice-clone",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "custom-voice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "voice-design",
}
_QWEN_TTS_MODEL_ID_BY_VARIANT: Final = {
    ("0b6", "base"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("1b7", "base"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("1b7", "custom_voice"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("1b7", "voice_design"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
_CHILD_ENVIRONMENT_KEYS: Final = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)


def validate_qwen_tts_model_package(model_package_path: Path) -> Path:
    """Return a resolved supported Safetensors package directory or raise ValueError."""

    try:
        package = model_package_path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Qwen3-TTS model package is unavailable") from exc
    if not package.is_dir():
        raise ValueError("Qwen3-TTS model package must be a directory")
    if any(not (package / filename).is_file() for filename in _REQUIRED_MODEL_PACKAGE_FILES):
        raise ValueError(
            "Qwen3-TTS model package is missing required Safetensors or tokenizer files"
        )
    if not (package / "speech_tokenizer").is_dir():
        raise ValueError("Qwen3-TTS model package is missing the speech_tokenizer package")
    _read_qwen_tts_model_identity(package)
    return package


def _read_qwen_tts_model_identity(package: Path) -> tuple[str, str]:
    """Read the supported exact Qwen3-TTS identity from package metadata."""

    try:
        metadata = json.loads((package / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Qwen3-TTS model package has invalid model metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Qwen3-TTS model package has invalid model metadata")
    declared_model_id = metadata.get(_MODEL_ID_CONFIG_KEY)
    if declared_model_id is not None and not isinstance(declared_model_id, str):
        raise ValueError("Qwen3-TTS model package has invalid model metadata")
    derived_model_id = _derive_qwen_tts_model_identity(metadata)
    if declared_model_id is None and derived_model_id is None:
        raise ValueError("Qwen3-TTS model package has invalid model metadata")
    if (
        declared_model_id is not None
        and derived_model_id is not None
        and declared_model_id != derived_model_id
    ):
        raise ValueError("Qwen3-TTS model package has conflicting model metadata")
    model_id = derived_model_id or declared_model_id
    assert model_id is not None
    mode = _QWEN_TTS_MODE_BY_MODEL_ID.get(model_id)
    if mode is None:
        raise ValueError("Qwen3-TTS model package has an unsupported model identity")
    return model_id, mode


def _derive_qwen_tts_model_identity(metadata: Mapping[str, object]) -> str | None:
    """Derive the exact supported identity when Qwen's variant fields are present."""

    model_size = metadata.get(_MODEL_SIZE_CONFIG_KEY)
    model_type = metadata.get(_MODEL_TYPE_CONFIG_KEY)
    if model_size is None and model_type is None:
        return None
    if not isinstance(model_size, str) or not isinstance(model_type, str):
        raise ValueError("Qwen3-TTS model package has invalid model metadata")
    model_id = _QWEN_TTS_MODEL_ID_BY_VARIANT.get((model_size, model_type))
    if model_id is None:
        raise ValueError("Qwen3-TTS model package has an unsupported model identity")
    return model_id


class AudioCppQwenTTSCLITransport:
    """Run Qwen3-TTS through the pinned Linux audio.cpp container CLI."""

    def __init__(
        self,
        *,
        model_package_path: Path,
        image: str = PINNED_AUDIO_CPP_CONTAINER_IMAGE,
        container_command: Sequence[str] = ("docker",),
        timeout_seconds: float = 300.0,
    ) -> None:
        if sys.platform.startswith("win"):
            raise ValueError("Qwen3-TTS container transport is unavailable on Windows")
        if image != PINNED_AUDIO_CPP_CONTAINER_IMAGE:
            raise ValueError("Qwen3-TTS container image must use the verified pinned digest")
        if not container_command or not all(
            part and "\x00" not in part for part in container_command
        ):
            raise ValueError("Qwen3-TTS container command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Qwen3-TTS container timeout must be positive")
        self._model_package_path = validate_qwen_tts_model_package(model_package_path)
        self._model_id, self._mode = _read_qwen_tts_model_identity(self._model_package_path)
        self._image = image
        self._container_command = tuple(container_command)
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
                output_directory = workspace / "output"
                output_directory.mkdir(mode=0o700)
                reference_path = self._stage_reference_audio(request, workspace)
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("Qwen3-TTS container invocation cancelled")
                    process = self._start_container(
                        self._build_command(
                            output_directory=output_directory,
                            request=request,
                            reference_path=reference_path,
                        )
                    )
                    self._processes[request_id] = process
                try:
                    process.communicate(timeout=self._timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    self._terminate_process(process)
                    raise RuntimeTransportError("Qwen3-TTS container invocation timed out") from exc
                with self._lock:
                    cancelled = request_id in self._cancelled
                if cancelled:
                    raise RuntimeTransportError("Qwen3-TTS container invocation cancelled")
                if process.returncode != 0:
                    raise RuntimeTransportError(
                        f"Qwen3-TTS container process exited with code {process.returncode}"
                    )
                self._validate_private_output(workspace, output_directory / _OUTPUT_FILENAME)
                return decode_audio_cpp_cli_response(
                    request_id=request_id,
                    family=_QWEN_TTS_FAMILY,
                    payload=request,
                    output_directory=output_directory,
                    wav_filename=_OUTPUT_FILENAME,
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

    def _decode_vop_request(self, payload: Mapping[str, object]) -> dict[str, object]:
        request = decode_audio_cpp_cli_request(payload)
        if request.get("family") != _QWEN_TTS_FAMILY:
            raise RuntimeProtocolError("Qwen3-TTS transport only supports the Qwen3-TTS family")
        raw_package_path = request.get("model_artifact_path")
        if not isinstance(raw_package_path, str) or not raw_package_path:
            raise RuntimeProtocolError("Qwen3-TTS request has no model package identity")
        try:
            request_package_path = Path(raw_package_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError("Qwen3-TTS request model package is unavailable") from exc
        if request_package_path != self._model_package_path:
            raise RuntimeProtocolError(
                "Qwen3-TTS request model package does not match the transport"
            )
        if request.get("model_id") != self._model_id or request.get("mode") != self._mode:
            raise RuntimeProtocolError(
                "Qwen3-TTS model package identity does not match the requested model and mode"
            )
        return request

    def _build_command(
        self,
        *,
        output_directory: Path,
        request: Mapping[str, object],
        reference_path: Path | None,
    ) -> tuple[str, ...]:
        command: list[str] = [
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
            self._readonly_mount(self._model_package_path, _MOUNTED_MODEL_PATH),
            "--mount",
            f"type=bind,src={output_directory},dst=/output",
        ]
        if reference_path is not None:
            command.extend(
                ("--mount", self._readonly_mount(reference_path, _MOUNTED_REFERENCE_PATH))
            )
        command.append(self._image)
        command.extend(
            build_audio_cpp_family_arguments(
                family=_QWEN_TTS_FAMILY,
                payload=request,
                model_argument=_MOUNTED_MODEL_PATH,
                output_directory=Path("/output"),
                reference_audio_argument=_MOUNTED_REFERENCE_PATH,
                wav_filename=_OUTPUT_FILENAME,
            )
        )
        return tuple(command)

    @staticmethod
    def _readonly_mount(source: Path, destination: str) -> str:
        return f"type=bind,src={source},dst={destination},readonly"

    @staticmethod
    def _stage_reference_audio(request: Mapping[str, object], workspace: Path) -> Path | None:
        if request.get("mode") != "voice-clone":
            return None
        raw_reference = request.get("reference_audio_path")
        if not isinstance(raw_reference, str) or not raw_reference:
            raise RuntimeProtocolError("Qwen3-TTS clone reference audio is unavailable")
        try:
            source = Path(raw_reference).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError("Qwen3-TTS clone reference audio is unavailable") from exc
        if not source.is_file():
            raise RuntimeProtocolError("Qwen3-TTS clone reference audio must be a regular file")
        staged = workspace / "reference.wav"
        try:
            shutil.copyfile(source, staged)
            os.chmod(staged, 0o600)
            with wave.open(str(staged), "rb") as audio:
                if audio.getnframes() <= 0:
                    raise RuntimeProtocolError(
                        "Qwen3-TTS clone reference audio contains no audio frames"
                    )
        except RuntimeProtocolError:
            raise
        except (EOFError, OSError, wave.Error) as exc:
            raise RuntimeProtocolError(
                "Qwen3-TTS clone reference audio is not a readable WAV file"
            ) from exc
        return staged

    @staticmethod
    def _validate_private_output(workspace: Path, output_path: Path) -> None:
        try:
            resolved_workspace = workspace.resolve(strict=True)
            resolved_candidate = output_path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError("Qwen3-TTS output audio is missing or invalid") from exc
        if not resolved_candidate.is_relative_to(resolved_workspace):
            raise RuntimeProtocolError("Qwen3-TTS output audio escaped the private workspace")
        try:
            resolved_output = output_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeProtocolError("Qwen3-TTS output audio is missing or invalid") from exc
        if not resolved_output.is_relative_to(resolved_workspace):
            raise RuntimeProtocolError("Qwen3-TTS output audio escaped the private workspace")
        if not resolved_output.is_file():
            raise RuntimeProtocolError("Qwen3-TTS output audio must be a regular file")

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
                shell=False,
            )
        except OSError as exc:
            raise RuntimeTransportError("Qwen3-TTS container process could not be started") from exc

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
