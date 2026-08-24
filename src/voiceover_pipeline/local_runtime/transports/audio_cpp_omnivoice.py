from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Final

from voiceover_pipeline.audio_cpp.inventory import find_family_inventory
from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    _stage_reference_audio,
    build_audio_cpp_family_arguments,
    decode_audio_cpp_cli_request,
    decode_audio_cpp_cli_response,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    PINNED_AUDIO_CPP_CONTAINER_IMAGE,
)

_PRIVATE_TMP_PREFIX: Final = "voiceover-audio-cpp-"
_OMNIVOICE_FAMILY: Final = "omnivoice"
_MOUNTED_MODEL_PATH: Final = "/models/omnivoice-q8_0.gguf"
_MOUNTED_REFERENCE_PATH: Final = "/input/reference.wav"
PINNED_AUDIO_CPP_OMNIVOICE_BINARY_SHA256: Final = (
    "d98b99f10355a018ddaec6d17999725ab7bdbcf5f164ab067c1288a15a4f51dd"
)
_CHILD_ENVIRONMENT_KEYS: Final = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)
_MODEL_HASH_CHUNK_BYTES: Final = 1024 * 1024
_DEFAULT_TIMEOUT_CHARS: Final = 420
_MAX_TIMEOUT_SECONDS: Final = 1800.0


@dataclass(frozen=True)
class VerifiedOmniVoiceModel:
    """Exact local artifact admission; public receipts intentionally omit local paths."""

    model_path: Path
    model_id: str
    sha256: str
    quantization: str
    license: str
    provenance: str

    def public_receipt(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "sha256": self.sha256,
            "quantization": self.quantization,
            "license": self.license,
            "provenance": self.provenance,
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_MODEL_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def admit_omnivoice_model(*, model_path: Path, model_id: str) -> VerifiedOmniVoiceModel:
    """Refuse every configured file except the exact inventory-pinned artifact."""
    inventory = find_family_inventory(_OMNIVOICE_FAMILY)
    if (
        model_id != inventory.model_id
        or inventory.model_sha256 is None
        or inventory.quantization is None
        or inventory.license is None
        or inventory.provenance is None
    ):
        raise ValueError("OmniVoice model identity is not admitted by the local inventory")
    if not model_path.is_file():
        raise ValueError("OmniVoice model artifact is unavailable")
    resolved_path = model_path.resolve()
    if _sha256_file(resolved_path) != inventory.model_sha256:
        raise ValueError("OmniVoice model artifact does not match the approved SHA-256")
    return VerifiedOmniVoiceModel(
        model_path=resolved_path,
        model_id=inventory.model_id,
        sha256=inventory.model_sha256,
        quantization=inventory.quantization,
        license=inventory.license,
        provenance=inventory.provenance,
    )


class AudioCppOmniVoiceCLITransport:
    """Run the pinned OmniVoice CLI through an isolated, immutable container."""

    def __init__(
        self,
        *,
        model: VerifiedOmniVoiceModel,
        image: str = PINNED_AUDIO_CPP_CONTAINER_IMAGE,
        container_command: Sequence[str] = ("docker",),
        timeout_seconds: float = 300.0,
        expected_sample_rate_hz: int = 24_000,
    ) -> None:
        if sys.platform.startswith("win"):
            raise ValueError("OmniVoice container transport is unavailable on Windows")
        if image != PINNED_AUDIO_CPP_CONTAINER_IMAGE:
            raise ValueError("OmniVoice container image must use the verified pinned digest")
        admitted_model = admit_omnivoice_model(
            model_path=model.model_path,
            model_id=model.model_id,
        )
        if not container_command or not all(
            part and "\x00" not in part for part in container_command
        ):
            raise ValueError("OmniVoice container command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OmniVoice container timeout must be positive")
        if expected_sample_rate_hz <= 0:
            raise ValueError("OmniVoice expected sample rate must be positive")
        self._model_path = admitted_model.model_path
        self._model_id = admitted_model.model_id
        self._image = image
        self._container_command = tuple(container_command)
        self._timeout_seconds = timeout_seconds
        self._expected_sample_rate_hz = expected_sample_rate_hz
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
                reference_path = _stage_reference_audio(
                    request, workspace, timeout_seconds=self._timeout_seconds
                )
                output_path = output_directory / "omnivoice.wav"
                with self._lock:
                    if request_id in self._cancelled:
                        raise RuntimeTransportError("OmniVoice invocation cancelled")
                    process = self._start_container(
                        self._build_command(
                            output_directory=output_directory,
                            request=request,
                            reference_path=reference_path,
                        )
                    )
                    self._processes[request_id] = process
                try:
                    process.communicate(timeout=self._timeout_for_request(request))
                except subprocess.TimeoutExpired as exc:
                    self._terminate_process(process)
                    raise RuntimeTransportError("OmniVoice invocation timed out") from exc
                with self._lock:
                    cancelled = request_id in self._cancelled
                if cancelled:
                    raise RuntimeTransportError("OmniVoice invocation cancelled")
                if process.returncode != 0:
                    raise RuntimeTransportError(
                        f"OmniVoice container process exited with code {process.returncode}"
                    )
                return self._encode_vop_response(request_id=request_id, output_path=output_path)
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
        if request.get("family") != _OMNIVOICE_FAMILY:
            raise RuntimeProtocolError("OmniVoice transport only supports the offline TTS family")
        if request.get("model_id") != self._model_id:
            raise RuntimeProtocolError("OmniVoice request has an unsupported model")
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
            self._readonly_mount(self._model_path, _MOUNTED_MODEL_PATH),
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
                family=_OMNIVOICE_FAMILY,
                payload=request,
                model_argument=_MOUNTED_MODEL_PATH,
                output_directory=Path("/output"),
                reference_audio_argument=(
                    _MOUNTED_REFERENCE_PATH if reference_path is not None else None
                ),
                wav_filename="omnivoice.wav",
            )
        )
        return tuple(command)

    def _timeout_for_request(self, request: Mapping[str, object]) -> float:
        text = request.get("text")
        text_length = len(text) if isinstance(text, str) else 0
        workload_units = max(1, ceil(text_length / _DEFAULT_TIMEOUT_CHARS))
        return min(_MAX_TIMEOUT_SECONDS, self._timeout_seconds * workload_units)

    @staticmethod
    def _readonly_mount(source: Path, destination: str) -> str:
        return f"type=bind,src={source},dst={destination},readonly"

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
            raise RuntimeTransportError("OmniVoice container process could not be started") from exc

    def _encode_vop_response(self, *, request_id: str, output_path: Path) -> Mapping[str, object]:
        return decode_audio_cpp_cli_response(
            request_id=request_id,
            family=_OMNIVOICE_FAMILY,
            payload={},
            output_directory=output_path.parent,
            wav_filename=output_path.name,
            expected_sample_rate_hz=self._expected_sample_rate_hz,
        )

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
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeTransportError(
                    "OmniVoice container process could not be terminated"
                ) from exc
