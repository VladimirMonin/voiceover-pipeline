from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from shutil import which
from typing import Any, Literal
from uuid import uuid4

from voiceover_pipeline.audio_cpp.inventory import PINNED_AUDIO_CPP_REVISION
from voiceover_pipeline.config import (
    QWEN_INSTRUCT,
    QWEN_LANGUAGE,
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)
from voiceover_pipeline.local_runtime.contracts import (
    LocalTTSRequest,
    RuntimeDriverHealth,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.drivers.audio_cpp import AudioCppRuntimeDriver
from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleOwner, probe_local_gpu_state
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry
from voiceover_pipeline.local_runtime.transports.audio_cpp_qwen_tts import (
    AudioCppQwenTTSCLITransport,
    validate_qwen_tts_model_package,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider

QWEN_TTS_FAMILY = "qwen3-tts"
QWEN_TTS_PROVIDER_ID = "qwen-local"
QwenTTSRuntimeMode = Literal["custom-voice", "voice-clone", "voice-design"]
_QWEN_TTS_MODEL_ENV = "VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL"
_QWEN_TTS_CONTAINER_COMMAND_ENV = "VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON"
_QWEN_TTS_MIN_FREE_VRAM_MB = 4096
_QWEN_TTS_MAX_GPU_UTILIZATION_PERCENT = 90
AUDIO_CPP_QWEN_TTS_INSTALL_REMEDIATION = (
    "audio.cpp Qwen TTS runtime is unavailable. Set VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL to "
    "an installed Safetensors package and configure a local container command before retrying."
)


def _container_command_from_environment() -> tuple[str, ...]:
    raw = os.environ.get(_QWEN_TTS_CONTAINER_COMMAND_ENV, "").strip()
    if not raw:
        return ("docker",)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_QWEN_TTS_CONTAINER_COMMAND_ENV} must be a JSON array of argv strings"
        ) from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(part, str) and part and "\x00" not in part for part in parsed)
    ):
        raise ValueError(f"{_QWEN_TTS_CONTAINER_COMMAND_ENV} must be a JSON array of argv strings")
    return tuple(parsed)


class AudioCppQwenTTSProvider(TTSProvider):
    """Qwen3-TTS route over the runtime-neutral audio.cpp JSON driver seam."""

    provider_id = QWEN_TTS_PROVIDER_ID

    def __init__(
        self,
        runtime: LocalAudioRuntime | Any | None = None,
        *,
        mode: str = "preset",
        voice: str | None = None,
        instruct: str = QWEN_INSTRUCT,
        language: str = QWEN_LANGUAGE,
        sample_path: str | None = None,
        sample_text: str = "",
        model_artifact_path: Path | str | None = None,
    ) -> None:
        self._runtime = runtime
        self._mode = mode
        self._voice = voice
        self._instruct = instruct
        self._language = language
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._model_artifact_path = model_artifact_path

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "AudioCppQwenTTSProvider":
        health = qwen_tts_audio_cpp_dependency_probe()
        if not health.available:
            return cls(**kwargs)
        model_package_path = validate_qwen_tts_model_package(
            Path(os.environ[_QWEN_TTS_MODEL_ENV].strip())
        )
        driver = AudioCppRuntimeDriver(
            binary_path=None,
            source_revision=PINNED_AUDIO_CPP_REVISION,
            transport=AudioCppQwenTTSCLITransport(
                model_package_path=model_package_path,
                container_command=_container_command_from_environment(),
            ),
            transport_name="container-cli",
        )
        return cls(
            LocalAudioRuntime(
                LocalRuntimeRegistry((driver,)),
                promoted_families=(QWEN_TTS_FAMILY,),
                lifecycle=_qwen_tts_gpu_lifecycle(),
            ),
            model_artifact_path=model_package_path,
            **kwargs,
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        if self._runtime is None:
            raise ModuleNotFoundError(AUDIO_CPP_QWEN_TTS_INSTALL_REMEDIATION)
        model_id, runtime_mode, voice, instruction, reference_audio_path, reference_text = (
            self._mode_request_fields()
        )
        response = self._runtime.execute_tts(
            LocalTTSRequest(
                request_id=uuid4().hex,
                family=QWEN_TTS_FAMILY,
                provider_id=self.provider_id,
                text=text,
                model_id=model_id,
                model_artifact_path=self._model_artifact_path,
                voice=voice,
                language=self._language,
                mode=runtime_mode,
                instruction=instruction,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
            ),
            runtime_choice="auto",
        )
        audio_bytes, sample_rate_hz = _validated_mono_wav(
            response.audio_bytes, response.audio_format
        )
        metadata: dict[str, Any] = {
            "provider": self.provider_id,
            "family": QWEN_TTS_FAMILY,
            "model_id": model_id,
            "mode": self._mode,
            "voice": voice if self._mode == "preset" else self._mode,
            "sample_rate_hz": sample_rate_hz,
        }
        if response.receipt is not None:
            metadata["runtime"] = {
                "driver_id": response.receipt.driver_id,
                "transport": response.receipt.transport,
                "source_revision": response.receipt.source_revision,
                "build_hash": response.receipt.build_hash,
            }
        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format="wav",
            transcript=text,
            client_path=self.provider_id,
            raw_metadata=metadata,
        )

    def _mode_request_fields(
        self,
    ) -> tuple[str, QwenTTSRuntimeMode, str | None, str | None, str | None, str | None]:
        if self._mode == "preset":
            return (
                QWEN_MODEL_CUSTOMVOICE,
                "custom-voice",
                self._voice or "Aiden",
                self._instruct,
                None,
                None,
            )
        if self._mode == "clone":
            if not self._sample_path or not Path(self._sample_path).is_file():
                raise FileNotFoundError(
                    f"Reference audio not found for clone mode: {self._sample_path}"
                )
            if not self._sample_text.strip():
                raise ValueError("Qwen3-TTS clone mode requires non-empty reference text")
            return (
                QWEN_MODEL_BASE,
                "voice-clone",
                None,
                None,
                self._sample_path,
                self._sample_text or None,
            )
        if self._mode == "design":
            if not self._instruct.strip():
                raise ValueError("VoiceDesign mode requires a non-empty --qwen-instruct value")
            return (
                QWEN_MODEL_VOICE_DESIGN,
                "voice-design",
                None,
                self._instruct,
                None,
                None,
            )
        raise ValueError(f"Unknown qwen mode: {self._mode}")


def qwen_tts_audio_cpp_dependency_probe() -> RuntimeDriverHealth:
    model_path = os.environ.get(_QWEN_TTS_MODEL_ENV, "").strip()
    if sys.platform.startswith("win") or not model_path:
        return RuntimeDriverHealth(
            available=False,
            remediation=AUDIO_CPP_QWEN_TTS_INSTALL_REMEDIATION,
        )
    try:
        validate_qwen_tts_model_package(Path(model_path))
        command = _container_command_from_environment()
    except ValueError:
        return RuntimeDriverHealth(
            available=False,
            remediation=AUDIO_CPP_QWEN_TTS_INSTALL_REMEDIATION,
        )
    if which(command[0]) is None:
        return RuntimeDriverHealth(
            available=False,
            remediation=AUDIO_CPP_QWEN_TTS_INSTALL_REMEDIATION,
        )
    return RuntimeDriverHealth(available=True)


def _qwen_tts_gpu_lifecycle() -> GPULifecycleOwner:
    return GPULifecycleOwner(
        GPULeaseManager(
            metadata_path=Path(tempfile.gettempdir()) / "voiceover-pipeline-gpu-lease.json"
        ),
        probe=probe_local_gpu_state,
        min_free_vram_mb=_QWEN_TTS_MIN_FREE_VRAM_MB,
        max_utilization_percent=_QWEN_TTS_MAX_GPU_UTILIZATION_PERCENT,
    )


def _validated_mono_wav(audio_bytes: bytes | None, audio_format: str | None) -> tuple[bytes, int]:
    if audio_bytes is None or audio_format != "wav":
        raise RuntimeUnavailableError("Qwen audio.cpp runtime did not return WAV audio")
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate_hz = audio.getframerate()
            frames = audio.getnframes()
    except (EOFError, ValueError, wave.Error) as exc:
        raise RuntimeUnavailableError(
            "Qwen audio.cpp runtime did not return valid WAV audio"
        ) from exc
    if channels != 1 or sample_rate_hz <= 0 or frames <= 0:
        raise RuntimeUnavailableError("Qwen audio.cpp runtime returned an invalid mono WAV")
    return audio_bytes, sample_rate_hz
