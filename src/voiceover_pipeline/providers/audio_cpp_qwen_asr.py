from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from voiceover_pipeline.local_runtime.contracts import (
    LocalASRRequest,
    LocalASRResponse,
    LocalRuntimeResponse,
    RuntimeExecutionReceipt,
)
from voiceover_pipeline.local_runtime.drivers.audio_cpp import (
    PINNED_AUDIO_CPP_REVISION,
    AudioCppRuntimeDriver,
)
from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleOwner, probe_local_gpu_state
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    AudioCppNativeCLITransport,
    discover_native_audio_cpp_install,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    AudioCppContainerCLITransport,
)
from voiceover_pipeline.models import (
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth
from voiceover_pipeline.providers.base import ASRProvider, validate_asr_response
from voiceover_pipeline.providers.qwen_asr_local import (
    QWEN_ASR_MODEL_ID,
    QWEN_ASR_PROVIDER_ID,
    _qwen_language_name,
)

QWEN_ASR_FAMILY = "qwen3-asr"
QWEN_AUDIO_CPP_MIN_FREE_VRAM_MB = 4096
QWEN_AUDIO_CPP_MAX_GPU_UTILIZATION_PERCENT = 90
AUDIO_CPP_QWEN_INSTALL_REMEDIATION = (
    "audio.cpp Qwen ASR runtime is unavailable. Set VOICEOVER_AUDIO_CPP_BINARY to a pinned "
    "JSON driver or configure the verified audio.cpp container image and model paths before retrying."
)
_CONTAINER_COMMAND_JSON_ENV = "VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON"


def _container_command_from_environment() -> tuple[str, ...]:
    raw = os.environ.get(_CONTAINER_COMMAND_JSON_ENV, "").strip()
    if not raw:
        return ("docker",)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_CONTAINER_COMMAND_JSON_ENV} must be a JSON array of argv strings"
        ) from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(part, str) and part and "\x00" not in part for part in parsed)
    ):
        raise ValueError(f"{_CONTAINER_COMMAND_JSON_ENV} must be a JSON array of argv strings")
    return tuple(parsed)


class AudioCppQwenASRProvider(ASRProvider):
    """Qwen3 ASR adapter with an explicit forced-aligner route for word timing."""

    provider_id = QWEN_ASR_PROVIDER_ID

    def __init__(self, runtime: LocalAudioRuntime | Any | None = None) -> None:
        self._runtime = runtime

    @classmethod
    def from_environment(cls) -> "AudioCppQwenASRProvider":
        binary = os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        container_image = os.environ.get("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", "").strip()
        asr_model = os.environ.get("VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL", "").strip()
        forced_aligner = os.environ.get("VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL", "").strip()
        if sys.platform.startswith("win"):
            native_executable = os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip()
            if not native_executable or not asr_model or not forced_aligner:
                return cls()
            try:
                install = discover_native_audio_cpp_install(
                    Path(native_executable),
                    required_model_paths=(Path(asr_model), Path(forced_aligner)),
                )
                transport = AudioCppNativeCLITransport(
                    executable_path=install.executable_path,
                    model_paths={
                        QWEN_ASR_FAMILY: Path(asr_model),
                        "qwen3-forced-aligner": Path(forced_aligner),
                    },
                )
            except ValueError:
                return cls()
            driver = AudioCppRuntimeDriver(
                binary_path=install.executable_path,
                source_revision=PINNED_AUDIO_CPP_REVISION,
                transport=transport,
                build_hash=install.files[install.executable_path.name],
                transport_name="native-cli",
            )
            return cls(
                LocalAudioRuntime(
                    LocalRuntimeRegistry((driver,)),
                    promoted_families=(QWEN_ASR_FAMILY,),
                    lifecycle=_audio_cpp_gpu_lifecycle(),
                )
            )
        if container_image:
            if not asr_model or not forced_aligner:
                return cls()
            container_command = _container_command_from_environment()
            driver = AudioCppRuntimeDriver(
                binary_path=None,
                source_revision=PINNED_AUDIO_CPP_REVISION,
                transport=AudioCppContainerCLITransport(
                    image=container_image,
                    asr_model_path=Path(asr_model),
                    forced_aligner_model_path=Path(forced_aligner),
                    container_command=container_command,
                ),
                transport_name="container-cli",
            )
            return cls(
                LocalAudioRuntime(
                    LocalRuntimeRegistry((driver,)),
                    promoted_families=(QWEN_ASR_FAMILY,),
                    lifecycle=_audio_cpp_gpu_lifecycle(),
                )
            )
        if not binary:
            return cls()
        driver = AudioCppRuntimeDriver(
            binary_path=Path(binary),
            source_revision=PINNED_AUDIO_CPP_REVISION,
        )
        return cls(
            LocalAudioRuntime(
                LocalRuntimeRegistry((driver,)),
                promoted_families=(QWEN_ASR_FAMILY,),
                lifecycle=_audio_cpp_gpu_lifecycle(),
            )
        )

    def transcribe(self, request: ASRRequest) -> ASRResult:
        if self._runtime is None:
            raise ModuleNotFoundError(AUDIO_CPP_QWEN_INSTALL_REMEDIATION)
        model_id = request.model_id or QWEN_ASR_MODEL_ID
        local_request = LocalASRRequest(
            request_id=uuid4().hex,
            family=QWEN_ASR_FAMILY,
            provider_id=self.provider_id,
            audio_path=request.audio_path,
            model_id=model_id,
            language=_qwen_language_name(request.language),
            timestamp_mode=request.timestamp_mode,
            context_text=request.hints.context_text,
        )
        response = self._runtime.execute(local_request.to_runtime_request(), runtime_choice="auto")
        if not isinstance(response, LocalRuntimeResponse):
            raise ValueError("audio.cpp Qwen ASR runtime returned an invalid response")
        local_response = LocalASRResponse.from_runtime_response(response)
        payload = local_response.payload
        duration_s = _optional_finite_number(payload, "duration_s")

        if request.timestamp_mode == "word":
            words = _forced_words(payload, transcript=local_response.transcript)
            result = ASRResult(
                transcript=local_response.transcript,
                provider_id=self.provider_id,
                model_id=model_id,
                language=local_response.language or request.language or "",
                duration_s=duration_s,
                words=words,
                alignment_origin="forced",
                execution=_execution_receipt(response.receipt, request),
            )
        else:
            result = ASRResult(
                transcript=local_response.transcript,
                provider_id=self.provider_id,
                model_id=model_id,
                language=local_response.language or request.language or "",
                duration_s=duration_s,
                execution=_execution_receipt(response.receipt, request),
            )
        return validate_asr_response(request, result)


def _audio_cpp_gpu_lifecycle() -> GPULifecycleOwner:
    return GPULifecycleOwner(
        GPULeaseManager(
            metadata_path=Path(tempfile.gettempdir()) / "voiceover-pipeline-gpu-lease.json"
        ),
        probe=probe_local_gpu_state,
        min_free_vram_mb=QWEN_AUDIO_CPP_MIN_FREE_VRAM_MB,
        max_utilization_percent=QWEN_AUDIO_CPP_MAX_GPU_UTILIZATION_PERCENT,
    )


def _forced_words(payload: Mapping[str, object], *, transcript: str) -> tuple[ASRWordSpan, ...]:
    if payload.get("forced_aligner_available") is False:
        raise ValueError(
            "Qwen word timestamps require Qwen3-ForcedAligner-0.6B; install or configure the aligner before retrying."
        )
    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        raise ValueError("Qwen forced alignment response must contain a words list")
    if not raw_words:
        if transcript.strip():
            raise ValueError("Qwen forced alignment returned no words for speech output")
        return ()
    response_offset = _finite_number(payload.get("chunk_offset_s", 0.0), "chunk_offset_s")
    words: list[ASRWordSpan] = []
    for index, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, Mapping):
            raise ValueError(f"Qwen forced alignment word {index} must be an object")
        text = raw_word.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Qwen forced alignment word {index} has no text")
        offset = response_offset + _finite_number(
            raw_word.get("chunk_offset_s", 0.0), f"word {index} chunk_offset_s"
        )
        start = _finite_number(raw_word.get("start_s"), f"word {index} start_s") + offset
        end = _finite_number(raw_word.get("end_s"), f"word {index} end_s") + offset
        confidence = raw_word.get("confidence")
        if confidence == 0.0:
            confidence = None
        if confidence is not None:
            confidence = _finite_number(confidence, f"word {index} confidence")
        try:
            words.append(ASRWordSpan(text=text, start_s=start, end_s=end, confidence=confidence))
        except ValueError as exc:
            raise ValueError(f"Qwen forced alignment word {index} is invalid: {exc}") from exc
    return tuple(words)


def _optional_finite_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _finite_number(value, key)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"Qwen forced alignment {label} must be a finite number")
    return float(value)


def _execution_receipt(
    receipt: RuntimeExecutionReceipt | None, request: ASRRequest
) -> ASRExecutionReceipt:
    return ASRExecutionReceipt(
        runtime=receipt.driver_id if receipt is not None else "audio-cpp",
        runtime_version=receipt.source_revision if receipt is not None else None,
        model_revision=receipt.source_revision if receipt is not None else None,
        resolved_device=request.device,
        resolved_compute=request.compute,
    )


def audio_cpp_qwen_asr_dependency_probe() -> ASRDependencyHealth:
    provider = AudioCppQwenASRProvider.from_environment()
    if provider._runtime is None:
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_QWEN_INSTALL_REMEDIATION,
            reason_code="invalid_native_package",
        )
    return ASRDependencyHealth(available=True, remediation="")
