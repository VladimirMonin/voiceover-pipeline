from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from voiceover_pipeline.audio_cpp.nemotron_words import normalize_nemotron_word_timestamps
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
from voiceover_pipeline.models import ASRContextHints, ASRExecutionReceipt, ASRRequest, ASRResult
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth
from voiceover_pipeline.providers.base import ASRProvider, validate_asr_response
from voiceover_pipeline.providers.nemotron_asr_local import (
    NEMOTRON_ASR_MODEL_ID,
    NEMOTRON_ASR_PROVIDER_ID,
)

NEMOTRON_ASR_FAMILY = "nemotron-3.5-asr"
NEMOTRON_AUDIO_CPP_MIN_FREE_VRAM_MB = 4096
NEMOTRON_AUDIO_CPP_MAX_GPU_UTILIZATION_PERCENT = 90
AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION = "audio.cpp Nemotron ASR runtime is unavailable. Set VOICEOVER_AUDIO_CPP_BINARY to the pinned JSON driver before retrying."


class AudioCppNemotronASRProvider(ASRProvider):
    """Nemotron family adapter using native audio.cpp RNN-T timestamp entries."""

    provider_id = NEMOTRON_ASR_PROVIDER_ID

    def __init__(self, runtime: LocalAudioRuntime | Any | None = None) -> None:
        self._runtime = runtime

    @classmethod
    def from_environment(cls) -> "AudioCppNemotronASRProvider":
        binary = os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        if not binary or sys.platform.startswith("win"):
            return cls()
        driver = AudioCppRuntimeDriver(
            binary_path=Path(binary),
            source_revision=PINNED_AUDIO_CPP_REVISION,
        )
        return cls(
            LocalAudioRuntime(
                LocalRuntimeRegistry((driver,)),
                promoted_families=(NEMOTRON_ASR_FAMILY,),
                lifecycle=_audio_cpp_gpu_lifecycle(),
            )
        )

    def transcribe(self, request: ASRRequest) -> ASRResult:
        if self._runtime is None:
            raise ModuleNotFoundError(AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION)
        _reject_unsupported_prompt_context(request.hints)
        model_id = request.model_id or NEMOTRON_ASR_MODEL_ID
        local_request = LocalASRRequest(
            request_id=uuid4().hex,
            family=NEMOTRON_ASR_FAMILY,
            provider_id=self.provider_id,
            audio_path=request.audio_path,
            model_id=model_id,
            language=request.language,
            timestamp_mode=request.timestamp_mode,
        )
        response = self._runtime.execute(local_request.to_runtime_request(), runtime_choice="auto")
        if not isinstance(response, LocalRuntimeResponse):
            raise ValueError("audio.cpp Nemotron ASR runtime returned an invalid response")
        local_response = LocalASRResponse.from_runtime_response(response)
        payload = local_response.payload
        duration_s = _optional_finite_number(payload, "duration_s")
        raw_timestamp_entries = _raw_timestamp_entries(
            payload, required=request.timestamp_mode == "word"
        )

        result_language = local_response.language or request.language or ""
        if request.timestamp_mode == "word":
            words = normalize_nemotron_word_timestamps(
                raw_timestamp_entries,
                response_offset_s=_finite_number(
                    payload.get("chunk_offset_s", 0.0), "chunk_offset_s"
                ),
            )
            if local_response.transcript.strip() and not words:
                raise ValueError("Nemotron native timestamps returned no words for speech output")
            result = ASRResult(
                transcript=local_response.transcript,
                provider_id=self.provider_id,
                model_id=model_id,
                language=result_language,
                duration_s=duration_s,
                words=words,
                alignment_origin="native",
                execution=_execution_receipt(response.receipt, request, raw_timestamp_entries),
            )
        else:
            result = ASRResult(
                transcript=local_response.transcript,
                provider_id=self.provider_id,
                model_id=model_id,
                language=result_language,
                duration_s=duration_s,
                execution=_execution_receipt(response.receipt, request, raw_timestamp_entries),
            )
        return validate_asr_response(request, result)


def _reject_unsupported_prompt_context(hints: ASRContextHints) -> None:
    if (
        hints.context_text is not None
        or hints.initial_prompt is not None
        or hints.glossary is not None
        or hints.phrase_hints
    ):
        raise ValueError(
            "audio.cpp Nemotron ASR does not expose free-text or phrase-boosting prompts; "
            "only its model-owned language prompt selection is supported."
        )


def _raw_timestamp_entries(
    payload: Mapping[str, object], *, required: bool
) -> tuple[Mapping[str, object], ...]:
    raw_entries = payload.get("word_timestamps")
    if raw_entries is None and not required:
        return ()
    if not isinstance(raw_entries, list):
        raise ValueError("Nemotron native response must contain a word_timestamps list")
    entries: list[Mapping[str, object]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Nemotron native word timestamp {index} must be an object")
        entries.append(entry)
    return tuple(entries)


def _audio_cpp_gpu_lifecycle() -> GPULifecycleOwner:
    return GPULifecycleOwner(
        GPULeaseManager(
            metadata_path=Path(tempfile.gettempdir()) / "voiceover-pipeline-gpu-lease.json"
        ),
        probe=probe_local_gpu_state,
        min_free_vram_mb=NEMOTRON_AUDIO_CPP_MIN_FREE_VRAM_MB,
        max_utilization_percent=NEMOTRON_AUDIO_CPP_MAX_GPU_UTILIZATION_PERCENT,
    )


def _optional_finite_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _finite_number(value, key)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"Nemotron native timestamp {label} must be a finite number")
    return float(value)


def _execution_receipt(
    receipt: RuntimeExecutionReceipt | None,
    request: ASRRequest,
    raw_timestamp_entries: tuple[Mapping[str, object], ...],
) -> ASRExecutionReceipt:
    return ASRExecutionReceipt(
        runtime=receipt.driver_id if receipt is not None else "audio-cpp",
        runtime_version=receipt.source_revision if receipt is not None else None,
        model_revision=receipt.source_revision if receipt is not None else None,
        resolved_device=request.device,
        resolved_compute=request.compute,
        raw_timestamp_entries=raw_timestamp_entries,
    )


def audio_cpp_nemotron_asr_dependency_probe() -> ASRDependencyHealth:
    binary = os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
    if not binary or sys.platform.startswith("win"):
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
            reason_code="native_unavailable",
        )
    driver = AudioCppRuntimeDriver(
        binary_path=Path(binary),
        source_revision=PINNED_AUDIO_CPP_REVISION,
    )
    health = driver.health()
    if not health.available:
        return ASRDependencyHealth(
            available=False,
            remediation=f"{AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION} {health.remediation}".strip(),
        )
    return ASRDependencyHealth(available=True, remediation="")
