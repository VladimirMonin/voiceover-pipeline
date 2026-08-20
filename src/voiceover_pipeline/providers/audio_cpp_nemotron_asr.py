from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
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
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    AudioCppNativeCLITransport,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    AudioCppPackageError,
    admit_audio_cpp_native_package,
)
from voiceover_pipeline.models import (
    ASRContextHints,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth
from voiceover_pipeline.providers.base import ASRProvider, validate_asr_response
from voiceover_pipeline.providers.nemotron_asr_local import (
    NEMOTRON_ASR_MODEL_ID,
    NEMOTRON_ASR_PROVIDER_ID,
)

NEMOTRON_ASR_FAMILY = "nemotron-3.5-asr"
NEMOTRON_AUDIO_CPP_MODEL_ENV = "VOICEOVER_AUDIO_CPP_NEMOTRON_MODEL"
NEMOTRON_AUDIO_CPP_MIN_FREE_VRAM_MB = 4096
NEMOTRON_AUDIO_CPP_MAX_GPU_UTILIZATION_PERCENT = 90
NEMOTRON_AUDIO_CPP_TRAILING_CLAMP_TOLERANCE_S = 0.2
AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION = "audio.cpp Nemotron ASR runtime is unavailable. Set VOICEOVER_AUDIO_CPP_BINARY to the pinned JSON driver before retrying."


class AudioCppNemotronASRProvider(ASRProvider):
    """Nemotron family adapter using native audio.cpp RNN-T timestamp entries."""

    provider_id = NEMOTRON_ASR_PROVIDER_ID

    def __init__(self, runtime: LocalAudioRuntime | Any | None = None) -> None:
        self._runtime = runtime

    @classmethod
    def from_environment(cls) -> "AudioCppNemotronASRProvider":
        binary = os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        if sys.platform.startswith("win"):
            return cls(_native_windows_runtime_from_environment())
        if not binary:
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
        measurements = _emission_measurements(payload)

        result_language = local_response.language or request.language or ""
        if request.timestamp_mode == "word":
            words = normalize_nemotron_word_timestamps(
                raw_timestamp_entries,
                response_offset_s=_finite_number(
                    payload.get("chunk_offset_s", 0.0), "chunk_offset_s"
                ),
            )
            words = _clamp_words_to_duration(words, duration_s)
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
                execution=_execution_receipt(
                    response.receipt, request, raw_timestamp_entries, measurements
                ),
            )
        else:
            result = ASRResult(
                transcript=local_response.transcript,
                provider_id=self.provider_id,
                model_id=model_id,
                language=result_language,
                duration_s=duration_s,
                execution=_execution_receipt(
                    response.receipt, request, raw_timestamp_entries, measurements
                ),
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


def _admitted_nemotron_model_from_environment() -> Path | None:
    model_path = os.environ.get(NEMOTRON_AUDIO_CPP_MODEL_ENV, "").strip()
    if not model_path:
        return None
    return Path(model_path)


def _native_windows_runtime_from_environment() -> LocalAudioRuntime | None:
    """Admit the native Windows package and build the CUDA-only Nemotron runtime.

    Fail closed: a missing native executable, model, or invalid package returns
    ``None`` so the provider stays unavailable instead of falling back to the
    Python route when the caller explicitly selected the native runtime.
    """
    native_executable = os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip()
    model_path = _admitted_nemotron_model_from_environment()
    if not native_executable or model_path is None:
        return None
    try:
        install = admit_audio_cpp_native_package(
            Path(native_executable), required_model_paths=(model_path,)
        )
        transport = AudioCppNativeCLITransport(
            executable_path=install.executable_path,
            model_paths={NEMOTRON_ASR_FAMILY: model_path},
        )
    except (AudioCppPackageError, ValueError):
        return None
    driver = AudioCppRuntimeDriver(
        binary_path=install.executable_path,
        source_revision=PINNED_AUDIO_CPP_REVISION,
        transport=transport,
        build_hash=install.files[install.executable_path.name],
        transport_name="native-cli",
    )
    return LocalAudioRuntime(
        LocalRuntimeRegistry((driver,)),
        promoted_families=(NEMOTRON_ASR_FAMILY,),
        lifecycle=_audio_cpp_gpu_lifecycle(),
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


def _clamp_words_to_duration(
    words: tuple[ASRWordSpan, ...], duration_s: float | None
) -> tuple[ASRWordSpan, ...]:
    """Clamp native word spans to the real staged-audio duration.

    The native Nemotron runtime may emit a trailing RNN-T token a few frames
    beyond the actual audio length; entries within a bounded trailing tolerance
    are clamped (not rejected) so the strict ASR bounds validation treats them
    as a tolerance at the audio boundary. Live evidence shows trailing
    overshoots of 0.08-0.12s, so the tolerance is set to 0.2s with headroom.
    Clamping preserves ``end >= start`` and word monotonicity because
    ``min(value, duration)`` is monotone. A word whose entire span lies beyond
    the duration collapses to a zero-length span at the boundary instead of
    being dropped, keeping transcript correspondence intact. Entries beyond the
    tolerance are rejected instead of silently normalized, so a wrong timebase
    or severe decoder failure is not hidden.
    """
    if duration_s is None or duration_s <= 0 or not words:
        return words
    for word in words:
        if word.end_s > duration_s + NEMOTRON_AUDIO_CPP_TRAILING_CLAMP_TOLERANCE_S:
            raise ValueError(
                "Nemotron native word timestamp exceeds the staged-audio duration beyond "
                "the trailing tolerance; refusing to clamp an invalid timebase"
            )
    clamped: list[ASRWordSpan] = []
    for word in words:
        start_s = min(word.start_s, duration_s)
        end_s = min(word.end_s, duration_s)
        if end_s < start_s:
            end_s = start_s
        clamped.append(replace(word, start_s=start_s, end_s=end_s))
    return tuple(clamped)


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
    measurements: Mapping[str, float] | None = None,
) -> ASRExecutionReceipt:
    return ASRExecutionReceipt(
        runtime=receipt.driver_id if receipt is not None else "audio-cpp",
        runtime_version=receipt.source_revision if receipt is not None else None,
        model_revision=None,
        resolved_device=request.device,
        resolved_compute=request.compute,
        measurements=measurements or {},
        raw_timestamp_entries=raw_timestamp_entries,
    )


def _emission_measurements(payload: Mapping[str, object]) -> dict[str, float]:
    """Mirror native words-emission status into numeric receipt measurements.

    ``words_emitted`` distinguishes a native run that truly produced no speech
    (no words file) from a run whose words were merely empty; both are legal,
    but only the former is direct evidence of a no-speech outcome.
    """
    words_emitted = payload.get("words_emitted")
    if isinstance(words_emitted, bool):
        return {"native_words_emitted": 1.0 if words_emitted else 0.0}
    return {}


def audio_cpp_nemotron_asr_dependency_probe() -> ASRDependencyHealth:
    if sys.platform.startswith("win"):
        return _native_windows_dependency_probe()
    binary = os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
    if not binary:
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


def _native_windows_dependency_probe() -> ASRDependencyHealth:
    native_executable = os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip()
    if not native_executable:
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
            reason_code="missing_native_executable",
        )
    model_path = _admitted_nemotron_model_from_environment()
    if model_path is None:
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
            reason_code="missing_model_artifact",
        )
    try:
        admit_audio_cpp_native_package(Path(native_executable), required_model_paths=(model_path,))
    except AudioCppPackageError as exc:
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
            reason_code=exc.code,
        )
    except ValueError:
        return ASRDependencyHealth(
            available=False,
            remediation=AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
            reason_code="invalid_native_package",
        )
    return ASRDependencyHealth(available=True, remediation="")
