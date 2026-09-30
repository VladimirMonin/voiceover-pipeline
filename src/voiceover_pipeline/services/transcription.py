"""ASR request preparation, provider execution, and result-capability checks.

This module owns everything after the CLI has resolved one ASR request: it
builds the request from parsed arguments, selects the runtime backend, probes
that backend, constructs the provider, invokes it through the optional long-form
orchestrator, validates the returned result, and invokes the dialogue-quality
route. It also owns the generic ASR timing request/bridge and the timing-provider
invocation for the legacy timings route, plus the per-turn dialogue-quality gate
that runs before the final concat. The CLI keeps argument validation,
provider-spec lookup, the unsupported-timing guard, and the fail-envelope
translation, and forwards its currently bound callables so the existing
monkeypatch seams keep steering these bodies without importing ``cli``.
"""

import argparse
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

from ..asr_timing_bridge import asr_result_to_timing
from ..config import DEFAULT_ASR_COMPUTE, DEFAULT_ASR_DEVICE, DEFAULT_TIMING_MODEL
from ..local_runtime.transports.audio_cpp_cli import NATIVE_AUDIO_CPP_EXECUTABLE_ENV
from ..models import (
    ASRContextHints,
    ASRRequest,
    ASRResult,
    ASRRuntimeChoice,
    ChunkArtifact,
    RunPaths,
    ScriptChunk,
    TimingResult,
)
from ..providers.asr_registry import ASRProviderSpec
from ..providers.base import ASRProvider, TranscriptionProvider, validate_asr_response
from ..run_state import atomic_write_json
from ..tts_quality import evaluate_tts_transcript

LongFormTranscribe = Callable[[ASRProvider, ASRRequest], ASRResult]
QualityTranscribe = Callable[[argparse.Namespace], tuple[ASRResult, Path]]
DialogueQualityTranscribe = Callable[
    [argparse.Namespace, Path], tuple[str, str, str | None, str, str | None]
]
FailQuality = Callable[[str, dict[str, Any]], NoReturn]


class ASRDependencyUnavailableError(RuntimeError):
    """The selected runtime backend or its optional dependency is unavailable."""


class ASRCapabilityError(RuntimeError):
    """A provider result exceeds the capabilities its spec declared."""


def build_asr_request(
    args: argparse.Namespace,
    spec: ASRProviderSpec,
    hints: ASRContextHints,
    audio_path: Path,
) -> ASRRequest:
    """Build an ASR request from parsed CLI arguments and a provider spec."""
    runtime = cast(ASRRuntimeChoice, getattr(args, "runtime", "auto"))
    model_id = args.model
    if model_id is None:
        model_id = next((model["id"] for model in spec.models if model.get("default")), None)
    return ASRRequest(
        audio_path=audio_path,
        model_id=model_id,
        language=args.language,
        device=args.device,
        compute=args.compute,
        hints=hints,
        timestamp_mode="word" if getattr(args, "word_timestamps", False) else "none",
        runtime_choice=runtime,
    )


def resolve_asr_provider(spec: ASRProviderSpec, request: ASRRequest) -> ASRProvider:
    """Select the runtime backend, probe it, and construct the provider.

    Runtime-specific probes and factories are imported lazily so an unselected
    runtime never imports another optional local package. A missing runtime
    backend or unavailable dependency raises ``ASRDependencyUnavailableError``;
    a ``factory()`` failure propagates unchanged, matching the constructor's
    former position outside the CLI invocation ``try``.
    """
    runtime = request.runtime_choice
    if spec.provider_id == "nemotron-local" and runtime == "audio-cpp":
        from ..providers.audio_cpp_nemotron_asr import audio_cpp_nemotron_asr_dependency_probe
        from ..providers.nemotron_asr_local import nemotron_asr_audio_cpp_provider_factory

        health = audio_cpp_nemotron_asr_dependency_probe()
        provider_factory = nemotron_asr_audio_cpp_provider_factory
    elif spec.provider_id == "nemotron-local" and runtime == "python":
        from ..providers.nemotron_asr_local import (
            nemotron_asr_python_dependency_probe,
            nemotron_asr_python_provider_factory,
        )

        health = nemotron_asr_python_dependency_probe()
        provider_factory = nemotron_asr_python_provider_factory
    elif runtime == "audio-cpp":
        if not os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip():
            raise ASRDependencyUnavailableError(
                f"ASR provider {spec.provider_id} does not support runtime=audio-cpp "
                "without a native audio.cpp package"
            )
        health = spec.dependency_probe()
        provider_factory = spec.factory
    else:
        health = spec.dependency_probe()
        provider_factory = spec.factory
    if not health.available:
        raise ASRDependencyUnavailableError(health.remediation)
    return provider_factory()


def transcribe_asr_request(
    provider: ASRProvider,
    spec: ASRProviderSpec,
    request: ASRRequest,
    *,
    long_form: bool = False,
    long_form_transcribe: LongFormTranscribe | None = None,
    capability_check: bool = False,
) -> ASRResult:
    """Invoke a resolved ASR provider once and validate the returned result.

    ``long_form`` selects the chunked orchestration for providers that declare
    it; the orchestrator is injected as ``long_form_transcribe`` so the CLI's
    bound ``transcribe_prerecorded_long_form`` stays authoritative. ``spec`` is
    read only for the declared-capability check that ``capability_check``
    enables, which the timing route leaves off. Provider and validation
    exceptions propagate unchanged, and a declared-capability mismatch raises
    ``ASRCapabilityError``.
    """
    if long_form:
        if long_form_transcribe is None:
            raise RuntimeError(
                "long-form ASR invocation requires an injected long_form_transcribe callable"
            )
        raw_result = long_form_transcribe(provider, request)
    else:
        raw_result = provider.transcribe(request)
    result = validate_asr_response(request, raw_result)
    if capability_check:
        capability_error = validate_result_capabilities(result, spec)
        if capability_error is not None:
            raise ASRCapabilityError(capability_error)
    return result


def transcribe_generic_asr_timing(
    spec: ASRProviderSpec,
    *,
    audio_path: Path,
    model: str | None,
    device: str,
    compute: str,
    language: str | None,
    mp3_duration_ms: Callable[[str, Path], int],
) -> TimingResult:
    """Run one generic ASR timing request and bridge its result into timings.

    The CLI validates provider options and probes the spec dependency before this
    point. This body then builds the word-timestamp request, invokes the spec
    factory exactly once through ``transcribe_asr_request``, requires ``ffprobe``
    for the authoritative source duration, and converts the result through
    ``asr_result_to_timing``. The duration callable is injected by the CLI so the
    existing ``mp3_duration_ms`` and ``shutil.which`` seams keep steering these
    bodies.
    """
    model_id = model or next((item["id"] for item in spec.models if item.get("default")), None)
    request = ASRRequest(
        audio_path=audio_path,
        model_id=model_id,
        language=language,
        device=device,
        compute=compute,
        timestamp_mode="word",
    )
    result = transcribe_asr_request(spec.factory(), spec, request)
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("FFprobe is required to validate generic ASR timestamp bounds")
    source_duration_s = mp3_duration_ms(ffprobe_path, audio_path) / 1000
    return asr_result_to_timing(
        result,
        source_audio=str(audio_path.resolve()),
        source_duration_s=source_duration_s,
    )


def transcribe_dialogue_quality_audio(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    transcribe_result: QualityTranscribe,
) -> tuple[str, str, str | None, str, str | None]:
    """Return transcript plus content-free ASR identity for a quality check.

    The ``xai-stt`` route calls its timing provider directly and is imported only
    for that branch, so the other routes never import the XAI adapter. Every
    other route delegates to the injected ``transcribe_result`` callable, which
    the CLI binds to its own ``_transcribe_result`` so the existing monkeypatch
    seam keeps intercepting dialogue-quality transcription.
    """
    provider_id = args.tts_quality_provider
    if provider_id == "xai-stt":
        from ..providers.xai_stt import XAISttProvider

        provider = XAISttProvider(model=args.tts_quality_model or "grok-stt")
        timing = provider.transcribe(
            audio_path=audio_path,
            language=args.tts_quality_language or "ru",
            word_timestamps=False,
            quiet=True,
        )
        transcript = " ".join(segment.text for segment in timing.segments).strip()
        return transcript, provider_id, provider.model, "cloud-api", None

    transcribe_args = argparse.Namespace(
        audio=audio_path,
        provider=provider_id,
        model=getattr(args, "tts_quality_model", None),
        language=getattr(args, "tts_quality_language", None),
        device=getattr(args, "tts_quality_device", DEFAULT_ASR_DEVICE),
        compute=getattr(args, "tts_quality_compute", DEFAULT_ASR_COMPUTE),
        runtime=getattr(args, "tts_quality_runtime", "auto"),
        context=None,
        context_file=None,
        word_timestamps=False,
    )
    result, _ = transcribe_result(transcribe_args)
    return (
        result.transcript,
        result.provider_id,
        result.model_id,
        result.execution.runtime,
        result.execution.model_revision,
    )


def transcribe_timing_audio(
    *,
    audio_path: Path,
    timing_provider: str,
    model: str | None,
    device: str,
    compute_type: str,
    language: str,
    word_timestamps: bool = False,
    quiet: bool = False,
    local_files_only: bool = False,
) -> TimingResult:
    """Select the timing provider and run one timestamped transcription.

    Each optional adapter is imported only for its own branch so an unselected
    runtime never imports another package. The model defaults match the former
    CLI branches; ``openrouter-whisper`` is rejected by the CLI before this
    point because it cannot return real timestamps. ``local_files_only`` is
    forwarded to the local faster-whisper adapter so the native timing route can
    forbid an implicit model download; cloud providers ignore it.
    """
    provider: TranscriptionProvider
    local_provider: Any | None = None
    if timing_provider == "groq-whisper":
        from ..providers.groq_whisper import GroqWhisperProvider

        effective_model = model or "whisper-large-v3-turbo"
        provider = GroqWhisperProvider(model=effective_model)
    elif timing_provider == "xai-stt":
        from ..providers.xai_stt import XAISttProvider

        effective_model = model or "grok-stt"
        provider = XAISttProvider(model=effective_model)
    else:
        from ..providers.faster_whisper import FasterWhisperProvider

        effective_model = model or DEFAULT_TIMING_MODEL
        local_provider = FasterWhisperProvider(
            model_size=effective_model,
            device=device,
            compute_type=compute_type,
        )
        provider = local_provider

    if local_provider is not None:
        # Only the local adapter understands ``local_files_only``; passing it to a
        # cloud adapter would be an unexpected keyword argument.
        return local_provider.transcribe(
            audio_path=audio_path,
            language=language,
            word_timestamps=word_timestamps,
            quiet=quiet,
            local_files_only=local_files_only,
        )
    return provider.transcribe(
        audio_path=audio_path,
        language=language,
        word_timestamps=word_timestamps,
        quiet=quiet,
    )


def verify_dialogue_turns_before_concat(
    *,
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    chunk_artifacts: list[ChunkArtifact],
    paths: RunPaths,
    transcribe_quality_audio: DialogueQualityTranscribe,
    sha256_file: Callable[[Path], str],
    fail_quality: FailQuality,
) -> dict[str, Any] | None:
    """Transcribe and strictly verify every dialogue turn before final concat.

    This is the fail-closed gate itself: it writes the running receipt, records
    one content-free turn receipt per chunk in turn order, and calls the injected
    ``fail_quality`` with the current receipt when a turn fails so no concat runs
    afterwards. The CLI binds ``transcribe_quality_audio`` to its own dialogue
    quality transcribe callable and ``sha256_file`` to its hashing helper so the
    existing monkeypatch seams keep steering this body.
    """
    provider_id = getattr(args, "tts_quality_provider", None)
    if provider_id is None:
        return None

    artifacts_by_number = {artifact.number: artifact for artifact in chunk_artifacts}
    receipt_path = paths.output_root / "tts_quality.json"
    aggregate: dict[str, Any] = {
        "artifact_type": "voiceover-dialogue-tts-quality-receipt",
        "status": "running",
        "passed": False,
        "provider": provider_id,
        "model": getattr(args, "tts_quality_model", None),
        "turn_count": len(chunks),
        "turns": [],
        "human_listening_required": True,
    }
    atomic_write_json(receipt_path, aggregate)

    for chunk in chunks:
        artifact = artifacts_by_number.get(chunk.number)
        if artifact is None:
            aggregate["status"] = "quality_failed"
            aggregate["failure_reason"] = "missing_turn_artifact"
            atomic_write_json(receipt_path, aggregate)
            fail_quality(
                f"Dialogue TTS quality gate failed: turn {chunk.number} has no audio artifact.",
                aggregate,
            )
        audio_path = paths.chunks_dir / artifact.file
        transcript, asr_provider, asr_model, asr_runtime, asr_revision = transcribe_quality_audio(
            args, audio_path
        )
        quality = evaluate_tts_transcript(
            expected_text=chunk.text,
            actual_transcript=transcript,
            minimum_similarity=1.0,
            maximum_missing_ratio=0.0,
            maximum_unexpected_ratio=0.0,
            maximum_repeated_ngram_excess=0,
            strip_audio_tags=True,
        )
        turn_receipt = quality.public_receipt(
            audio_sha256=sha256_file(audio_path),
            asr_provider=asr_provider,
            asr_model=asr_model,
            asr_runtime=asr_runtime,
            asr_model_revision=asr_revision,
        )
        turn_receipt["turn_index"] = chunk.number
        aggregate["turns"].append(turn_receipt)
        if not quality.passed:
            aggregate["status"] = "quality_failed"
            aggregate["failed_turn"] = chunk.number
            atomic_write_json(receipt_path, aggregate)
            fail_quality(
                f"Dialogue TTS quality gate failed for turn {chunk.number}; "
                "final concat was not created.",
                aggregate,
            )
        atomic_write_json(receipt_path, aggregate)

    aggregate["status"] = "success"
    aggregate["passed"] = True
    atomic_write_json(receipt_path, aggregate)
    return aggregate


def validate_result_capabilities(result: ASRResult, spec: ASRProviderSpec) -> str | None:
    """Return a provider error message when a result exceeds declared capabilities."""
    capabilities = spec.capabilities
    if result.provider_id != spec.provider_id:
        return f"ASR provider {spec.provider_id} returned provider ID {result.provider_id}"
    if (
        any(segment.start_s is not None for segment in result.segments)
        and not capabilities.segment_timestamps
    ):
        return f"ASR provider {spec.provider_id} returned undeclared segment timestamps"
    if result.words and not capabilities.word_timestamps:
        return f"ASR provider {spec.provider_id} returned undeclared word timestamps"
    if result.alignment_origin == "forced" and not capabilities.forced_alignment:
        return f"ASR provider {spec.provider_id} returned undeclared forced alignment"
    return None
