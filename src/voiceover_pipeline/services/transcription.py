"""ASR request preparation, provider execution, and result-capability checks.

This module owns everything after the CLI has resolved one ASR request: it
builds the request from parsed arguments, selects the runtime backend, probes
that backend, constructs the provider, invokes it through the optional long-form
orchestrator, validates the returned result, and invokes the dialogue-quality
route. The CLI keeps argument validation, provider-spec lookup, and the
fail-envelope translation, and forwards its currently bound callables so the
existing monkeypatch seams keep steering this body without importing ``cli``.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ..config import DEFAULT_ASR_COMPUTE, DEFAULT_ASR_DEVICE
from ..local_runtime.transports.audio_cpp_cli import NATIVE_AUDIO_CPP_EXECUTABLE_ENV
from ..models import ASRContextHints, ASRRequest, ASRResult, ASRRuntimeChoice
from ..providers.asr_registry import ASRProviderSpec
from ..providers.base import ASRProvider, validate_asr_response

LongFormTranscribe = Callable[[ASRProvider, ASRRequest], ASRResult]
QualityTranscribe = Callable[[argparse.Namespace], tuple[ASRResult, Path]]


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
