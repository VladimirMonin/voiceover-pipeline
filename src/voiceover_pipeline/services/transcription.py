"""Pure ASR request preparation and result-capability checks for the CLI."""

import argparse
from pathlib import Path
from typing import cast

from ..models import ASRContextHints, ASRRequest, ASRResult, ASRRuntimeChoice
from ..providers.asr_registry import ASRProviderSpec


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
