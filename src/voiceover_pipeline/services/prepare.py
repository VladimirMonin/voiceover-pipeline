"""Bounded script preparation and prepared speech parts for one CLI generation run."""

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..config import (
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_OPENROUTER_TTS_VOICE,
    DEFAULT_POLZA_TTS_VOICE,
    DEFAULT_QWEN_VOICE,
    DEFAULT_VOICE,
    OMNIVOICE_LOCAL_MODEL_ID,
)
from ..gemini_dialogue import DIALOGUE_FORMAT, is_dialogue_format
from ..models import ScriptChunk
from ..omnivoice_voice_bank import VoiceBankCatalog
from ..tts_prompting import resolve_prompt_mode
from ..voiceover_script import VOICEOVER_FORMAT, detect_frontmatter_format


class PreparationError(ValueError):
    """Script or run-preparation argument failure the CLI reports as a usage error."""


LocalChunkPreparation = Callable[[Iterable[ScriptChunk], str, str | None], list[ScriptChunk]]


class SessionFragmentMerge(Protocol):
    """The OmniVoice session-merge seam the CLI forwards for patchability."""

    def __call__(
        self,
        fragments: Iterable[ScriptChunk],
        *,
        mode: str = ...,
        reference_audio_path: Path | str | None = ...,
        reference_text: str | None = ...,
        design_instruction: str | None = ...,
    ) -> list[ScriptChunk]: ...


@dataclass(frozen=True)
class PreparedScriptFragments:
    """Limited script fragments plus the fragment counts a dry run reports.

    ``chunks`` holds the exact objects the run hashes and stores. A provider/model
    spoken-text profile may replace them, but ``--limit-chunks`` only slices, so no
    fragment is renumbered, reordered, or rewritten while limiting.
    """

    chunks: list[ScriptChunk]
    original_count: int
    requested_count: int


@dataclass(frozen=True)
class PreparedPart:
    """One original script chunk plus the per-part voice identity for its call.

    The wrapped chunk is the exact object the run hashes and stores, so part
    identity (number, id, text, speaker, pause) never changes between
    preparation and synthesis. ``voice`` is the chunk's own voice from the
    dialogue cast; ``None`` means the run-level voice applies.
    """

    chunk: ScriptChunk
    voice: str | None


@dataclass(frozen=True)
class PreparedRun:
    """Run-scoped provider identity plus the ordered parts to synthesize.

    ``voice`` is the run-level voice the CLI already resolved; a part without
    its own cast voice falls back to it.
    """

    provider: str
    model: str
    voice: str
    style_prompt: str | None
    prompt_mode: str
    parts: tuple[PreparedPart, ...]


def default_voice(args: argparse.Namespace) -> str | None:
    """Return the provider/model default voice when the run specifies none.

    ``omnivoice-local`` has no single default voice: its preset catalog or an
    explicit mode selects one, so this returns ``None``.
    """
    if args.provider == "polza-tts":
        if args.model and args.model.startswith("elevenlabs/"):
            return DEFAULT_ELEVENLABS_VOICE
        return DEFAULT_POLZA_TTS_VOICE
    if args.provider == "openrouter-tts":
        if args.model and args.model.startswith("openai/"):
            return DEFAULT_OPENAI_TTS_VOICE
        return DEFAULT_OPENROUTER_TTS_VOICE
    if args.provider == "qwen-local":
        return DEFAULT_QWEN_VOICE
    if args.provider == "omnivoice-local":
        return None
    return DEFAULT_VOICE


def resolve_script_format(script_path: Path, requested_format: str) -> str:
    """Resolve the run's script format from frontmatter and the requested flag.

    A ``markdown`` request is upgraded by voiceover/dialogue frontmatter, an
    explicit dialogue alias collapses to ``dialogue``, and an unset format
    falls back to ``markdown``.
    """
    detected_format = detect_frontmatter_format(script_path)
    script_format = requested_format
    if (
        detected_format is not None
        and script_format == "markdown"
        and (detected_format == VOICEOVER_FORMAT or is_dialogue_format(detected_format))
    ):
        script_format = detected_format
    if script_format is None:
        return "markdown"
    return DIALOGUE_FORMAT if is_dialogue_format(script_format) else script_format


def bind_omnivoice_dialogue_fingerprints(
    chunks: list[ScriptChunk], catalog: VoiceBankCatalog
) -> list[ScriptChunk]:
    """Bind each dialogue turn to its voice-bank profile's reference fingerprint.

    A turn without a cast voice or with a profile id absent from the admitted
    catalog raises ``PreparationError`` before provider work in CLI generation,
    preventing submission of a turn with an unknown clone identity.
    """
    profiles = {profile.id: profile for profile in catalog.profiles}
    bound: list[ScriptChunk] = []
    for chunk in chunks:
        if chunk.voice is None:
            raise PreparationError("OmniVoice dialogue turn is missing a voice-bank profile")
        profile = profiles.get(chunk.voice)
        if profile is None:
            raise PreparationError(f"voice '{chunk.voice}' not found in the voice bank")
        bound.append(replace(chunk, voice_fingerprint=profile.reference_sha256))
    return bound


@dataclass(frozen=True)
class PreparedGenerationIdentity:
    """Run-scoped cast identity plus the resolved style prompt and prompt mode.

    ``chunks`` are the exact objects the run hashes and stores: the CLI's
    dialogue cast may bind each chunk's OmniVoice fingerprint here, but no
    number, id, text, speaker, or pause changes.
    """

    chunks: list[ScriptChunk]
    style_prompt: str | None
    prompt_mode: str


def prepare_generation_identity(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    gemini_report: dict[str, Any] | None,
    *,
    resolve_style_prompt: Callable[[argparse.Namespace], str | None],
) -> PreparedGenerationIdentity:
    """Resolve the run's cast voice, style prompt, and prompt mode for one generation.

    The steps keep their exact order from the CLI: the cast voice and per-chunk
    OmniVoice fingerprints bind first, then ``resolve_style_prompt`` runs, then a
    dialogue report may supply the style prompt, then the prompt mode is resolved.
    ``resolve_style_prompt`` is injected so the service keeps no ``cli`` import and
    the CLI still owns reading/validating the provider's style input.
    """
    requested_voice = args.voice
    if gemini_report:
        args.speaker_voice_map = gemini_report["speaker_voice_map"]
        args.voice = requested_voice or next(iter(args.speaker_voice_map.values()))
        if args.provider == "omnivoice-local":
            chunks = bind_omnivoice_dialogue_fingerprints(chunks, args.voice_bank_catalog)
    else:
        args.speaker_voice_map = {}
        args.voice = requested_voice or default_voice(args)

    style_prompt = resolve_style_prompt(args)
    if (
        gemini_report
        and args.provider != "openrouter-tts"
        and not args.no_style_prompt
        and args.style_prompt is None
        and args.style_prompt_file is None
    ):
        style_prompt = gemini_report["style_prompt"]
    prompt_mode = resolve_prompt_mode(args.provider, args.model)
    return PreparedGenerationIdentity(
        chunks=chunks,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
    )


def prepare_run(
    args: argparse.Namespace,
    chunks: Sequence[ScriptChunk],
    style_prompt: str | None,
    prompt_mode: str,
) -> PreparedRun:
    """Wrap the already-resolved run identity and chunks without changing them."""
    return PreparedRun(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        parts=tuple(PreparedPart(chunk=chunk, voice=chunk.voice) for chunk in chunks),
    )


def prepare_script_fragments(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    script_format: str,
    *,
    local_prepare: LocalChunkPreparation,
) -> PreparedScriptFragments:
    """Normalize and limit the script fragments a run will synthesize.

    Non-dialogue scripts first pass through the caller's provider/model spoken-text
    profile, so ``--limit-chunks`` always slices already-prepared fragments. Dialogue
    chunks are one turn each and bypass that step. Empty scripts and non-positive
    limits raise ``PreparationError`` before any provider, key, or pricing work.
    """
    prepared = chunks
    if not is_dialogue_format(script_format):
        prepared = local_prepare(prepared, args.provider, args.model)
    if not prepared:
        raise PreparationError("Script produced no chunks. Check delimiter and content.")
    original_count = len(prepared)
    if args.limit_chunks is not None:
        if args.limit_chunks <= 0:
            raise PreparationError("--limit-chunks must be greater than zero")
        prepared = prepared[: args.limit_chunks]
    return PreparedScriptFragments(
        chunks=prepared,
        original_count=original_count,
        requested_count=len(prepared),
    )


def prepare_runtime_chunks(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    script_format: str,
    *,
    merge_session_fragments: SessionFragmentMerge,
) -> list[ScriptChunk]:
    """Fold prepared non-dialogue fragments into one OmniVoice request when eligible.

    A preset run with an admitted voice-bank profile always clones that profile's
    resolved reference; otherwise the explicit mode/reference/design arguments apply
    unchanged. Dialogue turns and every other provider/model keep their fragments.
    """
    if (
        args.provider == "omnivoice-local"
        and args.model == OMNIVOICE_LOCAL_MODEL_ID
        and not is_dialogue_format(script_format)
    ):
        bank_profile = getattr(args, "voice_bank_profile", None)
        bank_catalog = getattr(args, "voice_bank_catalog", None)
        if getattr(args, "mode", "preset") == "preset" and bank_profile is not None:
            reference_audio_path = (
                str(bank_catalog.root / bank_profile.reference_audio)
                if bank_catalog is not None
                else str(Path(bank_profile.reference_audio))
            )
            return merge_session_fragments(
                chunks,
                mode="clone",
                reference_audio_path=reference_audio_path,
                reference_text=bank_profile.reference_text,
            )
        return merge_session_fragments(
            chunks,
            mode=getattr(args, "mode", "preset"),
            reference_audio_path=getattr(args, "reference_audio", None),
            reference_text=getattr(args, "reference_text", None),
            design_instruction=getattr(args, "design_instruction", None),
        )
    return chunks
