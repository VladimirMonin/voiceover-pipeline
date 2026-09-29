"""Bounded script preparation and prepared speech parts for one CLI generation run."""

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from ..config import OMNIVOICE_LOCAL_MODEL_ID
from ..gemini_dialogue import is_dialogue_format
from ..models import ScriptChunk


class PreparationError(ValueError):
    """Explicit chunk-preparation argument failure the CLI reports as a usage error."""


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
