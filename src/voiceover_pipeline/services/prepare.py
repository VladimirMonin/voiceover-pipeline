"""Bounded prepared speech parts for one CLI generation run."""

import argparse
from dataclasses import dataclass
from typing import Sequence

from ..models import ScriptChunk


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
