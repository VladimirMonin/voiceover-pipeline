"""Deterministic spoken-text preparation for local TTS runtimes."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from .config import OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE, OMNIVOICE_LOCAL_MODEL_ID
from .models import ScriptChunk


@dataclass(frozen=True)
class LocalTTSChunkProfile:
    """Verified spoken-text preparation policy for one provider/model pair."""

    target_chars: int
    reject_raw_digits: bool


LOCAL_TTS_CHUNK_PROFILES = MappingProxyType(
    {
        ("omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID): LocalTTSChunkProfile(
            target_chars=OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE,
            reject_raw_digits=True,
        ),
    }
)
_DIGIT_PATTERN = re.compile(r"[0-9]")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?…])\s+")


def prepare_local_tts_chunks(
    chunks: Iterable[ScriptChunk],
    provider_id: str,
    model_id: str | None = None,
    *,
    max_chars: int | None = None,
) -> list[ScriptChunk]:
    """Apply the explicit provider/model spoken-text profile, if one exists."""
    source_chunks = list(chunks)
    if model_id is None:
        return source_chunks
    profile = LOCAL_TTS_CHUNK_PROFILES.get((provider_id, model_id))
    if profile is None:
        return source_chunks
    target_chars = profile.target_chars if max_chars is None else max_chars
    if target_chars <= 0:
        raise ValueError("local TTS max_chars must be greater than zero")

    prepared: list[ScriptChunk] = []
    for source in source_chunks:
        normalized = " ".join(source.text.split())
        if not normalized:
            continue
        if profile.reject_raw_digits and _DIGIT_PATTERN.search(normalized):
            raise ValueError(
                f"Local TTS spoken text contains raw digits in {source.id}; "
                "write numbers, dates, percentages, fractions, and versions in words"
            )
        parts = _pack_text(normalized, max_chars=target_chars)
        for part_index, text in enumerate(parts, start=1):
            chunk_id = source.id if len(parts) == 1 else f"{source.id}_part_{part_index:02d}"
            prepared.append(
                ScriptChunk(
                    number=len(prepared) + 1,
                    id=chunk_id,
                    text=text,
                )
            )
    return prepared


def merge_omnivoice_session_fragments(fragments: Iterable[ScriptChunk]) -> list[ScriptChunk]:
    """Create one OmniVoice request so its internal 420-character chunks share one session."""
    prepared = list(fragments)
    if not prepared:
        return []
    return [
        ScriptChunk(
            number=1,
            id="chunk_01_omnivoice_session",
            text=" ".join(fragment.text for fragment in prepared),
        )
    ]


def _pack_text(text: str, *, max_chars: int) -> tuple[str, ...]:
    units: list[str] = []
    for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        units.extend(_split_long_unit(sentence, max_chars=max_chars))

    packed: list[str] = []
    current_units: list[str] = []
    for unit in units:
        current = " ".join(current_units)
        candidate = unit if not current else f"{current} {unit}"
        if len(candidate) <= max_chars:
            current_units.append(unit)
            continue
        if (
            len(current_units) > 1
            and len(current_units[-1]) <= 40
            and len(f"{current_units[-1]} {unit}") <= max_chars
        ):
            packed.append(" ".join(current_units[:-1]))
            current_units = [current_units[-1], unit]
            continue
        if current_units:
            packed.append(current)
        current_units = [unit]
    if current_units:
        packed.append(" ".join(current_units))
    return tuple(packed)


def _split_long_unit(text: str, *, max_chars: int) -> tuple[str, ...]:
    if len(text) <= max_chars:
        return (text,)

    parts: list[str] = []
    current = ""
    for token in text.split():
        if len(token) > max_chars:
            raise ValueError("local TTS spoken text contains a token longer than max_chars")
        candidate = token if not current else f"{current} {token}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        parts.append(current)
        current = token
    if current:
        parts.append(current)
    return tuple(parts)
