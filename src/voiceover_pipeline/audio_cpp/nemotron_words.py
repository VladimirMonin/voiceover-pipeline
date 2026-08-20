from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from voiceover_pipeline.models import ASRWordSpan

_METASPACE = "▁"


@dataclass
class _OpenWord:
    text_parts: list[str]
    start_s: float
    end_s: float


def normalize_nemotron_word_timestamps(
    raw_entries: Sequence[Mapping[str, object]], *, response_offset_s: float = 0.0
) -> tuple[ASRWordSpan, ...]:
    """Merge emitted SentencePiece chunks into canonical VOP word spans.

    The runtime emits tokenizer chunks, possibly with several entries on the same
    RNN-T frame. A leading SentencePiece metaspace or the ASCII space emitted by
    audio.cpp starts a new canonical word; punctuation without either marker stays
    attached to the preceding word. Entries
    outside the driver's explicit keep span are retained in the raw receipt but
    excluded from the canonical word result.
    """

    offset = _finite_number(response_offset_s, "response_offset_s")
    words: list[ASRWordSpan] = []
    current: _OpenWord | None = None

    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Nemotron word timestamp {index} must be an object")
        keep = entry.get("keep", True)
        if not isinstance(keep, bool):
            raise ValueError(f"Nemotron word timestamp {index} keep must be a boolean")
        if not keep:
            continue

        text = entry.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Nemotron word timestamp {index} text must be a non-empty string")
        start_s = (
            offset
            + _entry_offset(entry, index)
            + _finite_number(entry.get("start_s"), f"word timestamp {index} start_s")
        )
        end_s = (
            offset
            + _entry_offset(entry, index)
            + _finite_number(entry.get("end_s"), f"word timestamp {index} end_s")
        )
        if end_s < start_s:
            raise ValueError(f"Nemotron word timestamp {index} end_s must not be before start_s")

        starts_word = text.startswith((_METASPACE, " "))
        piece = text.lstrip(_METASPACE) if text.startswith(_METASPACE) else text.removeprefix(" ")
        if _METASPACE in piece or any(character.isspace() for character in piece):
            raise ValueError(
                f"Nemotron word timestamp {index} must contain one SentencePiece chunk without whitespace"
            )
        if starts_word and current is not None:
            _finish_word(words, current, add_trailing_space=True)
            current = None
        if not piece:
            continue
        if current is None:
            current = _OpenWord(text_parts=[piece], start_s=start_s, end_s=end_s)
        else:
            current.text_parts.append(piece)
            current.end_s = max(current.end_s, end_s)

    if current is not None:
        _finish_word(words, current, add_trailing_space=False)
    return tuple(words)


def _finish_word(words: list[ASRWordSpan], current: _OpenWord, *, add_trailing_space: bool) -> None:
    text = "".join(current.text_parts)
    if not text:
        return
    if words and current.start_s < words[-1].end_s:
        raise ValueError("Nemotron canonical word timestamps must be monotonic")
    try:
        words.append(
            ASRWordSpan(
                text=f"{text} " if add_trailing_space else text,
                start_s=current.start_s,
                end_s=current.end_s,
            )
        )
    except ValueError as exc:
        raise ValueError(f"Nemotron canonical word timestamp is invalid: {exc}") from exc


def _entry_offset(entry: Mapping[str, object], index: int) -> float:
    return _finite_number(
        entry.get("chunk_offset_s", 0.0), f"word timestamp {index} chunk_offset_s"
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"Nemotron {label} must be a finite number")
    return float(value)
