"""Fail-closed mapping from observed ASR spans onto the canonical transcript text.

Plan section 7 stage S07 saves the transcript an ASR command actually produced
together with the timing it actually observed. This module is the single
translation between the two:

* :func:`build_observed_timing` turns an :class:`ASRResult`'s observed spans --
  its word spans when it has them, else its provider segment spans -- into a
  bounded, JSON-shaped provenance block: the alignment ``origin``, the ``unit``
  (always ``ms``), the ``model``/``model_path``/``model_revision`` that produced
  the spans, and a ``spans`` list of integer millisecond ranges mapped onto
  character offsets of the canonical transcript.
* :func:`observed_spans_from_snapshot` reads that block back from canonical
  SQLite, and :func:`chunk_time_range` intersects the stored character spans with
  a search chunk's character window.

It never interpolates. A span is recorded only when its own text can be located
unambiguously in the canonical transcript; a span without a real observed bound
keeps no entry; and a text-only result maps to no span at all, so both the run
provenance and every derived FTS row stay ``NULL`` rather than gaining an invented
timestamp. Reading is equally fail-closed: a malformed or truncated block yields
no spans instead of corrupting the index.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .models import ASRResult

# The snapshot key that holds the whole provenance block, so a reader never has
# to infer the unit or origin from an unrelated field.
OBSERVED_SPANS_KEY = "observed_spans"
# Timestamps are recorded in integer milliseconds only; ``seconds`` survives only
# in the command payload, never in the canonical timing provenance.
TIMING_UNIT_MS = "ms"
# The origin recorded for a result that observed no timing at all.
ORIGIN_NONE = "none"


@dataclass(frozen=True)
class ObservedTimingSpan:
    """One observed time range placed on the canonical transcript characters.

    ``char_start``/``char_end`` are offsets into the exact stored transcript text
    (``char_end`` exclusive) and ``start_ms``/``end_ms`` are the observed
    millisecond bounds. The four values are stored together so a later reader can
    intersect them with a chunk window without re-guessing the alignment.
    """

    char_start: int
    char_end: int
    start_ms: int
    end_ms: int


def seconds_to_ms(seconds: float | None) -> int | None:
    """Return an observed second bound as an integer millisecond, or ``None``.

    Only a finite, non-negative observed value converts; an absent or invalid
    bound stays ``None`` instead of becoming a fabricated zero.
    """
    if seconds is None or not isfinite(seconds) or seconds < 0:
        return None
    return round(seconds * 1000)


def build_observed_timing(result: ASRResult) -> dict[str, Any]:
    """Return the bounded timing provenance block for one ASR result.

    The block always carries the observed alignment origin, the ``ms`` unit, and
    the producing ``model``/``model_path``/``model_revision``. ``spans`` holds only
    the observed spans that map cleanly onto ``result.transcript``; a text-only
    result yields an empty list, not an invented range.
    """
    return {
        "origin": result.alignment_origin or ORIGIN_NONE,
        "unit": TIMING_UNIT_MS,
        "model": result.model_id,
        "model_path": result.execution.model_path,
        "model_revision": result.execution.model_revision,
        "source_text_sha256": hashlib.sha256(result.transcript.encode("utf-8")).hexdigest(),
        "spans": [_span_payload(span) for span in _observed_spans(result, result.transcript)],
    }


def observed_spans_from_snapshot(
    snapshot: object, *, source_text: str | None = None
) -> tuple[ObservedTimingSpan, ...]:
    """Read only spans bound to the exact canonical ASR transcript.

    Missing or malformed unit, origin, hash, offsets, or time ranges fail closed.
    The optional source text check is mandatory at the FTS indexing boundary:
    another text source in the same run must never inherit these speech times.
    """
    if not isinstance(snapshot, dict):
        return ()
    block = snapshot.get(OBSERVED_SPANS_KEY)
    if not isinstance(block, dict) or block.get("unit") != TIMING_UNIT_MS:
        return ()
    origin = block.get("origin")
    if origin not in {"native", "forced", "chunked", ORIGIN_NONE}:
        return ()
    digest = block.get("source_text_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return ()
    if (
        source_text is not None
        and digest != hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    ):
        return ()
    raw_spans = block.get("spans")
    if not isinstance(raw_spans, list) or (raw_spans and origin not in {"native", "forced"}):
        return ()
    spans: list[ObservedTimingSpan] = []
    for entry in raw_spans:
        if not isinstance(entry, dict):
            return ()
        char_start = entry.get("char_start")
        char_end = entry.get("char_end")
        start_ms = entry.get("start_ms")
        end_ms = entry.get("end_ms")
        if not all(_is_plain_int(value) for value in (char_start, char_end, start_ms, end_ms)):
            return ()
        assert isinstance(char_start, int)
        assert isinstance(char_end, int)
        assert isinstance(start_ms, int)
        assert isinstance(end_ms, int)
        if char_start < 0 or char_end <= char_start or start_ms < 0 or end_ms < start_ms:
            return ()
        if spans and (char_start < spans[-1].char_end or start_ms < spans[-1].start_ms):
            return ()
        if source_text is not None and char_end > len(source_text):
            return ()
        spans.append(
            ObservedTimingSpan(
                char_start=char_start,
                char_end=char_end,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return tuple(spans)


def chunk_time_range(
    spans: tuple[ObservedTimingSpan, ...], *, char_start: int, char_end: int, chunk_text: str
) -> tuple[int | None, int | None]:
    """Return a range only when every speech character has an observed span.

    Whitespace and punctuation may separate timed words, but an untimed word or
    a word cut by the chunk boundary makes the whole chunk's range unknown. A
    subset of timed words must not be reported as the timing of the full chunk.
    """
    if char_start < 0 or char_end <= char_start or len(chunk_text) != char_end - char_start:
        return None, None
    contained = [
        span for span in spans if char_start <= span.char_start and span.char_end <= char_end
    ]
    if not contained:
        return None, None
    covered = [False] * len(chunk_text)
    for span in contained:
        for index in range(span.char_start - char_start, span.char_end - char_start):
            covered[index] = True
    if any(
        any(folded.isalnum() for folded in character.casefold()) and not covered[index]
        for index, character in enumerate(chunk_text)
    ):
        return None, None
    return min(span.start_ms for span in contained), max(span.end_ms for span in contained)


def _observed_spans(result: ASRResult, transcript: str) -> tuple[ObservedTimingSpan, ...]:
    """Map only native/forced word or segment spans onto ``transcript``."""
    if result.alignment_origin not in {"native", "forced"}:
        return ()
    if result.words:
        units: Sequence[tuple[str, float | None, float | None]] = [
            (word.text, word.start_s, word.end_s) for word in result.words
        ]
    elif result.segments:
        units = [(segment.text, segment.start_s, segment.end_s) for segment in result.segments]
    else:
        return ()
    return _map_units(units, transcript)


def _map_units(
    units: Sequence[tuple[str, float | None, float | None]], transcript: str
) -> tuple[ObservedTimingSpan, ...]:
    """Walk observed units over the transcript and place each observed time.

    The whole walk is trusted only when the concatenated unit text reconstructs
    the transcript exactly under the same alphanumeric fold that
    :class:`ASRResult` already validates. Otherwise the mapping returns no spans:
    a partial or reordered alignment can never place an invented range. A unit
    whose own text maps cleanly but that observed no time contributes no span while
    still consuming its characters, so a mixed timed/untimed result keeps the timed
    intervals and leaves the rest ``NULL``.
    """
    normalized, offsets = _normalized_with_offsets(transcript)
    if not normalized:
        return ()
    spans: list[ObservedTimingSpan] = []
    position = 0
    for text, start_s, end_s in units:
        unit_normalized = _normalize(text)
        end_position = position + len(unit_normalized)
        if normalized[position:end_position] != unit_normalized:
            return ()
        if unit_normalized:
            start_ms = seconds_to_ms(start_s)
            end_ms = seconds_to_ms(end_s)
            if start_ms is not None and end_ms is not None and end_ms >= start_ms:
                spans.append(
                    ObservedTimingSpan(
                        char_start=offsets[position],
                        char_end=offsets[end_position - 1] + 1,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
        position = end_position
    if position != len(normalized):
        return ()
    return tuple(spans)


def _normalize(text: str) -> str:
    """Return the alphanumeric casefold used by :class:`ASRResult` validation."""
    return "".join(character for character in text.casefold() if character.isalnum())


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    """Return the folded text plus a folded-index -> original-index offset map.

    ``casefold`` can expand one character into several, so each folded character
    maps back to the original index it came from.
    """
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        for folded in character.casefold():
            if folded.isalnum():
                characters.append(folded)
                offsets.append(index)
    return "".join(characters), offsets


def _span_payload(span: ObservedTimingSpan) -> dict[str, int]:
    return {
        "char_start": span.char_start,
        "char_end": span.char_end,
        "start_ms": span.start_ms,
        "end_ms": span.end_ms,
    }


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
