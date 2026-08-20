from __future__ import annotations

from typing import Any

from voiceover_pipeline.models import ASRResult, TimingResult, TimingSegment


class ASRTimingBridgeError(ValueError):
    """ASR output cannot safely become a timing artifact."""


def asr_result_to_timing(
    result: ASRResult,
    *,
    source_audio: str = "",
    source_duration_s: float | None = None,
    max_words_per_segment: int = 8,
    max_duration_seconds: float = 5.0,
) -> TimingResult:
    """Convert validated word spans into the existing timing/SRT data model."""
    if not isinstance(result, ASRResult):
        raise ASRTimingBridgeError(
            "ASR timing bridge requires an ASRResult with an alignment origin"
        )
    if not result.words and result.transcript.strip():
        raise ASRTimingBridgeError(
            "ASR timing bridge requires word timestamps; text-only results cannot produce SRT"
        )
    if result.alignment_origin not in ("native", "forced"):
        raise ASRTimingBridgeError("ASR timing bridge requires a native or forced alignment origin")
    effective_duration_s = source_duration_s if source_duration_s is not None else result.duration_s
    if effective_duration_s is None:
        raise ASRTimingBridgeError(
            "ASR timing bridge requires a source duration for timestamp bounds"
        )
    try:
        result.validate_timestamp_bounds(effective_duration_s)
    except ValueError as exc:
        raise ASRTimingBridgeError(str(exc)) from exc
    if max_words_per_segment < 1:
        raise ValueError("max_words_per_segment must be positive")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")

    segments: list[TimingSegment] = []
    current: list[Any] = []
    start_s = 0.0
    for word in result.words:
        if current and (
            len(current) >= max_words_per_segment or word.end_s - start_s > max_duration_seconds
        ):
            segments.append(_segment_from_words(len(segments) + 1, current))
            current = []
        if not current:
            start_s = word.start_s
        current.append(word)
    if current:
        segments.append(_segment_from_words(len(segments) + 1, current))

    return TimingResult(
        segments=segments,
        model=result.model_id,
        backend=result.execution.runtime,
        provider=result.provider_id,
        device=result.execution.resolved_device,
        compute_type=result.execution.resolved_compute,
        language=result.language,
        source_audio=source_audio,
    )


def _segment_from_words(segment_id: int, words: list[Any]) -> TimingSegment:
    start_s = words[0].start_s
    end_s = words[-1].end_s
    start_ms = round(start_s * 1000)
    end_ms = round(end_s * 1000)
    if end_ms < start_ms:
        raise ASRTimingBridgeError("ASR word spans must be monotonic before timing conversion")
    return TimingSegment(
        id=segment_id,
        start_sec=start_s,
        end_sec=end_s,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        text=" ".join(word.text.strip() for word in words).strip(),
        words=[
            {
                "text": word.text,
                "start_s": word.start_s,
                "end_s": word.end_s,
                **({"confidence": word.confidence} if word.confidence is not None else {}),
            }
            for word in words
        ],
    )
