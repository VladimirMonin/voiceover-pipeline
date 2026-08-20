"""Bounded prerecorded-audio ASR orchestration for local long-form providers."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Iterable

from .media import mp3_duration_ms
from .models import (
    ASRAlignmentOrigin,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
)
from .providers.base import ASRProvider, validate_asr_response

LONG_FORM_PROVIDER_IDS = frozenset({"qwen-local", "nemotron-local"})
LONG_FORM_TARGET_S = 110.0
LONG_FORM_HARD_MAX_S = 120.0
LONG_FORM_OVERLAP_S = 1.0
LONG_FORM_MIN_TARGET_S = 90.0
LONG_FORM_EXTRACT_DURATION_TOLERANCE_S = 0.1
_EPSILON_S = 0.02


class LongFormASRError(RuntimeError):
    """Long-form execution could not prove a complete, bounded result."""


class LongFormASRMediaError(LongFormASRError):
    """Required local media inspection or chunk extraction failed."""


@dataclass(frozen=True)
class ASRChunkPlan:
    index: int
    input_start_s: float
    input_end_s: float
    coverage_start_s: float
    coverage_end_s: float

    @property
    def input_duration_s(self) -> float:
        return self.input_end_s - self.input_start_s


def uses_long_form_orchestration(provider_id: str) -> bool:
    return provider_id in LONG_FORM_PROVIDER_IDS


def plan_prerecorded_chunks(
    source_duration_s: float,
    *,
    silence_boundaries: Iterable[float] = (),
    overlap_s: float = LONG_FORM_OVERLAP_S,
) -> tuple[ASRChunkPlan, ...]:
    """Plan source-covering, bounded inputs with an optional adjacent overlap."""
    if not isfinite(source_duration_s) or source_duration_s <= 0:
        raise LongFormASRError("source audio duration must be finite and positive")
    if not isfinite(overlap_s) or overlap_s < 0 or overlap_s >= LONG_FORM_HARD_MAX_S:
        raise LongFormASRError("long-form overlap must be finite, non-negative, and bounded")
    if source_duration_s <= LONG_FORM_HARD_MAX_S:
        return (
            ASRChunkPlan(
                index=0,
                input_start_s=0.0,
                input_end_s=source_duration_s,
                coverage_start_s=0.0,
                coverage_end_s=source_duration_s,
            ),
        )

    normalized_silences = sorted(
        boundary
        for boundary in silence_boundaries
        if isfinite(boundary) and 0.0 < boundary < source_duration_s
    )
    plans: list[ASRChunkPlan] = []
    coverage_start_s = 0.0
    while coverage_start_s < source_duration_s - _EPSILON_S:
        max_coverage_end_s = min(
            source_duration_s,
            coverage_start_s + LONG_FORM_HARD_MAX_S - (overlap_s if plans else 0.0),
        )
        target_coverage_end_s = min(
            max_coverage_end_s,
            coverage_start_s + LONG_FORM_TARGET_S,
        )
        coverage_end_s = _prefer_silence_boundary(
            normalized_silences,
            coverage_start_s=coverage_start_s,
            target_coverage_end_s=target_coverage_end_s,
            max_coverage_end_s=max_coverage_end_s,
        )
        if coverage_end_s <= coverage_start_s + _EPSILON_S:
            raise LongFormASRError("long-form chunk planner did not advance source coverage")
        input_start_s = max(0.0, coverage_start_s - (overlap_s if plans else 0.0))
        plan = ASRChunkPlan(
            index=len(plans),
            input_start_s=input_start_s,
            input_end_s=coverage_end_s,
            coverage_start_s=coverage_start_s,
            coverage_end_s=coverage_end_s,
        )
        if plan.input_duration_s > LONG_FORM_HARD_MAX_S + _EPSILON_S:
            raise LongFormASRError("long-form plan exceeds the hard model input limit")
        plans.append(plan)
        coverage_start_s = coverage_end_s

    _assert_complete_coverage(plans, source_duration_s)
    return tuple(plans)


def transcribe_prerecorded_long_form(provider: ASRProvider, request: ASRRequest) -> ASRResult:
    """Serially execute one provider over bounded source-complete audio chunks."""
    source_path = Path(request.audio_path)
    source_duration_s = _source_duration_s(source_path)
    if source_duration_s <= LONG_FORM_HARD_MAX_S:
        return provider.transcribe(request)

    ffmpeg_path = _ffmpeg_path()
    silence_boundaries = _silence_boundaries(ffmpeg_path, source_path)
    overlap_s = LONG_FORM_OVERLAP_S if request.timestamp_mode == "word" else 0.0
    plans = plan_prerecorded_chunks(
        source_duration_s,
        silence_boundaries=silence_boundaries,
        overlap_s=overlap_s,
    )
    if len(plans) < 2:
        raise LongFormASRError("long-form source was not split into bounded inputs")

    chunk_results: list[tuple[ASRChunkPlan, ASRResult, float]] = []
    with tempfile.TemporaryDirectory(prefix="voiceover-asr-") as temp_dir:
        temp_path = Path(temp_dir)
        for plan in plans:
            chunk_path = temp_path / f"chunk-{plan.index:04d}.wav"
            output_duration_s = _extract_chunk(ffmpeg_path, source_path, plan, chunk_path)
            _validate_extracted_duration(plan, output_duration_s)
            chunk_request = replace(request, audio_path=chunk_path)
            try:
                result = validate_asr_response(chunk_request, provider.transcribe(chunk_request))
            except LongFormASRError:
                raise
            except Exception as exc:
                raise LongFormASRError(f"ASR chunk {plan.index} failed: {exc}") from exc
            _validate_chunk_result(plan, output_duration_s, result)
            chunk_results.append((plan, result, output_duration_s))

    return _merge_chunk_results(
        chunk_results,
        request=request,
        source_duration_s=source_duration_s,
        silence_boundary_count=len(silence_boundaries),
        overlap_s=overlap_s,
    )


def _prefer_silence_boundary(
    silence_boundaries: list[float],
    *,
    coverage_start_s: float,
    target_coverage_end_s: float,
    max_coverage_end_s: float,
) -> float:
    minimum_boundary_s = min(
        target_coverage_end_s,
        coverage_start_s + LONG_FORM_MIN_TARGET_S,
    )
    candidates = [
        boundary
        for boundary in silence_boundaries
        if minimum_boundary_s <= boundary <= max_coverage_end_s
    ]
    if not candidates:
        return target_coverage_end_s
    return min(candidates, key=lambda boundary: abs(boundary - target_coverage_end_s))


def _assert_complete_coverage(plans: Iterable[ASRChunkPlan], source_duration_s: float) -> None:
    expected_start_s = 0.0
    last_end_s: float | None = None
    for plan in plans:
        if abs(plan.coverage_start_s - expected_start_s) > _EPSILON_S:
            raise LongFormASRError("long-form plan has an unprocessed source gap")
        if plan.input_start_s > plan.coverage_start_s + _EPSILON_S:
            raise LongFormASRError("long-form plan starts after its coverage range")
        if plan.input_end_s + _EPSILON_S < plan.coverage_end_s:
            raise LongFormASRError("long-form plan ends before its coverage range")
        expected_start_s = plan.coverage_end_s
        last_end_s = plan.coverage_end_s
    if last_end_s is None or abs(last_end_s - source_duration_s) > _EPSILON_S:
        raise LongFormASRError("long-form plan does not cover the source tail")


def _source_duration_s(source_path: Path) -> float:
    return _measured_audio_duration_s(source_path, description="source audio")


def _measured_audio_duration_s(audio_path: Path, *, description: str) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise LongFormASRMediaError(f"FFprobe is required to prove {description} duration.")
    try:
        duration_s = mp3_duration_ms(ffprobe_path, audio_path) / 1000
    except (OSError, RuntimeError, ValueError) as exc:
        raise LongFormASRMediaError(f"FFprobe could not determine {description} duration.") from exc
    if not isfinite(duration_s) or duration_s <= 0:
        raise LongFormASRMediaError(f"FFprobe returned an invalid {description} duration.")
    return duration_s


def _ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise LongFormASRMediaError("FFmpeg is required for bounded long-form ASR chunks.")
    return ffmpeg_path


def _silence_boundaries(ffmpeg_path: str, source_path: Path) -> tuple[float, ...]:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(source_path),
            "-af",
            "silencedetect=noise=-40dB:d=0.25",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise LongFormASRMediaError("FFmpeg silence analysis failed.")

    ranges: list[tuple[float, float]] = []
    start_s: float | None = None
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            start_s = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and start_s is not None:
            ranges.append((start_s, float(end_match.group(1))))
            start_s = None
    return tuple((start_s + end_s) / 2 for start_s, end_s in ranges)


def _extract_chunk(
    ffmpeg_path: str,
    source_path: Path,
    plan: ASRChunkPlan,
    output_path: Path,
) -> float:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{plan.input_start_s:.3f}",
            "-t",
            f"{plan.input_duration_s:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise LongFormASRMediaError(f"FFmpeg could not extract ASR chunk {plan.index}.")
    output_duration_s = _measured_audio_duration_s(
        output_path,
        description=f"ASR chunk {plan.index} output",
    )
    _validate_extracted_duration(plan, output_duration_s)
    return output_duration_s


def _validate_extracted_duration(plan: ASRChunkPlan, output_duration_s: float) -> None:
    if not isfinite(output_duration_s) or output_duration_s <= 0:
        raise LongFormASRMediaError(f"ASR chunk {plan.index} output duration is invalid.")
    if abs(output_duration_s - plan.input_duration_s) > LONG_FORM_EXTRACT_DURATION_TOLERANCE_S:
        raise LongFormASRMediaError(
            f"ASR chunk {plan.index} output duration is {output_duration_s:.3f}s; "
            f"expected {plan.input_duration_s:.3f}s."
        )


def _validate_chunk_result(plan: ASRChunkPlan, output_duration_s: float, result: ASRResult) -> None:
    if result.duration_s is not None and result.duration_s > output_duration_s + _EPSILON_S:
        raise LongFormASRError(f"ASR chunk {plan.index} reported duration beyond its bounded input")
    if any(
        segment.end_s is not None and segment.end_s > output_duration_s + _EPSILON_S
        for segment in result.segments
    ) or any(word.end_s > output_duration_s + _EPSILON_S for word in result.words):
        raise LongFormASRError(
            f"ASR chunk {plan.index} returned timestamp beyond its bounded input"
        )
    truncation_signal = _truncation_signal(result.execution.measurements)
    if truncation_signal is not None:
        raise LongFormASRError(
            f"ASR chunk {plan.index} reported a truncation signal: {truncation_signal}"
        )


def _truncation_signal(measurements: object) -> str | None:
    if not isinstance(measurements, Mapping):
        return None
    values: dict[str, object] = {str(key): value for key, value in measurements.items()}
    for key in (
        "truncated",
        "truncation",
        "token_limit_reached",
        "max_tokens_reached",
        "generation_truncated",
    ):
        if _measurement_is_truthy(values.get(key)):
            return key
    for generated_key, limit_key in (
        ("generated_tokens", "max_new_tokens"),
        ("generated_tokens", "max_tokens"),
    ):
        generated = values.get(generated_key)
        limit = values.get(limit_key)
        if isinstance(generated, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
            if generated >= limit:
                return f"{generated_key} reached {limit_key}"
    return None


def _measurement_is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.casefold() in {"1", "true", "yes"}
    return False


def _merge_chunk_results(
    chunk_results: list[tuple[ASRChunkPlan, ASRResult, float]],
    *,
    request: ASRRequest,
    source_duration_s: float,
    silence_boundary_count: int,
    overlap_s: float,
) -> ASRResult:
    if not chunk_results:
        raise LongFormASRError("long-form execution produced no chunk results")
    _assert_complete_coverage(
        (plan for plan, _result, _output_duration_s in chunk_results), source_duration_s
    )

    first_result = chunk_results[0][1]
    merged_words: list[ASRWordSpan] = []
    transcript_parts: list[str] = []
    segments: list[ASRSegment] = []
    word_segment_indices: list[int] = []
    evidence_chunks: list[dict[str, object]] = []
    alignment_origins: set[str] = set()
    previous_plan: ASRChunkPlan | None = None
    deduplicated_word_count = 0
    deduplicated_text_token_count = 0

    for plan, result, output_duration_s in chunk_results:
        if result.provider_id != first_result.provider_id:
            raise LongFormASRError("long-form chunks returned inconsistent provider IDs")
        if result.model_id != first_result.model_id:
            raise LongFormASRError("long-form chunks returned inconsistent model IDs")

        offset_words = tuple(
            ASRWordSpan(
                text=word.text,
                start_s=word.start_s + plan.input_start_s,
                end_s=word.end_s + plan.input_start_s,
                confidence=word.confidence,
            )
            for word in result.words
        )
        if offset_words:
            if result.alignment_origin not in {"native", "forced"}:
                raise LongFormASRError(
                    "word-timed ASR chunks need a native or forced alignment origin"
                )
            alignment_origins.add(result.alignment_origin)
            retained_words, removed_words, trimmed_existing = _deduplicate_overlap_words(
                merged_words,
                offset_words,
                previous_input_end_s=previous_plan.input_end_s if previous_plan else None,
                incoming_input_start_s=plan.input_start_s,
            )
            retained_words = _with_word_boundary_space(merged_words, retained_words)
            merged_words.extend(retained_words)
            deduplicated_word_count += removed_words + trimmed_existing
            word_segment_indices.append(len(segments))
            segments.append(
                ASRSegment(
                    text="",
                    start_s=plan.coverage_start_s,
                    end_s=plan.coverage_end_s,
                )
            )
        else:
            merged_text, removed_tokens = _merge_text_without_word_timestamps(
                transcript_parts,
                result.transcript,
                has_input_overlap=previous_plan is not None
                and plan.input_start_s < previous_plan.input_end_s - _EPSILON_S,
            )
            if merged_text:
                transcript_parts.append(merged_text)
            deduplicated_text_token_count += removed_tokens
            segments.append(
                ASRSegment(
                    text=merged_text,
                    start_s=plan.coverage_start_s,
                    end_s=plan.coverage_end_s,
                )
            )
        evidence_chunks.append(
            {
                "index": plan.index,
                "input_start_s": plan.input_start_s,
                "input_end_s": plan.input_end_s,
                "input_duration_s": plan.input_duration_s,
                "output_duration_s": output_duration_s,
                "output_duration_delta_s": output_duration_s - plan.input_duration_s,
                "output_duration_tolerance_s": LONG_FORM_EXTRACT_DURATION_TOLERANCE_S,
                "output_status": "duration_verified",
                "coverage_start_s": plan.coverage_start_s,
                "coverage_end_s": plan.coverage_end_s,
                "status": "success",
                "transcript_chars": len(result.transcript),
                "segment_count": len(result.segments),
                "word_count": len(result.words),
            }
        )
        previous_plan = plan

    if merged_words:
        for segment_index in word_segment_indices:
            segment = segments[segment_index]
            if segment.start_s is None:
                continue
            segment_start_s = segment.start_s
            next_start_s = (
                segments[segment_index + 1].start_s if segment_index + 1 < len(segments) else None
            )
            assigned_text = "".join(
                word.text
                for word in merged_words
                if word.start_s >= segment_start_s
                and (next_start_s is None or word.start_s < next_start_s)
            ).strip()
            segments[segment_index] = replace(segment, text=assigned_text)

    if len(alignment_origins) > 1:
        raise LongFormASRError("long-form chunks returned inconsistent word-timestamp origins")
    transcript = (
        "".join(word.text for word in merged_words).strip()
        if merged_words
        else " ".join(transcript_parts).strip()
    )
    if not alignment_origins:
        alignment_origin: ASRAlignmentOrigin = "chunked"
    elif alignment_origins == {"native"}:
        alignment_origin = "native"
    else:
        alignment_origin = "forced"
    measurements = dict(first_result.execution.measurements)
    processed_duration_s = sum(
        output_duration_s for _plan, _result, output_duration_s in chunk_results
    )
    planned_processed_duration_s = sum(
        plan.input_duration_s for plan, _result, _output_duration_s in chunk_results
    )
    if abs(processed_duration_s - planned_processed_duration_s) > (
        len(chunk_results) * LONG_FORM_EXTRACT_DURATION_TOLERANCE_S
    ):
        raise LongFormASRError("long-form extracted duration does not match the bounded plan")
    measurements.update(
        {
            "long_form_source_duration_s": source_duration_s,
            "long_form_processed_duration_s": processed_duration_s,
            "long_form_chunk_count": float(len(chunk_results)),
        }
    )
    long_form_evidence: dict[str, object] = {
        "source_duration_s": source_duration_s,
        "covered_duration_s": source_duration_s,
        "processed_duration_s": processed_duration_s,
        "planned_processed_duration_s": planned_processed_duration_s,
        "chunk_count": len(chunk_results),
        "chunk_target_s": LONG_FORM_TARGET_S,
        "chunk_hard_max_s": LONG_FORM_HARD_MAX_S,
        "overlap_s": overlap_s,
        "extract_duration_tolerance_s": LONG_FORM_EXTRACT_DURATION_TOLERANCE_S,
        "silence_boundary_count": silence_boundary_count,
        "coverage_verified": True,
        "deduplicated_word_count": deduplicated_word_count,
        "deduplicated_text_token_count": deduplicated_text_token_count,
        "chunks": evidence_chunks,
    }
    result = ASRResult(
        transcript=transcript,
        provider_id=first_result.provider_id,
        model_id=first_result.model_id,
        language=next(
            (
                chunk_result.language
                for _plan, chunk_result, _output_duration_s in chunk_results
                if chunk_result.language
            ),
            request.language or "",
        ),
        duration_s=source_duration_s,
        segments=tuple(segments),
        words=tuple(merged_words),
        alignment_origin=alignment_origin,
        execution=ASRExecutionReceipt(
            runtime=first_result.execution.runtime,
            runtime_version=first_result.execution.runtime_version,
            model_revision=first_result.execution.model_revision,
            resolved_device=first_result.execution.resolved_device,
            resolved_compute=first_result.execution.resolved_compute,
            measurements=measurements,
            long_form=long_form_evidence,
        ),
    )
    result.validate_timestamp_bounds(source_duration_s)
    return result


def _deduplicate_overlap_words(
    existing_words: list[ASRWordSpan],
    incoming_words: tuple[ASRWordSpan, ...],
    *,
    previous_input_end_s: float | None,
    incoming_input_start_s: float,
) -> tuple[tuple[ASRWordSpan, ...], int, int]:
    """Reconcile the chunk seam without inventing timestamps.

    Returns ``(retained_incoming, removed_incoming, trimmed_existing)``.
    ``retained_incoming`` is appended after the merged tail, ``removed_incoming``
    counts incoming duplicate-window words that were dropped, and
    ``trimmed_existing`` counts merged-tail words that had to be removed to make
    the seam strictly monotonic; trimmed words are popped in place.

    Independent ASR runs may disagree on the exact words inside the shared
    overlap, so exact text matching is used only when the paired spans overlap
    temporally and the resulting seam is strictly monotonic. Otherwise a bounded
    temporal splice drops incoming words wholly inside the previous chunk input,
    retains incoming words that cross beyond it, and trims the minimum number of
    existing tail words that intersect the overlap window. A residual overlap is
    never accepted, even when smaller than ``_EPSILON_S``; if a valid splice
    would require discarding pre-overlap content, the seam fails closed.
    """
    if previous_input_end_s is None or not incoming_words or not existing_words:
        return incoming_words, 0, 0
    if incoming_words[0].start_s >= existing_words[-1].end_s:
        return incoming_words, 0, 0

    overlap_existing = [
        word for word in existing_words if word.end_s > incoming_input_start_s + _EPSILON_S
    ]
    maximum = min(len(overlap_existing), len(incoming_words))
    for count in range(maximum, 0, -1):
        candidate_words = incoming_words[:count]
        if candidate_words[-1].end_s > previous_input_end_s + _EPSILON_S:
            continue
        paired = list(zip(overlap_existing[-count:], candidate_words, strict=True))
        if not all(
            _word_key(existing.text) == _word_key(incoming.text) for existing, incoming in paired
        ):
            continue
        if not all(_spans_overlap(existing, incoming) for existing, incoming in paired):
            continue
        retained_words = incoming_words[count:]
        if retained_words and retained_words[0].start_s < existing_words[-1].end_s:
            continue
        return retained_words, count, 0

    retained_incoming = tuple(word for word in incoming_words if word.end_s > previous_input_end_s)
    removed_incoming = len(incoming_words) - len(retained_incoming)
    if not retained_incoming:
        return (), removed_incoming, 0
    trimmed_existing = 0
    while retained_incoming[0].start_s < existing_words[-1].end_s:
        word = existing_words[-1]
        if word.start_s < incoming_input_start_s:
            raise LongFormASRError(
                "cannot reconcile long-form chunk seam without discarding pre-overlap content"
            )
        existing_words.pop()
        trimmed_existing += 1
        if not existing_words:
            break
    return retained_incoming, removed_incoming, trimmed_existing


def _spans_overlap(first: ASRWordSpan, second: ASRWordSpan) -> bool:
    return first.start_s < second.end_s and second.start_s < first.end_s


def _with_word_boundary_space(
    existing_words: list[ASRWordSpan], incoming_words: tuple[ASRWordSpan, ...]
) -> tuple[ASRWordSpan, ...]:
    if not existing_words or not incoming_words:
        return incoming_words
    previous_text = existing_words[-1].text
    incoming_text = incoming_words[0].text
    if (
        previous_text
        and incoming_text
        and not previous_text[-1].isspace()
        and not incoming_text[0].isspace()
    ):
        return (replace(incoming_words[0], text=f" {incoming_text}"), *incoming_words[1:])
    return incoming_words


def _merge_text_without_word_timestamps(
    existing_parts: list[str],
    incoming_text: str,
    *,
    has_input_overlap: bool,
) -> tuple[str, int]:
    incoming_tokens = re.findall(r"\S+", incoming_text)
    if not incoming_tokens:
        return "", 0
    if not existing_parts or not has_input_overlap:
        return " ".join(incoming_tokens), 0
    existing_tokens = re.findall(r"\S+", " ".join(existing_parts))
    maximum = min(len(existing_tokens), len(incoming_tokens))
    for count in range(maximum, 0, -1):
        if [token.casefold() for token in existing_tokens[-count:]] == [
            token.casefold() for token in incoming_tokens[:count]
        ]:
            raise LongFormASRError(
                "cannot prove text-only boundary overlap without word timestamps; "
                "retry with word timestamps"
            )
    return " ".join(incoming_tokens), 0


def _word_key(text: str) -> str:
    key = "".join(character for character in text.casefold() if character.isalnum())
    return key or text.casefold().strip()
