"""Paid execution loop plus raw-recovery helpers for the CLI entry point.

This module owns the single prepared-part execution loop: the durable attempt
marker written before a paid provider submit, the accepted raw audio plus its
bounded receipt before any FFmpeg step, the conversion/trim/duration/artifact
sequence, and the completed upsert that clears the marker in the same atomic
state write. It also persists the paid safety evidence a provider already
produced (the accepted media task id before the first poll or download, the
observed cost before the signed-URL download) and rebuilds a chunk from those
already-paid raw bytes.

The loop calls the CLI's call-time compatibility and presentation seams through
an explicit ``PartExecutionHooks`` object rather than importing ``cli``: tests
patch ``cli.synthesize_part`` and the media functions, and the JSON events,
retry logging, fail envelope, and progress output stay CLI-owned. The raw
receipt digest reuses ``services.recovery``'s read-only streaming SHA-256 helper
rather than duplicating the hashing logic here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from ..models import ChunkArtifact, ScriptChunk, SynthesisResult
from ..providers import PolzaTTSProvider
from ..retry import RetryPolicy, run_with_retry
from ..run_state import (
    ATTEMPT_FAILED,
    ATTEMPT_OUTCOME_UNKNOWN,
    GenerationLogger,
    append_error,
    atomic_write_json,
    begin_chunk_attempt,
    clear_chunk_attempt,
    pending_raw_recovery,
    raw_audio_relative_path,
    record_chunk_attempt_outcome,
    record_media_observed_cost,
    record_media_task_accepted,
    record_raw_audio_saved,
    unconfirmed_attempt,
    upsert_completed_chunk,
)
from . import costs
from .prepare import PreparedPart, PreparedRun
from .recovery import _sha256_file


def bind_polza_media_attempt(
    provider: Any,
    *,
    state: dict[str, Any],
    state_path: Path,
    logger: GenerationLogger,
    chunk: ScriptChunk,
) -> None:
    """Bind the marker-persisting callbacks for this paid media chunk.

    ``on_media_task_accepted`` stores the accepted id atomically before the
    first poll or download, and ``on_media_completed`` stores the exact cost the
    completed poll reported before the signed-URL download, which can fail on
    its own. Both write only bounded marker fields, and both writes are what let
    a later ``--resume`` finish this chunk with GET calls instead of a POST.
    """

    def on_accepted(task_id: str) -> None:
        record_media_task_accepted(state, task_id=task_id, chunk_id=chunk.id, number=chunk.number)
        atomic_write_json(state_path, state)
        logger.event("info", "remote_accepted", chunk=chunk.number, id=chunk.id, task_id=task_id)

    def on_completed(_task_id: str, usage: dict | None, _generation_id: str | None) -> None:
        cost, cost_exact = costs.media_observed_cost(usage)
        if cost is None:
            return
        record_media_observed_cost(state, cost=cost, cost_exact=cost_exact)
        atomic_write_json(state_path, state)
        logger.event("info", "cost_observed", chunk=chunk.number, id=chunk.id, cost=cost)

    provider.on_media_task_accepted = on_accepted
    provider.on_media_completed = on_completed


def persist_paid_raw_audio(
    result: SynthesisResult,
    *,
    state: dict[str, Any],
    state_path: Path,
    paths,
    chunk: ScriptChunk,
    logger: GenerationLogger,
) -> None:
    """Write the accepted paid bytes to a run-local raw file and record the receipt.

    The raw file is the rebuild source rather than a disposable cache: it is
    written atomically before any FFmpeg step and survives a later conversion
    failure. Only the bounded format, the deterministic path, the digest, a
    bounded generation id, and the recognized direct cost join the attempt marker
    atomically, so a crash leaves either the whole receipt or none of it. A
    failure propagates to keep the paid outcome unconfirmed instead of allowing
    another submit.
    """
    relative_path = raw_audio_relative_path(chunk.id, result.audio_format)
    raw_path = paths.output_root / relative_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    temp_path.write_bytes(result.audio_bytes)
    temp_path.replace(raw_path)
    record_raw_audio_saved(
        state,
        chunk_id=chunk.id,
        number=chunk.number,
        audio_format=result.audio_format,
        relative_path=relative_path,
        sha256=_sha256_file(raw_path),
        generation_id=result.generation_id,
    )
    # The synchronous ``/audio/speech`` route reports its billed amount only in
    # this response, so it joins the same atomic write as the raw receipt; a later
    # conversion failure therefore cannot lose the observed cost. A response
    # without a recognized cost leaves an already stored observation untouched.
    cost, cost_exact = costs.media_observed_cost((result.raw_metadata or {}).get("usage_direct"))
    if cost is not None:
        record_media_observed_cost(state, cost=cost, cost_exact=cost_exact)
    atomic_write_json(state_path, state)
    logger.event(
        "info",
        "raw_audio_saved",
        chunk=chunk.number,
        id=chunk.id,
        format=result.audio_format,
        bytes=len(result.audio_bytes),
    )


def raw_recovery_result(
    args, chunk: ScriptChunk, run_root: Path, recovery: dict[str, Any]
) -> SynthesisResult:
    """Rebuild a SynthesisResult from saved raw paid bytes with no provider call.

    The bytes were already paid for, so a resume reads them from disk instead of
    issuing another POST or a media GET. The transcript is the exact submitted
    chunk text, and the stored exact cost is applied separately from the marker,
    so no unbounded provider payload is reconstructed.
    """
    audio_bytes = (run_root / recovery["raw_path"]).read_bytes()
    return SynthesisResult(
        audio_bytes=audio_bytes,
        audio_format=recovery["raw_format"],
        transcript=chunk.text,
        generation_id=recovery["generation_id"],
        client_path="requests",
        raw_metadata={
            "voice": args.voice,
            "provider": args.provider,
            "model": args.model,
        },
    )


@dataclass
class PartExecutionHooks:
    """The CLI's call-time compatibility and presentation callables.

    The part loop runs in this module, but these functions must stay the ones
    bound in ``cli`` at call time: tests patch ``cli.synthesize_part`` and the
    media functions, and the JSON events, retry log, fail envelope, and progress
    text are CLI-owned presentation seams rather than execution logic.
    ``fail_provider`` and ``fail_output`` are the CLI ``fail`` bound to its
    provider and output semantic exit codes.
    """

    synthesize_part: Callable[[Any, PreparedPart], SynthesisResult]
    write_audio_as_mp3: Callable[[str, bytes, str, Path], None]
    trim_final_silence: Callable[[str, str, Path], None]
    mp3_duration_ms: Callable[[str, Path], int]
    fail_provider: Callable[[str], NoReturn]
    fail_output: Callable[[str], NoReturn]
    emit_json_event: Callable[..., None]
    log_retry: Callable[..., None]
    reject_unconfirmed_paid_resume: Callable[..., NoReturn]
    paid_submit_attempt_status: Callable[[BaseException], str]
    public_projection: Callable[[SynthesisResult], dict[str, Any]]
    progress: Callable[[str], None]


@dataclass
class PartLoopState:
    """Mutable run-scoped execution state the CLI reads after the part loop.

    ``chunk_artifacts_by_number`` gains one artifact per saved chunk (or keeps the
    resumed one), and ``total_duration_ms`` advances by each chunk's duration plus
    its dialogue pause so the next chunk's start offset stays contiguous. The loop
    mutates this one object instead of returning loosely related values.
    """

    chunk_artifacts_by_number: dict[int, ChunkArtifact]
    total_duration_ms: int


def execute_prepared_parts(
    *,
    args,
    prepared: PreparedRun,
    chunks: list[ScriptChunk],
    paths,
    ffmpeg_path: str,
    ffprobe_path: str,
    state: dict[str, Any],
    state_path: Path,
    logger: GenerationLogger,
    provider: Any,
    paid_submit: bool,
    retry_policy: RetryPolicy,
    recoverable_attempts: dict[int, dict[str, Any]],
    dialogue_run: bool,
    completed: set[int],
    loop_state: PartLoopState,
    hooks: PartExecutionHooks,
) -> None:
    """Run one prepared-part loop with the paid-safety ordering intact.

    Every chunk follows the same sequence as the former ``cli._generate_step``
    body: skip a resumed chunk, print/emit start, write the durable paid attempt
    marker before any provider request, submit through the retry policy, persist
    the accepted raw bytes before FFmpeg, convert/trim/digest, then upsert the
    completed artifact and clear the marker in one atomic state write. Recovery
    for a known raw or media attempt replaces the submit with local bytes or GET
    calls only, so no chunk can send a second POST.
    """
    chunk_artifacts_by_number = loop_state.chunk_artifacts_by_number

    for part in prepared.parts:
        chunk = part.chunk
        output_path = paths.chunks_dir / f"{chunk.id}.mp3"
        if chunk.number in completed and output_path.exists():
            if dialogue_run:
                completed_artifact = chunk_artifacts_by_number.get(chunk.number)
                if (
                    completed_artifact is None
                    or completed_artifact.audio_sha256 is None
                    or completed_artifact.audio_sha256 != _sha256_file(output_path)
                ):
                    logger.event("error", "resume_rejected", reason="dialogue_audio_hash_mismatch")
                    hooks.fail_provider(
                        "Cannot resume: dialogue audio does not match the trusted run state."
                    )
            logger.event(
                "info",
                "chunk_skipped_resume",
                chunk=chunk.number,
                id=chunk.id,
                file=output_path.name,
            )
            hooks.emit_json_event(
                args, "chunk_skipped", chunk=chunk.number, id=chunk.id, reason="resume"
            )
            continue
        if not args.json_output:
            hooks.progress(f"Generating {chunk.id}/{len(chunks):02d}: {output_path.name}")
        logger.event(
            "info", "chunk_started", chunk=chunk.number, id=chunk.id, file=output_path.name
        )
        hooks.emit_json_event(args, "chunk_started", chunk=chunk.number, id=chunk.id)

        recovery = recoverable_attempts.get(chunk.number)
        if recovery is not None:
            recovery_event = (
                "paid_raw_recovery" if recovery.get("kind") == "raw" else "paid_media_recovery"
            )
            logger.event("info", recovery_event, chunk=chunk.number, id=chunk.id)

        def synthesize_current_chunk():
            if recovery is not None:
                if recovery.get("kind") == "raw":
                    # The paid bytes are already on disk, so this rebuilds the
                    # chunk with no provider request at all.
                    return raw_recovery_result(args, chunk, paths.output_root, recovery)
                # The paid submit for this chunk already happened, so only GET
                # calls may finish it; no POST is sent a second time.
                return provider.recover_media_task(recovery["remote_task_id"], chunk.text, chunk.id)
            return hooks.synthesize_part(provider, part)

        if paid_submit and isinstance(provider, PolzaTTSProvider):
            bind_polza_media_attempt(
                provider, state=state, state_path=state_path, logger=logger, chunk=chunk
            )
        if paid_submit and recovery is None:
            existing_attempt = unconfirmed_attempt(state)
            if existing_attempt is not None:
                # A marker for another chunk holds a known paid attempt that this
                # write would erase, so fail closed with the documented envelope
                # instead of risking a second POST for that chunk.
                hooks.reject_unconfirmed_paid_resume(existing_attempt, logger)
            # The marker is durable before the request leaves, so a crash between
            # this write and the POST resumes as unconfirmed instead of repeating
            # a possibly billed submit. A recovered attempt keeps its own marker,
            # which still binds the accepted id and any observed cost.
            begin_chunk_attempt(state, chunk_id=chunk.id, number=chunk.number)
            atomic_write_json(state_path, state)
            logger.event("info", "paid_submit_started", chunk=chunk.number, id=chunk.id)
        try:
            result = run_with_retry(
                synthesize_current_chunk,
                policy=retry_policy,
                on_retry=lambda attempt, error, delay: hooks.log_retry(
                    logger, args, chunk, attempt, error, delay
                ),
            )
        except Exception as e:
            paid_attempt_status: str | None = None
            if paid_submit:
                paid_attempt_status = hooks.paid_submit_attempt_status(e)
                record_chunk_attempt_outcome(state, status=paid_attempt_status)
            append_error(state, chunk_id=chunk.id, message=str(e))
            atomic_write_json(state_path, state)
            logger.event("error", "chunk_failed", chunk=chunk.number, id=chunk.id, error=str(e))
            if paid_attempt_status is not None:
                logger.event(
                    "warn",
                    "paid_submit_failed",
                    chunk=chunk.number,
                    id=chunk.id,
                    status=paid_attempt_status,
                )
            hooks.emit_json_event(
                args, "chunk_failed", chunk=chunk.number, id=chunk.id, error=str(e)
            )
            hooks.fail_provider(f"Failed to synthesize {chunk.id}: {e}")
        logger.event(
            "info",
            "chunk_provider_response",
            chunk=chunk.number,
            id=chunk.id,
            generation_id=result.generation_id,
        )
        if paid_submit:
            try:
                persist_paid_raw_audio(
                    result,
                    state=state,
                    state_path=state_path,
                    paths=paths,
                    chunk=chunk,
                    logger=logger,
                )
            except Exception as e:
                # The paid bytes could not be persisted, so the outcome stays
                # unconfirmed and no second POST may be sent; the marker is kept
                # as evidence and the error is reported honestly.
                record_chunk_attempt_outcome(state, status=ATTEMPT_OUTCOME_UNKNOWN)
                append_error(state, chunk_id=chunk.id, message=str(e))
                atomic_write_json(state_path, state)
                logger.event(
                    "error", "chunk_raw_save_failed", chunk=chunk.number, id=chunk.id, error=str(e)
                )
                hooks.emit_json_event(
                    args, "chunk_failed", chunk=chunk.number, id=chunk.id, error=str(e)
                )
                hooks.fail_output(f"Failed to save raw paid audio {chunk.id}: {e}")
        try:
            hooks.write_audio_as_mp3(
                ffmpeg_path, result.audio_bytes, result.audio_format, output_path
            )
        except Exception as e:
            if paid_submit and pending_raw_recovery(state) is None:
                # Without a saved raw receipt the paid response is lost, so the
                # attempt is a definite failure; with one it stays recoverable.
                record_chunk_attempt_outcome(state, status=ATTEMPT_FAILED)
            append_error(state, chunk_id=chunk.id, message=str(e))
            atomic_write_json(state_path, state)
            logger.event(
                "error", "chunk_write_failed", chunk=chunk.number, id=chunk.id, error=str(e)
            )
            hooks.fail_output(f"Failed to write chunk audio {output_path}: {e}")
        logger.event(
            "info", "chunk_file_saved", chunk=chunk.number, id=chunk.id, file=output_path.name
        )
        if not args.no_trim:
            try:
                hooks.trim_final_silence(ffmpeg_path, ffprobe_path, output_path)
                logger.event("info", "chunk_trimmed", chunk=chunk.number, id=chunk.id)
            except Exception as e:
                if paid_submit and pending_raw_recovery(state) is None:
                    record_chunk_attempt_outcome(state, status=ATTEMPT_FAILED)
                append_error(state, chunk_id=chunk.id, message=str(e))
                atomic_write_json(state_path, state)
                logger.event(
                    "error", "chunk_trim_failed", chunk=chunk.number, id=chunk.id, error=str(e)
                )
                hooks.fail_output(f"Failed to trim chunk audio {output_path}: {e}")

        duration_ms = hooks.mp3_duration_ms(ffprobe_path, output_path)
        start_ms = loop_state.total_duration_ms
        end_ms = start_ms + duration_ms
        loop_state.total_duration_ms = end_ms + (chunk.pause_after_ms if dialogue_run else 0)

        direct_cost_kwargs = costs.direct_cost_kwargs(args.provider, result)
        if not direct_cost_kwargs and recovery is not None:
            # A recovered attempt can see a completion payload without usage, so
            # the exact cost stored before the failed download stays observed.
            direct_cost_kwargs = costs.recovered_attempt_cost_kwargs(recovery)
        artifact = ChunkArtifact(
            number=chunk.number,
            id=chunk.id,
            file=output_path.name,
            duration_ms=duration_ms,
            duration_sec=round(duration_ms / 1000, 3),
            start_ms=start_ms,
            end_ms=end_ms,
            text_characters=len(chunk.text),
            transcript=None if dialogue_run else result.transcript,
            client_path=result.client_path,
            generation_id=result.generation_id,
            speaker=chunk.speaker,
            voice=part.voice or prepared.voice,
            voice_fingerprint=chunk.voice_fingerprint,
            turn_index=chunk.number if dialogue_run else None,
            speech_duration_ms=duration_ms if dialogue_run else None,
            audio_sha256=_sha256_file(output_path) if dialogue_run else None,
            pause_after_ms=chunk.pause_after_ms,
            **hooks.public_projection(result),
            **direct_cost_kwargs,
        )
        chunk_artifacts_by_number[chunk.number] = artifact
        upsert_completed_chunk(
            state,
            artifact=artifact,
            model=args.model,
            voice=part.voice or prepared.voice,
            text=chunk.text,
            include_text=not dialogue_run,
            include_transcript=not dialogue_run,
        )
        if paid_submit:
            # The completed entry and the cleared marker land in the same atomic
            # write, so a saved chunk always replaces its own attempt marker.
            clear_chunk_attempt(state)
        atomic_write_json(state_path, state)
        logger.event(
            "info", "chunk_state_saved", chunk=chunk.number, id=chunk.id, state=state_path.name
        )
        hooks.emit_json_event(
            args,
            "chunk_saved",
            chunk=chunk.number,
            id=chunk.id,
            file=output_path.name,
            duration_ms=duration_ms,
        )
        if not args.json_output:
            hooks.progress(f"Saved {output_path.name}: {duration_ms} ms")
