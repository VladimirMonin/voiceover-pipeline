"""Paid execution and raw-recovery helpers for the CLI entry point.

These helpers persist the paid safety evidence a provider already produced: the
accepted media task id before the first poll or download, the exact observed cost
before the signed-URL download, and the accepted raw audio plus a bounded receipt
before any FFmpeg step. They also rebuild a chunk from those already-paid raw
bytes. ``cli`` keeps thin wrappers under the former names so existing call sites,
imports, and monkeypatches stay identical, and the submit seam, the marker write
before the request, and every provider call still live in ``cli``. The raw
receipt digest reuses ``services.recovery``'s read-only streaming SHA-256 helper
rather than duplicating the hashing logic here.
"""

from pathlib import Path
from typing import Any

from ..models import ScriptChunk, SynthesisResult
from ..run_state import (
    GenerationLogger,
    atomic_write_json,
    raw_audio_relative_path,
    record_media_observed_cost,
    record_media_task_accepted,
    record_raw_audio_saved,
)
from .costs import media_observed_cost
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
        cost, cost_exact = media_observed_cost(usage)
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
    cost, cost_exact = media_observed_cost((result.raw_metadata or {}).get("usage_direct"))
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
