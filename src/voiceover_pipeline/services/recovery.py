"""Read-only paid raw/Media resume-eligibility projection for the CLI entry point.

These helpers decide whether a stored paid attempt may be resumed without a new
paid submit: saved raw audio rebuilt locally, or a stored Polza ``/media`` task
id finished with GET calls only. They only read bounded run state and
already-written files — they never import ``cli``, call a provider, reach the
network, or mutate run state. ``cli`` keeps thin wrappers under the former names
so existing call sites, imports, and monkeypatches stay identical, and the paid
marker write before the request plus every provider call still live in ``cli``.
"""

import hashlib
from pathlib import Path
from typing import Any

from ..models import ScriptChunk
from ..run_state import (
    completed_numbers,
    pending_media_recovery,
    pending_raw_recovery,
    script_hash,
)

# Providers whose TTS call is a paid network submit. A failure after the request
# was sent may still have been accepted and billed, so raw recovery and the
# documented paid-submit block share this exact set.
_PAID_SUBMIT_TTS_PROVIDERS = frozenset({"polza-tts", "polza-chat-audio", "openrouter-tts"})


def _sha256_file(path: Path) -> str:
    """Digest a run-local raw file; a tiny local copy so this module never imports ``cli``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_entry_number(entry: dict[str, Any]) -> int | None:
    number = entry.get("number")
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError):
        return None


def polza_media_route_model(model: object) -> bool:
    """Whether a Polza TTS model submits through the async ``/media`` route.

    Only the ElevenLabs ``elevenlabs/`` models POST to ``/media`` and can hold a
    recoverable task id; every other model uses the synchronous ``/audio/speech``
    route, so a marker next to it can never be an async media task and keeps the
    documented paid-submit block.
    """
    return isinstance(model, str) and model.startswith("elevenlabs/")


def known_media_recovery(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the bounded recovery marker of a known paid Polza media submit.

    Only a paid media run whose model uses the ``/media`` route and whose stored
    id is still a bounded opaque token can be finished with GET calls, and a
    marker next to a completed entry for the same chunk is stale evidence rather
    than a task to recover.
    """
    if not isinstance(state, dict) or state.get("provider") != "polza-tts":
        return None
    if not polza_media_route_model(state.get("model")):
        return None
    recovery = pending_media_recovery(state)
    if recovery is None or recovery["number"] in completed_numbers(state):
        return None
    return recovery


def preceding_chunks_ready(
    prior_ids: dict[int, str], chunks_dir: Path, completed: set[int], number: int
) -> bool:
    """Whether every earlier chunk is state-completed with its MP3 still on disk.

    A missing earlier MP3 would be regenerated into a new paid submit, which
    must not collide with a stored paid attempt, so recovery is refused until
    the run is genuinely contiguous up to this chunk. The file name follows the
    chunk id, which is ``chunk_NN`` for narration and ``turn_NNNN`` for dialogue.
    """
    for earlier in range(1, number):
        if earlier not in completed:
            return False
        chunk_id = prior_ids.get(earlier)
        if not isinstance(chunk_id, str) or not (chunks_dir / f"{chunk_id}.mp3").exists():
            return False
    return True


def state_completed_chunk_ids(state: dict[str, Any] | None) -> dict[int, str]:
    """Map completed state-chunk numbers to their stored ids for resume checks."""
    ids: dict[int, str] = {}
    if not isinstance(state, dict):
        return ids
    for entry in state.get("chunks", []):
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            continue
        number = state_entry_number(entry)
        chunk_id = entry.get("id")
        if number is not None and isinstance(chunk_id, str):
            ids[number] = chunk_id
    return ids


def known_raw_recovery(
    state: dict[str, Any] | None, run_root: Path, chunks_dir: Path
) -> dict[str, Any] | None:
    """Return the bounded raw receipt of a saved paid audio attempt, if usable.

    Raw recovery applies to every paid provider: the bytes were saved before any
    conversion, so a resume rebuilds the chunk locally with no POST and no GET.
    A marker next to a completed entry is stale evidence, and a missing or
    digest-mismatched raw file fails closed so the caller keeps the documented
    paid-submit block instead of resubmitting.
    """
    if not isinstance(state, dict) or state.get("provider") not in _PAID_SUBMIT_TTS_PROVIDERS:
        return None
    recovery = pending_raw_recovery(state)
    if recovery is None or recovery["number"] in completed_numbers(state):
        return None
    raw_path = run_root / recovery["raw_path"]
    if not raw_path.is_file() or _sha256_file(raw_path) != recovery["raw_sha256"]:
        return None
    return recovery


def recoverable_raw_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
    run_root: Path,
) -> dict[int, dict[str, Any]]:
    """Map chunk number to the saved raw audio this exact command may rebuild.

    The raw bytes count only while the whole identity still matches — same
    provider, model, voice, and script — the receipt is bound to one of that
    script's chunks, the file exists with the recorded digest, and every earlier
    chunk is finished. Every other marker keeps the documented block.
    """
    if provider not in _PAID_SUBMIT_TTS_PROVIDERS:
        return {}
    if not isinstance(state, dict):
        return {}
    if state.get("provider") != provider or state.get("model") != model:
        return {}
    if state.get("voice") != voice:
        return {}
    if state.get("script_hash") != script_hash(chunks):
        return {}
    recovery = known_raw_recovery(state, run_root, chunks_dir)
    if recovery is None:
        return {}
    completed = completed_numbers(state)
    prior_ids = {chunk.number: chunk.id for chunk in chunks}
    for chunk in chunks:
        if chunk.number in completed:
            continue
        if chunk.id == recovery["id"] and chunk.number == recovery["number"]:
            if not preceding_chunks_ready(prior_ids, chunks_dir, completed, chunk.number):
                return {}
            return {chunk.number: {**recovery, "kind": "raw"}}
        # The stored attempt belongs to a later chunk while this one is still
        # unfinished, so it is not the attempt that was in flight.
        return {}
    return {}


def recoverable_paid_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
    run_root: Path,
) -> dict[int, dict[str, Any]]:
    """Map chunk number to the paid attempt this exact command may recover.

    Saved raw audio takes precedence over a known media task id: it needs no
    provider request at all, so a chunk that somehow holds both is rebuilt
    locally. Every other marker keeps the documented PAID_SUBMIT_UNCONFIRMED
    block.
    """
    raw = recoverable_raw_attempts(
        state,
        provider=provider,
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_root,
    )
    if raw:
        return raw
    return recoverable_media_attempts(
        state,
        provider=provider,
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
    )


def recoverable_media_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Map chunk number to the paid media attempt this exact command may recover.

    Resume may reuse a stored paid media id only while the whole identity still
    matches — same provider, model, voice, and script — and the marker is bound
    to one of that script's chunks. Every other marker keeps the documented
    PAID_SUBMIT_UNCONFIRMED block, and the chunk is finished with GET calls only.
    """
    if provider != "polza-tts":
        return {}
    if not isinstance(state, dict):
        return {}
    if state.get("provider") != provider or state.get("model") != model:
        return {}
    if state.get("voice") != voice:
        return {}
    if state.get("script_hash") != script_hash(chunks):
        return {}
    recovery = known_media_recovery(state)
    if recovery is None:
        return {}
    completed = completed_numbers(state)
    prior_ids = {chunk.number: chunk.id for chunk in chunks}
    for chunk in chunks:
        if chunk.number in completed:
            continue
        if chunk.id == recovery["id"] and chunk.number == recovery["number"]:
            # A missing earlier MP3 must be regenerated first; that would
            # overwrite this paid marker, so block before key/provider work.
            if not preceding_chunks_ready(prior_ids, chunks_dir, completed, chunk.number):
                return {}
            return {chunk.number: {**recovery, "kind": "media"}}
        # The stored attempt belongs to a later chunk while this one is still
        # unfinished, so it is not the attempt that was in flight. Keeping the
        # documented block is safer than overwriting a known paid id.
        return {}
    return {}
