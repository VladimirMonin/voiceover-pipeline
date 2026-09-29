from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution_identity import build_execution_identity
from .models import ChunkArtifact, ScriptChunk

STATE_FILE = "run_state.json"
LOG_FILE = "generation.log"

# Stage of a paid chunk attempt that a later process must not repeat blindly.
PENDING_ATTEMPT_FIELD = "pending_attempt"
ATTEMPT_SUBMITTING = "submitting"
ATTEMPT_OUTCOME_UNKNOWN = "outcome_unknown"
ATTEMPT_FAILED = "failed"
# The paid audio response was persisted to a run-local raw file before any
# conversion, so a later process can rebuild the chunk from those exact bytes
# without another paid submit.
ATTEMPT_RAW_SAVED = "raw_saved"

# Stages a writer of this module can store. Any other value is reported as
# unknown instead of being echoed back into diagnostics.
_PENDING_ATTEMPT_STATUSES = frozenset(
    {ATTEMPT_SUBMITTING, ATTEMPT_OUTCOME_UNKNOWN, ATTEMPT_FAILED, ATTEMPT_RAW_SAVED}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chunk_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def script_hash(chunks: list[ScriptChunk]) -> str:
    payload = [{"id": chunk.id, "text": chunk.text} for chunk in chunks]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def initial_state(
    *,
    provider: str,
    model: str,
    voice: str,
    script_path: Path,
    chunks: list[ScriptChunk],
    script_format: str,
    run_id: str,
    limited_to_chunks: int | None = None,
    voice_identity: str | None = None,
    synthesis_identity: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    state: dict[str, Any] = {
        "artifact_type": "voiceover-run-state",
        "status": "running",
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "voice": voice,
        "script": str(script_path.resolve()),
        "script_format": script_format,
        "script_hash": script_hash(chunks),
        "execution_source": build_execution_identity(),
        "chunk_count": len(chunks),
        "limited_to_chunks": limited_to_chunks,
        "completed_count": 0,
        "started_at": now,
        "updated_at": now,
        "chunks": [],
        "errors": [],
    }
    if voice_identity is not None:
        state["voice_identity"] = voice_identity
    if synthesis_identity is not None:
        state["synthesis_identity"] = synthesis_identity
    return state


def completed_numbers(state: dict[str, Any] | None) -> set[int]:
    if not state:
        return set()
    return {
        int(item["number"])
        for item in state.get("chunks", [])
        if isinstance(item, dict)
        and item.get("status") == "completed"
        and item.get("number") is not None
    }


def begin_chunk_attempt(state: dict[str, Any], *, chunk_id: str, number: int) -> None:
    """Record that a paid submit for this chunk is about to be sent.

    The marker is written before the provider call, so a process that dies
    between the write and the request resumes as unconfirmed instead of
    repeating a possibly billed submit. Only the chunk identity and stage are
    stored: request text, signed URLs, and provider error bodies never enter run
    state.
    """
    state[PENDING_ATTEMPT_FIELD] = {
        "id": chunk_id,
        "number": number,
        "status": ATTEMPT_SUBMITTING,
        "at": utc_now(),
    }
    state["updated_at"] = utc_now()


def record_chunk_attempt_outcome(state: dict[str, Any], *, status: str) -> None:
    """Keep the attempt marker but record how the paid attempt ended.

    ``ATTEMPT_OUTCOME_UNKNOWN`` means the paid outcome was not confirmed, so
    ``ATTEMPT_FAILED`` is the definite case: the provider rejected the request,
    or its response never became a saved chunk result.
    """
    marker = state.get(PENDING_ATTEMPT_FIELD)
    if not isinstance(marker, dict):
        return
    marker["status"] = status
    marker["at"] = utc_now()
    state["updated_at"] = utc_now()


def clear_chunk_attempt(state: dict[str, Any]) -> None:
    """Drop the attempt marker once the chunk result is safely persisted."""
    if state.pop(PENDING_ATTEMPT_FIELD, None) is not None:
        state["updated_at"] = utc_now()


def record_media_task_accepted(
    state: dict[str, Any], *, task_id: str, chunk_id: str, number: int
) -> None:
    """Store the accepted paid media task id inside the pending attempt marker.

    The id is written before the first poll or audio download, so a later process
    can finish the same paid task with GET calls instead of a second POST. The
    call is bound to the chunk the caller is submitting: the marker written by
    ``begin_chunk_attempt`` must still name that exact chunk, or the id is not
    stored. Only the bounded opaque id joins the marker; request text, signed
    URLs, and provider bodies are never stored.
    """
    marker = state.get(PENDING_ATTEMPT_FIELD)
    if not isinstance(marker, dict):
        raise ValueError("media task accepted without a pending attempt marker")
    if marker.get("id") != chunk_id or marker.get("number") != number:
        raise ValueError("media task accepted for a different chunk than the pending attempt")
    if _bounded_media_task_id(task_id) is None:
        raise ValueError("media task id is not a bounded opaque token")
    marker["remote_task_id"] = task_id
    marker["at"] = utc_now()
    state["updated_at"] = utc_now()


def record_media_observed_cost(
    state: dict[str, Any], *, cost: float | None, cost_exact: str | None
) -> None:
    """Keep the exact cost a completed paid request already reported.

    The completed poll payload carries the billed amount and the download that
    follows can fail on its own, so the cost is stored before that download. A
    call without a usable cost leaves an already stored observation in place, so
    a later GET-only recovery that omits usage cannot erase it.
    """
    marker = state.get(PENDING_ATTEMPT_FIELD)
    if not isinstance(marker, dict) or cost is None:
        return
    marker["cost"] = cost
    if cost_exact is not None:
        marker["cost_exact"] = cost_exact
    marker["at"] = utc_now()
    state["updated_at"] = utc_now()


def record_raw_audio_saved(
    state: dict[str, Any],
    *,
    chunk_id: str,
    number: int,
    audio_format: str,
    relative_path: str,
    sha256: str,
    generation_id: str | None,
) -> None:
    """Store the run-local raw-audio receipt of an accepted paid response.

    The audio bytes are already on disk; this records only the bounded format,
    the deterministic generated path, the digest, and a bounded generation id in
    the existing attempt marker, so a later process can rebuild the chunk from
    the exact paid bytes without another submit. Request text, signed URLs, and
    provider bodies are never stored. A generation id already on the marker
    survives a later call that cannot supply one, and only the same chunk the
    ``begin_chunk_attempt`` marker names may be bound.
    """
    marker = state.get(PENDING_ATTEMPT_FIELD)
    if not isinstance(marker, dict):
        raise ValueError("raw audio saved without a pending attempt marker")
    if marker.get("id") != chunk_id or marker.get("number") != number:
        raise ValueError("raw audio saved for a different chunk than the pending attempt")
    bounded_format = _bounded_raw_format(audio_format)
    if bounded_format is None:
        raise ValueError("raw audio format is not a bounded supported format")
    bounded_path = _bounded_raw_path(
        relative_path, _bounded_chunk_id(chunk_id, _bounded_attempt_number(number)), bounded_format
    )
    if bounded_path is None:
        raise ValueError("raw audio path is not the deterministic generated path")
    bounded_sha = _bounded_sha256(sha256)
    if bounded_sha is None:
        raise ValueError("raw audio digest is not a sha256 hex value")
    bounded_generation = _bounded_generation_id(generation_id)
    if bounded_generation is None:
        previous = marker.get("raw")
        if isinstance(previous, dict):
            bounded_generation = _bounded_generation_id(previous.get("generation_id"))
    marker["raw"] = {
        "format": bounded_format,
        "path": bounded_path,
        "sha256": bounded_sha,
        "generation_id": bounded_generation,
    }
    marker["status"] = ATTEMPT_RAW_SAVED
    marker["at"] = utc_now()
    state["updated_at"] = utc_now()


# Largest chunk or turn number a bounded marker may echo back. Chunks and dialogue
# turns are numbered from 1, so a boolean, non-positive, or larger ``number`` was
# not written by this repository and is reported as unknown instead of echoed.
_MAX_ATTEMPT_NUMBER = 1_000_000


# Ids this module may echo back are the exact forms it generates: an ordinary
# chunk ``chunk_01``, a dialogue turn ``turn_0003``, or the single OmniVoice session
# chunk ``chunk_01_omnivoice_session`` (always number 1). ``run_state.json`` is
# user-editable, so an id that is not the generated form for the reported number
# could be a signed URL, request text, or another secret and is dropped instead of
# being reported. Matching the generated shape exactly rejects near-misses such as
# ``chunk_sk_live_secret`` even though they share the ``chunk_`` prefix.
def _bounded_attempt_number(value: Any) -> int | None:
    """Keep only a positive generated chunk number within the bounded maximum."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 1 <= value <= _MAX_ATTEMPT_NUMBER else None


def _bounded_chunk_id(value: Any, number: int | None) -> str | None:
    """Keep only an id that equals the generated form for this bounded number."""
    if number is None or not isinstance(value, str):
        return None
    if value == f"chunk_{number:02d}" or value == f"turn_{number:04d}":
        return value
    if value == "chunk_01_omnivoice_session" and number == 1:
        return value
    return None


def _bounded_attempt_marker(marker: Any) -> dict[str, Any]:
    """Project any stored marker onto the bounded fields a caller may report.

    ``run_state.json`` is user-editable, so a marker can be a string, a list, or
    a dict with unknown values. Only the chunk identity and a known stage survive
    the projection; request text, signed URLs, and provider error bodies are
    dropped instead of being echoed into diagnostics. The ``number`` survives only
    when it is a positive generated number within the bounded maximum, and the
    ``id`` survives only when it equals the generated form for that number, so a
    URL, request body, or ``chunk_sk_live_secret`` stored there is reported as
    ``null``.
    """
    if not isinstance(marker, dict):
        return {"id": None, "number": None, "status": "unknown"}
    number = _bounded_attempt_number(marker.get("number"))
    status = marker.get("status")
    bounded_status = status if isinstance(status, str) else None
    return {
        "id": _bounded_chunk_id(marker.get("id"), number),
        "number": number,
        "status": bounded_status if bounded_status in _PENDING_ATTEMPT_STATUSES else "unknown",
    }


def unconfirmed_attempt(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the persisted paid-attempt marker whenever one is present.

    The marker is written before a paid submit and removed by the same atomic
    write that stores the finished chunk, so any marker still in run state means
    the paid outcome was not safely resolved. A marker next to a matching
    ``completed`` entry is therefore stale or conflicting evidence, not proof
    that the submit may be repeated, and it blocks resume and overwrite exactly
    like a malformed marker. The returned marker is a bounded projection that
    exposes only the chunk identity and stage.
    """
    if not isinstance(state, dict) or PENDING_ATTEMPT_FIELD not in state:
        return None
    return _bounded_attempt_marker(state[PENDING_ATTEMPT_FIELD])


# A paid media task id becomes part of the GET path ``/media/<id>`` on recovery,
# so only an opaque token may be stored or reused. ``run_state.json`` is
# user-editable, so a value that could change the target endpoint (slash, query,
# scheme, dot traversal, whitespace) is dropped on read and never reaches a
# request or a report.
_OPAQUE_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _bounded_media_task_id(value: Any) -> str | None:
    """Keep only a bounded opaque paid media task id, else report unknown."""
    if isinstance(value, str) and _OPAQUE_TASK_ID_PATTERN.fullmatch(value) is not None:
        return value
    return None


# A paid response's audio is written to a run-local ``raw/<chunk-id>.<ext>``
# file before conversion. The stored format, path, digest, and generation id are
# projected onto exact bounded forms on read, because ``run_state.json`` is
# user-editable and must not be able to point a later process at an arbitrary
# path or an unbounded provider value.
_RAW_FILE_EXTENSIONS = {"mp3": "mp3", "wav": "wav", "pcm16": "pcm"}
_RAW_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _bounded_raw_format(value: Any) -> str | None:
    """Keep only a supported raw-audio format."""
    return value if isinstance(value, str) and value in _RAW_FILE_EXTENSIONS else None


def _bounded_raw_path(value: Any, chunk_id: str | None, raw_format: str | None) -> str | None:
    """Keep only the deterministic ``raw/<chunk-id>.<ext>`` path for this chunk."""
    if chunk_id is None or raw_format is None or not isinstance(value, str):
        return None
    expected = f"raw/{chunk_id}.{_RAW_FILE_EXTENSIONS[raw_format]}"
    return expected if value == expected else None


def _bounded_sha256(value: Any) -> str | None:
    """Keep only a lowercase sha256 hex digest."""
    if isinstance(value, str) and _RAW_SHA256_PATTERN.fullmatch(value) is not None:
        return value
    return None


def _bounded_generation_id(value: Any) -> str | None:
    """Keep only a bounded opaque generation id, exactly like a media task id."""
    return _bounded_media_task_id(value)


def raw_audio_relative_path(chunk_id: str, audio_format: str) -> str:
    """Return the deterministic run-local path for a paid raw audio file."""
    bounded_format = _bounded_raw_format(audio_format)
    if bounded_format is None:
        raise ValueError(f"Unsupported raw audio format: {audio_format}")
    return f"raw/{chunk_id}.{_RAW_FILE_EXTENSIONS[bounded_format]}"


def _bounded_marker_cost(marker: dict[str, Any]) -> tuple[float | None, str | None]:
    """Project the marker's stored cost onto a finite number and exact string."""
    cost = marker.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None, None
    legacy = float(cost)
    if not math.isfinite(legacy):
        return None, None
    return legacy, _exact_cost_string(marker.get("cost_exact"))


def pending_media_recovery(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the bounded GET-only recovery data of the persisted paid attempt.

    A paid Polza media submit that returned an id can be finished with GET calls
    only, so the id and any already observed cost belong in the marker. Only a
    marker bound to a generated chunk identity *and* holding a bounded opaque
    task id is usable; a marker without one, or one holding a URL or request
    text, reports ``None`` so the caller keeps failing closed. The returned
    mapping adds no unbounded provider value.
    """
    bounded_attempt = unconfirmed_attempt(state)
    if (
        bounded_attempt is None
        or bounded_attempt["id"] is None
        or bounded_attempt["number"] is None
    ):
        return None
    marker = state[PENDING_ATTEMPT_FIELD] if isinstance(state, dict) else None
    if not isinstance(marker, dict):
        return None
    task_id = _bounded_media_task_id(marker.get("remote_task_id"))
    if task_id is None:
        return None
    cost, cost_exact = _bounded_marker_cost(marker)
    return {**bounded_attempt, "remote_task_id": task_id, "cost": cost, "cost_exact": cost_exact}


def pending_raw_recovery(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the bounded raw-audio receipt of the persisted paid attempt.

    A paid response whose audio was saved before any conversion can be rebuilt
    from those exact bytes with no provider request at all. Only a marker bound
    to a generated chunk identity and holding a supported format, the
    deterministic generated path, and a sha256 digest is usable; a marker holding
    a URL, request text, or any other shape reports ``None`` so the caller keeps
    failing closed. The returned mapping adds no unbounded provider value.
    """
    bounded_attempt = unconfirmed_attempt(state)
    if (
        bounded_attempt is None
        or bounded_attempt["id"] is None
        or bounded_attempt["number"] is None
    ):
        return None
    marker = state[PENDING_ATTEMPT_FIELD] if isinstance(state, dict) else None
    if not isinstance(marker, dict):
        return None
    raw = marker.get("raw")
    if not isinstance(raw, dict):
        return None
    bounded_format = _bounded_raw_format(raw.get("format"))
    bounded_path = _bounded_raw_path(raw.get("path"), bounded_attempt["id"], bounded_format)
    bounded_sha = _bounded_sha256(raw.get("sha256"))
    if bounded_format is None or bounded_path is None or bounded_sha is None:
        return None
    cost, cost_exact = _bounded_marker_cost(marker)
    return {
        **bounded_attempt,
        "raw_format": bounded_format,
        "raw_path": bounded_path,
        "raw_sha256": bounded_sha,
        "generation_id": _bounded_generation_id(raw.get("generation_id")),
        "cost": cost,
        "cost_exact": cost_exact,
    }


def _exact_cost_string(value: Any) -> str | None:
    """Keep a canonical exact-cost string; any other state value is unknown.

    ``run_state.json`` is user-editable and may hold a legacy float, a bool, or a
    container where an exact cost string is expected. Only a string is a canonical
    exact value; everything else resumes as unknown so it can never be mistaken
    for an exact billed amount.
    """
    return value if isinstance(value, str) else None


def artifact_from_state(item: dict[str, Any]) -> ChunkArtifact:
    return ChunkArtifact(
        number=int(item["number"]),
        id=item["id"],
        file=item["file"],
        duration_ms=int(item["duration_ms"]),
        duration_sec=round(int(item["duration_ms"]) / 1000, 3),
        start_ms=int(item.get("start_ms", 0)),
        end_ms=int(item.get("end_ms", 0)),
        text_characters=int(item.get("text_characters", 0)),
        transcript=item.get("transcript"),
        client_path=item.get("client_path"),
        generation_id=item.get("generation_id"),
        speaker=item.get("speaker"),
        voice=item.get("voice"),
        voice_fingerprint=item.get("voice_fingerprint"),
        turn_index=item.get("turn_index"),
        speech_duration_ms=item.get("speech_duration_ms"),
        audio_sha256=item.get("audio_sha256"),
        pause_after_ms=int(item.get("pause_after_ms", 0)),
        runtime_receipt=item.get("runtime_receipt"),
        voice_selection=item.get("voice_selection"),
        voice_session=item.get("voice_session"),
        cost_rub=item.get("cost_rub"),
        cost_rub_exact=_exact_cost_string(item.get("cost_rub_exact")),
        cost=item.get("cost"),
        cost_exact=_exact_cost_string(item.get("cost_exact")),
        cost_currency=item.get("cost_currency"),
        usage=item.get("usage"),
        generation_time_ms=item.get("generation_time_ms"),
        generated_at=item.get("generated_at"),
        generation_detail_source=item.get("generation_detail_source"),
    )


def state_chunks_as_artifacts(
    state: dict[str, Any] | None, chunks_dir: Path
) -> list[ChunkArtifact]:
    if not state:
        return []
    artifacts = []
    for item in state.get("chunks", []):
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        file_name = item.get("file")
        if not file_name or not (chunks_dir / file_name).exists():
            continue
        artifacts.append(artifact_from_state(item))
    return sorted(artifacts, key=lambda artifact: artifact.number)


def upsert_completed_chunk(
    state: dict[str, Any],
    *,
    artifact: ChunkArtifact,
    model: str,
    voice: str,
    text: str,
    include_text: bool = True,
    include_transcript: bool = True,
) -> None:
    item = {
        "status": "completed",
        "number": artifact.number,
        "id": artifact.id,
        "file": artifact.file,
        "path": f"chunks/{artifact.file}",
        "duration_ms": artifact.duration_ms,
        "duration_sec": artifact.duration_sec,
        "start_ms": artifact.start_ms,
        "end_ms": artifact.end_ms,
        "generation_id": artifact.generation_id,
        "speaker": artifact.speaker,
        "voice": artifact.voice or voice,
        "voice_fingerprint": artifact.voice_fingerprint,
        "turn_index": artifact.turn_index,
        "speech_duration_ms": artifact.speech_duration_ms,
        "audio_sha256": artifact.audio_sha256,
        "pause_after_ms": artifact.pause_after_ms,
        "model": model,
        "text_hash": chunk_text_hash(text),
        "text_characters": artifact.text_characters,
        "transcript": artifact.transcript,
        "client_path": artifact.client_path,
        "runtime_receipt": artifact.runtime_receipt,
        "voice_selection": artifact.voice_selection,
        "voice_session": artifact.voice_session,
        "cost": artifact.cost,
        "cost_exact": artifact.cost_exact,
        "cost_currency": artifact.cost_currency,
        "cost_rub": artifact.cost_rub,
        "cost_rub_exact": artifact.cost_rub_exact,
        "usage": artifact.usage,
        "generated_at": utc_now(),
    }
    if include_text:
        item["text"] = text
    if not include_transcript:
        item.pop("transcript", None)
    state["chunks"] = [
        entry for entry in state.get("chunks", []) if entry.get("number") != artifact.number
    ]
    state["chunks"].append({key: value for key, value in item.items() if value is not None})
    state["chunks"].sort(key=lambda entry: int(entry["number"]))
    state["completed_count"] = len(
        [entry for entry in state["chunks"] if entry.get("status") == "completed"]
    )
    state["updated_at"] = utc_now()


def append_error(state: dict[str, Any], *, chunk_id: str | None, message: str) -> None:
    state.setdefault("errors", []).append(
        {"at": utc_now(), "chunk_id": chunk_id, "message": message}
    )
    state["status"] = "failed"
    state["updated_at"] = utc_now()


class GenerationLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, event: str, **fields: Any) -> None:
        parts = [utc_now(), level.upper(), event]
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value).replace("\n", " ")
            if any(ch.isspace() for ch in text) or text == "":
                text = json.dumps(text, ensure_ascii=False)
            parts.append(f"{key}={text}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(parts) + "\n")
