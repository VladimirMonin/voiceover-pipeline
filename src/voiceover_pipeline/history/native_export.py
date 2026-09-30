"""DB-derived compatibility JSON exports for one verified native TTS run.

Plan sections 5 and 6 make SQLite the single source of truth and demote the old
JSON files to compatibility projections. This module owns only that projection:
it reads one already-verified
:class:`~voiceover_pipeline.history.native_view.NativeTtsView` and rebuilds the
familiar ``run_state.json``, ``chunks.json``, run manifest, and bundle manifest
from the committed rows, so a consumer that still reads the JSON files sees the
same run the database records.

Contract:

* The view is the only input. Every value comes from a committed, fingerprint
  verified row: the run identity from the snapshot config, each chunk's text and
  cast from its ``NativeTtsPart``, each converted artifact's digest and bounded
  processing metadata, and each attempt's observed cost and remote id. Nothing is
  re-globbed from the filesystem, no provider or media function is called, and no
  paid action is authorized. A completed-run export therefore needs neither an
  API key nor a constructed provider.
* Assembly is ordered by committed part position, never by a directory glob, so
  two files that happen to share a directory cannot reorder the final run.
* Every projection carries the run UUID and the revision the view was read at, so
  a half-replaced mixed-revision set of JSON files is detectable. The exported
  ``run_state.json`` also carries a ``native_history`` marker and no
  ``pending_attempt``: the copy is evidence of committed history and never
  authorizes the legacy writer to resume or repeat paid work.
* A missing or malformed committed row that the projection needs raises
  :class:`NativeExportError` before any file is written, so a broken view can
  never publish a half-true manifest. The caller decides whether the failure is
  retryable and keeps the database and audio intact.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..artifacts import (
    build_chunks_manifest,
    build_manifest_json,
    build_run_manifest,
)
from ..models import ChunkArtifact, RunPaths, ScriptChunk
from ..run_state import STATE_FILE, initial_state, upsert_completed_chunk
from .native_view import NativeTtsPart, NativeTtsView, _config_required_str
from .repository import (
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_FINAL_AUDIO,
    ARTIFACT_ROLE_TTS_TURN_QUALITY,
    ATTEMPT_CALL_TYPE_TTS_CHUNK,
    AttemptRecord,
    Cost,
)

# Key added to every exported projection so a reader can correlate a JSON file
# with the exact committed history revision it was projected from.
HISTORY_RUN_UUID_KEY = "history_run_uuid"
HISTORY_REVISION_KEY = "history_revision"
# Marker written into the compatibility ``run_state.json``. Its presence means the
# file is a projection of canonical history, not a legacy state the legacy writer
# may resume or overwrite.
NATIVE_HISTORY_MARKER_KEY = "native_history"


class NativeExportError(RuntimeError):
    """A native run's committed history cannot be projected to compatible JSON."""


@dataclass(frozen=True)
class NativeExport:
    """The four compatibility documents projected from one committed view."""

    run_state: dict[str, Any]
    chunks_manifest: dict[str, Any]
    run_manifest: dict[str, Any]
    bundle_manifest: dict[str, Any]


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeExportError(f"committed {field_name} is missing or not an integer")
    return value


def _final_artifact(view: NativeTtsView):
    finals = [artifact for artifact in view.artifacts if artifact.role == ARTIFACT_ROLE_FINAL_AUDIO]
    if len(finals) != 1:
        raise NativeExportError(
            "a completed native run must carry exactly one final audio artifact"
        )
    return finals[0]


def _chunk_artifact_for_part(view: NativeTtsView, part_uuid: str):
    matches = [
        artifact
        for artifact in view.artifacts
        if artifact.role == ARTIFACT_ROLE_CHUNK_AUDIO and artifact.part_uuid == part_uuid
    ]
    if len(matches) != 1:
        raise NativeExportError(
            "each committed native part must carry exactly one converted chunk artifact"
        )
    return matches[0]


def _attempt_for_part(view: NativeTtsView, part_uuid: str) -> AttemptRecord | None:
    # Only a paid ``tts_chunk`` attempt is projected. A local ``omnivoice-local``
    # part may carry several ``local_tts_chunk`` attempts after local retries, and
    # those cost-free rows are not the paid attempt the export describes.
    matches = [
        attempt
        for attempt in view.attempts
        if attempt.part_uuid == part_uuid and attempt.call_type == ATTEMPT_CALL_TYPE_TTS_CHUNK
    ]
    if len(matches) > 1:
        raise NativeExportError("a committed native part must not carry more than one attempt")
    return matches[0] if matches else None


def _metadata_int(artifact, key: str) -> int:
    metadata = artifact.media_metadata
    if not isinstance(metadata, dict):
        raise NativeExportError("a converted chunk artifact is missing its processing metadata")
    return _require_int(metadata.get(key), key)


def _cost_fields(cost: Cost) -> dict[str, Any]:
    """Project an attempt's observed cost to the legacy per-chunk cost fields.

    Only a recognized RUB observation becomes a chunk cost; an unknown amount
    stays absent so the export never invents a billed value.
    """
    if cost.amount is None or cost.currency != "RUB":
        return {}
    try:
        amount = float(Decimal(cost.amount))
    except (ArithmeticError, ValueError):
        return {}
    fields: dict[str, Any] = {
        "cost": amount,
        "cost_rub": amount,
        "cost_currency": cost.currency,
    }
    if cost.exact_available:
        fields["cost_exact"] = cost.amount
        fields["cost_rub_exact"] = cost.amount
    return fields


def _chunk_artifact(
    view: NativeTtsView,
    part: NativeTtsPart,
    attempt: AttemptRecord | None,
    artifact,
    *,
    start_ms: int,
    dialogue: bool = False,
) -> tuple[ChunkArtifact, int]:
    """Build one ordered ``ChunkArtifact`` plus the exclusive end offset it reaches."""
    duration_ms = _metadata_int(artifact, "duration_ms")
    end_ms = start_ms + duration_ms
    generation_id = None
    if isinstance(artifact.media_metadata, dict):
        candidate = artifact.media_metadata.get("generation_id")
        if isinstance(candidate, str) and candidate:
            generation_id = candidate
    if generation_id is None and attempt is not None:
        generation_id = attempt.remote_id
    chunk = ChunkArtifact(
        number=part.number,
        id=part.chunk_id,
        file=Path(artifact.path).name,
        duration_ms=duration_ms,
        duration_sec=round(duration_ms / 1000, 3),
        start_ms=start_ms,
        end_ms=end_ms,
        text_characters=len(part.text),
        transcript=None if dialogue else part.text,
        client_path="requests",
        generation_id=generation_id,
        speaker=part.speaker,
        voice=part.effective_voice,
        voice_fingerprint=part.voice_fingerprint,
        turn_index=part.number if dialogue else None,
        speech_duration_ms=duration_ms if dialogue else None,
        audio_sha256=artifact.sha256 if dialogue else None,
        pause_after_ms=part.pause_after_ms,
        **(_cost_fields(attempt.cost) if attempt is not None else {}),
    )
    return chunk, end_ms


def _summarize_costs(
    chunk_artifacts: list[ChunkArtifact],
) -> tuple[float | None, str | None, str | None, str | None]:
    """Sum the per-chunk costs into the manifest's total, currency, and provenance.

    All-exact observations produce an exact total; any weaker provenance keeps
    the total as an ordinary float so a legacy lexeme is never promoted to an
    exact amount.
    """
    if not chunk_artifacts:
        return None, None, None, None
    if any(artifact.cost_rub is None for artifact in chunk_artifacts):
        return None, None, None, None
    if all(artifact.cost_rub_exact is not None for artifact in chunk_artifacts):
        exact_total = sum(
            (
                Decimal(artifact.cost_rub_exact)
                for artifact in chunk_artifacts
                if artifact.cost_rub_exact
            ),
            Decimal(0),
        )
        return (
            float(exact_total),
            str(exact_total),
            "RUB",
            "exact",
        )
    float_total = sum(
        artifact.cost_rub for artifact in chunk_artifacts if artifact.cost_rub is not None
    )
    return float_total, None, "RUB", "provider_observed_float"


def _ordered_chunk_artifacts(
    view: NativeTtsView, *, dialogue: bool = False
) -> tuple[list[ChunkArtifact], dict[str, ChunkArtifact]]:
    ordered: list[ChunkArtifact] = []
    by_part: dict[str, ChunkArtifact] = {}
    start_ms = 0
    for part in view.parts:
        artifact = _chunk_artifact_for_part(view, part.record.part_uuid)
        attempt = _attempt_for_part(view, part.record.part_uuid)
        chunk_artifact, start_ms = _chunk_artifact(
            view, part, attempt, artifact, start_ms=start_ms, dialogue=dialogue
        )
        ordered.append(chunk_artifact)
        by_part[part.record.part_uuid] = chunk_artifact
    return ordered, by_part


def _speaker_voice_map(view: NativeTtsView) -> dict[str, str] | None:
    """Rebuild the dialogue cast map from committed parts, in first-seen order.

    Only a run whose parts carry a speaker and a cast voice produces a map; an
    ordinary non-dialogue run returns ``None`` so its manifest stays unchanged.
    """
    cast: dict[str, str] = {}
    for part in view.parts:
        if part.speaker and part.cast_voice:
            cast.setdefault(part.speaker, part.cast_voice)
    return cast or None


def _dialogue_quality_receipt(view: NativeTtsView) -> dict[str, Any] | None:
    """Rebuild the dialogue quality receipt from the committed per-turn verdicts.

    Each turn's content-free receipt is read from its ``tts_turn_quality_receipt``
    artifact in part order, so the aggregate ``tts_quality`` projection carries the
    same fields the legacy ``tts_quality.json`` did and never the private
    transcript. A run that recorded no per-turn evidence returns ``None``.
    """
    receipts = {
        artifact.part_uuid: artifact
        for artifact in view.artifacts
        if artifact.role == ARTIFACT_ROLE_TTS_TURN_QUALITY and artifact.part_uuid is not None
    }
    if not receipts:
        return None
    turns: list[dict[str, Any]] = []
    passed_all = True
    provider: str | None = None
    model: str | None = None
    for part in view.parts:
        artifact = receipts.get(part.record.part_uuid)
        metadata = artifact.media_metadata if artifact is not None else None
        if not isinstance(metadata, dict):
            # A part without its committed verdict cannot be projected; the run is
            # not a complete dialogue run, so no aggregate receipt is claimed.
            return None
        turn = dict(metadata)
        turn.setdefault("turn_index", part.number)
        turns.append(turn)
        if metadata.get("quality_passed") is not True:
            passed_all = False
        asr = metadata.get("asr")
        if isinstance(asr, dict):
            if provider is None and isinstance(asr.get("provider"), str):
                provider = asr["provider"]
            if model is None and isinstance(asr.get("model"), str):
                model = asr["model"]
    return {
        "artifact_type": "voiceover-dialogue-tts-quality-receipt",
        "status": "success" if passed_all else "quality_failed",
        "passed": passed_all,
        "provider": provider,
        "model": model,
        "turn_count": len(view.parts),
        "turns": turns,
        "human_listening_required": True,
    }


def _script_chunks(view: NativeTtsView) -> list[ScriptChunk]:
    """Rebuild the exact script chunks the run hashed, from its committed parts."""
    return [
        ScriptChunk(
            number=part.number,
            id=part.chunk_id,
            text=part.text,
            speaker=part.speaker,
            voice=part.cast_voice,
            voice_fingerprint=part.voice_fingerprint,
            pause_after_ms=part.pause_after_ms,
        )
        for part in view.parts
    ]


def build_native_export(
    view: NativeTtsView,
    paths: RunPaths,
    *,
    script_path: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
) -> NativeExport:
    """Project one verified committed view onto the four compatibility documents.

    ``script_path`` is the still-readable source script when one is available;
    it is used only as the compatibility ``script`` field, never re-read. The
    ordered chunk artifacts, total duration, and cost provenance are derived from
    the committed parts and attempts. A missing final artifact or a part without
    its converted artifact raises :class:`NativeExportError` before any file is
    written.
    """
    config = view.run.config_snapshot
    if not isinstance(config, dict):
        raise NativeExportError("native run is missing its committed snapshot config")
    final = _final_artifact(view)
    if not isinstance(final.media_metadata, dict):
        raise NativeExportError("the final audio artifact is missing its processing metadata")
    main_duration_ms = _require_int(final.media_metadata.get("duration_ms"), "duration_ms")
    execution_source = final.media_metadata.get("execution_source")
    if execution_source is not None and not isinstance(execution_source, dict):
        raise NativeExportError("the final audio artifact carries a malformed execution source")

    ordered, by_part = _ordered_chunk_artifacts(view, dialogue=view.script_format == "dialogue")
    cost_total, cost_total_exact, cost_currency, cost_source = _summarize_costs(ordered)
    dialogue = view.script_format == "dialogue"

    chunks_manifest = build_chunks_manifest(
        provider=_config_required_str(config, "provider"),
        model=_config_required_str(config, "model"),
        voice=_config_required_str(config, "voice"),
        style_prompt=view.style_prompt,
        script=script_path,
        chunks_dir=paths.chunks_dir,
        pricing_snapshot=None,
        cost_exact_available=cost_total_exact is not None,
        cost_total=cost_total,
        cost_total_exact=cost_total_exact,
        cost_currency=cost_currency,
        cost_source=cost_source,
        chunk_artifacts=ordered,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        prompt_mode=_config_required_str(config, "prompt_mode"),
        script_format=view.script_format,
        speaker_voice_map=_speaker_voice_map(view) if dialogue else None,
        tts_quality_receipt=_dialogue_quality_receipt(view) if dialogue else None,
        execution_source=execution_source,
    )
    run_manifest = build_run_manifest(chunks_manifest, paths, main_duration_ms)
    bundle_manifest = build_manifest_json(paths, main_duration_ms)

    run_state = _build_run_state(
        view,
        paths,
        config=config,
        script_path=script_path,
        by_part=by_part,
        main_duration_ms=main_duration_ms,
    )

    revision = view.run.revision
    for document in (chunks_manifest, run_manifest, bundle_manifest):
        document[HISTORY_RUN_UUID_KEY] = view.run.run_uuid
        document[HISTORY_REVISION_KEY] = revision
    return NativeExport(
        run_state=run_state,
        chunks_manifest=chunks_manifest,
        run_manifest=run_manifest,
        bundle_manifest=bundle_manifest,
    )


def _build_run_state(
    view: NativeTtsView,
    paths: RunPaths,
    *,
    config: dict[str, Any],
    script_path: Path,
    by_part: dict[str, ChunkArtifact],
    main_duration_ms: int,
) -> dict[str, Any]:
    """Rebuild the legacy ``run_state.json`` shape from committed history.

    The state is completed and carries no ``pending_attempt``; the
    ``native_history`` marker plus the run UUID and revision identify it as a
    projection of canonical history rather than a legacy state file.
    """
    state = initial_state(
        provider=_config_required_str(config, "provider"),
        model=_config_required_str(config, "model"),
        voice=_config_required_str(config, "voice"),
        script_path=script_path,
        chunks=_script_chunks(view),
        script_format=view.script_format,
        run_id=paths.prefix,
    )
    dialogue = view.script_format == "dialogue"
    for part in view.parts:
        artifact = by_part.get(part.record.part_uuid)
        if artifact is None:  # pragma: no cover - guarded by the ordered projection
            raise NativeExportError("a committed part is missing its converted chunk artifact")
        upsert_completed_chunk(
            state,
            artifact=artifact,
            model=_config_required_str(config, "model"),
            voice=part.effective_voice,
            text=part.text,
            include_text=not dialogue,
            include_transcript=not dialogue,
        )
    state["status"] = "completed"
    state["full_mp3"] = str(paths.full_mp3)
    state["main_duration_ms"] = main_duration_ms
    state[NATIVE_HISTORY_MARKER_KEY] = {
        HISTORY_RUN_UUID_KEY: view.run.run_uuid,
        HISTORY_REVISION_KEY: view.run.revision,
    }
    state[HISTORY_RUN_UUID_KEY] = view.run.run_uuid
    state[HISTORY_REVISION_KEY] = view.run.revision
    return state


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one projection atomically without a predictable temporary name.

    The temporary file is created uniquely and exclusively beside the target with
    ``mkstemp``, flushed and fsynced, then renamed over the target. A fixed
    ``*.json.tmp`` name is never used, so a pre-existing symlink at that name
    cannot redirect the write to an external file. Only this call's own temporary
    file is removed on failure, and a failure raises the underlying ``OSError`` so
    the caller keeps the database and audio untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def write_native_export(export: NativeExport, paths: RunPaths) -> None:
    """Atomically replace every compatibility document from one committed view.

    Each file is written through a temp file and renamed, so a failure while
    writing leaves the previous copy intact. A failure raises the underlying
    ``OSError``: the caller keeps the database and audio untouched and reports a
    stable output error.
    """
    _atomic_write_json(paths.output_root / STATE_FILE, export.run_state)
    _atomic_write_json(paths.chunks_json, export.chunks_manifest)
    _atomic_write_json(paths.run_json, export.run_manifest)
    _atomic_write_json(paths.output_root / "manifest.json", export.bundle_manifest)
