"""Canonical-history persistence for the local ASR, timings, and verify commands.

Plan sections 6 and 7 require the existing ``transcribe``, ``timings``, and
``verify-tts`` commands to record what they observed in the canonical SQLite
history. This module is that writer for the already-implemented *local* routes:
the registered ASR providers (``qwen-local``, ``nemotron-local``) and the local
``faster-whisper`` timing provider. Cloud ASR and cloud timing routes are not
wired here: their paid-submit contract is not confirmed, so persisting them now
would claim an outcome the project cannot yet prove.

Contract:

* One call writes one run, its one attempt, its artifacts, and its private text
  sources inside a single short transaction. There is no second state engine, no
  resume marker, and no lock: a local ASR request has no paid POST to protect.
* The run root is a private ``0700`` directory under the managed
  ``VOICEOVER_HOME/runs/<uuid>`` layout, so an ASR run never shares a directory
  with a legacy/JSON TTS run or a native TTS run. Managed audio is never copied:
  a large source audio file is referenced by its external absolute path with its
  size and SHA-256, and a file that no longer exists is recorded as
  ``availability="missing"`` with no invented digest.
* Every stored value is caller-supplied and observed. No span is synthesized, a
  text-only result carries no timestamps, and a blank transcript is stored as an
  empty string marked ``text_completeness="incomplete"`` so a reader never treats
  an empty result as complete text.
* The attempt keeps :meth:`Cost.unknown` because these are local, keyless routes:
  no external API charge exists, and the aggregate ``history costs`` read counts
  such an attempt as ``local_attempts_without_api_charge``.
* Validation and database failures raise :class:`AsrHistoryError` with a fixed,
  content-free message. The CLI keeps the provider response it already produced
  and reports a fixed machine-visible persistence failure instead of claiming a
  save that did not happen.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import HistoryDatabase, HistoryDatabaseError
from .native_snapshot import NATIVE_SNAPSHOT_ORIGIN
from .paths import (
    HistoryPathsError,
    ensure_history_home,
    ensure_private_directory,
    history_database_path,
    history_runs_dir,
)
from .repository import (
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    MAX_QUERY_LIMIT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    RUN_STATUS_COMPLETED,
    TEXT_COMPLETENESS_COMPLETE,
    Cost,
    HistoryRepository,
)

# Operation names one per command so ``history list --operation`` filters them.
OPERATION_ASR = "asr"
OPERATION_TIMINGS = "timings"
OPERATION_VERIFY = "verify"

# One attempt call type per operation. They mirror the legacy importer's free-form
# call types and stay distinct from the paid TTS ``tts_chunk`` marker.
ATTEMPT_CALL_TYPE_ASR = "asr_transcription"
ATTEMPT_CALL_TYPE_TIMING = "asr_timing"
ATTEMPT_CALL_TYPE_VERIFY = "tts_quality_asr"
ATTEMPT_STATUS_COMPLETED = "completed"

# Artifact roles this writer records: the referenced source audio, the two
# legacy timings files, and the optional content-free TTS quality receipt.
ARTIFACT_ROLE_SOURCE_AUDIO = "asr_source_audio"
ARTIFACT_ROLE_TIMINGS_JSON = "timings_json"
ARTIFACT_ROLE_SRT = "srt"
ARTIFACT_ROLE_QUALITY_RECEIPT = "tts_quality_receipt"

# Provenance written into every run snapshot and text source, so a native ASR run
# is distinguishable from a ``legacy_import`` run.
NATIVE_ASR_ORIGIN = "native_asr"

# The fixed machine-visible reason a caller reports when the write failed. The
# provider response is kept; only the persistence step is marked failed.
HISTORY_PERSISTENCE_FAILED = "HISTORY_PERSISTENCE_FAILED"

_HISTORY_UNAVAILABLE = "the local history database could not be opened for this run"
_HISTORY_WRITE_FAILED = "the local history database could not store this run"


class AsrHistoryError(RuntimeError):
    """The canonical-history write for one ASR/timings/verify run failed."""

    def __init__(self, message: str, *, error_code: str = HISTORY_PERSISTENCE_FAILED) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class AsrHistoryText:
    """One private text source: its role, verbatim content, and completeness."""

    kind: str
    content: str
    language: str | None = None
    text_completeness: str = TEXT_COMPLETENESS_COMPLETE


@dataclass(frozen=True)
class AsrHistoryArtifact:
    """One external file reference recorded only when the file truly exists."""

    role: str
    path: Path
    mime: str | None = None
    media_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AsrHistorySave:
    """Everything one completed local ASR/timings/verify run contributes.

    ``config_snapshot`` is the run's bounded non-secret provenance map: provider,
    model, options, observed segments, and the timing basis. It holds no provider
    response text beyond the transcript role stored separately, and the history
    boundary redacts secret-looking values from its own copy.
    """

    operation: str
    attempt_call_type: str
    provider: str | None
    model: str | None
    config_snapshot: dict[str, Any]
    text_sources: Sequence[AsrHistoryText] = ()
    artifacts: Sequence[AsrHistoryArtifact] = ()
    source_audio: Path | None = None
    source_audio_mime: str | None = None
    parent_run_root: Path | None = None
    attempt_usage: dict[str, Any] | None = None
    user_label: str | None = None


@dataclass(frozen=True)
class AsrHistorySaveResult:
    """The CLI-visible outcome of one persistence attempt.

    ``active=False`` means history persistence does not apply to this command or
    route -- ``settings.toml`` disabled history, or a cloud timing route that is
    still legacy -- so the command behaves exactly as before and creates nothing.
    ``saved=False`` with ``active=True`` is a persistence failure the caller must
    surface instead of claiming a save.
    """

    active: bool
    saved: bool
    run_uuid: str | None = None
    error_code: str | None = None

    def metadata(self) -> dict[str, Any] | None:
        """Return the bounded ``history`` block, or ``None`` when not persisted."""
        if not self.active:
            return None
        if self.saved:
            return {"saved": True, "run_uuid": self.run_uuid}
        return {"saved": False, "error_code": self.error_code}


def inactive_history_result() -> AsrHistorySaveResult:
    """Return the result for a run whose history persistence does not apply."""
    return AsrHistorySaveResult(active=False, saved=False)


def failed_history_result() -> AsrHistorySaveResult:
    """Return the result for a persistence failure that must be reported."""
    return AsrHistorySaveResult(active=True, saved=False, error_code=HISTORY_PERSISTENCE_FAILED)


def saved_history_result(run_uuid: str) -> AsrHistorySaveResult:
    """Return the result for a committed run."""
    return AsrHistorySaveResult(active=True, saved=True, run_uuid=run_uuid)


def sha256_text(text: str) -> str:
    """Return the lowercase hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path) -> tuple[str, int | None, str | None, str]:
    """Return ``(availability, size, sha256, resolved path)`` for a real file.

    A missing or unreadable file keeps its resolved path but reports
    ``availability="missing"`` with no size and no digest: an absent file never
    gains a fabricated hash.
    """
    resolved = str(Path(path).expanduser().resolve())
    try:
        if not Path(path).is_file():
            return AVAILABILITY_MISSING, None, None, resolved
        size = Path(path).stat().st_size
        digest = _sha256_file(Path(path))
    except OSError:
        return AVAILABILITY_MISSING, None, None, resolved
    return AVAILABILITY_PRESENT, size, digest, resolved


def _run_root_text(run_uuid: str) -> str:
    return str((history_runs_dir() / run_uuid).resolve())


def _resolve_parent_run(repository: HistoryRepository, parent_run_root: Path | None) -> str | None:
    """Return the one native TTS run owning ``parent_run_root``, else ``None``.

    ``verify-tts`` may verify audio that already lives in a native TTS run root.
    Only an unambiguous native snapshot in that exact directory is linked; an
    imported run, a shared root, or an ambiguous match stays unlinked rather than
    guessing a parent.
    """
    if parent_run_root is None:
        return None
    root = str(Path(parent_run_root).expanduser().resolve())
    candidates = [
        run
        for run in repository.find_runs_by_root(root, limit=MAX_QUERY_LIMIT)
        if isinstance(run.config_snapshot, dict)
        and run.config_snapshot.get("operation_origin") == NATIVE_SNAPSHOT_ORIGIN
    ]
    if len(candidates) != 1:
        return None
    return candidates[0].run_uuid


def _write_run(repository: HistoryRepository, save: AsrHistorySave, run_uuid: str) -> None:
    """Write run, attempt, artifacts, and text sources inside one transaction."""
    with repository.transaction():
        parent_uuid = _resolve_parent_run(repository, save.parent_run_root)
        run = repository.create_run(
            operation=save.operation,
            run_root=_run_root_text(run_uuid),
            status=RUN_STATUS_COMPLETED,
            user_label=save.user_label,
            parent_uuid=parent_uuid,
            config_snapshot=save.config_snapshot,
            run_uuid=run_uuid,
        )
        attempt = repository.add_attempt(
            run.run_uuid,
            call_type=save.attempt_call_type,
            provider=save.provider,
            model=save.model,
            status=ATTEMPT_STATUS_COMPLETED,
            usage=save.attempt_usage,
            cost=Cost.unknown(),
        )
        if save.source_audio is not None:
            availability, size, digest, resolved = _file_evidence(save.source_audio)
            repository.add_artifact(
                run.run_uuid,
                role=ARTIFACT_ROLE_SOURCE_AUDIO,
                path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
                path=resolved,
                attempt_uuid=attempt.attempt_uuid,
                mime=save.source_audio_mime,
                size_bytes=size,
                sha256=digest,
                availability=availability,
            )
        for artifact in save.artifacts:
            availability, size, digest, resolved = _file_evidence(artifact.path)
            if availability != AVAILABILITY_PRESENT:
                # Only a real file is referenced: a timing/SRT/receipt artifact is
                # recorded when it truly exists, never as a placeholder.
                continue
            repository.add_artifact(
                run.run_uuid,
                role=artifact.role,
                path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
                path=resolved,
                attempt_uuid=attempt.attempt_uuid,
                mime=artifact.mime,
                size_bytes=size,
                sha256=digest,
                media_metadata=artifact.media_metadata,
            )
        for text in save.text_sources:
            repository.add_text_source(
                run.run_uuid,
                kind=text.kind,
                origin=NATIVE_ASR_ORIGIN,
                content=text.content,
                content_hash=sha256_text(text.content),
                language=text.language,
                text_completeness=text.text_completeness,
            )


def persist_asr_history(save: AsrHistorySave) -> str:
    """Persist one completed local ASR/timings/verify run and return its UUID.

    The managed ``runs/<uuid>`` directory is created private before any insert,
    and run, attempt, artifacts, and text sources commit in one transaction. A
    failure before or during the write raises :class:`AsrHistoryError` with a
    fixed message; no partial run is left behind, and the caller keeps the
    provider response it already produced.
    """
    run_uuid = str(uuid.uuid4())
    try:
        ensure_history_home()
        ensure_private_directory(history_runs_dir() / run_uuid)
        database = HistoryDatabase(history_database_path())
        database.connect()
        database.migrate()
    except (HistoryDatabaseError, HistoryPathsError, sqlite3.Error, OSError) as exc:
        raise AsrHistoryError(_HISTORY_UNAVAILABLE) from exc
    try:
        with database:
            _write_run(HistoryRepository(database), save, run_uuid)
    except (HistoryDatabaseError, HistoryPathsError, sqlite3.Error, OSError, ValueError) as exc:
        raise AsrHistoryError(_HISTORY_WRITE_FAILED) from exc
    return run_uuid
