"""Native DB-first generation and resume for the ordinary Polza Media TTS route.

Plan sections 5 and 6 make SQLite the single source of truth for run history:
one native run's prepared snapshot, paid attempts, raw evidence, converted
chunks, and final assembly live in the database, and the legacy JSON files become
compatibility exports. This module owns the bounded executor for exactly one
admitted route -- an ordinary, non-dialogue ``polza-tts`` run whose model uses the
existing ``elevenlabs/`` ``/media`` route, without integrated timing or quality
processing -- and its recovery decisions.

Contract:

* Ownership is resolved from two independent pieces of evidence: the committed
  run row found by canonical ``run_root`` and a tiny run-local ownership
  descriptor. A committed native row always wins, so the crash window between the
  snapshot commit and the descriptor write stays native-owned. A descriptor whose
  database row is missing, an unreadable or foreign database next to native
  evidence, or a malformed descriptor fails closed; there is no silent fallback
  to the legacy JSON writer and no replacement database is created.
* Selection and mutation are serialized by the same cross-process run lock the
  history layer already owns, so two writers can never execute one run root.
* A fresh run commits its prepared snapshot and then its descriptor *before* any
  provider request exists. A resume proves the exact synthesis identity through
  :func:`~voiceover_pipeline.history.native_resume.preflight_native_tts_resume`
  and the recorded output-processing settings before any paid action.
* The paid ordering is inherited unchanged from the reservation seam: reserve a
  never-attempted part, submit once, record the accepted opaque task id, record
  the exact observed cost, persist the accepted bytes and their bounded receipt,
  link them, and only then convert and commit. No prior marker ever triggers a
  second POST; a known accepted task id is finished with GET calls only, and a
  verified raw receipt is rebuilt locally with no provider work at all.
* Recovery decisions read and hash on-disk evidence *outside* every database
  transaction. A completed part is skipped only after its committed digest, size,
  and file presence verify; a mismatch fails closed instead of regenerating and
  re-billing. Final assembly uses the ordered database parts, never a directory
  glob.
* The provider is constructed lazily, only for a fresh unattempted submit or a
  known-id GET recovery. A local raw rebuild or a completed-run export repair
  never reads an API key and never builds a provider.

Known limits: this slice admits one non-dialogue provider/model route and the
fixed trimming/assembly semantics recorded in the snapshot. Dialogue, integrated
transcription/timing, quality gating, and the other speech providers stay on the
legacy executor until their own identity and processing paths are supported.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from ..execution_identity import build_execution_identity
from ..history.database import (
    HistoryDatabase,
    HistoryDatabaseError,
    connect_readonly_consistent,
)
from ..history.locking import HistoryRunLockedError, acquire_run_lock
from ..history.native_export import build_native_export, write_native_export
from ..history.native_resume import NativeResumeError, preflight_native_tts_resume
from ..history.native_snapshot import (
    NATIVE_SNAPSHOT_ORIGIN,
    NativeSnapshotError,
    persist_prepared_tts_snapshot,
)
from ..history.native_view import (
    NativeTtsPart,
    NativeTtsView,
    NativeViewError,
    load_native_tts_view,
)
from ..history.paths import HistoryPathsError, history_database_path
from ..history.raw_receipt import (
    PaidRawReceipt,
    PaidRawReceiptError,
    write_paid_raw_receipt,
)
from ..history.repository import (
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_FINAL_AUDIO,
    ARTIFACT_ROLE_PAID_RAW_AUDIO,
    ATTEMPT_STATUS_COMPLETED,
    ATTEMPT_STATUS_RAW_SAVED,
    ATTEMPT_STATUS_REMOTE_ACCEPTED,
    ATTEMPT_STATUS_SUBMITTING,
    MAX_QUERY_LIMIT,
    ArtifactRecord,
    AttemptRecord,
    HistoryRepository,
    HistoryRepositoryError,
    RunRecord,
)
from ..models import ScriptChunk, SynthesisResult
from ..run_state import LOG_FILE, GenerationLogger
from . import costs
from .prepare import PreparedRun

# Numeric exit codes duplicated from ``cli.py`` because the CLI imports this
# service; only the documented stable codes are used here.
_EXIT_ARGS = 2
_EXIT_PROVIDER = 30
_EXIT_OUTPUT = 50

# Run-local ownership descriptor. It identifies the committed native run and
# carries no progress and no execution permission; the run lock serializes writers
# and the database remains the only progress truth.
OWNERSHIP_FILE_NAME = ".voiceover-native-history.json"
OWNERSHIP_ARTIFACT_TYPE = "voiceover-native-history-ownership"
OWNERSHIP_VERSION = 1
# Version of the fixed trimming/assembly semantics recorded in the snapshot. A
# later change bumps it instead of silently resuming with different processing.
OUTPUT_PROCESSING_VERSION = 1
# Bounded account label for a single paid attempt; the provider identity itself is
# the model, and no user secret ever enters this field.
PAID_ACCOUNT_ALIAS = "default"

# Stable machine-readable error codes carried in the CLI error envelope.
_ERROR_OWNERSHIP_UNVERIFIABLE = "NATIVE_OWNERSHIP_UNVERIFIABLE"
_ERROR_OWNERSHIP_RUN_MISSING = "NATIVE_OWNERSHIP_RUN_MISSING"
_ERROR_PROCESSING_CHANGED = "NATIVE_PROCESSING_UNSUPPORTED"
_ERROR_EVIDENCE_INCONSISTENT = "NATIVE_EVIDENCE_INCONSISTENT"
_ERROR_RAWS_EVIDENCE_INVALID = "NATIVE_RAW_EVIDENCE_INVALID"
_ERROR_SUBMIT_UNCONFIRMED = "PAID_SUBMIT_UNCONFIRMED"
_ERROR_SYNTHESIS_FAILED = "NATIVE_SYNTHESIS_FAILED"
_ERROR_CONVERSION_FAILED = "NATIVE_CONVERSION_FAILED"
_ERROR_ASSEMBLY_FAILED = "NATIVE_ASSEMBLY_FAILED"
_ERROR_EXPORT_FAILED = "NATIVE_EXPORT_FAILED"
_ERROR_OUTPUT_UNAVAILABLE = "NATIVE_OUTPUT_UNAVAILABLE"
_ERROR_HISTORY_UNAVAILABLE = "NATIVE_HISTORY_UNAVAILABLE"
_ERROR_HISTORY_CONFLICT = "NATIVE_HISTORY_CONFLICT"

# Recovery route of one part.
_ROUTE_LEGACY = "legacy"
_ROUTE_NATIVE_EXISTING = "native_existing"
_ROUTE_NATIVE_NEW = "native_new"
_ROUTE_BLOCKED = "blocked"


class NativeGenerationError(RuntimeError):
    """A native generation step failed with a stable exit code and error code."""

    def __init__(self, message: str, *, code: int, error_code: str) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code


@dataclass(frozen=True)
class NativeOwnership:
    """Where a run root belongs before any provider or deletion step.

    ``route`` is one of the bounded routes above; ``run_uuid`` and ``revision``
    identify an existing committed native run, and ``error_code`` plus ``reason``
    describe a fail-closed block. ``reason`` never echoes a stored value.
    """

    route: str
    run_uuid: str | None = None
    revision: int | None = None
    error_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NativeExecutionHooks:
    """The CLI's local media presentation seams and streaming digest helper."""

    write_audio_as_mp3: Callable[[str, bytes, str, Path], None]
    trim_final_silence: Callable[[str, str, Path], None]
    mp3_duration_ms: Callable[[str, Path], int]
    concat_audio_files: Callable[[str, list[Path], Path], None]
    sha256_file: Callable[[Path], str]
    progress: Callable[[str], None]


@dataclass(frozen=True)
class NativeGenerationSummary:
    """What the CLI turns into its success JSON or human output."""

    run_uuid: str
    revision: int
    files: dict[str, str]
    duration_ms: int
    segment_count: int | None
    cost_total: float | None
    cost_currency: str | None


class _ReadOnlyConnection:
    """Minimal handle exposing ``.connection`` to :class:`HistoryRepository`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


# ── ownership descriptor ──────────────────────────────────────────────────────


def _descriptor_path(run_root: Path) -> Path:
    return run_root / OWNERSHIP_FILE_NAME


def _read_descriptor(run_root: Path) -> str | None:
    """Return the committed run UUID the descriptor names, or fail closed.

    A missing descriptor is ``None``. A symlinked descriptor is refused before
    its target is read, including a dangling link, so a user-controlled alias can
    never redirect the ownership probe. A present descriptor that is not a
    bounded ownership document raises :class:`NativeGenerationError`, so a
    malformed or tampered marker never becomes a legacy run.
    """
    path = _descriptor_path(run_root)
    if path.is_symlink():
        raise NativeGenerationError(
            "native run ownership descriptor must not be a symlink; refusing to trust it.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise NativeGenerationError(
            "native run ownership descriptor is unreadable; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type",
        "ownership_version",
        "run_uuid",
    }:
        raise NativeGenerationError(
            "native run ownership descriptor is malformed; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    if (
        payload["artifact_type"] != OWNERSHIP_ARTIFACT_TYPE
        or payload["ownership_version"] != OWNERSHIP_VERSION
    ):
        raise NativeGenerationError(
            "native run ownership descriptor has an unsupported version; refusing legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    run_uuid = payload["run_uuid"]
    if not isinstance(run_uuid, str) or not run_uuid:
        raise NativeGenerationError(
            "native run ownership descriptor names no run; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    return run_uuid


def _write_descriptor(run_root: Path, run_uuid: str) -> None:
    """Atomically write the tiny ownership descriptor for a committed native run.

    The temporary file is created uniquely and exclusively beside the descriptor
    with ``mkstemp``, flushed and fsynced, then renamed over it. A fixed
    ``*.json.tmp`` name is never used, so a pre-existing symlink at that name
    cannot redirect the write to an external file. Only this call's own temporary
    file is removed on failure.
    """
    payload = {
        "artifact_type": OWNERSHIP_ARTIFACT_TYPE,
        "ownership_version": OWNERSHIP_VERSION,
        "run_uuid": run_uuid,
    }
    path = _descriptor_path(run_root)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
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


def _ensure_descriptor(run_root: Path, run_uuid: str) -> None:
    """Restore a missing descriptor for a committed run without rewriting one."""
    existing = _read_descriptor(run_root)
    if existing is None:
        _write_descriptor(run_root, run_uuid)
    elif existing != run_uuid:
        raise NativeGenerationError(
            "native run ownership descriptor names a different run than the committed history.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )


def _native_local_evidence(run_root: Path) -> bool:
    """Whether the run root carries any marker the legacy writer must not own.

    A descriptor, a compatibility ``run_state.json`` that carries the native
    history marker, or a ``raw/`` directory with at least one paid receipt are all
    evidence that this directory belongs to canonical history. When the database
    cannot confirm that, the caller fails closed instead of letting a legacy
    writer mutate the directory.
    """
    if _descriptor_path(run_root).exists():
        return True
    state_path = run_root / "run_state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            payload = None
        if isinstance(payload, dict) and "native_history" in payload:
            return True
    raw_dir = run_root / "raw"
    if raw_dir.is_dir() and any(raw_dir.glob("*.receipt.json")):
        return True
    return False


def _find_native_run(
    canonical_root: str, database_path: Path | None
) -> tuple[RunRecord | None, bool]:
    """Return the committed native run for a root and whether the database was usable.

    The read is read-only and writes nothing: an absent database means no native
    ownership, while an unreadable, foreign, or newer database reports itself
    unusable so the caller can fail closed when native evidence exists. The reader
    is WAL-consistent, so a native row another live writer just committed is not
    missed as a stale main-file snapshot. Only a run carrying the native snapshot
    origin counts; an imported run that happens to share the directory does not.
    """
    try:
        db_path = history_database_path() if database_path is None else Path(database_path)
    except HistoryPathsError:
        return None, False
    if not db_path.exists():
        return None, True
    connection: sqlite3.Connection | None = None
    try:
        connection = connect_readonly_consistent(db_path)
        repository = HistoryRepository(cast(HistoryDatabase, _ReadOnlyConnection(connection)))
        rows = repository.find_runs_by_root(canonical_root, limit=MAX_QUERY_LIMIT)
    except (HistoryDatabaseError, sqlite3.Error, OSError):
        return None, False
    finally:
        if connection is not None:
            connection.close()
    native = [
        run
        for run in rows
        if run.legacy_source_root is None
        and isinstance(run.config_snapshot, dict)
        and run.config_snapshot.get("operation_origin") == NATIVE_SNAPSHOT_ORIGIN
    ]
    if len(native) > 1:
        # Two native runs on one root cannot be told apart; refuse instead of
        # choosing one.
        return None, False
    return (native[0] if native else None), True


def _fresh_run_root_available(run_root: Path) -> bool:
    """Whether a fresh native run may claim this root without another writer's state.

    The early ownership resolution already routes a native-owned root away from a
    fresh start; this is the re-check under the run lock. If a concurrent legacy
    or native writer created local state between the early decision and this
    point, the root is refused instead of letting a second writer interleave one
    directory. Any descriptor, native run state, or paid raw receipt, and any
    legacy ``run_state.json``, counts as another writer's state.
    """
    return not _native_local_evidence(run_root) and not (run_root / "run_state.json").exists()


def resolve_native_ownership(
    output_root: Path, *, database_path: Path | None = None
) -> NativeOwnership:
    """Resolve one run root's route without writing anything.

    A committed native run row, or failing that native local evidence, makes the
    root native-owned. Missing or unreadable database state next to native
    evidence fails closed with a stable error code. Every other root reports
    ``legacy``, which keeps existing legacy runs and unswitched configurations on
    the exact behavior they had.
    """
    canonical_root = str(Path(output_root).resolve())
    descriptor_uuid = _read_descriptor(output_root)
    run, database_usable = _find_native_run(canonical_root, database_path)
    if run is not None:
        return NativeOwnership(
            route=_ROUTE_NATIVE_EXISTING, run_uuid=run.run_uuid, revision=run.revision
        )
    if not database_usable:
        if descriptor_uuid is not None or _native_local_evidence(output_root):
            return NativeOwnership(
                route=_ROUTE_BLOCKED,
                error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
                reason=(
                    "this run directory carries native history evidence but the history "
                    "database could not be read; refusing to treat it as a legacy JSON run. "
                    "Restore the history database or choose a different --run-id."
                ),
            )
        return NativeOwnership(route=_ROUTE_LEGACY)
    if descriptor_uuid is not None or _native_local_evidence(output_root):
        return NativeOwnership(
            route=_ROUTE_BLOCKED,
            error_code=_ERROR_OWNERSHIP_RUN_MISSING,
            reason=(
                "this run directory is owned by native history, but its committed history run "
                "no longer exists; refusing to treat it as a legacy JSON run or to recreate "
                "history silently. Choose a different --run-id."
            ),
        )
    return NativeOwnership(route=_ROUTE_LEGACY)


def build_output_options(no_trim: bool) -> dict[str, Any]:
    """Return the fixed output-processing settings the native route records.

    The native slice always trims final silence; ``--no-trim`` is rejected before
    any provider work, so the recorded settings prove a later resume used the same
    semantics rather than inferring them from the current command line.
    """
    return {"trim_final_silence": not no_trim, "processing_version": OUTPUT_PROCESSING_VERSION}


# ── database helpers ──────────────────────────────────────────────────────────


@contextmanager
def _open_writable_history(database_path: Path | None) -> Iterator[HistoryRepository]:
    """Open the private history database for writing, migrating it if needed."""
    try:
        path = history_database_path() if database_path is None else Path(database_path)
    except HistoryPathsError as exc:
        raise NativeGenerationError(
            "the history home is not a usable absolute directory.",
            code=_EXIT_ARGS,
            error_code="NATIVE_HISTORY_HOME_INVALID",
        ) from exc
    try:
        database = HistoryDatabase(path)
        database.connect()
        database.migrate()
    except (HistoryDatabaseError, sqlite3.Error, OSError) as exc:
        raise NativeGenerationError(
            "the local history database could not be opened for this run.",
            code=_EXIT_OUTPUT,
            error_code=_ERROR_HISTORY_UNAVAILABLE,
        ) from exc
    try:
        yield HistoryRepository(database)
    finally:
        database.close()


# ── part evidence ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _PartEvidence:
    """One part's committed attempt and artifacts, read from a verified view."""

    part: NativeTtsPart
    attempt: AttemptRecord | None
    raw_artifact: ArtifactRecord | None
    chunk_artifact: ArtifactRecord | None


def _collect_evidence(view: NativeTtsView) -> list[_PartEvidence]:
    evidence: list[_PartEvidence] = []
    for part in view.parts:
        part_uuid = part.record.part_uuid
        attempts = [attempt for attempt in view.attempts if attempt.part_uuid == part_uuid]
        raws = [
            artifact
            for artifact in view.artifacts
            if artifact.part_uuid == part_uuid and artifact.role == ARTIFACT_ROLE_PAID_RAW_AUDIO
        ]
        chunks = [
            artifact
            for artifact in view.artifacts
            if artifact.part_uuid == part_uuid and artifact.role == ARTIFACT_ROLE_CHUNK_AUDIO
        ]
        if len(attempts) > 1 or len(raws) > 1 or len(chunks) > 1:
            # A part with more than one attempt or artifact cannot be told apart, so
            # the executor must not silently pick one and continue to a paid call.
            raise NativeGenerationError(
                "refusing to continue: a committed part carries duplicate attempts or "
                "artifacts, so its history cannot be reconstructed.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        evidence.append(
            _PartEvidence(
                part=part,
                attempt=attempts[0] if attempts else None,
                raw_artifact=raws[0] if raws else None,
                chunk_artifact=chunks[0] if chunks else None,
            )
        )
    return evidence


def _has_paid_evidence(evidence: _PartEvidence) -> bool:
    return (
        evidence.attempt is not None
        or evidence.raw_artifact is not None
        or evidence.chunk_artifact is not None
    )


def _verified_chunk_file(
    evidence: _PartEvidence, run_root: Path, sha256_file: Callable[[Path], str]
) -> Path | None:
    """Return the verified converted chunk path, or ``None`` when it is not usable.

    A committed converted artifact is skipped only when its file still exists and
    its size and digest match the committed evidence. Any mismatch returns
    ``None`` so the caller re-derives from retained raw bytes instead of trusting
    a foreign file.
    """
    artifact = evidence.chunk_artifact
    if artifact is None or artifact.sha256 is None or artifact.size_bytes is None:
        return None
    path = run_root / artifact.path
    if not path.is_file():
        return None
    try:
        if path.stat().st_size != artifact.size_bytes:
            return None
        if sha256_file(path) != artifact.sha256:
            return None
    except OSError:
        return None
    return path


def _verified_raw_bytes(
    raw: ArtifactRecord, run_root: Path, sha256_file: Callable[[Path], str]
) -> bytes:
    """Return retained raw paid bytes only when they still match their artifact.

    A crash between writing the raw bytes and committing the database row, or any
    later tampering, is caught here: the file must exist and match the committed
    size and digest, or the caller fails closed rather than converting foreign
    bytes.
    """
    if raw.sha256 is None or raw.size_bytes is None:
        raise NativeGenerationError(
            "the retained paid raw artifact is missing its digest or size.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        )
    path = run_root / raw.path
    if not path.is_file():
        raise NativeGenerationError(
            "the retained paid raw audio is missing; refusing to regenerate paid work.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        )
    try:
        if path.stat().st_size != raw.size_bytes or sha256_file(path) != raw.sha256:
            raise NativeGenerationError(
                "the retained paid raw audio does not match its committed evidence.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        return path.read_bytes()
    except OSError as exc:
        raise NativeGenerationError(
            "the retained paid raw audio could not be read.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        ) from exc


# ── executor ──────────────────────────────────────────────────────────────────


def _new_staging_path(target: Path) -> Path:
    """Return a fresh private staging path beside ``target`` for one output file.

    The file lives in the output's own directory so publishing is an atomic
    same-filesystem rename, and ``mkstemp`` creates it exclusively with private
    mode, so a crash can neither reuse a foreign file nor leave a world-readable
    partial artifact. The original suffix is preserved because the media layer
    selects its codec and bitrate from it.
    """
    descriptor, name = tempfile.mkstemp(
        prefix=".vo-native-", suffix=target.suffix, dir=str(target.parent)
    )
    os.close(descriptor)
    os.chmod(name, 0o600)
    return Path(name)


def _publish_staged(
    staged: Path,
    target: Path,
    *,
    error_code: str,
    size_bytes: int,
    sha256: str,
    sha256_file: Callable[[Path], str],
) -> None:
    """Publish once without replacing an existing output or following an alias.

    A byte-identical file already published just before a crash can be adopted
    without clobbering it; conflicting or symlinked files remain untouched. The
    hard link is an exclusive, same-filesystem publication of the private staging
    file, unlike a rename that silently replaces an existing paid artifact.
    """
    if target.parent.is_symlink() or target.is_symlink() or staged.is_symlink():
        raise NativeGenerationError(
            "refusing to publish audio through a symlink.",
            code=_EXIT_OUTPUT,
            error_code=error_code,
        )
    try:
        os.link(staged, target)
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            raise NativeGenerationError(
                "refusing to replace an existing non-regular audio output.",
                code=_EXIT_OUTPUT,
                error_code=error_code,
            ) from None
        try:
            same_bytes = target.stat().st_size == size_bytes and sha256_file(target) == sha256
        except OSError:
            same_bytes = False
        if not same_bytes:
            raise NativeGenerationError(
                "refusing to replace an existing conflicting audio output.",
                code=_EXIT_OUTPUT,
                error_code=error_code,
            ) from None
    staged.unlink()


class _Executor:
    """One locked native run's committed state, provider factory, and media hooks."""

    def __init__(
        self,
        *,
        repository: HistoryRepository,
        view: NativeTtsView,
        paths,
        run_root: Path,
        prepared: PreparedRun,
        chunks: list[ScriptChunk],
        script_format: str,
        script_path: Path,
        output_options: dict[str, Any],
        ffmpeg_path: str,
        ffprobe_path: str,
        provider_factory: Callable[[], Any],
        hooks: NativeExecutionHooks,
        logger: GenerationLogger,
        resume: bool,
    ) -> None:
        self.repository = repository
        self.view = view
        self.paths = paths
        self.run_root = run_root
        self.prepared = prepared
        self.chunks = chunks
        self.script_format = script_format
        self.script_path = script_path
        self.output_options = output_options
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.provider_factory = provider_factory
        self.hooks = hooks
        self.logger = logger
        self.resume = resume
        self._revision = view.run.revision
        self._provider: Any = None

    # -- provider laziness ----------------------------------------------------

    def _provider_instance(self) -> Any:
        """Build the provider on first real need; never for a local/export path."""
        if self._provider is None:
            self._provider = self.provider_factory()
        return self._provider

    def _run_uuid(self) -> str:
        return self.view.run.run_uuid

    def _model(self) -> str:
        return self.prepared.model

    # -- per-part processing --------------------------------------------------

    def _bind_media_callbacks(
        self,
        provider: Any,
        *,
        attempt: AttemptRecord,
        part: NativeTtsPart,
        accepted_id: dict[str, str],
    ) -> None:
        """Bind DB task-id and observed-cost transitions onto the provider callbacks.

        The accepted id is committed before the first poll or download, and the
        exact observed cost before the signed-URL download, exactly as the paid
        reservation order requires. A failure here propagates out of the provider
        call so the outcome stays unconfirmed and no second POST is ever sent.
        """
        run_uuid = self._run_uuid()
        part_uuid = part.record.part_uuid
        attempt_uuid = attempt.attempt_uuid

        def on_accepted(task_id: str) -> None:
            run, _updated = self.repository.record_polza_media_task_accepted(
                run_uuid,
                attempt_uuid=attempt_uuid,
                part_uuid=part_uuid,
                expected_revision=self._revision,
                remote_task_id=task_id,
            )
            self._revision = run.revision
            accepted_id["id"] = task_id
            self.logger.event("info", "remote_accepted", chunk=part.number, id=part.chunk_id)

        def on_completed(_task_id: str, usage: dict | None, _generation_id: str | None) -> None:
            cost, cost_exact = costs.media_observed_cost(usage)
            if cost is None:
                return
            amount: Decimal | float | str = cost_exact if cost_exact is not None else cost
            run, _updated = self.repository.record_polza_media_observed_cost(
                run_uuid,
                attempt_uuid=attempt_uuid,
                part_uuid=part_uuid,
                expected_revision=self._revision,
                amount=amount,
            )
            self._revision = run.revision
            self.logger.event("info", "cost_observed", chunk=part.number, id=part.chunk_id)

        provider.on_media_task_accepted = on_accepted
        provider.on_media_completed = on_completed

    def _convert_and_commit(
        self, evidence: _PartEvidence, result: SynthesisResult, attempt: AttemptRecord
    ) -> None:
        """Convert one accepted audio result into the canonical chunk artifact.

        The file is written and trimmed with the recorded semantics, hashed, and
        then committed with the digest, size, and bounded processing metadata in
        one short transaction. A conversion failure propagates as an output error
        while the raw and accepted-id evidence stay committed, so a later resume
        rebuilds locally instead of re-submitting.
        """
        part = evidence.part
        output_path = self.paths.chunks_dir / f"{part.chunk_id}.mp3"
        staged = _new_staging_path(output_path)
        try:
            self.hooks.write_audio_as_mp3(
                self.ffmpeg_path, result.audio_bytes, result.audio_format, staged
            )
            self.hooks.trim_final_silence(self.ffmpeg_path, self.ffprobe_path, staged)
            duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, staged)
            sha256 = self.hooks.sha256_file(staged)
            size_bytes = staged.stat().st_size
            _publish_staged(
                staged,
                output_path,
                error_code=_ERROR_CONVERSION_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            self.logger.event("error", "chunk_conversion_failed", chunk=part.number)
            raise NativeGenerationError(
                f"Failed to write chunk audio {output_path.name}: {exc}",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_CONVERSION_FAILED,
            ) from exc
        metadata = {
            "duration_ms": duration_ms,
            "generation_id": result.generation_id,
            "processing_version": OUTPUT_PROCESSING_VERSION,
        }
        try:
            run, _artifact = self.repository.record_tts_part_completed(
                self._run_uuid(),
                part_uuid=part.record.part_uuid,
                attempt_uuid=attempt.attempt_uuid,
                expected_revision=self._revision,
                path=f"chunks/{part.chunk_id}.mp3",
                mime="audio/mpeg",
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata=metadata,
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this part conflicts with the converted chunk; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        self.logger.event("info", "chunk_state_saved", chunk=part.number, id=part.chunk_id)

    def _link_raw(
        self,
        part: NativeTtsPart,
        attempt: AttemptRecord,
        *,
        audio_bytes: bytes,
        audio_format: str,
        remote_task_id: str | None,
        generation_id: str | None,
    ) -> AttemptRecord:
        """Persist accepted bytes and their receipt, then link them to the attempt."""
        receipt: PaidRawReceipt = write_paid_raw_receipt(
            run_root=self.run_root,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            synthesis_fingerprint=part.fingerprint,
            chunk_id=part.chunk_id,
            number=part.number,
            audio_format=audio_format,
            audio_bytes=audio_bytes,
            remote_task_id=remote_task_id,
            generation_id=generation_id,
        )
        run, updated, _artifact = self.repository.record_polza_media_raw_saved(
            self._run_uuid(),
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            receipt=receipt,
        )
        self._revision = run.revision
        return updated

    def _submit_fresh_part(self, evidence: _PartEvidence) -> None:
        """Reserve, submit once, persist raw evidence, and commit the conversion."""
        part = evidence.part
        run, attempt = self.repository.reserve_paid_tts_attempt(
            self._run_uuid(),
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            provider=self.prepared.provider,
            model=self._model(),
            account_alias=PAID_ACCOUNT_ALIAS,
        )
        self._revision = run.revision
        provider = self._provider_instance()
        accepted_id: dict[str, str] = {}
        self._bind_media_callbacks(provider, attempt=attempt, part=part, accepted_id=accepted_id)
        self.logger.event("info", "paid_submit_started", chunk=part.number, id=part.chunk_id)
        try:
            result = provider.synthesize_chunk(part.text, part.chunk_id)
        except Exception:
            # The provider exception body is untrusted (it may carry an echoed
            # Authorization header), so it never reaches the public error message
            # or a chained cause the machine envelope could reveal.
            self.logger.event("error", "chunk_provider_failed", chunk=part.number)
            raise NativeGenerationError(
                f"Failed to synthesize {part.chunk_id}.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SYNTHESIS_FAILED,
            ) from None
        updated = self._link_raw(
            part,
            attempt,
            audio_bytes=result.audio_bytes,
            audio_format=result.audio_format,
            remote_task_id=accepted_id.get("id"),
            generation_id=result.generation_id,
        )
        self._convert_and_commit(evidence, result, updated)

    def _recover_via_get(self, evidence: _PartEvidence, attempt: AttemptRecord) -> None:
        """Finish a known accepted task with GET calls only, then convert locally."""
        part = evidence.part
        assert attempt.remote_id is not None  # narrowed by the caller
        provider = self._provider_instance()
        accepted_id: dict[str, str] = {}
        self._bind_media_callbacks(provider, attempt=attempt, part=part, accepted_id=accepted_id)
        self.logger.event("info", "paid_media_recovery", chunk=part.number, id=part.chunk_id)
        try:
            result = provider.recover_media_task(attempt.remote_id, part.text, part.chunk_id)
        except Exception:
            # The provider exception body is untrusted and never becomes the public
            # error message or a chained cause.
            raise NativeGenerationError(
                f"Failed to recover media task for {part.chunk_id}.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SYNTHESIS_FAILED,
            ) from None
        updated = self._link_raw(
            part,
            attempt,
            audio_bytes=result.audio_bytes,
            audio_format=result.audio_format,
            remote_task_id=attempt.remote_id,
            generation_id=result.generation_id,
        )
        self._convert_and_commit(evidence, result, updated)

    def _convert_from_raw(self, evidence: _PartEvidence, attempt: AttemptRecord) -> None:
        """Rebuild one chunk from retained raw paid bytes with no provider call."""
        raw = evidence.raw_artifact
        assert raw is not None  # narrowed by the caller
        metadata = raw.media_metadata if isinstance(raw.media_metadata, dict) else {}
        audio_format = metadata.get("format")
        if not isinstance(audio_format, str):
            raise NativeGenerationError(
                "the retained paid raw artifact is missing its audio format.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        generation_id = metadata.get("generation_id")
        audio_bytes = _verified_raw_bytes(raw, self.run_root, self.hooks.sha256_file)
        result = SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            transcript=evidence.part.text,
            generation_id=generation_id if isinstance(generation_id, str) else None,
            client_path="requests",
        )
        self.logger.event(
            "info", "paid_raw_recovery", chunk=evidence.part.number, id=evidence.part.chunk_id
        )
        self._convert_and_commit(evidence, result, attempt)

    def _reconcile_raw_receipt(
        self, evidence: _PartEvidence, attempt: AttemptRecord
    ) -> ArtifactRecord | None:
        """Link an already-written raw receipt that has no database artifact yet.

        A crash between writing the paid bytes plus their receipt and committing
        the database row leaves valid evidence on disk. Re-verifying it and
        linking it here recovers that window without a provider request; the
        verification hashes the file outside any transaction. ``None`` means the
        on-disk evidence is absent or does not match, so the caller falls back to
        GET-only recovery; a return value is the newly committed raw artifact.
        """
        from ..history.raw_receipt import verify_paid_raw_receipt

        part = evidence.part
        receipt_format = _raw_format_from_receipt(self.run_root, part.chunk_id, part.number)
        if receipt_format is None:
            return None
        try:
            receipt = verify_paid_raw_receipt(
                run_root=self.run_root,
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                synthesis_fingerprint=part.fingerprint,
                chunk_id=part.chunk_id,
                number=part.number,
                audio_format=receipt_format,
                remote_task_id=attempt.remote_id,
            )
        except (PaidRawReceiptError, ValueError):
            return None
        try:
            run, _updated, artifact = self.repository.record_polza_media_raw_saved(
                self._run_uuid(),
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                receipt=receipt,
            )
        except HistoryRepositoryError:
            return None
        self._revision = run.revision
        return artifact

    def _process_first_incomplete(self, evidence: _PartEvidence) -> None:
        """Recover or submit exactly the first part without a verified chunk file."""
        attempt = evidence.attempt
        if attempt is None:
            # A converted artifact without an attempt cannot be reconciled safely.
            raise NativeGenerationError(
                "a committed part carries partial history without its paid attempt.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        status = attempt.status
        if status == ATTEMPT_STATUS_SUBMITTING:
            raise NativeGenerationError(
                "refusing to resume: the previous paid submit for this part was never confirmed. "
                "A paid outcome must not be repeated automatically.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SUBMIT_UNCONFIRMED,
            )
        if status not in (
            ATTEMPT_STATUS_REMOTE_ACCEPTED,
            ATTEMPT_STATUS_RAW_SAVED,
            ATTEMPT_STATUS_COMPLETED,
        ):
            raise NativeGenerationError(
                "refusing to resume: this part carries an unrecognized paid attempt state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        if evidence.raw_artifact is not None:
            self._convert_from_raw(evidence, attempt)
            return
        if status == ATTEMPT_STATUS_REMOTE_ACCEPTED and attempt.remote_id is not None:
            linked = self._reconcile_raw_receipt(evidence, attempt)
            if linked is not None:
                # The crash window between raw and database left valid evidence on
                # disk; rebuild that part locally with no provider request at all.
                self._convert_from_raw(replace(evidence, raw_artifact=linked), attempt)
                return
            self._recover_via_get(evidence, attempt)
            return
        if status == ATTEMPT_STATUS_RAW_SAVED:
            raise NativeGenerationError(
                "refusing to resume: this part records saved raw audio but its linked raw "
                "artifact is missing.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        raise NativeGenerationError(
            "refusing to resume: this part's paid evidence cannot be reconstructed locally.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_EVIDENCE_INCONSISTENT,
        )

    def _reload_view(self) -> None:
        """Reload the verified view so it reflects the parts this process just committed."""
        self.view = load_native_tts_view(self.repository, self._run_uuid())

    def run_parts(self) -> None:
        """Execute or recover every part, stopping before any inconsistent gap.

        Only the first part without a verified converted chunk may carry
        recoverable evidence. Any later part that already carries paid evidence
        while an earlier part is unfinished fails closed, so a partially written
        history can never be extended by a new paid submit.
        """
        evidence = _collect_evidence(self.view)
        first_incomplete_done = False
        for index, item in enumerate(evidence):
            if _verified_chunk_file(item, self.run_root, self.hooks.sha256_file) is not None:
                self.hooks.progress(
                    f"Skipping {item.part.chunk_id}/{len(evidence):02d}: already committed"
                )
                continue
            if any(_has_paid_evidence(later) for later in evidence[index + 1 :]):
                raise NativeGenerationError(
                    "refusing to resume: a later part carries paid evidence before this part "
                    "completed.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            if first_incomplete_done and _has_paid_evidence(item):
                raise NativeGenerationError(
                    "refusing to resume: paid evidence does not form a contiguous prefix.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            self.hooks.progress(
                f"Generating {item.part.chunk_id}/{len(evidence):02d}: {item.part.chunk_id}.mp3"
            )
            if _has_paid_evidence(item):
                self._process_first_incomplete(item)
            else:
                self._submit_fresh_part(item)
            first_incomplete_done = True

    # -- assembly and export --------------------------------------------------

    def _all_chunks_verified(self) -> list[Path]:
        """Return the ordered, verified chunk paths, or fail closed."""
        evidence = _collect_evidence(self.view)
        ordered: list[Path] = []
        for item in evidence:
            verified = _verified_chunk_file(item, self.run_root, self.hooks.sha256_file)
            if verified is None:
                raise NativeGenerationError(
                    "refusing to assemble: a committed chunk file is missing or does not match "
                    "its history.",
                    code=_EXIT_OUTPUT,
                    error_code=_ERROR_ASSEMBLY_FAILED,
                )
            ordered.append(verified)
        return ordered

    def assemble_and_complete(self) -> None:
        """Concatenate the ordered committed parts and close the run."""
        ordered = self._all_chunks_verified()
        output_path = self.paths.full_mp3
        staged = _new_staging_path(output_path)
        try:
            self.hooks.concat_audio_files(self.ffmpeg_path, ordered, staged)
            duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, staged)
            sha256 = self.hooks.sha256_file(staged)
            size_bytes = staged.stat().st_size
            _publish_staged(
                staged,
                output_path,
                error_code=_ERROR_ASSEMBLY_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            raise NativeGenerationError(
                f"Failed to concat MP3 chunks: {exc}",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_ASSEMBLY_FAILED,
            ) from exc
        try:
            run, _artifact = self.repository.record_tts_run_completed(
                self._run_uuid(),
                expected_revision=self._revision,
                path=self.paths.full_mp3.name,
                mime="audio/mpeg",
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata={
                    "duration_ms": duration_ms,
                    "execution_source": build_execution_identity(),
                },
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this run conflicts with the assembled final audio; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        self.logger.event("info", "run_completed", duration_ms=duration_ms)

    def _final_artifact_verified(self) -> bool:
        finals = [
            artifact
            for artifact in self.view.artifacts
            if artifact.role == ARTIFACT_ROLE_FINAL_AUDIO
        ]
        if len(finals) != 1:
            return False
        final = finals[0]
        if final.sha256 is None or final.size_bytes is None:
            return False
        path = self.run_root / final.path
        if not path.is_file():
            return False
        try:
            return (
                path.stat().st_size == final.size_bytes
                and self.hooks.sha256_file(path) == final.sha256
            )
        except OSError:
            return False

    def ensure_complete(self) -> None:
        """Complete the run only from verified committed parts.

        A run whose final artifact is already present and verified is left
        untouched, so an export-only repair neither reassembles audio nor reads an
        API key. Otherwise the ordered committed parts are assembled and the run
        is closed.
        """
        self._reload_view()
        if self.view.run.status == "completed" and self._final_artifact_verified():
            return
        self.assemble_and_complete()

    def export(self) -> NativeGenerationSummary:
        """Project one verified committed view onto the compatibility JSON files."""
        self._reload_view()
        view = self.view
        try:
            export = build_native_export(
                view,
                self.paths,
                script_path=self.script_path,
                ffmpeg_path=self.ffmpeg_path,
                ffprobe_path=self.ffprobe_path,
            )
            write_native_export(export, self.paths)
        except OSError as exc:
            raise NativeGenerationError(
                "Failed to write the compatibility JSON exports; the history database and audio "
                "are intact.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_EXPORT_FAILED,
            ) from exc
        except NativeGenerationError:
            raise
        except Exception as exc:
            raise NativeGenerationError(
                "Failed to project the committed history into the compatibility JSON exports; "
                "the history database and audio are intact.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_EXPORT_FAILED,
            ) from exc
        duration_ms = int(export.run_manifest.get("main_duration_ms") or 0)
        files = {
            "full_mp3": str(self.paths.full_mp3),
            "run_json": str(self.paths.run_json),
            "chunks_json": str(self.paths.chunks_json),
            "manifest_json": str(self.paths.output_root / "manifest.json"),
        }
        return NativeGenerationSummary(
            run_uuid=view.run.run_uuid,
            revision=view.run.revision,
            files=files,
            duration_ms=duration_ms,
            segment_count=None,
            cost_total=export.chunks_manifest.get("cost_total"),
            cost_currency=export.chunks_manifest.get("cost_currency"),
        )


def _raw_format_from_receipt(run_root: Path, chunk_id: str, number: int) -> str | None:
    """Return the bounded audio format an on-disk receipt names, or ``None``.

    Only the deterministic ``raw/<chunk-id>.<ext>.receipt.json`` files are read,
    and only their bounded ``format`` field is returned. A missing, unreadable,
    or malformed receipt reports ``None`` so the caller falls back to GET-only
    recovery instead of guessing a format.
    """
    raw_dir = run_root / "raw"
    if not raw_dir.is_dir():
        return None
    for candidate in sorted(raw_dir.glob(f"{chunk_id}.*.receipt.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            candidate_format = payload.get("format")
            if isinstance(candidate_format, str) and candidate_format:
                return candidate_format
    return None


def _find_native_run_in_repository(
    repository: HistoryRepository, canonical_root: str
) -> RunRecord | None:
    """Return the committed native run for a canonical root inside an open database.

    The locked executor re-verifies ownership against its own writable connection
    instead of trusting the pre-lock read. Two native runs on one root cannot be
    told apart, so that is a fail-closed conflict rather than a silent pick.
    """
    rows = repository.find_runs_by_root(canonical_root, limit=MAX_QUERY_LIMIT)
    native = [
        run
        for run in rows
        if run.legacy_source_root is None
        and isinstance(run.config_snapshot, dict)
        and run.config_snapshot.get("operation_origin") == NATIVE_SNAPSHOT_ORIGIN
    ]
    if len(native) > 1:
        raise NativeGenerationError(
            "more than one native history run owns this run directory; refusing to choose one.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    return native[0] if native else None


def _ensure_native_run_dirs(paths) -> None:
    """Create the native run's output directories after the lock and ownership check.

    The native route must not create a directory tree in a root before it has taken
    the run lock and re-verified ownership, so directory creation happens here
    instead of in the caller before ``run_native_generation`` takes the lock. A
    failure is an output error and leaves the database and audio untouched.
    """
    try:
        paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NativeGenerationError(
            "the native run output directory could not be created.",
            code=_EXIT_OUTPUT,
            error_code=_ERROR_OUTPUT_UNAVAILABLE,
        ) from exc


def execute_native_tts(
    *,
    repository: HistoryRepository,
    run_root: Path,
    paths,
    prepared: PreparedRun,
    chunks: list[ScriptChunk],
    script_format: str,
    script_path: Path,
    output_options: dict[str, Any],
    ffmpeg_path: str,
    ffprobe_path: str,
    user_label: str | None,
    resume: bool,
    provider_factory: Callable[[], Any],
    hooks: NativeExecutionHooks,
    logger: GenerationLogger,
) -> NativeGenerationSummary:
    """Run one locked native TTS generation, resume, or completed-run export.

    The caller has already taken the run lock and opened the writable history
    database. A fresh run commits its prepared snapshot and ownership descriptor
    before any provider exists; a resume proves the exact committed synthesis
    identity and output settings before any paid action; a completed run is only
    re-exported. The provider factory is invoked lazily and never for a local raw
    rebuild or an export-only repair.
    """
    canonical_root = str(run_root.resolve())
    existing = _find_native_run_in_repository(repository, canonical_root)
    if existing is None and not _fresh_run_root_available(run_root):
        raise NativeGenerationError(
            "the run directory already carries local run state; refusing to start a "
            "native run over it.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_RUN_MISSING,
        )
    _ensure_native_run_dirs(paths)
    run_uuid: str
    view: NativeTtsView
    if existing is None:
        script_text = "\n".join(chunk.text for chunk in chunks)
        try:
            snapshot = persist_prepared_tts_snapshot(
                repository,
                prepared=prepared,
                run_root=run_root,
                user_label=user_label,
                script_format=script_format,
                script_text=script_text,
                script_path=script_path,
                output_options=output_options,
            )
            run_uuid = snapshot.run.run_uuid
            _write_descriptor(run_root, run_uuid)
            view = load_native_tts_view(repository, run_uuid)
        except (NativeSnapshotError, NativeViewError) as exc:
            raise NativeGenerationError(
                f"refusing to start a native run: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_SNAPSHOT_FAILED",
            ) from exc
        resume = False
    else:
        run_uuid = existing.run_uuid
        _ensure_descriptor(run_root, run_uuid)
        try:
            view = load_native_tts_view(repository, run_uuid)
        except NativeViewError as exc:
            raise NativeGenerationError(
                f"refusing to resume: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_HISTORY_INTEGRITY",
            ) from exc
        if not resume:
            raise NativeGenerationError(
                "this run is owned by native history and already exists. Use --resume to "
                "continue it, or a different --run-id for a new run.",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_RUN_ALREADY_EXISTS",
            )
        if view.output_options is not None and view.output_options != output_options:
            raise NativeGenerationError(
                "refusing to resume: the recorded output-processing settings differ from this "
                "command.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_PROCESSING_CHANGED,
            )
        try:
            preflight_native_tts_resume(
                repository,
                run_uuid,
                expected_revision=view.run.revision,
                prepared=prepared,
                script_format=script_format,
                run_root=run_root,
            )
        except (NativeResumeError, NativeViewError) as exc:
            raise NativeGenerationError(
                f"refusing to resume: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_RESUME_IDENTITY_CHANGED",
            ) from exc

    executor = _Executor(
        repository=repository,
        view=view,
        paths=paths,
        run_root=run_root,
        prepared=prepared,
        chunks=chunks,
        script_format=script_format,
        script_path=script_path,
        output_options=output_options,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        provider_factory=provider_factory,
        hooks=hooks,
        logger=logger,
        resume=resume,
    )
    executor.run_parts()
    executor.ensure_complete()
    return executor.export()


def run_native_generation(
    *,
    paths,
    prepared: PreparedRun,
    chunks: list[ScriptChunk],
    script_format: str,
    script_path: Path,
    output_options: dict[str, Any],
    ffmpeg_path: str,
    ffprobe_path: str,
    user_label: str | None,
    resume: bool,
    provider_factory: Callable[[], Any],
    hooks: NativeExecutionHooks,
    database_path: Path | None = None,
) -> NativeGenerationSummary:
    """Hold the run lock and the history database around one native execution.

    The generation logger is constructed here, after the run lock is taken, so the
    native route creates no file in the run root before the lock and the ownership
    recheck inside :func:`execute_native_tts`.
    """
    try:
        with acquire_run_lock(paths.output_root):
            logger = GenerationLogger(paths.output_root / LOG_FILE)
            with _open_writable_history(database_path) as repository:
                return execute_native_tts(
                    repository=repository,
                    run_root=paths.output_root,
                    paths=paths,
                    prepared=prepared,
                    chunks=chunks,
                    script_format=script_format,
                    script_path=script_path,
                    output_options=output_options,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    user_label=user_label,
                    resume=resume,
                    provider_factory=provider_factory,
                    hooks=hooks,
                    logger=logger,
                )
    except HistoryRunLockedError as exc:
        raise NativeGenerationError(
            "another process is already writing this run directory; refusing to run two "
            "writers for one native run.",
            code=_EXIT_PROVIDER,
            error_code="NATIVE_RUN_LOCKED",
        ) from exc
