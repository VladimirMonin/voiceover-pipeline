"""Canonical-history boundary for one paid cloud transcription request.

The existing local ASR/timing/verify writer (:mod:`history.native_asr`) records
one *completed* run in a single transaction, which is safe because a local model
has no paid submit to protect. A cloud transcription route (Groq Whisper, xAI STT)
is different: one synchronous multipart POST is the paid call, and a crash or a
lost response must leave durable evidence instead of a silently repeatable
submit. This module is that boundary and nothing else.

Lifecycle (three durable phases, one run):

1. :func:`reserve_paid_transcription` commits the run row, its bounded operation
   identity, and one ``submitting`` attempt marker *before* the caller may POST,
   and writes a run-local ownership descriptor into the output root. A failure
   here means the caller makes zero requests.
2. :func:`record_paid_transcription_raw` writes a bounded private receipt and then
   the exact successful response body to the run-local ``raw`` directory, both
   fsynced, and only then links the body to the same attempt and moves it to
   ``raw_saved``. The receipt is published *before* the body, so a crash
   immediately after the body replacement can still validate the pair. It runs
   before any fallible JSON parse, so a successful paid response is never lost to
   a parsing or downstream failure.
3. :func:`complete_paid_transcription` records the observed transcript and
   artifact references and closes the run with the attempt at ``completed``.

A POST/stream/timeout failure or an unconfirmed response leaves the attempt in
``submitting``: the durable marker blocks an automatic retry, fallback, resume,
overwrite, or sync from issuing a second paid request. An explicit
:func:`recover_paid_transcription_after_crash` reconciles a receipt+body pair the
crash left unlinked and returns the same attempt, so a resume replays it locally
with no second POST; :func:`read_paid_transcription_raw` returns the validated
body for that replay.

Ownership is bound to the *canonical output root*, not the newly generated history
UUID: :func:`resolve_paid_timing_ownership` reads a bounded run-local descriptor
and an exact, read-only database lookup by ``run_root``, so a fresh invocation or
``--overwrite`` against any already-owned root fails closed before any deletion,
key access, or POST. A committed database run wins when the descriptor is missing;
a descriptor that names a different or missing run fails closed.

The store is deliberately not a second state engine: it opens the history
database, takes no lock of its own (the caller holds the shared run lock), makes
no provider call, polls nothing, and invents no remote id. Every attempt keeps
:meth:`Cost.unknown` (a ``NULL`` amount), because neither adapter reports an
observed cost. Every failure raises :class:`PaidTranscriptionError` with a fixed,
content-free message; the raw body is private run-local evidence and never reaches
a machine-visible payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .database import (
    HistoryDatabase,
    HistoryDatabaseError,
    connect_readonly_consistent,
    readonly_schema_is_empty,
)
from .native_asr import (
    OPERATION_TIMINGS,
    OPERATION_VERIFY,
    AsrHistoryArtifact,
    AsrHistoryText,
    _file_evidence,
    _resolve_parent_run,
    sha256_text,
)
from .native_snapshot import NATIVE_SNAPSHOT_ORIGIN
from .paths import (
    HistoryPathsError,
    ensure_history_home,
    ensure_private_directory,
    history_database_path,
)
from .repository import (
    ATTEMPT_STATUS_COMPLETED,
    ATTEMPT_STATUS_RAW_SAVED,
    ATTEMPT_STATUS_SUBMITTING,
    AVAILABILITY_PRESENT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    PATH_KIND_MANAGED_RELATIVE,
    RUN_STATUS_COMPLETED,
    Cost,
    HistoryRepository,
)

# Call types distinct from the local ``asr_timing``/``tts_quality_asr`` values, so
# a paid attempt is never read as a local one by a resume, a cost read, or a
# future migration.
ATTEMPT_CALL_TYPE_PAID_TIMING = "paid_asr_timing"
ATTEMPT_CALL_TYPE_PAID_QUALITY = "paid_tts_quality_asr"

# Provenance written into every run snapshot, so a paid cloud transcription run is
# distinguishable from a local ``native_asr`` run.
PAID_TRANSCRIPTION_ORIGIN = "native_paid_transcription"

# The run status a reservation commits and the completed transition writes.
RUN_STATUS_RUNNING_TEXT = "running"

# The one managed artifact that links a saved paid response body to its attempt.
ARTIFACT_ROLE_PROVIDER_RAW = "provider_raw_response"

# Run-local directory that holds the private response bodies and receipts.
RAW_RESPONSE_DIRECTORY_NAME = "raw"
# A provider response body is private but also attacker-influenceable if a proxy
# is compromised, so the reader never allocates an unbounded file before it can be
# validated. A body larger than this bound is refused instead of parsed.
_MAX_RAW_BODY_BYTES = 32 * 1024 * 1024
# A receipt is a bounded structural document with a fixed field set.
_MAX_RECEIPT_BYTES = 64 * 1024

# Run-local ownership descriptor. It names the committed paid run and carries no
# progress and no execution permission; the run lock serializes writers and the
# database remains the only progress truth.
PAID_OWNERSHIP_FILE_NAME = ".voiceover-paid-timing.json"
PAID_OWNERSHIP_ARTIFACT_TYPE = "voiceover-paid-timing-ownership"
PAID_OWNERSHIP_VERSION = 1

# The bounded private receipt that binds one saved response body to its attempt.
RAW_RECEIPT_ARTIFACT_TYPE = "voiceover-paid-transcription-raw-receipt"
RAW_RECEIPT_VERSION = 1
# The receipt suffix must not match the native TTS ``*.receipt.json`` glob, so a
# paid timing run root is never mistaken for native TTS ownership evidence.
_RECEIPT_SUFFIX = ".receipt"

# The operation a call type belongs to. A caller passes the operation explicitly,
# but the boundary refuses a mismatched pair so a run never carries a timing
# attempt under the verify operation or the reverse.
_OPERATION_BY_CALL_TYPE = {
    ATTEMPT_CALL_TYPE_PAID_TIMING: OPERATION_TIMINGS,
    ATTEMPT_CALL_TYPE_PAID_QUALITY: OPERATION_VERIFY,
}

# Fixed, content-free failure messages. None echoes a stored value or path.
_RESERVE_FAILED = "the paid transcription attempt could not be reserved"
_RAW_WRITE_FAILED = "the paid transcription response could not be stored"
_COMPLETE_FAILED = "the paid transcription result could not be stored"
_READ_FAILED = "the paid transcription history could not be read"


class PaidTranscriptionError(RuntimeError):
    """A paid-transcription history contract violation with a stable code."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class PaidTranscriptionRequest:
    """The bounded identity one paid transcription request commits before POST.

    ``config_snapshot`` carries the effective request the caller will send: the
    provider, model, language, timestamp granularity, and provider-specific
    request options. It never carries a key, a signed URL, or the transcript.
    ``output_root`` is the canonical output directory the run owns; ownership is
    bound to it.
    """

    call_type: str
    provider: str
    model: str
    language: str | None
    word_timestamps: bool
    source_audio: Path
    output_root: Path
    source_audio_mime: str | None = None
    request_options: dict[str, Any] = field(default_factory=dict)
    parent_run_root: Path | None = None
    user_label: str | None = None


@dataclass
class PaidTranscriptionReservation:
    """The committed identity of one in-flight paid transcription attempt.

    ``revision`` is the run revision this reservation owns; each phase advances it
    through the same compare-and-swap the native TTS reservation uses, so a
    concurrent writer cannot double-commit a phase. ``output_root`` is the
    canonical directory the run owns and where its private raw evidence lives.
    """

    run_uuid: str
    attempt_uuid: str
    run_root: Path
    output_root: Path
    revision: int
    operation: str
    call_type: str
    provider: str
    model: str
    operation_fingerprint: str


@dataclass(frozen=True)
class PaidTranscriptionState:
    """The durable state a resume or sync reads without touching a provider.

    ``snapshot`` is the bounded, non-secret operation identity the reservation
    committed (effective request options plus the caller's output locator), so a
    resume can replay the same attempt locally without re-reading the original
    command line. ``run_root`` is the independently committed canonical database
    binding used to reject a retargeted output locator before local replay.
    """

    run_uuid: str
    run_root: str
    revision: int
    status: str
    attempt_uuid: str
    attempt_status: str
    raw_saved: bool
    raw_missing: bool
    operation: str
    call_type: str
    provider: str
    model: str
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class PaidTimingOwnership:
    """Whether one output root is already owned by a committed paid timing run."""

    run_uuid: str | None

    @property
    def owned(self) -> bool:
        return self.run_uuid is not None


class _ReadOnlyConnection:
    """Minimal handle exposing ``.connection`` to :class:`HistoryRepository`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


def _require_cloud_provider(call_type: str, provider: str) -> None:
    """Refuse a call type this boundary does not own before any write."""
    if call_type not in _OPERATION_BY_CALL_TYPE:
        raise PaidTranscriptionError(
            "the paid transcription boundary does not own this call type",
            error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
        )
    if not provider or not provider.strip():
        raise PaidTranscriptionError(
            "a paid transcription attempt requires a provider",
            error_code="PAID_TRANSCRIPTION_INVALID",
        )


def _open_database() -> HistoryDatabase:
    """Open and migrate the history database for writing, or raise a fixed error."""
    try:
        ensure_history_home()
        database = HistoryDatabase(history_database_path())
        database.connect()
        database.migrate()
    except (HistoryDatabaseError, HistoryPathsError, sqlite3.Error, OSError) as exc:
        raise PaidTranscriptionError(_RESERVE_FAILED, error_code="HISTORY_UNAVAILABLE") from exc
    return database


def _raw_file_name(attempt_uuid: str) -> str:
    return f"{attempt_uuid}.body"


def _receipt_file_name(attempt_uuid: str) -> str:
    return f"{_raw_file_name(attempt_uuid)}{_RECEIPT_SUFFIX}"


def _raw_relative_path(attempt_uuid: str) -> str:
    return f"{RAW_RESPONSE_DIRECTORY_NAME}/{_raw_file_name(attempt_uuid)}"


def _raw_directory(output_root: Path) -> Path:
    return output_root / RAW_RESPONSE_DIRECTORY_NAME


def _raw_body_path(output_root: Path, attempt_uuid: str) -> Path:
    return _raw_directory(output_root) / _raw_file_name(attempt_uuid)


def _receipt_path(output_root: Path, attempt_uuid: str) -> Path:
    return _raw_directory(output_root) / _receipt_file_name(attempt_uuid)


def _ensure_raw_directory(output_root: Path) -> Path:
    """Create or verify the private ``raw`` directory inside the output root."""
    raw_dir = _raw_directory(output_root)
    ensure_private_directory(raw_dir)
    return raw_dir


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a crash where supported."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to a private regular file, replacing nothing on failure."""
    try:
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        raise PaidTranscriptionError(_RAW_WRITE_FAILED, error_code="PAID_RAW_WRITE_FAILED") from exc
    temp = Path(name)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except OSError as exc:
            raise PaidTranscriptionError(
                _RAW_WRITE_FAILED, error_code="PAID_RAW_WRITE_FAILED"
            ) from exc
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    _fsync_directory(path.parent)


def _operation_fingerprint(
    *,
    provider: str,
    model: str,
    language: str | None,
    word_timestamps: bool,
    request_options: dict[str, Any],
) -> str:
    """Return a stable digest of the effective request identity.

    The fingerprint binds a receipt to the exact request shape, so a receipt from
    a different provider/model/options is never adopted for this run. It carries
    no secret and no transcript.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "language": language,
            "word_timestamps": bool(word_timestamps),
            "request_options": request_options,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_readable_source(source_audio: Path | str) -> tuple[str, int, str]:
    """Return ``(canonical path, size, sha256)`` for paid-ready source audio.

    A paid reservation never proceeds on a missing or unreadable source: unlike a
    local run, an absent file cannot be reconstructed, and a later resume needs a
    stable identity to prove the audio did not change. A non-regular file is
    refused as well.
    """
    availability, size, digest, resolved = _file_evidence(Path(source_audio))
    if availability != AVAILABILITY_PRESENT or size is None or digest is None:
        raise PaidTranscriptionError(
            "the paid transcription source audio is not a readable regular file",
            error_code="PAID_SOURCE_INVALID",
        )
    return resolved, size, digest


def verify_source_identity(snapshot: dict[str, Any]) -> Path:
    """Return the recorded source audio path, or fail closed when it changed.

    A resume verifies the stored path, size, and digest before parsing, probing, or
    publishing, so a replaced, removed, or truncated source never yields a
    transcript with misleading provenance.
    """
    path = snapshot.get("source_audio")
    size = snapshot.get("source_audio_size_bytes")
    digest = snapshot.get("source_audio_sha256")
    if not isinstance(path, str) or not path:
        raise PaidTranscriptionError(
            "the paid transcription run does not record its source audio",
            error_code="PAID_TRANSCRIPTION_INVALID",
        )
    availability, actual_size, actual_digest, resolved = _file_evidence(Path(path))
    if availability != AVAILABILITY_PRESENT:
        raise PaidTranscriptionError(
            "the paid transcription source audio is missing",
            error_code="PAID_SOURCE_UNAVAILABLE",
        )
    if actual_size != size or actual_digest != digest:
        raise PaidTranscriptionError(
            "the paid transcription source audio changed",
            error_code="PAID_SOURCE_CHANGED",
        )
    return Path(resolved)


# -- ownership descriptor -----------------------------------------------------


def _ownership_descriptor_path(output_root: Path) -> Path:
    return output_root / PAID_OWNERSHIP_FILE_NAME


def _read_ownership_descriptor(output_root: Path) -> str | None:
    """Return the paid run UUID the descriptor names, or fail closed.

    A missing descriptor is ``None``. A symlinked descriptor is refused before its
    target is read, and a present descriptor that is not a bounded ownership
    document raises :class:`PaidTranscriptionError`, so a malformed marker never
    becomes an unowned output root.
    """
    path = _ownership_descriptor_path(output_root)
    if path.is_symlink():
        raise PaidTranscriptionError(
            "the paid timing output descriptor must not be a symlink",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise PaidTranscriptionError(
            "the paid timing output descriptor is unreadable",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type",
        "ownership_version",
        "run_uuid",
    }:
        raise PaidTranscriptionError(
            "the paid timing output descriptor is malformed",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        )
    if (
        payload["artifact_type"] != PAID_OWNERSHIP_ARTIFACT_TYPE
        or payload["ownership_version"] != PAID_OWNERSHIP_VERSION
    ):
        raise PaidTranscriptionError(
            "the paid timing output descriptor has an unsupported version",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        )
    run_uuid = payload["run_uuid"]
    if not isinstance(run_uuid, str) or not run_uuid:
        raise PaidTranscriptionError(
            "the paid timing output descriptor names no run",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        )
    return run_uuid


def _write_ownership_descriptor(output_root: Path, run_uuid: str) -> None:
    """Atomically write the tiny ownership descriptor for a committed paid run."""
    payload = {
        "artifact_type": PAID_OWNERSHIP_ARTIFACT_TYPE,
        "ownership_version": PAID_OWNERSHIP_VERSION,
        "run_uuid": run_uuid,
    }
    path = _ownership_descriptor_path(output_root)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        raise PaidTranscriptionError(
            "the paid timing ownership descriptor could not be written",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        ) from exc
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
    _fsync_directory(path.parent)


def paid_local_evidence(run_root: Path) -> bool:
    """Whether the run root carries local evidence of a paid timing run.

    A paid timing run writes its ownership descriptor into the canonical output
    root and keeps its response body under ``raw/``. Either marker means another
    writer already owns this directory, so a native or legacy writer must not
    freshly admit it. The check is read-only and never raises for a missing or
    unreadable path.
    """
    if _ownership_descriptor_path(run_root).exists():
        return True
    raw_dir = _raw_directory(run_root)
    return raw_dir.is_dir() and any(raw_dir.glob("*.body"))


def committed_paid_run_for_root(repository: HistoryRepository, canonical_root: str) -> str | None:
    """Return the paid run UUID committed for ``canonical_root``, or ``None``.

    The lookup is the exact, untruncated ``run_root`` plus paid-origin query, so an
    older paid run is never hidden behind newer unrelated rows at the same root.
    Two paid runs on one root cannot be told apart, so that state fails closed with
    a fixed ownership error instead of choosing one. The caller holds the shared
    run lock for the root, so this read decides one interleaved writer.
    """
    paid = repository.find_runs_by_root_origin(canonical_root, PAID_TRANSCRIPTION_ORIGIN)
    if len(paid) > 1:
        raise PaidTranscriptionError(
            "the output directory is bound to more than one paid timing run",
            error_code="PAID_TIMING_OWNERSHIP_MISMATCH",
        )
    return paid[0].run_uuid if paid else None


def _find_paid_run(canonical_root: str) -> tuple[str | None, bool]:
    """Return the committed paid run for a root and whether the database was usable.

    The lookup is one exact, untruncated SQL statement over ``run_root`` plus the
    paid operation origin, so an older paid run is never hidden behind newer
    unrelated rows at the same root. The read is read-only and writes nothing: an
    absent database means no paid ownership, while an unreadable, foreign, or
    newer database reports itself unusable so the caller fails closed rather than
    treating its root as unowned. A genuinely empty database that the read-only
    schema gate validated (no tables, ``user_version`` 0) also means no paid
    ownership, so it is not misread as unusable. Two paid runs on one root cannot
    be told apart and also report unusable.
    """
    try:
        db_path = history_database_path()
    except HistoryPathsError:
        return None, False
    if not db_path.exists():
        return None, True
    connection: sqlite3.Connection | None = None
    run_uuid: str | None
    try:
        connection = connect_readonly_consistent(db_path)
        if readonly_schema_is_empty(connection):
            # A read-only-validated empty schema has no runs table and therefore no
            # paid owner; it is a fresh database, not an unreadable one.
            return None, True
        repository = HistoryRepository(cast(HistoryDatabase, _ReadOnlyConnection(connection)))
        run_uuid = committed_paid_run_for_root(repository, canonical_root)
    except (PaidTranscriptionError, HistoryDatabaseError, sqlite3.Error, OSError):
        return None, False
    finally:
        if connection is not None:
            connection.close()
    return run_uuid, True


def resolve_paid_timing_ownership(output_root: Path | str) -> PaidTimingOwnership:
    """Resolve who owns one output root without writing anything.

    A committed paid run row, or a present descriptor, marks the root as owned. A
    descriptor naming a different or missing run, or an unusable history database
    (unreadable, foreign, newer, or ambiguous), fails closed with a fixed error
    before the caller may delete or overwrite the root. Every other root reports
    unowned, which keeps a genuinely fresh output directory usable.
    """
    root = Path(output_root).expanduser()
    if root.is_symlink():
        raise PaidTranscriptionError(
            "the paid timing output directory must not be a symlink",
            error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE",
        )
    canonical_root = str(root.resolve())
    descriptor_uuid = _read_ownership_descriptor(root)
    run_uuid, database_usable = _find_paid_run(canonical_root)
    if run_uuid is not None:
        if descriptor_uuid is not None and descriptor_uuid != run_uuid:
            raise PaidTranscriptionError(
                "the paid timing output descriptor names a different run than the committed "
                "history",
                error_code="PAID_TIMING_OWNERSHIP_MISMATCH",
            )
        return PaidTimingOwnership(run_uuid)
    if not database_usable:
        # The database could not be read, so it cannot prove this root is unowned.
        # Fail closed before any caller deletes or overwrites it.
        raise PaidTranscriptionError(
            "the history database could not be read; refusing to treat this output directory "
            "as unowned",
            error_code="HISTORY_UNAVAILABLE",
        )
    if descriptor_uuid is not None:
        raise PaidTranscriptionError(
            "this output directory is owned by a paid timing run, but its committed history "
            "run no longer exists",
            error_code="PAID_TIMING_OWNERSHIP_MISMATCH",
        )
    return PaidTimingOwnership(None)


# -- raw receipt --------------------------------------------------------------


def _receipt_payload(
    *,
    run_uuid: str,
    attempt_uuid: str,
    operation_fingerprint: str,
    relative_path: str,
    size: int,
    sha256: str,
    content_type: str,
) -> dict[str, Any]:
    return {
        "artifact_type": RAW_RECEIPT_ARTIFACT_TYPE,
        "receipt_version": RAW_RECEIPT_VERSION,
        "run_uuid": run_uuid,
        "attempt_uuid": attempt_uuid,
        "operation_fingerprint": operation_fingerprint,
        "relative_path": relative_path,
        "size": size,
        "sha256": sha256,
        "content_type": content_type,
    }


def _read_receipt(path: Path) -> dict[str, Any] | None:
    """Read a bounded receipt as a fixed-field mapping, or ``None`` when invalid."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_RECEIPT_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("artifact_type") != RAW_RECEIPT_ARTIFACT_TYPE:
        return None
    if payload.get("receipt_version") != RAW_RECEIPT_VERSION:
        return None
    return payload


def _verify_raw_evidence(
    output_root: Path,
    run_uuid: str,
    attempt_uuid: str,
    operation_fingerprint: str | None,
) -> tuple[str, int, str, str] | None:
    """Return validated ``(relative_path, size, sha256, content_type)``, else None.

    The receipt and body must both be regular non-symlink files inside a private
    ``raw`` directory, the receipt must carry exactly this run/attempt/fingerprint
    identity, and the body's size and digest must match. A half, foreign,
    mismatched, oversized, or tampered pair is refused, so a crash window can only
    be reconciled from exactly this attempt's evidence.
    """
    raw_dir = _raw_directory(output_root)
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        return None
    receipt_path = _receipt_path(output_root, attempt_uuid)
    body_path = _raw_body_path(output_root, attempt_uuid)
    payload = _read_receipt(receipt_path)
    if payload is None:
        return None
    if payload.get("run_uuid") != run_uuid or payload.get("attempt_uuid") != attempt_uuid:
        return None
    if operation_fingerprint is not None and (
        payload.get("operation_fingerprint") != operation_fingerprint
    ):
        return None
    relative_path = payload.get("relative_path")
    size = payload.get("size")
    digest = payload.get("sha256")
    content_type = payload.get("content_type")
    if (
        relative_path != _raw_relative_path(attempt_uuid)
        or not isinstance(size, int)
        or not isinstance(digest, str)
        or not isinstance(content_type, str)
    ):
        return None
    if size < 0 or size > _MAX_RAW_BODY_BYTES:
        return None
    try:
        if body_path.is_symlink() or not body_path.is_file():
            return None
        actual_size = body_path.stat().st_size
        if actual_size != size:
            return None
        actual_digest = _sha256_file(body_path)
    except OSError:
        return None
    if actual_digest != digest:
        return None
    return relative_path, size, digest, content_type


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- write phases -------------------------------------------------------------


def _committed_native_run_for_root(
    repository: HistoryRepository, canonical_root: str
) -> str | None:
    """Return a committed native TTS run that already owns ``canonical_root``, or ``None``.

    The native TTS route binds its ownership descriptor, chunks, and final audio to
    the same canonical output root, and no reader reconciles one directory holding
    both a native run and a paid one. A reservation must therefore refuse a root a
    native run already committed instead of creating the second owner. The lookup
    reuses the exact run-root-plus-origin repository query, so an older native run
    is never hidden behind newer unrelated rows at the same root, and a
    legacy-imported run that only shares the directory is not a native owner.
    """
    native = [
        run
        for run in repository.find_runs_by_root_origin(canonical_root, NATIVE_SNAPSHOT_ORIGIN)
        if run.legacy_source_root is None
    ]
    return native[0].run_uuid if native else None


def reserve_paid_transcription(request: PaidTranscriptionRequest) -> PaidTranscriptionReservation:
    """Commit the durable operation identity and attempt marker before any POST.

    The readable regular source audio is validated first, and a run root already
    committed to a native TTS run is refused before the reservation; then the run
    row, its ``running`` status, the operation identity snapshot (including the
    canonical output root binding), and one ``submitting`` attempt are written in a
    single transaction, and the run-local ownership descriptor is written after it
    commits. The caller may only start the provider request *after* this returns; a
    failure here raises :class:`PaidTranscriptionError` and the caller makes zero
    requests.
    """
    _require_cloud_provider(request.call_type, request.provider)
    operation = _OPERATION_BY_CALL_TYPE[request.call_type]
    resolved_source, size, digest = require_readable_source(request.source_audio)
    output_root = Path(request.output_root).expanduser().resolve()
    run_uuid = str(uuid.uuid4())
    request_options: dict[str, Any] = json.loads(json.dumps(dict(request.request_options)))
    fingerprint = _operation_fingerprint(
        provider=request.provider,
        model=request.model,
        language=request.language,
        word_timestamps=bool(request.word_timestamps),
        request_options=request_options,
    )
    snapshot: dict[str, Any] = {
        "operation_origin": PAID_TRANSCRIPTION_ORIGIN,
        "provider": request.provider,
        "model": request.model,
        "language": request.language,
        "word_timestamps_requested": bool(request.word_timestamps),
        "request_options": request_options,
        "output_root": str(output_root),
        "operation_fingerprint": fingerprint,
        "source_audio": resolved_source,
        "source_audio_availability": AVAILABILITY_PRESENT,
        "source_audio_size_bytes": size,
        "source_audio_sha256": digest,
        "cost_known": False,
    }
    database = _open_database()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            # One directory, one owner: a root a committed native TTS run already
            # owns is refused before the paid run row, its attempt marker, or the
            # ownership descriptor exist, so no paid evidence can be created over it.
            if _committed_native_run_for_root(repository, str(output_root)) is not None:
                raise PaidTranscriptionError(
                    "the output directory is owned by a committed native TTS run; refusing to "
                    "reserve a paid timing attempt over its run root.",
                    error_code="NATIVE_TIMING_OUTPUT_OWNED",
                )
            parent_uuid = _resolve_parent_run(repository, request.parent_run_root)
            run = repository.create_run(
                operation=operation,
                run_root=str(output_root),
                status=RUN_STATUS_RUNNING_TEXT,
                user_label=request.user_label,
                parent_uuid=parent_uuid,
                config_snapshot=snapshot,
                run_uuid=run_uuid,
            )
            attempt = repository.add_attempt(
                run.run_uuid,
                call_type=request.call_type,
                provider=request.provider,
                model=request.model,
                status=ATTEMPT_STATUS_SUBMITTING,
                cost=Cost.unknown(),
            )
    except PaidTranscriptionError:
        raise
    except (HistoryDatabaseError, sqlite3.Error, OSError, ValueError) as exc:
        raise PaidTranscriptionError(_RESERVE_FAILED, error_code="HISTORY_WRITE_FAILED") from exc
    finally:
        database.close()
    revision = run.revision
    _write_ownership_descriptor(output_root, run_uuid)
    return PaidTranscriptionReservation(
        run_uuid=run_uuid,
        attempt_uuid=attempt.attempt_uuid,
        run_root=output_root,
        output_root=output_root,
        revision=revision,
        operation=operation,
        call_type=request.call_type,
        provider=request.provider,
        model=request.model,
        operation_fingerprint=fingerprint,
    )


def record_paid_transcription_raw(
    reservation: PaidTranscriptionReservation,
    *,
    body: bytes,
    content_type: str,
) -> None:
    """Persist the exact successful provider body before any fallible parse.

    The bounded private receipt is written and fsynced first, then the bounded
    private body, so a crash immediately after the body replacement leaves a
    validated receipt+body pair for a later explicit resume. Only then are the
    body artifact, the attempt transition, and the run revision committed in one
    transaction. The reservation's revision is advanced after that commit. Any
    failure raises and leaves the attempt ``submitting`` (never ``raw_saved``), so
    no later phase may treat a missing body as a completed request.
    """
    if not isinstance(body, (bytes, bytearray)) or not body:
        raise PaidTranscriptionError(
            "the paid transcription response body is empty",
            error_code="PAID_TRANSCRIPTION_INVALID",
        )
    data = bytes(body)
    if len(data) > _MAX_RAW_BODY_BYTES:
        raise PaidTranscriptionError(
            "the paid transcription response body exceeds the bounded maximum",
            error_code="PAID_RAW_TOO_LARGE",
        )
    digest = hashlib.sha256(data).hexdigest()
    relative_path = _raw_relative_path(reservation.attempt_uuid)
    output_root = reservation.output_root
    receipt_payload = _receipt_payload(
        run_uuid=reservation.run_uuid,
        attempt_uuid=reservation.attempt_uuid,
        operation_fingerprint=reservation.operation_fingerprint,
        relative_path=relative_path,
        size=len(data),
        sha256=digest,
        content_type=content_type or "application/json",
    )
    receipt_bytes = (
        json.dumps(receipt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _ensure_raw_directory(output_root)
    _atomic_write_private(_receipt_path(output_root, reservation.attempt_uuid), receipt_bytes)
    _atomic_write_private(_raw_body_path(output_root, reservation.attempt_uuid), data)
    database = _open_database()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            advanced = repository.advance_run_revision(
                reservation.run_uuid, expected_revision=reservation.revision
            )
            repository.add_artifact(
                reservation.run_uuid,
                role=ARTIFACT_ROLE_PROVIDER_RAW,
                path_kind=PATH_KIND_MANAGED_RELATIVE,
                path=relative_path,
                attempt_uuid=reservation.attempt_uuid,
                mime=receipt_payload["content_type"],
                size_bytes=len(data),
                sha256=digest,
            )
            repository.set_paid_transcription_attempt_status(
                reservation.run_uuid,
                attempt_uuid=reservation.attempt_uuid,
                call_type=reservation.call_type,
                expected_status=ATTEMPT_STATUS_SUBMITTING,
                status=ATTEMPT_STATUS_RAW_SAVED,
            )
    except PaidTranscriptionError:
        raise
    except (HistoryDatabaseError, sqlite3.Error, OSError, ValueError) as exc:
        raise PaidTranscriptionError(_RAW_WRITE_FAILED, error_code="PAID_RAW_WRITE_FAILED") from exc
    finally:
        database.close()
    reservation.revision = advanced.revision


def complete_paid_transcription(
    reservation: PaidTranscriptionReservation,
    *,
    artifacts: Sequence[AsrHistoryArtifact] = (),
    text_sources: Sequence[AsrHistoryText] = (),
    result_metadata: dict[str, Any] | None = None,
    required_artifact_roles: Sequence[str] = (),
) -> None:
    """Store the observed transcript/artifacts and close the run.

    Runs only when the attempt is ``raw_saved``: a request whose body was never
    persisted can never be marked complete. Every required artifact role must
    resolve to a present regular file or completion fails closed, so a missing
    timings/SRT leaf is never silently accepted. Each present artifact is
    referenced, each private text source is stored verbatim, and the attempt moves
    to ``completed`` while the run compare-and-swaps to ``completed`` in the same
    transaction. ``result_metadata`` is bounded, non-secret provenance attached to
    the attempt usage; it never carries transcript text.
    """
    present_roles: set[str] = set()
    resolved_artifacts: list[tuple[AsrHistoryArtifact, str, int, str]] = []
    for artifact in artifacts:
        availability, size, digest, resolved = _file_evidence(artifact.path)
        if availability != AVAILABILITY_PRESENT or size is None or digest is None:
            continue
        present_roles.add(artifact.role)
        resolved_artifacts.append((artifact, resolved, size, digest))
    missing = [role for role in required_artifact_roles if role not in present_roles]
    if missing:
        raise PaidTranscriptionError(
            _COMPLETE_FAILED, error_code="PAID_TRANSCRIPTION_ARTIFACT_MISSING"
        )
    database = _open_database()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            for artifact, resolved, size, digest in resolved_artifacts:
                repository.add_artifact(
                    reservation.run_uuid,
                    role=artifact.role,
                    path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
                    path=resolved,
                    attempt_uuid=reservation.attempt_uuid,
                    mime=artifact.mime,
                    size_bytes=size,
                    sha256=digest,
                    media_metadata=artifact.media_metadata,
                )
            for text in text_sources:
                repository.add_text_source(
                    reservation.run_uuid,
                    kind=text.kind,
                    origin=PAID_TRANSCRIPTION_ORIGIN,
                    content=text.content,
                    content_hash=sha256_text(text.content),
                    language=text.language,
                    text_completeness=text.text_completeness,
                )
            advanced = repository.advance_run_revision(
                reservation.run_uuid,
                expected_revision=reservation.revision,
                status=RUN_STATUS_COMPLETED,
            )
            repository.set_paid_transcription_attempt_status(
                reservation.run_uuid,
                attempt_uuid=reservation.attempt_uuid,
                call_type=reservation.call_type,
                expected_status=ATTEMPT_STATUS_RAW_SAVED,
                status=ATTEMPT_STATUS_COMPLETED,
                usage=result_metadata,
            )
    except PaidTranscriptionError:
        raise
    except (HistoryDatabaseError, sqlite3.Error, OSError, ValueError) as exc:
        raise PaidTranscriptionError(_COMPLETE_FAILED, error_code="HISTORY_WRITE_FAILED") from exc
    finally:
        database.close()
    reservation.revision = advanced.revision


# -- read phases --------------------------------------------------------------


def _run_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise PaidTranscriptionError(
            "a paid transcription run requires its internal UUID",
            error_code="PAID_TRANSCRIPTION_NOT_FOUND",
        ) from None


def _paid_attempt(repository: HistoryRepository, run_uuid: str) -> Any:
    """Return the run's one paid-transcription attempt, or raise a fixed error."""
    attempts = [
        attempt
        for attempt in repository.get_attempts(run_uuid)
        if attempt.call_type in _OPERATION_BY_CALL_TYPE
    ]
    if len(attempts) != 1:
        raise PaidTranscriptionError(
            "the run does not carry exactly one paid transcription attempt",
            error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
        )
    return attempts[0]


def _raw_artifacts(repository: HistoryRepository, run_uuid: str) -> list[Any]:
    """Return every raw-response artifact row a run carries."""
    return [
        artifact
        for artifact in repository.get_artifacts(run_uuid)
        if artifact.role == ARTIFACT_ROLE_PROVIDER_RAW
    ]


def _matching_raw_artifact(
    repository: HistoryRepository,
    run_uuid: str,
    *,
    attempt_uuid: str,
    relative_path: str,
    size_bytes: int,
    sha256: str,
) -> Any:
    """Return the run's one raw-response artifact when it exactly matches.

    A run must carry exactly one raw-response row, and that row counts only when
    it names this attempt, is a managed relative path equal to the deterministic
    body path, and carries a mandatory size and digest equal to the validated
    body. Zero rows, a duplicate or conflicting row, and any identity mismatch all
    return ``None`` so the caller fails closed instead of replaying the first row.
    """
    raw_rows = _raw_artifacts(repository, run_uuid)
    if len(raw_rows) != 1:
        return None
    artifact = raw_rows[0]
    if (
        artifact.attempt_uuid == attempt_uuid
        and artifact.path_kind == PATH_KIND_MANAGED_RELATIVE
        and artifact.path == relative_path
        and artifact.size_bytes is not None
        and artifact.size_bytes == size_bytes
        and artifact.sha256 is not None
        and artifact.sha256 == sha256
    ):
        return artifact
    return None


def _open_read(run_uuid: str) -> tuple[sqlite3.Connection, HistoryRepository, Any]:
    """Open the paid run read-only in one snapshot, or raise a non-dispatching error.

    The reader never migrates, switches the journal mode, creates a sidecar, or
    creates the database or home. After the run row is identified it opens one
    explicit deferred read transaction, so the run, attempt, and artifact rows the
    caller reads next all come from a single consistent snapshot even when a live
    writer commits between those selects. A missing database or run raises a
    ``PAID_TRANSCRIPTION_NOT_FOUND`` error the caller may dispatch onward; a
    corrupt, foreign, newer, or otherwise unreadable database raises
    ``HISTORY_UNAVAILABLE``, which must not be silently reclassified. A run whose
    snapshot is not a paid origin raises ``PAID_TRANSCRIPTION_NOT_PAID``.
    """
    canonical = _run_uuid(run_uuid)
    try:
        database_path = history_database_path()
    except HistoryPathsError as exc:
        raise PaidTranscriptionError(_READ_FAILED, error_code="HISTORY_UNAVAILABLE") from exc
    if not database_path.exists():
        raise PaidTranscriptionError(
            "no paid transcription history run with that UUID",
            error_code="PAID_TRANSCRIPTION_NOT_FOUND",
        )
    try:
        connection = connect_readonly_consistent(database_path)
    except FileNotFoundError as exc:
        raise PaidTranscriptionError(
            "no paid transcription history run with that UUID",
            error_code="PAID_TRANSCRIPTION_NOT_FOUND",
        ) from exc
    except (HistoryDatabaseError, sqlite3.Error, OSError) as exc:
        raise PaidTranscriptionError(_READ_FAILED, error_code="HISTORY_UNAVAILABLE") from exc
    try:
        # One deferred read transaction brackets every select the caller makes, so
        # an interleaved writer commit cannot tear the run/attempt/artifact view.
        connection.execute("BEGIN")
        if readonly_schema_is_empty(connection):
            # A read-only-validated empty schema holds no runs table, so no paid run
            # can exist; dispatch it onward exactly like an absent database.
            raise PaidTranscriptionError(
                "no paid transcription history run with that UUID",
                error_code="PAID_TRANSCRIPTION_NOT_FOUND",
            )
        repository = HistoryRepository(cast(HistoryDatabase, _ReadOnlyConnection(connection)))
        run = repository.get_run(canonical)
        if run is None:
            raise PaidTranscriptionError(
                "no paid transcription history run with that UUID",
                error_code="PAID_TRANSCRIPTION_NOT_FOUND",
            )
        snapshot = run.config_snapshot if isinstance(run.config_snapshot, dict) else {}
        if snapshot.get("operation_origin") != PAID_TRANSCRIPTION_ORIGIN:
            raise PaidTranscriptionError(
                "the run is not a paid transcription run",
                error_code="PAID_TRANSCRIPTION_NOT_PAID",
            )
    except BaseException:
        connection.close()
        raise
    return connection, repository, run


def load_paid_transcription_state(run_uuid: str) -> PaidTranscriptionState:
    """Return the durable state of one paid transcription run, reading only rows."""
    connection, repository, run = _open_read(run_uuid)
    try:
        attempt = _paid_attempt(repository, run.run_uuid)
        raw_saved = attempt.status == ATTEMPT_STATUS_RAW_SAVED
        raw_missing = raw_saved and not _raw_artifacts(repository, run.run_uuid)
        snapshot = run.config_snapshot if isinstance(run.config_snapshot, dict) else {}
        return PaidTranscriptionState(
            run_uuid=run.run_uuid,
            run_root=run.run_root,
            revision=run.revision,
            status=run.status,
            attempt_uuid=attempt.attempt_uuid,
            attempt_status=attempt.status or "",
            raw_saved=raw_saved,
            raw_missing=raw_missing,
            operation=run.operation,
            call_type=attempt.call_type,
            provider=attempt.provider or "",
            model=attempt.model or "",
            snapshot=snapshot,
        )
    finally:
        connection.close()


def _require_canonical_output_root(run_root: str, snapshot: dict[str, Any]) -> Path:
    """Require the stored output locator to still name its committed DB root.

    A symlink inserted into any ancestor after reservation may make a saved raw
    body reachable at a new location without changing its bytes or database
    rows. Refuse such a relocation before replay, receipt recovery, or artifact
    publication; never silently follow the new location.
    """
    value = snapshot.get("output_root")
    if not isinstance(value, str) or not value or not run_root:
        raise PaidTranscriptionError(
            "the paid transcription run does not record its output root",
            error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
        )
    root = Path(value)
    if value != run_root or not root.is_absolute():
        raise PaidTranscriptionError(
            "the paid timing output directory no longer matches its committed root",
            error_code="PAID_OUTPUT_UNSAFE",
        )
    try:
        if root.resolve(strict=True) != root or not root.is_dir():
            raise PaidTranscriptionError(
                "the paid timing output directory no longer names its committed root",
                error_code="PAID_OUTPUT_UNSAFE",
            )
    except (OSError, RuntimeError) as exc:
        raise PaidTranscriptionError(
            "the paid timing output directory cannot be verified",
            error_code="PAID_OUTPUT_UNSAFE",
        ) from exc
    return root


def require_paid_output_root(state: PaidTranscriptionState) -> Path:
    """Validate a paid run's saved output root against the committed DB binding."""
    return _require_canonical_output_root(state.run_root, state.snapshot)


def reservation_from_state(state: PaidTranscriptionState) -> PaidTranscriptionReservation:
    """Rebuild the in-flight reservation a resume replays under.

    The rebuilt reservation carries the run's current revision, so the next phase
    still compare-and-swaps against the same run a concurrent writer would see, and
    the canonical output root the run owns.
    """
    output_root = require_paid_output_root(state)
    fingerprint = state.snapshot.get("operation_fingerprint")
    return PaidTranscriptionReservation(
        run_uuid=state.run_uuid,
        attempt_uuid=state.attempt_uuid,
        run_root=output_root,
        output_root=output_root,
        revision=state.revision,
        operation=state.operation,
        call_type=state.call_type,
        provider=state.provider,
        model=state.model,
        operation_fingerprint=fingerprint if isinstance(fingerprint, str) else "",
    )


def read_paid_transcription_raw(run_uuid: str) -> bytes:
    """Return the validated private response body of one paid attempt.

    The attempt must be ``raw_saved`` and exactly one managed artifact must name
    this attempt at its deterministic body path with a present size and digest
    equal to the bytes the file still holds. A missing or duplicated row, a
    different attempt, path kind, or path, a missing size or digest, or an
    oversized, symlinked, or tampered body all fail closed before the caller can
    parse or republish. At most the bound plus one byte is read, so an oversized
    body is refused before it is allocated.
    """
    connection, repository, run = _open_read(run_uuid)
    try:
        attempt = _paid_attempt(repository, run.run_uuid)
        if attempt.status != ATTEMPT_STATUS_RAW_SAVED:
            raise PaidTranscriptionError(
                "the paid transcription attempt does not carry a saved response",
                error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
            )
        snapshot = run.config_snapshot if isinstance(run.config_snapshot, dict) else {}
        output_root = _require_canonical_output_root(run.run_root, snapshot)
        relative_path = _raw_relative_path(attempt.attempt_uuid)
        body_path = output_root / relative_path
        try:
            if body_path.parent.is_symlink():
                raise PaidTranscriptionError(
                    "the saved response directory must not be a symlink",
                    error_code="PAID_OUTPUT_UNSAFE",
                )
            if body_path.is_symlink() or not body_path.is_file():
                raise PaidTranscriptionError(
                    "the saved response body is missing",
                    error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
                )
            if body_path.stat().st_size > _MAX_RAW_BODY_BYTES:
                raise PaidTranscriptionError(
                    "the saved response body exceeds the bounded maximum",
                    error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
                )
            with body_path.open("rb") as handle:
                data = handle.read(_MAX_RAW_BODY_BYTES + 1)
        except OSError as exc:
            raise PaidTranscriptionError(
                "the saved response body cannot be read",
                error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
            ) from exc
        if len(data) > _MAX_RAW_BODY_BYTES:
            raise PaidTranscriptionError(
                "the saved response body exceeds the bounded maximum",
                error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
            )
        artifact = _matching_raw_artifact(
            repository,
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            relative_path=relative_path,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        if artifact is None:
            raise PaidTranscriptionError(
                "the saved response body does not match exactly one stored artifact",
                error_code="PAID_TRANSCRIPTION_UNSUPPORTED",
            )
        return data
    finally:
        connection.close()


def recover_paid_transcription_after_crash(
    run_uuid: str,
) -> PaidTranscriptionReservation | None:
    """Reconcile a validated receipt+body pair the crash left unlinked.

    Called only by an explicit resume, under the shared run lock, for a run whose
    attempt is still ``submitting``. If this exact attempt's receipt and body are
    both present and match its stored fingerprint, size, and digest, the body is
    linked and the attempt is atomically moved ``submitting -> raw_saved``; the
    caller may then replay it with no POST. Anything else — a missing half, a
    foreign attempt, a changed digest, an oversized body — returns ``None``, so the
    caller keeps failing closed. The function never contacts a provider and never
    infers success from a filename.
    """
    state = load_paid_transcription_state(run_uuid)
    if state.attempt_status != ATTEMPT_STATUS_SUBMITTING:
        return None
    try:
        output_root = require_paid_output_root(state)
    except PaidTranscriptionError:
        return None
    fingerprint = state.snapshot.get("operation_fingerprint")
    verified = _verify_raw_evidence(
        output_root,
        state.run_uuid,
        state.attempt_uuid,
        fingerprint if isinstance(fingerprint, str) else None,
    )
    if verified is None:
        return None
    relative_path, size, digest, content_type = verified
    database = _open_database()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            present = _raw_artifacts(repository, state.run_uuid)
            existing = _matching_raw_artifact(
                repository,
                state.run_uuid,
                attempt_uuid=state.attempt_uuid,
                relative_path=relative_path,
                size_bytes=size,
                sha256=digest,
            )
            if present and existing is None:
                # A raw-role row already exists that is not this attempt's validated
                # pair (a different attempt, path, size, or digest, or a duplicate).
                # Replaying the first row would attribute the wrong body, so fail
                # closed.
                return None
            advanced = repository.advance_run_revision(
                state.run_uuid, expected_revision=state.revision
            )
            if existing is None:
                repository.add_artifact(
                    state.run_uuid,
                    role=ARTIFACT_ROLE_PROVIDER_RAW,
                    path_kind=PATH_KIND_MANAGED_RELATIVE,
                    path=relative_path,
                    attempt_uuid=state.attempt_uuid,
                    mime=content_type,
                    size_bytes=size,
                    sha256=digest,
                )
            repository.set_paid_transcription_attempt_status(
                state.run_uuid,
                attempt_uuid=state.attempt_uuid,
                call_type=state.call_type,
                expected_status=ATTEMPT_STATUS_SUBMITTING,
                status=ATTEMPT_STATUS_RAW_SAVED,
            )
    except (HistoryDatabaseError, sqlite3.Error, OSError, ValueError, PaidTranscriptionError):
        return None
    finally:
        database.close()
    return PaidTranscriptionReservation(
        run_uuid=state.run_uuid,
        attempt_uuid=state.attempt_uuid,
        run_root=output_root,
        output_root=output_root,
        revision=advanced.revision,
        operation=state.operation,
        call_type=state.call_type,
        provider=state.provider,
        model=state.model,
        operation_fingerprint=fingerprint if isinstance(fingerprint, str) else "",
    )


# -- safe publication ---------------------------------------------------------


def _require_safe_artifact_leaf(leaf: str) -> str:
    if not leaf or "/" in leaf or "\\" in leaf or leaf in (".", "..") or "\x00" in leaf:
        raise PaidTranscriptionError(
            "the paid timing artifact name is not a safe plain filename",
            error_code="PAID_OUTPUT_UNSAFE",
        )
    return leaf


def atomic_write_artifact(output_root: Path | str, leaf: str, data: bytes) -> Path:
    """Atomically publish one private-free artifact inside a validated output root.

    The output root must be a real directory (not a symlink), the leaf a plain
    filename, and the destination must not already be a symlink. The bytes are
    written to a unique temp file in the root, flushed and fsynced, then renamed
    over the destination, so a crash never leaves a half-written artifact and an
    attacker-planted symlink is never followed.
    """
    root = Path(output_root)
    try:
        if root.resolve(strict=True) != root or not root.is_dir():
            raise PaidTranscriptionError(
                "the paid timing output directory is missing or has a symlinked ancestor",
                error_code="PAID_OUTPUT_UNSAFE",
            )
    except (OSError, RuntimeError) as exc:
        raise PaidTranscriptionError(
            "the paid timing output directory cannot be verified",
            error_code="PAID_OUTPUT_UNSAFE",
        ) from exc
    safe_leaf = _require_safe_artifact_leaf(leaf)
    target = root / safe_leaf
    if target.is_symlink():
        raise PaidTranscriptionError(
            "the paid timing artifact destination must not be a symlink",
            error_code="PAID_OUTPUT_UNSAFE",
        )
    try:
        fd, name = tempfile.mkstemp(dir=root, prefix=f".{safe_leaf}.", suffix=".tmp")
    except OSError as exc:
        raise PaidTranscriptionError(
            "the paid timing artifact could not be written",
            error_code="PAID_OUTPUT_UNSAFE",
        ) from exc
    temp = Path(name)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError as exc:
            raise PaidTranscriptionError(
                "the paid timing artifact could not be written",
                error_code="PAID_OUTPUT_UNSAFE",
            ) from exc
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    _fsync_directory(root)
    return target
