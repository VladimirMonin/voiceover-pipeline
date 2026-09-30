"""Read, import, resume, and sync command handlers for the local SQLite history.

Plan section 6 stages S04 and S05 expose the canonical history through the CLI:
``history list`` and ``history show ID`` read run metadata, ``history import DIR
[--dry-run]`` previews or performs a safe offline import of old ``out/<run-id>``
trees, and ``history resume ID``/``history sync ID`` load one committed native TTS
run so the CLI can reconstruct and continue or inspect it. The handlers here stay
free of CLI printing and exit codes; :mod:`voiceover_pipeline.cli` owns the machine
envelope and human output.

Read guarantees:

* Nothing is created for an absent database. ``history list`` reports an empty
  history, ``history show`` reports a missing run, and ``history resume``/
  ``history sync`` report a missing run, all without creating the home directory or
  the database file.
* Reads go through the narrow :func:`voiceover_pipeline.history.database.connect_readonly`
  seam, so no migration, DDL, WAL switch, or sidecar write happens and a
  corrupt, foreign, or newer database fails closed instead of being queried.
* ``history costs`` reads the same way but through the WAL-consistent
  :func:`voiceover_pipeline.history.database.connect_readonly_consistent` seam, so
  an attempt a run in progress just committed (a live ``-wal``) is never missed;
  it still runs no migration, DDL, or sidecar write of its own.
* :func:`load_native_history_view` returns one verified read-only native
  prepared-TTS view for ``history resume ID``/``history sync ID``; a UUID that is
  not such a snapshot fails closed with a fixed, content-free ``NATIVE_HISTORY_UNSUPPORTED``
  reason instead of reconstructing a run whose identity was never captured.
* stdout metadata omits prepared/source text, transcripts, raw snapshots, signed
  URLs, and secrets. Text sources and parts report completeness as booleans; a
  run's ``config_snapshot`` is never emitted. Free-form stored values
  (remote IDs, provider/model/voice names, account aliases, artifact paths,
  legacy preview labels, cost currency/source/raw, part stage, attempt
  call/status, artifact role/hash, text-source kind/origin/language, and
  timestamps) are echoed only when they are bounded and free of a URL scheme,
  query, fragment, user-info, control character, or secret-shaped value, so a
  signed URL or Authorization header is never printed. A human label or
  filesystem path may keep its spaces; only an unsafe value is omitted as
  ``null``. Exact numeric cost fields (an amount or a legacy decimal lexeme) are
  ordinary metadata and still print; the fix is display-only and never rewrites
  the stored value.
  A UUID-shaped identifier that is not an internal run UUID falls back to an
  exact user-label lookup.
* A public error never forwards a stored database string: a
  ``HistoryDatabaseError`` subclass message can embed a tampered migration-ledger
  name, and a SQLite diagnostic can quote an unrecognized identifier, so every
  known failure type reports one fixed, content-free message and keeps its stable
  ``details.error_code`` and numeric exit code.
* A symlinked database file, or a database reached through a symlinked home
  directory, fails closed before any connect, so a read cannot follow a link
  outside the private home.

Import guarantees: the preview delegates to
:func:`voiceover_pipeline.history.legacy_import.preview_legacy_import`, which
writes nothing, and the real import calls
:func:`voiceover_pipeline.history.legacy_import.import_legacy_runs` once, which
is transactional, idempotent, and leaves the original legacy files untouched.
The dry-run projects an explicit, sanitized preview shape instead of serializing
the parser object wholesale.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from ..history.database import (
    HistoryDatabase,
    HistoryDatabaseError,
    HistoryDatabaseReadOnlyError,
    MigrationChecksumError,
    SchemaVersionTooNewError,
    connect_readonly,
    connect_readonly_consistent,
)
from ..history.legacy_import import (
    LegacyImportPreview,
    LegacyRunImportResult,
    LegacyRunPreview,
    import_legacy_runs,
    preview_legacy_import,
)
from ..history.native_view import (
    NativeTtsView,
    NativeViewError,
    NativeViewNotFoundError,
    NativeViewValidationError,
    load_native_tts_view,
)
from ..history.paths import (
    HistoryHomePermissionError,
    HistoryPathsError,
    history_database_path,
)
from ..history.repository import (
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    ArtifactRecord,
    AttemptRecord,
    Cost,
    HistoryRepository,
    PartRecord,
    RunRecord,
    TextSourceRecord,
    _is_sensitive_string,
)

# Numeric exit codes are duplicated from ``cli.py`` because the CLI imports this
# module; only the documented stable codes are used here.
_EXIT_ARGS = 2
_EXIT_PROVIDER = 30
_EXIT_OUTPUT = 50

# A database exception message is not a safe public string: a
# ``MigrationChecksumError`` quotes the stored ledger name and a SQLite
# diagnostic can quote an unrecognized identifier. Every known failure type
# therefore maps to one fixed, content-free message while keeping its stable
# ``details.error_code`` and numeric exit code.
_DATABASE_TOO_NEW_MESSAGE = "History database was written by a newer schema version."
_DATABASE_CHECKSUM_MESSAGE = "History database migration ledger does not match this binary."
_DATABASE_UNREADABLE_MESSAGE = "History database could not be read."

# A label that resolves to more than one run is ambiguous; never pick the first.
_MAX_LABEL_CANDIDATES = 20
# Fetch at most this many label matches before deciding ambiguity, so a hostile
# label cannot make the lookup unbounded.
_LABEL_LOOKUP_LIMIT = MAX_QUERY_LIMIT

# Free-form database values (remote IDs, provider/model/voice names, account
# aliases, artifact paths) can be an arbitrary string, including a temporary
# signed URL. Only a bounded plain token is echoed: a value carrying a URL
# scheme, query, fragment, user-info, or whitespace is reported as absent rather
# than leaking a temporary link, Authorization header, or secret to stdout.
_MAX_PUBLIC_METADATA_LENGTH = 200
_UNSAFE_PUBLIC_METADATA_MARKERS = ("://", "?", "#", "@")
# A human label (``production run``), operation, status, or filesystem path may
# legitimately contain spaces, so it is not held to the whitespace-free token
# rule. It is still rejected when it is non-printable, over the bound, or carries
# a URL scheme, query, fragment, user-info, or secret-shaped value, so a stored
# signed URL cannot reach stdout as a label or root path.
_UNSAFE_PUBLIC_TEXT_MARKERS = ("://", "?", "#", "@")
# A provider remote ID is an opaque token, never a URL or a path.
_OPAQUE_REMOTE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

# Local providers run inference on this machine, so their attempts can never carry
# an external API charge. This is the documented local allowlist, drawn from the
# provider registry and the CLI provider choices: TTS ``qwen-local`` and
# ``omnivoice-local``; ASR ``qwen-local``, ``nemotron-local``, and
# ``faster-whisper``. Any other value -- including an unrecognized provider --
# stays a potentially paid attempt, so an unknown provider is never assumed free.
_LOCAL_PROVIDER_IDS = frozenset(
    {"qwen-local", "omnivoice-local", "nemotron-local", "faster-whisper"}
)


class HistoryCommandError(RuntimeError):
    """A history command failed with a stable numeric code and error code.

    ``code`` is the CLI exit code (:data:`_EXIT_ARGS`, :data:`_EXIT_PROVIDER`, or
    :data:`_EXIT_OUTPUT`) and ``details`` is always an object that carries the
    stable string ``error_code`` plus any bounded, content-free context.
    """

    def __init__(self, message: str, code: int, *, error_code: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code
        self.details: dict[str, Any] = {"error_code": error_code, **extra}


class _ReadOnlyConnection:
    """Minimal handle exposing ``.connection`` to :class:`HistoryRepository`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


@dataclass(frozen=True)
class _OpenHistory:
    """A validated read-only history database, or ``None`` when unusable/absent."""

    path: Path
    connection: sqlite3.Connection | None
    repository: HistoryRepository | None
    schema_version: int | None


def _resolve_database_path(database_path: Path | str | None) -> Path:
    if database_path is not None:
        return Path(database_path).expanduser().absolute()
    try:
        return history_database_path().absolute()
    except HistoryPathsError as exc:
        raise HistoryCommandError(str(exc), _EXIT_ARGS, error_code="HISTORY_HOME_INVALID") from exc


def _open_history(database_path: Path, *, consistent: bool = False) -> _OpenHistory:
    """Open the database read-only, or report it absent/empty without writing.

    ``consistent`` selects the WAL-consistent reader for ``history costs``: it
    accepts a live ``-wal``/``-shm`` pair written by a run in progress, because an
    aggregate must never miss a just-committed attempt or create a sidecar of its
    own. The default reader pins ``immutable=1`` and refuses any adjacent
    sidecar, so ``list``/``show`` keep their post-run semantics.
    """
    if database_path.is_symlink() or database_path.parent.is_symlink():
        raise HistoryCommandError(
            f"History database must not be a symlink: {database_path}",
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        )
    if not database_path.exists():
        return _OpenHistory(database_path, None, None, None)
    opener = connect_readonly_consistent if consistent else connect_readonly
    try:
        connection = opener(database_path)
    except FileNotFoundError:
        return _OpenHistory(database_path, None, None, None)
    except SchemaVersionTooNewError as exc:
        raise HistoryCommandError(
            _DATABASE_TOO_NEW_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_TOO_NEW"
        ) from exc
    except MigrationChecksumError as exc:
        raise HistoryCommandError(
            _DATABASE_CHECKSUM_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_CHECKSUM_MISMATCH",
        ) from exc
    except HistoryDatabaseReadOnlyError as exc:
        raise HistoryCommandError(
            _DATABASE_UNREADABLE_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        ) from exc
    except HistoryDatabaseError as exc:
        raise HistoryCommandError(
            _DATABASE_UNREADABLE_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        ) from exc
    except sqlite3.Error as exc:
        raise HistoryCommandError(
            _DATABASE_UNREADABLE_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        ) from exc

    has_runs = (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        is not None
    )
    if not has_runs:
        connection.close()
        return _OpenHistory(database_path, None, None, None)
    schema_version = int(
        connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()["version"]
    )
    # ``HistoryRepository`` only reads ``database.connection``; a duck-typed
    # read-only handle avoids exposing the writable ``HistoryDatabase`` surface.
    repository = HistoryRepository(cast(HistoryDatabase, _ReadOnlyConnection(connection)))
    return _OpenHistory(database_path, connection, repository, schema_version)


def _database_metadata(open_history: _OpenHistory, *, exists: bool) -> dict[str, Any]:
    return {
        "path": str(open_history.path),
        "exists": exists,
        "schema_version": open_history.schema_version,
    }


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise HistoryCommandError(
            f"--limit must be an integer between 1 and {MAX_QUERY_LIMIT}.",
            _EXIT_ARGS,
            error_code="HISTORY_INVALID_LIMIT",
        )


def _validate_offset(offset: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise HistoryCommandError(
            "--offset must be a non-negative integer.",
            _EXIT_ARGS,
            error_code="HISTORY_INVALID_OFFSET",
        )


# -- metadata projections (content-free) --------------------------------------


def _safe_public_value(value: object) -> str | None:
    """Return a bounded plain metadata token, or ``None`` when it may leak.

    A value that is not a bounded, whitespace-free token without a URL scheme,
    query, fragment, or user-info is dropped instead of being echoed. The same
    fail-closed secret detection used for stored config snapshots
    (:func:`voiceover_pipeline.history.repository._is_sensitive_string`) rejects
    a value that looks like an API key, bearer token, or signed-URL parameter, so
    a stored secret cannot reach stdout.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > _MAX_PUBLIC_METADATA_LENGTH:
        return None
    if any(marker in candidate for marker in _UNSAFE_PUBLIC_METADATA_MARKERS):
        return None
    if any(character.isspace() for character in candidate):
        return None
    if _is_sensitive_string(candidate):
        return None
    return candidate


def _safe_remote_id(value: object) -> str | None:
    """Return a bounded opaque remote ID, or ``None`` when it is not one."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if _OPAQUE_REMOTE_ID_PATTERN.fullmatch(candidate) is None:
        return None
    if _is_sensitive_string(candidate):
        return None
    return candidate


def _safe_public_text(value: object) -> str | None:
    """Return a bounded free-form label/operation/status/path, or ``None``.

    Unlike :func:`_safe_public_value`, spaces are preserved so a normal human
    label such as ``production run`` or a path under a directory with spaces
    still prints. A value that is non-printable, over the bound, or carries a URL
    scheme, query, fragment, or user-info is dropped, and the stored-secret
    policy :func:`voiceover_pipeline.history.repository._is_sensitive_string`
    rejects a key- or signed-URL-shaped value, so a stored signed URL is omitted
    rather than echoed.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > _MAX_PUBLIC_METADATA_LENGTH:
        return None
    if any(marker in candidate for marker in _UNSAFE_PUBLIC_TEXT_MARKERS):
        return None
    if any(not character.isprintable() for character in candidate):
        return None
    if _is_sensitive_string(candidate):
        return None
    return candidate


def _run_metadata(run: RunRecord) -> dict[str, Any]:
    return {
        "run_uuid": run.run_uuid,
        "operation": _safe_public_text(run.operation),
        "user_label": _safe_public_text(run.user_label),
        "parent_uuid": run.parent_uuid,
        "status": _safe_public_text(run.status),
        "run_root": _safe_public_text(run.run_root),
        "legacy_source_root": _safe_public_text(run.legacy_source_root),
        "record_version": run.record_version,
        "created_at": _safe_public_value(run.created_at),
        "updated_at": _safe_public_value(run.updated_at),
        "legacy_import": run.legacy_source_root is not None,
    }


def _part_metadata(part: PartRecord) -> dict[str, Any]:
    return {
        "part_uuid": part.part_uuid,
        "position": part.position,
        "voice": _safe_public_value(part.voice),
        "fingerprint": _safe_public_value(part.fingerprint),
        "stage": _safe_public_value(part.stage),
        "has_prepared_text": part.prepared_text is not None,
        "has_shared_vibe": part.vibe_shared is not None,
        "has_specific_vibe": part.vibe_specific is not None,
        "has_effective_vibe": part.vibe_effective is not None,
    }


def _cost_metadata(cost: Cost) -> dict[str, Any]:
    # An exact amount (or a legacy decimal lexeme) is ordinary metadata and still
    # prints; the free-form currency/source/raw provenance is bounded the same way
    # as every other stored string.
    return {
        "amount": _safe_public_value(cost.amount),
        "currency": _safe_public_value(cost.currency),
        "source": _safe_public_value(cost.source),
        "exact_available": cost.exact_available,
        "raw": _safe_public_value(cost.raw),
    }


def _attempt_metadata(attempt: AttemptRecord) -> dict[str, Any]:
    return {
        "attempt_uuid": attempt.attempt_uuid,
        "part_uuid": attempt.part_uuid,
        "call_type": _safe_public_value(attempt.call_type),
        "provider": _safe_public_value(attempt.provider),
        "model": _safe_public_value(attempt.model),
        "account_alias": _safe_public_value(attempt.account_alias),
        "remote_id": _safe_remote_id(attempt.remote_id),
        "status": _safe_public_value(attempt.status),
        "cost": _cost_metadata(attempt.cost),
        "has_usage": attempt.usage is not None,
        # A stored provider message is not echoed by default: it is unredacted
        # free text that could contain a secret. Only its presence is reported.
        "has_error": attempt.error is not None,
        "created_at": _safe_public_value(attempt.created_at),
        "updated_at": _safe_public_value(attempt.updated_at),
    }


def _artifact_metadata(artifact: ArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_uuid": artifact.artifact_uuid,
        "part_uuid": artifact.part_uuid,
        "attempt_uuid": artifact.attempt_uuid,
        "role": _safe_public_value(artifact.role),
        "path_kind": _safe_public_value(artifact.path_kind),
        "path": _safe_public_text(artifact.path),
        "mime": _safe_public_value(artifact.mime),
        "size_bytes": artifact.size_bytes,
        "sha256": _safe_public_value(artifact.sha256),
        "availability": _safe_public_value(artifact.availability),
        "has_media_metadata": artifact.media_metadata is not None,
    }


def _text_source_metadata(text_source: TextSourceRecord) -> dict[str, Any]:
    return {
        "text_source_uuid": text_source.text_source_uuid,
        "part_uuid": text_source.part_uuid,
        "artifact_uuid": text_source.artifact_uuid,
        "kind": _safe_public_value(text_source.kind),
        "origin": _safe_public_value(text_source.origin),
        "content_hash": _safe_public_value(text_source.content_hash),
        "language": _safe_public_value(text_source.language),
        "text_completeness": _safe_public_value(text_source.text_completeness),
        "has_content": text_source.content is not None,
    }


def _run_detail(repository: HistoryRepository, run: RunRecord) -> dict[str, Any]:
    return {
        "run": _run_metadata(run),
        "parts": [_part_metadata(part) for part in repository.get_parts(run.run_uuid)],
        "attempts": [
            _attempt_metadata(attempt) for attempt in repository.get_attempts(run.run_uuid)
        ],
        "artifacts": [
            _artifact_metadata(artifact) for artifact in repository.get_artifacts(run.run_uuid)
        ],
        "text_sources": [
            _text_source_metadata(text_source)
            for text_source in repository.get_text_sources(run.run_uuid)
        ],
    }


def _candidate_metadata(run: RunRecord) -> dict[str, Any]:
    return {
        "run_uuid": run.run_uuid,
        "user_label": _safe_public_text(run.user_label),
        "operation": _safe_public_text(run.operation),
        "status": _safe_public_text(run.status),
        "run_root": _safe_public_text(run.run_root),
        "created_at": _safe_public_value(run.created_at),
    }


def _run_not_found_error(identifier: str) -> HistoryCommandError:
    """Report a missing run without echoing a URL-shaped identifier."""
    safe_identifier = _safe_public_text(identifier)
    if safe_identifier is None:
        message = "No history run matches the given identifier."
    else:
        message = f"No history run matches {safe_identifier!r}."
    return HistoryCommandError(
        message,
        _EXIT_ARGS,
        error_code="HISTORY_RUN_NOT_FOUND",
        id=safe_identifier,
    )


def _ambiguous_label_error(identifier: str, candidates: list[RunRecord]) -> HistoryCommandError:
    """Report bounded candidates for an ambiguous label without echoing a URL."""
    safe_identifier = _safe_public_text(identifier)
    label = safe_identifier if safe_identifier is not None else "(redacted label)"
    return HistoryCommandError(
        f"Label {label!r} matches {len(candidates)} runs; show one by UUID.",
        _EXIT_ARGS,
        error_code="HISTORY_LABEL_AMBIGUOUS",
        id=safe_identifier,
        candidate_count=len(candidates),
        candidates=[
            _candidate_metadata(candidate) for candidate in candidates[:_MAX_LABEL_CANDIDATES]
        ],
    )


# -- list ---------------------------------------------------------------------


def list_history(
    *,
    label: str | None = None,
    operation: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    offset: int = 0,
    database_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return bounded run metadata, newest first, filtered by exact metadata.

    An absent or empty database yields an empty list with the resolved database
    path and ``exists`` flag; nothing is created.
    """
    _validate_limit(limit)
    _validate_offset(offset)
    resolved = _resolve_database_path(database_path)
    open_history = _open_history(resolved)
    exists = open_history.connection is not None or resolved.exists()
    try:
        if open_history.repository is None:
            runs: list[dict[str, Any]] = []
        else:
            runs = [
                _run_metadata(run)
                for run in open_history.repository.list_runs(
                    user_label=label,
                    operation=operation,
                    status=status,
                    limit=limit,
                    offset=offset,
                )
            ]
    finally:
        if open_history.connection is not None:
            open_history.connection.close()
    return {
        "dry_run": False,
        "database": _database_metadata(open_history, exists=exists),
        "filters": {
            "label": _safe_public_text(label),
            "operation": _safe_public_text(operation),
            "status": _safe_public_text(status),
        },
        "limit": limit,
        "offset": offset,
        "count": len(runs),
        "runs": runs,
    }


# -- show ---------------------------------------------------------------------


def show_history(
    identifier: str,
    *,
    database_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return one run's content-free metadata resolved by UUID or user label.

    A UUID-shaped identifier is looked up by exact UUID first; if no internal run
    carries it, the same string is looked up as an exact user label, so a legacy
    run whose ``run_id`` happens to be UUID-shaped still resolves. A label that
    matches runs in more than one root fails with ``HISTORY_LABEL_AMBIGUOUS`` and
    bounded candidates instead of choosing the first match.
    """
    resolved = _resolve_database_path(database_path)
    open_history = _open_history(resolved)
    try:
        if open_history.repository is None:
            raise _run_not_found_error(identifier)
        repository = open_history.repository
        try:
            canonical_uuid = str(uuid.UUID(identifier))
        except (ValueError, AttributeError, TypeError):
            canonical_uuid = None
        if canonical_uuid is not None:
            run = repository.get_run(canonical_uuid)
            if run is not None:
                return {
                    "dry_run": False,
                    "database": _database_metadata(open_history, exists=True),
                    **_run_detail(repository, run),
                }
        # A UUID-shaped identifier that is not an internal run UUID can still be
        # an exact legacy user label, so fall back to label resolution.
        candidates = repository.find_runs_by_label(identifier, limit=_LABEL_LOOKUP_LIMIT)
        if not candidates:
            raise _run_not_found_error(identifier)
        if len(candidates) > 1:
            raise _ambiguous_label_error(identifier, candidates)
        return {
            "dry_run": False,
            "database": _database_metadata(open_history, exists=True),
            **_run_detail(repository, candidates[0]),
        }
    finally:
        if open_history.connection is not None:
            open_history.connection.close()


# -- costs --------------------------------------------------------------------

# A stored cost the aggregate cannot read as an exact decimal or a stored
# currency it cannot render safely fails closed with one fixed message, so no
# malformed value is echoed and no inexact value is silently rounded or dropped.
_COST_UNREADABLE_MESSAGE = "A stored cost could not be read as an exact decimal."


@dataclass
class _CostBucket:
    """Accumulated money observations for one currency or ``(operation, currency)``.

    ``known_total`` is a :class:`~decimal.Decimal` sum of the stored decimal
    texts, so ``0.1 + 0.2`` stays ``0.3`` and a binary float never enters the
    money path. It stays ``None`` until a known amount arrives, so a group with
    only unknown attempts reports ``known_amount: null`` instead of an invented
    zero. ``exact_attempts`` and ``non_exact_attempts`` split the known attempts
    by the stored ``exact_available`` guarantee; a non-exact legacy lexeme still
    contributes to ``known_amount`` but is never counted as exact.
    """

    known_total: Decimal | None = None
    known_attempts: int = 0
    exact_attempts: int = 0
    non_exact_attempts: int = 0
    unknown_attempts: int = 0

    def observe_known(self, amount: Decimal, *, exact: bool) -> None:
        self.known_total = amount if self.known_total is None else self.known_total + amount
        self.known_attempts += 1
        if exact:
            self.exact_attempts += 1
        else:
            self.non_exact_attempts += 1

    def observe_unknown(self) -> None:
        self.unknown_attempts += 1

    def project(self) -> dict[str, Any]:
        return {
            "known_amount": None if self.known_total is None else str(self.known_total),
            "known_attempts": self.known_attempts,
            "exact_attempts": self.exact_attempts,
            "non_exact_attempts": self.non_exact_attempts,
            "unknown_attempts": self.unknown_attempts,
        }


def _cost_decimal(amount: str) -> Decimal:
    """Parse one stored cost text exactly, or fail closed without echoing it."""
    try:
        value = Decimal(amount)
    except (InvalidOperation, ValueError) as exc:
        raise HistoryCommandError(
            _COST_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_COST_UNREADABLE"
        ) from exc
    if not value.is_finite():
        raise HistoryCommandError(
            _COST_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_COST_UNREADABLE"
        )
    return value


def _safe_cost_currency(currency: str | None) -> str | None:
    """Return the safe currency label, or ``None`` for a missing currency.

    A stored currency that cannot be safely rendered (a URL, whitespace, or a
    secret-shaped value) is neither echoed nor silently folded into a real
    currency: the read fails closed instead. A genuinely missing currency
    (``None``) stays its own unknown-currency bucket, never an implicit ``RUB``.
    """
    if currency is None:
        return None
    safe = _safe_public_value(currency)
    if safe is None:
        raise HistoryCommandError(
            "A stored cost currency could not be read safely.",
            _EXIT_PROVIDER,
            error_code="HISTORY_COST_UNREADABLE",
        )
    return safe


def _sorted_currencies(buckets: dict[str | None, _CostBucket]) -> list[str | None]:
    """Order currency buckets deterministically with the unknown bucket last."""
    return sorted(buckets, key=lambda currency: (currency is None, currency or ""))


def _sorted_operation_keys(
    buckets: dict[tuple[str | None, str | None], _CostBucket],
) -> list[tuple[str | None, str | None]]:
    """Order ``(operation, currency)`` buckets deterministically."""
    return sorted(
        buckets,
        key=lambda key: (key[0] is None, key[0] or "", key[1] is None, key[1] or ""),
    )


def costs_history(*, database_path: Path | str | None = None) -> dict[str, Any]:
    """Return read-only money totals grouped by currency and operation.

    Every attempt row is one unique financial attempt: summed exactly once, never
    parent plus child, and a repeated status observation stays on the same row.
    The stored decimal text is accumulated with :class:`~decimal.Decimal`, so an
    exact total is reported and a binary float never enters the money path. A
    ``NULL`` amount is an unknown attempt and stays distinct from a real ``"0"``.

    ``known_amount`` is the sum of the stored non-NULL observations, including a
    non-exact legacy lexeme, and does not claim those were exact; ``exact_attempts``
    and ``non_exact_attempts`` expose the split and ``completeness`` is ``partial``
    when any cloud attempt is unknown or any known amount is not exact. A local
    provider attempt (the documented local allowlist) with no amount is counted in
    ``local_attempts_without_api_charge``, not as a cloud unknown. The read uses
    the WAL-consistent reader, so a run in progress is still visible, and it
    creates nothing for an absent database.
    """
    resolved = _resolve_database_path(database_path)
    open_history = _open_history(resolved, consistent=True)
    exists = open_history.connection is not None or resolved.exists()
    totals: dict[str | None, _CostBucket] = {}
    operations: dict[tuple[str | None, str | None], _CostBucket] = {}
    attempts = 0
    local_attempts_without_api_charge = 0
    try:
        if open_history.repository is not None:
            for row in open_history.repository.iter_attempt_costs():
                attempts += 1
                operation = _safe_public_text(row.operation)
                if row.amount is None:
                    provider = _safe_public_value(row.provider)
                    if provider is not None and provider in _LOCAL_PROVIDER_IDS:
                        local_attempts_without_api_charge += 1
                        continue
                    currency = _safe_cost_currency(row.currency)
                    operations.setdefault((operation, currency), _CostBucket()).observe_unknown()
                    totals.setdefault(currency, _CostBucket()).observe_unknown()
                    continue
                amount = _cost_decimal(row.amount)
                currency = _safe_cost_currency(row.currency)
                exact = row.exact_available
                operations.setdefault((operation, currency), _CostBucket()).observe_known(
                    amount, exact=exact
                )
                totals.setdefault(currency, _CostBucket()).observe_known(amount, exact=exact)
    finally:
        if open_history.connection is not None:
            open_history.connection.close()
    unknown_attempts = sum(bucket.unknown_attempts for bucket in totals.values())
    non_exact_attempts = sum(bucket.non_exact_attempts for bucket in totals.values())
    operation_rows: list[dict[str, Any]] = []
    for operation, currency in _sorted_operation_keys(operations):
        bucket = operations[(operation, currency)]
        operation_rows.append({"operation": operation, "currency": currency, **bucket.project()})
    return {
        "dry_run": False,
        "database": _database_metadata(open_history, exists=exists),
        "attempts": attempts,
        "totals": [
            {"currency": currency, **totals[currency].project()}
            for currency in _sorted_currencies(totals)
        ],
        "operations": operation_rows,
        "completeness": "partial" if unknown_attempts or non_exact_attempts else "complete",
        "local_attempts_without_api_charge": local_attempts_without_api_charge,
    }


# -- reconstructable native view ----------------------------------------------


def load_native_history_view(
    identifier: str, *, database_path: Path | str | None = None
) -> NativeTtsView:
    """Load one verified native TTS view for ``history resume``/``history sync``.

    ``identifier`` must be the internal run UUID; a label is not resolved here,
    because resume and sync mutate one exact run rather than reporting metadata.
    The run is read through the same read-only seam as ``history show``, so an
    absent database reports ``HISTORY_RUN_NOT_FOUND`` and an unreadable, too-new,
    or foreign database fails closed without creating anything. A UUID that is not
    a native prepared-TTS snapshot -- a legacy-imported run, another operation, an
    unsupported route, or a tampered snapshot -- is refused with a fixed message
    that echoes no stored value, so the caller never reconstructs a run whose
    identity was never captured. Reading never creates the home directory, the
    database, or a run row.
    """
    try:
        canonical_uuid = str(uuid.UUID(identifier))
    except (ValueError, AttributeError, TypeError):
        raise HistoryCommandError(
            "history resume and history sync require the internal run UUID.",
            _EXIT_ARGS,
            error_code="HISTORY_INVALID_RUN_ID",
        ) from None
    resolved = _resolve_database_path(database_path)
    open_history = _open_history(resolved)
    try:
        if open_history.repository is None:
            raise _run_not_found_error(identifier)
        try:
            return load_native_tts_view(open_history.repository, canonical_uuid)
        except NativeViewNotFoundError:
            raise _run_not_found_error(identifier) from None
        except NativeViewValidationError:
            raise HistoryCommandError(
                "history resume and history sync require the internal run UUID.",
                _EXIT_ARGS,
                error_code="HISTORY_INVALID_RUN_ID",
            ) from None
        except NativeViewError:
            raise HistoryCommandError(
                "This history run is not a reconstructable native TTS snapshot; only an admitted "
                "openrouter-tts dialogue or polza-tts/openrouter-tts native run can be resumed "
                "or synced.",
                _EXIT_PROVIDER,
                error_code="NATIVE_HISTORY_UNSUPPORTED",
            ) from None
    finally:
        if open_history.connection is not None:
            open_history.connection.close()


# -- import -------------------------------------------------------------------


def _require_source_directory(source: Path | str) -> Path:
    source_path = Path(source).expanduser()
    if not source_path.is_dir():
        raise HistoryCommandError(
            f"Source directory not found: {source_path}",
            _EXIT_ARGS,
            error_code="HISTORY_SOURCE_NOT_FOUND",
        )
    return source_path


def _run_preview_metadata(preview: LegacyRunPreview) -> dict[str, Any]:
    """Project a bounded, content-free preview of one legacy run.

    Fields are projected explicitly rather than serialized wholesale, and the
    free-form provider/model/voice/label/currency values pass through
    :func:`_safe_public_value`, so a signed URL stored in a legacy JSON string
    cannot reach the dry-run output. Counts, cost provenance, and conflict codes
    are preserved.
    """
    return {
        "source_root": _safe_public_text(preview.source_root),
        "run_id": _safe_public_value(preview.run_id),
        "operation": _safe_public_text(preview.operation),
        "status": _safe_public_text(preview.status),
        "provider": _safe_public_value(preview.provider),
        "model": _safe_public_value(preview.model),
        "voice": _safe_public_value(preview.voice),
        "chunk_count": preview.chunk_count,
        "chunks_with_text": preview.chunks_with_text,
        "chunks_missing_text": preview.chunks_missing_text,
        "chunks_with_audio": preview.chunks_with_audio,
        "chunks_missing_audio": preview.chunks_missing_audio,
        "chunks_with_cost": preview.chunks_with_cost,
        "cost_total": _safe_public_value(preview.cost_total),
        "cost_total_exact": _safe_public_value(preview.cost_total_exact),
        "cost_currency": _safe_public_value(preview.cost_currency),
        "script_text_available": preview.script_text_available,
        "already_imported": preview.already_imported,
        "importable": preview.importable,
        "conflicts": list(preview.conflicts),
    }


def preview_history_import(
    source: Path | str, *, database_path: Path | str | None = None
) -> dict[str, Any]:
    """Report what an import would do, writing nothing to disk or database."""
    source_path = _require_source_directory(source)
    try:
        preview: LegacyImportPreview = preview_legacy_import(
            source_path, database_path=database_path
        )
    except HistoryPathsError as exc:
        raise HistoryCommandError(str(exc), _EXIT_ARGS, error_code="HISTORY_HOME_INVALID") from exc
    return {
        "dry_run": True,
        "source": preview.source,
        "database": {
            "path": preview.database_path,
            "exists": preview.database_exists,
            "readable": preview.database_readable,
        },
        "discovered_count": preview.discovered_count,
        "importable_count": preview.importable_count,
        "already_imported_count": preview.already_imported_count,
        "missing_text_count": preview.missing_text_count,
        "missing_audio_count": preview.missing_audio_count,
        "scan_conflicts": list(preview.scan_conflicts),
        "runs": [_run_preview_metadata(run) for run in preview.runs],
    }


def _import_result_metadata(run: LegacyRunImportResult) -> dict[str, Any]:
    """Project one imported run's bounded result without echoing a URL path."""
    return {
        "source_root": _safe_public_text(run.source_root),
        "run_uuid": run.run_uuid,
        "created": run.created,
        "parts": run.parts,
        "attempts": run.attempts,
        "artifacts": run.artifacts,
        "text_sources": run.text_sources,
        "conflicts": list(run.conflicts),
    }


def run_history_import(
    source: Path | str, *, database_path: Path | str | None = None
) -> dict[str, Any]:
    """Import every importable legacy run under ``source`` exactly once."""
    source_path = _require_source_directory(source)
    try:
        result = import_legacy_runs(source_path, database_path=database_path)
    except SchemaVersionTooNewError as exc:
        raise HistoryCommandError(
            _DATABASE_TOO_NEW_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_TOO_NEW"
        ) from exc
    except MigrationChecksumError as exc:
        raise HistoryCommandError(
            _DATABASE_CHECKSUM_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_CHECKSUM_MISMATCH",
        ) from exc
    except HistoryHomePermissionError as exc:
        raise HistoryCommandError(
            str(exc), _EXIT_OUTPUT, error_code="HISTORY_HOME_PERMISSION"
        ) from exc
    except HistoryPathsError as exc:
        raise HistoryCommandError(str(exc), _EXIT_ARGS, error_code="HISTORY_HOME_INVALID") from exc
    except HistoryDatabaseError as exc:
        raise HistoryCommandError(
            _DATABASE_UNREADABLE_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        ) from exc
    except OSError as exc:
        # A filesystem error can quote a path or an untrusted file name from the
        # imported tree, so only a bounded, sanitized detail is echoed.
        detail = _safe_public_text(str(exc))
        message = "History import failed." if detail is None else f"History import failed: {detail}"
        raise HistoryCommandError(message, _EXIT_OUTPUT, error_code="HISTORY_WRITE_ERROR") from exc
    return {
        "dry_run": False,
        "source": result.source,
        "database": {"path": result.database_path},
        "imported_count": result.imported_count,
        "skipped_count": result.skipped_count,
        "rejected_count": result.rejected_count,
        "runs": [_import_result_metadata(run) for run in result.runs],
    }
