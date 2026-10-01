"""Offline ``search`` and ``index`` command handlers for saved history text.

Plan section 8 stage S08 exposes the derived lexical layer through the CLI:

* ``voiceover search QUERY --mode lexical`` runs one offline BM25 query over the
  saved scripts, transcripts, run labels, and (only with ``--scope directions``)
  director instructions. It needs no key, FFmpeg, Torch, or ``sqlite-vec``; a
  semantic or hybrid mode is refused with a clear deferred message rather than
  silently returning nothing.
* ``voiceover index status|build|rebuild`` inspects or maintains the derived
  index. ``status`` opens the database read-only and writes no sidecar;
  ``build``/``rebuild`` open it for writing, create a missing private home, and
  rebuild only from SQLite -- never by scanning user files.

The handlers here stay free of CLI printing and exit codes;
:mod:`voiceover_pipeline.cli` owns the machine envelope and human output. A search
on an absent database reports an empty result without creating anything, and a
database whose lexical layer is not built reports the same plus one instruction to
run ``voiceover index build``.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..history.database import (
    HistoryDatabase,
    HistoryDatabaseError,
    HistoryDatabaseReadOnlyError,
    MigrationChecksumError,
    SchemaVersionTooNewError,
    connect_readonly_consistent,
)
from ..history.paths import (
    HistoryHomePermissionError,
    HistoryPathsError,
    ensure_history_home,
    history_database_path,
)
from ..history.repository import (
    MAX_QUERY_LIMIT,
    TEXT_KIND_ASR_CONTEXT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
)
from ..search import indexing
from ..search.lexical import (
    DEFAULT_SCOPE,
    ROLES,
    SCOPES,
    query_tokens,
    roles_for_scope,
    search_lexical,
)

# Numeric exit codes are duplicated from ``cli.py`` because the CLI imports this
# module; only the documented stable codes are used here.
_EXIT_ARGS = 2
_EXIT_PROVIDER = 30
_EXIT_OUTPUT = 50

DEFAULT_SEARCH_LIMIT = 20

# Known text-source kinds accepted by ``--kind``; a typo fails loudly instead of
# silently returning nothing. ``asr_context`` is a real stored kind but is not
# indexed, so filtering by it is allowed and yields an empty result.
_SEARCH_KINDS = (
    TEXT_KIND_TTS_SCRIPT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_ASR_CONTEXT,
)

# A database exception message is not a safe public string, so each known failure
# type maps to one fixed, content-free message with its stable error code.
_DATABASE_TOO_NEW_MESSAGE = "History database was written by a newer schema version."
_DATABASE_CHECKSUM_MESSAGE = "History database migration ledger does not match this binary."
_DATABASE_UNREADABLE_MESSAGE = "History database could not be read."


class SearchCommandError(RuntimeError):
    """A search or index command failed with a stable code and error code."""

    def __init__(self, message: str, code: int, *, error_code: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code
        self.details: dict[str, Any] = {"error_code": error_code, **extra}


def _resolve_database_path(database_path: Path | str | None) -> Path:
    if database_path is not None:
        return Path(database_path).expanduser().absolute()
    try:
        return history_database_path().absolute()
    except HistoryPathsError as exc:
        raise SearchCommandError(str(exc), _EXIT_ARGS, error_code="HISTORY_HOME_INVALID") from exc


def _open_read_connection(database_path: Path) -> sqlite3.Connection | None:
    """Open the history database read-only, or ``None`` when absent.

    Uses the consistent read-only reader: a quiescent database is read immutably
    and creates no sidecar, a database a writer just committed to is read over its
    committed WAL frames, and an absent or empty database yields ``None`` rather
    than creating a file. A symlinked database fails closed before any connect.
    """
    if database_path.is_symlink() or database_path.parent.is_symlink():
        raise SearchCommandError(
            f"History database must not be a symlink: {database_path}",
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_UNREADABLE",
        )
    if not database_path.exists():
        return None
    try:
        connection = connect_readonly_consistent(database_path)
    except FileNotFoundError:
        return None
    except SchemaVersionTooNewError as exc:
        raise SearchCommandError(
            _DATABASE_TOO_NEW_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_TOO_NEW"
        ) from exc
    except MigrationChecksumError as exc:
        raise SearchCommandError(
            _DATABASE_CHECKSUM_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_CHECKSUM_MISMATCH",
        ) from exc
    except HistoryDatabaseReadOnlyError as exc:
        raise SearchCommandError(
            _DATABASE_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_UNREADABLE"
        ) from exc
    except HistoryDatabaseError as exc:
        raise SearchCommandError(
            _DATABASE_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_UNREADABLE"
        ) from exc
    except sqlite3.Error as exc:
        raise SearchCommandError(
            _DATABASE_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_UNREADABLE"
        ) from exc
    return connection


def _schema_version(connection: sqlite3.Connection) -> int | None:
    """Return the highest applied migration version, or ``None`` without a ledger."""
    has_ledger = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if has_ledger is None:
        return None
    return int(
        connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
    )


def _database_metadata(path: Path, connection: sqlite3.Connection | None) -> dict[str, Any]:
    exists = connection is not None or path.exists()
    schema_version = None if connection is None else _schema_version(connection)
    return {"path": str(path), "exists": exists, "schema_version": schema_version}


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise SearchCommandError(
            f"--limit must be an integer between 1 and {MAX_QUERY_LIMIT}.",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_LIMIT",
        )


def _validate_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise SearchCommandError(
            f"--scope must be one of {', '.join(SCOPES)}.",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_SCOPE",
        )


def _validate_role(role: str | None) -> None:
    if role is not None and role not in ROLES:
        raise SearchCommandError(
            f"--role must be one of {', '.join(ROLES)}.",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_ROLE",
        )


def _validate_kind(kind: str | None) -> None:
    if kind is not None and kind not in _SEARCH_KINDS:
        raise SearchCommandError(
            f"--kind must be one of {', '.join(_SEARCH_KINDS)}.",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_KIND",
        )


def _validate_run_uuid(run_uuid: str | None) -> str | None:
    if run_uuid is None:
        return None
    try:
        return str(uuid.UUID(run_uuid))
    except (ValueError, AttributeError, TypeError) as exc:
        raise SearchCommandError(
            "--run must be an internal run UUID.",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_RUN",
        ) from exc


def _validate_date(value: str | None, *, flag: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        if parsed.strftime("%Y-%m-%d") != value:
            raise ValueError("date is not zero-padded")
    except (ValueError, TypeError) as exc:
        raise SearchCommandError(
            f"{flag} must be an ISO date (YYYY-MM-DD).",
            _EXIT_ARGS,
            error_code="SEARCH_INVALID_DATE",
        ) from exc
    return value


def search_history(
    query: str,
    *,
    mode: str = "lexical",
    scope: str = DEFAULT_SCOPE,
    limit: int = DEFAULT_SEARCH_LIMIT,
    kind: str | None = None,
    role: str | None = None,
    run_uuid: str | None = None,
    operation: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    database_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run one offline lexical search and return the bounded result payload.

    The query is validated to hold at least one searchable term before any
    database read. Every filter is applied in SQL before ``LIMIT``. A semantic or
    hybrid mode is refused with ``SEARCH_MODE_DEFERRED``; the embedding backends
    are a later stage. An absent database, or one without the lexical layer, yields
    an empty result and a warning instead of creating anything.
    """
    if mode != "lexical":
        raise SearchCommandError(
            "Semantic and hybrid search are deferred to a later release; pass --mode lexical.",
            _EXIT_ARGS,
            error_code="SEARCH_MODE_DEFERRED",
        )
    _validate_limit(limit)
    _validate_scope(scope)
    _validate_role(role)
    _validate_kind(kind)
    canonical_run = _validate_run_uuid(run_uuid)
    canonical_since = _validate_date(since, flag="--since")
    canonical_until = _validate_date(until, flag="--until")
    if query_tokens(query) == []:
        raise SearchCommandError(
            "The query must contain at least one searchable term.",
            _EXIT_ARGS,
            error_code="SEARCH_EMPTY_QUERY",
        )

    resolved = _resolve_database_path(database_path)
    connection = _open_read_connection(resolved)
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    schema_version: int | None = None
    try:
        if connection is None:
            warnings.append("History database is absent; nothing has been saved yet.")
        else:
            schema_version = _schema_version(connection)
            status = indexing.index_status(connection)
            if not status.get("available"):
                warnings.append(
                    "The lexical index is not built for this database; run `voiceover index build`."
                )
            else:
                roles = [role] if role is not None else roles_for_scope(scope)
                outcome = search_lexical(
                    connection,
                    query,
                    limit=limit,
                    roles=roles,
                    kind=kind,
                    run_uuid=canonical_run,
                    operation=operation,
                    provider=provider,
                    since=canonical_since,
                    until=canonical_until,
                )
                results = outcome["results"]
                # An upgraded v2 corpus starts with no chunks. Incremental
                # saves only index new sources, so absence of pending markers
                # does not mean older rows are searchable yet.
                unindexed = max(
                    0, status.get("sources_indexable", 0) - status.get("sources_indexed", 0)
                )
                if unindexed:
                    warnings.append(
                        f"{unindexed} saved text source(s) are not yet indexed; "
                        "run `voiceover index build`."
                    )
                pending = status.get("sources_pending", 0)
                if pending:
                    warnings.append(
                        f"{pending} text source(s) are pending indexing; run `voiceover index build`."
                    )
                labels_missing = status.get("labels_missing", 0)
                if labels_missing:
                    warnings.append(
                        f"{labels_missing} run label(s) are missing from the index; "
                        "run `voiceover index build`."
                    )
                missing_part_directions = status.get("part_directions_missing", 0)
                if missing_part_directions:
                    warnings.append(
                        f"{missing_part_directions} saved part direction(s) are not indexed; "
                        "run `voiceover index build`."
                    )
                inactive_style_chunks = status.get("inactive_style_chunks", 0)
                if inactive_style_chunks:
                    warnings.append(
                        f"{inactive_style_chunks} unused run-level direction chunk(s) remain "
                        "indexed; run `voiceover index build`."
                    )
                incomplete = status.get("sources_incomplete", 0)
                if incomplete:
                    warnings.append(
                        f"{incomplete} saved text source(s) lack retrievable text and "
                        "cannot be searched."
                    )
                if status.get("needs_rebuild") and (
                    status.get("indexed_chunks")
                    or status.get("sources_indexable")
                    or status.get("label_runs")
                ):
                    warnings.append(
                        "The search index has a different or missing chunker version; "
                        "run `voiceover index rebuild`."
                    )
    finally:
        if connection is not None:
            connection.close()

    return {
        "dry_run": False,
        "database": {
            "path": str(resolved),
            "exists": connection is not None or resolved.exists(),
            "schema_version": schema_version,
        },
        "query": query,
        "mode": mode,
        "scope": scope,
        "filters": {
            "kind": kind,
            "role": role,
            "run_uuid": canonical_run,
            "operation": operation,
            "provider": provider,
            "since": canonical_since,
            "until": canonical_until,
        },
        "limit": limit,
        "count": len(results),
        "results": results,
        "warnings": warnings,
    }


def index_status_report(*, database_path: Path | str | None = None) -> dict[str, Any]:
    """Return the truthful derived-index status without creating anything."""
    resolved = _resolve_database_path(database_path)
    connection = _open_read_connection(resolved)
    try:
        if connection is None:
            return {
                "dry_run": False,
                "available": False,
                "database": {"path": str(resolved), "exists": False, "schema_version": None},
            }
        status = indexing.index_status(connection)
        schema_version = _database_metadata(resolved, connection)["schema_version"]
        return {
            "dry_run": False,
            "database": {
                "path": str(resolved),
                "exists": True,
                "schema_version": schema_version,
            },
            **status,
        }
    finally:
        if connection is not None:
            connection.close()


def _open_writable_database(
    database_path: Path | str | None,
) -> HistoryDatabase:
    """Open and migrate the writable history database, creating a missing home."""
    if database_path is None:
        try:
            ensure_history_home()
        except HistoryHomePermissionError as exc:
            raise SearchCommandError(
                str(exc), _EXIT_OUTPUT, error_code="HISTORY_HOME_PERMISSION"
            ) from exc
        except HistoryPathsError as exc:
            raise SearchCommandError(
                str(exc), _EXIT_ARGS, error_code="HISTORY_HOME_INVALID"
            ) from exc
    resolved = _resolve_database_path(database_path)
    database = HistoryDatabase(resolved)
    try:
        database.migrate()
    except SchemaVersionTooNewError as exc:
        database.close()
        raise SearchCommandError(
            _DATABASE_TOO_NEW_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_TOO_NEW"
        ) from exc
    except MigrationChecksumError as exc:
        database.close()
        raise SearchCommandError(
            _DATABASE_CHECKSUM_MESSAGE,
            _EXIT_PROVIDER,
            error_code="HISTORY_DATABASE_CHECKSUM_MISMATCH",
        ) from exc
    except (HistoryDatabaseError, sqlite3.Error) as exc:
        database.close()
        raise SearchCommandError(
            _DATABASE_UNREADABLE_MESSAGE, _EXIT_PROVIDER, error_code="HISTORY_DATABASE_UNREADABLE"
        ) from exc
    return database


def build_lexical_index(*, database_path: Path | str | None = None) -> dict[str, Any]:
    """Incrementally index everything not yet indexed; offline and idempotent."""
    database = _open_writable_database(database_path)
    try:
        result = indexing.build_index(database.connection)
        status = indexing.index_status(database.connection)
        schema_version = database.schema_version
    finally:
        database.close()
    return {
        "dry_run": False,
        "database": {
            "path": str(database.path),
            "exists": True,
            "schema_version": schema_version,
        },
        **result,
        "status": status,
    }


def rebuild_lexical_index(*, database_path: Path | str | None = None) -> dict[str, Any]:
    """Drop and rebuild the whole derived index from SQLite only."""
    database = _open_writable_database(database_path)
    try:
        result = indexing.rebuild_index(database.connection)
        status = indexing.index_status(database.connection)
        schema_version = database.schema_version
    finally:
        database.close()
    return {
        "dry_run": False,
        "database": {
            "path": str(database.path),
            "exists": True,
            "schema_version": schema_version,
        },
        **result,
        "status": status,
    }
