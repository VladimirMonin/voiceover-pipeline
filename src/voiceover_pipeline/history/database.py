"""SQLite connection, pragmas, and versioned checksummed migrations for history.

Plan section 6 makes SQLite the canonical state store for the pipeline. This
module owns only the connection contract and the migration ledger; the typed
entity API lives in :mod:`voiceover_pipeline.history.repository`.

Guarantees enforced here:

* Before any schema change, a connection is opened and its identity is checked
  read-only: ``PRAGMA user_version`` and the ``schema_migrations`` ledger must be
  consistent with this binary. A too-new version, a stale checksum, or a
  nonempty database without a ledger fails closed *before* ``journal_mode=WAL``
  or any DDL runs, so a rejected database keeps its original journal mode.
* Only after that check does the connection switch to WAL and set the bounded
  ``busy_timeout`` and ``foreign_keys``.
* Each migration carries a checksum over its exact statements. The initial
  ledger and every migration run in one transaction, so a failed migration
  leaves neither a ledger row nor the ledger table behind.
* A migration marked ``backup_before`` takes a consistent copy through the
  SQLite online backup API first, which captures un-checkpointed WAL frames
  that a plain file copy of the main database would miss. Backups are reserved
  exclusively (never overwritten) with private file mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

_logger = logging.getLogger("voiceover_pipeline.history")

DEFAULT_BUSY_TIMEOUT_MS: Final = 5000
SCHEMA_MIGRATIONS_TABLE: Final = "schema_migrations"
_MAX_BACKUP_NAME_ATTEMPTS: Final = 1000
_BACKUP_FILE_MODE: Final = 0o600

_SCHEMA_MIGRATIONS_DDL: Final = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "checksum TEXT NOT NULL, "
    "applied_at TEXT NOT NULL)"
)

_INSERT_MIGRATION: Final = (
    "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)"
)


def utc_now() -> str:
    """Return the current UTC time as a second-precision ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class HistoryDatabaseError(RuntimeError):
    """Base class for history database contract violations."""


class SchemaVersionTooNewError(HistoryDatabaseError):
    """The database was written by a newer schema than this binary knows."""


class MigrationChecksumError(HistoryDatabaseError):
    """A recorded migration no longer matches this binary's statements."""


class HistoryDatabaseReadOnlyError(HistoryDatabaseError):
    """An existing database cannot be read read-only without writing a sidecar."""


@dataclass(frozen=True)
class Migration:
    """One versioned schema change with a checksum over its exact statements."""

    version: int
    name: str
    statements: tuple[str, ...]
    backup_before: bool = False

    @property
    def checksum(self) -> str:
        """Return the sha256 of the version, name, and statements."""
        payload = json.dumps(
            {"version": self.version, "name": self.name, "statements": list(self.statements)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The initial schema keeps only the entities plan section 6 marks as necessary.
# search_chunks / FTS5 / embedding tables are deferred to the stages that use
# them, so this migration stays small and honest about what it provides.
#
# Run ownership is enforced at the database level: a part, attempt, artifact, or
# text source may only link to a part/attempt/artifact that belongs to the same
# run. The composite foreign keys resolve against the extra UNIQUE keys on the
# parent tables, and a NULL child column simply means "no such link". Ordinary
# runs may share a ``run_root``; the one-per-source identity for legacy imports
# is the nullable, unique ``legacy_source_root`` column.
_CORE_SCHEMA: Final = Migration(
    version=1,
    name="create_history_core",
    statements=(
        "CREATE TABLE runs ("
        "run_uuid TEXT PRIMARY KEY, "
        "operation TEXT NOT NULL, "
        "user_label TEXT, "
        "parent_uuid TEXT REFERENCES runs(run_uuid) ON DELETE SET NULL, "
        "status TEXT NOT NULL, "
        "run_root TEXT NOT NULL, "
        "legacy_source_root TEXT, "
        "config_snapshot TEXT, "
        "record_version INTEGER NOT NULL DEFAULT 1, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)",
        "CREATE INDEX idx_runs_run_root ON runs (run_root)",
        "CREATE INDEX idx_runs_user_label ON runs (user_label)",
        "CREATE INDEX idx_runs_operation ON runs (operation)",
        "CREATE UNIQUE INDEX idx_runs_legacy_source_root ON runs (legacy_source_root)",
        "CREATE TABLE parts ("
        "part_uuid TEXT PRIMARY KEY, "
        "run_uuid TEXT NOT NULL REFERENCES runs(run_uuid) ON DELETE CASCADE, "
        "position INTEGER NOT NULL, "
        "prepared_text TEXT, "
        "voice TEXT, "
        "vibe_shared TEXT, "
        "vibe_specific TEXT, "
        "vibe_effective TEXT, "
        "fingerprint TEXT, "
        "stage TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "UNIQUE (run_uuid, position), "
        "UNIQUE (run_uuid, part_uuid))",
        "CREATE INDEX idx_parts_run ON parts (run_uuid)",
        "CREATE TABLE attempts ("
        "attempt_uuid TEXT PRIMARY KEY, "
        "run_uuid TEXT NOT NULL REFERENCES runs(run_uuid) ON DELETE CASCADE, "
        "part_uuid TEXT, "
        "call_type TEXT NOT NULL, "
        "provider TEXT, "
        "model TEXT, "
        "account_alias TEXT, "
        "remote_id TEXT, "
        "status TEXT, "
        "usage_json TEXT, "
        "cost TEXT, "
        "cost_raw TEXT, "
        "cost_currency TEXT, "
        "cost_source TEXT, "
        "cost_exact_available INTEGER NOT NULL DEFAULT 0, "
        "error TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "UNIQUE (run_uuid, attempt_uuid), "
        "FOREIGN KEY (run_uuid, part_uuid) "
        "REFERENCES parts(run_uuid, part_uuid) ON DELETE CASCADE)",
        "CREATE INDEX idx_attempts_run ON attempts (run_uuid)",
        "CREATE INDEX idx_attempts_part ON attempts (part_uuid)",
        "CREATE TABLE artifacts ("
        "artifact_uuid TEXT PRIMARY KEY, "
        "run_uuid TEXT NOT NULL REFERENCES runs(run_uuid) ON DELETE CASCADE, "
        "part_uuid TEXT, "
        "attempt_uuid TEXT, "
        "role TEXT NOT NULL, "
        "path_kind TEXT NOT NULL, "
        "path TEXT NOT NULL, "
        "mime TEXT, "
        "size_bytes INTEGER, "
        "sha256 TEXT, "
        "media_metadata_json TEXT, "
        "availability TEXT NOT NULL DEFAULT 'present', "
        "created_at TEXT NOT NULL, "
        "UNIQUE (run_uuid, artifact_uuid), "
        "FOREIGN KEY (run_uuid, part_uuid) "
        "REFERENCES parts(run_uuid, part_uuid) ON DELETE CASCADE, "
        "FOREIGN KEY (run_uuid, attempt_uuid) "
        "REFERENCES attempts(run_uuid, attempt_uuid) ON DELETE CASCADE)",
        "CREATE INDEX idx_artifacts_run ON artifacts (run_uuid)",
        "CREATE INDEX idx_artifacts_sha256 ON artifacts (sha256)",
        "CREATE TABLE text_sources ("
        "text_source_uuid TEXT PRIMARY KEY, "
        "run_uuid TEXT NOT NULL REFERENCES runs(run_uuid) ON DELETE CASCADE, "
        "part_uuid TEXT, "
        "artifact_uuid TEXT, "
        "kind TEXT NOT NULL, "
        "content TEXT, "
        "content_hash TEXT, "
        "language TEXT, "
        "origin TEXT NOT NULL, "
        "text_completeness TEXT NOT NULL DEFAULT 'complete', "
        "created_at TEXT NOT NULL, "
        "FOREIGN KEY (run_uuid, part_uuid) "
        "REFERENCES parts(run_uuid, part_uuid) ON DELETE CASCADE, "
        "FOREIGN KEY (run_uuid, artifact_uuid) "
        "REFERENCES artifacts(run_uuid, artifact_uuid) ON DELETE CASCADE)",
        "CREATE INDEX idx_text_sources_run ON text_sources (run_uuid)",
        "CREATE INDEX idx_text_sources_kind ON text_sources (kind)",
        "CREATE INDEX idx_text_sources_hash ON text_sources (content_hash)",
    ),
)

MIGRATIONS: Final = (
    _CORE_SCHEMA,
    # v2 adds a monotonic per-run revision used as a compare-and-swap guard for
    # resume and the compatibility state export. It is an additive ALTER TABLE,
    # so it neither rewrites nor backfills existing rows: every pre-existing run
    # reads revision 1. The v1 bytes and checksum stay frozen, so a database
    # migrated by an earlier binary still validates unchanged.
    Migration(
        version=2,
        name="add_run_revision",
        statements=("ALTER TABLE runs ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",),
    ),
)
LATEST_SCHEMA_VERSION: Final = max(migration.version for migration in MIGRATIONS)


def _reserve_exclusive(path: Path) -> bool:
    """Create ``path`` exclusively with private mode; ``False`` if it exists."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _BACKUP_FILE_MODE)
    except FileExistsError:
        return False
    os.close(descriptor)
    return True


def _reserve_unique_backup(dest_dir: Path, base_name: str) -> Path:
    """Reserve a collision-free default backup path in ``dest_dir``."""
    for index in range(_MAX_BACKUP_NAME_ATTEMPTS):
        suffix = "" if index == 0 else f"-{index}"
        candidate = dest_dir / f"{base_name}{suffix}.sqlite3"
        if _reserve_exclusive(candidate):
            return candidate
    raise HistoryDatabaseError(f"could not reserve a unique backup name for {base_name!r}")


def _reserve_named_backup(dest_dir: Path, name: str) -> Path:
    """Reserve an explicitly named backup path, refusing to overwrite one."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("backup name must be a plain file name")
    candidate = dest_dir / name
    if not _reserve_exclusive(candidate):
        raise HistoryDatabaseError(f"backup {candidate} already exists; refusing to overwrite it")
    return candidate


def backup_database(source: sqlite3.Connection, dest_dir: Path, *, name: str | None = None) -> Path:
    """Write a consistent, never-overwriting copy of ``source`` into ``dest_dir``.

    ``sqlite3.Connection.backup`` copies the full database state including
    committed frames still living in the WAL, which a plain copy of the main
    file would miss. The destination is reserved exclusively with private file
    mode before any bytes are written: a default name that already exists in the
    same second gets a numeric suffix, and an explicit name that already exists
    raises instead of replacing an earlier recovery point.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest_path = _reserve_unique_backup(dest_dir, f"history-{stamp}")
    else:
        dest_path = _reserve_named_backup(dest_dir, name)
    try:
        destination = sqlite3.connect(str(dest_path))
        try:
            source.backup(destination)
        finally:
            destination.close()
    except BaseException:
        dest_path.unlink(missing_ok=True)
        raise
    return dest_path


def _recorded_migrations(connection: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    rows = connection.execute(
        f"SELECT version, name, checksum FROM {SCHEMA_MIGRATIONS_TABLE}"
    ).fetchall()
    return {int(row[0]): (row[1], row[2]) for row in rows}


def _validate_schema(
    connection: sqlite3.Connection, migrations: Sequence[Migration]
) -> dict[int, tuple[str, str]]:
    """Check database identity read-only and return the applied ledger.

    Runs before any write, WAL switch, or DDL. Rejects a database whose
    ``user_version`` or ledger names a version newer than this binary knows, a
    nonempty database without a migration ledger, a ledger record whose name or
    checksum no longer matches this binary, and an empty ledger or a ledger that
    disagrees with ``user_version`` or is not a prefix of known migrations.
    """
    ordered = sorted(migrations, key=lambda migration: migration.version)
    known = {migration.version: migration for migration in ordered}
    known_max = max(known, default=0)

    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }

    if SCHEMA_MIGRATIONS_TABLE not in tables:
        if user_version > known_max:
            raise SchemaVersionTooNewError(
                f"database user_version {user_version} is newer than the known {known_max}"
            )
        if user_version != 0 or tables:
            raise HistoryDatabaseError(
                "database has no schema_migrations ledger; refusing to touch an "
                "unrecognized or nonempty database"
            )
        return {}

    recorded = _recorded_migrations(connection)
    recorded_max = max(recorded) if recorded else 0
    if recorded_max > known_max:
        raise SchemaVersionTooNewError(
            f"database schema version {recorded_max} is newer than the known {known_max}"
        )
    if user_version > known_max:
        raise SchemaVersionTooNewError(
            f"database user_version {user_version} is newer than the known {known_max}"
        )

    for version in sorted(recorded):
        name, checksum = recorded[version]
        migration = known.get(version)
        if migration is None:
            raise MigrationChecksumError(
                f"recorded migration {version} ({name}) is not known to this binary"
            )
        if migration.name != name:
            raise MigrationChecksumError(
                f"migration {version} name mismatch: ledger has {name!r}, "
                f"binary expects {migration.name!r}"
            )
        if migration.checksum != checksum:
            raise MigrationChecksumError(
                f"migration {version} ({name}) checksum no longer matches the applied one"
            )

    # A ledger table without any committed migration cannot result from our
    # transactional bootstrap; do not adopt a foreign database simply because it
    # happens to contain an empty table with that name.
    if not recorded:
        raise HistoryDatabaseError("database has an empty migration ledger; refusing to migrate it")

    # Names and checksums match, so any remaining disagreement is a structural
    # inconsistency: ``user_version`` must equal the ledger's highest applied
    # version, and the recorded ledger must be a prefix of this binary's known
    # migration sequence. Applying the pending migrations of a database whose
    # ``user_version`` or ledger was tampered with would fail after the WAL
    # switch, so it is rejected here before any write.
    if user_version != recorded_max:
        raise HistoryDatabaseError(
            f"database user_version {user_version} disagrees with the recorded "
            f"migration ledger version {recorded_max}; refusing to migrate it"
        )
    known_versions = [migration.version for migration in ordered]
    recorded_versions = sorted(recorded)
    if recorded_versions != known_versions[: len(recorded_versions)]:
        raise HistoryDatabaseError(
            "database migration ledger is not a prefix of this binary's known "
            f"migrations: recorded {recorded_versions}, known {known_versions}"
        )
    return recorded


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    *,
    backup_dir: Path | None = None,
) -> list[Migration]:
    """Apply pending migrations, validating the ledger before any write.

    Raises :class:`SchemaVersionTooNewError` / :class:`HistoryDatabaseError` for
    an unrecognized database and :class:`MigrationChecksumError` for a stale
    ledger record. When a pending migration needs a backup and ``backup_dir`` is
    ``None`` the call fails closed instead of upgrading without a recoverable
    copy. The ledger table and all pending migrations share one transaction, so a
    failure leaves neither a ledger nor a partial schema. Applied migrations are
    logged as ``migration_applied`` with version and checksum only.
    """
    ordered = sorted(migrations, key=lambda migration: migration.version)
    recorded = _validate_schema(connection, ordered)
    pending = [migration for migration in ordered if migration.version not in recorded]
    if not pending:
        return []

    if recorded and any(migration.backup_before for migration in pending):
        if backup_dir is None:
            raise HistoryDatabaseError(
                "an incompatible migration needs a backup directory but none is configured"
            )
        backup_database(connection, backup_dir)

    newest_applied = max(recorded.keys() | {migration.version for migration in pending})
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(_SCHEMA_MIGRATIONS_DDL)
        for migration in pending:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                _INSERT_MIGRATION,
                (migration.version, migration.name, migration.checksum, utc_now()),
            )
        connection.execute(f"PRAGMA user_version = {int(newest_applied)}")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")

    for migration in pending:
        _logger.info(
            "migration_applied version=%d checksum=%s", migration.version, migration.checksum
        )
    return pending


def _sidecar_path(target: Path, suffix: str) -> Path:
    """Return the adjacent SQLite sidecar path for a suffix such as ``"-wal"``."""
    return target.with_name(target.name + suffix)


def _reject_symlinked_database(path: Path) -> None:
    """Refuse a database reached through a symlinked file or home directory.

    A symlinked ``history.sqlite3``, or a database under a symlinked home
    directory, would redirect a read or import outside the OS-private tree, so
    the target is rejected before any ``sqlite3.connect``, probe, DDL, or write.
    """
    if path.is_symlink():
        raise HistoryDatabaseError(f"history database must not be a symlink: {path}")
    if path.parent.is_symlink():
        raise HistoryDatabaseError(
            f"history database home directory must not be a symlink: {path.parent}"
        )


def connect_readonly(
    path: Path | str,
    *,
    migrations: Sequence[Migration] | None = None,
) -> sqlite3.Connection:
    """Open an existing history database read-only for metadata queries, writing nothing.

    The reader is deliberately narrower than :meth:`HistoryDatabase.connect`: it
    runs no migration or DDL, never switches the journal mode, and never creates
    a ``-wal`` or ``-shm`` sidecar. The main file is opened with
    ``mode=ro&immutable=1``, which is only trustworthy when no committed state can
    live outside the main file, so an adjacent ``-wal`` or ``-journal`` sidecar
    raises :class:`HistoryDatabaseReadOnlyError` instead of reading a possibly
    stale snapshot.

    The migration ledger and schema are validated with the same read-only gate as
    :meth:`HistoryDatabase.migrate`, so a foreign, corrupt, or newer database
    raises the corresponding :class:`HistoryDatabaseError` rather than being
    queried. A missing file raises :class:`FileNotFoundError`; a genuinely empty
    database (no tables, ``user_version`` 0) is accepted and yields a connection
    with no schema. A symlinked database file, or one under a symlinked home
    directory, raises :class:`HistoryDatabaseError` before any connect.
    """
    target = Path(path).expanduser().absolute()
    _reject_symlinked_database(target)
    if not target.exists():
        raise FileNotFoundError(str(target))
    for suffix in ("-wal", "-journal"):
        sidecar = _sidecar_path(target, suffix)
        if sidecar.exists():
            raise HistoryDatabaseReadOnlyError(
                f"history database {target} has a {suffix.lstrip('-')} sidecar; refusing to "
                "read a possibly stale main file or create a sidecar"
            )
    connection = sqlite3.connect(f"{target.as_uri()}?mode=ro&immutable=1", uri=True, timeout=2.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        _validate_schema(connection, MIGRATIONS if migrations is None else migrations)
    except BaseException:
        connection.close()
        raise
    return connection


def connect_readonly_consistent(
    path: Path | str,
    *,
    migrations: Sequence[Migration] | None = None,
) -> sqlite3.Connection:
    """Open an existing history database read-only with the most trustworthy view.

    This reader writes nothing of its own: no migration, no DDL, no journal-mode
    switch, no database creation, and no sidecar of its own. Which view it opens
    is decided by the ``-wal`` sidecar at connect time:

    * Without a ``-wal`` every committed byte is already in the main file, so the
      read falls through to :func:`connect_readonly`, the ``immutable=1``
      no-sidecar reader. That is what keeps a quiescent database quiescent: a
      read-only WAL connection would create an empty ``-wal``/``-shm`` pair that a
      read-only connection cannot remove again, and every later immutable read of
      that home would then fail closed.
    * With a ``-wal`` the main file alone may be a stale snapshot, so the database
      is opened ``mode=ro`` over the WAL and the committed frames are read, and a
      run another process just committed is never missed. A live ``-wal``/``-shm``
      pair is reused as it is: this reader never deletes and never overwrites a
      sidecar.

    A hot ``-journal`` (an unclean non-WAL rollback) is refused because the main
    file may be mid-rollback, and a symlinked database, home directory, or sidecar
    is rejected before any connect. The migration ledger is validated with the same
    read-only gate as :meth:`HistoryDatabase.migrate`, so a foreign, corrupt, or
    newer database raises the corresponding :class:`HistoryDatabaseError`; a
    missing file raises :class:`FileNotFoundError`.

    Reader selection is a lock-free best-effort gate, so it carries the same
    quiescence assumption :func:`connect_readonly` already documents for
    ``list``/``show``: a writer that creates its ``-wal`` only after this check is
    read through the quiescent branch instead of failing closed, so a commit that
    races the check is missed rather than reported. A writer whose WAL frames are
    already committed is always read WAL-consistently, which is the case
    ``history costs`` cares about, and no read ever fabricates a missing row or
    writes a sidecar.
    """
    target = Path(path).expanduser().absolute()
    _reject_symlinked_database(target)
    if not target.exists():
        raise FileNotFoundError(str(target))
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = _sidecar_path(target, suffix)
        if sidecar.is_symlink():
            raise HistoryDatabaseError(f"history database sidecar must not be a symlink: {sidecar}")
    journal = _sidecar_path(target, "-journal")
    if journal.exists():
        raise HistoryDatabaseReadOnlyError(
            f"history database {target} has a journal sidecar; refusing to read a "
            "possibly unrolled-back main file"
        )
    if not _sidecar_path(target, "-wal").exists():
        return connect_readonly(target, migrations=migrations)
    connection = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=2.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        _validate_schema(connection, MIGRATIONS if migrations is None else migrations)
    except BaseException:
        connection.close()
        raise
    return connection


class HistoryDatabase:
    """Owns the SQLite connection and migration state for one history database.

    Use :meth:`migrate` before touching :attr:`connection`. The default
    ``backup_dir`` is a ``backups`` directory beside the database file. A failed
    open or migration closes the connection, so no writable handle is exposed
    for a database this binary refused.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        migrations: Sequence[Migration] | None = None,
        backup_dir: Path | str | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.path = Path(path)
        self.migrations: tuple[Migration, ...] = (
            tuple(migrations) if migrations is not None else MIGRATIONS
        )
        self.backup_dir = (
            Path(backup_dir) if backup_dir is not None else self.path.parent / "backups"
        )
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._connection: sqlite3.Connection | None = None

    @property
    def _is_memory(self) -> bool:
        return str(self.path) == ":memory:"

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the open connection; raises before :meth:`migrate` has succeeded."""
        if self._connection is None:
            raise HistoryDatabaseError("history database is not open; call migrate() first")
        return self._connection

    @property
    def schema_version(self) -> int:
        """Return the highest applied migration version."""
        row = self.connection.execute(
            f"SELECT COALESCE(MAX(version), 0) AS version FROM {SCHEMA_MIGRATIONS_TABLE}"
        ).fetchone()
        return int(row["version"])

    def connect(self) -> sqlite3.Connection:
        """Open the connection, validate the schema read-only, then set pragmas."""
        if self._connection is not None:
            return self._connection
        _reject_symlinked_database(self.path)
        connection = sqlite3.connect(
            str(self.path), timeout=self.busy_timeout_ms / 1000.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        try:
            _validate_schema(connection, self.migrations)
        except BaseException:
            connection.close()
            raise
        if not self._is_memory:
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        self._connection = connection
        return connection

    def migrate(self) -> list[Migration]:
        """Connect if needed and apply pending migrations; returns what was applied."""
        try:
            connection = self._connection if self._connection is not None else self.connect()
            return apply_migrations(connection, self.migrations, backup_dir=self.backup_dir)
        except BaseException:
            self.close()
            raise

    def backup(self, dest_dir: Path | str | None = None, *, name: str | None = None) -> Path:
        """Take a backup of the open database via the SQLite backup API."""
        target = Path(dest_dir) if dest_dir is not None else self.backup_dir
        return backup_database(self.connection, target, name=name)

    def close(self) -> None:
        """Close the connection if it is open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> HistoryDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
