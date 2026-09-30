"""Contract tests for the history SQLite connection and migration ledger.

These tests exercise ``voiceover_pipeline.history.database`` in isolation: the
pragmas every connection must carry, the read-only identity gate that runs
before WAL or any DDL, the versioned/checksummed migration ledger, and the
consistent, never-overwriting backup taken through the SQLite online backup API.
The typed entity API is covered separately in ``test_history_repository.py``.
"""

import logging
import shutil
import sqlite3
import stat

import pytest

from voiceover_pipeline.history.database import (
    MIGRATIONS,
    HistoryDatabase,
    HistoryDatabaseError,
    HistoryDatabaseReadOnlyError,
    Migration,
    MigrationChecksumError,
    SchemaVersionTooNewError,
    apply_migrations,
    connect_readonly,
    connect_readonly_consistent,
)

# Frozen sha256 of the v1 core migration; the v2 revision migration must never
# rewrite v1 bytes, so an already-migrated v1 database still validates against it.
_V1_CHECKSUM = "e802c869914ac25365b5ae9859bcfce81a4ef4937984988b2cfcdcd867e27232"

_LEDGER_DDL = (
    "CREATE TABLE schema_migrations ("
    "version INTEGER PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "checksum TEXT NOT NULL, "
    "applied_at TEXT NOT NULL)"
)


def _widgets_v1() -> Migration:
    return Migration(
        version=1,
        name="create_widgets",
        statements=("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)",),
    )


def _journal_mode(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("PRAGMA journal_mode").fetchone()[0].lower()
    finally:
        connection.close()


def _has_table(path, name):
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _write_ledger_only_database(path, rows):
    connection = sqlite3.connect(path)
    try:
        connection.execute(_LEDGER_DDL)
        for row in rows:
            connection.execute("INSERT INTO schema_migrations VALUES (?, ?, ?, ?)", row)
        connection.commit()
    finally:
        connection.close()


def _make_real_database(path, migrations):
    """Build a genuine migrated database with the given migration sequence."""
    with HistoryDatabase(path, migrations=migrations) as database:
        database.migrate()


def _force_delete_journal_and_metadata(path, *, user_version, drop_ledger_version=None):
    """Tamper a real database into a deterministic, non-WAL inconsistent state.

    Switches the journal mode back to ``delete`` so a rejected open cannot be
    mistaken for a WAL sidecar, optionally drops one recorded ledger row, then
    sets ``PRAGMA user_version`` to a value that no longer matches the ledger.
    """
    connection = sqlite3.connect(path)
    try:
        switched = connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        assert switched.lower() == "delete"
        if drop_ledger_version is not None:
            connection.execute(
                "DELETE FROM schema_migrations WHERE version = ?", (drop_ledger_version,)
            )
        connection.execute(f"PRAGMA user_version = {int(user_version)}")
        connection.commit()
    finally:
        connection.close()


def _assert_rejected_before_wal(path, *, ledger_versions, has_revision_column):
    """Assert migrate() rejects a tampered database without any write or WAL switch."""
    before_bytes = path.read_bytes()
    before_mode = stat.S_IMODE(path.stat().st_mode)

    database = HistoryDatabase(path)
    with pytest.raises(HistoryDatabaseError):
        database.migrate()

    assert path.read_bytes() == before_bytes
    assert stat.S_IMODE(path.stat().st_mode) == before_mode
    assert _journal_mode(path) == "delete"
    assert not path.with_name(path.name + "-wal").exists()
    assert not path.with_name(path.name + "-shm").exists()
    assert not path.with_name(path.name + "-journal").exists()
    with pytest.raises(HistoryDatabaseError):
        _ = database.connection

    connection = sqlite3.connect(path)
    try:
        recorded = [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    finally:
        connection.close()
    assert recorded == ledger_versions
    assert ("revision" in columns) is has_revision_column


def test_connection_enables_required_pragmas(tmp_path):
    database_path = tmp_path / "history.sqlite3"

    with HistoryDatabase(database_path, busy_timeout_ms=2500) as database:
        database.migrate()
        connection = database.connection

        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 2500


def test_foreign_key_violation_is_rejected(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as database:
        database.migrate()

        with pytest.raises(sqlite3.IntegrityError):
            database.connection.execute(
                "INSERT INTO parts (part_uuid, run_uuid, position, created_at, updated_at) "
                "VALUES ('part-1', 'missing-run', 1, '2026-01-01T00:00:00Z', "
                "'2026-01-01T00:00:00Z')"
            )


def test_connection_refuses_use_before_migrate(tmp_path):
    database = HistoryDatabase(tmp_path / "history.sqlite3")

    with pytest.raises(HistoryDatabaseError):
        _ = database.connection


def test_migrate_records_known_migrations_and_is_idempotent(tmp_path):
    database_path = tmp_path / "history.sqlite3"

    with HistoryDatabase(database_path) as database:
        applied_first = database.migrate()
        applied_second = database.migrate()

        assert [migration.version for migration in applied_first] == [1, 2]
        assert applied_second == []
        assert database.schema_version == 2
        assert database.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        row = database.connection.execute(
            "SELECT name, checksum FROM schema_migrations WHERE version = 1"
        ).fetchone()
        assert row["name"] == "create_history_core"
        assert len(row["checksum"]) == 64


def test_v1_core_migration_bytes_and_checksum_are_frozen():
    assert MIGRATIONS[0].version == 1
    assert MIGRATIONS[0].name == "create_history_core"
    assert MIGRATIONS[0].checksum == _V1_CHECKSUM


def test_v2_upgrade_of_real_v1_database_preserves_rows_and_fks(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    run_uuid = "11111111-2222-3333-4444-555555555555"

    with HistoryDatabase(database_path, migrations=[MIGRATIONS[0]]) as database:
        database.migrate()
        assert database.schema_version == 1
        columns = {
            row[1] for row in database.connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "revision" not in columns
        database.connection.execute(
            "INSERT INTO runs (run_uuid, operation, status, run_root, created_at, updated_at) "
            "VALUES (?, 'tts', 'completed', '/tmp/run', 't', 't')",
            (run_uuid,),
        )
        database.connection.execute(
            "INSERT INTO parts (part_uuid, run_uuid, position, created_at, updated_at) "
            "VALUES ('part-1', ?, 1, 't', 't')",
            (run_uuid,),
        )

    with HistoryDatabase(database_path) as database:
        applied = database.migrate()
        assert [migration.version for migration in applied] == [2]
        assert database.schema_version == 2
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        revised = database.connection.execute(
            "SELECT revision FROM runs WHERE run_uuid = ?", (run_uuid,)
        ).fetchone()
        assert revised["revision"] == 1
        assert (
            database.connection.execute(
                "SELECT COUNT(*) FROM parts WHERE run_uuid = ?", (run_uuid,)
            ).fetchone()[0]
            == 1
        )
        ledger = database.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row["version"] for row in ledger] == [1, 2]


def test_v2_migration_is_not_reapplied_on_reopen(tmp_path):
    database_path = tmp_path / "history.sqlite3"

    with HistoryDatabase(database_path) as database:
        assert [migration.version for migration in database.migrate()] == [1, 2]

    with HistoryDatabase(database_path) as database:
        assert database.migrate() == []
        assert database.schema_version == 2
        count = database.connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        assert count == 2


def test_migration_applied_is_logged_without_content(tmp_path, caplog):
    database_path = tmp_path / "history.sqlite3"

    with caplog.at_level(logging.INFO, logger="voiceover_pipeline.history"):
        with HistoryDatabase(database_path) as database:
            database.migrate()

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("migration_applied version=1 checksum=") for message in messages)


def test_migration_upgrade_preserves_existing_data(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    add_color = Migration(
        version=2,
        name="add_widget_color",
        statements=("ALTER TABLE widgets ADD COLUMN color TEXT",),
    )

    with HistoryDatabase(database_path, migrations=[_widgets_v1()]) as database:
        database.migrate()
        database.connection.execute("INSERT INTO widgets (id, name) VALUES (1, 'первый')")

    with HistoryDatabase(database_path, migrations=[_widgets_v1(), add_color]) as database:
        applied = database.migrate()

        assert [migration.version for migration in applied] == [2]
        assert database.schema_version == 2
        row = database.connection.execute(
            "SELECT id, name, color FROM widgets WHERE id = 1"
        ).fetchone()
        assert row["name"] == "первый"
        assert row["color"] is None


def test_failed_migration_leaves_no_ledger_or_partial_schema(tmp_path):
    broken = Migration(
        version=1,
        name="broken_widgets",
        statements=(
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY)",
            "THIS IS NOT VALID SQL",
        ),
    )
    database_path = tmp_path / "history.sqlite3"
    database = HistoryDatabase(database_path, migrations=[broken])

    with pytest.raises(sqlite3.OperationalError):
        database.migrate()

    assert not _has_table(database_path, "schema_migrations")
    assert not _has_table(database_path, "widgets")
    with pytest.raises(HistoryDatabaseError):
        _ = database.connection


def test_checksum_mismatch_is_rejected_before_wal_or_any_write(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _write_ledger_only_database(database_path, [(1, "create_widgets", "0" * 64, "t")])
    before = database_path.read_bytes()
    assert _journal_mode(database_path) == "delete"

    database = HistoryDatabase(database_path, migrations=[_widgets_v1()])
    with pytest.raises(MigrationChecksumError):
        database.migrate()

    assert database_path.read_bytes() == before
    assert _journal_mode(database_path) == "delete"
    with pytest.raises(HistoryDatabaseError):
        _ = database.connection


def test_ledger_name_mismatch_is_rejected(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _write_ledger_only_database(
        database_path, [(1, "renamed_widgets", _widgets_v1().checksum, "t")]
    )

    database = HistoryDatabase(database_path, migrations=[_widgets_v1()])
    with pytest.raises(MigrationChecksumError):
        database.migrate()

    assert not _has_table(database_path, "widgets")


def test_user_version_ahead_of_ledger_is_rejected_before_wal(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _make_real_database(database_path, [MIGRATIONS[0]])
    _force_delete_journal_and_metadata(database_path, user_version=2)
    assert _journal_mode(database_path) == "delete"

    _assert_rejected_before_wal(database_path, ledger_versions=[1], has_revision_column=False)


def test_user_version_behind_ledger_is_rejected_before_wal(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _make_real_database(database_path, MIGRATIONS)
    _force_delete_journal_and_metadata(database_path, user_version=1)
    assert _journal_mode(database_path) == "delete"

    _assert_rejected_before_wal(database_path, ledger_versions=[1, 2], has_revision_column=True)


def test_ledger_missing_an_earlier_version_is_rejected_before_wal(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _make_real_database(database_path, MIGRATIONS)
    _force_delete_journal_and_metadata(database_path, user_version=2, drop_ledger_version=1)
    assert _journal_mode(database_path) == "delete"

    _assert_rejected_before_wal(database_path, ledger_versions=[2], has_revision_column=True)


def test_ledgerless_future_user_version_is_rejected_without_writes(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")
        raw.execute("PRAGMA user_version = 999")
        raw.commit()
    finally:
        raw.close()
    before = database_path.read_bytes()
    assert _journal_mode(database_path) == "delete"

    database = HistoryDatabase(database_path)
    with pytest.raises(SchemaVersionTooNewError):
        database.migrate()

    assert database_path.read_bytes() == before
    assert _journal_mode(database_path) == "delete"
    assert not _has_table(database_path, "schema_migrations")
    assert _has_table(database_path, "foreign_table")
    with pytest.raises(HistoryDatabaseError):
        _ = database.connection


def test_ledgerless_user_version_within_known_range_is_rejected(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")
        raw.execute("PRAGMA user_version = 1")
        raw.commit()
    finally:
        raw.close()
    before = database_path.read_bytes()

    database = HistoryDatabase(database_path)
    with pytest.raises(HistoryDatabaseError):
        database.migrate()

    assert database_path.read_bytes() == before
    assert not _has_table(database_path, "schema_migrations")


def test_nonempty_ledgerless_database_is_rejected(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")
        raw.commit()
    finally:
        raw.close()
    before = database_path.read_bytes()

    database = HistoryDatabase(database_path)
    with pytest.raises(HistoryDatabaseError):
        database.migrate()

    assert database_path.read_bytes() == before
    assert not _has_table(database_path, "schema_migrations")


def test_future_schema_version_is_rejected(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    future = Migration(
        version=999,
        name="future_widgets",
        statements=("CREATE TABLE future_widgets (id INTEGER PRIMARY KEY)",),
    )

    with HistoryDatabase(database_path, migrations=[_widgets_v1(), future]) as database:
        database.migrate()
        assert database.schema_version == 999

    database = HistoryDatabase(database_path, migrations=[_widgets_v1()])
    with pytest.raises(SchemaVersionTooNewError):
        database.migrate()

    assert _has_table(database_path, "future_widgets")
    with pytest.raises(HistoryDatabaseError):
        _ = database.connection


def test_unknown_recorded_migration_is_rejected(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    other = Migration(
        version=2,
        name="other_binary_change",
        statements=("CREATE TABLE other_thing (id INTEGER PRIMARY KEY)",),
    )

    with HistoryDatabase(database_path, migrations=[_widgets_v1(), other]) as database:
        database.migrate()

    with HistoryDatabase(database_path, migrations=[_widgets_v1()]) as database:
        with pytest.raises(SchemaVersionTooNewError):
            database.migrate()


def test_backup_api_captures_wal_resident_data(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    database = HistoryDatabase(database_path)
    database.migrate()
    database.connection.execute(
        "INSERT INTO runs (run_uuid, operation, status, run_root, "
        "created_at, updated_at) "
        "VALUES ('run-1', 'tts', 'completed', '/tmp/run', 't', 't')"
    )

    wal_path = database_path.with_name(database_path.name + "-wal")
    assert wal_path.exists()

    backup_path = database.backup(tmp_path / "backups")
    database.close()

    # The backup is self-contained: opening it with no adjacent WAL file still
    # yields the committed row.
    assert not backup_path.with_name(backup_path.name + "-wal").exists()
    restored = HistoryDatabase(backup_path)
    restored.connect()
    try:
        row = restored.connection.execute("SELECT run_uuid FROM runs").fetchone()
        assert row["run_uuid"] == "run-1"
    finally:
        restored.close()


def test_backup_restores_after_original_files_are_lost(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    database = HistoryDatabase(database_path)
    database.migrate()
    database.connection.execute(
        "INSERT INTO runs (run_uuid, operation, status, run_root, "
        "created_at, updated_at) "
        "VALUES ('run-1', 'tts', 'completed', '/tmp/run', 't', 't')"
    )
    backup_path = database.backup(tmp_path / "backups")
    database.close()

    for suffix in ("", "-wal", "-shm"):
        leftover = database_path.with_name(database_path.name + suffix)
        if leftover.exists():
            leftover.unlink()
    shutil.copyfile(backup_path, database_path)

    restored = HistoryDatabase(database_path)
    restored.connect()
    try:
        row = restored.connection.execute("SELECT run_uuid FROM runs").fetchone()
        assert row["run_uuid"] == "run-1"
    finally:
        restored.close()


def _insert_run(database, run_uuid, run_root):
    database.connection.execute(
        "INSERT INTO runs (run_uuid, operation, status, run_root, created_at, updated_at) "
        "VALUES (?, 'tts', 'completed', ?, 't', 't')",
        (run_uuid, run_root),
    )


def test_explicit_backup_name_never_overwrites(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    backups_dir = tmp_path / "backups"
    database = HistoryDatabase(database_path)
    database.migrate()
    _insert_run(database, "run-1", "/tmp/run-1")
    first = database.backup(backups_dir, name="snapshot.sqlite3")
    first_bytes = first.read_bytes()

    _insert_run(database, "run-2", "/tmp/run-2")
    with pytest.raises(HistoryDatabaseError):
        database.backup(backups_dir, name="snapshot.sqlite3")
    database.close()

    assert first.read_bytes() == first_bytes
    assert len(list(backups_dir.glob("*.sqlite3"))) == 1


def test_default_backup_names_never_overwrite(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    backups_dir = tmp_path / "backups"
    database = HistoryDatabase(database_path)
    database.migrate()
    _insert_run(database, "run-1", "/tmp/run-1")
    first = database.backup(backups_dir)
    first_bytes = first.read_bytes()

    _insert_run(database, "run-2", "/tmp/run-2")
    second = database.backup(backups_dir)
    database.close()

    assert second != first
    assert first.exists() and second.exists()
    assert first.read_bytes() == first_bytes


def test_backup_files_use_private_mode(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    database = HistoryDatabase(database_path)
    database.migrate()
    backup_path = database.backup(tmp_path / "backups")
    database.close()

    assert stat.S_IMODE(backup_path.stat().st_mode) & 0o077 == 0


def test_incompatible_upgrade_takes_backup_before_migrating(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    backups_dir = tmp_path / "backups"
    upgrade = Migration(
        version=2,
        name="rebuild_widgets",
        statements=(
            "DROP TABLE widgets",
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY, label TEXT)",
        ),
        backup_before=True,
    )

    with HistoryDatabase(database_path, migrations=[_widgets_v1()]) as database:
        database.migrate()
        database.connection.execute("INSERT INTO widgets (id, name) VALUES (1, 'legacy')")

    with HistoryDatabase(
        database_path, migrations=[_widgets_v1(), upgrade], backup_dir=backups_dir
    ) as database:
        applied = database.migrate()
        assert [migration.version for migration in applied] == [2]
        assert database.connection.execute("SELECT COUNT(*) AS c FROM widgets").fetchone()["c"] == 0

    backup_files = sorted(backups_dir.glob("*.sqlite3"))
    assert len(backup_files) == 1
    connection = sqlite3.connect(backup_files[0])
    try:
        row = connection.execute("SELECT name FROM widgets WHERE id = 1").fetchone()
    finally:
        connection.close()
    assert row[0] == "legacy"


def test_required_backup_without_directory_fails_closed(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    upgrade = Migration(
        version=2,
        name="rebuild_widgets",
        statements=("DROP TABLE widgets",),
        backup_before=True,
    )

    with HistoryDatabase(database_path, migrations=[_widgets_v1()]) as database:
        database.migrate()

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        with pytest.raises(HistoryDatabaseError):
            apply_migrations(connection, [_widgets_v1(), upgrade], backup_dir=None)
        kept = connection.execute("SELECT COUNT(*) AS c FROM schema_migrations").fetchone()
        assert kept[0] == 1
    finally:
        connection.close()


def test_connect_readonly_reads_without_sidecar_or_modification(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as database:
        database.migrate()
        _insert_run(database, "run-1", "/tmp/run-1")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert sorted(before) == ["history.sqlite3"]

    connection = connect_readonly(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"] == 1
    finally:
        connection.close()

    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def test_connect_readonly_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "missing.sqlite3")


def test_connect_readonly_refuses_wal_sidecar_without_writes(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as database:
        database.migrate()
    wal_path = database_path.with_name(database_path.name + "-wal")
    wal_path.write_bytes(b"uncheckpointed-frames")

    with pytest.raises(HistoryDatabaseReadOnlyError):
        connect_readonly(database_path)

    assert wal_path.read_bytes() == b"uncheckpointed-frames"
    assert not database_path.with_name(database_path.name + "-shm").exists()


def test_connect_readonly_accepts_empty_database(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    database_path.write_bytes(b"")

    connection = connect_readonly(database_path)
    try:
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
    finally:
        connection.close()

    assert sorted(path.name for path in tmp_path.iterdir()) == ["history.sqlite3"]


def test_connect_readonly_consistent_reads_quiescent_database_without_sidecar(tmp_path):
    """A database with no ``-wal`` is read without creating one.

    Regression: the consistent reader used to open ``mode=ro`` unconditionally,
    which created an empty ``-wal``/``-shm`` pair that a read-only connection
    cannot remove again and that every later immutable read then refused.
    """
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as database:
        database.migrate()
        _insert_run(database, "run-1", "/tmp/run-1")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert sorted(before) == ["history.sqlite3"]

    connection = connect_readonly_consistent(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"] == 1
    finally:
        connection.close()

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_connect_readonly_consistent_reads_live_wal_without_touching_sidecars(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    writer = HistoryDatabase(database_path)
    writer.migrate()
    _insert_run(writer, "run-1", "/tmp/run-1")
    wal_path = database_path.with_name(database_path.name + "-wal")
    committed_frames = wal_path.read_bytes()
    assert committed_frames

    connection = connect_readonly_consistent(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"] == 1
    finally:
        connection.close()

    # The live writer's committed frames and WAL index are reused as they are.
    assert wal_path.read_bytes() == committed_frames
    assert database_path.with_name(database_path.name + "-shm").exists()
    assert writer.connection.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"] == 1
    writer.close()


def test_connect_readonly_consistent_reads_past_a_stale_empty_wal(tmp_path):
    """A stale zero-frame ``-wal`` is read through, never deleted or repaired.

    An empty ``-wal`` carries no committed state, so the consistent reader still
    reads the database, while the immutable reader keeps its documented
    fail-closed refusal instead of deleting a sidecar that a live writer could
    own.
    """
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as database:
        database.migrate()
        _insert_run(database, "run-1", "/tmp/run-1")
    wal_path = database_path.with_name(database_path.name + "-wal")
    shm_path = database_path.with_name(database_path.name + "-shm")
    wal_path.write_bytes(b"")
    shm_path.write_bytes(b"\0" * 32768)
    shm_size = shm_path.stat().st_size

    connection = connect_readonly_consistent(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"] == 1
    finally:
        connection.close()

    with pytest.raises(HistoryDatabaseReadOnlyError):
        connect_readonly(database_path)

    assert wal_path.read_bytes() == b""
    assert shm_path.stat().st_size == shm_size
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "history.sqlite3",
        "history.sqlite3-shm",
        "history.sqlite3-wal",
    ]


def test_connect_readonly_rejects_corrupt_database(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    database_path.write_bytes(b"not a database at all")

    with pytest.raises(sqlite3.DatabaseError):
        connect_readonly(database_path)


def test_connect_readonly_rejects_stale_ledger(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _write_ledger_only_database(database_path, [(1, "create_widgets", "0" * 64, "t")])

    with pytest.raises(MigrationChecksumError):
        connect_readonly(database_path, migrations=[_widgets_v1()])


@pytest.mark.parametrize("foreign_table", [False, True])
def test_empty_migration_ledger_is_rejected_before_wal(tmp_path, foreign_table):
    database_path = tmp_path / "history.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(_LEDGER_DDL)
        if foreign_table:
            connection.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    before = database_path.read_bytes()
    assert _journal_mode(database_path) == "delete"

    database = HistoryDatabase(database_path)
    with pytest.raises(HistoryDatabaseError):
        database.migrate()

    assert database_path.read_bytes() == before
    assert _journal_mode(database_path) == "delete"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["history.sqlite3"]
    assert not _has_table(database_path, "runs")


def test_connect_readonly_rejects_version_ledger_disagreement(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    _make_real_database(database_path, [MIGRATIONS[0]])
    _force_delete_journal_and_metadata(database_path, user_version=2)
    before = database_path.read_bytes()

    with pytest.raises(HistoryDatabaseError):
        connect_readonly(database_path)

    assert database_path.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["history.sqlite3"]


def _file_bytes(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_connect_readonly_rejects_symlinked_database_file(tmp_path):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_db = external_dir / "history.sqlite3"
    with HistoryDatabase(external_db) as database:
        database.migrate()
        _insert_run(database, "run-1", "/tmp/run-1")
    before = _file_bytes(external_dir)

    link = tmp_path / "history.sqlite3"
    link.symlink_to(external_db)

    with pytest.raises(HistoryDatabaseError):
        connect_readonly(link)

    assert _file_bytes(external_dir) == before


def test_connect_readonly_rejects_symlinked_home_directory(tmp_path):
    external_dir = tmp_path / "external-home"
    external_dir.mkdir()
    external_db = external_dir / "history.sqlite3"
    with HistoryDatabase(external_db) as database:
        database.migrate()
    before = _file_bytes(external_dir)

    linked_dir = tmp_path / "linked-home"
    linked_dir.symlink_to(external_dir, target_is_directory=True)

    with pytest.raises(HistoryDatabaseError):
        connect_readonly(linked_dir / "history.sqlite3")

    assert _file_bytes(external_dir) == before


def test_history_database_connect_rejects_symlinked_database_file(tmp_path):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_db = external_dir / "history.sqlite3"
    with HistoryDatabase(external_db) as database:
        database.migrate()
    before = _file_bytes(external_dir)

    link = tmp_path / "history.sqlite3"
    link.symlink_to(external_db)

    database = HistoryDatabase(link)
    with pytest.raises(HistoryDatabaseError):
        database.migrate()

    assert _file_bytes(external_dir) == before
    assert not link.with_name(link.name + "-wal").exists()
    assert not link.with_name(link.name + "-shm").exists()
