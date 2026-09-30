"""Contract tests for offline lexical index maintenance (plan section 8, S08).

These cover the derived-index lifecycle: immediate indexing on a normal save,
idempotent incremental build, deterministic rebuild from SQLite only, a failed
derived write that keeps the canonical text and records ``index_pending``,
truthful status counts for an incomplete imported corpus and for excluded private
sources, read-only status that creates nothing, and atomic rebuild recovery. They
run fully offline on temporary SQLite databases.
"""

from __future__ import annotations

import sqlite3

import pytest

import voiceover_pipeline.search.indexing as indexing_module
from voiceover_pipeline.commands import search as search_commands
from voiceover_pipeline.history.database import MIGRATIONS, HistoryDatabase
from voiceover_pipeline.history.repository import (
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_ASR_CONTEXT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
)
from voiceover_pipeline.search import indexing
from voiceover_pipeline.search.lexical import roles_for_scope, search_lexical


@pytest.fixture
def database(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as db:
        db.migrate()
        yield db


@pytest.fixture
def repository(database):
    return HistoryRepository(database)


def _search(connection, query):
    return search_lexical(connection, query, limit=20, roles=roles_for_scope("speech"))


def test_save_indexes_immediately(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="немедленная индексация"
    )

    # No separate index build: the normal save already made it searchable.
    assert len(_search(database.connection, "немедленная")["results"]) == 1


def test_created_run_label_is_indexed_immediately(repository, database, tmp_path):
    repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "a"), user_label="метка запуска"
    )

    assert len(_search(database.connection, "метка")["results"]) == 1


def test_legacy_imported_run_label_is_indexed(repository, database, tmp_path):
    source_root = str(tmp_path / "out" / "prod")
    with repository.transaction():
        repository.create_legacy_run(
            operation="tts",
            run_root=source_root,
            legacy_source_root=source_root,
            user_label="legacy prod",
        )

    assert len(_search(database.connection, "legacy")["results"]) == 1


def test_canonical_run_delete_removes_derived_fts_terms(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "delete"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="удаляемые слова"
    )
    assert database.connection.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0] == 1

    # SQLite's canonical ON DELETE CASCADE removes chunks. The FTS mirror must
    # not retain words that no longer have any canonical text even before build.
    database.connection.execute("DELETE FROM runs WHERE run_uuid = ?", (run.run_uuid,))
    assert database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0] == 0
    assert database.connection.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0] == 0


def test_index_build_is_idempotent(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="текст для сборки"
    )

    first = indexing.build_index(database.connection)
    second = indexing.build_index(database.connection)

    assert first["sources_written"] == 0  # already indexed by the save
    assert second["sources_written"] == 0
    assert second["chunks_written"] == 0
    before = database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0]
    indexing.build_index(database.connection)
    after = database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0]
    assert before == after


def test_index_rebuild_is_deterministic(repository, database, tmp_path):
    run = repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "a"), user_label="сборка подкаста"
    )
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="первый текст истории"
    )
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_ASR_TRANSCRIPT, origin="x", content="второй текст истории"
    )

    before = database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0]
    result = indexing.rebuild_index(database.connection)
    after = database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0]

    assert after == before == result["chunks_written"]
    assert len(_search(database.connection, "истории")["results"]) == 2
    assert len(_search(database.connection, "сборка")["results"]) == 1


def test_failed_derived_index_keeps_text_and_records_pending(
    repository, database, tmp_path, monkeypatch, caplog
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))

    def _boom(*args, **kwargs):
        raise RuntimeError("derived index unavailable")

    monkeypatch.setattr(indexing_module, "index_committed_text_sources", _boom)

    with caplog.at_level("WARNING", logger="voiceover_pipeline.history"):
        source = repository.add_text_source(
            run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="канонический текст"
        )

    # Canonical text is committed even though the derived write failed.
    assert repository.get_text_sources(run.run_uuid)[0].content == "канонический текст"
    pending = database.connection.execute(
        "SELECT text_source_uuid, reason FROM search_index_pending"
    ).fetchall()
    assert [(row[0], row[1]) for row in pending] == [
        (source.text_source_uuid, "derived_write_failed")
    ]
    assert any("search_index_failed" in record.getMessage() for record in caplog.records)

    # A later offline index build recovers it from SQLite alone.
    monkeypatch.undo()
    indexing.build_index(database.connection)
    assert len(_search(database.connection, "канонический")["results"]) == 1
    remaining = database.connection.execute("SELECT COUNT(*) FROM search_index_pending").fetchone()[
        0
    ]
    assert remaining == 0


def test_label_only_index_failure_is_incomplete_until_build(
    repository, database, tmp_path, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("derived index unavailable")

    monkeypatch.setattr(indexing_module, "index_committed_text_sources", _boom)
    run = repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "label"), user_label="поисковая метка"
    )
    assert repository.get_run(run.run_uuid).user_label == "поисковая метка"
    status = indexing.index_status(database.connection)
    assert status["label_runs"] == 1
    assert status["labels_missing"] == 1
    assert status["complete"] is False
    assert _search(database.connection, "поисковая")["results"] == []
    warning_result = search_commands.search_history(
        "поисковая", database_path=tmp_path / "history.sqlite3"
    )
    assert any("run label(s)" in message for message in warning_result["warnings"])

    monkeypatch.undo()
    indexing.build_index(database.connection)
    assert indexing.index_status(database.connection)["labels_missing"] == 0
    assert indexing.index_status(database.connection)["complete"] is True
    assert len(_search(database.connection, "поисковая")["results"]) == 1


def test_index_status_reports_incomplete_corpus_not_complete(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="legacy_import", content="полный текст"
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="legacy_import",
        content=None,
        content_hash="deadbeef",
        text_completeness=TEXT_COMPLETENESS_INCOMPLETE,
    )

    status = indexing.index_status(database.connection)

    assert status["sources_total"] == 2
    assert status["sources_indexable"] == 1
    assert status["sources_indexed"] == 1
    assert status["sources_incomplete"] == 1
    assert status["complete"] is False
    search = search_commands.search_history("полный", database_path=tmp_path / "history.sqlite3")
    assert search["count"] == 1
    assert any("cannot be searched" in warning for warning in search["warnings"])


def test_search_warns_when_chunker_version_needs_rebuild(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "old-index"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="старая нормализация"
    )
    database.connection.execute("UPDATE search_index_state SET chunker_version = 0")
    assert indexing.index_status(database.connection)["needs_rebuild"] is True

    search = search_commands.search_history("старая", database_path=tmp_path / "history.sqlite3")
    assert search["count"] == 1
    assert any("index rebuild" in warning for warning in search["warnings"])


def test_index_status_counts_private_excluded(repository, database, tmp_path):
    run = repository.create_run(operation="asr", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_ASR_CONTEXT, origin="x", content="промпт"
    )

    status = indexing.index_status(database.connection)

    assert status["sources_private_excluded"] == 1
    assert status["sources_indexed"] == 0
    first = indexing.build_index(database.connection)
    second = indexing.build_index(database.connection)
    rebuilt = indexing.rebuild_index(database.connection)
    assert first["sources_written"] == 0
    assert second["sources_written"] == 0
    assert second["chunks_written"] == 0
    assert rebuilt["sources_written"] == 0
    assert indexing.index_status(database.connection)["sources_private_excluded"] == 1


def test_upgraded_v2_corpus_missing_old_sources_warns_until_build(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path, migrations=MIGRATIONS[:2]) as old_database:
        old_database.migrate()
        old_repository = HistoryRepository(old_database)
        old_run = old_repository.create_run(operation="tts", run_root=str(tmp_path / "old-run"))
        old_repository.add_text_source(
            old_run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="native_snapshot",
            content="старый индексируемый текст",
        )

    with HistoryDatabase(database_path) as upgraded:
        upgraded.migrate()
        repository = HistoryRepository(upgraded)
        new_run = repository.create_run(operation="tts", run_root=str(tmp_path / "new-run"))
        repository.add_text_source(
            new_run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="native_snapshot",
            content="новый индексируемый текст",
        )
        status = indexing.index_status(upgraded.connection)
        assert status["sources_indexable"] == 2
        assert status["sources_indexed"] == 1
        assert status["sources_pending"] == 0
        assert status["labels_missing"] == 0
        assert status["complete"] is False

    missing = search_commands.search_history("старый", database_path=database_path)
    assert missing["count"] == 0
    assert any("index build" in warning for warning in missing["warnings"])
    with HistoryDatabase(database_path) as repaired:
        repaired.migrate()
        indexing.build_index(repaired.connection)
    recovered = search_commands.search_history("старый", database_path=database_path)
    assert recovered["count"] == 1
    assert recovered["warnings"] == []


def test_index_status_unavailable_without_search_schema(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path, migrations=[MIGRATIONS[0]]) as db:
        db.migrate()
        status = indexing.index_status(db.connection)

    assert status["available"] is False
    assert status["schema_version"] == 1


def test_index_status_handles_empty_database_file(tmp_path):
    database_path = tmp_path / "empty.sqlite3"
    sqlite3.connect(database_path).close()

    payload = search_commands.index_status_report(database_path=database_path)

    assert payload["available"] is False
    assert payload["database"]["exists"] is True
    assert payload["database"]["schema_version"] is None


def test_rebuild_is_atomic_on_failure(repository, database, tmp_path, monkeypatch):
    run = repository.create_run(
        operation="tts",
        run_root=str(tmp_path / "runs" / "a"),
        user_label="метка сборки",
    )
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="исходный поисковый текст"
    )
    original_chunks = database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[
        0
    ]

    calls = {"count": 0}
    real_insert = indexing_module._insert_chunk

    def _insert(connection, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("rebuild interrupted")
        return real_insert(connection, **kwargs)

    monkeypatch.setattr(indexing_module, "_insert_chunk", _insert)

    with pytest.raises(RuntimeError):
        indexing.rebuild_index(database.connection)

    # The transaction rolled back: the previous index is intact and searchable.
    monkeypatch.undo()
    assert (
        database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0]
        == original_chunks
    )
    assert len(_search(database.connection, "поисковый")["results"]) == 1


def test_rebuild_clears_pending(repository, database, tmp_path, monkeypatch):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    monkeypatch.setattr(
        indexing_module,
        "index_committed_text_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")),
    )
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="текст после сбоя"
    )
    monkeypatch.undo()

    indexing.rebuild_index(database.connection)

    assert (
        database.connection.execute("SELECT COUNT(*) FROM search_index_pending").fetchone()[0] == 0
    )
    assert len(_search(database.connection, "сбоя")["results"]) == 1


# -- command-layer handlers ----------------------------------------------------


def test_index_status_report_is_readonly_and_creates_no_sidecar(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as db:
        db.migrate()
        repository = HistoryRepository(db)
        run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
        repository.add_text_source(
            run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="статусный текст"
        )

    payload = search_commands.index_status_report(database_path=database_path)

    assert payload["available"] is True
    assert payload["sources_indexed"] == 1
    assert not (tmp_path / "history.sqlite3-wal").exists()
    assert not (tmp_path / "history.sqlite3-shm").exists()


def test_search_history_command_returns_results_and_warns_on_absent_db(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as db:
        db.migrate()
        repository = HistoryRepository(db)
        run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
        repository.add_text_source(
            run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="сценарный поиск"
        )

    payload = search_commands.search_history("сценарный", database_path=database_path)
    assert payload["count"] == 1

    absent = search_commands.search_history(
        "сценарный", database_path=tmp_path / "missing" / "history.sqlite3"
    )
    assert absent["count"] == 0
    assert absent["warnings"]


def test_search_history_rejects_semantic_mode_and_empty_query(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with pytest.raises(search_commands.SearchCommandError) as deferred:
        search_commands.search_history("что-то", mode="semantic", database_path=database_path)
    assert deferred.value.error_code == "SEARCH_MODE_DEFERRED"
    assert deferred.value.code == 2

    with pytest.raises(search_commands.SearchCommandError) as empty:
        search_commands.search_history("   ", database_path=database_path)
    assert empty.value.error_code == "SEARCH_EMPTY_QUERY"


def test_build_and_rebuild_commands_report_truthful_counts(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as db:
        db.migrate()
        repository = HistoryRepository(db)
        run = repository.create_run(
            operation="tts", run_root=str(tmp_path / "runs" / "a"), user_label="командная метка"
        )
        repository.add_text_source(
            run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="командный текст"
        )

    built = search_commands.build_lexical_index(database_path=database_path)
    assert built["status"]["complete"] is True
    rebuilt = search_commands.rebuild_lexical_index(database_path=database_path)
    assert rebuilt["status"]["indexed_chunks"] == built["status"]["indexed_chunks"]


def test_index_writer_refuses_to_join_an_open_transaction(database):
    database.connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(Exception):
            indexing.build_index(database.connection)
    finally:
        database.connection.execute("ROLLBACK")
