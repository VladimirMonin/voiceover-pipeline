"""CLI contract tests for ``voiceover search`` and ``voiceover index``.

These exercise the machine-facing JSON envelope and exit codes of the S08 offline
search commands through the real CLI entry point: lexical results, human output,
the deferred semantic/hybrid refusal, the index status/build/rebuild commands, and
the help exits. They stay offline and use an isolated ``VOICEOVER_HOME``.
"""

from __future__ import annotations

import pytest
from conftest import cli_json, run_cli

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.paths import ensure_history_home, history_database_path
from voiceover_pipeline.history.repository import TEXT_KIND_TTS_SCRIPT, HistoryRepository


def _build_home(home):
    ensure_history_home(home)
    database_path = history_database_path(home)
    with HistoryDatabase(database_path) as database:
        database.migrate()
        repository = HistoryRepository(database)
        run = repository.create_run(
            operation="tts", run_root=str(home / "runs" / "a"), user_label="метка поиска"
        )
        repository.add_text_source(
            run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="native_snapshot",
            content="Проверяемый сценарный текст",
        )
    return database_path


@pytest.fixture
def home(tmp_path, monkeypatch):
    resolved = tmp_path / "home"
    monkeypatch.setenv("VOICEOVER_HOME", str(resolved))
    _build_home(resolved)
    return resolved


def test_search_json_returns_results(home):
    returncode, payload = cli_json("search", "сценарный", "--mode", "lexical", "--json")

    assert returncode == 0
    assert payload["status"] == "success"
    assert payload["mode"] == "lexical"
    assert payload["count"] == 1
    assert payload["results"][0]["role"] == "speech"


def test_search_until_includes_whole_named_day(home):
    with HistoryDatabase(history_database_path(home)) as database:
        database.migrate()
        database.connection.execute("UPDATE runs SET created_at = ?", ("2026-05-01T23:59:59Z",))

    same_day_code, same_day = cli_json("search", "сценарный", "--until", "2026-05-01", "--json")
    earlier_code, earlier_day = cli_json("search", "сценарный", "--until", "2026-04-30", "--json")
    assert (same_day_code, same_day["count"]) == (0, 1)
    assert (earlier_code, earlier_day["count"]) == (0, 0)


def test_search_date_filters_require_zero_padded_iso_date(home):
    for flag in ("--since", "--until"):
        code, payload = cli_json("search", "сценарный", flag, "2026-5-1", "--json")
        assert code == 2
        assert payload["details"]["error_code"] == "SEARCH_INVALID_DATE"


def test_search_human_output_exit_zero(home):
    proc = run_cli("search", "метка")

    assert proc.returncode == 0
    assert "Search results" in proc.stdout
    assert "метка" in proc.stdout


def test_search_semantic_and_hybrid_modes_are_deferred(home):
    for mode in ("semantic", "hybrid"):
        returncode, payload = cli_json("search", "текст", "--mode", mode, "--json")
        assert returncode == 2
        assert payload["status"] == "error"
        assert payload["details"]["error_code"] == "SEARCH_MODE_DEFERRED"


def test_search_absent_database_warns_without_creating(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEOVER_HOME", str(tmp_path / "empty"))

    returncode, payload = cli_json("search", "что-нибудь", "--json")

    assert returncode == 0
    assert payload["count"] == 0
    assert payload["warnings"]
    assert not (tmp_path / "empty" / "history.sqlite3").exists()


def test_index_status_json(home):
    returncode, payload = cli_json("index", "status", "--json")

    assert returncode == 0
    assert payload["available"] is True
    assert payload["sources_indexed"] == 1
    assert payload["complete"] is True


def test_index_status_human_shows_missing_label_count(home):
    proc = run_cli("index", "status")
    assert proc.returncode == 0
    assert "labels_missing=0" in proc.stdout


def test_index_build_and_rebuild_json(home):
    build_code, build_payload = cli_json("index", "build", "--json")
    assert build_code == 0
    assert build_payload["status"]["indexed_chunks"] >= 1

    rebuild_code, rebuild_payload = cli_json("index", "rebuild", "--json")
    assert rebuild_code == 0
    assert rebuild_payload["mode"] == "rebuild"
    assert rebuild_payload["status"]["indexed_chunks"] == build_payload["status"]["indexed_chunks"]


def test_search_and_index_help_exit_zero():
    assert run_cli("search", "--help").returncode == 0
    assert run_cli("index", "--help").returncode == 0
    assert run_cli("index", "status", "--help").returncode == 0
