"""Contract tests for the ``voiceover history`` CLI surface.

Every fixture is synthetic and lives under ``tmp_path``; ``VOICEOVER_HOME`` is
redirected by an autouse fixture so no real user data directory is touched, and
no ``.env`` value, provider, ASR, FFmpeg, model, embedding, or network path is
used. The tests cover parser shape, the single-JSON-object/exit-code contract,
absent-database zero creation, bounded metadata reads, exact filters, label
disambiguation, dry-run zero writes, transactional idempotent import, missing
text/audio metadata, exact versus legacy-float money, corrupt/future database
fail-closed behavior, and the no-network guarantee.
"""

import hashlib
import json
import logging
import socket
import sqlite3
import sys
from pathlib import Path

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.history.database import MIGRATIONS, HistoryDatabase, Migration

_SLUG = "openai-gpt-4o-mini-tts"
_MODEL = "openai/gpt-4o-mini-tts"
_SCRIPT_HASH = "a" * 64


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point ``VOICEOVER_HOME`` at a temp path so no real data store is touched."""
    home = tmp_path / "voiceover-home"
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    return home


def _run_cli(monkeypatch, *argv):
    """Run ``cli.main`` in-process and return its numeric exit code."""
    monkeypatch.setattr(sys, "argv", ["voiceover-pipeline", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _decode_payload(out):
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _invoke(monkeypatch, capsys, *argv):
    """Run ``cli.main`` in-process and return ``(exit_code, json_payload, stdout)``."""
    code = _run_cli(monkeypatch, *argv)
    out = capsys.readouterr().out
    return code, _decode_payload(out), out


def _invoke_streams(monkeypatch, capsys, *argv):
    """Like :func:`_invoke` but also returns captured stderr for leak checks."""
    code = _run_cli(monkeypatch, *argv)
    captured = capsys.readouterr()
    return code, _decode_payload(captured.out), captured.out, captured.err


def _default_chunk(number=1, text="Привет, мир!", **overrides):
    chunk = {
        "status": "completed",
        "number": number,
        "id": f"chunk_{number:02d}",
        "file": f"chunk_{number:02d}.mp3",
        "duration_ms": 1200,
        "text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None,
        "cost": 0.1,
        "cost_currency": "RUB",
    }
    chunk.update(overrides)
    return {key: value for key, value in chunk.items() if value is not None}


def _write_legacy_run(
    source,
    *,
    run_id="prod",
    chunks=None,
    provider="polza-tts",
    model=_MODEL,
    voice="alloy",
    cost_currency="RUB",
    status="completed",
    write_manifest=True,
    missing_files=(),
    json_run_id=None,
):
    """Create a synthetic legacy run tree and return its run root.

    ``json_run_id`` lets a test declare a JSON ``run_id`` that differs from the
    directory name, which a real signed-URL run ID cannot match because a POSIX
    directory name cannot contain ``/``.
    """
    root = Path(source) / run_id
    label = run_id if json_run_id is None else json_run_id
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    entries = [_default_chunk()] if chunks is None else list(chunks)

    for entry in entries:
        file_name = entry.get("file")
        if (
            isinstance(file_name, str)
            and file_name
            and "/" not in file_name
            and "\\" not in file_name
            and file_name not in missing_files
        ):
            (chunks_dir / file_name).write_bytes(b"ID3chunk")

    full_mp3 = root / f"{run_id}-voiceover-{_SLUG}.mp3"
    full_mp3.write_bytes(b"ID3full")
    run_json_path = root / f"{run_id}-voiceover-{_SLUG}.json"
    chunks_json_path = chunks_dir / "chunks.json"

    state = {
        "artifact_type": "voiceover-run-state",
        "status": status,
        "run_id": label,
        "provider": provider,
        "model": model,
        "voice": voice,
        "script": str(root / "missing-script.md"),
        "script_hash": _SCRIPT_HASH,
        "chunk_count": len(entries),
        "completed_count": len(entries),
        "chunks": entries,
    }
    if cost_currency is not None:
        state["cost_currency"] = cost_currency
    (root / "run_state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    run_json = {
        "artifact_type": "voiceover-run",
        "run_id": label,
        "provider": provider,
        "model": model,
        "voice": voice,
        "full_mp3": str(full_mp3),
        "run_json": str(run_json_path),
        "chunks_json": str(chunks_json_path),
        "chunk_count": len(entries),
        "chunks": entries,
    }
    if cost_currency is not None:
        run_json["cost_currency"] = cost_currency
    run_json_path.write_text(json.dumps(run_json, ensure_ascii=False), encoding="utf-8")

    if write_manifest:
        manifest = {
            "artifact_type": "voiceover-production-bundle",
            "run_id": label,
            "full_mp3": str(full_mp3),
            "run_json": str(run_json_path),
            "chunks_json": str(chunks_json_path),
            "duration_ms": 1200,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

    chunks_json = {
        "artifact_type": "voiceover-chunks",
        "provider": provider,
        "model": model,
        "voice": voice,
        "chunk_count": len(entries),
        "chunks": entries,
    }
    if cost_currency is not None:
        chunks_json["cost_currency"] = cost_currency
    chunks_json_path.write_text(json.dumps(chunks_json, ensure_ascii=False), encoding="utf-8")
    return root


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _entries(root: Path) -> set[str]:
    return (
        {str(path.relative_to(root)) for path in sorted(root.rglob("*"))}
        if root.exists()
        else set()
    )


def _seed_external_database(database_path):
    """Create a valid history database with one run and return its UUID."""
    from voiceover_pipeline.history.repository import HistoryRepository

    database = HistoryDatabase(database_path)
    database.migrate()
    try:
        run = HistoryRepository(database).create_run(
            operation="tts", run_root="/tmp/run", status="completed"
        )
    finally:
        database.close()
    return run.run_uuid


def _seed_run(database_path, **overrides):
    """Insert one run (optionally with hostile fields) and return the record."""
    from voiceover_pipeline.history.repository import HistoryRepository

    database = HistoryDatabase(database_path)
    database.migrate()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            run = repository.create_run(
                operation=overrides.pop("operation", "tts"),
                run_root=overrides.pop("run_root", "/tmp/run"),
                status=overrides.pop("status", "completed"),
                **overrides,
            )
    finally:
        database.close()
    return run


def _rewrite_ledger_name(database_path, name):
    """Replace the stored migration-ledger name to simulate a tampered database."""
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("UPDATE schema_migrations SET name = ? WHERE version = 1", (name,))
        connection.commit()
    finally:
        connection.close()


# -- parser -------------------------------------------------------------------


def test_history_parser_places_json_at_end_of_leaf_command():
    parser = cli.build_parser()

    args = parser.parse_args(["history", "list", "--label", "prod", "--limit", "5", "--json"])

    assert args.command == "history"
    assert args.history_command == "list"
    assert args.label == "prod"
    assert args.limit == 5
    assert args.json_output is True


def test_history_parser_import_dry_run_and_show_id():
    parser = cli.build_parser()

    show = parser.parse_args(["history", "show", "abcdef", "--json"])
    assert show.history_command == "show"
    assert show.run == "abcdef"

    imp = parser.parse_args(["history", "import", "out", "--dry-run", "--json"])
    assert imp.history_command == "import"
    assert imp.source == "out"
    assert imp.dry_run is True


# -- absent database ----------------------------------------------------------


def test_history_list_absent_database_is_empty_and_creates_nothing(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 0
    assert payload["status"] == "success"
    assert payload["runs"] == []
    assert payload["count"] == 0
    assert payload["database"]["exists"] is False
    assert not _isolated_home.exists()


def test_history_show_missing_id_is_args_error_without_creation(
    monkeypatch, capsys, _isolated_home
):
    code, payload, _ = _invoke(monkeypatch, capsys, "history", "show", "does-not-exist", "--json")

    assert code == 2
    assert payload["status"] == "error"
    assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"
    assert not _isolated_home.exists()


def test_history_list_limit_out_of_range_is_args_error(monkeypatch, capsys):
    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "list", "--limit", "100000", "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_INVALID_LIMIT"


def test_history_list_negative_offset_is_args_error(monkeypatch, capsys):
    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--offset", "-1", "--json")

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_INVALID_OFFSET"


def test_history_list_argparse_error_is_single_json_object(monkeypatch, capsys):
    code, payload, out = _invoke(monkeypatch, capsys, "history", "list", "--limit", "abc", "--json")

    assert code == 2
    assert payload == {"status": "error", "error": "Invalid command-line arguments", "code": 2}
    assert out.strip().count("\n") == 0


def test_history_list_relative_home_is_args_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOICEOVER_HOME", "relative-home")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_HOME_INVALID"
    assert not (tmp_path / "relative-home").exists()


# -- dry-run import -----------------------------------------------------------


def test_history_import_dry_run_writes_nothing(tmp_path, monkeypatch, capsys, _isolated_home):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    before = _snapshot_tree(source)

    code, payload, out = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["status"] == "success"
    assert payload["dry_run"] is True
    assert payload["discovered_count"] == 1
    assert payload["importable_count"] == 1
    assert payload["already_imported_count"] == 0
    assert payload["database"]["exists"] is False
    assert not _isolated_home.exists()
    assert _snapshot_tree(source) == before
    assert "Привет" not in out


def test_history_import_dry_run_missing_source_is_args_error(monkeypatch, capsys, tmp_path):
    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(tmp_path / "nope"), "--dry-run", "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_SOURCE_NOT_FOUND"


def test_history_import_dry_run_unknown_database_is_not_conflict_free(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    (_isolated_home / "history.sqlite3").write_bytes(b"not a database at all")
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    before = _entries(_isolated_home)

    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["database"]["exists"] is True
    assert payload["database"]["readable"] is False
    assert "database_status_unknown" in payload["scan_conflicts"]
    assert payload["runs"][0]["already_imported"] is None
    assert _entries(_isolated_home) == before


def test_history_import_dry_run_with_existing_database_writes_nothing(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    before_home = _snapshot_tree(_isolated_home)
    before_source = _snapshot_tree(source)

    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["already_imported_count"] == 1
    assert payload["database"]["readable"] is True
    assert _snapshot_tree(_isolated_home) == before_home
    assert _snapshot_tree(source) == before_source


def test_history_import_dry_run_leaves_wal_and_shm_sidecars_untouched(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    (_isolated_home / "history.sqlite3-wal").write_bytes(b"wal-frames")
    (_isolated_home / "history.sqlite3-shm").write_bytes(b"shm-pages")
    before = _snapshot_tree(_isolated_home)

    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["database"]["readable"] is False
    assert "database_status_unknown" in payload["scan_conflicts"]
    assert payload["runs"][0]["already_imported"] is None
    assert _snapshot_tree(_isolated_home) == before


def test_history_import_insecure_home_is_output_error(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir()
    _isolated_home.chmod(0o755)
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    assert code == 50
    assert payload["details"]["error_code"] == "HISTORY_HOME_PERMISSION"
    assert not (_isolated_home / "history.sqlite3").exists()
    assert not (_isolated_home / "runs").exists()


def test_history_import_real_missing_source_is_args_error(monkeypatch, capsys, tmp_path):
    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(tmp_path / "nope"), "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_SOURCE_NOT_FOUND"


# -- real import + read -------------------------------------------------------


def test_history_import_then_list_and_show_are_metadata_only(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    secret = "СЕКРЕТНЫЙ_ТЕКСТ_СЦЕНАРИЯ"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=[_default_chunk(text=secret)])

    code, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    assert code == 0
    assert imported["imported_count"] == 1
    assert imported["skipped_count"] == 0
    assert imported["rejected_count"] == 0
    run_uuid = imported["runs"][0]["run_uuid"]
    assert (_isolated_home / "runs").is_dir()

    code, listed, out = _invoke(monkeypatch, capsys, "history", "list", "--json")
    assert code == 0
    assert listed["count"] == 1
    assert listed["runs"][0]["run_uuid"] == run_uuid
    assert listed["runs"][0]["user_label"] == "prod"
    assert listed["runs"][0]["operation"] == "tts"
    assert listed["runs"][0]["legacy_import"] is True
    assert "config_snapshot" not in listed["runs"][0]
    assert secret not in out

    code, detail, out = _invoke(monkeypatch, capsys, "history", "show", run_uuid, "--json")
    assert code == 0
    assert detail["run"]["run_uuid"] == run_uuid
    assert len(detail["parts"]) == 1
    assert len(detail["attempts"]) == 1
    assert detail["attempts"][0]["cost"]["amount"] == "0.1"
    assert detail["attempts"][0]["cost"]["source"] == "legacy_import"
    assert detail["attempts"][0]["cost"]["exact_available"] is False
    assert len(detail["text_sources"]) == 2
    assert all("has_content" in text_source for text_source in detail["text_sources"])
    # No prepared/source text, raw snapshot, secret, or signed URL leaks to stdout.
    assert secret not in out
    assert "prepared_text" not in detail["parts"][0]
    assert "config_snapshot" not in out
    assert "http" not in out


def test_history_show_resolves_user_label(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    code, detail, _ = _invoke(monkeypatch, capsys, "history", "show", "prod", "--json")

    assert code == 0
    assert detail["run"]["user_label"] == "prod"


def test_history_show_duplicate_label_across_roots_is_ambiguous(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _write_legacy_run(source / "nested", run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "show", "prod", "--json")

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_LABEL_AMBIGUOUS"
    assert payload["details"]["candidate_count"] == 2
    assert len(payload["details"]["candidates"]) == 2
    assert {candidate["run_root"] for candidate in payload["details"]["candidates"]} == {
        str(source / "prod"),
        str(source / "nested" / "prod"),
    }


def test_history_show_unknown_uuid_is_args_error(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "show", "00000000-0000-0000-0000-000000000000", "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"


def test_history_list_filters_are_exact(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _write_legacy_run(source, run_id="beta", chunks=[_default_chunk(number=1, cost=None)])
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    _, prod, _ = _invoke(monkeypatch, capsys, "history", "list", "--label", "prod", "--json")
    assert prod["count"] == 1
    assert prod["runs"][0]["user_label"] == "prod"

    _, none, _ = _invoke(monkeypatch, capsys, "history", "list", "--label", "missing", "--json")
    assert none["count"] == 0

    _, by_operation, _ = _invoke(
        monkeypatch, capsys, "history", "list", "--operation", "tts", "--json"
    )
    assert by_operation["count"] == 2

    _, by_status, _ = _invoke(
        monkeypatch, capsys, "history", "list", "--status", "completed", "--json"
    )
    assert by_status["count"] == 2

    _, limited, _ = _invoke(monkeypatch, capsys, "history", "list", "--limit", "1", "--json")
    assert limited["count"] == 1
    _, offset, _ = _invoke(
        monkeypatch, capsys, "history", "list", "--limit", "1", "--offset", "1", "--json"
    )
    assert offset["count"] == 1
    assert offset["runs"][0]["run_uuid"] != limited["runs"][0]["run_uuid"]


def test_history_import_is_idempotent_and_preserves_original_bytes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    before = _snapshot_tree(source)

    _, first, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    _, second, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    assert first["imported_count"] == 1
    assert second["imported_count"] == 0
    assert second["skipped_count"] == 1
    assert second["runs"][0]["run_uuid"] == first["runs"][0]["run_uuid"]
    assert _snapshot_tree(source) == before

    _, detail, _ = _invoke(
        monkeypatch, capsys, "history", "show", first["runs"][0]["run_uuid"], "--json"
    )
    assert len(detail["attempts"]) == 1
    assert len(detail["parts"]) == 1


def test_history_show_reports_missing_text_and_audio_metadata(tmp_path, monkeypatch, capsys):
    chunks = [_default_chunk(number=1, text=None, text_hash="b" * 64, file="chunk_01.mp3")]
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=chunks, missing_files=("chunk_01.mp3",))
    _, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    code, detail, out = _invoke(
        monkeypatch, capsys, "history", "show", imported["runs"][0]["run_uuid"], "--json"
    )

    assert code == 0
    part_texts = [
        text_source
        for text_source in detail["text_sources"]
        if text_source["part_uuid"] is not None
    ]
    assert part_texts[0]["has_content"] is False
    assert part_texts[0]["text_completeness"] == "incomplete"
    chunk_artifact = next(
        artifact for artifact in detail["artifacts"] if artifact["role"] == "chunk_audio"
    )
    assert chunk_artifact["availability"] == "missing"
    assert "prepared_text" not in detail["parts"][0]


def test_history_show_does_not_echo_stored_attempt_error(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import HistoryRepository

    _isolated_home.mkdir(mode=0o700)
    secret = "sk-live-SENTINEL0123456789"
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(operation="tts", run_root="/tmp/run", status="failed")
    repository.add_attempt(
        run.run_uuid,
        call_type="tts_chunk",
        status="failed",
        error=f"provider rejected the request with {secret}",
    )
    database.close()

    code, detail, out = _invoke(monkeypatch, capsys, "history", "show", run.run_uuid, "--json")

    assert code == 0
    assert detail["attempts"][0]["has_error"] is True
    assert "error" not in detail["attempts"][0]
    assert secret not in out


def test_history_show_preserves_exact_zero_null_and_legacy_cost(tmp_path, monkeypatch, capsys):
    exact_amount = "0.123456789012345678901234567"
    chunks = [
        _default_chunk(number=1, cost=0.0),
        _default_chunk(number=2, cost=None),
        _default_chunk(number=3, cost=None, cost_exact=exact_amount),
        _default_chunk(number=4, cost="0.1000"),
    ]
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=chunks)
    _, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    _, detail, _ = _invoke(
        monkeypatch, capsys, "history", "show", imported["runs"][0]["run_uuid"], "--json"
    )

    costs = {attempt["cost"]["amount"]: attempt["cost"] for attempt in detail["attempts"]}
    assert costs["0.0"]["exact_available"] is False
    assert costs["0.0"]["source"] == "legacy_import"
    assert costs[exact_amount]["exact_available"] is True
    assert costs["0.1000"]["exact_available"] is False
    assert costs["0.1000"]["raw"] == "0.1000"
    assert any(attempt["cost"]["amount"] is None for attempt in detail["attempts"])


# -- corrupt / future / read-only database ------------------------------------


def test_history_list_empty_database_is_empty_and_not_found(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    database_path.write_bytes(b"")
    before = _entries(_isolated_home)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")
    assert code == 0
    assert payload["count"] == 0
    assert payload["database"]["exists"] is True
    assert payload["database"]["schema_version"] is None

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "show", "prod", "--json")
    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"
    assert _entries(_isolated_home) == before


def test_history_list_reads_legacy_v1_database_read_only(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    run_uuid = "11111111-2222-3333-4444-555555555555"
    with HistoryDatabase(database_path, migrations=[MIGRATIONS[0]]) as database:
        database.migrate()
        database.connection.execute(
            "INSERT INTO runs (run_uuid, operation, status, run_root, created_at, updated_at) "
            "VALUES (?, 'tts', 'completed', '/tmp/run', 't', 't')",
            (run_uuid,),
        )
    before = _snapshot_tree(_isolated_home)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")
    assert code == 0
    assert payload["count"] == 1
    assert payload["database"]["schema_version"] == 1
    assert payload["runs"][0]["run_uuid"] == run_uuid
    assert "revision" not in payload["runs"][0]

    code, detail, _ = _invoke(monkeypatch, capsys, "history", "show", run_uuid, "--json")
    assert code == 0
    assert detail["run"]["run_uuid"] == run_uuid

    assert _snapshot_tree(_isolated_home) == before
    assert not (database_path.with_name(database_path.name + "-wal")).exists()
    assert not (database_path.with_name(database_path.name + "-shm")).exists()


def test_history_list_corrupt_database_is_provider_error(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    database_path.write_bytes(b"not a database at all")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_UNREADABLE"


def test_history_list_redacts_tampered_ledger_name(monkeypatch, capsys, _isolated_home):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    _seed_run(database_path)
    _rewrite_ledger_name(database_path, f"https://example.invalid/?signature={_SENTINEL}")
    before = _snapshot_tree(_isolated_home)

    code, payload, out, err = _invoke_streams(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_CHECKSUM_MISMATCH"
    assert payload["error"]
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out
    assert "signature=" not in err
    assert _snapshot_tree(_isolated_home) == before


def test_history_import_redacts_tampered_ledger_name_without_writes(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    _seed_run(database_path)
    _rewrite_ledger_name(database_path, f"https://example.invalid/?signature={_SENTINEL}")
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    before = _snapshot_tree(_isolated_home)

    code, payload, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "import", str(source), "--json"
    )

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_CHECKSUM_MISMATCH"
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert _snapshot_tree(_isolated_home) == before


def test_history_list_future_schema_is_provider_error(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    future = Migration(
        version=999,
        name="future_schema",
        statements=("CREATE TABLE future_only (id INTEGER PRIMARY KEY)",),
    )
    with HistoryDatabase(database_path, migrations=[*MIGRATIONS, future]) as database:
        database.migrate()
    before = _entries(_isolated_home)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_TOO_NEW"
    assert _entries(_isolated_home) == before


def test_history_list_sidecar_present_is_unreadable_without_writes(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    wal = _isolated_home / "history.sqlite3-wal"
    wal.write_bytes(b"uncheckpointed-frames")
    before = _entries(_isolated_home)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_UNREADABLE"
    assert _entries(_isolated_home) == before
    assert not (_isolated_home / "history.sqlite3-shm").exists()
    assert wal.read_bytes() == b"uncheckpointed-frames"


def test_history_import_future_schema_is_provider_error_without_writes(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    future = Migration(
        version=999,
        name="future_schema",
        statements=("CREATE TABLE future_only (id INTEGER PRIMARY KEY)",),
    )
    with HistoryDatabase(database_path, migrations=[*MIGRATIONS, future]) as database:
        database.migrate()
    before = _snapshot_tree(_isolated_home)
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_TOO_NEW"
    assert not (_isolated_home / "runs").exists()
    assert _snapshot_tree(_isolated_home) == before


# -- human output and no-network ---------------------------------------------


def test_history_human_output_is_readable(tmp_path, monkeypatch, capsys):
    code, _, out = _invoke(monkeypatch, capsys, "history", "list")
    assert code == 0
    assert "No history runs found" in out

    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _invoke(monkeypatch, capsys, "history", "import", str(source))
    _, _, out = _invoke(monkeypatch, capsys, "history", "list")
    assert "History runs (1)" in out
    assert "label=prod" in out


def test_history_import_and_preview_attempt_no_network(tmp_path, monkeypatch, capsys):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    def _forbid_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _forbid_network)
    monkeypatch.setattr(socket, "create_connection", _forbid_network)

    code, preview, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )
    assert code == 0
    assert preview["discovered_count"] == 1

    code, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    assert code == 0
    assert imported["imported_count"] == 1


def test_history_import_default_explicit_database_path_is_read_only_after_close(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    # A cleanly closed WAL database leaves no sidecar, so the read path can open
    # it without creating one.
    assert not (_isolated_home / "history.sqlite3-wal").exists()
    assert not (_isolated_home / "history.sqlite3-shm").exists()

    connection = sqlite3.connect(_isolated_home / "history.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    finally:
        connection.close()


# -- privacy: free-form metadata is never echoed -------------------------------

_SENTINEL = "SIGNATURE_SENTINEL_DO_NOT_ECHO"


def test_history_show_does_not_echo_signed_url_remote_id(monkeypatch, capsys, _isolated_home):
    from voiceover_pipeline.history.repository import HistoryRepository

    _isolated_home.mkdir(mode=0o700)
    signed_url = f"https://example.invalid/audio?signature={_SENTINEL}"
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(operation="tts", run_root="/tmp/run", status="completed")
    repository.add_attempt(
        run.run_uuid,
        call_type="tts_chunk",
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        account_alias="work",
        remote_id=signed_url,
        status="completed",
    )
    database.close()

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["attempts"][0]["remote_id"] is None
    assert detail["attempts"][0]["provider"] == "polza-tts"
    assert detail["attempts"][0]["model"] == "openai/gpt-4o-mini-tts"
    assert detail["attempts"][0]["account_alias"] == "work"
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert signed_url not in out
    assert "signature=" not in out


def test_history_show_drops_raw_api_key_shaped_metadata(monkeypatch, capsys, _isolated_home):
    from voiceover_pipeline.history.repository import HistoryRepository

    _isolated_home.mkdir(mode=0o700)
    raw_key = "sk-live-ABCdef0123456789"
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(operation="tts", run_root="/tmp/run", status="completed")
    repository.add_attempt(
        run.run_uuid,
        call_type="tts_chunk",
        provider=raw_key,
        remote_id=raw_key,
        status="completed",
    )
    database.close()

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["attempts"][0]["provider"] is None
    assert detail["attempts"][0]["remote_id"] is None
    assert raw_key not in out
    assert raw_key not in err


def test_history_show_sanitizes_freeform_provider_and_voice_fields(
    monkeypatch, capsys, _isolated_home
):
    from voiceover_pipeline.history.repository import HistoryRepository

    provider_url = f"https://provider.invalid/auth?token={_SENTINEL}"
    voice_url = f"https://voices.invalid/pick?signature={_SENTINEL}"
    account_url = f"https://account.invalid/?key={_SENTINEL}"
    _isolated_home.mkdir(mode=0o700)
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(operation="tts", run_root="/tmp/run", status="completed")
    repository.add_part(run.run_uuid, position=1, voice=voice_url)
    repository.add_attempt(
        run.run_uuid,
        call_type="tts_chunk",
        provider=provider_url,
        account_alias=account_url,
        status="completed",
    )
    database.close()

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["attempts"][0]["provider"] is None
    assert detail["attempts"][0]["account_alias"] is None
    assert detail["parts"][0]["voice"] is None
    assert _SENTINEL not in out
    assert _SENTINEL not in err


def test_history_show_redacts_signed_url_legacy_cost_currency(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    signed_currency = f"https://example.invalid/?signature={_SENTINEL}"
    chunks = [
        _default_chunk(number=1, cost=0.1, cost_currency=signed_currency),
        _default_chunk(number=2, cost=0.2, cost_currency="RUB"),
        _default_chunk(number=3, cost=0.3, cost_currency="USD"),
    ]
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=chunks)
    _, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    run_uuid = imported["runs"][0]["run_uuid"]

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run_uuid, "--json"
    )

    assert code == 0
    costs = {attempt["cost"]["amount"]: attempt["cost"] for attempt in detail["attempts"]}
    assert costs["0.1"]["currency"] is None
    assert costs["0.1"]["source"] == "legacy_import"
    assert costs["0.1"]["exact_available"] is False
    assert costs["0.2"]["currency"] == "RUB"
    assert costs["0.3"]["currency"] == "USD"
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out

    # Only the public projection is sanitized: the legacy amount and its stored
    # currency are preserved verbatim for a later exact read or export.
    connection = sqlite3.connect(_isolated_home / "history.sqlite3")
    try:
        stored = connection.execute(
            "SELECT cost_raw, cost_currency FROM attempts WHERE cost = '0.1'"
        ).fetchone()
    finally:
        connection.close()
    assert stored[0] == "0.1"
    assert stored[1] == signed_currency


def test_history_show_sanitizes_freeform_classification_and_cost_fields(
    monkeypatch, capsys, _isolated_home
):
    from voiceover_pipeline.history.repository import (
        AVAILABILITY_PRESENT,
        PATH_KIND_EXTERNAL_ABSOLUTE,
        Cost,
        HistoryRepository,
    )

    signed_value = f"https://example.invalid/audio?signature={_SENTINEL}"
    _isolated_home.mkdir(mode=0o700)
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(
        operation="tts", run_root="/tmp/run", status="completed", created_at=signed_value
    )
    part = repository.add_part(
        run.run_uuid, position=1, stage=signed_value, fingerprint=signed_value
    )
    repository.add_attempt(
        run.run_uuid,
        call_type=signed_value,
        part_uuid=part.part_uuid,
        status=signed_value,
        cost=Cost(amount="0.5", currency=signed_value, source=signed_value, raw=signed_value),
    )
    repository.add_artifact(
        run.run_uuid,
        role=signed_value,
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path="/tmp/run/audio.mp3",
        sha256=signed_value,
        availability=AVAILABILITY_PRESENT,
    )
    repository.add_text_source(
        run.run_uuid,
        kind=signed_value,
        origin=signed_value,
        language=signed_value,
        content_hash=signed_value,
    )
    database.close()

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["run"]["created_at"] is None
    assert detail["parts"][0]["stage"] is None
    assert detail["parts"][0]["fingerprint"] is None
    assert detail["attempts"][0]["call_type"] is None
    assert detail["attempts"][0]["status"] is None
    assert detail["attempts"][0]["cost"] == {
        "amount": "0.5",
        "currency": None,
        "source": None,
        "exact_available": False,
        "raw": None,
    }
    assert detail["artifacts"][0]["role"] is None
    assert detail["artifacts"][0]["sha256"] is None
    # A validated classification value is ordinary metadata and still prints.
    assert detail["artifacts"][0]["path_kind"] == PATH_KIND_EXTERNAL_ABSOLUTE
    assert detail["artifacts"][0]["availability"] == AVAILABILITY_PRESENT
    assert detail["text_sources"][0]["kind"] is None
    assert detail["text_sources"][0]["origin"] is None
    assert detail["text_sources"][0]["language"] is None
    assert detail["text_sources"][0]["content_hash"] is None
    assert _SENTINEL not in out
    assert _SENTINEL not in err


def test_history_show_preserves_iso_timestamps(monkeypatch, capsys, _isolated_home):
    from voiceover_pipeline.history.repository import HistoryRepository

    _isolated_home.mkdir(mode=0o700)
    database = HistoryDatabase(_isolated_home / "history.sqlite3")
    database.migrate()
    repository = HistoryRepository(database)
    run = repository.create_run(
        operation="tts",
        run_root="/tmp/run",
        status="completed",
        created_at="2026-01-02T03:04:05Z",
    )
    database.close()

    code, detail, _, _ = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["run"]["created_at"] == "2026-01-02T03:04:05Z"
    assert detail["run"]["updated_at"] == "2026-01-02T03:04:05Z"


def test_history_import_conflict_log_does_not_leak_signed_url_run_id(
    tmp_path, monkeypatch, capsys, caplog, _isolated_home
):
    signed_run_id = f"https://x.invalid/?sig={_SENTINEL}"
    source = tmp_path / "out"
    _write_legacy_run(
        source,
        run_id="prod",
        json_run_id=signed_run_id,
        chunks=[_default_chunk(file="../escape.mp3")],
    )

    with caplog.at_level(logging.WARNING, logger="voiceover_pipeline.history"):
        code, payload, out, err = _invoke_streams(
            monkeypatch, capsys, "history", "import", str(source), "--json"
        )

    assert code == 0
    assert payload["imported_count"] == 1
    assert payload["runs"][0]["conflicts"] == ["unsafe_chunk_path"]
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("history_conflict") for message in messages)
    assert all(_SENTINEL not in message for message in messages)
    assert all(signed_run_id not in message for message in messages)
    assert all("sig=" not in message for message in messages)
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "sig=" not in out
    assert "sig=" not in err


def test_history_import_dry_run_does_not_echo_signed_url_legacy_fields(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    signed_voice = f"https://legacy.invalid/voice?signature={_SENTINEL}"
    signed_provider = f"https://legacy.invalid/provider?key={_SENTINEL}"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", provider=signed_provider, voice=signed_voice)

    code, payload, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["discovered_count"] == 1
    assert payload["importable_count"] == 1
    preview = payload["runs"][0]
    assert preview["provider"] is None
    assert preview["voice"] is None
    assert preview["model"] == _MODEL
    assert preview["chunk_count"] == 1
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out


# -- symlinked database fail-closed --------------------------------------------


def test_history_read_rejects_symlinked_database_file_without_touching_target(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    external_dir = tmp_path / "external"
    external_dir.mkdir(mode=0o700)
    external_db = external_dir / "history.sqlite3"
    _seed_external_database(external_db)
    _isolated_home.mkdir(mode=0o700)
    (_isolated_home / "history.sqlite3").symlink_to(external_db)
    before = _snapshot_tree(external_dir)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_UNREADABLE"
    assert _snapshot_tree(external_dir) == before


def test_history_import_dry_run_reports_unknown_for_symlinked_database_file(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    external_dir = tmp_path / "external"
    external_dir.mkdir(mode=0o700)
    external_db = external_dir / "history.sqlite3"
    _seed_external_database(external_db)
    _isolated_home.mkdir(mode=0o700)
    (_isolated_home / "history.sqlite3").symlink_to(external_db)
    before = _snapshot_tree(external_dir)
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    code, payload, _ = _invoke(
        monkeypatch, capsys, "history", "import", str(source), "--dry-run", "--json"
    )

    assert code == 0
    assert payload["database"]["exists"] is True
    assert payload["database"]["readable"] is False
    assert "database_status_unknown" in payload["scan_conflicts"]
    assert payload["runs"][0]["already_imported"] is None
    assert _snapshot_tree(external_dir) == before


def test_history_import_rejects_symlinked_database_file_without_writing_target(
    tmp_path, monkeypatch, capsys, _isolated_home
):
    external_dir = tmp_path / "external"
    external_dir.mkdir(mode=0o700)
    external_db = external_dir / "history.sqlite3"
    _seed_external_database(external_db)
    _isolated_home.mkdir(mode=0o700)
    (_isolated_home / "history.sqlite3").symlink_to(external_db)
    before = _snapshot_tree(external_dir)
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_UNREADABLE"
    assert _snapshot_tree(external_dir) == before
    assert not (_isolated_home / "runs").exists()


def test_history_read_rejects_symlinked_home_directory_without_touching_target(
    tmp_path, monkeypatch, capsys
):
    external_home = tmp_path / "external-home"
    external_home.mkdir(mode=0o700)
    _seed_external_database(external_home / "history.sqlite3")
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(external_home, target_is_directory=True)
    monkeypatch.setenv("VOICEOVER_HOME", str(linked_home))
    before = _snapshot_tree(external_home)

    code, payload, _ = _invoke(monkeypatch, capsys, "history", "list", "--json")

    assert code == 30
    assert payload["details"]["error_code"] == "HISTORY_DATABASE_UNREADABLE"
    assert _snapshot_tree(external_home) == before


# -- UUID-shaped legacy labels -------------------------------------------------


def test_history_show_resolves_uuid_shaped_legacy_label(tmp_path, monkeypatch, capsys):
    legacy_label = "11111111-2222-3333-4444-555555555555"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id=legacy_label)
    _, imported, _ = _invoke(monkeypatch, capsys, "history", "import", str(source), "--json")
    internal_uuid = imported["runs"][0]["run_uuid"]
    assert internal_uuid != legacy_label

    code, detail, _ = _invoke(monkeypatch, capsys, "history", "show", legacy_label, "--json")

    assert code == 0
    assert detail["run"]["run_uuid"] == internal_uuid
    assert detail["run"]["user_label"] == legacy_label


# -- privacy: stored labels/paths are never echoed as a signed URL --------------

_SIGNED_LABEL = f"https://example.invalid/audio?signature={_SENTINEL}"


def test_history_list_redacts_signed_url_user_label(monkeypatch, capsys, _isolated_home):
    _isolated_home.mkdir(mode=0o700)
    run = _seed_run(_isolated_home / "history.sqlite3", user_label=_SIGNED_LABEL)

    code, payload, out, err = _invoke_streams(monkeypatch, capsys, "history", "list", "--json")

    assert code == 0
    assert payload["count"] == 1
    assert payload["runs"][0]["run_uuid"] == run.run_uuid
    assert payload["runs"][0]["user_label"] is None
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic ZmFrZTpmYWtl",
        "Authorization: Digest ZmFrZTpmYWtl",
        "Proxy-Authorization: Custom ZmFrZTpmYWtl",
    ],
)
def test_history_list_redacts_authorization_header_user_label(
    monkeypatch, capsys, _isolated_home, header
):
    _isolated_home.mkdir(mode=0o700)
    run = _seed_run(_isolated_home / "history.sqlite3", user_label=header)

    code, payload, out, err = _invoke_streams(monkeypatch, capsys, "history", "list", "--json")

    assert code == 0
    assert payload["runs"][0]["run_uuid"] == run.run_uuid
    assert payload["runs"][0]["user_label"] is None
    assert "ZmFrZTpmYWtl" not in out  # synthetic "fake:fake", not a real credential
    assert "ZmFrZTpmYWtl" not in err
    assert header not in out


def test_history_show_redacts_signed_url_user_label(monkeypatch, capsys, _isolated_home):
    _isolated_home.mkdir(mode=0o700)
    run = _seed_run(_isolated_home / "history.sqlite3", user_label=_SIGNED_LABEL)

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["run"]["run_uuid"] == run.run_uuid
    assert detail["run"]["user_label"] is None
    assert _SENTINEL not in out
    assert _SENTINEL not in err


def test_history_show_redacts_signed_url_run_root_and_operation(
    monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    run = _seed_run(
        _isolated_home / "history.sqlite3",
        operation=_SIGNED_LABEL,
        status=_SIGNED_LABEL,
        run_root=_SIGNED_LABEL,
        legacy_source_root=_SIGNED_LABEL,
    )

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", run.run_uuid, "--json"
    )

    assert code == 0
    assert detail["run"]["operation"] is None
    assert detail["run"]["status"] is None
    assert detail["run"]["run_root"] is None
    assert detail["run"]["legacy_source_root"] is None
    assert detail["run"]["legacy_import"] is True
    assert _SENTINEL not in out
    assert _SENTINEL not in err


def test_history_list_and_show_preserve_human_label_with_spaces(
    monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    run = _seed_run(_isolated_home / "history.sqlite3", user_label="production run")

    code, listed, out, err = _invoke_streams(monkeypatch, capsys, "history", "list", "--json")
    assert code == 0
    assert listed["runs"][0]["user_label"] == "production run"

    code, detail, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", "production run", "--json"
    )
    assert code == 0
    assert detail["run"]["run_uuid"] == run.run_uuid
    assert detail["run"]["user_label"] == "production run"


def test_history_show_ambiguous_candidates_redact_signed_url_labels(
    monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    database_path = _isolated_home / "history.sqlite3"
    _seed_run(database_path, user_label=_SIGNED_LABEL, run_root="/tmp/run-a")
    _seed_run(database_path, user_label=_SIGNED_LABEL, run_root="/tmp/run-b")

    code, payload, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", _SIGNED_LABEL, "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_LABEL_AMBIGUOUS"
    assert payload["details"]["candidate_count"] == 2
    candidates = payload["details"]["candidates"]
    assert len(candidates) == 2
    assert all(candidate["user_label"] is None for candidate in candidates)
    assert {candidate["run_root"] for candidate in candidates} == {"/tmp/run-a", "/tmp/run-b"}
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out


def test_history_show_not_found_does_not_echo_signed_url_identifier(
    monkeypatch, capsys, _isolated_home
):
    _isolated_home.mkdir(mode=0o700)
    _seed_run(_isolated_home / "history.sqlite3", user_label="prod")

    code, payload, out, err = _invoke_streams(
        monkeypatch, capsys, "history", "show", _SIGNED_LABEL, "--json"
    )

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"
    assert _SENTINEL not in out
    assert _SENTINEL not in err
    assert "signature=" not in out
