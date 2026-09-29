"""Contract tests for the safe offline legacy-run importer.

Every fixture here is synthetic and lives under ``tmp_path``; no ``.env`` value,
user output directory, provider, ASR, FFmpeg, model, embedding, or network path
is touched. The tests cover the dry-run zero-write guarantee, idempotent and
multi-root import, malformed/symlinked/traversing inputs, incomplete text and
missing files, exact versus legacy-float money, rollback with retry, redacted
logging, future-schema rejection, and post-import metadata queries.
"""

import hashlib
import json
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from voiceover_pipeline.history.database import (
    MIGRATIONS,
    HistoryDatabase,
    Migration,
    SchemaVersionTooNewError,
)
from voiceover_pipeline.history.legacy_import import (
    import_legacy_runs,
    preview_legacy_import,
)
from voiceover_pipeline.history.paths import HistoryHomePermissionError, HistoryPathsError
from voiceover_pipeline.history.repository import (
    AVAILABILITY_MISSING,
    COST_SOURCE_LEGACY_FLOAT,
    TEXT_COMPLETENESS_COMPLETE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
)

_SLUG = "openai-gpt-4o-mini-tts"
_MODEL = "openai/gpt-4o-mini-tts"
_SCRIPT_HASH = "a" * 64


@contextmanager
def _repository(database_path):
    database = HistoryDatabase(database_path)
    database.migrate()
    try:
        yield HistoryRepository(database)
    finally:
        database.close()


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
    cost_total=None,
    cost_total_exact=None,
    script_path=None,
    style_prompt=None,
    state_status="completed",
    write_state=True,
    write_manifest=True,
    write_run_json=True,
    missing_files=(),
):
    """Create a synthetic legacy run tree and return its run root."""
    root = Path(source) / run_id
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

    if write_state:
        state = {
            "artifact_type": "voiceover-run-state",
            "status": state_status,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "voice": voice,
            "script": str(script_path if script_path is not None else root / "missing-script.md"),
            "script_hash": _SCRIPT_HASH,
            "chunk_count": len(entries),
            "completed_count": len(entries),
            "chunks": entries,
        }
        if cost_currency is not None:
            state["cost_currency"] = cost_currency
        if style_prompt is not None:
            state["style_prompt"] = style_prompt
        (root / "run_state.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )

    if write_run_json:
        run_json = {
            "artifact_type": "voiceover-run",
            "run_id": run_id,
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
        if cost_total is not None:
            run_json["cost_total"] = cost_total
        if cost_total_exact is not None:
            run_json["cost_total_exact"] = cost_total_exact
        if style_prompt is not None:
            run_json["style_prompt"] = style_prompt
        run_json_path.write_text(json.dumps(run_json, ensure_ascii=False), encoding="utf-8")

    if write_manifest:
        manifest = {
            "artifact_type": "voiceover-production-bundle",
            "run_id": run_id,
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
    if cost_total is not None:
        chunks_json["cost_total"] = cost_total
    chunks_json_path.write_text(json.dumps(chunks_json, ensure_ascii=False), encoding="utf-8")
    return root


def test_preview_creates_no_home_database_or_directory(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "home" / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)

    assert not database_path.exists()
    assert not database_path.parent.exists()
    assert preview.database_exists is False
    assert preview.database_readable is True
    assert preview.discovered_count == 1
    assert preview.importable_count == 1
    assert preview.runs[0].already_imported is False
    assert preview.runs[0].importable is True
    assert preview.runs[0].conflicts == ()


def test_preview_leaves_existing_database_and_sidecars_untouched(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "home" / "history.sqlite3"
    import_legacy_runs(source, database_path=database_path)
    before = {path.name: path.read_bytes() for path in database_path.parent.iterdir()}

    preview = preview_legacy_import(source, database_path=database_path)

    after = {path.name: path.read_bytes() for path in database_path.parent.iterdir()}
    assert after == before
    assert preview.runs[0].already_imported is True


def test_preview_omits_prepared_text(tmp_path):
    text = "СЕКРЕТНЫЙ_ТЕКСТ_СЦЕНАРИЯ"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=[_default_chunk(text=text)])
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.runs[0].chunks_with_text == 1
    assert text not in repr(preview)


def test_preview_reports_missing_text_and_files(tmp_path):
    chunks = [
        _default_chunk(number=1, text=None, text_hash="b" * 64, file="chunk_01.mp3"),
        _default_chunk(number=2, file="chunk_02.mp3"),
    ]
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=chunks, missing_files=("chunk_01.mp3",))
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.runs[0].chunks_missing_text == 1
    assert preview.runs[0].chunks_missing_audio == 1
    assert preview.missing_text_count == 1
    assert preview.missing_audio_count == 1


def test_preview_reports_unknown_when_database_cannot_be_read(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"
    database_path.write_bytes(b"not a database")

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.database_exists is True
    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None
    assert "database_status_unknown" in preview.scan_conflicts


def test_preview_reports_unknown_when_nonempty_wal_exists(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"
    import_legacy_runs(source, database_path=database_path)
    database_path.with_name(database_path.name + "-wal").write_bytes(b"uncheckedpointed")

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None
    assert "database_status_unknown" in preview.scan_conflicts


def test_preview_reports_unknown_when_empty_wal_exists(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"
    import_legacy_runs(source, database_path=database_path)
    database_path.with_name(database_path.name + "-wal").write_bytes(b"")

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None


def test_preview_leaves_entries_and_bytes_untouched_with_existing_wal(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "home" / "history.sqlite3"
    import_legacy_runs(source, database_path=database_path)
    database_path.with_name(database_path.name + "-wal").write_bytes(b"wal-frames")
    database_path.with_name(database_path.name + "-shm").write_bytes(b"shm-pages")
    before = {path.name: path.read_bytes() for path in database_path.parent.iterdir()}

    preview = preview_legacy_import(source, database_path=database_path)

    after = {path.name: path.read_bytes() for path in database_path.parent.iterdir()}
    assert after == before
    assert set(after) == {"history.sqlite3", "history.sqlite3-wal", "history.sqlite3-shm"}
    assert preview.database_exists is True
    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None


def test_repeated_import_of_same_root_does_not_duplicate(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"

    first = import_legacy_runs(source, database_path=database_path)
    second = import_legacy_runs(source, database_path=database_path)

    assert first.imported_count == 1
    assert second.imported_count == 0
    assert second.skipped_count == 1
    assert second.runs[0].run_uuid == first.runs[0].run_uuid

    with _repository(database_path) as repository:
        assert len(repository.list_runs()) == 1
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert len(repository.get_parts(run.run_uuid)) == 1
        assert len(repository.get_attempts(run.run_uuid)) == 1


def test_same_run_id_in_two_roots_gets_distinct_uuids(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    _write_legacy_run(source / "nested", run_id="prod")
    database_path = tmp_path / "history.sqlite3"

    result = import_legacy_runs(source, database_path=database_path)

    assert result.imported_count == 2
    assert len({run.run_uuid for run in result.runs}) == 2
    with _repository(database_path) as repository:
        candidates = repository.find_runs_by_label("prod")
        assert len(candidates) == 2
        assert {candidate.run_root for candidate in candidates} == {
            str(source / "prod"),
            str(source / "nested" / "prod"),
        }


def test_incomplete_text_and_missing_chunk_file(tmp_path):
    chunks = [_default_chunk(number=1, text=None, text_hash="b" * 64, file="chunk_01.mp3")]
    source = tmp_path / "out"
    run_root = _write_legacy_run(
        source, run_id="prod", chunks=chunks, missing_files=("chunk_01.mp3",)
    )
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        part = repository.get_parts(run.run_uuid)[0]
        assert part.prepared_text is None
        assert part.fingerprint == "b" * 64

        chunk_artifact = next(
            artifact
            for artifact in repository.get_artifacts(run.run_uuid)
            if artifact.role == "chunk_audio"
        )
        assert chunk_artifact.availability == AVAILABILITY_MISSING

        part_text_sources = [
            source_record
            for source_record in repository.get_text_sources(run.run_uuid)
            if source_record.part_uuid == part.part_uuid
        ]
        assert part_text_sources[0].content is None
        assert part_text_sources[0].content_hash == "b" * 64
        assert part_text_sources[0].text_completeness == TEXT_COMPLETENESS_INCOMPLETE


def test_exact_cost_string_is_preserved_with_legacy_provenance(tmp_path):
    amount = "0.123456789012345678901234567"
    chunks = [_default_chunk(number=1, cost=None, cost_exact=amount)]
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        attempt = repository.get_attempts(run.run_uuid)[0]
        assert attempt.cost.amount == amount
        assert attempt.cost.source == COST_SOURCE_LEGACY_FLOAT
        assert attempt.cost.exact_available is True
        assert attempt.cost.currency == "RUB"


def test_legacy_numeric_lexeme_is_preserved_without_binary_float(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"
    state_path = run_root / "run_state.json"
    state_path.write_text(
        state_path.read_text(encoding="utf-8").replace('"cost": 0.1', '"cost": 0.1000'),
        encoding="utf-8",
    )

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        attempt = repository.get_attempts(run.run_uuid)[0]
        assert attempt.cost.amount == "0.1000"
        assert attempt.cost.raw == "0.1000"
        assert attempt.cost.exact_available is False
        assert attempt.cost.source == COST_SOURCE_LEGACY_FLOAT


def test_zero_cost_is_kept_and_missing_cost_is_unknown(tmp_path):
    chunks = [
        _default_chunk(number=1, cost=0.0),
        _default_chunk(number=2, cost=None),
    ]
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        attempts = {attempt.part_uuid: attempt for attempt in repository.get_attempts(run.run_uuid)}
        parts = repository.get_parts(run.run_uuid)
        assert attempts[parts[0].part_uuid].cost.amount == "0.0"
        assert attempts[parts[0].part_uuid].cost.exact_available is False
        assert attempts[parts[1].part_uuid].cost.amount is None


def test_currencies_are_kept_separate(tmp_path):
    chunks = [
        _default_chunk(number=1, cost=1.5, cost_currency="RUB"),
        _default_chunk(number=2, cost=2.5, cost_currency="USD"),
    ]
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        currencies = {
            attempt.cost.currency: attempt.cost.amount
            for attempt in repository.get_attempts(run.run_uuid)
        }
        assert currencies == {"RUB": "1.5", "USD": "2.5"}


def test_run_total_is_imported_once_when_no_per_chunk_cost(tmp_path):
    chunks = [_default_chunk(number=1, cost=None), _default_chunk(number=2, cost=None)]
    source = tmp_path / "out"
    run_root = _write_legacy_run(
        source, run_id="prod", chunks=chunks, cost_total=0.5, cost_total_exact="0.5000"
    )
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        attempts = repository.get_attempts(run.run_uuid)
        assert len(attempts) == 1
        assert attempts[0].call_type == "tts_run_total"
        assert attempts[0].cost.amount == "0.5000"
        assert attempts[0].cost.exact_available is True


def test_per_chunk_cost_suppresses_run_total(tmp_path):
    chunks = [_default_chunk(number=1, cost=0.1)]
    source = tmp_path / "out"
    run_root = _write_legacy_run(
        source, run_id="prod", chunks=chunks, cost_total=0.1, cost_total_exact="0.1"
    )
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        attempts = repository.get_attempts(run.run_uuid)
        assert len(attempts) == 1
        assert attempts[0].call_type == "tts_chunk"


def test_failed_import_rolls_back_entirely_and_retry_completes(tmp_path, monkeypatch):
    chunks = [_default_chunk(number=1), _default_chunk(number=2)]
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    database_path = tmp_path / "history.sqlite3"

    original_add_part = HistoryRepository.add_part
    calls = {"count": 0}

    def flaky_add_part(self, run_uuid, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("injected failure")
        return original_add_part(self, run_uuid, **kwargs)

    monkeypatch.setattr(HistoryRepository, "add_part", flaky_add_part)
    with pytest.raises(RuntimeError):
        import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        assert repository.list_runs() == []

    monkeypatch.undo()
    result = import_legacy_runs(source, database_path=database_path)

    assert result.imported_count == 1
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert len(repository.get_parts(run.run_uuid)) == 2
        assert len(repository.get_attempts(run.run_uuid)) == 2
        assert len(repository.get_text_sources(run.run_uuid)) == 3


def test_import_events_do_not_leak_prepared_text(tmp_path, caplog):
    text = "СОВЕРШЕННО_СЕКРЕТНЫЙ_ТЕКСТ"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod", chunks=[_default_chunk(text=text)])
    database_path = tmp_path / "history.sqlite3"

    with caplog.at_level(logging.INFO, logger="voiceover_pipeline.history"):
        import_legacy_runs(source, database_path=database_path)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("history_imported run_uuid=") for message in messages)
    assert all(text not in message for message in messages)


def test_conflict_event_is_logged_without_content(tmp_path, caplog):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    (run_root / "run_state.json").write_text("{not json", encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    with caplog.at_level(logging.INFO, logger="voiceover_pipeline.history"):
        result = import_legacy_runs(source, database_path=database_path)

    assert result.rejected_count == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("history_conflict") for message in messages)
    assert any("malformed_json" in message for message in messages)


def test_malformed_identity_fails_closed_without_writes(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    (run_root / "run_state.json").write_text("{oops", encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    result = import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].importable is False
    assert "run_state_malformed_json" in preview.runs[0].conflicts
    assert result.rejected_count == 1
    assert result.runs[0].run_uuid is None
    with _repository(database_path) as repository:
        assert repository.list_runs() == []


def test_symlinked_identity_is_not_followed(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps({"run_id": "hijack", "chunks": [{"status": "completed", "number": 1}]}),
        encoding="utf-8",
    )
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    state_path = run_root / "run_state.json"
    state_path.unlink()
    state_path.symlink_to(outside)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    result = import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].importable is False
    assert "run_state_symlinked_json" in preview.runs[0].conflicts
    assert result.rejected_count == 1
    with _repository(database_path) as repository:
        assert repository.list_runs() == []


def test_chunk_path_traversal_is_rejected(tmp_path):
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"secret-outside")
    source = tmp_path / "out"
    chunks = [_default_chunk(number=1, file="../outside.mp3")]
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert "unsafe_chunk_path" in preview.runs[0].conflicts
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        roles = {artifact.role for artifact in repository.get_artifacts(run.run_uuid)}
        assert "chunk_audio" not in roles
    assert outside.read_bytes() == b"secret-outside"


def test_script_inside_source_is_imported_outside_is_incomplete(tmp_path):
    inside_source = tmp_path / "out-inside"
    inside_root = inside_source / "inside"
    _write_legacy_run(inside_source, run_id="inside", script_path=inside_root / "script.md")
    (inside_root / "script.md").write_text("Полный сценарий", encoding="utf-8")

    outside_script = tmp_path / "in" / "script.md"
    outside_script.parent.mkdir(parents=True, exist_ok=True)
    outside_script.write_text("Внешний сценарий", encoding="utf-8")
    outside_source = tmp_path / "out-outside"
    _write_legacy_run(outside_source, run_id="outside", script_path=outside_script)

    database_path = tmp_path / "history.sqlite3"

    inside_preview = preview_legacy_import(inside_source, database_path=database_path)
    outside_preview = preview_legacy_import(outside_source, database_path=database_path)
    assert inside_preview.runs[0].script_text_available is True
    assert outside_preview.runs[0].script_text_available is False

    import_legacy_runs(inside_source, database_path=database_path)
    import_legacy_runs(outside_source, database_path=database_path)

    with _repository(database_path) as repository:
        inside_run = repository.find_run_by_legacy_source_root(str(inside_root))
        outside_run = repository.find_run_by_legacy_source_root(str(outside_source / "outside"))
        inside_texts = [
            source_record
            for source_record in repository.get_text_sources(inside_run.run_uuid)
            if source_record.kind == TEXT_KIND_TTS_SCRIPT and source_record.part_uuid is None
        ]
        outside_texts = [
            source_record
            for source_record in repository.get_text_sources(outside_run.run_uuid)
            if source_record.kind == TEXT_KIND_TTS_SCRIPT and source_record.part_uuid is None
        ]
        assert inside_texts[0].content == "Полный сценарий"
        assert inside_texts[0].text_completeness == TEXT_COMPLETENESS_COMPLETE
        assert outside_texts[0].content is None
        assert outside_texts[0].content_hash == _SCRIPT_HASH
        assert outside_texts[0].text_completeness == TEXT_COMPLETENESS_INCOMPLETE


def test_hidden_script_is_not_read_and_stays_incomplete(tmp_path):
    source = tmp_path / "out"
    run_root = source / "prod"
    env_file = run_root / ".env"
    _write_legacy_run(source, run_id="prod", script_path=env_file)
    env_file.write_text("POLZA_API_KEY=sk-sentinel-value", encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].script_text_available is False
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert "sk-sentinel-value" not in json.dumps(run.config_snapshot)
        run_level_scripts = [
            text_source
            for text_source in repository.get_text_sources(run.run_uuid)
            if text_source.kind == TEXT_KIND_TTS_SCRIPT and text_source.part_uuid is None
        ]
        assert run_level_scripts[0].content is None
        assert run_level_scripts[0].content_hash == _SCRIPT_HASH
        assert run_level_scripts[0].text_completeness == TEXT_COMPLETENESS_INCOMPLETE


def test_style_prompt_is_preserved_as_direction(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", style_prompt="Спокойный голос")
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        directions = [
            source_record
            for source_record in repository.get_text_sources(run.run_uuid)
            if source_record.kind == TEXT_KIND_TTS_DIRECTION
        ]
        assert directions[0].content == "Спокойный голос"
        assert directions[0].origin == "legacy_import"


def test_future_schema_is_rejected_without_modification(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"
    future = Migration(
        version=999,
        name="future_schema",
        statements=("CREATE TABLE future_only (id INTEGER PRIMARY KEY)",),
    )
    with HistoryDatabase(database_path, migrations=[*MIGRATIONS, future]) as database:
        database.migrate()
    before = database_path.read_bytes()

    with pytest.raises(SchemaVersionTooNewError):
        import_legacy_runs(source, database_path=database_path)

    assert database_path.read_bytes() == before
    with HistoryDatabase(database_path, migrations=[*MIGRATIONS, future]) as database:
        database.migrate()
        assert database.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_imported_run_is_queryable_by_metadata(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"

    result = import_legacy_runs(source, database_path=database_path)
    run_uuid = result.runs[0].run_uuid

    with _repository(database_path) as repository:
        assert [run.run_uuid for run in repository.list_runs(operation="tts")] == [run_uuid]
        assert [run.run_uuid for run in repository.find_runs_by_label("prod")] == [run_uuid]
        assert [run.run_uuid for run in repository.find_runs_by_root(str(run_root))] == [run_uuid]
        legacy = repository.find_run_by_legacy_source_root(str(run_root))
        assert legacy.run_uuid == run_uuid
        assert legacy.user_label == "prod"
        roles = {artifact.role for artifact in repository.get_artifacts(run_uuid)}
        assert {"chunk_audio", "main_audio", "manifest", "run_state", "chunks_manifest"} <= roles


def test_preview_with_default_home_creates_nothing(tmp_path, monkeypatch):
    home = tmp_path / "voiceover-home"
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    preview = preview_legacy_import(source)

    assert preview.database_path == str(home / "history.sqlite3")
    assert preview.database_exists is False
    assert not home.exists()


def test_import_with_default_home_creates_private_layout(tmp_path, monkeypatch):
    home = tmp_path / "voiceover-home"
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    result = import_legacy_runs(source)

    assert result.imported_count == 1
    assert (home / "history.sqlite3").is_file()
    assert (home / "runs").is_dir()
    assert (home / "backups").is_dir()
    assert (home / "logs").is_dir()


def test_manifest_only_run_still_imports(tmp_path):
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", write_state=False)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    result = import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].importable is True
    assert result.imported_count == 1
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert len(repository.get_parts(run.run_uuid)) == 1


def test_import_and_preview_attempt_no_network_access(tmp_path, monkeypatch):
    import socket

    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "history.sqlite3"

    def _forbid_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _forbid_network)
    monkeypatch.setattr(socket, "create_connection", _forbid_network)

    preview = preview_legacy_import(source, database_path=database_path)
    result = import_legacy_runs(source, database_path=database_path)

    assert preview.discovered_count == 1
    assert result.imported_count == 1


def test_config_snapshot_is_whitelisted_and_redacted(tmp_path):
    source = tmp_path / "out"
    secret = "sk-live-SENTINEL0123456789"
    run_root = _write_legacy_run(source, run_id="prod")
    state_path = run_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["api_key"] = secret
    state["download_url"] = f"https://cdn.example.invalid/a.mp3?X-Amz-Signature={secret}"
    state["unsupported_object"] = {"nested": {"api_key": secret}}
    state["pricing_snapshot"] = {
        "api_key": secret,
        "source_url": f"https://cdn.example.invalid/price?token={secret}",
        "audio_per_million": 5.0,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        snapshot = run.config_snapshot
        assert snapshot["provider"] == "polza-tts"
        assert snapshot["run_id"] == "prod"
        assert "api_key" not in snapshot
        assert "unsupported_object" not in snapshot
        assert snapshot["pricing_snapshot"]["audio_per_million"] == "5.0"
        assert "api_key" not in snapshot["pricing_snapshot"]
        assert secret not in json.dumps(snapshot)


def test_dialogue_receipt_is_preserved_and_transcript_is_not_imported(tmp_path):
    receipt = {
        "artifact_type": "voiceover-tts-quality-receipt",
        "passed": True,
        "audio_sha256": "c" * 64,
    }
    transcript = "секретная расшифровка"
    chunks = [_default_chunk(number=1, text=None, text_hash="b" * 64, transcript=transcript)]
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", chunks=chunks)
    state_path = run_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["tts_quality"] = receipt
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert run.config_snapshot["tts_quality"] == receipt
        assert transcript not in json.dumps(run.config_snapshot)
        for part in repository.get_parts(run.run_uuid):
            assert part.prepared_text != transcript
        for text_source in repository.get_text_sources(run.run_uuid):
            assert text_source.content != transcript


def _run_level_script_sources(repository, run_uuid):
    return [
        source_record
        for source_record in repository.get_text_sources(run_uuid)
        if source_record.kind == TEXT_KIND_TTS_SCRIPT and source_record.part_uuid is None
    ]


def test_preview_accepts_fresh_empty_database(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "home" / "history.sqlite3"
    database_path.parent.mkdir()
    database_path.write_bytes(b"")

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.database_exists is True
    assert preview.database_readable is True
    assert preview.runs[0].already_imported is False
    assert "database_status_unknown" not in preview.scan_conflicts
    assert sorted(path.name for path in database_path.parent.iterdir()) == ["history.sqlite3"]


def test_preview_reports_unknown_for_foreign_ledgerless_database(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_path = tmp_path / "foreign.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE unrelated_data (id INTEGER PRIMARY KEY)")
        raw.commit()
    finally:
        raw.close()

    preview = preview_legacy_import(source, database_path=database_path)

    assert preview.database_exists is True
    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None
    assert "database_status_unknown" in preview.scan_conflicts


def test_preview_reports_unknown_for_future_schema_database(tmp_path):
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")
    database_parent = tmp_path / "home"
    database_parent.mkdir()
    database_path = database_parent / "future.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE future_only (id INTEGER PRIMARY KEY)")
        raw.execute("PRAGMA user_version = 999")
        raw.commit()
    finally:
        raw.close()
    before = {path.name: path.read_bytes() for path in database_parent.iterdir()}

    preview = preview_legacy_import(source, database_path=database_path)

    after = {path.name: path.read_bytes() for path in database_parent.iterdir()}
    assert after == before
    assert preview.database_readable is False
    assert preview.runs[0].already_imported is None
    assert "database_status_unknown" in preview.scan_conflicts


def test_import_future_schema_creates_no_home_layout(tmp_path, monkeypatch):
    home = tmp_path / "voiceover-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    database_path = home / "history.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE future_only (id INTEGER PRIMARY KEY)")
        raw.execute("PRAGMA user_version = 999")
        raw.commit()
    finally:
        raw.close()
    before = {path.name for path in home.iterdir()}
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(SchemaVersionTooNewError):
        import_legacy_runs(source)

    assert {path.name for path in home.iterdir()} == before
    assert not (home / "runs").exists()
    assert not (home / "backups").exists()
    assert not (home / "logs").exists()


def test_hidden_ancestor_script_is_not_read(tmp_path):
    secret = "SYNTHETIC_HIDDEN_ANCESTOR_SECRET"
    source = tmp_path / "out"
    run_root = source / "prod"
    hidden_dir = run_root / ".private"
    _write_legacy_run(source, run_id="prod", script_path=hidden_dir / "credentials.txt")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (hidden_dir / "credentials.txt").write_text(secret, encoding="utf-8")
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].script_text_available is False
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        assert secret not in json.dumps(run.config_snapshot)
        run_sources = _run_level_script_sources(repository, run.run_uuid)
        assert run_sources[0].content is None
        assert run_sources[0].text_completeness == TEXT_COMPLETENESS_INCOMPLETE
        for source_record in repository.get_text_sources(run.run_uuid):
            assert source_record.content != secret


def test_symlink_ancestor_script_is_not_read(tmp_path):
    secret = "SYNTHETIC_SYMLINK_ANCESTOR_SECRET"
    source = tmp_path / "out"
    run_root = source / "prod"
    target = source / "sub"
    _write_legacy_run(source, run_id="prod", script_path=run_root / "link" / "script.md")
    target.mkdir(parents=True, exist_ok=True)
    (target / "script.md").write_text(secret, encoding="utf-8")
    (run_root / "link").symlink_to(target, target_is_directory=True)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].script_text_available is False
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        run_sources = _run_level_script_sources(repository, run.run_uuid)
        assert run_sources[0].content is None
        for source_record in repository.get_text_sources(run.run_uuid):
            assert source_record.content != secret


def test_symlinked_chunk_audio_is_not_referenced(tmp_path):
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"outside-secret-audio")
    source = tmp_path / "out"
    chunks = [_default_chunk(number=1, file="chunk_01.mp3")]
    run_root = _write_legacy_run(
        source, run_id="prod", chunks=chunks, missing_files=("chunk_01.mp3",)
    )
    (run_root / "chunks" / "chunk_01.mp3").symlink_to(outside)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert "symlinked_artifact" in preview.runs[0].conflicts
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        roles = {artifact.role for artifact in repository.get_artifacts(run.run_uuid)}
        assert "chunk_audio" not in roles
        paths = {artifact.path for artifact in repository.get_artifacts(run.run_uuid)}
        assert str(run_root / "chunks" / "chunk_01.mp3") not in paths
        assert str(outside) not in paths


def test_symlinked_main_audio_is_not_referenced(tmp_path):
    outside = tmp_path / "outside-main.mp3"
    outside.write_bytes(b"outside-main-secret")
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    main_name = f"prod-voiceover-{_SLUG}.mp3"
    (run_root / main_name).unlink()
    (run_root / main_name).symlink_to(outside)
    database_path = tmp_path / "history.sqlite3"

    import_legacy_runs(source, database_path=database_path)

    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        paths = {artifact.path for artifact in repository.get_artifacts(run.run_uuid)}
        assert str(run_root / main_name) not in paths
        assert str(outside) not in paths
        assert "main_audio" not in {
            artifact.role for artifact in repository.get_artifacts(run.run_uuid)
        }


def test_declared_symlinked_run_manifest_blocks_import(tmp_path):
    outside = tmp_path / "outside-run.json"
    outside.write_text(json.dumps({"run_id": "hijack"}), encoding="utf-8")
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod", write_run_json=False)
    declared_name = f"prod-voiceover-{_SLUG}.json"
    (run_root / declared_name).symlink_to(outside)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    result = import_legacy_runs(source, database_path=database_path)

    assert preview.runs[0].importable is False
    assert "symlinked_run_manifest" in preview.runs[0].conflicts
    assert result.rejected_count == 1
    with _repository(database_path) as repository:
        assert repository.list_runs() == []


def _strip_identity_chunks(run_root):
    for name in ("run_state.json", "manifest.json", f"prod-voiceover-{_SLUG}.json"):
        path = run_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("chunks", None)
        payload.pop("chunk_count", None)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_symlinked_chunks_directory_json_is_not_read(tmp_path):
    sentinel_text = "EXTERNAL_CHUNKS_SENTINEL_TEXT"
    external = tmp_path / "external"
    external.mkdir()
    (external / "chunks.json").write_text(
        json.dumps(
            {
                "artifact_type": "voiceover-chunks",
                "chunks": [
                    {
                        "status": "completed",
                        "number": 7,
                        "id": "external-chunk",
                        "file": "chunk_01.mp3",
                        "text": sentinel_text,
                        "text_hash": hashlib.sha256(sentinel_text.encode("utf-8")).hexdigest(),
                        "cost": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (external / "chunk_01.mp3").write_bytes(b"external-chunk-audio")

    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    # The identity files are left without chunks, so the external chunks.json is
    # the only chunk source: if the importer read it, the sentinel would appear.
    _strip_identity_chunks(run_root)
    shutil.rmtree(run_root / "chunks")
    (run_root / "chunks").symlink_to(external, target_is_directory=True)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert "chunks_manifest_symlinked_json" in preview.runs[0].conflicts
    assert preview.runs[0].chunk_count == 0
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        for text_source in repository.get_text_sources(run.run_uuid):
            assert text_source.content != sentinel_text
        for part in repository.get_parts(run.run_uuid):
            assert part.prepared_text != sentinel_text
        for artifact in repository.get_artifacts(run.run_uuid):
            assert Path(artifact.path).resolve().is_relative_to(run_root)


def test_symlinked_chunks_directory_audio_is_not_referenced(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "chunk_01.mp3").write_bytes(b"external-secret-audio")

    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    shutil.rmtree(run_root / "chunks")
    (run_root / "chunks").symlink_to(external, target_is_directory=True)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert "symlinked_artifact" in preview.runs[0].conflicts
    assert preview.runs[0].chunks_missing_audio == 0
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        artifacts = repository.get_artifacts(run.run_uuid)
        assert "chunk_audio" not in {artifact.role for artifact in artifacts}
        paths = {artifact.path for artifact in artifacts}
        assert str(external / "chunk_01.mp3") not in paths
        assert str(run_root / "chunks" / "chunk_01.mp3") not in paths


def test_symlinked_chunks_directory_even_inside_root_is_rejected(tmp_path):
    # The ancestor is rejected even when it resolves inside the run: only the
    # leaf check would otherwise let a symlinked ``chunks`` through.
    source = tmp_path / "out"
    run_root = _write_legacy_run(source, run_id="prod")
    real_chunks = source / "real-chunks"
    (run_root / "chunks").rename(real_chunks)
    (run_root / "chunks").symlink_to(real_chunks, target_is_directory=True)
    database_path = tmp_path / "history.sqlite3"

    preview = preview_legacy_import(source, database_path=database_path)
    import_legacy_runs(source, database_path=database_path)

    assert "chunks_manifest_symlinked_json" in preview.runs[0].conflicts
    assert "symlinked_artifact" in preview.runs[0].conflicts
    assert preview.runs[0].chunk_count == 1
    with _repository(database_path) as repository:
        run = repository.find_run_by_legacy_source_root(str(run_root))
        roles = {artifact.role for artifact in repository.get_artifacts(run.run_uuid)}
        assert "chunk_audio" not in roles
        assert "chunks_manifest" not in roles


def test_import_rejects_relative_voiceover_home_before_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOICEOVER_HOME", "relative-home")
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(HistoryPathsError):
        import_legacy_runs(source)

    assert not (tmp_path / "relative-home").exists()


def test_preview_rejects_relative_voiceover_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOICEOVER_HOME", "relative-home")
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(HistoryPathsError):
        preview_legacy_import(source)

    assert not (tmp_path / "relative-home").exists()


def test_import_rejects_insecure_existing_home_before_db_write(tmp_path, monkeypatch):
    home = tmp_path / "voiceover-home"
    home.mkdir()
    home.chmod(0o755)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(HistoryHomePermissionError):
        import_legacy_runs(source)

    assert not (home / "history.sqlite3").exists()
    assert not (home / "runs").exists()
    assert (home.stat().st_mode & 0o077) != 0


def test_import_rejects_insecure_home_even_when_database_is_unknown(tmp_path, monkeypatch):
    home = tmp_path / "voiceover-home"
    home.mkdir()
    home.chmod(0o755)
    database_path = home / "history.sqlite3"
    raw = sqlite3.connect(database_path)
    try:
        raw.execute("CREATE TABLE future_only (id INTEGER PRIMARY KEY)")
        raw.execute("PRAGMA user_version = 999")
        raw.commit()
    finally:
        raw.close()
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(HistoryHomePermissionError):
        import_legacy_runs(source)

    assert not (home / "runs").exists()
    assert not (home / "backups").exists()
    assert not (home / "logs").exists()


def test_import_rejects_insecure_existing_explicit_parent(tmp_path):
    parent = tmp_path / "open-parent"
    parent.mkdir()
    parent.chmod(0o755)
    database_path = parent / "history.sqlite3"
    source = tmp_path / "out"
    _write_legacy_run(source, run_id="prod")

    with pytest.raises(HistoryHomePermissionError):
        import_legacy_runs(source, database_path=database_path)

    assert not database_path.exists()
    assert (parent.stat().st_mode & 0o077) != 0
