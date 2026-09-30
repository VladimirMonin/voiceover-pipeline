"""Contract tests for the typed history repository.

These tests cover the importer-facing API in
``voiceover_pipeline.history.repository``: stable run UUIDs, ``run_root`` scope
that is independent of the user label, one-per-root legacy import idempotency,
atomic multi-entity inserts through one transaction, database-enforced run
ownership, exact ``Decimal`` and preserved legacy-number money, fail-closed
snapshot redaction and path classification, UTF-8 text completeness, and
bounded metadata queries.
"""

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

import voiceover_pipeline.history.repository as history_repository_module
from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.repository import (
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    COST_SOURCE_LEGACY_FLOAT,
    COST_SOURCE_UNKNOWN,
    MAX_QUERY_LIMIT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    PATH_KIND_MANAGED_RELATIVE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_SCRIPT,
    Cost,
    HistoryPaidAttemptConflictError,
    HistoryPaidAttemptInTransactionError,
    HistoryPartNotFoundError,
    HistoryRepository,
    HistoryRevisionConflictError,
    HistoryRunNotFoundError,
    HistoryRunNotReservableError,
)


@pytest.fixture
def repository(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as database:
        database.migrate()
        yield HistoryRepository(database)


def test_create_run_generates_stable_uuid(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))

    assert run.run_uuid
    assert repository.get_run(run.run_uuid).run_uuid == run.run_uuid
    assert repository.get_run("does-not-exist") is None


def test_caller_supplied_run_uuid_is_preserved(repository, tmp_path):
    supplied = "11111111-2222-3333-4444-555555555555"

    run = repository.create_run(operation="asr", run_root=str(tmp_path), run_uuid=supplied)

    assert run.run_uuid == supplied
    assert repository.get_run(supplied).run_uuid == supplied


def test_invalid_run_uuid_is_rejected(repository, tmp_path):
    with pytest.raises(ValueError):
        repository.create_run(operation="tts", run_root=str(tmp_path), run_uuid="not-a-uuid")


def test_distinct_roots_with_same_label_are_distinct_runs(repository, tmp_path):
    first_root = str(tmp_path / "out-a" / "prod")
    second_root = str(tmp_path / "out-b" / "prod")

    first = repository.create_run(operation="tts", run_root=first_root, user_label="prod")
    second = repository.create_run(operation="tts", run_root=second_root, user_label="prod")

    assert first.run_uuid != second.run_uuid
    candidates = repository.find_runs_by_label("prod")
    assert {candidate.run_uuid for candidate in candidates} == {first.run_uuid, second.run_uuid}
    assert {candidate.run_root for candidate in candidates} == {first_root, second_root}
    assert [run.run_uuid for run in repository.find_runs_by_root(first_root)] == [first.run_uuid]


def test_shared_run_root_still_allows_independent_runs(repository, tmp_path):
    shared_root = str(tmp_path / "out" / "prod")

    first = repository.create_run(operation="tts", run_root=shared_root)
    second = repository.create_run(operation="tts", run_root=shared_root)

    assert first.run_uuid != second.run_uuid
    assert {run.run_uuid for run in repository.find_runs_by_root(shared_root)} == {
        first.run_uuid,
        second.run_uuid,
    }


def test_same_legacy_source_root_is_imported_once(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")

    with repository.transaction():
        first, created_first = repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )
    with repository.transaction():
        second, created_second = repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )

    assert created_first is True
    assert created_second is False
    assert first.run_uuid == second.run_uuid
    assert len(repository.list_runs()) == 1


def test_create_legacy_run_requires_outer_transaction(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")

    with pytest.raises(history_repository_module.HistoryRepositoryError):
        repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )

    assert repository.list_runs() == []
    assert repository.find_run_by_legacy_source_root(legacy_root) is None


def test_create_run_with_legacy_source_root_requires_outer_transaction(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")

    with pytest.raises(history_repository_module.HistoryRepositoryError):
        repository.create_run(operation="tts", run_root=legacy_root, legacy_source_root=legacy_root)

    assert repository.list_runs() == []


def test_failed_legacy_import_rolls_back_and_retry_completes(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")

    with pytest.raises(sqlite3.IntegrityError):
        with repository.transaction():
            run, created = repository.create_legacy_run(
                operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
            )
            assert created is True
            repository.add_part(run.run_uuid, position=1)
            repository.add_part(run.run_uuid, position=1)

    assert repository.list_runs() == []
    assert repository.find_run_by_legacy_source_root(legacy_root) is None

    with repository.transaction():
        run, created = repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )
        assert created is True
        part = repository.add_part(run.run_uuid, position=1)
        repository.add_attempt(
            run.run_uuid,
            call_type="tts_chunk",
            part_uuid=part.part_uuid,
            cost=Cost.legacy_float("0.1000", currency="RUB"),
        )

    assert len(repository.list_runs()) == 1
    assert len(repository.get_parts(run.run_uuid)) == 1
    assert len(repository.get_attempts(run.run_uuid)) == 1


def test_two_connections_import_same_legacy_root_once(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    legacy_root = str(tmp_path / "legacy-run")

    with HistoryDatabase(database_path) as first_database:
        first_database.migrate()
        with HistoryDatabase(database_path) as second_database:
            second_database.connect()
            first_repository = HistoryRepository(first_database)
            second_repository = HistoryRepository(second_database)

            with first_repository.transaction():
                first, created_first = first_repository.create_legacy_run(
                    operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
                )
            with second_repository.transaction():
                second, created_second = second_repository.create_legacy_run(
                    operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
                )

            assert created_first is True
            assert created_second is False
            assert first.run_uuid == second.run_uuid
            assert len(first_repository.list_runs()) == 1


def test_different_legacy_source_roots_are_distinct(repository, tmp_path):
    with repository.transaction():
        first, _ = repository.create_legacy_run(
            operation="tts",
            run_root=str(tmp_path / "a"),
            legacy_source_root=str(tmp_path / "legacy-a"),
        )
    with repository.transaction():
        second, _ = repository.create_legacy_run(
            operation="tts",
            run_root=str(tmp_path / "b"),
            legacy_source_root=str(tmp_path / "legacy-b"),
        )

    assert first.run_uuid != second.run_uuid
    assert {run.legacy_source_root for run in repository.list_runs()} == {
        str(Path(tmp_path / "legacy-a").resolve()),
        str(Path(tmp_path / "legacy-b").resolve()),
    }


def test_legacy_source_root_unique_index_is_enforced(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )

    with pytest.raises(sqlite3.IntegrityError):
        with repository.transaction():
            repository.create_run(
                operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
            )


def test_find_run_by_legacy_source_root(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        record, _ = repository.create_legacy_run(
            operation="tts", run_root=legacy_root, legacy_source_root=legacy_root
        )

    found = repository.find_run_by_legacy_source_root(legacy_root)
    assert found is not None and found.run_uuid == record.run_uuid
    assert repository.find_run_by_legacy_source_root(str(tmp_path / "other")) is None


def test_run_config_snapshot_roundtrips(repository, tmp_path):
    snapshot = {
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
        "voice": "alloy",
        "style_prompt": "Спокойный голос, ёлка",
    }

    run = repository.create_run(operation="tts", run_root=str(tmp_path), config_snapshot=snapshot)
    reloaded = repository.get_run(run.run_uuid)

    assert reloaded is not None
    assert reloaded.config_snapshot == snapshot


def test_config_snapshot_redacts_secrets_before_storage(repository, tmp_path):
    sentinel = "sk-live-SENTINEL0123456789"
    snapshot = {
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
        "api_key": sentinel,
        "authorization": f"Bearer {sentinel}",
        "download_url": f"https://cdn.example.com/a.mp3?X-Amz-Signature={sentinel}&Expires=1",
        "nested": {"access_token": sentinel, "voice": "alloy"},
        "pricing_snapshot": {"audio_per_million": 5.0, "currency": "RUB"},
    }

    run = repository.create_run(operation="tts", run_root=str(tmp_path), config_snapshot=snapshot)
    stored = repository.get_run(run.run_uuid).config_snapshot

    assert stored["provider"] == "polza-tts"
    assert stored["model"] == "openai/gpt-4o-mini-tts"
    assert stored["pricing_snapshot"] == {"audio_per_million": 5.0, "currency": "RUB"}
    assert stored["nested"] == {"voice": "alloy"}
    assert "api_key" not in stored
    assert "authorization" not in stored
    assert sentinel not in json.dumps(stored)

    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    try:
        raw = connection.execute(
            "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run.run_uuid,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert sentinel not in raw
    assert "sk-live" not in raw
    assert "X-Amz-Signature" not in raw


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic ZmFrZTpmYWtl",
        "Authorization: Digest ZmFrZTpmYWtl",
        "Proxy-Authorization: Custom ZmFrZTpmYWtl",
    ],
)
def test_config_snapshot_redacts_authorization_header_value(repository, tmp_path, header):
    run = repository.create_run(
        operation="tts",
        run_root=str(tmp_path),
        config_snapshot={"note": header, "label": "safe"},
    )

    stored = repository.get_run(run.run_uuid).config_snapshot
    assert stored == {"note": history_repository_module.REDACTED_VALUE, "label": "safe"}
    assert "ZmFrZTpmYWtl" not in json.dumps(stored)  # synthetic "fake:fake"


def test_config_snapshot_redacts_google_signed_url(repository, tmp_path):
    sentinel = "SYNTHETIC_SENTINEL"
    snapshot = {
        "provider": "polza-tts",
        "download_url": f"https://example.invalid/file?X-Goog-Signature={sentinel}",
    }

    run = repository.create_run(operation="tts", run_root=str(tmp_path), config_snapshot=snapshot)
    stored = repository.get_run(run.run_uuid).config_snapshot

    assert stored["provider"] == "polza-tts"
    assert sentinel not in json.dumps(stored)
    assert "X-Goog-Signature" not in json.dumps(stored)

    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    try:
        raw = connection.execute(
            "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run.run_uuid,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert sentinel not in raw
    assert "X-Goog-Signature" not in raw


def test_config_snapshot_redacts_unfamiliar_query_url(repository, tmp_path):
    sentinel = "UNFAMILIAR_PARAM_VALUE"
    snapshot = {
        "provider": "polza-tts",
        "callback": f"https://example.invalid/cb?totally_unknown_param={sentinel}",
        "safe_url": "https://example.invalid/audio.mp3",
    }

    run = repository.create_run(operation="tts", run_root=str(tmp_path), config_snapshot=snapshot)
    stored = repository.get_run(run.run_uuid).config_snapshot

    assert stored["provider"] == "polza-tts"
    assert sentinel not in json.dumps(stored)
    assert stored["safe_url"] == "https://example.invalid/audio.mp3"


def test_metadata_search_does_not_interpret_sql(repository, tmp_path):
    repository.create_run(operation="tts", run_root=str(tmp_path / "a"), user_label="prod")

    assert repository.find_runs_by_label("prod' OR '1'='1") == []


def test_bundle_insert_is_atomic_and_readable(repository, tmp_path):
    text = "Привет, мир! «Ёжик» — ёлка, 日本語, emoji 🎙️"
    with repository.transaction():
        run = repository.create_run(operation="tts", run_root=str(tmp_path / "run"))
        part = repository.add_part(
            run.run_uuid, position=1, prepared_text=text, voice="alloy", stage="synthesis"
        )
        attempt = repository.add_attempt(
            run.run_uuid,
            call_type="tts_chunk",
            part_uuid=part.part_uuid,
            provider="polza-tts",
            model="openai/gpt-4o-mini-tts",
            cost=Cost.legacy_float(0.1, currency="RUB"),
        )
        chunk_artifact = repository.add_artifact(
            run.run_uuid,
            role="chunk_audio",
            path_kind=PATH_KIND_MANAGED_RELATIVE,
            path="chunks/chunk_01.mp3",
            part_uuid=part.part_uuid,
            attempt_uuid=attempt.attempt_uuid,
            mime="audio/mpeg",
            size_bytes=1234,
            sha256="a" * 64,
        )
        repository.add_artifact(
            run.run_uuid,
            role="main_audio",
            path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
            path="/media/voice/out.mp3",
            availability=AVAILABILITY_MISSING,
        )
        source = repository.add_text_source(
            run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="legacy_import",
            content=text,
            part_uuid=part.part_uuid,
            artifact_uuid=chunk_artifact.artifact_uuid,
            content_hash="b" * 64,
            language="ru",
        )

    parts = repository.get_parts(run.run_uuid)
    attempts = repository.get_attempts(run.run_uuid)
    artifacts = repository.get_artifacts(run.run_uuid)
    sources = repository.get_text_sources(run.run_uuid)

    assert len(parts) == 1
    assert parts[0].prepared_text == text
    assert len(attempts) == 1
    assert attempts[0].part_uuid == part.part_uuid
    assert {artifact.role for artifact in artifacts} == {"chunk_audio", "main_audio"}
    assert {artifact.availability for artifact in artifacts} == {
        AVAILABILITY_PRESENT,
        AVAILABILITY_MISSING,
    }
    assert len(sources) == 1
    assert sources[0].content == text
    assert sources[0].artifact_uuid == chunk_artifact.artifact_uuid
    assert source.text_source_uuid == sources[0].text_source_uuid


def test_cross_run_associations_are_rejected(repository, tmp_path):
    run_a = repository.create_run(operation="tts", run_root=str(tmp_path / "a"))
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_a = repository.add_part(run_a.run_uuid, position=1)
    attempt_a = repository.add_attempt(
        run_a.run_uuid, call_type="tts_chunk", part_uuid=part_a.part_uuid
    )
    artifact_a = repository.add_artifact(
        run_a.run_uuid,
        role="chunk_audio",
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="chunks/a.mp3",
        part_uuid=part_a.part_uuid,
        attempt_uuid=attempt_a.attempt_uuid,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.add_attempt(run_b.run_uuid, call_type="tts_chunk", part_uuid=part_a.part_uuid)
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_artifact(
            run_b.run_uuid,
            role="chunk_audio",
            path_kind=PATH_KIND_MANAGED_RELATIVE,
            path="chunks/b.mp3",
            part_uuid=part_a.part_uuid,
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_artifact(
            run_b.run_uuid,
            role="chunk_audio",
            path_kind=PATH_KIND_MANAGED_RELATIVE,
            path="chunks/b.mp3",
            attempt_uuid=attempt_a.attempt_uuid,
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_text_source(
            run_b.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="legacy_import",
            part_uuid=part_a.part_uuid,
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_text_source(
            run_b.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="legacy_import",
            artifact_uuid=artifact_a.artifact_uuid,
        )

    assert repository.get_parts(run_b.run_uuid) == []
    assert repository.get_attempts(run_b.run_uuid) == []
    assert repository.get_artifacts(run_b.run_uuid) == []


def test_failed_transaction_rolls_back_every_entity(repository, tmp_path):
    run_uuid = None

    with pytest.raises(sqlite3.IntegrityError):
        with repository.transaction():
            run = repository.create_run(operation="tts", run_root=str(tmp_path / "run"))
            run_uuid = run.run_uuid
            repository.add_part(run_uuid, position=1)
            repository.add_part(run_uuid, position=1)

    assert run_uuid is not None
    assert repository.list_runs() == []
    assert repository.get_run(run_uuid) is None
    assert repository.get_parts(run_uuid) == []


def test_exact_decimal_cost_roundtrips_without_loss(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    amount = Decimal("0.123456789012345678901234567")

    attempt = repository.add_attempt(
        run.run_uuid, call_type="tts_chunk", cost=Cost.exact(amount, currency="RUB")
    )
    reloaded = repository.get_attempt(attempt.attempt_uuid)

    assert reloaded is not None
    assert reloaded.cost.amount == "0.123456789012345678901234567"
    assert reloaded.cost.as_decimal() == amount
    assert reloaded.cost.currency == "RUB"
    assert reloaded.cost.exact_available is True


def test_zero_cost_is_not_unknown(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    zero = repository.add_attempt(
        run.run_uuid, call_type="tts_chunk", cost=Cost.exact(Decimal("0"), currency="RUB")
    )
    unknown = repository.add_attempt(run.run_uuid, call_type="tts_chunk")

    zero_reloaded = repository.get_attempt(zero.attempt_uuid)
    unknown_reloaded = repository.get_attempt(unknown.attempt_uuid)

    assert zero_reloaded is not None and unknown_reloaded is not None
    assert zero_reloaded.cost.amount == "0"
    assert zero_reloaded.cost.as_decimal() == Decimal(0)
    assert zero_reloaded.cost.exact_available is True
    assert unknown_reloaded.cost.amount is None
    assert unknown_reloaded.cost.as_decimal() is None
    assert unknown_reloaded.cost.exact_available is False


def test_legacy_float_cost_records_missing_exact_guarantee(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    attempt = repository.add_attempt(
        run.run_uuid, call_type="tts_chunk", cost=Cost.legacy_float(0.1, currency="RUB")
    )
    reloaded = repository.get_attempt(attempt.attempt_uuid)

    assert reloaded is not None
    assert reloaded.cost.source == COST_SOURCE_LEGACY_FLOAT
    assert reloaded.cost.exact_available is False
    assert reloaded.cost.raw == "0.1"
    assert reloaded.cost.as_decimal() == Decimal("0.1")


def test_legacy_decimal_cost_preserves_full_precision(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    amount = Decimal("0.100000000000000000001")

    attempt = repository.add_attempt(
        run.run_uuid, call_type="tts_chunk", cost=Cost.legacy_float(amount, currency="RUB")
    )
    reloaded = repository.get_attempt(attempt.attempt_uuid)

    assert reloaded is not None
    assert reloaded.cost.amount == "0.100000000000000000001"
    assert reloaded.cost.raw == "0.100000000000000000001"
    assert reloaded.cost.as_decimal() == amount
    assert reloaded.cost.exact_available is False


def test_legacy_json_lexeme_is_preserved(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    attempt = repository.add_attempt(
        run.run_uuid,
        call_type="tts_chunk",
        cost=Cost.legacy_float(0.1, currency="RUB", raw="0.1000"),
    )
    reloaded = repository.get_attempt(attempt.attempt_uuid)

    assert reloaded is not None
    assert reloaded.cost.raw == "0.1000"
    assert reloaded.cost.amount == "0.1"
    assert reloaded.cost.exact_available is False


def test_legacy_decimal_string_lexeme_is_preserved(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    attempt = repository.add_attempt(
        run.run_uuid, call_type="tts_chunk", cost=Cost.legacy_float("0.1000", currency="RUB")
    )
    reloaded = repository.get_attempt(attempt.attempt_uuid)

    assert reloaded is not None
    assert reloaded.cost.amount == "0.1000"
    assert reloaded.cost.raw == "0.1000"
    assert reloaded.cost.as_decimal() == Decimal("0.1000")
    assert reloaded.cost.exact_available is False


def test_text_source_preserves_utf8_and_incomplete_marker(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    text = "Привет, мир! «Ёжик» — ёлка, 日本語, emoji 🎙️"

    complete = repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="legacy_import", content=text
    )
    incomplete = repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="legacy_import",
        content=None,
        content_hash="c" * 64,
        text_completeness=TEXT_COMPLETENESS_INCOMPLETE,
    )

    sources = {
        source.text_source_uuid: source for source in repository.get_text_sources(run.run_uuid)
    }
    assert sources[complete.text_source_uuid].content == text
    assert sources[complete.text_source_uuid].kind == TEXT_KIND_TTS_SCRIPT
    assert sources[complete.text_source_uuid].origin == "legacy_import"
    assert sources[incomplete.text_source_uuid].content is None
    assert sources[incomplete.text_source_uuid].kind == TEXT_KIND_ASR_TRANSCRIPT
    assert sources[incomplete.text_source_uuid].content_hash == "c" * 64
    assert sources[incomplete.text_source_uuid].text_completeness == TEXT_COMPLETENESS_INCOMPLETE


def test_list_runs_filters_and_bounds_limit(repository, tmp_path):
    for index in range(3):
        repository.create_run(
            operation="tts", run_root=str(tmp_path / f"root-{index}"), user_label="prod"
        )
    repository.create_run(operation="asr", run_root=str(tmp_path / "asr-root"))

    assert len(repository.list_runs()) == 4
    assert len(repository.list_runs(operation="tts")) == 3
    assert len(repository.list_runs(user_label="prod", limit=2)) == 2
    assert len(repository.list_runs(user_label="prod", limit=2, offset=2)) == 1

    with pytest.raises(ValueError):
        repository.list_runs(limit=0)
    with pytest.raises(ValueError):
        repository.list_runs(limit=MAX_QUERY_LIMIT + 1)
    with pytest.raises(ValueError):
        repository.list_runs(offset=-1)


def test_invalid_path_kind_is_rejected(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    with pytest.raises(ValueError):
        repository.add_artifact(
            run.run_uuid, role="chunk_audio", path_kind="relative", path="chunks/a.mp3"
        )


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/abs/out.mp3",
        "../outside",
        "chunks/../x",
        "chunks\\a.mp3",
        "C:/x",
        "chunks//a.mp3",
        ".",
    ],
)
def test_managed_relative_path_rejects_unsafe_values(repository, tmp_path, bad_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    with pytest.raises(ValueError):
        repository.add_artifact(
            run.run_uuid,
            role="chunk_audio",
            path_kind=PATH_KIND_MANAGED_RELATIVE,
            path=bad_path,
        )


def test_managed_relative_path_accepts_nested_path(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    artifact = repository.add_artifact(
        run.run_uuid,
        role="chunk_audio",
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="chunks/chunk_01.mp3",
    )

    assert artifact.path == "chunks/chunk_01.mp3"


@pytest.mark.parametrize("bad_path", ["", "relative/out.mp3", "out.mp3"])
def test_external_absolute_path_rejects_relative_values(repository, tmp_path, bad_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    with pytest.raises(ValueError):
        repository.add_artifact(
            run.run_uuid,
            role="main_audio",
            path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
            path=bad_path,
        )


@pytest.mark.parametrize("good_path", ["/media/voice/out.mp3", "C:/media/out.mp3"])
def test_external_absolute_path_accepts_absolute_values(repository, tmp_path, good_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    artifact = repository.add_artifact(
        run.run_uuid,
        role="main_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=good_path,
    )

    assert artifact.path == good_path


# -- run revision compare-and-swap -------------------------------------------


def test_new_run_starts_at_revision_one(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    assert run.revision == 1
    assert repository.get_run(run.run_uuid).revision == 1


def test_advance_run_revision_increments_once_and_sets_status(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")

    advanced = repository.advance_run_revision(
        run.run_uuid, expected_revision=1, status="completed"
    )

    assert advanced.revision == 2
    assert advanced.status == "completed"
    reloaded = repository.get_run(run.run_uuid)
    assert reloaded is not None
    assert reloaded.revision == 2
    assert reloaded.status == "completed"
    assert reloaded.updated_at >= run.updated_at


def test_advance_run_revision_without_status_keeps_status(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")

    advanced = repository.advance_run_revision(run.run_uuid, expected_revision=1)

    assert advanced.revision == 2
    assert advanced.status == "running"


def test_advance_run_revision_rejects_stale_expected_without_change(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")
    repository.advance_run_revision(run.run_uuid, expected_revision=1, status="completed")

    with pytest.raises(HistoryRevisionConflictError):
        repository.advance_run_revision(run.run_uuid, expected_revision=1, status="failed")

    unchanged = repository.get_run(run.run_uuid)
    assert unchanged is not None
    assert unchanged.revision == 2
    assert unchanged.status == "completed"


def test_advance_run_revision_missing_uuid_is_typed_not_found(repository):
    with pytest.raises(HistoryRunNotFoundError):
        repository.advance_run_revision("11111111-2222-3333-4444-555555555555", expected_revision=1)


@pytest.mark.parametrize("bad_revision", [0, -1, True, False, 1.0, "1"])
def test_advance_run_revision_rejects_invalid_expected_revision(repository, tmp_path, bad_revision):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    with pytest.raises(ValueError):
        repository.advance_run_revision(run.run_uuid, expected_revision=bad_revision)

    unchanged = repository.get_run(run.run_uuid)
    assert unchanged is not None
    assert unchanged.revision == 1


def test_advance_run_revision_rolls_back_with_outer_transaction(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")

    with pytest.raises(RuntimeError):
        with repository.transaction():
            repository.advance_run_revision(run.run_uuid, expected_revision=1, status="completed")
            raise RuntimeError("boom after advance")

    unchanged = repository.get_run(run.run_uuid)
    assert unchanged is not None
    assert unchanged.revision == 1
    assert unchanged.status == "running"


def test_advance_run_revision_conflict_rolls_back_outer_transaction(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")
    repository.advance_run_revision(run.run_uuid, expected_revision=1, status="completed")

    with pytest.raises(HistoryRevisionConflictError):
        with repository.transaction():
            repository.advance_run_revision(run.run_uuid, expected_revision=1, status="failed")
            repository.advance_run_revision(run.run_uuid, expected_revision=2, status="failed")

    unchanged = repository.get_run(run.run_uuid)
    assert unchanged is not None
    assert unchanged.revision == 2
    assert unchanged.status == "completed"


# -- paid TTS attempt reservation -------------------------------------------


def test_reserve_paid_tts_attempt_bumps_revision_and_commits_marker(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")
    part = repository.add_part(run.run_uuid, position=1, prepared_text="chunk text")

    advanced, attempt = repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="elevenlabs/eleven_multilingual_v2",
        account_alias="main",
    )

    assert advanced.revision == 2
    assert advanced.status == "running"
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2

    assert attempt.run_uuid == run.run_uuid
    assert attempt.part_uuid == part.part_uuid
    assert attempt.call_type == history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK
    assert attempt.provider == "polza-tts"
    assert attempt.model == "elevenlabs/eleven_multilingual_v2"
    assert attempt.account_alias == "main"
    assert attempt.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING
    assert attempt.remote_id is None
    assert attempt.cost.amount is None
    assert attempt.cost.source == COST_SOURCE_UNKNOWN
    assert attempt.cost.exact_available is False

    reloaded_attempt = repository.get_attempt(attempt.attempt_uuid)
    assert reloaded_attempt is not None
    assert reloaded_attempt.remote_id is None
    assert reloaded_attempt.cost.amount is None


def test_reserve_paid_tts_attempt_commits_marker_before_external_call(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)

    _, attempt = repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="m",
        account_alias="a",
    )

    # A separate connection already sees the committed marker, so the durable
    # unconfirmed attempt exists before any caller could make a paid request.
    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    try:
        revision = connection.execute(
            "SELECT revision FROM runs WHERE run_uuid = ?", (run.run_uuid,)
        ).fetchone()[0]
        row = connection.execute(
            "SELECT remote_id, status, cost, cost_source, cost_exact_available "
            "FROM attempts WHERE attempt_uuid = ?",
            (attempt.attempt_uuid,),
        ).fetchone()
    finally:
        connection.close()

    assert revision == 2
    assert row[0] is None  # remote_id: no remote id exists before the submit
    assert row[1] == history_repository_module.ATTEMPT_STATUS_SUBMITTING
    assert row[2] is None  # cost stays unknown, never fabricated
    assert row[3] == COST_SOURCE_UNKNOWN
    assert row[4] == 0


def test_reserve_paid_tts_attempt_rejects_stale_revision_without_bump(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    first = repository.add_part(run.run_uuid, position=1)
    second = repository.add_part(run.run_uuid, position=2)

    repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=first.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="m",
        account_alias="a",
    )

    with pytest.raises(HistoryRevisionConflictError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=second.part_uuid,
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    assert len(repository.get_attempts(run.run_uuid)) == 1


def test_reserve_paid_tts_attempt_rejects_foreign_part_without_bump(repository, tmp_path):
    run_a = repository.create_run(operation="tts", run_root=str(tmp_path / "a"))
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_a = repository.add_part(run_a.run_uuid, position=1)

    with pytest.raises(HistoryPartNotFoundError):
        repository.reserve_paid_tts_attempt(
            run_b.run_uuid,
            part_uuid=part_a.part_uuid,
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    unchanged_b = repository.get_run(run_b.run_uuid)
    unchanged_a = repository.get_run(run_a.run_uuid)
    assert unchanged_b is not None and unchanged_b.revision == 1
    assert unchanged_a is not None and unchanged_a.revision == 1
    assert repository.get_attempts(run_b.run_uuid) == []


def test_reserve_paid_tts_attempt_rejects_missing_part_without_bump(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))

    with pytest.raises(HistoryPartNotFoundError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid="11111111-2222-3333-4444-555555555555",
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert repository.get_attempts(run.run_uuid) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": ""},
        {"provider": "   "},
        {"model": ""},
        {"account_alias": ""},
    ],
)
def test_reserve_paid_tts_attempt_rejects_empty_identity(repository, tmp_path, overrides):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    arguments = {"provider": "polza-tts", "model": "m", "account_alias": "a", **overrides}

    with pytest.raises(ValueError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid, part_uuid=part.part_uuid, expected_revision=1, **arguments
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert repository.get_attempts(run.run_uuid) == []


@pytest.mark.parametrize("bad_revision", [0, -1, True, False, 1.0, "1"])
def test_reserve_paid_tts_attempt_rejects_invalid_expected_revision(
    repository, tmp_path, bad_revision
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)

    with pytest.raises(ValueError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=part.part_uuid,
            expected_revision=bad_revision,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert repository.get_attempts(run.run_uuid) == []


def test_reserve_paid_tts_attempt_missing_run_is_typed_not_found(repository):
    with pytest.raises(HistoryRunNotFoundError):
        repository.reserve_paid_tts_attempt(
            "11111111-2222-3333-4444-555555555555",
            part_uuid="22222222-3333-4444-5555-666666666666",
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )


def test_reserve_paid_tts_attempt_rejects_duplicate_on_fresh_revision(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)

    repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="m",
        account_alias="a",
    )

    with pytest.raises(HistoryPaidAttemptConflictError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=part.part_uuid,
            expected_revision=2,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    assert len(repository.get_attempts(run.run_uuid)) == 1


@pytest.mark.parametrize(
    "status, remote_id",
    [
        ("outcome_unknown", None),
        ("remote_accepted", "task-abc"),
        ("submitting", None),
    ],
)
def test_reserve_paid_tts_attempt_refuses_any_prior_attempt(
    repository, tmp_path, status, remote_id
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    repository.add_attempt(
        run.run_uuid,
        call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        status=status,
        remote_id=remote_id,
    )

    with pytest.raises(HistoryPaidAttemptConflictError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert len(repository.get_attempts(run.run_uuid)) == 1


def test_reserve_paid_tts_attempt_ignores_other_call_types(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    repository.add_attempt(run.run_uuid, call_type="asr_transcript", part_uuid=part.part_uuid)

    advanced, attempt = repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="m",
        account_alias="a",
    )

    assert advanced.revision == 2
    assert attempt.call_type == history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK


def test_reserve_paid_tts_attempt_refuses_open_outer_transaction(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    submitted = False

    # A reservation inside a caller's still-open transaction could return a
    # usable marker, be followed by the paid POST, and then be erased by the
    # caller's rollback. The seam must refuse before it can return.
    with pytest.raises(HistoryPaidAttemptInTransactionError):
        with repository.transaction():
            repository.reserve_paid_tts_attempt(
                run.run_uuid,
                part_uuid=part.part_uuid,
                expected_revision=1,
                provider="polza-tts",
                model="m",
                account_alias="a",
            )
            submitted = True

    assert submitted is False
    assert repository._connection.in_transaction is False
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert repository.get_attempts(run.run_uuid) == []


def test_reserve_paid_tts_attempt_refuses_legacy_imported_run(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        run, _ = repository.create_legacy_run(
            operation="tts",
            run_root=legacy_root,
            legacy_source_root=legacy_root,
            status="interrupted",
        )
        part = repository.add_part(run.run_uuid, position=1)
        # An import with only a run-total price records no part-linked chunk, so
        # the per-part duplicate guard alone would miss its paid work.
        repository.add_attempt(
            run.run_uuid,
            call_type="tts_run_total",
            cost=Cost.legacy_float("0.1000", currency="RUB"),
        )

    with pytest.raises(HistoryRunNotReservableError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert reloaded_run.legacy_source_root is not None
    assert [attempt.call_type for attempt in repository.get_attempts(run.run_uuid)] == [
        "tts_run_total"
    ]


def test_reserve_paid_tts_attempt_refuses_completed_run(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="completed")
    part = repository.add_part(run.run_uuid, position=1)

    # Repeating already completed work belongs to a new run, not a new paid
    # attempt on the finished one.
    with pytest.raises(HistoryRunNotReservableError):
        repository.reserve_paid_tts_attempt(
            run.run_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            provider="polza-tts",
            model="m",
            account_alias="a",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert repository.get_attempts(run.run_uuid) == []


def test_reserve_paid_tts_attempt_cross_connection_cannot_double_reserve(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as first_database:
        first_database.migrate()
        with HistoryDatabase(database_path) as second_database:
            second_database.connect()
            first = HistoryRepository(first_database)
            second = HistoryRepository(second_database)
            run = first.create_run(operation="tts", run_root=str(tmp_path))
            part = first.add_part(run.run_uuid, position=1)

            first.reserve_paid_tts_attempt(
                run.run_uuid,
                part_uuid=part.part_uuid,
                expected_revision=1,
                provider="polza-tts",
                model="m",
                account_alias="a",
            )

            with pytest.raises(HistoryPaidAttemptConflictError):
                second.reserve_paid_tts_attempt(
                    run.run_uuid,
                    part_uuid=part.part_uuid,
                    expected_revision=1,
                    provider="polza-tts",
                    model="m",
                    account_alias="a",
                )

            reloaded_run = second.get_run(run.run_uuid)
            assert reloaded_run is not None and reloaded_run.revision == 2
            assert len(second.get_attempts(run.run_uuid)) == 1


def test_advance_run_revision_two_connections_serialize(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as first_database:
        first_database.migrate()
        with HistoryDatabase(database_path) as second_database:
            second_database.connect()
            first = HistoryRepository(first_database)
            second = HistoryRepository(second_database)
            run = first.create_run(operation="tts", run_root=str(tmp_path), status="running")

            advanced = first.advance_run_revision(
                run.run_uuid, expected_revision=1, status="completed"
            )
            assert advanced.revision == 2

            with pytest.raises(HistoryRevisionConflictError):
                second.advance_run_revision(run.run_uuid, expected_revision=1)

            advanced_again = second.advance_run_revision(run.run_uuid, expected_revision=2)
            assert advanced_again.revision == 3
