"""Contract tests for the typed history repository.

These tests cover the importer-facing API in
``voiceover_pipeline.history.repository``: stable run UUIDs, ``run_root`` scope
that is independent of the user label, one-per-root legacy import idempotency,
atomic multi-entity inserts through one transaction, database-enforced run
ownership, exact ``Decimal`` and preserved legacy-number money, fail-closed
snapshot redaction and path classification, UTF-8 text completeness, and
bounded metadata queries. The paid Polza seams cover the pre-submit reservation,
the accepted media task id, the provider-observed media cost that must be
durable before a signed-URL download, and the verified raw-artifact link that
joins already-saved paid bytes to their attempt without holding a database
transaction across file hashing.
"""

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

import voiceover_pipeline.history.raw_receipt as raw_receipt_module
import voiceover_pipeline.history.repository as history_repository_module
from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.raw_receipt import write_paid_raw_receipt
from voiceover_pipeline.history.repository import (
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    COST_SOURCE_EXACT,
    COST_SOURCE_LEGACY_FLOAT,
    COST_SOURCE_PROVIDER_OBSERVED_FLOAT,
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
    HistoryPaidMediaCostConflictError,
    HistoryPaidMediaTaskConflictError,
    HistoryPaidRawConflictError,
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


# -- paid Polza Media task acceptance ---------------------------------------


def _reserve_paid_attempt(repository, run_uuid, part_uuid, *, expected_revision):
    """Reserve one paid Polza Media attempt as setup for an acceptance test."""
    _, attempt = repository.reserve_paid_tts_attempt(
        run_uuid,
        part_uuid=part_uuid,
        expected_revision=expected_revision,
        provider="polza-tts",
        model="elevenlabs/eleven_multilingual_v2",
        account_alias="main",
    )
    return attempt


def test_record_polza_media_task_accepted_binds_id_and_advances_revision(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="running")
    part = repository.add_part(run.run_uuid, position=1, prepared_text="chunk text")
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    advanced, accepted = repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id="media_task_01",
    )

    assert advanced.run_uuid == run.run_uuid
    assert advanced.revision == 3
    assert advanced.status == "running"
    assert accepted.attempt_uuid == reserved.attempt_uuid
    assert accepted.part_uuid == part.part_uuid
    assert accepted.call_type == history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK
    assert accepted.remote_id == "media_task_01"
    assert accepted.status == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
    # The immutable provider/model/account identity and the unknown cost survive.
    assert accepted.provider == "polza-tts"
    assert accepted.model == "elevenlabs/eleven_multilingual_v2"
    assert accepted.account_alias == "main"
    assert accepted.cost.amount is None
    assert accepted.cost.source == COST_SOURCE_UNKNOWN
    assert accepted.cost.exact_available is False


def test_record_polza_media_task_accepted_commits_before_poll(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id="task_abc",
    )

    # A separate connection already sees the committed transition, so a crash
    # before the first synthetic GET leaves the accepted task id durable.
    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    try:
        revision = connection.execute(
            "SELECT revision FROM runs WHERE run_uuid = ?", (run.run_uuid,)
        ).fetchone()[0]
        row = connection.execute(
            "SELECT remote_id, status, provider, model, account_alias, cost, cost_source, "
            "cost_exact_available FROM attempts WHERE attempt_uuid = ?",
            (reserved.attempt_uuid,),
        ).fetchone()
    finally:
        connection.close()

    assert revision == 3
    assert row[0] == "task_abc"
    assert row[1] == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
    assert row[2] == "polza-tts"
    assert row[3] == "elevenlabs/eleven_multilingual_v2"
    assert row[4] == "main"
    assert row[5] is None  # cost stays unknown, never fabricated
    assert row[6] == COST_SOURCE_UNKNOWN
    assert row[7] == 0


def test_record_polza_media_task_accepted_rejects_stale_revision_without_mutation(
    repository, tmp_path
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    first = repository.add_part(run.run_uuid, position=1)
    second = repository.add_part(run.run_uuid, position=2)
    first_reserved = _reserve_paid_attempt(
        repository, run.run_uuid, first.part_uuid, expected_revision=1
    )
    repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=first_reserved.attempt_uuid,
        part_uuid=first.part_uuid,
        expected_revision=2,
        remote_task_id="id-first",
    )
    second_reserved = _reserve_paid_attempt(
        repository, run.run_uuid, second.part_uuid, expected_revision=3
    )

    with pytest.raises(HistoryRevisionConflictError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=second_reserved.attempt_uuid,
            part_uuid=second.part_uuid,
            expected_revision=3,
            remote_task_id="id-second",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 4
    unchanged = repository.get_attempt(second_reserved.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_rejects_foreign_attempt_run(repository, tmp_path):
    run_a = repository.create_run(operation="tts", run_root=str(tmp_path / "a"))
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_a = repository.add_part(run_a.run_uuid, position=1)
    part_b = repository.add_part(run_b.run_uuid, position=1)
    reserved_a = _reserve_paid_attempt(
        repository, run_a.run_uuid, part_a.part_uuid, expected_revision=1
    )

    # The attempt belongs to run_a, so naming run_b must not rebind it.
    with pytest.raises(HistoryPaidMediaTaskConflictError):
        repository.record_polza_media_task_accepted(
            run_b.run_uuid,
            attempt_uuid=reserved_a.attempt_uuid,
            part_uuid=part_b.part_uuid,
            expected_revision=2,
            remote_task_id="id-1",
        )

    unchanged_a = repository.get_attempt(reserved_a.attempt_uuid)
    assert unchanged_a is not None
    assert unchanged_a.remote_id is None
    assert unchanged_a.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING
    reloaded_a = repository.get_run(run_a.run_uuid)
    reloaded_b = repository.get_run(run_b.run_uuid)
    assert reloaded_a is not None and reloaded_a.revision == 2
    assert reloaded_b is not None and reloaded_b.revision == 1


def test_record_polza_media_task_accepted_rejects_foreign_part(repository, tmp_path):
    run_a = repository.create_run(operation="tts", run_root=str(tmp_path / "a"))
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_a = repository.add_part(run_a.run_uuid, position=1)
    reserved_a = _reserve_paid_attempt(
        repository, run_a.run_uuid, part_a.part_uuid, expected_revision=1
    )

    with pytest.raises(HistoryPartNotFoundError):
        repository.record_polza_media_task_accepted(
            run_b.run_uuid,
            attempt_uuid=reserved_a.attempt_uuid,
            part_uuid=part_a.part_uuid,
            expected_revision=1,
            remote_task_id="id-1",
        )

    reloaded_b = repository.get_run(run_b.run_uuid)
    assert reloaded_b is not None and reloaded_b.revision == 1


def test_record_polza_media_task_accepted_rejects_missing_part_and_attempt(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    with pytest.raises(HistoryPartNotFoundError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=reserved.attempt_uuid,
            part_uuid="11111111-2222-3333-4444-555555555555",
            expected_revision=2,
            remote_task_id="id-1",
        )
    with pytest.raises(HistoryPaidMediaTaskConflictError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid="11111111-2222-3333-4444-555555555555",
            part_uuid=part.part_uuid,
            expected_revision=2,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None and unchanged.remote_id is None


@pytest.mark.parametrize(
    "call_type, provider, status, remote_id",
    [
        ("asr_transcript", "polza-tts", "submitting", None),
        ("tts_chunk", "openrouter-tts", "submitting", None),
        ("tts_chunk", "polza-tts", "outcome_unknown", None),
        ("tts_chunk", "polza-tts", "failed", None),
        ("tts_chunk", "polza-tts", "remote_accepted", "task-abc"),
    ],
)
def test_record_polza_media_task_accepted_rejects_wrong_type_provider_or_state(
    repository, tmp_path, call_type, provider, status, remote_id
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=call_type,
        part_uuid=part.part_uuid,
        provider=provider,
        status=status,
        remote_id=remote_id,
    )

    with pytest.raises(HistoryPaidMediaTaskConflictError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id == remote_id
    assert unchanged.status == status


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "https://cdn.example.invalid/media/SENTINEL?X-Amz-Signature=SENTINEL",
        "task\nid",
        "task id",
        "task/id",
        "a" * 129,
    ],
)
def test_record_polza_media_task_accepted_rejects_unsafe_id_without_echo(
    repository, tmp_path, unsafe_id
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    with pytest.raises(ValueError) as excinfo:
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=reserved.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=2,
            remote_task_id=unsafe_id,
        )

    # The fixed message never echoes the unsafe value.
    assert str(excinfo.value) == "remote_task_id is not a bounded opaque media task id"
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_refuses_second_or_different_id(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id="id-first",
    )

    # The accepted id is immutable: neither the same id nor a different one may
    # be rebound, and neither may advance the revision a second time.
    for duplicate in ("id-first", "id-second"):
        with pytest.raises(HistoryPaidMediaTaskConflictError):
            repository.record_polza_media_task_accepted(
                run.run_uuid,
                attempt_uuid=reserved.attempt_uuid,
                part_uuid=part.part_uuid,
                expected_revision=3,
                remote_task_id=duplicate,
            )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    reloaded = repository.get_attempt(reserved.attempt_uuid)
    assert reloaded is not None
    assert reloaded.remote_id == "id-first"
    assert reloaded.status == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED


def test_record_polza_media_task_accepted_refuses_open_outer_transaction(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    recorded = False

    # A recording inside a caller's still-open transaction could return a usable
    # accepted marker, be followed by the GET, and then be erased by a rollback.
    # The seam must refuse before it can return.
    with pytest.raises(HistoryPaidAttemptInTransactionError):
        with repository.transaction():
            repository.record_polza_media_task_accepted(
                run.run_uuid,
                attempt_uuid=reserved.attempt_uuid,
                part_uuid=part.part_uuid,
                expected_revision=2,
                remote_task_id="id-1",
            )
            recorded = True

    assert recorded is False
    assert repository._connection.in_transaction is False
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_two_connections_cannot_overwrite(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as first_database:
        first_database.migrate()
        with HistoryDatabase(database_path) as second_database:
            second_database.connect()
            first = HistoryRepository(first_database)
            second = HistoryRepository(second_database)
            run = first.create_run(operation="tts", run_root=str(tmp_path))
            first_part = first.add_part(run.run_uuid, position=1)
            second_part = first.add_part(run.run_uuid, position=2)
            first_reserved = _reserve_paid_attempt(
                first, run.run_uuid, first_part.part_uuid, expected_revision=1
            )
            second_reserved = _reserve_paid_attempt(
                first, run.run_uuid, second_part.part_uuid, expected_revision=2
            )

            first.record_polza_media_task_accepted(
                run.run_uuid,
                attempt_uuid=first_reserved.attempt_uuid,
                part_uuid=first_part.part_uuid,
                expected_revision=3,
                remote_task_id="id-first",
            )

            # The second connection's stale view cannot advance the revision or
            # bind part two's still-open attempt.
            with pytest.raises(HistoryRevisionConflictError):
                second.record_polza_media_task_accepted(
                    run.run_uuid,
                    attempt_uuid=second_reserved.attempt_uuid,
                    part_uuid=second_part.part_uuid,
                    expected_revision=2,
                    remote_task_id="id-second",
                )
            # The already accepted id is immutable even at the current revision.
            with pytest.raises(HistoryPaidMediaTaskConflictError):
                second.record_polza_media_task_accepted(
                    run.run_uuid,
                    attempt_uuid=first_reserved.attempt_uuid,
                    part_uuid=first_part.part_uuid,
                    expected_revision=4,
                    remote_task_id="id-second",
                )

            reloaded_run = second.get_run(run.run_uuid)
            assert reloaded_run is not None and reloaded_run.revision == 4
            accepted = second.get_attempt(first_reserved.attempt_uuid)
            assert accepted is not None and accepted.remote_id == "id-first"
            assert accepted.status == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
            untouched = second.get_attempt(second_reserved.attempt_uuid)
            assert untouched is not None and untouched.remote_id is None
            assert untouched.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


@pytest.mark.parametrize("bad_revision", [0, -1, True, False, 1.0, "1"])
def test_record_polza_media_task_accepted_rejects_invalid_expected_revision(
    repository, tmp_path, bad_revision
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    with pytest.raises(ValueError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=reserved.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=bad_revision,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None and unchanged.remote_id is None


def test_record_polza_media_task_accepted_rejects_synchronous_polza_model(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    # ``openai/gpt-4o-mini-tts`` is the default ``polza-tts`` model. It submits
    # through the synchronous ``/audio/speech`` route, so no recoverable async
    # media task id can exist for it even though the provider matches.
    _, reserved = repository.reserve_paid_tts_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        account_alias="main",
    )

    with pytest.raises(HistoryPaidMediaTaskConflictError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=reserved.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=2,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_refuses_legacy_imported_run(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        run, _ = repository.create_legacy_run(
            operation="tts",
            run_root=legacy_root,
            legacy_source_root=legacy_root,
            status="interrupted",
        )
        part = repository.add_part(run.run_uuid, position=1)
        # A part-linked submitting marker on an imported run is not a recoverable
        # async task; the run-level legacy guard must refuse it like the
        # pre-submit reservation does.
        attempt = repository.add_attempt(
            run.run_uuid,
            call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
            part_uuid=part.part_uuid,
            provider="polza-tts",
            model="elevenlabs/eleven_multilingual_v2",
            status=history_repository_module.ATTEMPT_STATUS_SUBMITTING,
        )

    with pytest.raises(HistoryRunNotReservableError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert reloaded_run.legacy_source_root is not None
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_refuses_completed_run(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="completed")
    part = repository.add_part(run.run_uuid, position=1)
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        provider="polza-tts",
        model="elevenlabs/eleven_multilingual_v2",
        status=history_repository_module.ATTEMPT_STATUS_SUBMITTING,
    )

    # Finishing a completed run's stale marker belongs to a new run, so the
    # accepted-id transition must leave the closed run and attempt unchanged.
    with pytest.raises(HistoryRunNotReservableError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


def test_record_polza_media_task_accepted_fails_closed_when_guarded_update_is_ignored(
    repository, tmp_path
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)

    # A ``BEFORE UPDATE`` trigger that silently skips the row models any database
    # condition under which the guarded UPDATE affects no row. The seam must not
    # then return a successful accepted id with an advanced revision: it raises
    # and rolls the revision bump back together with the binding.
    repository._connection.execute(
        "CREATE TRIGGER attempts_ignore_accept BEFORE UPDATE ON attempts "
        "WHEN NEW.status = 'remote_accepted' "
        "BEGIN SELECT RAISE(IGNORE); END"
    )

    with pytest.raises(HistoryPaidMediaTaskConflictError):
        repository.record_polza_media_task_accepted(
            run.run_uuid,
            attempt_uuid=reserved.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=2,
            remote_task_id="id-1",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 2
    unchanged = repository.get_attempt(reserved.attempt_uuid)
    assert unchanged is not None
    assert unchanged.remote_id is None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_SUBMITTING


# -- paid Polza Media observed cost -----------------------------------------


def _accepted_media_attempt(repository, tmp_path, *, remote_task_id="media_task_01"):
    """Reserve and accept one paid Polza Media attempt as observation setup."""
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    _, accepted = repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id=remote_task_id,
    )
    return run, part, accepted


def test_record_polza_media_observed_cost_keeps_exact_string_and_advances_revision(
    repository, tmp_path
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    advanced, observed = repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    assert advanced.run_uuid == run.run_uuid
    assert advanced.revision == 4
    # The exact lexeme keeps its trailing zeros instead of collapsing to "0.1".
    assert observed.cost.amount == "0.1000"
    assert observed.cost.currency == "RUB"
    assert observed.cost.source == COST_SOURCE_EXACT
    assert observed.cost.exact_available is True
    # The accepted id and status are untouched by the cost observation.
    assert observed.remote_id == "media_task_01"
    assert observed.status == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
    assert observed.provider == "polza-tts"


@pytest.mark.parametrize(
    "amount, expected_text",
    [
        (Decimal("0.0831"), "0.0831"),
        (Decimal("0.1000"), "0.1000"),
        (12, "12"),
        ("3.14", "3.14"),
    ],
)
def test_record_polza_media_observed_cost_accepts_exact_decimal_int_and_string(
    repository, tmp_path, amount, expected_text
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    _, observed = repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount=amount,
    )

    assert observed.cost.amount == expected_text
    assert observed.cost.source == COST_SOURCE_EXACT
    assert observed.cost.exact_available is True


@pytest.mark.parametrize("amount, expected_text", [(0, "0"), ("0.0000", "0.0000")])
def test_record_polza_media_observed_cost_zero_is_known_not_unknown(
    repository, tmp_path, amount, expected_text
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    _, observed = repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount=amount,
    )

    # A reported zero is a real billing fact, not the unknown NULL cost.
    assert observed.cost.amount == expected_text
    assert observed.cost.amount is not None
    assert observed.cost.source == COST_SOURCE_EXACT
    assert observed.cost.exact_available is True


def test_record_polza_media_observed_cost_float_carries_approximate_provenance(
    repository, tmp_path
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    _, observed = repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount=0.1,
    )

    # A binary float has no exact decimal representation: its shortest text is
    # stored with a dedicated source and no exact guarantee, never as legacy
    # import and never as a false exact value.
    assert observed.cost.amount == repr(0.1)
    assert observed.cost.amount == "0.1"
    assert observed.cost.raw == "0.1"
    assert observed.cost.currency == "RUB"
    assert observed.cost.source == COST_SOURCE_PROVIDER_OBSERVED_FLOAT
    assert observed.cost.source != COST_SOURCE_LEGACY_FLOAT
    assert observed.cost.source != COST_SOURCE_EXACT
    assert observed.cost.exact_available is False


def test_record_polza_media_observed_cost_commits_before_signed_url_download(repository, tmp_path):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    # A separate connection already sees the committed cost, so a crash during
    # the synthetic signed-URL download cannot lose the reported amount.
    connection = sqlite3.connect(tmp_path / "history.sqlite3")
    try:
        revision = connection.execute(
            "SELECT revision FROM runs WHERE run_uuid = ?", (run.run_uuid,)
        ).fetchone()[0]
        row = connection.execute(
            "SELECT cost, cost_currency, cost_source, cost_exact_available, status, remote_id "
            "FROM attempts WHERE attempt_uuid = ?",
            (accepted.attempt_uuid,),
        ).fetchone()
    finally:
        connection.close()

    assert revision == 4
    assert row[0] == "0.1000"
    assert row[1] == "RUB"
    assert row[2] == COST_SOURCE_EXACT
    assert row[3] == 1
    assert row[4] == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
    assert row[5] == "media_task_01"


def test_record_polza_media_observed_cost_repeats_identical_without_bump(repository, tmp_path):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)
    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    # The same billed amount polled again is one observation, not a second
    # charge: it stays idempotent and must not advance the revision.
    current, repeated = repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=4,
        amount="0.1000",
    )

    assert current.revision == 4
    assert repeated.cost.amount == "0.1000"
    assert repeated.cost.source == COST_SOURCE_EXACT
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 4


def test_record_polza_media_observed_cost_repeat_with_stale_revision_conflicts(
    repository, tmp_path
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)
    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    with pytest.raises(HistoryRevisionConflictError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=3,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 4


def test_record_polza_media_observed_cost_refuses_a_different_price(repository, tmp_path):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)
    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    with pytest.raises(HistoryPaidMediaCostConflictError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=4,
            amount="0.2",
        )

    # The first billing observation is never silently overwritten.
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 4
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None
    assert unchanged.cost.amount == "0.1000"
    assert unchanged.cost.source == COST_SOURCE_EXACT


def test_record_polza_media_observed_cost_stale_revision_leaves_attempt_unknown(
    repository, tmp_path
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    with pytest.raises(HistoryRevisionConflictError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=2,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None
    assert unchanged.cost.amount is None
    assert unchanged.cost.source == COST_SOURCE_UNKNOWN
    assert unchanged.cost.exact_available is False


def test_record_polza_media_observed_cost_none_never_erases_observed_amount(repository, tmp_path):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)
    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=accepted.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1000",
    )

    # A later poll that omits usage must not blank the observed amount.
    with pytest.raises(ValueError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=4,
            amount=None,
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 4
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount == "0.1000"


@pytest.mark.parametrize(
    "call_type, provider, model, status, remote_id",
    [
        (
            "asr_transcript",
            "polza-tts",
            "elevenlabs/eleven_multilingual_v2",
            "remote_accepted",
            "t1",
        ),
        (
            "tts_chunk",
            "openrouter-tts",
            "elevenlabs/eleven_multilingual_v2",
            "remote_accepted",
            "t1",
        ),
        ("tts_chunk", "polza-tts", "openai/gpt-4o-mini-tts", "remote_accepted", "t1"),
        ("tts_chunk", "polza-tts", "elevenlabs/eleven_multilingual_v2", "submitting", None),
        ("tts_chunk", "polza-tts", "elevenlabs/eleven_multilingual_v2", "remote_accepted", None),
        (
            "tts_chunk",
            "polza-tts",
            "elevenlabs/eleven_multilingual_v2",
            "remote_accepted",
            "https://cdn.example.invalid/media?X-Amz-Signature=SENTINEL",
        ),
    ],
)
def test_record_polza_media_observed_cost_rejects_wrong_identity_state_or_remote_id(
    repository, tmp_path, call_type, provider, model, status, remote_id
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1)
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=call_type,
        part_uuid=part.part_uuid,
        provider=provider,
        model=model,
        status=status,
        remote_id=remote_id,
    )

    with pytest.raises(HistoryPaidMediaCostConflictError) as excinfo:
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            amount="0.1000",
        )

    assert "SENTINEL" not in str(excinfo.value)
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


def test_record_polza_media_observed_cost_rejects_foreign_run_attempt(repository, tmp_path):
    run_a, part_a, accepted_a = _accepted_media_attempt(repository, tmp_path / "a")
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_b = repository.add_part(run_b.run_uuid, position=1)

    with pytest.raises(HistoryPaidMediaCostConflictError):
        repository.record_polza_media_observed_cost(
            run_b.run_uuid,
            attempt_uuid=accepted_a.attempt_uuid,
            part_uuid=part_b.part_uuid,
            expected_revision=1,
            amount="0.1000",
        )

    unchanged_a = repository.get_attempt(accepted_a.attempt_uuid)
    assert unchanged_a is not None and unchanged_a.cost.amount is None
    reloaded_a = repository.get_run(run_a.run_uuid)
    reloaded_b = repository.get_run(run_b.run_uuid)
    assert reloaded_a is not None and reloaded_a.revision == 3
    assert reloaded_b is not None and reloaded_b.revision == 1


def test_record_polza_media_observed_cost_rejects_foreign_or_missing_part_and_attempt(
    repository, tmp_path
):
    run_a, _, accepted_a = _accepted_media_attempt(repository, tmp_path / "a")
    run_b = repository.create_run(operation="tts", run_root=str(tmp_path / "b"))
    part_b = repository.add_part(run_b.run_uuid, position=1)

    with pytest.raises(HistoryPartNotFoundError):
        repository.record_polza_media_observed_cost(
            run_a.run_uuid,
            attempt_uuid=accepted_a.attempt_uuid,
            part_uuid=part_b.part_uuid,
            expected_revision=3,
            amount="0.1000",
        )
    with pytest.raises(HistoryPartNotFoundError):
        repository.record_polza_media_observed_cost(
            run_a.run_uuid,
            attempt_uuid=accepted_a.attempt_uuid,
            part_uuid="11111111-2222-3333-4444-555555555555",
            expected_revision=3,
            amount="0.1000",
        )
    with pytest.raises(HistoryPaidMediaCostConflictError):
        repository.record_polza_media_observed_cost(
            run_b.run_uuid,
            attempt_uuid="11111111-2222-3333-4444-555555555555",
            part_uuid=part_b.part_uuid,
            expected_revision=1,
            amount="0.1000",
        )

    reloaded_a = repository.get_run(run_a.run_uuid)
    reloaded_b = repository.get_run(run_b.run_uuid)
    assert reloaded_a is not None and reloaded_a.revision == 3
    assert reloaded_b is not None and reloaded_b.revision == 1


@pytest.mark.parametrize(
    "bad_amount",
    [
        None,
        True,
        False,
        "not-a-decimal",
        "https://cdn.example.invalid/media?X-Amz-Signature=SENTINEL",
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        {"cost_rub": 1},
        [1],
    ],
)
def test_record_polza_media_observed_cost_rejects_unusable_amount_without_echo(
    repository, tmp_path, bad_amount
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    with pytest.raises(ValueError) as excinfo:
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=3,
            amount=bad_amount,
        )

    # The fixed message never echoes an unsafe value such as a signed URL.
    assert str(excinfo.value) == (
        "observed media cost must be an exact Decimal, int, or decimal string, or a finite float"
    )
    assert "SENTINEL" not in str(excinfo.value)
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


@pytest.mark.parametrize("bad_revision", [0, -1, True, False, 1.0, "3"])
def test_record_polza_media_observed_cost_rejects_invalid_expected_revision(
    repository, tmp_path, bad_revision
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    with pytest.raises(ValueError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=bad_revision,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


def test_record_polza_media_observed_cost_refuses_open_outer_transaction(repository, tmp_path):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)
    observed = False

    # An observation inside a caller's still-open transaction could be followed
    # by the download and then erased by a rollback, so the seam must refuse.
    with pytest.raises(HistoryPaidAttemptInTransactionError):
        with repository.transaction():
            repository.record_polza_media_observed_cost(
                run.run_uuid,
                attempt_uuid=accepted.attempt_uuid,
                part_uuid=part.part_uuid,
                expected_revision=3,
                amount="0.1000",
            )
            observed = True

    assert observed is False
    assert repository._connection.in_transaction is False
    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


def test_record_polza_media_observed_cost_refuses_legacy_imported_run(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        run, _ = repository.create_legacy_run(
            operation="tts",
            run_root=legacy_root,
            legacy_source_root=legacy_root,
            status="interrupted",
        )
        part = repository.add_part(run.run_uuid, position=1)
        attempt = repository.add_attempt(
            run.run_uuid,
            call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
            part_uuid=part.part_uuid,
            provider="polza-tts",
            model="elevenlabs/eleven_multilingual_v2",
            status=history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED,
            remote_id="legacy-task",
        )

    with pytest.raises(HistoryRunNotReservableError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    assert reloaded_run.legacy_source_root is not None
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


def test_record_polza_media_observed_cost_refuses_completed_run(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="completed")
    part = repository.add_part(run.run_uuid, position=1)
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        provider="polza-tts",
        model="elevenlabs/eleven_multilingual_v2",
        status=history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED,
        remote_id="completed-task",
    )

    with pytest.raises(HistoryRunNotReservableError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=1,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 1
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None and unchanged.cost.amount is None


def test_record_polza_media_observed_cost_fails_closed_when_guarded_update_is_ignored(
    repository, tmp_path
):
    run, part, accepted = _accepted_media_attempt(repository, tmp_path)

    # A ``BEFORE UPDATE`` trigger that silently skips the row models any database
    # condition under which the guarded cost UPDATE affects no row. The seam must
    # not then return a recorded amount with an advanced revision: it raises and
    # rolls the revision bump back together with the cost write.
    repository._connection.execute(
        "CREATE TRIGGER attempts_ignore_observed_cost BEFORE UPDATE ON attempts "
        "WHEN NEW.cost IS NOT NULL "
        "BEGIN SELECT RAISE(IGNORE); END"
    )

    with pytest.raises(HistoryPaidMediaCostConflictError):
        repository.record_polza_media_observed_cost(
            run.run_uuid,
            attempt_uuid=accepted.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=3,
            amount="0.1000",
        )

    reloaded_run = repository.get_run(run.run_uuid)
    assert reloaded_run is not None and reloaded_run.revision == 3
    unchanged = repository.get_attempt(accepted.attempt_uuid)
    assert unchanged is not None
    assert unchanged.cost.amount is None
    assert unchanged.cost.source == COST_SOURCE_UNKNOWN


# -- paid Polza Media verified raw artifact link ----------------------------

_RAW_FINGERPRINT = "a" * 64
_RAW_AUDIO = b"PAID-RAW-MARKER" + bytes(range(16))
_RAW_REMOTE_TASK_ID = "media_task_01"
_RAW_GENERATION_ID = "gen-media-123"


def _write_media_receipt(tmp_path, attempt_uuid, part_uuid, overrides=None):
    """Write the bounded paid raw receipt for a synthetic media attempt."""
    kwargs = {
        "run_root": str(tmp_path),
        "attempt_uuid": attempt_uuid,
        "part_uuid": part_uuid,
        "synthesis_fingerprint": _RAW_FINGERPRINT,
        "chunk_id": "chunk_01",
        "number": 1,
        "audio_format": "mp3",
        "audio_bytes": _RAW_AUDIO,
        "remote_task_id": _RAW_REMOTE_TASK_ID,
        "generation_id": _RAW_GENERATION_ID,
    }
    if overrides:
        kwargs.update(overrides)
    return write_paid_raw_receipt(**kwargs)


def _accepted_media_attempt_with_raw(repository, tmp_path, **overrides):
    """Reserve, accept, and save raw evidence for one paid media attempt."""
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    _, accepted = repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id=_RAW_REMOTE_TASK_ID,
    )
    receipt = _write_media_receipt(
        tmp_path, accepted.attempt_uuid, part.part_uuid, overrides=overrides
    )
    return run, part, accepted, receipt


def _record_raw_saved(repository, run, part, attempt, receipt, *, expected_revision=3):
    """Call the raw-saved seam for the synthetic media fixture."""
    return repository.record_polza_media_raw_saved(
        run.run_uuid,
        attempt_uuid=attempt.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=expected_revision,
        receipt=receipt,
    )


def _assert_no_raw_link(repository, run, attempt, *, revision=3, status=None):
    """Assert the raw-saved seam left run, attempt, parts, and artifacts unchanged."""
    reloaded = repository.get_run(run.run_uuid)
    assert reloaded is not None and reloaded.revision == revision
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None
    if status is not None:
        assert unchanged.status == status
    assert repository.get_artifacts(run.run_uuid) == []
    assert all(stored_part.stage is None for stored_part in repository.get_parts(run.run_uuid))


def test_record_polza_media_raw_saved_links_verified_evidence(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    advanced, updated, artifact = _record_raw_saved(repository, run, part, attempt, receipt)

    assert advanced.run_uuid == run.run_uuid
    assert advanced.revision == 4
    assert updated.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED
    # The immutable accepted remote id, provider identity, and still-unknown cost
    # survive the transition untouched.
    assert updated.remote_id == _RAW_REMOTE_TASK_ID
    assert updated.provider == "polza-tts"
    assert updated.model == "elevenlabs/eleven_multilingual_v2"
    assert updated.cost.amount is None
    assert updated.cost.source == COST_SOURCE_UNKNOWN

    artifacts = repository.get_artifacts(run.run_uuid)
    assert [stored.artifact_uuid for stored in artifacts] == [artifact.artifact_uuid]
    assert artifact.role == history_repository_module.ARTIFACT_ROLE_PAID_RAW_AUDIO
    assert artifact.path_kind == PATH_KIND_MANAGED_RELATIVE
    assert artifact.path == "raw/chunk_01.mp3"
    assert artifact.part_uuid == part.part_uuid
    assert artifact.attempt_uuid == attempt.attempt_uuid
    assert artifact.sha256 == receipt.sha256
    assert artifact.size_bytes == len(_RAW_AUDIO)
    assert artifact.mime == "audio/mpeg"
    assert artifact.media_metadata == {
        "format": "mp3",
        "chunk_id": "chunk_01",
        "number": 1,
        "generation_id": _RAW_GENERATION_ID,
    }
    # The part stage records the paid progression only while it was unset.
    stored_part = repository.get_parts(run.run_uuid)[0]
    assert stored_part.stage == history_repository_module.PART_STAGE_RAW_SAVED


def test_record_polza_media_raw_saved_preserves_observed_cost(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    repository.record_polza_media_observed_cost(
        run.run_uuid,
        attempt_uuid=attempt.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=3,
        amount="0.1250",
    )

    advanced, updated, _artifact = _record_raw_saved(
        repository, run, part, attempt, receipt, expected_revision=4
    )

    assert advanced.revision == 5
    assert updated.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED
    assert updated.cost.amount == "0.1250"
    assert updated.cost.source == COST_SOURCE_EXACT
    assert updated.cost.exact_available is True
    assert updated.remote_id == _RAW_REMOTE_TASK_ID


def test_record_polza_media_raw_saved_hashes_outside_any_transaction(
    repository, tmp_path, monkeypatch
):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    observed: list[bool] = []
    real_sha256_file = raw_receipt_module._sha256_file

    def recording_sha256_file(path, *, error, label):
        observed.append(repository._connection.in_transaction)
        return real_sha256_file(path, error=error, label=label)

    monkeypatch.setattr(raw_receipt_module, "_sha256_file", recording_sha256_file)

    _record_raw_saved(repository, run, part, attempt, receipt)

    assert observed, "the raw hashing was not instrumented"
    assert all(in_transaction is False for in_transaction in observed)


def test_record_polza_media_raw_saved_repeat_is_idempotent_without_bump(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    first_run, _first_attempt, first_artifact = _record_raw_saved(
        repository, run, part, attempt, receipt
    )
    assert first_run.revision == 4

    # The repository owns no JSON compatibility export, so a lost or failed
    # export after the commit cannot roll back this durable row. A repeated
    # verified call at the current revision is a safe no-op: same revision, one
    # artifact, and untouched bytes on disk.
    second_run, second_attempt, second_artifact = _record_raw_saved(
        repository, run, part, attempt, receipt, expected_revision=4
    )

    assert second_run.revision == 4
    assert second_attempt.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED
    assert second_artifact.artifact_uuid == first_artifact.artifact_uuid
    assert len(repository.get_artifacts(run.run_uuid)) == 1
    assert receipt.raw_path.read_bytes() == _RAW_AUDIO


def test_record_polza_media_raw_saved_rejects_receipt_number_off_part_position(
    repository, tmp_path
):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
    # A second part with the identical fingerprint is indistinguishable by
    # fingerprint alone, so only the receipt number and the part position bind
    # the paid bytes to the right chunk.
    repository.add_part(run.run_uuid, position=2, fingerprint=_RAW_FINGERPRINT)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    _, accepted = repository.record_polza_media_task_accepted(
        run.run_uuid,
        attempt_uuid=reserved.attempt_uuid,
        part_uuid=part.part_uuid,
        expected_revision=2,
        remote_task_id=_RAW_REMOTE_TASK_ID,
    )
    # This is a valid bounded receipt for chunk 2 that names part position 1 and
    # that part's fingerprint, so only a positional cross-check can refuse it.
    receipt = _write_media_receipt(
        tmp_path,
        accepted.attempt_uuid,
        part.part_uuid,
        overrides={"number": 2, "chunk_id": "chunk_02"},
    )

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, accepted, receipt)

    _assert_no_raw_link(repository, run, accepted)


@pytest.mark.parametrize("case", ["part_uuid", "availability", "mime", "media_metadata"])
def test_record_polza_media_raw_saved_rejects_inconsistent_existing_artifact(
    repository, tmp_path, case
):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    _record_raw_saved(repository, run, part, attempt, receipt)
    assert repository.get_run(run.run_uuid).revision == 4

    if case == "part_uuid":
        other = repository.add_part(run.run_uuid, position=2, fingerprint=_RAW_FINGERPRINT)
        statement = "UPDATE artifacts SET part_uuid = ? WHERE attempt_uuid = ?"
        parameters = (other.part_uuid, attempt.attempt_uuid)
    elif case == "availability":
        statement = "UPDATE artifacts SET availability = ? WHERE attempt_uuid = ?"
        parameters = (AVAILABILITY_MISSING, attempt.attempt_uuid)
    elif case == "mime":
        statement = "UPDATE artifacts SET mime = ? WHERE attempt_uuid = ?"
        parameters = ("audio/ogg", attempt.attempt_uuid)
    else:
        statement = "UPDATE artifacts SET media_metadata_json = ? WHERE attempt_uuid = ?"
        parameters = (
            json.dumps(
                {
                    "format": "mp3",
                    "chunk_id": "chunk_02",
                    "number": 2,
                    "generation_id": _RAW_GENERATION_ID,
                }
            ),
            attempt.attempt_uuid,
        )
    repository._connection.execute(statement, parameters)

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=4)

    reloaded = repository.get_run(run.run_uuid)
    assert reloaded is not None and reloaded.revision == 4
    assert len(repository.get_artifacts(run.run_uuid)) == 1
    unchanged = repository.get_attempt(attempt.attempt_uuid)
    assert unchanged is not None
    assert unchanged.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED


def test_record_polza_media_raw_saved_recovers_after_pre_commit_crash(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    # A failing artifact INSERT models a crash between the file+receipt write and
    # the database commit: the raw evidence stays on disk while the transaction
    # rolls back the artifact, the attempt status, the part stage, and the bump.
    repository._connection.execute(
        "CREATE TRIGGER artifacts_crash_insert BEFORE INSERT ON artifacts "
        "BEGIN SELECT RAISE(ABORT, 'synthetic pre-commit crash'); END"
    )

    with pytest.raises(sqlite3.Error):
        _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)
    assert receipt.raw_path.read_bytes() == _RAW_AUDIO
    assert (receipt.raw_path.parent / f"{receipt.raw_path.name}.receipt.json").exists()

    # After the fault is removed the same evidence links with no provider call:
    # the repository has no provider, network, or FFmpeg path.
    repository._connection.execute("DROP TRIGGER artifacts_crash_insert")
    advanced, updated, artifact = _record_raw_saved(repository, run, part, attempt, receipt)

    assert advanced.revision == 4
    assert updated.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED
    assert artifact.path == "raw/chunk_01.mp3"
    assert len(repository.get_artifacts(run.run_uuid)) == 1


def test_record_polza_media_raw_saved_fails_closed_when_guarded_update_is_ignored(
    repository, tmp_path
):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    # A ``BEFORE UPDATE`` trigger that silently skips the row models any database
    # condition under which the guarded status UPDATE affects no row. The seam
    # must not then return a linked artifact with an advanced revision: it raises
    # and rolls the artifact, part stage, and revision bump back together.
    repository._connection.execute(
        "CREATE TRIGGER attempts_ignore_raw_saved BEFORE UPDATE ON attempts "
        "WHEN NEW.status = 'raw_saved' BEGIN SELECT RAISE(IGNORE); END"
    )

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)


@pytest.mark.parametrize(
    "overrides",
    [
        {"synthesis_fingerprint": "b" * 64},
        {"remote_task_id": "a-different-task"},
        {"part_uuid": "33333333-3333-4333-8333-333333333333"},
    ],
)
def test_record_polza_media_raw_saved_rejects_mismatched_receipt_identity(
    repository, tmp_path, overrides
):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(
        repository, tmp_path, **overrides
    )

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_tampered_digest(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    receipt.raw_path.write_bytes(b"PAID-RAW-MARKER-tampered")

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)


@pytest.mark.parametrize("half", ["raw", "receipt"])
def test_record_polza_media_raw_saved_rejects_missing_evidence(repository, tmp_path, half):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    if half == "raw":
        receipt.raw_path.unlink()
    else:
        (receipt.raw_path.parent / f"{receipt.raw_path.name}.receipt.json").unlink()

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_stale_revision_without_mutation(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    with pytest.raises(HistoryRevisionConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=2)

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_attempt_not_remote_accepted(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
    reserved = _reserve_paid_attempt(repository, run.run_uuid, part.part_uuid, expected_revision=1)
    receipt = _write_media_receipt(tmp_path, reserved.attempt_uuid, part.part_uuid)

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, reserved, receipt, expected_revision=2)

    _assert_no_raw_link(
        repository,
        run,
        reserved,
        revision=2,
        status=history_repository_module.ATTEMPT_STATUS_SUBMITTING,
    )


def test_record_polza_media_raw_saved_rejects_foreign_part(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
    foreign = repository.add_part(run.run_uuid, position=2, fingerprint=_RAW_FINGERPRINT)

    with pytest.raises(HistoryPaidRawConflictError):
        repository.record_polza_media_raw_saved(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=foreign.part_uuid,
            expected_revision=3,
            receipt=receipt,
        )

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_missing_part(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    with pytest.raises(HistoryPartNotFoundError):
        repository.record_polza_media_raw_saved(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid="11111111-2222-3333-4444-555555555555",
            expected_revision=3,
            receipt=receipt,
        )

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_synchronous_polza_model(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path))
    part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
    # ``openai/gpt-4o-mini-tts`` submits through the synchronous
    # ``/audio/speech`` route, so it can never hold a recoverable async media
    # task id even though the provider matches.
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        status=history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED,
        remote_id="sync-task",
    )
    receipt = _write_media_receipt(
        tmp_path, attempt.attempt_uuid, part.part_uuid, overrides={"remote_task_id": "sync-task"}
    )

    with pytest.raises(HistoryPaidRawConflictError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=1)

    _assert_no_raw_link(repository, run, attempt, revision=1)


def test_record_polza_media_raw_saved_refuses_open_outer_transaction(repository, tmp_path):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    with pytest.raises(HistoryPaidAttemptInTransactionError):
        with repository.transaction():
            _record_raw_saved(repository, run, part, attempt, receipt)

    _assert_no_raw_link(repository, run, attempt)


@pytest.mark.parametrize("bad_revision", [0, -1, True, False, 1.0, "3"])
def test_record_polza_media_raw_saved_rejects_invalid_expected_revision(
    repository, tmp_path, bad_revision
):
    run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    with pytest.raises(ValueError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=bad_revision)

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_rejects_non_receipt(repository, tmp_path):
    run, part, attempt, _receipt = _accepted_media_attempt_with_raw(repository, tmp_path)

    with pytest.raises(ValueError):
        repository.record_polza_media_raw_saved(
            run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.part_uuid,
            expected_revision=3,
            receipt="not-a-receipt",
        )

    _assert_no_raw_link(repository, run, attempt)


def test_record_polza_media_raw_saved_refuses_legacy_imported_run(repository, tmp_path):
    legacy_root = str(tmp_path / "legacy-run")
    with repository.transaction():
        run, _ = repository.create_legacy_run(
            operation="tts",
            run_root=legacy_root,
            legacy_source_root=legacy_root,
            status="interrupted",
        )
        part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
        attempt = repository.add_attempt(
            run.run_uuid,
            call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
            part_uuid=part.part_uuid,
            provider="polza-tts",
            model="elevenlabs/eleven_multilingual_v2",
            status=history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED,
            remote_id="legacy-task",
        )
    # An imported run's paid work belongs to a run-level total, so the guard must
    # refuse before any receipt verification or database mutation.
    receipt = _write_media_receipt(
        tmp_path, attempt.attempt_uuid, part.part_uuid, overrides={"remote_task_id": "legacy-task"}
    )

    with pytest.raises(HistoryRunNotReservableError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=1)

    _assert_no_raw_link(repository, run, attempt, revision=1)


def test_record_polza_media_raw_saved_refuses_completed_run(repository, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path), status="completed")
    part = repository.add_part(run.run_uuid, position=1, fingerprint=_RAW_FINGERPRINT)
    attempt = repository.add_attempt(
        run.run_uuid,
        call_type=history_repository_module.ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        provider="polza-tts",
        model="elevenlabs/eleven_multilingual_v2",
        status=history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED,
        remote_id="completed-task",
    )
    receipt = _write_media_receipt(
        tmp_path,
        attempt.attempt_uuid,
        part.part_uuid,
        overrides={"remote_task_id": "completed-task"},
    )

    with pytest.raises(HistoryRunNotReservableError):
        _record_raw_saved(repository, run, part, attempt, receipt, expected_revision=1)

    _assert_no_raw_link(repository, run, attempt, revision=1)


def test_record_polza_media_raw_saved_two_connections_stale_cannot_duplicate(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as first_database:
        first_database.migrate()
        with HistoryDatabase(database_path) as second_database:
            second_database.connect()
            first = HistoryRepository(first_database)
            second = HistoryRepository(second_database)
            run, part, attempt, receipt = _accepted_media_attempt_with_raw(first, tmp_path)
            first.record_polza_media_raw_saved(
                run.run_uuid,
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.part_uuid,
                expected_revision=3,
                receipt=receipt,
            )

            # The second connection's stale view cannot link a second artifact or
            # advance the already-committed revision.
            with pytest.raises(HistoryRevisionConflictError):
                second.record_polza_media_raw_saved(
                    run.run_uuid,
                    attempt_uuid=attempt.attempt_uuid,
                    part_uuid=part.part_uuid,
                    expected_revision=3,
                    receipt=receipt,
                )

            reloaded_run = second.get_run(run.run_uuid)
            assert reloaded_run is not None and reloaded_run.revision == 4
            assert len(second.get_artifacts(run.run_uuid)) == 1
            saved = second.get_attempt(attempt.attempt_uuid)
            assert saved is not None
            assert saved.status == history_repository_module.ATTEMPT_STATUS_RAW_SAVED


def test_record_polza_media_raw_saved_rejects_run_root_change_after_verify(
    repository, tmp_path, monkeypatch
):
    # The verified evidence is read from the run_root fetched before the
    # transaction. A concurrent writer can move runs.run_root between that file
    # verification and BEGIN IMMEDIATE without bumping the run revision, which
    # would leave the linked relative artifact path pointing into a root whose
    # bytes were never verified. Deterministically reproduce that interleaving by
    # letting the verification seam run for real, then moving the root through a
    # separate connection before the transaction starts.
    moved_root = tmp_path / "moved-root"
    moved_root.mkdir()
    with HistoryDatabase(tmp_path / "history.sqlite3") as second_database:
        second_database.connect()
        second = HistoryRepository(second_database)
        run, part, attempt, receipt = _accepted_media_attempt_with_raw(repository, tmp_path)
        original_bytes = receipt.raw_path.read_bytes()
        real_verify = repository._verify_media_raw_receipt

        def verify_then_move_run_root(run_record, part_record, attempt_record, raw_receipt):
            verified = real_verify(run_record, part_record, attempt_record, raw_receipt)
            # Model the race: the on-disk evidence verified against the old
            # run_root, and now the root moves without a revision bump.
            second._connection.execute(
                "UPDATE runs SET run_root = ? WHERE run_uuid = ?",
                (str(moved_root), run_record.run_uuid),
            )
            moved = second.get_run(run_record.run_uuid)
            assert moved is not None and moved.run_root == str(moved_root)
            assert moved.revision == run_record.revision
            return verified

        monkeypatch.setattr(repository, "_verify_media_raw_receipt", verify_then_move_run_root)

        with pytest.raises(HistoryPaidRawConflictError):
            _record_raw_saved(repository, run, part, attempt, receipt)

        # No link, bump, or stage change reached the database, and the bytes
        # stay in the original, verified root untouched.
        _assert_no_raw_link(second, run, attempt)
        saved = second.get_attempt(attempt.attempt_uuid)
        assert saved is not None
        assert saved.status == history_repository_module.ATTEMPT_STATUS_REMOTE_ACCEPTED
        assert receipt.raw_path.read_bytes() == original_bytes
        assert (receipt.raw_path.parent / f"{receipt.raw_path.name}.receipt.json").exists()
        assert not (moved_root / "raw").exists()
