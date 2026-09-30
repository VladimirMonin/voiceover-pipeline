"""Offline tests for observed ASR timing provenance and safe FTS5 search ranges.

Plan section 7 stage S07 records the transcript an ASR command observed together
with exact integer millisecond offsets, and plan section 8 stage S08 derives the
lexical chunk ranges from that same canonical evidence. These tests cover the
fail-first contract:

* a long-form word offset converts exactly (``10.125`` s -> ``10125`` ms);
* a text-only result maps to no span, so its chunk ranges stay ``NULL``;
* a mixed timed/untimed result keeps only the observed intervals;
* an unmappable or malformed mapping fails closed instead of corrupting the index;
* the run snapshot persists the observed origin, model path, and model revision;
* the FTS ranges are derived from canonical SQLite, and an offline rebuild after a
  history import reproduces them byte-for-byte;
* search and history logs never carry the query or the transcript text.

Everything is synthetic, uses a temporary SQLite database, and needs no model,
key, network, or user file.
"""

from __future__ import annotations

import logging
import uuid

import pytest

from voiceover_pipeline.asr_timing_map import (
    ObservedTimingSpan,
    build_observed_timing,
    chunk_time_range,
    observed_spans_from_snapshot,
    seconds_to_ms,
)
from voiceover_pipeline.commands import search as search_commands
from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.repository import (
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
)
from voiceover_pipeline.models import (
    ASRExecutionReceipt,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
)
from voiceover_pipeline.search import indexing
from voiceover_pipeline.search.lexical import roles_for_scope, search_lexical

MODEL = "Qwen/Qwen3-ASR-0.6B"
MODEL_PATH = "/opt/models/qwen3-asr-0.6b"
MODEL_REVISION = "rev-0.6b-2026-09"
BOUND_ASR_SOURCE_UUID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def database(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as db:
        db.migrate()
        yield db


@pytest.fixture
def repository(database):
    return HistoryRepository(database)


def _execution() -> ASRExecutionReceipt:
    return ASRExecutionReceipt(
        runtime="fixture-runtime",
        runtime_version="1.0",
        model_revision=MODEL_REVISION,
        model_path=MODEL_PATH,
        resolved_device="cpu",
        resolved_compute="float32",
    )


def _result(
    transcript: str,
    *,
    words: tuple[ASRWordSpan, ...] = (),
    segments: tuple[ASRSegment, ...] = (),
    origin: str | None = None,
) -> ASRResult:
    return ASRResult(
        transcript=transcript,
        provider_id="qwen-local",
        model_id=MODEL,
        language="ru",
        segments=segments,
        words=words,
        alignment_origin=origin,
        execution=_execution(),
    )


def _snapshot(block: dict) -> dict:
    return {
        "operation_origin": "native_asr",
        "observed_spans": {"text_source_uuid": BOUND_ASR_SOURCE_UUID, **block},
    }


def _add_bound_transcript(repository, run_uuid: str, content: str):
    return repository.add_text_source(
        run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content=content,
        text_source_uuid=BOUND_ASR_SOURCE_UUID,
    )


def _chunk_ranges(connection, run_uuid):
    return connection.execute(
        "SELECT chunk_index, char_start, char_end, start_ms, end_ms FROM search_chunks "
        "WHERE run_uuid = ? ORDER BY chunk_index",
        (run_uuid,),
    ).fetchall()


def _search(connection, query):
    return search_lexical(connection, query, limit=20, roles=roles_for_scope("speech"))


# -- mapping ------------------------------------------------------------------


def test_seconds_to_ms_converts_only_observed_values():
    assert seconds_to_ms(10.125) == 10125
    assert seconds_to_ms(0.5) == 500
    assert seconds_to_ms(0.0) == 0
    assert seconds_to_ms(None) is None
    assert seconds_to_ms(-1.0) is None
    assert seconds_to_ms(float("nan")) is None
    assert seconds_to_ms(float("inf")) is None


def test_word_spans_place_ms_offsets_on_the_transcript():
    result = _result(
        "привет мир",
        words=(
            ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
            ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
        ),
        origin="forced",
    )

    block = build_observed_timing(result)

    assert block["origin"] == "forced"
    assert block["unit"] == "ms"
    assert block["model"] == MODEL
    assert block["model_path"] == MODEL_PATH
    assert block["model_revision"] == MODEL_REVISION
    assert len(block["source_text_sha256"]) == 64
    assert block["spans"] == [
        {"char_start": 0, "char_end": 6, "start_ms": 0, "end_ms": 500},
        {"char_start": 7, "char_end": 10, "start_ms": 500, "end_ms": 1000},
    ]


def test_long_form_word_offset_keeps_exact_milliseconds():
    result = _result(
        "слово",
        words=(ASRWordSpan(text="слово", start_s=10.125, end_s=10.375),),
        origin="native",
    )

    block = build_observed_timing(result)

    assert block["spans"] == [{"char_start": 0, "char_end": 5, "start_ms": 10125, "end_ms": 10375}]


def test_text_only_result_has_no_observed_span():
    block = build_observed_timing(_result("привет"))

    assert block["origin"] == "none"
    assert block["unit"] == "ms"
    assert block["spans"] == []


def test_mixed_timed_and_untimed_segments_keep_only_observed_intervals():
    result = _result(
        "привет мир",
        segments=(
            ASRSegment(text="привет", start_s=0.0, end_s=1.0),
            ASRSegment(text="мир", start_s=None, end_s=None),
        ),
        origin="native",
    )

    block = build_observed_timing(result)

    assert block["spans"] == [{"char_start": 0, "char_end": 6, "start_ms": 0, "end_ms": 1000}]


def test_chunked_origin_never_persists_acoustic_spans():
    result = _result(
        "привет",
        segments=(ASRSegment(text="привет", start_s=0.0, end_s=1.0),),
        origin="chunked",
    )

    block = build_observed_timing(result)

    assert block["origin"] == "chunked"
    assert block["spans"] == []


def test_unmappable_segments_fail_closed_to_no_spans():
    result = _result(
        "привет мир",
        segments=(ASRSegment(text="совсем другое", start_s=0.0, end_s=1.0),),
        origin="native",
    )

    assert build_observed_timing(result)["spans"] == []


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"observed_spans": []},
        {"observed_spans": {"spans": "garbage"}},
        {"observed_spans": {"spans": [{}]}},
        {
            "observed_spans": {
                "spans": [{"char_start": "0", "char_end": 1, "start_ms": 0, "end_ms": 1}]
            }
        },
        {
            "observed_spans": {
                "spans": [{"char_start": 5, "char_end": 2, "start_ms": 0, "end_ms": 1}]
            }
        },
        {
            "observed_spans": {
                "spans": [{"char_start": 0, "char_end": 2, "start_ms": 7, "end_ms": 1}]
            }
        },
        {
            "observed_spans": {
                "spans": [{"char_start": -1, "char_end": 2, "start_ms": 0, "end_ms": 1}]
            }
        },
    ],
)
def test_malformed_snapshot_yields_no_spans(snapshot):
    assert observed_spans_from_snapshot(snapshot) == ()


def test_snapshot_round_trip_reads_the_stored_spans():
    block = build_observed_timing(
        _result(
            "привет мир",
            words=(
                ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
                ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
            ),
            origin="native",
        )
    )

    spans = observed_spans_from_snapshot({"observed_spans": block})

    assert spans == (
        ObservedTimingSpan(char_start=0, char_end=6, start_ms=0, end_ms=500),
        ObservedTimingSpan(char_start=7, char_end=10, start_ms=500, end_ms=1000),
    )


def test_chunk_time_range_requires_full_containment():
    spans = (
        ObservedTimingSpan(char_start=0, char_end=6, start_ms=0, end_ms=500),
        ObservedTimingSpan(char_start=7, char_end=10, start_ms=500, end_ms=1000),
    )

    assert chunk_time_range(spans, char_start=0, char_end=10, chunk_text="привет мир") == (0, 1000)
    assert chunk_time_range(spans, char_start=0, char_end=6, chunk_text="привет") == (0, 500)
    assert chunk_time_range(spans, char_start=6, char_end=10, chunk_text=" мир") == (500, 1000)
    # A window that splits the first span claims no boundary it straddles.
    assert chunk_time_range(spans, char_start=0, char_end=3, chunk_text="при") == (None, None)
    assert chunk_time_range((), char_start=0, char_end=10, chunk_text="привет мир") == (None, None)
    assert chunk_time_range(spans[:1], char_start=0, char_end=10, chunk_text="привет мир") == (
        None,
        None,
    )


# -- derived FTS ranges -------------------------------------------------------


def test_indexed_asr_chunk_uses_the_observed_range(repository, database, tmp_path):
    result = _result(
        "привет мир",
        words=(
            ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
            ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
        ),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    _add_bound_transcript(repository, run.run_uuid, result.transcript)

    row = _search(database.connection, "привет")["results"][0]

    assert row["start_ms"] == 0
    assert row["end_ms"] == 1000


def test_other_indexed_text_in_same_run_never_inherits_transcript_timing(
    repository, database, tmp_path
):
    result = _result(
        "привет мир",
        words=(ASRWordSpan(text="привет мир", start_s=1.0, end_s=2.0),),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr-context"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_asr",
        content=result.transcript,
    )
    row = database.connection.execute(
        "SELECT start_ms, end_ms FROM search_chunks WHERE run_uuid = ? AND kind = ?",
        (run.run_uuid, TEXT_KIND_TTS_SCRIPT),
    ).fetchone()
    assert row is not None
    assert tuple(row) == (None, None)


def test_other_transcript_in_same_run_never_inherits_timing(repository, database, tmp_path):
    result = _result(
        "привет мир",
        words=(ASRWordSpan(text="привет мир", start_s=1.0, end_s=2.0),),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "other-source"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content="другая расшифровка",
    )
    row = database.connection.execute(
        "SELECT start_ms, end_ms FROM search_chunks WHERE run_uuid = ?",
        (run.run_uuid,),
    ).fetchone()
    assert row is not None
    assert tuple(row) == (None, None)


def test_identical_second_asr_source_in_same_run_has_no_timing(repository, database, tmp_path):
    result = _result(
        "привет мир",
        words=(ASRWordSpan(text="привет мир", start_s=1.0, end_s=2.0),),
        origin="native",
    )
    source_uuid = str(uuid.uuid4())
    block = build_observed_timing(result)
    block["text_source_uuid"] = source_uuid
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "duplicate-source"),
        config_snapshot=_snapshot(block),
    )
    first = repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content=result.transcript,
        text_source_uuid=source_uuid,
    )
    second = repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content=result.transcript,
    )
    rows = database.connection.execute(
        "SELECT text_source_uuid, start_ms, end_ms FROM search_chunks WHERE run_uuid = ?",
        (run.run_uuid,),
    ).fetchall()
    assert {row["text_source_uuid"]: (row["start_ms"], row["end_ms"]) for row in rows} == {
        first.text_source_uuid: (1000, 2000),
        second.text_source_uuid: (None, None),
    }


def test_mixed_timed_and_untimed_text_keeps_fts_chunk_range_null(repository, database, tmp_path):
    result = _result(
        "привет мир",
        segments=(
            ASRSegment(text="привет", start_s=0.0, end_s=1.0),
            ASRSegment(text="мир", start_s=None, end_s=None),
        ),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "partial"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    _add_bound_transcript(repository, run.run_uuid, result.transcript)
    row = _search(database.connection, "привет")["results"][0]
    assert row["start_ms"] is None
    assert row["end_ms"] is None


def test_text_only_asr_chunk_range_stays_null(repository, database, tmp_path):
    block = build_observed_timing(_result("привет мир"))
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr"),
        config_snapshot=_snapshot(block),
    )
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_ASR_TRANSCRIPT, origin="native_asr", content="привет мир"
    )

    row = _search(database.connection, "привет")["results"][0]

    assert row["start_ms"] is None
    assert row["end_ms"] is None


def test_non_asr_chunk_range_stays_null(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "tts"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        content="сценарный привет",
    )

    row = _search(database.connection, "сценарный")["results"][0]

    assert row["start_ms"] is None
    assert row["end_ms"] is None


def test_malformed_snapshot_does_not_corrupt_the_index(repository, database, tmp_path):
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "bad"),
        config_snapshot={"observed_spans": {"spans": "not-a-list", "unit": "ms"}},
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content="повреждённый текст",
    )

    status = indexing.index_status(database.connection)
    assert status["complete"] is True
    row = _search(database.connection, "повреждённый")["results"][0]
    assert row["start_ms"] is None
    assert row["end_ms"] is None


def test_offline_rebuild_reproduces_identical_ranges(repository, database, tmp_path):
    result = _result(
        "привет мир",
        words=(
            ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
            ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
        ),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    _add_bound_transcript(repository, run.run_uuid, result.transcript)

    before = [tuple(row) for row in _chunk_ranges(database.connection, run.run_uuid)]
    indexing.rebuild_index(database.connection)
    after = [tuple(row) for row in _chunk_ranges(database.connection, run.run_uuid)]

    assert after == before
    assert any(row[3] is not None for row in before)


def test_rebuild_after_history_import_reproduces_asr_ranges(repository, database, tmp_path):
    # One imported legacy transcript carries no observed span: its derived range
    # stays NULL rather than being back-filled from the source duration.
    with repository.transaction():
        legacy, _created = repository.create_legacy_run(
            operation="asr",
            run_root=str(tmp_path / "out" / "legacy"),
            legacy_source_root=str(tmp_path / "out" / "legacy"),
        )
        repository.add_text_source(
            legacy.run_uuid,
            kind=TEXT_KIND_ASR_TRANSCRIPT,
            origin="legacy_import",
            content="старый привет мир",
        )

    result = _result(
        "привет мир",
        words=(
            ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
            ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
        ),
        origin="native",
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr"),
        config_snapshot=_snapshot(build_observed_timing(result)),
    )
    _add_bound_transcript(repository, run.run_uuid, result.transcript)

    before = [tuple(row) for row in _chunk_ranges(database.connection, run.run_uuid)]
    indexing.rebuild_index(database.connection)
    after = [tuple(row) for row in _chunk_ranges(database.connection, run.run_uuid)]

    assert after == before
    legacy_ranges = _chunk_ranges(database.connection, legacy.run_uuid)
    assert legacy_ranges
    assert all(row[3] is None and row[4] is None for row in legacy_ranges)


def test_search_history_reports_derived_range_from_canonical_database(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    with HistoryDatabase(database_path) as db:
        db.migrate()
        repository = HistoryRepository(db)
        result = _result(
            "привет мир",
            words=(
                ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
                ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
            ),
            origin="native",
        )
        run = repository.create_run(
            operation="asr",
            run_root=str(tmp_path / "runs" / "asr"),
            config_snapshot=_snapshot(build_observed_timing(result)),
        )
        _add_bound_transcript(repository, run.run_uuid, result.transcript)

    payload = search_commands.search_history("привет", database_path=database_path)

    assert payload["count"] == 1
    assert payload["results"][0]["start_ms"] == 0
    assert payload["results"][0]["end_ms"] == 1000


def test_search_and_index_logs_never_carry_the_query_or_transcript(
    repository, database, tmp_path, caplog
):
    transcript = "приватная расшифровка с секретом"
    block = build_observed_timing(
        _result(
            transcript,
            words=(ASRWordSpan(text=transcript, start_s=0.0, end_s=1.0),),
            origin="native",
        )
    )
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "asr"),
        config_snapshot=_snapshot(block),
    )
    _add_bound_transcript(repository, run.run_uuid, transcript)

    with caplog.at_level(logging.DEBUG):
        indexing.build_index(database.connection)
        indexing.rebuild_index(database.connection)
        search_commands.search_history("приватная", database_path=database.path)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert transcript not in logged
    assert "приватная" not in logged
    assert "секретом" not in logged
