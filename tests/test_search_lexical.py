"""Contract tests for the offline lexical search layer (plan section 8, S08).

These cover the observable behavior of the FTS5 query over saved history text:
Russian/English and ``ё``/``е`` matching without changing the stored text, role
separation between speech, ASR, directions, and run label, filters applied before
the limit, missing audio that does not erase history, literal safe handling of
punctuation and SQL-looking input, and bounded snippets. They run fully offline on
a temporary SQLite database with no key, model, FFmpeg, Torch, or network.
"""

from __future__ import annotations

import pytest

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.native_asr import (
    AsrHistorySave,
    AsrHistoryText,
    persist_asr_history,
)
from voiceover_pipeline.history.paths import history_database_path
from voiceover_pipeline.history.repository import (
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_ASR_CONTEXT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
    HistoryRepository,
)
from voiceover_pipeline.search.lexical import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    ROLE_ASR,
    ROLE_DIRECTIONS,
    ROLE_LABEL,
    ROLE_SPEECH,
    chunk_text,
    make_snippet,
    normalize_search_text,
    query_tokens,
    roles_for_scope,
    search_lexical,
    split_text_chunks,
)


@pytest.fixture
def database(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as db:
        db.migrate()
        yield db


@pytest.fixture
def repository(database):
    return HistoryRepository(database)


def _search(database, query, **kwargs):
    kwargs.setdefault("limit", 20)
    kwargs.setdefault("roles", roles_for_scope("speech"))
    return search_lexical(database.connection, query, **kwargs)


def test_russian_and_english_terms_match(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        content="Здесь мы обсуждаем индексы SQLite и транзакции.",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content="The transaction rolls back on conflict.",
    )

    assert len(_search(database, "индексы")["results"]) == 1
    assert len(_search(database, "sqlite")["results"]) == 1
    assert len(_search(database, "transaction")["results"]) == 1


def test_yo_fold_matches_without_changing_stored_text(repository, database, tmp_path):
    run = repository.create_run(operation="asr", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="native_asr",
        content="Ёжик бежит по дороге",
    )

    # Query normalized to ``е`` matches stored ``ё``; the stored text is untouched.
    results = _search(database, "ежик")["results"]
    assert len(results) == 1
    assert results[0]["snippet"] == "Ёжик бежит по дороге"
    stored = repository.get_text_sources(run.run_uuid)[0].content
    assert stored == "Ёжик бежит по дороге"


def test_directions_excluded_from_default_scope_and_opt_in(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_DIRECTION,
        origin="native_snapshot",
        content="спокойный подкаст",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        content="обычный сценарий",
    )

    assert _search(database, "спокойный")["results"] == []
    directions = search_lexical(
        database.connection, "спокойный", limit=20, roles=roles_for_scope("directions")
    )
    assert len(directions["results"]) == 1
    assert directions["results"][0]["role"] == ROLE_DIRECTIONS


def test_label_role_is_searchable_without_a_text_source(repository, database, tmp_path):
    repository.create_run(
        operation="tts",
        run_root=str(tmp_path / "runs" / "a"),
        user_label="подкаст про SQLite",
    )

    results = _search(database, "подкаст")["results"]
    assert len(results) == 1
    assert results[0]["role"] == ROLE_LABEL
    assert results[0]["kind"] == "run_label"
    assert results[0]["text_source_uuid"] is None


def test_private_asr_context_is_not_indexed(repository, database, tmp_path):
    run = repository.create_run(operation="asr", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_CONTEXT,
        origin="native_asr",
        content="приватный промпт про ёжика",
    )

    assert _search(database, "ёжика")["results"] == []
    assert _search(database, "приватный")["results"] == []


def test_role_and_kind_filters(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="x",
        content="общая тема сценария",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_ASR_TRANSCRIPT,
        origin="x",
        content="общая тема распознавания",
    )

    all_roles = search_lexical(database.connection, "тема", limit=20, roles=roles_for_scope("all"))
    assert len(all_roles["results"]) == 2
    only_asr = search_lexical(database.connection, "тема", limit=20, roles=[ROLE_ASR])
    assert [row["role"] for row in only_asr["results"]] == [ROLE_ASR]
    only_script = search_lexical(
        database.connection,
        "тема",
        limit=20,
        roles=roles_for_scope("all"),
        kind=TEXT_KIND_TTS_SCRIPT,
    )
    assert [row["role"] for row in only_script["results"]] == [ROLE_SPEECH]


def test_provider_model_are_attributed_by_text_role_or_left_ambiguous(
    repository, database, tmp_path
):
    run = repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "dialogue"), user_label="mixed pipeline"
    )
    part = repository.add_part(run.run_uuid, position=0, prepared_text="сценарий")
    repository.add_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        call_type="tts_chunk",
        provider="openrouter-tts",
        model="gemini-tts",
    )
    repository.add_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        call_type="tts_quality_asr",
        provider="xai-stt",
        model="grok-stt",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        part_uuid=part.part_uuid,
        content="сценарий диалога",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_VERIFICATION_TRANSCRIPT,
        origin="native_asr",
        part_uuid=part.part_uuid,
        content="распознано слово",
    )

    spoken = _search(database, "диалога")["results"][0]
    recognized = _search(database, "распознано")["results"][0]
    label = _search(database, "mixed")["results"][0]
    assert (spoken["provider"], spoken["model"]) == ("openrouter-tts", "gemini-tts")
    assert (recognized["provider"], recognized["model"]) == ("xai-stt", "grok-stt")
    assert (label["provider"], label["model"]) == ("openrouter-tts", "gemini-tts")

    # A second candidate for the same spoken part makes that attribution
    # ambiguous; do not assign either provider to this part's saved text.
    repository.add_attempt(
        run.run_uuid,
        part_uuid=part.part_uuid,
        call_type="tts_chunk",
        provider="polza-tts",
        model="another-model",
    )
    ambiguous = _search(database, "диалога")["results"][0]
    assert (ambiguous["provider"], ambiguous["model"]) == (None, None)
    assert _search(database, "распознано")["results"][0]["provider"] == "xai-stt"


def test_filters_are_applied_before_limit(repository, database, tmp_path):
    for index in range(5):
        run = repository.create_run(
            operation="tts", run_root=str(tmp_path / "runs" / f"script-{index}")
        )
        repository.add_text_source(
            run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content="общая тема"
        )
    asr_run = repository.create_run(operation="asr", run_root=str(tmp_path / "runs" / "asr"))
    repository.add_text_source(
        asr_run.run_uuid, kind=TEXT_KIND_ASR_TRANSCRIPT, origin="x", content="общая тема"
    )

    # The only ASR match is the oldest run. With limit 1 it must still be returned,
    # which only holds because the kind filter runs before the limit.
    results = search_lexical(
        database.connection,
        "тема",
        limit=1,
        roles=roles_for_scope("speech"),
        kind=TEXT_KIND_ASR_TRANSCRIPT,
    )["results"]
    assert len(results) == 1
    assert results[0]["run_uuid"] == asr_run.run_uuid


def test_run_operation_provider_and_date_filters(repository, database, tmp_path):
    run = repository.create_run(
        operation="asr",
        run_root=str(tmp_path / "runs" / "a"),
        created_at="2026-05-01T00:00:00Z",
    )
    repository.add_attempt(run.run_uuid, call_type="asr_transcription", provider="qwen-local")
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_ASR_TRANSCRIPT, origin="x", content="найти запись"
    )

    base = roles_for_scope("speech")
    assert len(_search(database, "найти", roles=base, operation="asr")["results"]) == 1
    assert _search(database, "найти", roles=base, operation="tts")["results"] == []
    assert len(_search(database, "найти", roles=base, run_uuid=run.run_uuid)["results"]) == 1
    assert _search(database, "найти", roles=base, run_uuid=str(tmp_path))["results"] == []
    assert len(_search(database, "найти", roles=base, provider="qwen-local")["results"]) == 1
    assert _search(database, "найти", roles=base, provider="other")["results"] == []
    assert len(_search(database, "найти", roles=base, since="2026-01-01")["results"]) == 1
    assert _search(database, "найти", roles=base, since="2027-01-01")["results"] == []
    assert len(_search(database, "найти", roles=base, until="2026-12-31")["results"]) == 1
    # An inclusive date includes timestamps throughout that calendar day, but
    # not the following day; filtering is still performed before LIMIT.
    assert len(_search(database, "найти", roles=base, until="2026-05-01")["results"]) == 1
    assert _search(database, "найти", roles=base, until="2026-04-30")["results"] == []


def test_punctuation_quotes_hyphens_and_sql_are_literal(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="x",
        content="правильный ответ; выбери 'вариант-один'",
    )

    # FTS operators and SQL fragments in the query are tokenized into literals and
    # never reach SQL as syntax.
    assert _search(database, 'ответ "вариант"')["results"]
    assert _search(database, "вариант-один")["results"]
    assert _search(database, "ответ AND NOT")["results"] == []
    assert _search(database, "'; DROP TABLE text_sources; --")["results"] == []
    # The table still exists and holds rows after a SQL-looking query.
    assert repository.get_text_sources(run.run_uuid)[0].content is not None


def test_missing_audio_does_not_erase_history(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    artifact = repository.add_artifact(
        run.run_uuid,
        role="final_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=str(tmp_path / "gone.mp3"),
        availability=AVAILABILITY_MISSING,
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="x",
        artifact_uuid=artifact.artifact_uuid,
        content="текст без файла",
    )

    results = _search(database, "файла")["results"]
    assert len(results) == 1
    audio = results[0]["audio"]
    assert audio is not None
    assert audio["availability"] == AVAILABILITY_MISSING
    assert audio["path"].endswith("gone.mp3")


def test_audio_deleted_after_save_is_reported_missing(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"offline fixture, not real sound")
    artifact = repository.add_artifact(
        run.run_uuid,
        role="final_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=str(audio_path),
        availability=AVAILABILITY_PRESENT,
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="x",
        artifact_uuid=artifact.artifact_uuid,
        content="исчезнувший аудиофайл",
    )
    assert _search(database, "исчезнувший")["results"][0]["audio"]["availability"] == "present"
    audio_path.unlink()
    result = _search(database, "исчезнувший")["results"][0]
    assert result["audio"]["availability"] == "missing"
    assert result["snippet"] == "исчезнувший аудиофайл"


def test_native_asr_transcript_links_to_source_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEOVER_HOME", str(tmp_path / "private-home"))
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"offline fixture, not real sound")
    run_uuid = persist_asr_history(
        AsrHistorySave(
            operation="asr",
            attempt_call_type="asr_transcription",
            provider="qwen-local",
            model="qwen3-asr-0.6b",
            config_snapshot={},
            user_label="передача",
            source_audio=audio_path,
            text_sources=[
                AsrHistoryText(kind=TEXT_KIND_ASR_TRANSCRIPT, content="сохранённая речь")
            ],
        )
    )
    with HistoryDatabase(history_database_path()) as stored:
        stored.migrate()
        results = _search(stored, "сохранённая")["results"]
    assert len(results) == 1
    assert results[0]["run_uuid"] == run_uuid
    assert results[0]["audio"]["role"] == "asr_source_audio"
    assert results[0]["audio"]["path"] == str(audio_path.resolve())
    assert results[0]["audio"]["availability"] == "present"
    with HistoryDatabase(history_database_path()) as stored:
        stored.migrate()
        label = _search(stored, "передача")["results"][0]
    assert label["role"] == ROLE_LABEL
    assert label["audio"]["path"] == str(audio_path.resolve())


def test_partial_native_part_does_not_link_to_another_parts_audio(repository, database, tmp_path):
    run = repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "unfinished"), status="running"
    )
    first = repository.add_part(run.run_uuid, position=0, prepared_text="первая часть")
    second = repository.add_part(run.run_uuid, position=1, prepared_text="вторая часть")
    first_audio = tmp_path / "part-one.wav"
    first_audio.write_bytes(b"fixture, not real sound")
    artifact = repository.add_artifact(
        run.run_uuid,
        part_uuid=first.part_uuid,
        role="chunk_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=str(first_audio),
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        part_uuid=first.part_uuid,
        content="первая часть",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        part_uuid=second.part_uuid,
        content="вторая часть",
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        content="общий сценарий",
    )

    assert _search(database, "первая")["results"][0]["audio"]["artifact_uuid"] == (
        artifact.artifact_uuid
    )
    second_result = _search(database, "вторая")["results"][0]
    assert second_result["part_uuid"] == second.part_uuid
    assert second_result["audio"] is None
    assert _search(database, "общий")["results"][0]["audio"] is None


def test_completed_native_part_prefers_final_audio(repository, database, tmp_path):
    run = repository.create_run(
        operation="tts", run_root=str(tmp_path / "runs" / "finished"), status="completed"
    )
    part = repository.add_part(run.run_uuid, position=0, prepared_text="готовая часть")
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"offline chunk fixture")
    repository.add_artifact(
        run.run_uuid,
        part_uuid=part.part_uuid,
        role="chunk_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=str(chunk),
    )
    final = tmp_path / "final.wav"
    final.write_bytes(b"offline final fixture")
    final_artifact = repository.add_artifact(
        run.run_uuid,
        role="final_audio",
        path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
        path=str(final),
    )
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="native_snapshot",
        part_uuid=part.part_uuid,
        content="готовая часть",
    )

    result = _search(database, "готовая")["results"][0]
    assert result["audio"]["artifact_uuid"] == final_artifact.artifact_uuid
    assert result["audio"]["path"] == str(final)


def test_verification_transcript_is_searchable_as_asr_role(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_VERIFICATION_TRANSCRIPT,
        origin="x",
        content="проверочная расшифровка",
    )

    results = _search(database, "проверочная")["results"]
    assert len(results) == 1
    assert results[0]["role"] == ROLE_ASR


def test_snippet_is_bounded_and_centered(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    long_text = ("слово " * 400) + "маркеруникальный" + (" слово" * 400)
    repository.add_text_source(
        run.run_uuid, kind=TEXT_KIND_TTS_SCRIPT, origin="x", content=long_text
    )

    snippet = _search(database, "маркеруникальный")["results"][0]["snippet"]
    assert "маркеруникальный" in snippet
    assert snippet.startswith("…")
    assert len(snippet) <= 320


def test_empty_query_yields_no_results(database):
    assert _search(database, "   ")["results"] == []
    assert _search(database, "!!! ???")["results"] == []


# -- chunking and normalization units -----------------------------------------


def test_normalize_search_text_only_folds_case_and_yo():
    assert normalize_search_text("Ёжик МОДЕЛЬ") == "ежик модель"
    assert normalize_search_text("plain ASCII") == "plain ascii"


def test_query_tokens_drop_operators_and_punctuation():
    assert query_tokens('"индексы" AND sqlite-транзакции') == [
        "индексы",
        "and",
        "sqlite",
        "транзакции",
    ]
    assert query_tokens("") == []


def test_split_text_chunks_covers_text_with_bounded_overlap():
    text = "предложение. " * 400
    spans = split_text_chunks(text)

    assert spans
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for start, end in spans:
        assert end - start <= CHUNK_MAX_CHARS
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start < previous_end
        assert previous_end - next_start <= CHUNK_OVERLAP_CHARS


def test_split_text_chunks_empty_and_short():
    assert split_text_chunks("") == []
    assert split_text_chunks("   ") == []
    assert split_text_chunks("короткий текст") == [(0, len("короткий текст"))]


def test_chunk_text_returns_original_slices():
    text = "А" * (CHUNK_MAX_CHARS + 10)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for start, end, chunk in chunks:
        assert chunk == text[start:end]


def test_make_snippet_without_match_starts_at_beginning():
    text = "начало " + ("слово " * 200)
    snippet = make_snippet(text, ["отсутствует"])
    assert snippet.startswith("начало")


def test_incomplete_text_source_is_not_indexed(repository, database, tmp_path):
    run = repository.create_run(operation="tts", run_root=str(tmp_path / "runs" / "a"))
    repository.add_text_source(
        run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin="legacy_import",
        content=None,
        content_hash="deadbeef",
        text_completeness=TEXT_COMPLETENESS_INCOMPLETE,
    )

    assert _search(database, "deadbeef")["results"] == []
    assert database.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0] == 0
