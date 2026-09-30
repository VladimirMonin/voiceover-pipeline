"""Contract tests for the atomic native prepared-TTS snapshot writer.

Every fixture here is synthetic and lives under ``tmp_path``. No ``.env`` value,
user output directory, provider, model, FFmpeg, or network path is touched: the
service under test only validates the objects it is handed and writes SQLite
rows, and it never reads a script file or ``run_state.json``. The tests cover
exact text/voice/structural persistence, identity stability and variation,
``run_root`` collision protection (including an imported run), atomic rollback
when a part or text insert aborts, zero filesystem side effects, fail-closed
rejection of a secret-looking identity, rejection of a sparse, reordered, or
malformed chunk identity, refusal to join a caller's open transaction, acceptance
of the two allowlisted routes, and fail-closed rejection of every non-allowlisted
route (OmniVoice dialogue and preset/clone/design, Qwen preset/clone/design, Polza
chat-audio, and an unknown provider) before any write even when the provider, ids,
and fingerprints all look valid.
"""

import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.native_snapshot import (
    _UNSUPPORTED_TTS_ROUTE_REJECTED,
    NATIVE_SNAPSHOT_OPERATION,
    NATIVE_SNAPSHOT_ORIGIN,
    NativeSnapshotInTransactionError,
    NativeSnapshotRunRootConflictError,
    NativeSnapshotValidationError,
    persist_prepared_tts_snapshot,
)
from voiceover_pipeline.history.repository import (
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
)
from voiceover_pipeline.models import ScriptChunk
from voiceover_pipeline.services.prepare import PreparedPart, PreparedRun

_DEFAULT_SCRIPT_TEXT = "Первый.\n\nВторой."


@pytest.fixture
def repository(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as database:
        database.migrate()
        yield HistoryRepository(database)


def _chunk(number, text, **overrides):
    """Build one synthetic script chunk with a stable id derived from ``number``."""
    fields = {"id": f"chunk_{number:02d}", "text": text}
    fields.update(overrides)
    return ScriptChunk(number=number, **fields)


def _scenario(
    *,
    provider="polza-tts",
    model="openai/gpt-4o-mini-tts",
    voice="alloy",
    style_prompt="Read warmly.",
    prompt_mode="plain",
    texts=_DEFAULT_SCRIPT_TEXT.split("\n\n"),
    cast_voices=(None, None),
    pauses=(0, 0),
    order=None,
):
    """Build a two-part prepared run whose knobs drive one identity dimension each.

    ``order`` permutes the ``(text, cast voice, pause)`` inputs before the chunks
    are numbered, so every chunk keeps its generated number/id while the mapping
    of text, cast, and pause to positions still changes the identity.
    """
    indexes = list(range(len(texts))) if order is None else list(order)
    chunks = [
        _chunk(position + 1, texts[index], voice=cast_voices[index], pause_after_ms=pauses[index])
        for position, index in enumerate(indexes)
    ]
    parts = tuple(PreparedPart(chunk=chunk, voice=chunk.voice) for chunk in chunks)
    return PreparedRun(
        provider=provider,
        model=model,
        voice=voice,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        parts=parts,
    )


def _script_text(prepared):
    return "\n\n".join(part.chunk.text for part in prepared.parts)


def _persist(repository, prepared, run_root, **overrides):
    params = {
        "run_root": run_root,
        "user_label": "prod",
        "script_format": "markdown",
        "script_text": _script_text(prepared),
        "script_path": None,
    }
    params.update(overrides)
    return persist_prepared_tts_snapshot(repository, prepared=prepared, **params)


def _row_count(repository, table):
    return repository._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


_OMNIVOICE_MODEL = "audio-cpp/omnivoice-q8_0"
_OMNIVOICE_REFERENCE_SHA = "b" * 64


def _omnivoice_dialogue_prepared():
    """A valid-looking OmniVoice dialogue run: generated turn ids, cast voices,
    and bound reference fingerprints, with the nonsecret provider/model."""
    turns = (
        ScriptChunk(
            number=1,
            id="turn_0001",
            text="Первая реплика.",
            speaker="host",
            voice="voice_a",
            voice_fingerprint=_OMNIVOICE_REFERENCE_SHA,
        ),
        ScriptChunk(
            number=2,
            id="turn_0002",
            text="Вторая реплика.",
            speaker="guest",
            voice="voice_b",
            voice_fingerprint=_OMNIVOICE_REFERENCE_SHA,
        ),
    )
    parts = tuple(PreparedPart(chunk=turn, voice=turn.voice) for turn in turns)
    return PreparedRun(
        provider="omnivoice-local",
        model=_OMNIVOICE_MODEL,
        voice="voice_a",
        style_prompt=None,
        prompt_mode="plain",
        parts=parts,
    )


def _omnivoice_single_mode_prepared():
    """A valid-looking non-dialogue OmniVoice run that preset, clone, and design
    share: one generated chunk id with a bound cast voice."""
    chunk = ScriptChunk(
        number=1,
        id="chunk_01",
        text="Одиночный текст.",
        voice="voice_a",
        voice_fingerprint=_OMNIVOICE_REFERENCE_SHA,
    )
    return PreparedRun(
        provider="omnivoice-local",
        model=_OMNIVOICE_MODEL,
        voice="voice_a",
        style_prompt=None,
        prompt_mode="plain",
        parts=(PreparedPart(chunk=chunk, voice=chunk.voice),),
    )


# Each route pairs a valid-looking prepared run with the identity inputs that
# route would carry; every one must fail closed before any write with the same
# fixed privacy-safe message.
_QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_POLZA_CHAT_MODEL = "openai/gpt-4o-audio-preview"
_UNKNOWN_PROVIDER = "acme-tts-9000"


def _single_chunk_prepared(*, provider, model, voice="voice_a"):
    """A valid-looking single-part run for a route whose resume identity this
    foundation does not persist, so the provider allowlist alone decides it."""
    chunk = ScriptChunk(number=1, id="chunk_01", text="Одиночный текст.", voice=voice)
    return PreparedRun(
        provider=provider,
        model=model,
        voice=voice,
        style_prompt=None,
        prompt_mode="plain",
        parts=(PreparedPart(chunk=chunk, voice=chunk.voice),),
    )


_UNSUPPORTED_ROUTES = {
    "omnivoice-dialogue": lambda: (
        _omnivoice_dialogue_prepared(),
        {"synthesis_identity": "a" * 64},
    ),
    "omnivoice-preset": lambda: (
        _omnivoice_single_mode_prepared(),
        {"voice_identity": f"preset:voice_a:{_OMNIVOICE_REFERENCE_SHA}"},
    ),
    "omnivoice-clone": lambda: (
        _omnivoice_single_mode_prepared(),
        {"voice_identity": f"clone:{'c' * 64}:{'d' * 64}"},
    ),
    "omnivoice-design": lambda: (
        _omnivoice_single_mode_prepared(),
        {"voice_identity": f"design:{'e' * 64}"},
    ),
    "qwen-preset": lambda: (
        _single_chunk_prepared(provider="qwen-local", model=_QWEN_MODEL),
        {"voice_identity": "preset:speaker_1"},
    ),
    "qwen-clone": lambda: (
        _single_chunk_prepared(provider="qwen-local", model=_QWEN_MODEL),
        {"voice_identity": f"clone:{'c' * 64}"},
    ),
    "qwen-design": lambda: (
        _single_chunk_prepared(provider="qwen-local", model=_QWEN_MODEL),
        {"voice_identity": f"design:{'e' * 64}"},
    ),
    "polza-chat-audio": lambda: (
        _single_chunk_prepared(provider="polza-chat-audio", model=_POLZA_CHAT_MODEL),
        {},
    ),
    "unknown-provider": lambda: (
        _single_chunk_prepared(provider=_UNKNOWN_PROVIDER, model="mystery-model"),
        {},
    ),
}


def test_persists_exact_parts_text_voice_and_structural_snapshot(repository, tmp_path):
    script = tmp_path / "script.md"
    script.write_text(_DEFAULT_SCRIPT_TEXT, encoding="utf-8")
    prepared = _scenario()

    result = _persist(
        repository,
        prepared,
        tmp_path / "runs" / "one",
        script_text=_DEFAULT_SCRIPT_TEXT,
        script_path=script,
    )

    run = repository.get_run(result.run.run_uuid)
    assert run is not None
    assert run.operation == NATIVE_SNAPSHOT_OPERATION
    assert run.status == "running"
    assert run.user_label == "prod"
    assert run.run_root == str((tmp_path / "runs" / "one").resolve())
    assert result.run.run_uuid == run.run_uuid

    snapshot = run.config_snapshot
    assert snapshot["script_format"] == "markdown"
    assert snapshot["script_path"] == str(script.resolve())
    assert (
        snapshot["script_sha256"]
        == hashlib.sha256(_DEFAULT_SCRIPT_TEXT.encode("utf-8")).hexdigest()
    )
    assert snapshot["provider"] == "polza-tts"
    assert snapshot["model"] == "openai/gpt-4o-mini-tts"
    assert snapshot["voice"] == "alloy"
    assert snapshot["prompt_mode"] == "plain"
    assert snapshot["snapshot_fingerprint"] == result.fingerprint
    assert snapshot["part_count"] == 2
    assert [entry["number"] for entry in snapshot["parts"]] == [1, 2]
    assert [entry["position"] for entry in snapshot["parts"]] == [1, 2]
    assert [entry["id"] for entry in snapshot["parts"]] == ["chunk_01", "chunk_02"]
    assert [entry["effective_voice"] for entry in snapshot["parts"]] == ["alloy", "alloy"]
    # Structural fields only: the full text lives in text_sources, not here.
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "Первый." not in serialized
    assert "Read warmly." not in serialized

    parts = repository.get_parts(result.run.run_uuid)
    assert [(p.position, p.prepared_text, p.voice) for p in parts] == [
        (1, "Первый.", "alloy"),
        (2, "Второй.", "alloy"),
    ]
    assert all(p.fingerprint for p in parts)
    assert [p.part_uuid for p in parts] == [p.part_uuid for p in result.parts]
    # No premature stage: the raw-saved seam owns the NULL -> raw_saved move.
    assert all(p.stage is None for p in parts)
    # Vibe stays NULL until the speech-parts stage defines a real value.
    assert all(
        p.vibe_shared is None and p.vibe_specific is None and p.vibe_effective is None
        for p in parts
    )

    sources = repository.get_text_sources(result.run.run_uuid)
    assert all(source.origin == NATIVE_SNAPSHOT_ORIGIN for source in sources)
    run_level = [s for s in sources if s.kind == TEXT_KIND_TTS_SCRIPT and s.part_uuid is None]
    assert len(run_level) == 1
    assert run_level[0].content == _DEFAULT_SCRIPT_TEXT
    assert (
        run_level[0].content_hash
        == hashlib.sha256(_DEFAULT_SCRIPT_TEXT.encode("utf-8")).hexdigest()
    )
    per_part = [s for s in sources if s.kind == TEXT_KIND_TTS_SCRIPT and s.part_uuid is not None]
    # ``get_text_sources`` orders by created_at then UUID, and same-second rows
    # tie on the UUID, so compare as a set rather than a fixed order.
    assert {s.content for s in per_part} == {"Первый.", "Второй."}
    direction = [s for s in sources if s.kind == TEXT_KIND_TTS_DIRECTION]
    assert len(direction) == 1
    assert direction[0].content == "Read warmly."
    assert direction[0].part_uuid is None


def test_missing_style_prompt_writes_no_direction_source(repository, tmp_path):
    result = _persist(repository, _scenario(style_prompt=None), tmp_path / "runs" / "bare")

    sources = repository.get_text_sources(result.run.run_uuid)
    assert not [s for s in sources if s.kind == TEXT_KIND_TTS_DIRECTION]
    assert result.run.config_snapshot["script_path"] is None


def test_cast_voice_overrides_run_voice_per_part(repository, tmp_path):
    prepared = _scenario(cast_voices=("Rachel", None), voice="alloy")

    result = _persist(repository, prepared, tmp_path / "runs" / "cast")

    parts = repository.get_parts(result.run.run_uuid)
    assert [p.voice for p in parts] == ["Rachel", "alloy"]
    assert [entry["effective_voice"] for entry in result.run.config_snapshot["parts"]] == [
        "Rachel",
        "alloy",
    ]
    assert [entry["cast_voice"] for entry in result.run.config_snapshot["parts"]] == [
        "Rachel",
        None,
    ]


def test_single_omnivoice_session_chunk_id_is_accepted_at_number_one(repository, tmp_path):
    prepared = replace(
        _scenario(),
        parts=(
            PreparedPart(
                chunk=_chunk(1, "Единственный.", id="chunk_01_omnivoice_session"),
                voice=None,
            ),
        ),
    )

    result = _persist(repository, prepared, tmp_path / "runs" / "omni", script_text="Единственный.")

    parts = repository.get_parts(result.run.run_uuid)
    assert [(p.position, p.voice) for p in parts] == [(1, "alloy")]
    assert result.run.config_snapshot["parts"][0]["id"] == "chunk_01_omnivoice_session"


def test_identity_is_stable_for_identical_inputs_on_a_different_root(repository, tmp_path):
    first = _persist(repository, _scenario(), tmp_path / "runs" / "one")
    second = _persist(repository, _scenario(), tmp_path / "runs" / "two")

    assert first.run.run_uuid != second.run.run_uuid
    assert first.fingerprint == second.fingerprint
    assert [p.fingerprint for p in first.parts] == [p.fingerprint for p in second.parts]


def test_identity_depends_on_the_provided_snapshot_not_the_script_file(repository, tmp_path):
    script = tmp_path / "script.md"
    script.write_text(_DEFAULT_SCRIPT_TEXT, encoding="utf-8")
    prepared = _scenario()

    first = _persist(
        repository,
        prepared,
        tmp_path / "runs" / "before",
        script_text=_DEFAULT_SCRIPT_TEXT,
        script_path=script,
    )
    script.write_text("Полностью другой текст.", encoding="utf-8")
    second = _persist(
        repository,
        prepared,
        tmp_path / "runs" / "after",
        script_text=_DEFAULT_SCRIPT_TEXT,
        script_path=script,
    )

    assert first.fingerprint == second.fingerprint
    # The stored snapshot keeps the text that was supplied, not the new file bytes.
    stored = repository.get_text_sources(first.run.run_uuid)
    assert any(s.content == _DEFAULT_SCRIPT_TEXT for s in stored)
    assert not any(s.content == "Полностью другой текст." for s in stored)


@pytest.mark.parametrize(
    "variant",
    [
        {"voice": "verse"},
        {"model": "openai/gpt-4o-mini-tts-preview"},
        {"texts": ("Первый.", "Второй изменён.")},
        {"style_prompt": "Read coldly."},
        {"cast_voices": ("Rachel", "Rachel")},
        {"order": (1, 0)},
        {"pauses": (0, 350)},
    ],
)
def test_identity_changes_with_each_synthesis_variation(repository, tmp_path, variant):
    baseline = _persist(repository, _scenario(), tmp_path / "runs" / "baseline")

    changed = _scenario(**variant)
    variant_result = _persist(
        repository,
        changed,
        tmp_path / "runs" / "variant",
        script_text=_script_text(changed),
    )

    assert variant_result.fingerprint != baseline.fingerprint
    assert [p.fingerprint for p in variant_result.parts] != [p.fingerprint for p in baseline.parts]


def test_duplicate_run_root_is_blocked_without_partial_duplicate(repository, tmp_path):
    run_root = tmp_path / "runs" / "shared"
    first = _persist(repository, _scenario(), run_root)
    parts_before = len(repository.get_parts(first.run.run_uuid))
    sources_before = len(repository.get_text_sources(first.run.run_uuid))

    duplicate = _scenario(texts=("Совсем другой.", "И ещё один."))
    with pytest.raises(NativeSnapshotRunRootConflictError):
        _persist(repository, duplicate, run_root)

    runs = repository.find_runs_by_root(str(run_root.resolve()))
    assert [run.run_uuid for run in runs] == [first.run.run_uuid]
    assert len(repository.get_parts(first.run.run_uuid)) == parts_before
    assert len(repository.get_text_sources(first.run.run_uuid)) == sources_before
    assert len(repository.list_runs()) == 1


def test_imported_run_root_blocks_a_native_snapshot_of_the_same_root(repository, tmp_path):
    run_root = tmp_path / "runs" / "legacy"
    run_root.mkdir(parents=True)
    with repository.transaction():
        repository.create_legacy_run(
            operation="tts",
            run_root=str(run_root.resolve()),
            legacy_source_root=str(run_root.resolve()),
        )

    with pytest.raises(NativeSnapshotRunRootConflictError):
        _persist(repository, _scenario(), run_root)

    assert len(repository.list_runs()) == 1


def test_same_label_on_a_different_root_is_an_independent_run(repository, tmp_path):
    first = _persist(repository, _scenario(), tmp_path / "runs" / "a", user_label="prod")
    second = _persist(repository, _scenario(), tmp_path / "runs" / "b", user_label="prod")

    assert first.run.run_uuid != second.run.run_uuid
    assert len(repository.find_runs_by_label("prod")) == 2


def test_aborted_text_source_insert_leaves_no_partial_run(repository, tmp_path):
    repository._connection.execute(
        "CREATE TRIGGER text_sources_crash_insert BEFORE INSERT ON text_sources "
        "BEGIN SELECT RAISE(ABORT, 'synthetic pre-commit crash'); END"
    )

    with pytest.raises(sqlite3.Error):
        _persist(repository, _scenario(), tmp_path / "runs" / "crash")

    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_aborted_part_insert_rolls_back_the_run_and_its_texts(repository, tmp_path):
    repository._connection.execute(
        "CREATE TRIGGER parts_crash_insert BEFORE INSERT ON parts "
        "BEGIN SELECT RAISE(ABORT, 'synthetic pre-commit crash'); END"
    )

    with pytest.raises(sqlite3.Error):
        _persist(repository, _scenario(), tmp_path / "runs" / "crash")

    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_snapshot_writes_no_json_or_run_directory(repository, tmp_path):
    run_root = tmp_path / "runs" / "never-created"

    _persist(
        repository,
        _scenario(),
        run_root,
        script_path=tmp_path / "script.md",
    )

    assert not run_root.exists()
    assert list(tmp_path.rglob("run_state.json")) == []
    assert list(tmp_path.rglob("*.json")) == []


def test_part_voice_diverging_from_the_wrapped_chunk_voice_is_rejected(repository, tmp_path):
    prepared = replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(1, "Первый.", voice="Rachel"), voice="alloy"),
            PreparedPart(chunk=_chunk(2, "Второй."), voice=None),
        ),
    )

    with pytest.raises(NativeSnapshotValidationError):
        _persist(repository, prepared, tmp_path / "runs" / "divergent")

    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0


def test_snapshot_inside_a_caller_transaction_is_refused_before_insertion(repository, tmp_path):
    prepared = _scenario()

    with repository.transaction():
        with pytest.raises(NativeSnapshotInTransactionError):
            _persist(repository, prepared, tmp_path / "runs" / "outer")

    # A joined write would have returned a snapshot this transaction could still
    # roll back; the fail-closed refusal never inserts a row at all.
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_secret_looking_identity_is_rejected_without_a_row(repository, tmp_path):
    secret = "sk-syntheticSyntheticSynthetic"

    with pytest.raises(NativeSnapshotValidationError) as excinfo:
        _persist(
            repository,
            _scenario(),
            tmp_path / "runs" / "secret",
            voice_identity=secret,
        )

    assert secret not in str(excinfo.value)
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


_INVALID_CASES = {
    "blank provider": lambda: replace(_scenario(), provider="  "),
    "blank model": lambda: replace(_scenario(), model=""),
    "blank run voice": lambda: replace(_scenario(), voice=""),
    "blank prompt mode": lambda: replace(_scenario(), prompt_mode="  "),
    "no parts": lambda: replace(_scenario(), parts=()),
    "zero number": lambda: replace(
        _scenario(), parts=(PreparedPart(chunk=_chunk(0, "Ноль."), voice=None),)
    ),
    "negative number": lambda: replace(
        _scenario(), parts=(PreparedPart(chunk=_chunk(-1, "Минус."), voice=None),)
    ),
    "duplicate number": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(1, "А.", id="chunk_01"), voice=None),
            PreparedPart(chunk=_chunk(1, "Б.", id="chunk_01"), voice=None),
        ),
    ),
    "sparse chunk number": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(3, "Третий."), voice=None),
            PreparedPart(chunk=_chunk(7, "Седьмой."), voice=None),
        ),
    ),
    "reordered chunk numbers": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(2, "Второй."), voice=None),
            PreparedPart(chunk=_chunk(1, "Первый."), voice=None),
        ),
    ),
    "malformed chunk id": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(1, "Первый.", id="c1"), voice=None),
            PreparedPart(chunk=_chunk(2, "Второй."), voice=None),
        ),
    ),
    "part voice differs from chunk voice": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(1, "Первый.", voice="Rachel"), voice="alloy"),
            PreparedPart(chunk=_chunk(2, "Второй."), voice=None),
        ),
    ),
    "blank cast voice": lambda: replace(
        _scenario(),
        parts=(
            PreparedPart(chunk=_chunk(1, "А.", voice="  "), voice="  "),
            PreparedPart(chunk=_chunk(2, "Б."), voice=None),
        ),
    ),
}


@pytest.mark.parametrize("label", sorted(_INVALID_CASES))
def test_invalid_prepared_run_never_writes_a_row(repository, tmp_path, label):
    prepared = _INVALID_CASES[label]()

    with pytest.raises(NativeSnapshotValidationError):
        _persist(repository, prepared, tmp_path / "runs" / "invalid")

    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_blank_script_format_is_rejected_before_any_insert(repository, tmp_path):
    with pytest.raises(NativeSnapshotValidationError):
        _persist(repository, _scenario(), tmp_path / "runs" / "invalid", script_format="  ")

    assert _row_count(repository, "runs") == 0


@pytest.mark.parametrize("route", sorted(_UNSUPPORTED_ROUTES))
def test_unsupported_route_is_rejected_before_any_write(repository, tmp_path, route):
    prepared, overrides = _UNSUPPORTED_ROUTES[route]()
    run_root = tmp_path / "runs" / route

    with pytest.raises(NativeSnapshotValidationError) as excinfo:
        _persist(repository, prepared, run_root, **overrides)

    # One fixed privacy-safe message: it names no provider and echoes no supplied
    # identity value, so an unknown identifier cannot leak a secret.
    assert str(excinfo.value) == _UNSUPPORTED_TTS_ROUTE_REJECTED
    assert prepared.provider not in str(excinfo.value)
    for value in overrides.values():
        assert value not in str(excinfo.value)
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0
    assert not run_root.exists()
    assert list(tmp_path.rglob("*.json")) == []


def test_unknown_provider_is_rejected_without_echoing_the_identifier(repository, tmp_path):
    secret = "sk-syntheticUnknownProviderValue"

    with pytest.raises(NativeSnapshotValidationError) as excinfo:
        _persist(repository, replace(_scenario(), provider=secret), tmp_path / "runs" / "unknown")

    assert str(excinfo.value) == _UNSUPPORTED_TTS_ROUTE_REJECTED
    assert secret not in str(excinfo.value)
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_openrouter_tts_route_is_accepted_and_reconstructable(repository, tmp_path):
    prepared = _scenario(
        provider="openrouter-tts",
        model="openai/gpt-4o-mini-tts",
        voice="alloy",
        style_prompt="Read warmly.",
        prompt_mode="json",
    )

    result = _persist(repository, prepared, tmp_path / "runs" / "openrouter")

    snapshot = result.run.config_snapshot
    assert snapshot["provider"] == "openrouter-tts"
    assert snapshot["model"] == "openai/gpt-4o-mini-tts"
    assert snapshot["voice"] == "alloy"
    assert snapshot["prompt_mode"] == "json"
    direction = [
        source
        for source in repository.get_text_sources(result.run.run_uuid)
        if source.kind == TEXT_KIND_TTS_DIRECTION
    ]
    assert [source.content for source in direction] == ["Read warmly."]


def test_omnivoice_rejection_precedes_part_validation(repository, tmp_path):
    # A run whose parts would separately fail validation is refused by the route
    # check first, proving the rejection is fail-first and pre-write.
    prepared = replace(
        _omnivoice_dialogue_prepared(),
        parts=(
            PreparedPart(
                chunk=ScriptChunk(number=1, id="not_generated", text="x", voice="voice_a"),
                voice="voice_a",
            ),
        ),
    )

    with pytest.raises(NativeSnapshotValidationError) as excinfo:
        _persist(repository, prepared, tmp_path / "runs" / "omnivoice-invalid")

    assert str(excinfo.value) == _UNSUPPORTED_TTS_ROUTE_REJECTED
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0


def test_route_rejection_precedes_run_voice_validation(repository, tmp_path):
    # qwen-local design mode resolves no run voice, so the allowlist must reject the
    # route before the non-empty-voice check; this proves the rejection is fail-first
    # even for a run whose voice would separately fail.
    prepared = replace(_scenario(), provider="qwen-local", model=_QWEN_MODEL, voice="")

    with pytest.raises(NativeSnapshotValidationError) as excinfo:
        _persist(repository, prepared, tmp_path / "runs" / "qwen-design")

    assert str(excinfo.value) == _UNSUPPORTED_TTS_ROUTE_REJECTED
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0
