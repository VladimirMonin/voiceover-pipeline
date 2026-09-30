"""Contract tests for the read-only native TTS resume identity preflight.

These tests cover ``voiceover_pipeline.history.native_resume``: the read-only
synthesis-identity check a later ``generate --resume`` uses to prove that a
candidate prepared run is the exact run the committed native snapshot stores.

Covered here are the passing case (same prepared run, same canonical run root,
same revision), canonical run-root resolution through an equivalent path,
fail-closed rejection of a changed provider, model, run voice, style prompt,
prompt mode, opaque voice identity, opaque synthesis identity, script format, run
root, text, speaker, cast voice, voice fingerprint, pause, chunk number,
generated chunk id, part order, part count, or ``PreparedPart.voice``, the
``expected_revision``/missing-run/invalid-revision failures inherited from the
native view, the already-open-transaction refusal, the proof that the raw source
script file is neither read nor able to change the identity, the proof that a
failed and a successful preflight write no row and no file, the proof that a stored
``submitting`` paid attempt is returned as a plain record and grants no action,
and the fixed privacy-safe conflict and validation messages that echo no secret.

Every fixture is synthetic and lives under ``tmp_path``. No ``.env`` value, user
output directory, provider, model, FFmpeg, or network path is touched: the
preflight only reads already verified SQLite rows and resolves the candidate run
root.
"""

import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.native_resume import (
    NativeResumeError,
    NativeResumeIdentityConflictError,
    NativeResumeValidationError,
    preflight_native_tts_resume,
)
from voiceover_pipeline.history.native_snapshot import persist_prepared_tts_snapshot
from voiceover_pipeline.history.native_view import (
    NativeViewInTransactionError,
    NativeViewNotFoundError,
    NativeViewRevisionConflictError,
    NativeViewValidationError,
)
from voiceover_pipeline.history.repository import (
    ATTEMPT_CALL_TYPE_TTS_CHUNK,
    ATTEMPT_STATUS_SUBMITTING,
    HistoryRepository,
)
from voiceover_pipeline.models import ScriptChunk
from voiceover_pipeline.services.prepare import PreparedPart, PreparedRun

_TABLES = ("runs", "parts", "attempts", "artifacts", "text_sources")

_SECRET = "sk-abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def repository(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as database:
        database.migrate()
        yield HistoryRepository(database)


def _chunks(
    *,
    texts=("Первый.", "Второй."),
    speakers=("host", "guest"),
    cast_voices=("Rachel", None),
    voice_fingerprints=("a" * 64, None),
    pauses=(120, 0),
):
    return [
        ScriptChunk(
            number=position,
            id=f"chunk_{position:02d}",
            text=text,
            speaker=speaker,
            voice=cast,
            voice_fingerprint=voice_fingerprint,
            pause_after_ms=pause,
        )
        for position, (text, speaker, cast, voice_fingerprint, pause) in enumerate(
            zip(texts, speakers, cast_voices, voice_fingerprints, pauses), start=1
        )
    ]


def _prepared(chunks=None, **run_overrides):
    """Build the candidate run whose knobs drive one identity dimension each."""
    run = {
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
        "voice": "alloy",
        "style_prompt": "Read warmly.",
        "prompt_mode": "plain",
    }
    run.update(run_overrides)
    resolved = _chunks() if chunks is None else chunks
    parts = tuple(PreparedPart(chunk=chunk, voice=chunk.voice) for chunk in resolved)
    return PreparedRun(parts=parts, **run)


def _with_chunk(prepared, index, **overrides):
    """Replace one chunk and keep its part voice bound to the new chunk voice."""
    parts = list(prepared.parts)
    chunk = replace(parts[index].chunk, **overrides)
    parts[index] = PreparedPart(chunk=chunk, voice=chunk.voice)
    return replace(prepared, parts=tuple(parts))


def _with_part_voice(prepared, index, voice):
    """Give one part a voice that no longer matches its wrapped chunk."""
    parts = list(prepared.parts)
    parts[index] = replace(parts[index], voice=voice)
    return replace(prepared, parts=tuple(parts))


def _persist(repository, prepared, run_root, **overrides):
    params = {
        "run_root": run_root,
        "user_label": "prod",
        "script_format": "markdown",
        "script_text": "\n\n".join(part.chunk.text for part in prepared.parts),
        "script_path": None,
    }
    params.update(overrides)
    return persist_prepared_tts_snapshot(repository, prepared=prepared, **params)


def _preflight(repository, snapshot, prepared, run_root, **overrides):
    params = {
        "repository": repository,
        "run_uuid": snapshot.run.run_uuid,
        "expected_revision": snapshot.run.revision,
        "prepared": prepared,
        "script_format": "markdown",
        "run_root": run_root,
    }
    params.update(overrides)
    return preflight_native_tts_resume(**params)


def _row_counts(repository):
    return {
        table: repository._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _TABLES
    }


def test_same_prepared_run_root_and_revision_passes(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    view = _preflight(repository, snapshot, prepared, run_root)

    assert view.run.run_uuid == snapshot.run.run_uuid
    assert view.run.revision == snapshot.run.revision
    assert view.script_format == "markdown"
    assert [part.fingerprint for part in view.parts] == [
        part.fingerprint for part in snapshot.parts
    ]


def test_run_voice_applies_to_a_part_without_its_own_cast_voice(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    view = _preflight(repository, snapshot, prepared, run_root)

    assert view.parts[1].cast_voice is None
    assert view.parts[1].effective_voice == "alloy"


def test_equivalent_run_root_string_still_matches(tmp_path, repository):
    run_root = tmp_path / "run"
    nested = run_root / "nested"
    nested.mkdir(parents=True)
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    view = _preflight(repository, snapshot, prepared, run_root / "nested" / "..")

    assert view.run.run_root == str(run_root.resolve())


def test_matching_opaque_identity_passes_and_a_changed_one_fails(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(
        repository,
        prepared,
        run_root,
        voice_identity="preset:voice_a:" + "c" * 64,
        synthesis_identity="d" * 64,
    )

    view = _preflight(
        repository,
        snapshot,
        prepared,
        run_root,
        voice_identity="preset:voice_a:" + "c" * 64,
        synthesis_identity="d" * 64,
    )
    assert view.run.run_uuid == snapshot.run.run_uuid

    with pytest.raises(NativeResumeIdentityConflictError):
        _preflight(
            repository,
            snapshot,
            prepared,
            run_root,
            voice_identity="preset:voice_b:" + "c" * 64,
            synthesis_identity="d" * 64,
        )
    with pytest.raises(NativeResumeIdentityConflictError):
        _preflight(
            repository,
            snapshot,
            prepared,
            run_root,
            voice_identity="preset:voice_a:" + "c" * 64,
            synthesis_identity=None,
        )


_CANDIDATE_MUTATIONS = {
    "provider": lambda prepared: replace(prepared, provider="openrouter-tts"),
    "model": lambda prepared: replace(prepared, model="openai/gpt-4o"),
    "run_voice": lambda prepared: replace(prepared, voice="verse"),
    "style_prompt": lambda prepared: replace(prepared, style_prompt="Read coldly."),
    "no_style_prompt": lambda prepared: replace(prepared, style_prompt=None),
    "prompt_mode": lambda prepared: replace(prepared, prompt_mode="ssml"),
    "text": lambda prepared: _with_chunk(prepared, 0, text="Изменённый текст."),
    "speaker": lambda prepared: _with_chunk(prepared, 0, speaker="narrator"),
    "cast_voice": lambda prepared: _with_chunk(prepared, 0, voice="OtherVoice"),
    "voice_fingerprint": lambda prepared: _with_chunk(prepared, 0, voice_fingerprint="b" * 64),
    "pause": lambda prepared: _with_chunk(prepared, 0, pause_after_ms=42),
    "chunk_number": lambda prepared: _with_chunk(prepared, 0, number=2),
    "generated_id": lambda prepared: _with_chunk(prepared, 0, id="chunk_99"),
    "chunk_order": lambda prepared: replace(prepared, parts=(prepared.parts[1], prepared.parts[0])),
    "part_count": lambda prepared: replace(prepared, parts=(prepared.parts[0],)),
    "part_voice": lambda prepared: _with_part_voice(prepared, 0, "Rachel2"),
}


@pytest.mark.parametrize(
    "mutate", list(_CANDIDATE_MUTATIONS.values()), ids=list(_CANDIDATE_MUTATIONS)
)
def test_changed_candidate_identity_is_rejected(tmp_path, repository, mutate):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    with pytest.raises(NativeResumeIdentityConflictError):
        _preflight(repository, snapshot, mutate(prepared), run_root)


_ARGUMENT_MUTATIONS = {
    "run_root": lambda tmp_path: {"run_root": tmp_path / "other"},
    "script_format": lambda tmp_path: {"script_format": "dialogue"},
    "voice_identity": lambda tmp_path: {"voice_identity": "preset:other:abc"},
    "synthesis_identity": lambda tmp_path: {"synthesis_identity": "deadbeef"},
}


@pytest.mark.parametrize(
    "overrides", list(_ARGUMENT_MUTATIONS.values()), ids=list(_ARGUMENT_MUTATIONS)
)
def test_changed_run_argument_is_rejected(tmp_path, repository, overrides):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    kwargs = overrides(tmp_path)
    candidate_root = kwargs.pop("run_root", run_root)

    with pytest.raises(NativeResumeIdentityConflictError):
        _preflight(repository, snapshot, prepared, candidate_root, **kwargs)


def test_stale_expected_revision_is_rejected(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    repository.advance_run_revision(snapshot.run.run_uuid, expected_revision=snapshot.run.revision)

    with pytest.raises(NativeViewRevisionConflictError):
        _preflight(repository, snapshot, prepared, run_root)


def test_none_revision_cannot_bypass_stale_run_check(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    advanced = repository.advance_run_revision(
        snapshot.run.run_uuid, expected_revision=snapshot.run.revision
    )

    with pytest.raises(NativeResumeValidationError):
        _preflight(repository, snapshot, prepared, run_root, expected_revision=None)
    assert repository.get_run(snapshot.run.run_uuid).revision == advanced.revision


def test_unknown_run_is_rejected(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(NativeViewNotFoundError):
        preflight_native_tts_resume(
            repository,
            str(uuid.uuid4()),
            expected_revision=1,
            prepared=_prepared(),
            script_format="markdown",
            run_root=run_root,
        )


@pytest.mark.parametrize("revision", [0, -1, True])
def test_invalid_expected_revision_is_rejected(tmp_path, repository, revision):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    with pytest.raises(NativeViewValidationError):
        _preflight(repository, snapshot, prepared, run_root, expected_revision=revision)


def test_open_caller_transaction_is_refused_without_touching_rows(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    before = _row_counts(repository)

    with pytest.raises(NativeViewInTransactionError):
        with repository.transaction():
            _preflight(repository, snapshot, prepared, run_root)

    assert _row_counts(repository) == before


def test_run_root_resolution_error_is_typed_and_privacy_safe(tmp_path, repository, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    def fail_resolution(_path):
        raise OSError(f"private root {_SECRET}")

    monkeypatch.setattr(
        "voiceover_pipeline.history.native_resume._normalize_run_root", fail_resolution
    )
    with pytest.raises(NativeResumeValidationError) as excinfo:
        _preflight(repository, snapshot, prepared, run_root)
    assert _SECRET not in str(excinfo.value)
    assert not repository._connection.in_transaction


def test_malformed_candidate_identity_fails_typed_without_echoing_values(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    with pytest.raises(NativeResumeIdentityConflictError) as changed:
        _preflight(repository, snapshot, replace(prepared, model=_SECRET), run_root)
    assert _SECRET not in str(changed.value)

    with pytest.raises(NativeResumeIdentityConflictError) as mismatched:
        _preflight(repository, snapshot, _with_part_voice(prepared, 0, _SECRET), run_root)
    assert _SECRET not in str(mismatched.value)

    with pytest.raises(NativeResumeValidationError) as unusable:
        _preflight(repository, snapshot, replace(prepared, prompt_mode=""), run_root)
    assert _SECRET not in str(unusable.value)


@pytest.mark.parametrize("prepared", ["not-a-prepared-run", None])
def test_unusable_candidate_shape_is_rejected_before_any_read(tmp_path, repository, prepared):
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(NativeResumeValidationError) as excinfo:
        preflight_native_tts_resume(
            repository,
            str(uuid.uuid4()),
            expected_revision=1,
            prepared=prepared,
            script_format="markdown",
            run_root=run_root,
        )
    # A nonexistent run would raise NativeViewNotFoundError, so the typed candidate
    # error proves the shape is checked before the read.
    assert "prepared" in str(excinfo.value)
    assert isinstance(excinfo.value, NativeResumeError)


@pytest.mark.parametrize("malformed", ["wrong-part-type", "wrong-chunk-text"])
def test_malformed_part_is_rejected_before_any_database_read(tmp_path, repository, malformed):
    prepared = _prepared()
    if malformed == "wrong-part-type":
        prepared = replace(prepared, parts=("sk-synthetic-part",))
    else:
        prepared = _with_chunk(prepared, 0, text=object())

    with pytest.raises(NativeResumeValidationError) as excinfo:
        preflight_native_tts_resume(
            repository,
            str(uuid.uuid4()),
            expected_revision=1,
            prepared=prepared,
            script_format="markdown",
            run_root=tmp_path / "run",
        )
    assert "sk-synthetic-part" not in str(excinfo.value)
    assert not repository._connection.in_transaction


def test_altered_source_script_file_is_not_read_and_keeps_identity(
    tmp_path, repository, monkeypatch
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    # An edited comment or whitespace block that would change a hash of the raw file
    # while every prepared chunk stays identical.
    (run_root / "script.md").write_text(
        "# Совсем другой сценарий\n\nизменены комментарии и пробелы\n", encoding="utf-8"
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("the preflight must not read the raw source script file")

    monkeypatch.setattr(Path, "read_text", _forbidden)
    monkeypatch.setattr(Path, "read_bytes", _forbidden)
    monkeypatch.setattr(Path, "open", _forbidden)

    view = _preflight(repository, snapshot, prepared, run_root)

    assert view.run.run_uuid == snapshot.run.run_uuid
    assert view.script_text == "\n\n".join(part.chunk.text for part in prepared.parts)


def test_preflight_writes_no_database_row_and_no_file(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)
    before_counts = _row_counts(repository)
    before_revision = repository.get_run(snapshot.run.run_uuid).revision
    before_entries = sorted(path.name for path in run_root.iterdir())

    _preflight(repository, snapshot, prepared, run_root)
    with pytest.raises(NativeResumeIdentityConflictError):
        _preflight(repository, snapshot, replace(prepared, voice="verse"), run_root)

    assert _row_counts(repository) == before_counts
    assert repository.get_run(snapshot.run.run_uuid).revision == before_revision
    assert sorted(path.name for path in run_root.iterdir()) == before_entries
    assert not (run_root / "run_state.json").exists()


def test_stored_paid_attempt_record_does_not_grant_action(tmp_path, repository):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared()
    snapshot = _persist(repository, prepared, run_root)

    without_marker = _preflight(repository, snapshot, prepared, run_root)

    attempt = repository.add_attempt(
        snapshot.run.run_uuid,
        call_type=ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=snapshot.parts[0].part_uuid,
        provider="polza-tts",
        model="elevenlabs/multilingual-v2",
        status=ATTEMPT_STATUS_SUBMITTING,
    )
    before_counts = _row_counts(repository)
    before_revision = repository.get_run(snapshot.run.run_uuid).revision

    with_marker = _preflight(repository, snapshot, prepared, run_root)

    # The marker is evidence only: the preflight outcome is unchanged, the marker is
    # returned as a plain record, and nothing was reserved, advanced, or written.
    assert with_marker.run_fingerprint == without_marker.run_fingerprint
    assert [record.attempt_uuid for record in with_marker.attempts] == [attempt.attempt_uuid]
    assert with_marker.attempts[0].status == ATTEMPT_STATUS_SUBMITTING
    assert _row_counts(repository) == before_counts
    assert repository.get_run(snapshot.run.run_uuid).revision == before_revision


def test_chat_audio_fallback_voice_change_is_rejected(tmp_path, repository):
    """The ordinary polza-chat-audio route binds its fallback voice into identity."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    prepared = _prepared(
        provider="polza-chat-audio",
        model="openai/gpt-audio-mini",
        voice="ash",
        fallback_voice="onyx",
    )
    snapshot = _persist(repository, prepared, run_root)

    # The unchanged candidate passes; a different fallback voice fails closed with
    # the fixed conflict message that echoes no value.
    view = _preflight(repository, snapshot, prepared, run_root)
    assert view.run.run_uuid == snapshot.run.run_uuid

    with pytest.raises(NativeResumeIdentityConflictError) as excinfo:
        _preflight(repository, snapshot, replace(prepared, fallback_voice="echo"), run_root)

    assert "fallback voice changed" in str(excinfo.value)
    assert "echo" not in str(excinfo.value)
