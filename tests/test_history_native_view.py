"""Contract tests for the read-only native prepared-TTS view.

These tests cover ``voiceover_pipeline.history.native_view``: one coherent view
read under a single deferred read transaction, a full prepared-snapshot
roundtrip (exact text, style, effective and cast voice, pause, and structural
identity), a multi-part run whose per-part sources are indexed and resolved by
their own part, fail-closed rejection of a legacy import, a non-``tts``
operation, an unsupported origin or version, a malformed config, a non-contiguous
part, an altered part row, a part source naming a part outside the run, a text
source whose non-null part or artifact link names another run's row, a source
part that contradicts the part its own artifact names, an artifact naming a
different part than its attempt, and a missing, duplicate, or altered native text
or style source, the stale ``expected_revision`` check, the
already-open-transaction refusal, the writer's pre-insert refusal of an empty
style prompt, a provider outside the native writer's allowlist that is refused
even after its part and run fingerprints are recomputed to stay internally
consistent, malformed attempt/artifact JSON returning a typed integrity error,
and the deterministic proof that a concurrent writer committing
after the first ``SELECT`` cannot tear the view or be blocked by a write lock.

Every fixture is synthetic and lives under ``tmp_path``. No ``.env`` value, user
output directory, provider, model, FFmpeg, or network path is touched: the view
only reads SQLite rows and never opens a JSON file, run directory, provider, or
file.
"""

import hashlib
import json
import uuid

import pytest

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.native_snapshot import (
    NATIVE_SNAPSHOT_ORIGIN,
    NativeSnapshotValidationError,
    _part_fingerprint,
    _run_identity_payload,
    _sha256_text,
    _snapshot_fingerprint,
    persist_prepared_tts_snapshot,
)
from voiceover_pipeline.history.native_view import (
    NativeViewIntegrityError,
    NativeViewInTransactionError,
    NativeViewNotFoundError,
    NativeViewRevisionConflictError,
    NativeViewValidationError,
    load_native_tts_view,
)
from voiceover_pipeline.history.repository import (
    ARTIFACT_ROLE_PAID_RAW_AUDIO,
    PATH_KIND_MANAGED_RELATIVE,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
)
from voiceover_pipeline.models import ScriptChunk
from voiceover_pipeline.services.prepare import PreparedPart, PreparedRun

_TABLES = ("runs", "parts", "attempts", "artifacts", "text_sources")


@pytest.fixture
def repository(tmp_path):
    with HistoryDatabase(tmp_path / "history.sqlite3") as database:
        database.migrate()
        yield HistoryRepository(database)


def _scenario(
    *,
    texts=("Первый.", "Второй."),
    cast_voices=("Rachel", None),
    pauses=(0, 250),
    style_prompt="Read warmly.",
    provider="polza-tts",
    model="openai/gpt-4o-mini-tts",
    voice="alloy",
    prompt_mode="plain",
):
    """Build a two-part prepared run whose knobs drive one identity dimension each."""
    chunks = [
        ScriptChunk(
            number=position,
            id=f"chunk_{position:02d}",
            text=text,
            voice=cast,
            pause_after_ms=pause,
        )
        for position, (text, cast, pause) in enumerate(zip(texts, cast_voices, pauses), start=1)
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


def _persist(repository, prepared, run_root, *, label="prod", script_format="markdown"):
    return persist_prepared_tts_snapshot(
        repository,
        prepared=prepared,
        run_root=run_root,
        user_label=label,
        script_format=script_format,
        script_text=_script_text(prepared),
        script_path=None,
    )


def _row_count(repository, table):
    return repository._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _apply_config_tamper(repository, snapshot, mutate):
    config = json.loads(json.dumps(snapshot.run.config_snapshot, ensure_ascii=False))
    mutate(config)
    repository._connection.execute(
        "UPDATE runs SET config_snapshot = ? WHERE run_uuid = ?",
        (json.dumps(config, ensure_ascii=False), snapshot.run.run_uuid),
    )


def _insert_cross_run_text_source(
    repository,
    *,
    run_uuid,
    part_uuid=None,
    artifact_uuid=None,
    content="Чужой первый.",
    kind=TEXT_KIND_TTS_SCRIPT,
):
    """Insert one source whose non-null links may not resolve inside its own run.

    The composite foreign keys normally block a cross-run part or artifact link,
    so this direct insert disables them to prove a reader refuses the row itself
    instead of relying on the database constraint.
    """
    repository._connection.execute("PRAGMA foreign_keys = OFF")
    try:
        repository._connection.execute(
            "INSERT INTO text_sources (text_source_uuid, run_uuid, part_uuid, artifact_uuid, "
            "kind, content, content_hash, language, origin, text_completeness, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, 'complete', ?)",
            (
                str(uuid.uuid4()),
                run_uuid,
                part_uuid,
                artifact_uuid,
                kind,
                content,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                NATIVE_SNAPSHOT_ORIGIN,
                "2020-01-01T00:00:00Z",
            ),
        )
    finally:
        repository._connection.execute("PRAGMA foreign_keys = ON")


def _reforge_provider(repository, snapshot, provider):
    """Rewrite ``provider`` and every fingerprint it feeds, keeping the snapshot consistent.

    Changing only the stored provider would leave a stale part and run fingerprint,
    so the view's ordinary fingerprint checks -- not the provider rule -- would
    already refuse it. This rebuilds the run identity, each part fingerprint, and
    the run fingerprint exactly as the committed rows read after the change, so a
    snapshot that must still be refused can only be refused by the provider rule.
    """
    run_uuid = snapshot.run.run_uuid
    config = json.loads(json.dumps(snapshot.run.config_snapshot, ensure_ascii=False))
    config["provider"] = provider
    sources = repository.get_text_sources(run_uuid)
    direction = next(source for source in sources if source.kind == TEXT_KIND_TTS_DIRECTION)
    script = next(
        source
        for source in sources
        if source.kind == TEXT_KIND_TTS_SCRIPT
        and source.part_uuid is None
        and source.artifact_uuid is None
    )
    identity = _run_identity_payload(
        provider=provider,
        model=config["model"],
        voice=config["voice"],
        style_prompt=direction.content,
        prompt_mode=config["prompt_mode"],
        voice_identity=config["voice_identity"],
        synthesis_identity=config["synthesis_identity"],
    )
    part_fingerprints = []
    parts = repository.get_parts(run_uuid)
    for position, (entry, part) in enumerate(zip(config["parts"], parts), start=1):
        fingerprint = _part_fingerprint(
            identity,
            chunk=ScriptChunk(
                number=entry["number"],
                id=entry["id"],
                text=part.prepared_text,
                speaker=entry["speaker"],
                voice=entry["cast_voice"],
                voice_fingerprint=entry["voice_fingerprint"],
                pause_after_ms=entry["pause_after_ms"],
            ),
            position=position,
            effective_voice=entry["effective_voice"],
        )
        entry["fingerprint"] = fingerprint
        part_fingerprints.append(fingerprint)
        repository._connection.execute(
            "UPDATE parts SET fingerprint = ? WHERE part_uuid = ?",
            (fingerprint, part.part_uuid),
        )
    config["snapshot_fingerprint"] = _snapshot_fingerprint(
        identity,
        script_format=config["script_format"],
        script_sha256=_sha256_text(script.content),
        part_fingerprints=part_fingerprints,
    )
    repository._connection.execute(
        "UPDATE runs SET config_snapshot = ? WHERE run_uuid = ?",
        (json.dumps(config, ensure_ascii=False), run_uuid),
    )


def _duplicate_part_source(repository, snapshot, part_index=0):
    part = snapshot.parts[part_index]
    text = part.prepared_text
    repository._connection.execute(
        "INSERT INTO text_sources (text_source_uuid, run_uuid, part_uuid, artifact_uuid, "
        "kind, content, content_hash, language, origin, text_completeness, created_at) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, ?, 'complete', ?)",
        (
            str(uuid.uuid4()),
            snapshot.run.run_uuid,
            part.part_uuid,
            TEXT_KIND_TTS_SCRIPT,
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            NATIVE_SNAPSHOT_ORIGIN,
            "2020-01-01T00:00:00Z",
        ),
    )


_TAMPER_CASES = {
    "part text": lambda repo, snap: repo._connection.execute(
        "UPDATE parts SET prepared_text = 'tampered' WHERE part_uuid = ?",
        (snap.parts[0].part_uuid,),
    ),
    "part fingerprint": lambda repo, snap: repo._connection.execute(
        "UPDATE parts SET fingerprint = ? WHERE part_uuid = ?",
        ("0" * 64, snap.parts[0].part_uuid),
    ),
    "part voice": lambda repo, snap: repo._connection.execute(
        "UPDATE parts SET voice = 'Other' WHERE part_uuid = ?", (snap.parts[0].part_uuid,)
    ),
    "missing part": lambda repo, snap: repo._connection.execute(
        "DELETE FROM parts WHERE part_uuid = ?", (snap.parts[1].part_uuid,)
    ),
    "missing part source": lambda repo, snap: repo._connection.execute(
        "DELETE FROM text_sources WHERE kind = ? AND part_uuid = ?",
        (TEXT_KIND_TTS_SCRIPT, snap.parts[0].part_uuid),
    ),
    "duplicate part source": _duplicate_part_source,
    "duplicate second part source": lambda repo, snap: _duplicate_part_source(repo, snap, 1),
    "run operation": lambda repo, snap: repo._connection.execute(
        "UPDATE runs SET operation = 'asr' WHERE run_uuid = ?", (snap.run.run_uuid,)
    ),
    "missing run source": lambda repo, snap: repo._connection.execute(
        "DELETE FROM text_sources WHERE kind = ? AND part_uuid IS NULL", (TEXT_KIND_TTS_SCRIPT,)
    ),
    "altered run source": lambda repo, snap: repo._connection.execute(
        "UPDATE text_sources SET content = 'other script' WHERE kind = ? AND part_uuid IS NULL",
        (TEXT_KIND_TTS_SCRIPT,),
    ),
    "missing style source": lambda repo, snap: repo._connection.execute(
        "DELETE FROM text_sources WHERE kind = ?", (TEXT_KIND_TTS_DIRECTION,)
    ),
    "altered style source": lambda repo, snap: repo._connection.execute(
        "UPDATE text_sources SET content = 'Read coldly.' WHERE kind = ?",
        (TEXT_KIND_TTS_DIRECTION,),
    ),
    "unsupported origin": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"operation_origin": "legacy_import"})
    ),
    "unsupported version": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"snapshot_version": 2})
    ),
    "config without origin": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.pop("operation_origin")
    ),
    "config model": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"model": "other/model"})
    ),
    "config run fingerprint": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"snapshot_fingerprint": "a" * 64})
    ),
    "config script format": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"script_format": "plain"})
    ),
    "config script hash": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"script_sha256": "b" * 64})
    ),
    "config part count": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c.update({"part_count": 1})
    ),
    "config part cast voice": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c["parts"][0].update({"cast_voice": "Other"})
    ),
    "config part number": lambda repo, snap: _apply_config_tamper(
        repo, snap, lambda c: c["parts"][0].update({"number": 3})
    ),
    "config not a mapping": lambda repo, snap: repo._connection.execute(
        "UPDATE runs SET config_snapshot = ? WHERE run_uuid = ?",
        ('"not a mapping"', snap.run.run_uuid),
    ),
    "config invalid json": lambda repo, snap: repo._connection.execute(
        "UPDATE runs SET config_snapshot = ? WHERE run_uuid = ?",
        ("not json", snap.run.run_uuid),
    ),
}


def test_view_roundtrips_exact_prepared_snapshot(repository, tmp_path):
    prepared = _scenario()
    snapshot = _persist(repository, prepared, tmp_path / "runs" / "roundtrip")

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.run.run_uuid == snapshot.run.run_uuid
    assert view.run.revision == 1
    assert view.run.operation == "tts"
    assert view.run_fingerprint == snapshot.fingerprint
    assert view.script_format == "markdown"
    assert view.script_path is None
    assert view.script_text == _script_text(prepared)
    assert view.style_prompt == "Read warmly."
    assert [part.text for part in view.parts] == ["Первый.", "Второй."]
    assert [part.number for part in view.parts] == [1, 2]
    assert [part.chunk_id for part in view.parts] == ["chunk_01", "chunk_02"]
    assert [part.effective_voice for part in view.parts] == ["Rachel", "alloy"]
    assert [part.cast_voice for part in view.parts] == ["Rachel", None]
    assert [part.pause_after_ms for part in view.parts] == [0, 250]
    assert [part.fingerprint for part in view.parts] == [p.fingerprint for p in snapshot.parts]
    assert [part.record.part_uuid for part in view.parts] == [
        part.part_uuid for part in snapshot.parts
    ]
    assert view.attempts == ()
    assert view.artifacts == ()
    assert {source.kind for source in view.text_sources} == {
        TEXT_KIND_TTS_SCRIPT,
        TEXT_KIND_TTS_DIRECTION,
    }


def test_multi_part_run_indexes_each_part_source_by_its_part(repository, tmp_path):
    prepared = _scenario(
        texts=("Первый.", "Второй.", "Третий."),
        cast_voices=("Rachel", None, "alloy"),
        pauses=(0, 120, 240),
    )

    snapshot = _persist(repository, prepared, tmp_path / "runs" / "multi-parts")

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    # Each part resolves its own indexed source, so a source shuffled between parts
    # would break the text-to-part agreement instead of silently passing.
    assert [part.number for part in view.parts] == [1, 2, 3]
    assert [part.chunk_id for part in view.parts] == ["chunk_01", "chunk_02", "chunk_03"]
    assert [part.text for part in view.parts] == ["Первый.", "Второй.", "Третий."]
    assert [part.cast_voice for part in view.parts] == ["Rachel", None, "alloy"]
    assert [part.effective_voice for part in view.parts] == ["Rachel", "alloy", "alloy"]
    assert [part.pause_after_ms for part in view.parts] == [0, 120, 240]
    assert [part.fingerprint for part in view.parts] == [p.fingerprint for p in snapshot.parts]
    assert view.run_fingerprint == snapshot.fingerprint


def test_tampered_operation_is_rejected_without_a_revision_bump(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "operation")

    repository._connection.execute(
        "UPDATE runs SET operation = 'asr' WHERE run_uuid = ?", (snapshot.run.run_uuid,)
    )

    # Nothing else moved: the revision and every committed identity still verify,
    # so only the operation check can refuse this run as a prepared-TTS snapshot.
    assert repository.get_run(snapshot.run.run_uuid).revision == 1
    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_view_without_a_style_prompt_reports_none(repository, tmp_path):
    snapshot = _persist(repository, _scenario(style_prompt=None), tmp_path / "runs" / "no-style")

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.style_prompt is None
    assert not [s for s in view.text_sources if s.kind == TEXT_KIND_TTS_DIRECTION]


def test_view_returns_paid_markers_as_records_without_authorizing_resume(repository, tmp_path):
    snapshot = _persist(repository, _scenario(model="elevenlabs/tts"), tmp_path / "runs" / "paid")
    part_uuid = snapshot.parts[0].part_uuid

    advanced, marker = repository.reserve_paid_tts_attempt(
        snapshot.run.run_uuid,
        part_uuid=part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="elevenlabs/tts",
        account_alias="acct",
    )
    assert marker.status == "submitting" and marker.remote_id is None
    accepted_run, accepted = repository.record_polza_media_task_accepted(
        snapshot.run.run_uuid,
        attempt_uuid=marker.attempt_uuid,
        part_uuid=part_uuid,
        expected_revision=advanced.revision,
        remote_task_id="task_ABC123",
    )

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.run.revision == accepted_run.revision
    assert [attempt.status for attempt in view.attempts] == ["remote_accepted"]
    assert [attempt.remote_id for attempt in view.attempts] == ["task_ABC123"]
    # A stored marker is a record, not permission to poll or download, and no
    # committed artifact is inferred without on-disk evidence.
    assert view.artifacts == ()


def test_artifact_linked_to_its_attempts_own_part_is_accepted(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "artifact-ok")
    part_uuid = snapshot.parts[0].part_uuid
    _, attempt = repository.reserve_paid_tts_attempt(
        snapshot.run.run_uuid,
        part_uuid=part_uuid,
        expected_revision=1,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        account_alias="acct",
    )
    raw = repository.add_artifact(
        snapshot.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_01.mp3",
        part_uuid=part_uuid,
        attempt_uuid=attempt.attempt_uuid,
    )
    # A run-level role carries no part of its own, so no part equality applies.
    run_level = repository.add_artifact(
        snapshot.run.run_uuid,
        role="main_audio",
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="out/main.mp3",
        attempt_uuid=attempt.attempt_uuid,
    )

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    # ``get_artifacts`` orders by created_at then UUID, and same-tick rows tie on
    # the UUID, so compare as a set rather than a fixed order.
    assert {artifact.artifact_uuid for artifact in view.artifacts} == {
        raw.artifact_uuid,
        run_level.artifact_uuid,
    }
    assert [record.attempt_uuid for record in view.attempts] == [attempt.attempt_uuid]


def test_artifact_linked_to_another_parts_attempt_is_rejected(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "artifact-mismatch")
    first_part = snapshot.parts[0].part_uuid
    second_part = snapshot.parts[1].part_uuid
    _, attempt = repository.reserve_paid_tts_attempt(
        snapshot.run.run_uuid,
        part_uuid=first_part,
        expected_revision=1,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        account_alias="acct",
    )
    # Both links are inside the run, so only the part-to-attempt equality can tell
    # that this artifact is not evidence for the part it names.
    repository.add_artifact(
        snapshot.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_02.mp3",
        part_uuid=second_part,
        attempt_uuid=attempt.attempt_uuid,
    )

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_completed_run_is_readable_but_infers_no_resume_permission(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "completed")
    repository.advance_run_revision(snapshot.run.run_uuid, expected_revision=1, status="completed")

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.run.status == "completed"
    assert view.run.revision == 2
    assert [part.text for part in view.parts] == ["Первый.", "Второй."]
    # The view exposes committed rows only; it carries no resume authorization.
    assert not hasattr(view, "resumable")


def test_stale_expected_revision_is_rejected_against_the_pinned_revision(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "stale")
    repository.advance_run_revision(snapshot.run.run_uuid, expected_revision=1)

    with pytest.raises(NativeViewRevisionConflictError):
        load_native_tts_view(repository, snapshot.run.run_uuid, expected_revision=1)

    current = load_native_tts_view(repository, snapshot.run.run_uuid, expected_revision=2)
    assert current.run.revision == 2


@pytest.mark.parametrize("bad", ["", "not-a-uuid", "123", None, 12])
def test_invalid_run_uuid_is_rejected_before_any_read(repository, tmp_path, bad):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "uuid")

    with pytest.raises(NativeViewValidationError):
        load_native_tts_view(repository, bad)

    assert not repository._connection.in_transaction
    assert _row_count(repository, "runs") == 1
    assert snapshot.run.run_uuid


@pytest.mark.parametrize("bad", [0, -1, True, "1", 1.0])
def test_invalid_expected_revision_is_rejected(repository, tmp_path, bad):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "revision")

    with pytest.raises(NativeViewValidationError):
        load_native_tts_view(repository, snapshot.run.run_uuid, expected_revision=bad)


def test_view_refuses_to_run_inside_an_open_transaction(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "outer")

    with repository.transaction():
        with pytest.raises(NativeViewInTransactionError):
            load_native_tts_view(repository, snapshot.run.run_uuid)

    assert not repository._connection.in_transaction


def test_missing_run_is_reported(repository, tmp_path):
    with pytest.raises(NativeViewNotFoundError):
        load_native_tts_view(repository, str(uuid.uuid4()))


def test_legacy_imported_run_is_rejected(repository, tmp_path):
    run_root = tmp_path / "out" / "prod"
    run_root.mkdir(parents=True)
    with repository.transaction():
        legacy, _ = repository.create_legacy_run(
            operation="tts",
            run_root=str(run_root.resolve()),
            legacy_source_root=str(run_root.resolve()),
        )

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, legacy.run_uuid)


@pytest.mark.parametrize("label", sorted(_TAMPER_CASES))
def test_tampered_snapshot_fails_closed(repository, tmp_path, label):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "tamper")

    _TAMPER_CASES[label](repository, snapshot)

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_integrity_failure_never_echoes_the_tampered_value(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "leak")
    secret = "sk-syntheticSyntheticSyntheticValue"

    _apply_config_tamper(repository, snapshot, lambda c: c.update({"model": secret}))
    with pytest.raises(NativeViewIntegrityError) as config_error:
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert secret not in str(config_error.value)

    repository._connection.execute(
        "UPDATE parts SET prepared_text = ? WHERE part_uuid = ?",
        (secret, snapshot.parts[0].part_uuid),
    )
    with pytest.raises(NativeViewIntegrityError) as text_error:
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert secret not in str(text_error.value)


def test_view_writes_no_rows_and_touches_no_disk(repository, tmp_path):
    run_root = tmp_path / "runs" / "untouched"
    snapshot = _persist(repository, _scenario(), run_root)
    before = {table: _row_count(repository, table) for table in _TABLES}

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.run.run_uuid == snapshot.run.run_uuid
    assert {table: _row_count(repository, table) for table in _TABLES} == before
    assert not run_root.exists()
    assert list(tmp_path.rglob("*.json")) == []
    assert not repository._connection.in_transaction


def test_concurrent_writer_mid_read_keeps_a_coherent_old_view_without_a_write_lock(
    repository, tmp_path, monkeypatch
):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "concurrent")
    run_uuid = snapshot.run.run_uuid
    part_uuid = snapshot.parts[0].part_uuid
    observed_in_transaction = []

    with HistoryDatabase(tmp_path / "history.sqlite3") as second_database:
        second_database.connect()
        second = HistoryRepository(second_database)
        original_get_parts = repository.get_parts

        def injecting_get_parts(target_uuid):
            # Runs after the view's first SELECT pinned the WAL snapshot and
            # before it reads the parts, so this second-connection commit lands
            # strictly inside the read transaction.
            observed_in_transaction.append(repository._connection.in_transaction)
            second.reserve_paid_tts_attempt(
                target_uuid,
                part_uuid=part_uuid,
                expected_revision=1,
                provider="polza-tts",
                model="elevenlabs/tts",
                account_alias="acct",
            )
            return original_get_parts(target_uuid)

        with monkeypatch.context() as patched:
            patched.setattr(repository, "get_parts", injecting_get_parts)
            view = load_native_tts_view(repository, run_uuid)

    # The second writer committed while the read transaction was open, so the
    # view held no write lock; and the view itself is the coherent old revision.
    assert observed_in_transaction == [True]
    assert view.run.revision == 1
    assert view.attempts == ()
    assert [part.text for part in view.parts] == ["Первый.", "Второй."]
    assert not repository._connection.in_transaction

    fresh = load_native_tts_view(repository, run_uuid)
    assert fresh.run.revision == 2
    assert [attempt.status for attempt in fresh.attempts] == ["submitting"]


def test_part_source_naming_a_part_outside_the_run_is_rejected(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "foreign-source")
    other = _persist(
        repository,
        _scenario(texts=("Чужой первый.", "Чужой второй.")),
        tmp_path / "runs" / "foreign-other",
    )
    _insert_cross_run_text_source(
        repository, run_uuid=snapshot.run.run_uuid, part_uuid=other.parts[0].part_uuid
    )

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_source_linked_to_an_artifact_but_naming_another_runs_part_is_rejected(
    repository, tmp_path
):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "artifact-source")
    other = _persist(
        repository,
        _scenario(texts=("Чужой первый.", "Чужой второй.")),
        tmp_path / "runs" / "artifact-source-other",
    )
    artifact = repository.add_artifact(
        snapshot.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_01.mp3",
    )
    # A non-null ``artifact_uuid`` takes the source out of the per-part script
    # index, so only a reader that validates the source's own part link can tell
    # that this run-level-looking record carries another run's part.
    _insert_cross_run_text_source(
        repository,
        run_uuid=snapshot.run.run_uuid,
        part_uuid=other.parts[0].part_uuid,
        artifact_uuid=artifact.artifact_uuid,
    )

    before = {table: _row_count(repository, table) for table in _TABLES}
    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    # A fail-closed read leaves every committed row exactly where it was.
    assert {table: _row_count(repository, table) for table in _TABLES} == before
    assert not repository._connection.in_transaction


def test_source_naming_another_runs_artifact_is_rejected(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "foreign-artifact")
    other = _persist(
        repository,
        _scenario(texts=("Чужой первый.", "Чужой второй.")),
        tmp_path / "runs" / "foreign-artifact-other",
    )
    foreign_artifact = repository.add_artifact(
        other.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_01.mp3",
    )
    _insert_cross_run_text_source(
        repository,
        run_uuid=snapshot.run.run_uuid,
        artifact_uuid=foreign_artifact.artifact_uuid,
    )

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_source_part_contradicting_its_artifacts_part_is_rejected(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "source-artifact-part")
    first, second = snapshot.parts
    artifact = repository.add_artifact(
        snapshot.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_02.mp3",
        part_uuid=second.part_uuid,
    )
    # Both links resolve inside this run, so only the part-to-artifact equality can
    # tell that this source is attached to another part's bytes.
    repository.add_text_source(
        snapshot.run.run_uuid,
        kind=TEXT_KIND_TTS_SCRIPT,
        origin=NATIVE_SNAPSHOT_ORIGIN,
        content=first.prepared_text,
        part_uuid=first.part_uuid,
        artifact_uuid=artifact.artifact_uuid,
        content_hash=hashlib.sha256(first.prepared_text.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(NativeViewIntegrityError):
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert not repository._connection.in_transaction


def test_source_attached_to_a_same_run_run_level_artifact_stays_readable(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "run-level-source")
    artifact = repository.add_artifact(
        snapshot.run.run_uuid,
        role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
        path_kind=PATH_KIND_MANAGED_RELATIVE,
        path="raw/chunk_01.mp3",
    )
    # A later ASR or timing stage may store its text against the run-level audio an
    # artifact names, so a same-run run-level link must stay readable.
    source = repository.add_text_source(
        snapshot.run.run_uuid,
        kind="asr",
        origin="asr_local",
        content="Recognized text.",
        artifact_uuid=artifact.artifact_uuid,
    )

    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert source.text_source_uuid in {record.text_source_uuid for record in view.text_sources}
    assert [part.text for part in view.parts] == ["Первый.", "Второй."]


def test_writer_rejects_an_empty_style_prompt_instead_of_committing_an_unreadable_run(
    repository, tmp_path
):
    run_root = tmp_path / "runs" / "empty-style"

    with pytest.raises(NativeSnapshotValidationError):
        _persist(repository, _scenario(style_prompt=""), run_root)

    # Before the writer rejected this, it committed a run whose identity hashed
    # ``""`` while writing no ``tts_direction`` source, so this view then refused
    # the writer's own committed run. No row and no JSON survive now.
    assert _row_count(repository, "runs") == 0
    assert _row_count(repository, "parts") == 0
    assert _row_count(repository, "text_sources") == 0
    assert list(tmp_path.rglob("*.json")) == []

    # A missing prompt and a real prompt both still round-trip.
    bare = _persist(repository, _scenario(style_prompt=None), tmp_path / "runs" / "no-style")
    assert load_native_tts_view(repository, bare.run.run_uuid).style_prompt is None
    styled = _persist(repository, _scenario(), tmp_path / "runs" / "styled")
    prompt = load_native_tts_view(repository, styled.run.run_uuid).style_prompt
    assert prompt == "Read warmly."


def test_reforged_allowlisted_provider_keeps_the_run_readable(repository, tmp_path):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "reforged-allowlisted")

    _reforge_provider(repository, snapshot, "openrouter-tts")

    # Control for the rejection below: the reforge leaves one internally consistent
    # snapshot, which this view still accepts when the provider is allowlisted.
    view = load_native_tts_view(repository, snapshot.run.run_uuid)

    assert view.run.config_snapshot["provider"] == "openrouter-tts"
    assert [part.text for part in view.parts] == ["Первый.", "Второй."]
    assert view.run_fingerprint == view.run.config_snapshot["snapshot_fingerprint"]


@pytest.mark.parametrize("provider", ["qwen-local", "omnivoice-local", "unknown-route"])
def test_unsupported_provider_with_consistent_fingerprints_is_rejected(
    repository, tmp_path, provider
):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / "unsupported-provider")

    _reforge_provider(repository, snapshot, provider)

    assert repository.get_run(snapshot.run.run_uuid).config_snapshot["provider"] == provider
    with pytest.raises(NativeViewIntegrityError) as excinfo:
        load_native_tts_view(repository, snapshot.run.run_uuid)
    # The writer never stored a route whose resume identity it could not capture, so
    # the reader refuses the provider itself, with a message that echoes none.
    assert provider not in str(excinfo.value)
    assert not repository._connection.in_transaction


@pytest.mark.parametrize(
    ("table", "column", "id_column"),
    [
        ("attempts", "usage_json", "attempt_uuid"),
        ("artifacts", "media_metadata_json", "artifact_uuid"),
    ],
)
def test_malformed_attempt_or_artifact_json_fails_with_typed_integrity_error(
    repository, tmp_path, table, column, id_column
):
    snapshot = _persist(repository, _scenario(), tmp_path / "runs" / table)
    if table == "attempts":
        record = repository.add_attempt(
            snapshot.run.run_uuid,
            call_type="tts_chunk",
            part_uuid=snapshot.parts[0].part_uuid,
            provider="polza-tts",
            model="openai/gpt-4o-mini-tts",
            status="submitting",
            usage={"tokens": 1},
        )
        identifier = record.attempt_uuid
    else:
        record = repository.add_artifact(
            snapshot.run.run_uuid,
            role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
            path_kind=PATH_KIND_MANAGED_RELATIVE,
            path="raw/chunk_01.mp3",
            part_uuid=snapshot.parts[0].part_uuid,
            media_metadata={"format": "mp3"},
        )
        identifier = record.artifact_uuid
    repository._connection.execute(
        f"UPDATE {table} SET {column} = ? WHERE {id_column} = ?",
        ('{"secret":"synthetic-placeholder"', identifier),
    )

    with pytest.raises(NativeViewIntegrityError) as excinfo:
        load_native_tts_view(repository, snapshot.run.run_uuid)
    assert "synthetic-placeholder" not in str(excinfo.value)
    assert not repository._connection.in_transaction
