"""Contract tests for the native TTS completion CAS transitions.

These pin the two narrowly scoped compare-and-swap transitions the native
vertical slice adds: committing a converted part and closing a run. Every case
is offline and synthetic. The tests assert revision movement, idempotent repeat,
stale-revision rejection, and ordering/evidence guards so a partial or repeated
commit cannot silently close a run.
"""

import pytest

from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.repository import (
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_FINAL_AUDIO,
    ATTEMPT_CALL_TYPE_TTS_CHUNK,
    ATTEMPT_STATUS_COMPLETED,
    ATTEMPT_STATUS_RAW_SAVED,
    ATTEMPT_STATUS_SUBMITTING,
    PART_STAGE_COMPLETED,
    HistoryPartCompletionConflictError,
    HistoryRepository,
    HistoryRevisionConflictError,
    HistoryRunCompletionConflictError,
    HistoryRunNotFoundError,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


@pytest.fixture
def repository():
    database = HistoryDatabase(":memory:")
    database.connect()
    database.migrate()
    try:
        yield HistoryRepository(database)
    finally:
        database.close()


def _run_with_part(repository):
    run = repository.create_run(operation="tts", run_root="/tmp/native-completion")
    part = repository.add_part(
        run.run_uuid, position=1, prepared_text="hello", voice="Rachel", fingerprint=DIGEST_A
    )
    return run, part


def _raw_saved_attempt(repository, run, part, status=ATTEMPT_STATUS_RAW_SAVED):
    return repository.add_attempt(
        run.run_uuid,
        call_type=ATTEMPT_CALL_TYPE_TTS_CHUNK,
        part_uuid=part.part_uuid,
        provider="polza-tts",
        model="elevenlabs/text-to-speech-turbo-2-5",
        status=status,
    )


def _complete_part(repository, run, part, attempt, *, expected_revision, digest=DIGEST_A):
    return repository.record_tts_part_completed(
        run.run_uuid,
        part_uuid=part.part_uuid,
        attempt_uuid=attempt.attempt_uuid,
        expected_revision=expected_revision,
        path="chunks/chunk_01.mp3",
        mime="audio/mpeg",
        size_bytes=4,
        sha256=digest,
        media_metadata={"duration_ms": 1000, "processing_version": 1},
    )


def test_part_completion_advances_revision_and_is_idempotent(repository):
    """A converted part commit advances the revision; an identical repeat does not."""
    run, part = _run_with_part(repository)
    attempt = _raw_saved_attempt(repository, run, part)

    advanced, artifact = _complete_part(
        repository, run, part, attempt, expected_revision=run.revision
    )
    assert advanced.revision == run.revision + 1
    assert artifact.role == ARTIFACT_ROLE_CHUNK_AUDIO
    assert artifact.path == "chunks/chunk_01.mp3"
    assert repository._get_part(part.part_uuid).stage == PART_STAGE_COMPLETED
    assert repository.get_attempt(attempt.attempt_uuid).status == ATTEMPT_STATUS_COMPLETED

    repeated_run, repeated_artifact = _complete_part(
        repository, advanced, part, attempt, expected_revision=advanced.revision
    )
    assert repeated_run.revision == advanced.revision
    assert repeated_artifact.artifact_uuid == artifact.artifact_uuid


def test_part_completion_rejects_stale_revision_and_different_evidence(repository):
    """A stale revision and a conflicting artifact never advance the run."""
    run, part = _run_with_part(repository)
    attempt = _raw_saved_attempt(repository, run, part)
    advanced, _artifact = _complete_part(
        repository, run, part, attempt, expected_revision=run.revision
    )

    with pytest.raises(HistoryRevisionConflictError):
        _complete_part(repository, advanced, part, attempt, expected_revision=run.revision)

    with pytest.raises(HistoryPartCompletionConflictError):
        _complete_part(
            repository,
            advanced,
            part,
            attempt,
            expected_revision=advanced.revision,
            digest=DIGEST_B,
        )


def test_part_completion_requires_linked_raw(repository):
    """A submitting attempt cannot be completed before its raw bytes are linked."""
    run, part = _run_with_part(repository)
    attempt = _raw_saved_attempt(repository, run, part, status=ATTEMPT_STATUS_SUBMITTING)

    with pytest.raises(HistoryPartCompletionConflictError):
        _complete_part(repository, run, part, attempt, expected_revision=run.revision)
    assert repository.get_run(run.run_uuid).revision == run.revision


def test_run_completion_requires_every_part_completed(repository):
    """A run with an unfinished part cannot be closed."""
    run = repository.create_run(operation="tts", run_root="/tmp/native-completion-2")
    part_one = repository.add_part(
        run.run_uuid, position=1, prepared_text="one", voice="Rachel", fingerprint=DIGEST_A
    )
    part_two = repository.add_part(
        run.run_uuid, position=2, prepared_text="two", voice="Rachel", fingerprint=DIGEST_B
    )
    attempt = _raw_saved_attempt(repository, run, part_one)
    advanced, _artifact = _complete_part(
        repository, run, part_one, attempt, expected_revision=run.revision
    )

    with pytest.raises(HistoryRunCompletionConflictError):
        repository.record_tts_run_completed(
            run.run_uuid,
            expected_revision=advanced.revision,
            path="out.mp3",
            mime="audio/mpeg",
            size_bytes=8,
            sha256=DIGEST_B,
            media_metadata={"duration_ms": 2000},
        )
    assert repository.get_run(run.run_uuid).status == "running"
    assert repository.get_parts(run.run_uuid)[1].position == part_two.position


def test_run_completion_closes_run_and_is_idempotent(repository):
    """Closing a run sets its status and an identical repeat changes nothing."""
    run, part = _run_with_part(repository)
    attempt = _raw_saved_attempt(repository, run, part)
    advanced, _artifact = _complete_part(
        repository, run, part, attempt, expected_revision=run.revision
    )

    closed, final = repository.record_tts_run_completed(
        run.run_uuid,
        expected_revision=advanced.revision,
        path="native-completion.mp3",
        mime="audio/mpeg",
        size_bytes=8,
        sha256=DIGEST_B,
        media_metadata={"duration_ms": 1000},
    )
    assert closed.status == "completed"
    assert final.role == ARTIFACT_ROLE_FINAL_AUDIO

    repeated, repeated_final = repository.record_tts_run_completed(
        run.run_uuid,
        expected_revision=closed.revision,
        path="native-completion.mp3",
        mime="audio/mpeg",
        size_bytes=8,
        sha256=DIGEST_B,
        media_metadata={"duration_ms": 1000},
    )
    assert repeated.revision == closed.revision
    assert repeated_final.artifact_uuid == final.artifact_uuid
    assert repository.get_run("does-not-exist-or-uuid") is None


def test_run_completion_rejects_unknown_run(repository):
    """Completing a run that does not exist reports a missing run."""
    with pytest.raises(HistoryRunNotFoundError):
        repository.record_tts_run_completed(
            "8e2f1c8e-0000-4000-8000-000000000000",
            expected_revision=1,
            path="out.mp3",
            mime="audio/mpeg",
            size_bytes=1,
            sha256=DIGEST_A,
            media_metadata={},
        )
