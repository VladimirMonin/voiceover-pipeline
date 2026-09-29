"""Focused contract tests for ``services.execution`` paid-execution helpers.

These pin the paid-safety evidence the CLI relies on after the helpers moved out
of ``cli.py``: the accepted media id is persisted before any poll/download, the
observed cost joins the same marker, and the accepted raw bytes plus their
bounded receipt are written before conversion. They also pin that raw recovery
rebuilds the chunk from stored bytes alone with no provider call. All work is
offline: temp directories, synthetic values, and no network or provider seam.
"""

import argparse
import hashlib
import json
from types import SimpleNamespace

from conftest import fixture_path


def _run_dir(tmp_path, name):
    run_dir = tmp_path / "out" / name
    (run_dir / "chunks").mkdir(parents=True)
    return run_dir


def _state_for(run_dir, chunks, *, provider, model, voice, run_id):
    from voiceover_pipeline.run_state import initial_state

    return initial_state(
        provider=provider,
        model=model,
        voice=voice,
        script_path=fixture_path("smoke_test.md"),
        chunks=chunks,
        script_format="markdown",
        run_id=run_id,
    )


def test_bind_polza_media_attempt_persists_accepted_id_then_exact_cost(tmp_path):
    """The bound callbacks store the accepted id then the exact cost, in order.

    The accepted write happens inside the callback before any poll or download,
    so it lands on disk as soon as the id is known; the completed callback adds
    the reported cost to the same marker. A payload without a recognized cost
    must not erase the stored observation.
    """
    from voiceover_pipeline.run_state import GenerationLogger, begin_chunk_attempt
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution

    run_dir = _run_dir(tmp_path, "paid-media-bind")
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")
    chunk = chunks[0]
    state = _state_for(
        run_dir,
        chunks,
        provider="polza-tts",
        model="elevenlabs/text-to-speech-turbo-2-5",
        voice="Rachel",
        run_id="paid-media-bind",
    )
    begin_chunk_attempt(state, chunk_id=chunk.id, number=chunk.number)
    state_path = run_dir / "run_state.json"
    logger = GenerationLogger(run_dir / "generation.log")

    class FakePolzaProvider:
        pass

    provider = FakePolzaProvider()
    execution.bind_polza_media_attempt(
        provider, state=state, state_path=state_path, logger=logger, chunk=chunk
    )
    assert callable(provider.on_media_task_accepted)
    assert callable(provider.on_media_completed)

    # The accepted id is written to the state file before any poll/download.
    provider.on_media_task_accepted("task-1")
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["pending_attempt"]["remote_task_id"] == "task-1"

    # The completed poll stores the exact cost on the same marker. The history
    # boundary parses the JSON with ``parse_float=Decimal``, so an exact value
    # arrives as a string/Decimal rather than a binary float.
    provider.on_media_completed("task-1", {"cost_rub": "0.3"}, None)
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["remote_task_id"] == "task-1"
    assert marker["cost"] == 0.3
    assert marker["cost_exact"] == "0.3"

    # A later payload without a recognized cost cannot erase the observation.
    provider.on_media_completed("task-1", None, None)
    assert (
        json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]["cost_exact"] == "0.3"
    )

    log = (run_dir / "generation.log").read_text(encoding="utf-8")
    assert "remote_accepted" in log
    assert "cost_observed" in log
    # Accepted evidence precedes observed cost.
    assert log.index("remote_accepted") < log.index("cost_observed")


def test_persist_paid_raw_audio_writes_exact_receipt_and_direct_cost(tmp_path):
    """Persist accepted bytes before one atomic marker-and-direct-cost write.

    The raw file keeps the exact accepted bytes, the marker records only the
    bounded format, deterministic path, digest, and generation id, and the
    synchronous direct cost joins the same atomic state write. The raw file and
    state are not jointly atomic across a crash. No temp file survives.
    """
    from voiceover_pipeline.models import SynthesisResult
    from voiceover_pipeline.run_state import (
        GenerationLogger,
        begin_chunk_attempt,
        raw_audio_relative_path,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution

    run_dir = _run_dir(tmp_path, "paid-raw-receipt")
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")
    chunk = chunks[0]
    state = _state_for(
        run_dir,
        chunks,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        run_id="paid-raw-receipt",
    )
    begin_chunk_attempt(state, chunk_id=chunk.id, number=chunk.number)
    state_path = run_dir / "run_state.json"
    logger = GenerationLogger(run_dir / "generation.log")
    paths = SimpleNamespace(output_root=run_dir)
    result = SynthesisResult(
        audio_bytes=b"paid-mp3",
        audio_format="mp3",
        transcript=chunk.text,
        generation_id="gen-1",
        client_path="requests",
        raw_metadata={"usage_direct": {"cost_rub": "0.3"}},
    )

    execution.persist_paid_raw_audio(
        result, state=state, state_path=state_path, paths=paths, chunk=chunk, logger=logger
    )

    relative = raw_audio_relative_path(chunk.id, "mp3")
    raw_path = run_dir / relative
    assert raw_path.read_bytes() == b"paid-mp3"
    assert not raw_path.with_suffix(raw_path.suffix + ".tmp").exists()
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["raw"] == {
        "format": "mp3",
        "path": relative,
        "sha256": hashlib.sha256(b"paid-mp3").hexdigest(),
        "generation_id": "gen-1",
    }
    assert marker["cost"] == 0.3
    assert marker["cost_exact"] == "0.3"
    assert "raw_audio_saved" in (run_dir / "generation.log").read_text(encoding="utf-8")


def test_raw_recovery_result_reads_stored_bytes_without_provider(tmp_path):
    """Raw recovery rebuilds the chunk from stored bytes alone, with no provider.

    The helper takes no provider and performs a single disk read of the receipt
    path, so an already-paid chunk resumes from exact local bytes and the
    submitted chunk text/identity rather than any new provider payload.
    """
    from voiceover_pipeline.models import SynthesisResult
    from voiceover_pipeline.run_state import raw_audio_relative_path
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution

    run_root = tmp_path / "out" / "paid-raw-recovery"
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")
    chunk = chunks[0]
    relative = raw_audio_relative_path(chunk.id, "mp3")
    raw_path = run_root / relative
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"paid-mp3")
    args = argparse.Namespace(provider="polza-tts", model="openai/gpt-4o-mini-tts", voice="ash")
    recovery = {
        "kind": "raw",
        "raw_path": relative,
        "raw_format": "mp3",
        "generation_id": "gen-1",
    }

    result = execution.raw_recovery_result(args, chunk, run_root, recovery)

    assert isinstance(result, SynthesisResult)
    assert result.audio_bytes == b"paid-mp3"
    assert result.audio_format == "mp3"
    assert result.transcript == chunk.text
    assert result.generation_id == "gen-1"
    assert result.client_path == "requests"
    assert result.raw_metadata == {
        "voice": "ash",
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
    }
