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

import pytest
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


def _hooks(execution, **overrides):
    """Build the per-run hooks with safe, non-crashing defaults for one test part loop."""

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unexpected hook call")

    fields = {
        "synthesize_part": unexpected,
        "write_audio_as_mp3": unexpected,
        "trim_final_silence": unexpected,
        "mp3_duration_ms": unexpected,
        "fail_provider": unexpected,
        "fail_output": unexpected,
        "emit_json_event": lambda *_args, **_kwargs: None,
        "log_retry": lambda *_args, **_kwargs: None,
        "reject_unconfirmed_paid_resume": unexpected,
        "paid_submit_attempt_status": lambda _error: "failed",
        "public_projection": lambda _result: {},
        "progress": lambda _message: None,
    }
    fields.update(overrides)
    return execution.PartExecutionHooks(**fields)


def test_execute_prepared_parts_marks_then_submits_then_saves_in_order(tmp_path):
    """The moved part loop persists the paid marker before submit and saves after.

    The durable attempt marker is written to disk before the provider submit, the
    paid raw bytes are persisted before the FFmpeg write, and the completed entry
    replaces its own marker in one atomic state write. The loop drives the CLI's
    call-time seams (synthesis, media write, duration) rather than importing them.
    """
    import argparse
    from types import SimpleNamespace

    from voiceover_pipeline.models import SynthesisResult
    from voiceover_pipeline.retry import RetryPolicy
    from voiceover_pipeline.run_state import GenerationLogger, atomic_write_json
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution
    from voiceover_pipeline.services.prepare import prepare_run

    run_dir = _run_dir(tmp_path, "execute-order")
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")[:2]
    args = argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        json_output=True,
        no_trim=True,
    )
    prepared = prepare_run(args, chunks, None, "auto")
    state = _state_for(
        run_dir,
        chunks,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        run_id="execute-order",
    )
    state_path = run_dir / "run_state.json"
    atomic_write_json(state_path, state)
    logger = GenerationLogger(run_dir / "generation.log")
    paths = SimpleNamespace(output_root=run_dir, chunks_dir=run_dir / "chunks")
    order: list[tuple[str, str]] = []
    marker_before_submit: list[bool] = []

    class FakeProvider:
        def synthesize_chunk(self, text, chunk_id):
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            marker_before_submit.append(
                persisted.get("pending_attempt", {}).get("status") == "submitting"
            )
            order.append(("submit", chunk_id))
            return SynthesisResult(
                audio_bytes=b"paid-mp3",
                audio_format="mp3",
                transcript=text,
                generation_id=f"gen-{chunk_id}",
                client_path="fake",
                raw_metadata={"usage_direct": {"cost_rub": "0.3"}},
            )

    def synthesize_part(provider, part):
        return provider.synthesize_chunk(part.chunk.text, part.chunk.id)

    def write_hook(_ffmpeg, _audio, _fmt, path):
        # The paid raw bytes are already on disk before conversion starts.
        assert (run_dir / "raw" / f"{path.stem}.mp3").read_bytes() == b"paid-mp3"
        assert _audio == b"paid-mp3"
        order.append(("write", path.name))
        path.write_bytes(b"mp3")

    hooks = _hooks(
        execution,
        synthesize_part=synthesize_part,
        write_audio_as_mp3=write_hook,
        mp3_duration_ms=lambda *_args, **_kwargs: 1000,
    )
    loop_state = execution.PartLoopState(chunk_artifacts_by_number={}, total_duration_ms=0)

    execution.execute_prepared_parts(
        args=args,
        prepared=prepared,
        chunks=chunks,
        paths=paths,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        state=state,
        state_path=state_path,
        logger=logger,
        provider=FakeProvider(),
        paid_submit=True,
        retry_policy=RetryPolicy(attempts=1, delay_seconds=0, max_delay_seconds=0, enabled=False),
        recoverable_attempts={},
        dialogue_run=False,
        completed=set(),
        loop_state=loop_state,
        hooks=hooks,
    )

    assert marker_before_submit == [True, True]
    assert order == [
        ("submit", "chunk_01"),
        ("write", "chunk_01.mp3"),
        ("submit", "chunk_02"),
        ("write", "chunk_02.mp3"),
    ]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["completed_count"] == 2
    assert "pending_attempt" not in persisted
    assert [item["id"] for item in persisted["chunks"]] == ["chunk_01", "chunk_02"]
    assert sorted(loop_state.chunk_artifacts_by_number) == [1, 2]
    assert loop_state.total_duration_ms == 2000
    assert loop_state.chunk_artifacts_by_number[1].cost_exact == "0.3"
    assert (run_dir / "raw" / "chunk_01.mp3").read_bytes() == b"paid-mp3"


def test_execute_prepared_parts_rebuilds_recovered_raw_without_a_provider_submit(tmp_path):
    """A recoverable raw attempt is rebuilt locally with no synthesis submit.

    When the receipt for this exact chunk is known, the loop must read the stored
    paid bytes and reuse the stored exact cost instead of calling the synthesis
    seam, writing any marker, or sending a second provider request.
    """
    import argparse
    from types import SimpleNamespace

    from voiceover_pipeline.retry import RetryPolicy
    from voiceover_pipeline.run_state import (
        GenerationLogger,
        atomic_write_json,
        begin_chunk_attempt,
        raw_audio_relative_path,
        record_raw_audio_saved,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution
    from voiceover_pipeline.services.prepare import prepare_run

    run_dir = _run_dir(tmp_path, "execute-recovery")
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")[:1]
    chunk = chunks[0]
    args = argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        json_output=True,
        no_trim=True,
    )
    prepared = prepare_run(args, chunks, None, "auto")
    state = _state_for(
        run_dir,
        chunks,
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        run_id="execute-recovery",
    )
    relative = raw_audio_relative_path(chunk.id, "mp3")
    (run_dir / relative).parent.mkdir(parents=True)
    (run_dir / relative).write_bytes(b"recovered-audio")
    # A recoverable raw attempt by definition still holds its persisted marker from
    # the original submit; the loop must rebuild from the bytes without a new POST.
    begin_chunk_attempt(state, chunk_id=chunk.id, number=chunk.number)
    record_raw_audio_saved(
        state,
        chunk_id=chunk.id,
        number=chunk.number,
        audio_format="mp3",
        relative_path=relative,
        sha256=hashlib.sha256(b"recovered-audio").hexdigest(),
        generation_id="gen-1",
    )
    state_path = run_dir / "run_state.json"
    atomic_write_json(state_path, state)
    logger = GenerationLogger(run_dir / "generation.log")
    paths = SimpleNamespace(output_root=run_dir, chunks_dir=run_dir / "chunks")
    recovery = {
        "kind": "raw",
        "raw_path": relative,
        "raw_format": "mp3",
        "generation_id": "gen-1",
        "cost": 0.3,
        "cost_exact": "0.3",
    }

    class ForbiddenProvider:
        def synthesize_chunk(self, *_args, **_kwargs):
            raise AssertionError("a recovered raw attempt must not reach synthesis")

        def recover_media_task(self, *_args, **_kwargs):
            raise AssertionError("a recovered raw attempt must not poll the provider")

    def forbidden_synthesis(**_kwargs):
        raise AssertionError("a recovered raw attempt must not call the synthesis seam")

    written: list[str] = []

    def write_hook(_ffmpeg, _audio, _fmt, path):
        assert _audio == b"recovered-audio"
        written.append(path.name)
        path.write_bytes(b"mp3")

    hooks = _hooks(
        execution,
        synthesize_part=forbidden_synthesis,
        write_audio_as_mp3=write_hook,
        mp3_duration_ms=lambda *_args, **_kwargs: 1000,
    )
    loop_state = execution.PartLoopState(chunk_artifacts_by_number={}, total_duration_ms=0)

    execution.execute_prepared_parts(
        args=args,
        prepared=prepared,
        chunks=chunks,
        paths=paths,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        state=state,
        state_path=state_path,
        logger=logger,
        provider=ForbiddenProvider(),
        paid_submit=True,
        retry_policy=RetryPolicy(attempts=1, delay_seconds=0, max_delay_seconds=0, enabled=False),
        recoverable_attempts={chunk.number: recovery},
        dialogue_run=False,
        completed=set(),
        loop_state=loop_state,
        hooks=hooks,
    )

    assert written == [f"{chunk.id}.mp3"]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["completed_count"] == 1
    assert "pending_attempt" not in persisted
    assert loop_state.chunk_artifacts_by_number[chunk.number].cost == 0.3
    assert loop_state.chunk_artifacts_by_number[chunk.number].cost_exact == "0.3"
    log_text = (run_dir / "generation.log").read_text(encoding="utf-8")
    assert "paid_raw_recovery" in log_text


def _generation_setup_hooks(execution, **overrides):
    """Build the generation-setup hooks with safe defaults for one focused call."""

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unexpected generation-setup hook call")

    fields = {
        "voice_identity": lambda _args: None,
        "dialogue_synthesis_identity": lambda *_args, **_kwargs: None,
        "recover_existing_chunks": unexpected,
        "reject_unconfirmed_paid_resume": unexpected,
        "fail_provider": unexpected,
    }
    fields.update(overrides)
    return execution.GenerationSetupHooks(**fields)


def test_prepare_generation_state_keeps_identity_and_paid_policy_in_order(tmp_path):
    """A fresh run resolves identity, prepared parts, and paid retry policy before the loop.

    The setup seam computes the voice identity through the CLI callable, wraps the
    exact chunks into prepared parts in order, and derives the one-attempt disabled
    retry policy for a paid provider while a local provider keeps its retry budget.
    """
    from voiceover_pipeline.run_state import GenerationLogger
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution

    run_dir = _run_dir(tmp_path, "prepare-fresh")
    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")[:2]
    args = argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        script=fixture_path("smoke_test.md"),
        format="markdown",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        resume=False,
    )
    state_path = run_dir / "run_state.json"
    logger = GenerationLogger(run_dir / "generation.log")
    identity_calls: list[argparse.Namespace] = []

    setup = execution.prepare_generation_state(
        args,
        chunks=chunks,
        style_prompt="calm",
        prompt_mode="auto",
        paths=SimpleNamespace(
            output_root=run_dir, chunks_dir=run_dir / "chunks", prefix="prepare-fresh"
        ),
        ffprobe_path="ffprobe",
        state_path=state_path,
        logger=logger,
        hooks=_generation_setup_hooks(
            execution,
            voice_identity=lambda received: identity_calls.append(received) or "voice-identity",
        ),
    )

    assert identity_calls == [args]
    assert setup.dialogue_run is False
    assert setup.paid_submit is True
    assert setup.retry_policy.attempts == 1
    assert setup.retry_policy.enabled is False
    assert setup.completed == set()
    assert setup.recoverable_attempts == {}
    assert setup.loop_state.chunk_artifacts_by_number == {}
    assert setup.loop_state.total_duration_ms == 0
    assert [part.chunk for part in setup.prepared.parts] == chunks
    assert [part.voice for part in setup.prepared.parts] == [chunk.voice for chunk in chunks]
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["voice_identity"] == "voice-identity"
    assert persisted["completed_count"] == 0
    assert setup.state["voice_identity"] == "voice-identity"

    # A non-paid local provider keeps the requested retry budget.
    local_dir = _run_dir(tmp_path, "prepare-local")
    args.provider = "qwen-local"
    local_setup = execution.prepare_generation_state(
        args,
        chunks=chunks[:1],
        style_prompt=None,
        prompt_mode="auto",
        paths=SimpleNamespace(
            output_root=local_dir, chunks_dir=local_dir / "chunks", prefix="prepare-local"
        ),
        ffprobe_path="ffprobe",
        state_path=local_dir / "run_state.json",
        logger=GenerationLogger(local_dir / "generation.log"),
        hooks=_generation_setup_hooks(execution),
    )

    assert local_setup.paid_submit is False
    assert local_setup.retry_policy.attempts == 3
    assert local_setup.retry_policy.enabled is True


def test_prepare_generation_state_blocks_unconfirmed_resume_and_orphan_dialogue(tmp_path):
    """Setup refuses an unconfirmed paid resume and orphan dialogue before any state write.

    A stored unconfirmed paid marker reaches the CLI reject hook with the bounded
    marker and logger and never rewrites run state; a dialogue resume with orphan
    audio and no trusted state reaches the provider-failure hook before any run
    state is written.
    """
    from voiceover_pipeline.run_state import (
        GenerationLogger,
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import execution

    class Rejected(Exception):
        pass

    chunks = split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")[:1]
    run_dir = _run_dir(tmp_path, "prepare-unconfirmed")
    args = argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        script=fixture_path("smoke_test.md"),
        format="markdown",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        resume=True,
    )
    state = initial_state(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        script_path=fixture_path("smoke_test.md"),
        chunks=chunks,
        script_format="markdown",
        run_id="prepare-unconfirmed",
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    state_path = run_dir / "run_state.json"
    atomic_write_json(state_path, state)
    before = state_path.read_text(encoding="utf-8")
    rejected: list[tuple[dict, object]] = []

    def reject(marker, logger):
        rejected.append((marker, logger))
        raise Rejected

    def unexpected(*_args, **_kwargs):
        raise AssertionError("setup must not continue after a blocked resume")

    with pytest.raises(Rejected):
        execution.prepare_generation_state(
            args,
            chunks=chunks,
            style_prompt=None,
            prompt_mode="auto",
            paths=SimpleNamespace(
                output_root=run_dir, chunks_dir=run_dir / "chunks", prefix="prepare-unconfirmed"
            ),
            ffprobe_path="ffprobe",
            state_path=state_path,
            logger=GenerationLogger(run_dir / "generation.log"),
            hooks=_generation_setup_hooks(
                execution,
                reject_unconfirmed_paid_resume=reject,
                recover_existing_chunks=unexpected,
                fail_provider=unexpected,
            ),
        )

    assert len(rejected) == 1
    marker, _logger = rejected[0]
    assert marker["id"] == chunks[0].id
    assert marker["status"] == "submitting"
    assert state_path.read_text(encoding="utf-8") == before

    # A dialogue resume with orphan audio and no trusted state fails through the
    # provider-failure hook before any run state is written.
    orphan_dir = _run_dir(tmp_path, "prepare-orphan")
    (orphan_dir / "chunks" / "turn_0001.mp3").write_bytes(b"orphan")
    orphan_args = argparse.Namespace(
        provider="openrouter-tts",
        model="google/gemini-3.1-flash-tts-preview",
        voice="Kore",
        script=fixture_path("smoke_test.md"),
        format="dialogue",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        resume=True,
    )
    failures: list[str] = []

    def fail_provider(message):
        failures.append(message)
        raise Rejected

    with pytest.raises(Rejected):
        execution.prepare_generation_state(
            orphan_args,
            chunks=chunks,
            style_prompt=None,
            prompt_mode="auto",
            paths=SimpleNamespace(
                output_root=orphan_dir, chunks_dir=orphan_dir / "chunks", prefix="prepare-orphan"
            ),
            ffprobe_path="ffprobe",
            state_path=orphan_dir / "run_state.json",
            logger=GenerationLogger(orphan_dir / "generation.log"),
            hooks=_generation_setup_hooks(
                execution,
                dialogue_synthesis_identity=lambda *_args, **_kwargs: "dialogue-identity",
                fail_provider=fail_provider,
            ),
        )

    assert len(failures) == 1
    assert "orphan dialogue audio" in failures[0]
    assert not (orphan_dir / "run_state.json").exists()
