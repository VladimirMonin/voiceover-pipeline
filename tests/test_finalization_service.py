"""Focused contract tests for ``services.finalization``.

These pin the post-loop finalization seam ``cli._generate_step`` delegates to
after the prepared-part loop: the run cost is merged into trusted state and
written before any artifact, the dialogue quality gate runs before the final
concat, and a concat failure leaves the error envelope and a non-completed run.

All work is offline: temp directories, synthetic chunk artifacts, and hook
callables that never touch a provider or the network. The real
``services.cost_enrichment`` merge and Decimal summary run against the synthetic
state so the cost path is exercised rather than stubbed.
"""

import argparse
import json
from datetime import datetime, timezone

import pytest
from conftest import fixture_path


def _paths(tmp_path):
    from voiceover_pipeline.artifacts import build_run_paths

    paths = build_run_paths(tmp_path / "out", "test-model", "finalize-run")
    paths.chunks_dir.mkdir(parents=True)
    return paths


def _chunks():
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    return split_markdown_by_delimiter(fixture_path("smoke_test.md"), "******")[:2]


def _args(tmp_path):
    return argparse.Namespace(
        provider="polza-tts",
        model="test-model",
        voice="ash",
        script=fixture_path("smoke_test.md"),
        format="markdown",
        json_output=True,
        with_timings=False,
    )


def _prepared(chunks):
    from voiceover_pipeline.services.prepare import prepare_run

    return prepare_run(_args(None), chunks, "style", "auto")


def _state_with_completed_chunks(chunks):
    return {
        "status": "running",
        "run_id": "finalize-run",
        "execution_source": "test-source",
        "chunks": [
            {"status": "completed", "id": chunk.id, "number": chunk.number} for chunk in chunks
        ],
        "errors": [],
    }


def _artifacts(chunks, *, cost="0.25"):
    from voiceover_pipeline.models import ChunkArtifact

    return {
        chunk.number: ChunkArtifact(
            number=chunk.number,
            id=chunk.id,
            file=f"{chunk.id}.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=(chunk.number - 1) * 1000,
            end_ms=chunk.number * 1000,
            text_characters=len(chunk.text),
            transcript=None,
            client_path="fake",
            generation_id=f"gen-{chunk.id}",
            cost=float(cost),
            cost_exact=cost,
            cost_currency="RUB",
        )
        for chunk in chunks
    }


def _fail(code):
    import voiceover_pipeline.cli as cli

    def _raise(message):
        raise cli.CliError(message, code)

    return _raise


def _hooks(*, calls, concat=None, verify=None):
    from voiceover_pipeline.services.finalization import FinalizationHooks

    def attach(_provider, _api_key, _model, _started, artifacts):
        calls["attach_costs"] = calls.get("attach_costs", 0) + 1
        return artifacts

    def concat_mp3(_ffmpeg, _chunks_dir, _output):
        calls["concat"] = calls.get("concat", 0) + 1
        if concat is not None:
            concat()

    return FinalizationHooks(
        attach_costs=attach,
        verify_dialogue_turns_before_concat=verify or (lambda *_a, **_k: None),
        concat_dialogue_turns=concat_mp3,
        concat_mp3_chunks=concat_mp3,
        mp3_duration_ms=lambda _ffprobe, _audio: 1000,
        extract_timings=lambda **_kwargs: None,
        fail_provider=_fail(30),
        fail_output=_fail(50),
        fail_missing_dep=_fail(10),
        fail_whisper=_fail(40),
    )


def _finalize(tmp_path, *, state, artifacts, chunks, paths, hooks, dialogue_run=False):
    from voiceover_pipeline.run_state import GenerationLogger
    from voiceover_pipeline.services import finalization

    state_path = paths.output_root / "run_state.json"
    logger = GenerationLogger(paths.output_root / "generation.log")
    return finalization.finalize_generation(
        args=_args(tmp_path),
        paths=paths,
        state=state,
        state_path=state_path,
        logger=logger,
        chunks=chunks,
        chunk_artifacts_by_number=artifacts,
        prepared=_prepared(chunks),
        pricing_snapshot=None,
        api_key="key",
        run_started_at=datetime.now(timezone.utc),
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        dialogue_run=dialogue_run,
        hooks=hooks,
    )


def test_finalize_generation_writes_artifacts_and_completed_state(tmp_path, monkeypatch):
    """One successful pass writes each artifact once and marks the run completed."""
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.services import finalization

    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    writes = []
    original_write_json = finalization.write_json

    def record_write(path, data):
        writes.append(path)
        return original_write_json(path, data)

    monkeypatch.setattr(finalization, "write_json", record_write)
    paths = _paths(tmp_path)
    chunks = _chunks()
    state = _state_with_completed_chunks(chunks)
    artifacts = _artifacts(chunks)
    calls: dict[str, int] = {}

    summary = _finalize(
        tmp_path,
        state=state,
        artifacts=artifacts,
        chunks=chunks,
        paths=paths,
        hooks=_hooks(calls=calls),
    )

    assert calls == {"attach_costs": 1, "concat": 1}
    assert writes == [paths.chunks_json, paths.run_json, paths.output_root / "manifest.json"]
    assert paths.chunks_json.exists()
    assert paths.run_json.exists()
    assert (paths.output_root / "manifest.json").exists()
    assert summary.files["chunks_json"] == str(paths.chunks_json)
    assert summary.duration_ms == 1000
    assert summary.segment_count is None
    assert summary.cost_total == 0.5
    assert summary.cost_currency == "RUB"

    persisted = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    assert persisted["full_mp3"] == str(paths.full_mp3)
    assert persisted["main_duration_ms"] == 1000
    assert persisted["chunks"][0]["cost_exact"] == "0.25"

    log = (paths.output_root / "generation.log").read_text(encoding="utf-8")
    assert "run_complete" in log
    assert "concat_complete" in log


def test_finalize_generation_quality_failure_prevents_concat_and_completion(tmp_path, monkeypatch):
    """The quality gate runs before concat and leaves no completed run."""
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    paths = _paths(tmp_path)
    chunks = _chunks()
    state = _state_with_completed_chunks(chunks)
    artifacts = _artifacts(chunks)
    calls: dict[str, int] = {}

    def fail_quality(*_args, **_kwargs):
        raise cli.CliError("quality gate failed", 60)

    hooks = _hooks(calls=calls, verify=fail_quality)

    with pytest.raises(cli.CliError, match="quality gate failed") as error:
        _finalize(
            tmp_path,
            state=state,
            artifacts=artifacts,
            chunks=chunks,
            paths=paths,
            hooks=hooks,
            dialogue_run=True,
        )

    assert error.value.code == 60
    assert "concat" not in calls
    assert not paths.run_json.exists()
    assert not (paths.output_root / "manifest.json").exists()
    # The trusted cost write precedes the quality gate even when it fails.
    persisted = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert persisted["chunks"][0]["cost_exact"] == "0.25"


def test_finalize_generation_concat_failure_preserves_state_error_envelope(tmp_path, monkeypatch):
    """A concat failure records the error and state before the output fail exit."""
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    paths = _paths(tmp_path)
    chunks = _chunks()
    state = _state_with_completed_chunks(chunks)
    artifacts = _artifacts(chunks)
    calls: dict[str, int] = {}

    def explode():
        raise RuntimeError("ffmpeg exploded")

    hooks = _hooks(calls=calls, concat=explode)

    with pytest.raises(cli.CliError, match="Failed to concat MP3 chunks") as error:
        _finalize(
            tmp_path,
            state=state,
            artifacts=artifacts,
            chunks=chunks,
            paths=paths,
            hooks=hooks,
        )

    assert error.value.code == 50
    persisted = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["errors"][0]["message"] == "ffmpeg exploded"
    assert not paths.run_json.exists()
    log = (paths.output_root / "generation.log").read_text(encoding="utf-8")
    assert "concat_failed" in log
    assert "run_complete" not in log
