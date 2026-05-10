import argparse
import builtins
import json

import pytest

from conftest import cli_json, fixture_path


class FakeProvider:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def synthesize_chunk(self, text, chunk_id):
        from voiceover_pipeline.models import SynthesisResult

        self.calls.append(chunk_id)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("HTTP 503: temporary server error")
        return SynthesisResult(
            audio_bytes=b"audio",
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="fake",
        )


def make_args(tmp_path, run_id="stable-run", resume=False):
    return argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        script=fixture_path("smoke_test.md"),
        output_dir=tmp_path / "out",
        run_id=run_id,
        format="markdown",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        no_trim=True,
        json_output=True,
        json_events=False,
        resume=resume,
        with_timings=False,
    )


def patch_generation_io(monkeypatch):
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(cli, "write_audio_as_mp3", lambda _ffmpeg, _audio, _fmt, path: path.write_bytes(b"mp3"))
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda *_args, **_kwargs: 1000)
    monkeypatch.setattr(cli, "concat_mp3_chunks", lambda _ffmpeg, _chunks_dir, output_path: output_path.write_bytes(b"full"))
    monkeypatch.setattr(cli, "attach_costs", lambda _provider, _api_key, _model, _started, chunks: chunks)


def test_generate_step_writes_state_and_log_after_each_chunk(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto")

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["completed_count"] == 2
    assert [item["id"] for item in state["chunks"]] == ["chunk_01", "chunk_02"]
    assert "gen-chunk_01" == state["chunks"][0]["generation_id"]
    log_text = (paths.output_root / "generation.log").read_text(encoding="utf-8")
    assert "chunk_started" in log_text
    assert "chunk_state_saved" in log_text


def test_generate_step_retries_retryable_provider_errors(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.retry as retry
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(retry.time, "sleep", lambda _delay: None)
    args = make_args(tmp_path)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = FakeProvider(failures=1)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto")

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_01", "chunk_01"]


def test_resume_does_not_regenerate_completed_chunks(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import atomic_write_json, initial_state, upsert_completed_chunk
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]
    (paths.chunks_dir / "chunk_01.mp3").write_bytes(b"existing")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    provider = FakeProvider()

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto")

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_02"]


def test_overwrite_refuses_to_delete_existing_paid_chunks_without_confirmation(tmp_path):
    run_dir = tmp_path / "out" / "paid-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_01.mp3").write_bytes(b"paid")

    code, data = cli_json(
        "generate",
        "--provider", "polza-chat-audio",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "paid-run",
        "--overwrite",
        "--json",
    )

    assert code == 30
    assert "confirm-delete-paid-audio" in data["error"]
    assert (chunks_dir / "chunk_01.mp3").exists()


def test_dry_run_cost_with_limit_chunks_makes_no_tts_request(tmp_path):
    code, data = cli_json(
        "generate",
        "--provider", "polza-tts",
        "--model", "openai/gpt-4o-mini-tts",
        "--voice", "ash",
        "--script", str(fixture_path("smoke_test.md")),
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "dry-run",
        "--dry-run-cost",
        "--limit-chunks", "1",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["chunks"] == 1
    assert data["original_chunks"] == 2
    assert not (tmp_path / "out" / "dry-run").exists()


def test_status_reports_partial_run_12_of_105(tmp_path):
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "partial-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    for number in range(1, 13):
        chunk_id = f"chunk_{number:02d}"
        (chunks_dir / f"{chunk_id}.mp3").write_bytes(b"mp3")
        chunks.append({"status": "completed", "number": number, "id": chunk_id, "file": f"{chunk_id}.mp3"})
    atomic_write_json(run_dir / "run_state.json", {"chunk_count": 105, "completed_count": 12, "chunks": chunks, "errors": []})

    code, data = cli_json(
        "status",
        "--output-dir", str(tmp_path / "out"),
        "--run-id", "partial-run",
        "--json",
    )

    assert code == 0
    assert data["total_chunks"] == 105
    assert data["completed_chunks"] == 12
    assert data["next_chunk"] == 13
    assert data["can_resume"] is True


def test_concat_writes_partial_ogg_name(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "partial-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    for number in range(1, 13):
        (chunks_dir / f"chunk_{number:02d}.mp3").write_bytes(b"mp3")
    atomic_write_json(run_dir / "run_state.json", {"chunk_count": 105, "completed_count": 12, "chunks": []})
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "concat_audio_files", lambda _ffmpeg, _files, output_path: output_path.write_bytes(b"ogg"))

    args = argparse.Namespace(output_dir=tmp_path / "out", run_id="partial-run", format="ogg", json_output=False)
    cli.concat_cmd(args)

    assert (run_dir / "partial-12-of-105.ogg").read_bytes() == b"ogg"


def test_whisper_install_message_uses_extra(monkeypatch):
    import voiceover_pipeline.cli as cli

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(cli.CliError) as error:
        cli._preflight_timing_dependency()

    assert "uv sync --extra timing-whisper" in str(error.value)
