import argparse
import builtins
import json

import pytest
from conftest import cli_json, fixture_path


class FakeProvider:
    def __init__(self, failures=0, raw_metadata=None):
        self.failures = failures
        self.calls = []
        self.raw_metadata = raw_metadata or {}

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
            raw_metadata=self.raw_metadata,
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

    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda _ffmpeg, _audio, _fmt, path: path.write_bytes(b"mp3")
    )
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda *_args, **_kwargs: 1000)
    monkeypatch.setattr(
        cli,
        "concat_mp3_chunks",
        lambda _ffmpeg, _chunks_dir, output_path: output_path.write_bytes(b"full"),
    )
    monkeypatch.setattr(
        cli, "attach_costs", lambda _provider, _api_key, _model, _started, chunks: chunks
    )


def build_voice_bank(tmp_path):
    import hashlib
    import json
    import wave

    bank_root = tmp_path / "bank"
    (bank_root / "voices").mkdir(parents=True)
    reference = bank_root / "voices" / "main.wav"
    with wave.open(str(reference), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * 4000)
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    catalog_path = bank_root / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_voice": "main",
                "voices": [
                    {
                        "id": "main",
                        "display_name": "Main Narrator",
                        "description": "",
                        "language": "ru",
                        "reference_audio": "voices/main.wav",
                        "reference_text": "Эталонная фраза.",
                        "reference_sha256": digest,
                        "origin": {"mode": "owner-reference", "instruction": None, "seed": 7},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


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
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["completed_count"] == 2
    assert [item["id"] for item in state["chunks"]] == ["chunk_01", "chunk_02"]
    assert "gen-chunk_01" == state["chunks"][0]["generation_id"]
    log_text = (paths.output_root / "generation.log").read_text(encoding="utf-8")
    assert "chunk_started" in log_text
    assert "chunk_state_saved" in log_text


def test_generate_step_persists_the_public_omnivoice_runtime_receipt(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    receipt = {
        "model_id": "audio-cpp/omnivoice-q8_0",
        "sha256": "2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b",
        "quantization": "Q8_0 GGUF",
        "license": "CC-BY-NC-4.0 upstream weights; local noncommercial research only",
        "provenance": "audio-cpp/audio.cpp-gguf@fixture; converted from OmniVoice",
    }
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    args = make_args(tmp_path, run_id="omnivoice-receipt")
    args.provider = "omnivoice-local"
    args.model = "audio-cpp/omnivoice-q8_0"
    args.voice = "built-in-female-style-condition"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args,
            FakeProvider(
                raw_metadata={
                    "runtime_receipt": receipt,
                    "voice_selection": {
                        "kind": "built-in-style-condition",
                        "condition": "female",
                        "named_preset": False,
                        "voice_cloning": False,
                        "voice_design": False,
                    },
                    "voice_session": {
                        "strategy": "single-native-invocation-internal-text-chunking",
                        "seed": 1234,
                        "internal_text_chunk_size": 420,
                    },
                }
            ),
            "ffmpeg",
            "ffprobe",
            chunks,
            "",
            None,
            paths,
            None,
            "none",
        )

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(paths.run_json.read_text(encoding="utf-8"))
    assert state["chunks"][0]["runtime_receipt"] == receipt
    assert run_manifest["chunks"][0]["runtime_receipt"] == receipt
    assert state["chunks"][0]["voice_selection"]["condition"] == "female"
    assert state["chunks"][0]["voice_session"] == {
        "strategy": "single-native-invocation-internal-text-chunking",
        "seed": 1234,
        "internal_text_chunk_size": 420,
    }
    assert run_manifest["chunks"][0]["voice_selection"] == state["chunks"][0]["voice_selection"]
    assert run_manifest["chunks"][0]["voice_session"] == state["chunks"][0]["voice_session"]
    assert str(tmp_path) not in json.dumps(run_manifest["chunks"][0]["runtime_receipt"])


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
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_01", "chunk_01"]


def test_resume_does_not_regenerate_completed_chunks(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        initial_state,
        upsert_completed_chunk,
    )
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
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_02"]


def test_clone_voice_identity_is_deterministic_across_processes(tmp_path, monkeypatch):
    import subprocess
    import sys

    import voiceover_pipeline.cli as cli

    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"not-a-real-wav-but-fine")
    args = argparse.Namespace(
        provider="omnivoice-local",
        mode="clone",
        reference_audio=str(reference),
        reference_text="Всем привет. Это образец голоса.",
    )
    expected = cli._omnivoice_voice_identity(args)
    assert expected is not None
    assert "hash(" not in expected and "0x" not in expected

    script = (
        "import sys, types\n"
        "import voiceover_pipeline.cli as cli\n"
        "ref = r'%s'\n"
        "args = types.SimpleNamespace(provider='omnivoice-local', mode='clone',"
        " reference_audio=ref, reference_text='Всем привет. Это образец голоса.')\n"
        "print(cli._omnivoice_voice_identity(args))\n" % (str(reference),)
    )
    for seed in ("1", "987654"):
        env = dict(monkeypatch.__dict__.get("env", {}))
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected


def test_resume_rejects_changed_voice_identity(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json, initial_state
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="voice-identity", resume=True)
    args.provider = "omnivoice-local"
    args.model = "audio-cpp/omnivoice-q8_0"
    args.voice = "main"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
        voice_identity="preset:main:" + "1" * 64,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)

    args.voice_bank_catalog = object()
    args.voice_bank_profile = type(
        "Profile",
        (),
        {"id": "main", "reference_sha256": "2" * 64},
    )()

    with pytest.raises(cli.CliError, match="voice identity changed") as error:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "none"
        )

    assert error.value.code == 30


def test_overwrite_refuses_to_delete_existing_paid_chunks_without_confirmation(tmp_path):
    run_dir = tmp_path / "out" / "paid-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_01.mp3").write_bytes(b"paid")

    code, data = cli_json(
        "generate",
        "--provider",
        "polza-chat-audio",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "paid-run",
        "--overwrite",
        "--json",
    )

    assert code == 30
    assert "confirm-delete-paid-audio" in data["error"]
    assert (chunks_dir / "chunk_01.mp3").exists()


def test_dry_run_cost_with_limit_chunks_makes_no_tts_request(tmp_path):
    code, data = cli_json(
        "generate",
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "dry-run",
        "--dry-run-cost",
        "--limit-chunks",
        "1",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["chunks"] == 1
    assert data["original_chunks"] == 2
    assert not (tmp_path / "out" / "dry-run").exists()


def test_local_tts_dry_run_reports_sentence_packed_inference_chunks(tmp_path):
    script = tmp_path / "local-report.md"
    script.write_text(
        " ".join(
            f"Предложение номер словами содержит достаточно полезного текста {word}."
            for word in (
                "первое",
                "второе",
                "третье",
                "четвёртое",
                "пятое",
                "шестое",
                "седьмое",
                "восьмое",
            )
        ),
        encoding="utf-8",
    )

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "local-dry-run",
        "--voice-bank",
        str(build_voice_bank(tmp_path)),
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 1
    assert data["original_chunks"] == data["chunks"]
    assert not (tmp_path / "out" / "local-dry-run").exists()


def test_local_tts_cli_rejects_raw_digits_before_runtime(tmp_path):
    script = tmp_path / "digits.md"
    script.write_text("Версия 3.5 готова на 25 процентов.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "digits",
        "--voice-bank",
        str(build_voice_bank(tmp_path)),
        "--dry-run-cost",
        "--json",
    )

    assert code == 2
    assert "raw digits" in data["error"]
    assert not (tmp_path / "out" / "digits").exists()


def test_omnivoice_cli_rejects_style_controls_before_dry_run(tmp_path):
    script = tmp_path / "style.md"
    script.write_text("Проверка готового автоматического голоса.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--style-prompt",
        "тёплый голос",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "style",
        "--dry-run-cost",
        "--json",
    )

    assert code == 2
    assert "style controls" in data["error"]
    assert not (tmp_path / "out" / "style").exists()


def test_unprofiled_qwen_cli_dry_run_does_not_apply_omnivoice_digit_policy(tmp_path):
    script = tmp_path / "qwen-digits.md"
    script.write_text("Версия 3.5 готова на 25 процентов.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "qwen-local",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "qwen-digits",
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["chunks"] == 1
    assert not (tmp_path / "out" / "qwen-digits").exists()


def test_status_reports_partial_run_12_of_105(tmp_path):
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "partial-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    for number in range(1, 13):
        chunk_id = f"chunk_{number:02d}"
        (chunks_dir / f"{chunk_id}.mp3").write_bytes(b"mp3")
        chunks.append(
            {"status": "completed", "number": number, "id": chunk_id, "file": f"{chunk_id}.mp3"}
        )
    atomic_write_json(
        run_dir / "run_state.json",
        {"chunk_count": 105, "completed_count": 12, "chunks": chunks, "errors": []},
    )

    code, data = cli_json(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "partial-run",
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
    atomic_write_json(
        run_dir / "run_state.json", {"chunk_count": 105, "completed_count": 12, "chunks": []}
    )
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        cli,
        "concat_audio_files",
        lambda _ffmpeg, _files, output_path: output_path.write_bytes(b"ogg"),
    )

    args = argparse.Namespace(
        output_dir=tmp_path / "out", run_id="partial-run", format="ogg", json_output=False
    )
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


def test_timing_artifact_writer_keeps_legacy_file_names_and_manifest_shape(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.models import TimingResult, TimingSegment

    audio = tmp_path / "fixture.mp3"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _audio: 1000)
    timing = TimingResult(
        segments=[
            TimingSegment(
                id=1,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text="fixture",
            )
        ],
        model="small",
        backend="faster-whisper",
    )

    result = cli._write_timing_artifacts(audio, tmp_path, "legacy", timing)

    manifest = json.loads((tmp_path / "legacy.timings.json").read_text(encoding="utf-8"))
    assert result == {"segment_count": 1, "total_duration_ms": 1000}
    assert "provider" not in manifest
    assert (tmp_path / "legacy.srt").read_text(
        encoding="utf-8"
    ) == "1\n00:00:00,000 --> 00:00:01,000\nfixture\n"
