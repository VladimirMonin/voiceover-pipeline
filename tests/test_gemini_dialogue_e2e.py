import hashlib
import json
import sys

import pytest


def _write_dialogue_script(tmp_path):
    script = tmp_path / "podcast.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: gemini-dialogue",
                "language: ru",
                "model: google/gemini-3.1-flash-tts-preview",
                "speakers:",
                "  Host:",
                "    display_name: Ведущая",
                "    voice: Kore",
                "    profile: warm host",
                "  Guest:",
                "    display_name: Гость",
                "    voice: Puck",
                "    profile: calm expert",
                "vibe: >",
                "  Russian technical podcast. Natural question-and-answer conversation.",
                "allowed_tags:",
                "  - warmly",
                "  - curious",
                "max_chunk_bytes: 3500",
                "---",
                "Host: [warmly] Что умеет утилита?",
                "Guest: Она создаёт озвучку и субтитры.",
                "******",
                "Host: [curious] Можно работать локально?",
                "Guest: Да, для одноголосой озвучки есть OmniVoice.",
            ]
        ),
        encoding="utf-8",
    )
    return script


def _patch_generation_io(monkeypatch):
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda _ffmpeg, _audio, _fmt, path: path.write_bytes(b"mp3")
    )
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda *_args, **_kwargs: 1000)
    monkeypatch.setattr(
        cli,
        "concat_dialogue_turns",
        lambda _ffmpeg, _chunks_dir, output_path: output_path.write_bytes(b"full"),
    )
    monkeypatch.setattr(cli, "attach_costs", lambda *args, **kwargs: args[-1])
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda _provider, _api_key, _model: None)
    monkeypatch.setattr(cli, "_preflight_tts_quality_provider", lambda _args: None)
    monkeypatch.setattr(
        cli,
        "_verify_dialogue_turns_before_concat",
        lambda *_args, **_kwargs: {
            "artifact_type": "voiceover-dialogue-tts-quality-receipt",
            "status": "success",
            "passed": True,
            "turns": [],
            "human_listening_required": True,
        },
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)


def _write_dialogue_raw_resume_fixture(script, argv, *, state_voice):
    """Seed a one-turn dialogue run whose paid attempt saved raw audio pre-FFmpeg.

    The state mirrors exactly what ``generate --limit-chunks 1`` would persist with
    no ``--voice``: the cast voice, script hash, and synthesis identity all come
    from the helpers the CLI itself uses, so only the resume voice selection under
    test can differ. ``state_voice`` overrides just the stored ``run_state.json``
    voice to model a foreign or older run.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        raw_audio_relative_path,
        record_raw_audio_saved,
    )
    from voiceover_pipeline.tts_prompting import resolve_prompt_mode

    args = cli.build_parser().parse_args(argv[1:])
    args.format = cli._resolve_script_format(args.script, args.format)
    args.provider = args.provider or "openrouter-tts"
    cli._resolve_model(args)
    gemini_report = cli.validate_gemini_dialogue_file(
        args.script,
        delimiter=args.delimiter,
        model=args.model,
        speaker_voice_overrides=args.speaker_voice,
        agent=True,
        provider=args.provider,
        allowed_voices=None,
    )
    args.speaker_voice_map = gemini_report["speaker_voice_map"]
    chunks = cli.dialogue_turns_from_validation(gemini_report)[: args.limit_chunks]
    synthesis_identity = cli._dialogue_synthesis_identity(
        args, None, resolve_prompt_mode(args.provider, args.model), chunks
    )

    paths = build_run_paths(args.output_dir, args.model, args.run_id or None)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=state_voice,
        script_path=args.script,
        chunks=chunks,
        script_format="dialogue",
        run_id=paths.prefix,
        limited_to_chunks=args.limit_chunks,
        synthesis_identity=synthesis_identity,
    )
    chunk = chunks[0]
    begin_chunk_attempt(state, chunk_id=chunk.id, number=chunk.number)
    raw_relative = raw_audio_relative_path(chunk.id, "pcm16")
    raw_path = paths.output_root / raw_relative
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = b"paid-gemini-pcm-bytes"
    raw_path.write_bytes(raw_bytes)
    record_raw_audio_saved(
        state,
        chunk_id=chunk.id,
        number=chunk.number,
        audio_format="pcm16",
        relative_path=raw_relative,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        generation_id="gen-resume-1",
    )
    atomic_write_json(paths.output_root / cli.STATE_FILE, state)
    return paths, chunk.id, raw_path


def _dialogue_resume_argv(tmp_path, script):
    return [
        "voiceover",
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "resume-dialogue",
        "--limit-chunks",
        "1",
        "--resume",
        "--tts-quality-provider",
        "fixture-asr",
        "--json",
    ]


def test_openrouter_dialogue_resume_raw_recovery_uses_cast_voice_not_default(
    tmp_path, monkeypatch, capsys
):
    from unittest.mock import MagicMock

    import voiceover_pipeline.cli as cli

    class ForbiddenProvider:
        def synthesize_chunk(self, *_args, **_kwargs):
            pytest.fail("saved raw must not trigger a second paid synthesis")

        def recover_media_task(self, *_args, **_kwargs):
            pytest.fail("saved raw must not trigger provider GET recovery")

    script = _write_dialogue_script(tmp_path)
    argv = _dialogue_resume_argv(tmp_path, script)
    paths, chunk_id, raw_path = _write_dialogue_raw_resume_fixture(script, argv, state_voice="Kore")

    mock_post = MagicMock(side_effect=AssertionError("resume must not send a paid request"))
    _patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg-fixture", "ffprobe-fixture"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-or-v1-test-only-placeholder")
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: ForbiddenProvider())
    monkeypatch.setattr("voiceover_pipeline.providers.openrouter_tts.requests.post", mock_post)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert data["provider"] == "openrouter-tts"
    assert captured.err == ""
    assert mock_post.call_count == 0

    run_state = json.loads((paths.output_root / cli.STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["voice"] == "Kore"
    assert "pending_attempt" not in run_state
    assert [entry["status"] for entry in run_state["chunks"]] == ["completed"]
    assert run_state["chunks"][0]["id"] == chunk_id
    assert run_state["chunks"][0]["voice"] == "Kore"
    assert (paths.chunks_dir / f"{chunk_id}.mp3").exists()
    assert raw_path.exists()


def test_openrouter_dialogue_resume_rejects_mismatched_saved_voice_early(
    tmp_path, monkeypatch, capsys
):
    import voiceover_pipeline.cli as cli

    script = _write_dialogue_script(tmp_path)
    argv = _dialogue_resume_argv(tmp_path, script)
    paths, _chunk_id, raw_path = _write_dialogue_raw_resume_fixture(
        script, argv, state_voice="Puck"
    )
    state_before = (paths.output_root / cli.STATE_FILE).read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mismatched saved voice must block before key/provider/pricing work")

    _patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg-fixture", "ffprobe-fixture"))
    monkeypatch.setattr(cli, "read_api_key", forbidden)
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", forbidden)
    monkeypatch.setattr(cli, "build_provider", forbidden)
    monkeypatch.setattr("voiceover_pipeline.providers.openrouter_tts.requests.post", forbidden)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "error"
    assert data["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert data["details"]["chunk_id"] == _chunk_id
    assert (paths.output_root / cli.STATE_FILE).read_bytes() == state_before
    assert raw_path.exists()
    assert not (paths.chunks_dir / f"{_chunk_id}.mp3").exists()


def test_gemini_dialogue_e2e_mocked_generation(tmp_path, monkeypatch, capsys):
    from unittest.mock import MagicMock, patch

    import voiceover_pipeline.cli as cli

    script = _write_dialogue_script(tmp_path)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake-audio-gemini"
    mock_response.headers = {"X-Generation-Id": "gen-e2e-1"}
    _patch_generation_io(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-only-placeholder")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover",
            "generate",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "e2e-dialogue",
            "--tts-quality-provider",
            "fixture-asr",
            "--json",
        ],
    )

    with patch(
        "voiceover_pipeline.providers.openrouter_tts.requests.post",
        return_value=mock_response,
    ) as mock_post:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert data["provider"] == "openrouter-tts"
    assert data["run_id"] == "e2e-dialogue"
    json_lines = [line for line in captured.out.splitlines() if line.strip().startswith("{")]
    assert len(json_lines) == 1
    assert captured.err.strip() == ""

    assert mock_post.call_count == 4
    bodies = [call[1]["json"] for call in mock_post.call_args_list]
    for body, voice, text in zip(
        bodies,
        ["Kore", "Puck", "Kore", "Puck"],
        [
            "[warmly] Что умеет утилита?",
            "Она создаёт озвучку и субтитры.",
            "[curious] Можно работать локально?",
            "Да, для одноголосой озвучки есть OmniVoice.",
        ],
    ):
        assert body["model"] == "google/gemini-3.1-flash-tts-preview"
        assert body["voice"] == voice
        assert body["input"] == text
        assert set(body) == {"model", "input", "voice", "response_format"}
        assert body["response_format"] == "pcm"
        assert "multi_speaker_voice_config" not in body

    run_dir = tmp_path / "out" / "e2e-dialogue"
    chunks_manifest = json.loads((run_dir / "chunks" / "chunks.json").read_text(encoding="utf-8"))
    assert chunks_manifest["script_format"] == "dialogue"
    assert chunks_manifest["speaker_voice_map"] == {"Host": "Kore", "Guest": "Puck"}
    assert len(chunks_manifest["chunks"]) == 4
    assert [chunk["turn_index"] for chunk in chunks_manifest["chunks"]] == [1, 2, 3, 4]
    assert [chunk["speech_duration_ms"] for chunk in chunks_manifest["chunks"]] == [1000] * 4
    assert [chunk["pause_after_ms"] for chunk in chunks_manifest["chunks"]] == [250, 600, 250, 0]
    assert [(chunk["start_ms"], chunk["end_ms"]) for chunk in chunks_manifest["chunks"]] == [
        (0, 1000),
        (1250, 2250),
        (2850, 3850),
        (4100, 5100),
    ]
    for chunk in chunks_manifest["chunks"]:
        assert len(chunk["audio_sha256"]) == 64
        int(chunk["audio_sha256"], 16)
        assert "transcript" not in chunk

    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["script_format"] == "dialogue"
    synthesis_identity = run_state.get("synthesis_identity")
    assert synthesis_identity is not None
    assert len(synthesis_identity) == 64
    int(synthesis_identity, 16)
    assert all("text" not in turn and "transcript" not in turn for turn in run_state["chunks"])

    assert (run_dir / "chunks" / "turn_0001.mp3").exists()
    assert (run_dir / "chunks" / "turn_0004.mp3").exists()
    assert (run_dir / "e2e-dialogue-voiceover-google-gemini-3-1-flash-tts-preview.mp3").exists()
    assert (run_dir / "manifest.json").exists()


@pytest.mark.parametrize("no_retry", [False, True])
def test_openrouter_dialogue_failure_makes_one_request_and_stops(
    tmp_path, monkeypatch, capsys, no_retry
):
    from unittest.mock import MagicMock, patch

    import voiceover_pipeline.cli as cli

    script = _write_dialogue_script(tmp_path)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b""
    mock_response.headers = {"Content-Type": "audio/pcm"}
    _patch_generation_io(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "«redacted:sk-…»")
    argv = [
        "voiceover",
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "failed-dialogue",
        "--tts-quality-provider",
        "fixture-asr",
        "--json",
    ]
    if no_retry:
        argv.append("--no-retry")
    monkeypatch.setattr(sys, "argv", argv)

    with patch(
        "voiceover_pipeline.providers.openrouter_tts.requests.post",
        return_value=mock_response,
    ) as mock_post:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()

    assert exit_info.value.code == 30
    assert mock_post.call_count == 1
    captured = capsys.readouterr()
    assert "empty audio body" in captured.out
    assert captured.err == ""
    run_state = json.loads(
        (tmp_path / "out" / "failed-dialogue" / "run_state.json").read_text(encoding="utf-8")
    )
    assert run_state["completed_count"] == 0
    assert run_state["chunk_count"] == 4
    assert run_state["chunks"] == []


def test_omnivoice_dialogue_validation_accepts_admitted_bank_profiles(tmp_path):
    from voiceover_pipeline.gemini_dialogue import (
        dialogue_turns_from_validation,
        validate_gemini_dialogue_file,
    )

    script = tmp_path / "omnivoice-dialogue.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: gemini-dialogue",
                "speakers:",
                "  Female:",
                "    voice: omni-female-neutral-01",
                "  Male:",
                "    voice: omni-male-deep-01",
                "---",
                "Female: Привет.",
                "Male: Здравствуйте.",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_gemini_dialogue_file(
        script,
        provider="omnivoice-local",
        allowed_voices={"omni-female-neutral-01", "omni-male-deep-01"},
    )

    assert report["valid"] is True
    assert [turn.voice for turn in dialogue_turns_from_validation(report)] == [
        "omni-female-neutral-01",
        "omni-male-deep-01",
    ]


def _dialogue_gate_fixture(tmp_path):
    import argparse

    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact, ScriptChunk

    paths = build_run_paths(tmp_path / "out", "google/gemini-3.1-flash-tts-preview", "gate")
    paths.chunks_dir.mkdir(parents=True)
    turns = ["Первая реплика.", "Вторая реплика."]
    chunks = [
        ScriptChunk(number=number, id=f"turn_{number:04d}", text=text)
        for number, text in enumerate(turns, 1)
    ]
    for chunk in chunks:
        (paths.chunks_dir / f"{chunk.id}.mp3").write_bytes(f"audio-{chunk.number}".encode())
    artifacts = [
        ChunkArtifact(
            number=chunk.number,
            id=chunk.id,
            file=f"{chunk.id}.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunk.text),
            transcript=None,
            client_path=None,
            generation_id=None,
        )
        for chunk in chunks
    ]
    args = argparse.Namespace(tts_quality_provider="fixture-asr", tts_quality_model="fixture-model")
    return args, chunks, artifacts, paths, turns


def test_dialogue_quality_gate_service_writes_exact_success_receipt(tmp_path):
    import json

    from voiceover_pipeline.services import transcription

    args, chunks, artifacts, paths, turns = _dialogue_gate_fixture(tmp_path)
    transcripts = iter(turns)

    def transcribe_quality_audio(_args, _audio_path):
        return (next(transcripts), "fixture-asr", "fixture-model", "fixture-runtime", None)

    receipt = transcription.verify_dialogue_turns_before_concat(
        args=args,
        chunks=chunks,
        chunk_artifacts=artifacts,
        paths=paths,
        transcribe_quality_audio=transcribe_quality_audio,
        sha256_file=lambda path: f"sha-{path.name}",
        fail_quality=lambda message, _receipt: pytest.fail(message),
    )

    assert receipt["status"] == "success"
    assert receipt["passed"] is True
    assert receipt["provider"] == "fixture-asr"
    assert receipt["turn_count"] == 2
    assert [turn["turn_index"] for turn in receipt["turns"]] == [1, 2]
    assert receipt["turns"][0]["audio_sha256"] == "sha-turn_0001.mp3"
    assert all(turn["passed"] for turn in receipt["turns"])
    written = json.loads((paths.output_root / "tts_quality.json").read_text(encoding="utf-8"))
    assert written["status"] == "success"
    assert written["passed"] is True


def test_dialogue_quality_gate_service_fails_closed_before_remaining_turns(tmp_path):
    from voiceover_pipeline.services import transcription

    args, chunks, artifacts, paths, _turns = _dialogue_gate_fixture(tmp_path)
    transcribed = []
    seen_receipts = []

    def transcribe_quality_audio(_args, audio_path):
        transcribed.append(audio_path)
        return ("Совсем другой текст.", "fixture-asr", "fixture-model", "fixture-runtime", None)

    class QualityStop(Exception):
        pass

    def fail_quality(message, receipt):
        seen_receipts.append(receipt)
        raise QualityStop(message)

    with pytest.raises(QualityStop, match="turn 1"):
        transcription.verify_dialogue_turns_before_concat(
            args=args,
            chunks=chunks,
            chunk_artifacts=artifacts,
            paths=paths,
            transcribe_quality_audio=transcribe_quality_audio,
            sha256_file=lambda _path: "sha",
            fail_quality=fail_quality,
        )

    assert len(transcribed) == 1
    assert seen_receipts[0]["status"] == "quality_failed"
    assert seen_receipts[0]["passed"] is False
    assert seen_receipts[0]["failed_turn"] == 1
