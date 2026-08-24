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
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)


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
        assert body["input"].endswith(f"\n\n{text}")
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
