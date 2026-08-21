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
        "concat_mp3_chunks",
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

    assert mock_post.call_count == 2
    bodies = [call[1]["json"] for call in mock_post.call_args_list]
    for body in bodies:
        assert body["model"] == "google/gemini-3.1-flash-tts-preview"
        assert body["voice"] == "Kore"
        configs = body["multi_speaker_voice_config"]["speaker_voice_configs"]
        assert len(configs) == 2
        assert {config["speaker"] for config in configs} == {"Host", "Guest"}
        for config in configs:
            assert config["voice_config"]["prebuilt_voice_config"]["voice_name"] in ("Kore", "Puck")

    run_dir = tmp_path / "out" / "e2e-dialogue"
    chunks_manifest = json.loads((run_dir / "chunks" / "chunks.json").read_text(encoding="utf-8"))
    assert chunks_manifest["script_format"] == "gemini-dialogue"
    assert chunks_manifest["speaker_voice_map"] == {"Host": "Kore", "Guest": "Puck"}
    assert len(chunks_manifest["chunks"]) == 2

    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["script_format"] == "gemini-dialogue"
    voice_identity = run_state.get("voice_identity")
    assert voice_identity is not None
    assert len(voice_identity) == 64
    int(voice_identity, 16)

    assert (run_dir / "chunks" / "chunk_01.mp3").exists()
    assert (run_dir / "chunks" / "chunk_02.mp3").exists()
    assert (run_dir / "e2e-dialogue-voiceover-google-gemini-3-1-flash-tts-preview.mp3").exists()
    assert (run_dir / "manifest.json").exists()
