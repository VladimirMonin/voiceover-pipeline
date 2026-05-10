import pytest
from conftest import cli_json, fixture_path


INVALID_RUN_IDS = [
    (".", "not allowed"),
    ("..", "not allowed"),
    ("a/b", "path separators"),
    ("a\\b", "path separators"),
    ("CON", "reserved name"),
    ("con", "reserved name"),
    ("COM1", "reserved name"),
    ("LPT9", "reserved name"),
    ("PRN", "reserved name"),
    ("AUX", "reserved name"),
    ("NUL", "reserved name"),
    ("prod.", "trailing dot"),
    (" prod", "leading whitespace"),
    ("prod ", "trailing whitespace"),
    (" ", "whitespace only"),
    ("C:\\test", "absolute path"),
    ("test<bad", "illegal char"),
]


@pytest.mark.parametrize("run_id,reason", INVALID_RUN_IDS)
def test_invalid_run_id(run_id, reason):
    code, data = cli_json("generate", "--run-id", run_id, "--json")
    assert code == 2, f"run_id={run_id!r} ({reason}) expected exit 2, got {code}"
    assert data["code"] == 2


VALID_RUN_IDS = [
    "prod",
    "prod-01",
    "prod_01",
    "prod.v1",
    "v0.3.0",
    "test_run",
    "my-run",
]


@pytest.mark.parametrize("run_id", VALID_RUN_IDS)
def test_valid_run_id_does_not_fail(run_id):
    from voiceover_pipeline.cli import _validate_run_id
    result = _validate_run_id(run_id)
    assert result == run_id


INVALID_OUTPUT_DIRS = [
    ("C:\\", "drive root"),
    ("C:\\Windows", "system dir — skip for now"),
]


def test_output_dir_is_drive_root_fails():
    code, data = cli_json("generate", "--output-dir", "C:\\", "--json")
    assert code == 2, f"expected exit 2, got {code}"
    assert "drive root" in data["error"].lower() or "output-dir" in data["error"].lower()


def test_output_dir_is_cwd_fails():
    from pathlib import Path
    cwd = str(Path.cwd())
    code, data = cli_json("generate", "--output-dir", cwd, "--json")
    assert code == 2, f"expected exit 2, got {code}"
    assert "current working directory" in data["error"].lower() or "output-dir" in data["error"].lower()


def test_valid_output_dir_ok(tmp_path):
    out = tmp_path / "builds"
    code, data = cli_json(
        "generate",
        "--output-dir", str(out),
        "--run-id", "valid-dir-test",
        "--script", str(fixture_path("smoke_test.md")),
        "--skip-existing",
        "--json",
    )
    assert code in (0, 30), f"expected 0 or 30, got {code}: {data}"


class TestStylePromptFlags:
    def test_no_style_prompt_flag(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=None, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_style_prompt_file(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        pf = tmp_path / "prompt.txt"
        pf.write_text("File-based narration style", encoding="utf-8")
        ns = argparse.Namespace(style_prompt=None, style_prompt_file=pf, no_style_prompt=False)
        assert _resolve_style_prompt(ns) == "File-based narration style"

    def test_style_prompt_file_missing(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=tmp_path / "missing.txt", no_style_prompt=False)
        with pytest.raises(FileNotFoundError):
            _resolve_style_prompt(ns)

    def test_no_style_prompt_has_priority_over_file(self, tmp_path):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        pf = tmp_path / "prompt.txt"
        pf.write_text("Should not be read", encoding="utf-8")
        ns = argparse.Namespace(style_prompt=None, style_prompt_file=pf, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_no_style_prompt_has_priority_over_cli(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt

        ns = argparse.Namespace(style_prompt="cli style", style_prompt_file=None, no_style_prompt=True)
        assert _resolve_style_prompt(ns) is None

    def test_default_prompt_when_no_flags(self):
        import argparse
        from voiceover_pipeline.cli import _resolve_style_prompt, PODCAST_NARRATION_PROMPT

        ns = argparse.Namespace(style_prompt=None, style_prompt_file=None, no_style_prompt=False)
        assert _resolve_style_prompt(ns) == PODCAST_NARRATION_PROMPT


def write_gemini_dialogue(tmp_path, body, extra_meta=""):
    script = tmp_path / "dialogue.md"
    script.write_text(
        "\n".join([
            "---",
            "format: gemini-dialogue",
            "language: ru",
            "model: google/gemini-3.1-flash-tts-preview",
            "speakers:",
            "  Speaker1:",
            "    display_name: Первый диктор",
            "    voice: Puck",
            "    profile: calm host",
            "  Speaker2:",
            "    display_name: Второй диктор",
            "    voice: Kore",
            "    profile: energetic co-host",
            "allowed_tags:",
            "  - warmly",
            "  - calmly",
            "  - laughs",
            "  - cough",
            "  - crying",
            "  - curious",
            "  - gasp",
            "  - medium pause",
            "  - thoughtfully",
            "  - uhm",
            "max_chunk_bytes: 3500",
            extra_meta.rstrip(),
            "---",
            body,
        ]),
        encoding="utf-8",
    )
    return script


class TestGeminiDialogueValidation:
    def test_valid_gemini_dialogue_script(self, tmp_path):
        script = write_gemini_dialogue(
            tmp_path,
            "Speaker1: [warmly] Привет.\nSpeaker2: [curious] Проверяем два голоса.",
        )
        code, data = cli_json("validate", "--script", str(script), "--format", "gemini-dialogue", "--json")
        assert code == 0
        assert data["valid"] is True
        assert data["speaker_voice_map"] == {"Speaker1": "Puck", "Speaker2": "Kore"}
        assert data["chunk_reports"][0]["turn_count"] == 2

    def test_gemini_dialogue_reports_all_errors(self, tmp_path):
        script = write_gemini_dialogue(
            tmp_path,
            "Speaker3: [angry] Неизвестный спикер.\n"
            "Это строка без спикера.\n"
            "Speaker1:",
        )
        code, data = cli_json(
            "validate", "--script", str(script), "--format", "gemini-dialogue", "--agent", "--json"
        )
        assert code == 0
        assert data["valid"] is False
        codes = {item["code"] for item in data["errors"]}
        assert "UNKNOWN_SPEAKER" in codes
        assert "INVALID_AUDIO_TAG" in codes
        assert "LINE_WITHOUT_SPEAKER" in codes
        assert "EMPTY_REPLICA" in codes

    def test_gemini_dialogue_checks_last_chunk_size(self, tmp_path):
        long_text = "очень длинный текст " * 240
        script = write_gemini_dialogue(
            tmp_path,
            "Speaker1: Короткий первый чанк.\n******\nSpeaker2: " + long_text,
        )
        code, data = cli_json("validate", "--script", str(script), "--format", "gemini-dialogue", "--json")
        assert code == 0
        assert data["valid"] is False
        chunk_errors = [item for item in data["errors"] if item["code"] == "CHUNK_TOO_LARGE"]
        assert chunk_errors
        assert chunk_errors[0]["chunk"] == 2

    def test_gemini_dialogue_default_safe_tags_from_prompting_guide(self, tmp_path):
        script = write_gemini_dialogue(
            tmp_path,
            "Speaker1: [thoughtfully] Сначала спокойно. [medium pause] Потом пауза.\n"
            "Speaker2: [gasp] Ого. [cough] Простите. [uhm] Продолжим.",
        )
        code, data = cli_json("validate", "--script", str(script), "--format", "gemini-dialogue", "--json")
        assert code == 0
        assert data["valid"] is True

    def test_gemini_dialogue_detects_prompt_skeleton_in_body(self, tmp_path):
        script = write_gemini_dialogue(
            tmp_path,
            "#### TRANSCRIPT\nSpeaker1: Это уже реплика.",
        )
        code, data = cli_json("validate", "--script", str(script), "--format", "gemini-dialogue", "--agent", "--json")
        assert code == 0
        assert data["valid"] is False
        assert any(item["code"] == "PROMPT_SKELETON_IN_DIALOGUE_BODY" for item in data["errors"])


def write_voiceover_script(tmp_path, meta_lines, body="Первый чанк.\n******\nВторой чанк."):
    script = tmp_path / "voiceover.md"
    script.write_text(
        "\n".join(["---", *meta_lines, "---", body]),
        encoding="utf-8",
    )
    return script


class TestVoiceoverMetadataValidation:
    def test_valid_polza_tts_voiceover_metadata(self, tmp_path):
        script = write_voiceover_script(tmp_path, [
            "format: voiceover",
            "provider: polza-tts",
            "model: openai/gpt-4o-mini-tts",
            "voice: ash",
            "max_chunk_chars: 2000",
        ])
        code, data = cli_json("validate", "--script", str(script), "--json")
        assert code == 0
        assert data["valid"] is True
        assert data["format"] == "voiceover"
        assert data["effective_config"]["provider"] == "polza-tts"
        assert data["effective_config"]["model"] == "openai/gpt-4o-mini-tts"
        assert data["effective_config"]["voice"] == "ash"

    def test_valid_openrouter_gemini_voiceover_metadata_with_style(self, tmp_path):
        script = write_voiceover_script(tmp_path, [
            "format: voiceover",
            "provider: openrouter-tts",
            "model: google/gemini-3.1-flash-tts-preview",
            "voice: Puck",
            "style_prompt: >",
            "  Speak as a calm technical podcast narrator.",
        ])
        code, data = cli_json("validate", "--script", str(script), "--json")
        assert code == 0
        assert data["valid"] is True
        assert data["warnings"] == []
        assert data["effective_config"]["style_prompt"] == "Speak as a calm technical podcast narrator."

    def test_voiceover_metadata_reports_all_errors(self, tmp_path):
        script = write_voiceover_script(tmp_path, [
            "format: voiceover",
            "provider: polza-tts",
            "model: elevenlabs/text-to-speech-turbo-2-5",
            "voice: ash",
        ], body="длинно " * 500)
        code, data = cli_json("validate", "--script", str(script), "--format", "voiceover", "--agent", "--json")
        assert code == 0
        assert data["valid"] is False
        codes = {item["code"] for item in data["errors"]}
        assert "INVALID_VOICE" in codes
        assert "CHUNK_TOO_LARGE" in codes

    def test_voiceover_metadata_cli_overrides_provider_model_voice(self, tmp_path):
        script = write_voiceover_script(tmp_path, [
            "format: voiceover",
            "provider: polza-tts",
            "model: openai/gpt-4o-mini-tts",
            "voice: ash",
        ])
        code, data = cli_json(
            "validate",
            "--script", str(script),
            "--format", "voiceover",
            "--provider", "openrouter-tts",
            "--model", "openai/gpt-4o-mini-tts-2025-12-15",
            "--voice", "alloy",
            "--json",
        )
        assert code == 0
        assert data["valid"] is True
        assert data["effective_config"]["provider"] == "openrouter-tts"
        assert data["effective_config"]["model"] == "openai/gpt-4o-mini-tts-2025-12-15"
        assert data["effective_config"]["voice"] == "alloy"

    def test_voiceover_metadata_warns_about_prompt_skeleton_in_body(self, tmp_path):
        script = write_voiceover_script(tmp_path, [
            "format: voiceover",
            "provider: openrouter-tts",
            "model: google/gemini-3.1-flash-tts-preview",
            "voice: Puck",
        ], body="#### TRANSCRIPT\n[thoughtfully] Это тело, но маркер prompt skeleton здесь лишний.")
        code, data = cli_json("validate", "--script", str(script), "--format", "voiceover", "--json")
        assert code == 0
        assert data["valid"] is True
        assert any(item["code"] == "PROMPT_SKELETON_IN_BODY" for item in data["warnings"])
