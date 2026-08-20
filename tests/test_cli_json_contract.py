import json

import pytest
from conftest import cli_json, fixture_path


def test_doctor_json_parseable():
    code, data = cli_json("doctor", "--json")
    assert code == 0
    assert data["status"] == "success"
    assert "required_ok" in data
    assert "workflow_ok" in data
    assert "warnings" in data
    assert isinstance(data["workflow_ok"], bool)


def test_doctor_qwen_workflow():
    code, data = cli_json("doctor", "--provider", "qwen-local", "--json")
    assert code == 0
    checks = data["checks"]
    assert isinstance(checks, dict)
    cuda_check = checks["cuda"]
    assert isinstance(cuda_check, dict)
    assert cuda_check["required"] is True
    assert data["workflow_ok"] is cuda_check["ok"]


def test_doctor_with_timings():
    code, data = cli_json("doctor", "--with-timings", "--json")
    assert code == 0
    assert data["checks"]["faster_whisper"]["required"] is True


def test_validate_json_parseable():
    code, data = cli_json("validate", "--script", str(fixture_path("smoke_test.md")), "--json")
    assert code == 0
    assert data["status"] == "success"
    assert data["valid"] is True
    assert data["chunks"] > 0


def test_validate_missing_script():
    code, data = cli_json("validate", "--script", str(fixture_path("missing.md")), "--json")
    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2


def test_split_json_parseable():
    code, data = cli_json("split", "--script", str(fixture_path("smoke_test.md")), "--json")
    assert code == 0
    assert data["status"] == "success"
    assert len(data["chunks"]) > 0


def test_split_missing_script():
    code, data = cli_json("split", "--script", str(fixture_path("missing.md")), "--json")
    assert code == 2
    assert data["code"] == 2


def test_list_providers():
    code, data = cli_json("list", "providers", "--json")
    assert code == 0
    assert data["status"] == "success"
    ids = [p["id"] for p in data["providers"]]
    assert "polza-chat-audio" in ids


def test_list_timing_models():
    code, data = cli_json("list", "timing-models", "--json")
    assert code == 0
    assert data["status"] == "success"
    models = [m["id"] for m in data["timing_models"]]
    assert "small" in models


def test_list_voices():
    code, data = cli_json("list", "voices", "--json")
    assert code == 0
    assert data["status"] == "success"
    voices = data["voices"]
    assert isinstance(voices, list)
    assert len(voices) > 0


def test_list_qwen_voices_match_official_custom_voice_model():
    code, data = cli_json("list", "voices", "--provider", "qwen-local", "--json")
    assert code == 0
    assert data["voices"] == [
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    ]


def test_list_omnivoice_voice_contract_exposes_style_condition_not_a_named_voice():
    code, data = cli_json("list", "voices", "--provider", "omnivoice-local", "--json")

    assert code == 0
    assert data["voices"] == []
    assert data["voice_selection"] == {
        "kind": "built-in-style-condition",
        "condition": "female",
        "named_preset": False,
        "voice_cloning": False,
        "voice_design": False,
    }


def test_list_polza_tts_providers():
    code, data = cli_json("list", "providers", "--json")
    assert code == 0
    ids = [p["id"] for p in data["providers"]]
    assert "polza-tts" in ids
    polza_tts = next(p for p in data["providers"] if p["id"] == "polza-tts")
    models = polza_tts["models"]
    assert "elevenlabs/text-to-speech-turbo-2-5" in models
    assert "elevenlabs/text-to-speech-multilingual-v2" in models
    assert "openai/gpt-4o-mini-tts" in models


def test_openrouter_tts_models_include_openai():
    code, data = cli_json("list", "providers", "--json")
    assert code == 0
    or_tts = next(p for p in data["providers"] if p["id"] == "openrouter-tts")
    assert "openai/gpt-4o-mini-tts-2025-12-15" in or_tts["models"]


def test_list_polza_tts_voices():
    code, data = cli_json("list", "voices", "--provider", "polza-tts", "--json")
    assert code == 0
    voices = data["voices"]
    assert isinstance(voices, list)
    assert len(voices) > 0
    categories = data.get("voice_categories")
    assert categories is not None
    assert "openai" in categories
    assert "elevenlabs" in categories
    assert len(categories["elevenlabs"]) > 0


def test_doctor_polza_tts_requires_polza_key():
    code, data = cli_json("doctor", "--provider", "polza-tts", "--json")
    assert code == 0
    assert data["checks"]["polza_key"]["required"] is True


def test_polza_tts_provider_importable():
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider

    p = PolzaTTSProvider(api_key="test-key", model="openai/gpt-4o-mini-tts", voice="ash")
    assert p.provider_id == "polza-tts"
    assert p.model == "openai/gpt-4o-mini-tts"
    assert p.voice == "ash"


def test_openrouter_openai_model_voice_default(tmp_path):
    import argparse

    from voiceover_pipeline.cli import _default_voice

    ns = argparse.Namespace(provider="openrouter-tts", model="openai/gpt-4o-mini-tts-2025-12-15")
    assert _default_voice(ns) == "alloy"


def test_openrouter_gemini_model_voice_default(tmp_path):
    import argparse

    from voiceover_pipeline.cli import _default_voice

    ns = argparse.Namespace(provider="openrouter-tts", model="google/gemini-3.1-flash-tts-preview")
    assert _default_voice(ns) == "Puck"


def test_json_stdout_is_single_object(tmp_path):
    from conftest import run_cli

    proc = run_cli("doctor", "--json")
    assert proc.returncode == 0
    json.loads(proc.stdout)
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    assert len(lines) == 1, f"Expected 1 JSON line, got {len(lines)}"


def test_empty_cli_exit_code():
    from conftest import run_cli

    proc = run_cli()
    assert proc.returncode == 2


def test_timings_missing_audio():
    code, data = cli_json("timings", "--audio", "nonexistent.mp3", "--json")
    assert code == 2
    assert data["code"] == 2


def test_polza_tts_model_default():
    from voiceover_pipeline.config import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["polza-tts"] == "openai/gpt-4o-mini-tts"


def test_openrouter_tts_model_default():
    from voiceover_pipeline.config import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["openrouter-tts"] == "google/gemini-3.1-flash-tts-preview"


def test_model_validation_rejects_invalid():
    import pytest

    from voiceover_pipeline.cli import CliError, _validate_model_for_provider

    with pytest.raises(CliError) as exc_info:
        _validate_model_for_provider("polza-tts", "openai/gpt-audio-mini")
    assert exc_info.value.code == 2


def test_direct_cost_kwargs_populates_for_polza_tts():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": 0.1575, "cost": 0.1575}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.1575
    assert kwargs["cost_currency"] == "RUB"
    assert kwargs["cost_rub"] == 0.1575


def test_direct_cost_kwargs_none_for_other_providers():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": 1.0}},
    )
    assert _direct_cost_kwargs("openrouter-tts", result) == {}
    assert _direct_cost_kwargs("polza-chat-audio", result) == {}


def test_gemini_prompt_mode_in_manifest_is_native():
    from voiceover_pipeline.tts_prompting import resolve_prompt_mode

    mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview")
    assert mode == "native"


def test_qwen_instruct_flag_reaches_local_provider():
    from voiceover_pipeline.cli import _resolve_provider_style_prompt, build_parser, build_provider

    parser = build_parser()
    parsed = parser.parse_args(
        "generate --provider qwen-local --voice Serena --qwen-instruct calm_confident".split()
    )
    provider = build_provider(parsed, api_key="", style_prompt=None, prompt_mode="none")

    assert parsed.qwen_instruct == "calm_confident"
    assert getattr(provider, "_instruct") == "calm_confident"
    assert _resolve_provider_style_prompt(parsed) == "calm_confident"


def test_qwen_instruct_defaults_for_backward_compatibility():
    import argparse

    from voiceover_pipeline.cli import _resolve_provider_style_prompt, build_provider
    from voiceover_pipeline.config import QWEN_INSTRUCT

    args = argparse.Namespace(
        provider="qwen-local",
        mode="preset",
        voice="Serena",
        sample=None,
        sample_text="",
    )
    provider = build_provider(args, api_key="", style_prompt=None, prompt_mode="none")

    assert getattr(provider, "_instruct") == QWEN_INSTRUCT
    assert _resolve_provider_style_prompt(args) == QWEN_INSTRUCT


def test_style_prompt_flags_accepted_by_parser():
    from pathlib import Path

    from voiceover_pipeline.cli import build_parser

    parser = build_parser()
    parsed = parser.parse_args(
        "generate --style-prompt test_prompt --no-style-prompt --style-prompt-file prompt.txt".split()
    )
    assert parsed.style_prompt == "test_prompt"
    assert parsed.no_style_prompt is True
    assert parsed.style_prompt_file == Path("prompt.txt")


def test_timings_asr_provider_is_explicit_and_faster_whisper_remains_default():
    from voiceover_pipeline.cli import build_parser

    parser = build_parser()
    legacy = parser.parse_args("timings --audio recording.wav".split())
    generic = parser.parse_args("timings --audio recording.wav --asr-provider qwen-local".split())

    assert legacy.timing_provider == "faster-whisper"
    assert legacy.asr_provider is None
    assert generic.asr_provider == "qwen-local"
    assert generic.compute is None
    assert (
        parser.parse_args(
            "timings --audio recording.wav --asr-provider qwen-local --compute bfloat16".split()
        ).compute
        == "bfloat16"
    )


def test_transcribe_mutually_exclusive_context_sources_are_json_errors(tmp_path):
    context_path = tmp_path / "context.txt"
    context_path.write_text("context", encoding="utf-8")
    from conftest import run_cli

    proc = run_cli(
        "transcribe",
        "--audio",
        str(fixture_path("smoke_test.md")),
        "--provider",
        "qwen-local",
        "--context",
        "inline",
        "--context-file",
        str(context_path),
        "--json",
    )

    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data == {"status": "error", "error": "Invalid command-line arguments", "code": 2}
    assert len([line for line in proc.stdout.splitlines() if line.strip().startswith("{")]) == 1


@pytest.mark.parametrize("context_kind", ["missing", "unreadable", "blank"])
def test_transcribe_context_file_errors_are_json_and_do_not_leak_content(tmp_path, context_kind):
    context_path = tmp_path / "context.txt"
    secret = "private-context-value"
    if context_kind == "unreadable":
        context_path.mkdir()
    elif context_kind == "blank":
        context_path.write_text(" \n", encoding="utf-8")

    code, data = cli_json(
        "transcribe",
        "--audio",
        str(fixture_path("smoke_test.md")),
        "--provider",
        "qwen-local",
        "--context-file",
        str(context_path),
        "--json",
    )

    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2
    assert secret not in json.dumps(data)
    if context_kind == "blank":
        assert "blank" in data["error"]
    else:
        assert "read" in data["error"]


def test_transcribe_blank_inline_context_is_an_argument_error():
    code, data = cli_json(
        "transcribe",
        "--audio",
        str(fixture_path("smoke_test.md")),
        "--provider",
        "qwen-local",
        "--context",
        " \t",
        "--json",
    )

    assert code == 2
    assert data == {"status": "error", "error": "ASR context must not be blank", "code": 2}


def test_transcribe_explicit_audio_cpp_is_structured_fail_closed_error():
    code, data = cli_json(
        "transcribe",
        "--audio",
        str(fixture_path("smoke_test.md")),
        "--provider",
        "qwen-local",
        "--runtime",
        "audio-cpp",
        "--json",
    )

    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2
    assert "audio-cpp" in data["error"]


@pytest.mark.parametrize(
    ("mode", "mode_args"),
    [
        ("clone", ("--reference-audio", "{reference_audio}", "--reference-text", "reference")),
        ("design", ("--design-instruction", "warm and clear")),
    ],
)
def test_omnivoice_clone_and_design_fail_closed_as_single_json_error(tmp_path, mode, mode_args):
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"fixture")
    resolved_mode_args = [value.format(reference_audio=reference_audio) for value in mode_args]

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--mode",
        mode,
        *resolved_mode_args,
        "--json",
    )

    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2
    assert "not implemented" in data["error"]
    assert len(data) == 3
