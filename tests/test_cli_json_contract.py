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


def test_prepare_split_chunks_extracts_filesystem_preparation(tmp_path):
    from voiceover_pipeline.commands.split import ScriptNotFoundError, prepare_split_chunks

    script = tmp_path / "script.md"
    script.write_text("alpha******beta******", encoding="utf-8")
    chunks = prepare_split_chunks(script)

    assert [(c.id, c.text) for c in chunks] == [
        ("chunk_01", "alpha"),
        ("chunk_02", "beta"),
    ]
    assert [(c.id, len(c.text)) for c in prepare_split_chunks(script, "******")] == [
        ("chunk_01", 5),
        ("chunk_02", 4),
    ]

    missing = tmp_path / "missing.md"
    with pytest.raises(ScriptNotFoundError, match="Script file not found"):
        prepare_split_chunks(missing)


def test_split_json_contract_is_exact_and_invalid_path_preserved():
    code, data = cli_json("split", "--script", str(fixture_path("smoke_test.md")), "--json")
    assert code == 0
    assert data == {
        "status": "success",
        "chunks": [
            {"id": "chunk_01", "chars": 254},
            {"id": "chunk_02", "chars": 116},
        ],
    }

    missing = fixture_path("missing.md")
    code, data = cli_json("split", "--script", str(missing), "--json")
    assert code == 2
    assert data == {
        "status": "error",
        "error": f"Script file not found: {missing}",
        "code": 2,
    }


def test_split_text_output_and_invalid_path_stderr():
    from conftest import run_cli

    proc = run_cli("split", "--script", str(fixture_path("smoke_test.md")))
    assert proc.returncode == 0
    assert proc.stdout.splitlines() == ["chunk_01: 254 chars", "chunk_02: 116 chars"]
    assert proc.stderr == ""

    missing = fixture_path("missing.md")
    proc = run_cli("split", "--script", str(missing))
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr == f"Error: Script file not found: {missing}\n"


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


def test_long_russian_omnivoice_design_json_rejects_before_provider_construction(
    monkeypatch, tmp_path, capsys
):
    import sys

    from voiceover_pipeline import cli

    script = tmp_path / "long-russian.md"
    script.write_text(" ".join(["слово"] * 76), encoding="utf-8")
    constructed = False

    def forbidden_provider_construction(**kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError(f"provider must not be constructed: {kwargs}")

    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        cli.OmniVoiceLocalTTSProvider,
        "from_environment",
        forbidden_provider_construction,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "omnivoice-local",
            "--mode",
            "design",
            "--design-instruction",
            "female",
            "--script",
            str(script),
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert constructed is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["code"] == 2
    assert payload["details"]["error_code"] == "OMNIVOICE_DESIGN_UNSUPPORTED_LONG_LANGUAGE"
    assert len(payload["details"]["alternatives"]) == 4
    assert "unreliable" in payload["error"]


def test_short_russian_omnivoice_design_emits_experimental_warning(capsys):
    import argparse

    from voiceover_pipeline.cli import _enforce_omnivoice_design_route
    from voiceover_pipeline.models import ScriptChunk

    args = argparse.Namespace(provider="omnivoice-local", mode="design")
    _enforce_omnivoice_design_route(
        args,
        [ScriptChunk(number=1, id="chunk_01", text="Короткий русский текст.")],
    )

    warning = capsys.readouterr().err
    assert "experimental" in warning
    assert "trained only for Chinese and English" in warning


def test_omnivoice_design_help_names_long_russian_limit_and_alternatives():
    from conftest import run_cli

    proc = run_cli("generate", "--help")

    assert proc.returncode == 0
    help_text = " ".join(proc.stdout.split())
    assert "30-second" in help_text
    assert "clone, preset, short experimental clips, or another provider" in help_text


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
    assert "google/gemini-3.8-flash-tts" in models
    assert "google/gemini-3.8-flash-lite-tts" in models


def test_openrouter_tts_models_match_current_speech_catalog():
    code, data = cli_json("list", "providers", "--json")
    assert code == 0
    or_tts = next(p for p in data["providers"] if p["id"] == "openrouter-tts")
    assert or_tts["models"] == [
        "google/gemini-3.1-flash-tts-preview",
        "google/gemini-3.8-flash-tts",
        "google/gemini-3.8-flash-lite-tts",
    ]


def test_openrouter_tts_voices_match_current_speech_catalog():
    code, data = cli_json("list", "voices", "--provider", "openrouter-tts", "--json")
    assert code == 0
    categories = data["voice_categories"]
    assert isinstance(categories, dict)
    assert data["voices"] == categories["gemini"]
    assert set(categories) == {"gemini"}


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


def test_model_validation_rejects_stale_openrouter_tts_before_request():
    import pytest

    from voiceover_pipeline.cli import CliError, _validate_model_for_provider

    with pytest.raises(CliError, match="not valid for provider 'openrouter-tts'") as exc_info:
        _validate_model_for_provider("openrouter-tts", "openai/gpt-4o-mini-tts-2025-12-15")
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


def test_direct_cost_kwargs_zero_cost_rub_wins_over_nonzero_cost():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": 0, "cost": 1.5}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.0
    assert kwargs["cost_exact"] == "0"
    assert kwargs["cost_rub"] == 0.0
    assert kwargs["cost_rub_exact"] == "0"
    assert kwargs["cost_currency"] == "RUB"


def test_direct_cost_kwargs_zero_cost_rub_without_fallback_cost():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": 0.0}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.0
    assert kwargs["cost_exact"] is None
    assert kwargs["cost_currency"] == "RUB"


def test_direct_cost_kwargs_falls_back_to_cost_when_cost_rub_missing():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost": 0.25}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.25
    assert kwargs["cost_exact"] is None
    assert kwargs["cost_rub"] == 0.25
    assert kwargs["cost_currency"] == "RUB"


def test_direct_cost_kwargs_falls_back_to_cost_when_cost_rub_none():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": None, "cost": 0.25}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.25
    assert kwargs["cost_rub"] == 0.25
    assert kwargs["cost_currency"] == "RUB"


def test_direct_cost_kwargs_preserves_usage_and_direct_source():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    usage = {"cost_rub": 0.1575, "cost": 0.1575, "tokens": 12}
    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": usage},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["usage"] == usage
    assert kwargs["generation_detail_source"] == "Polza API usage.cost_rub (direct)"


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


def test_direct_cost_kwargs_keeps_integer_and_string_exact_values():
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    integer = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": 0}},
    )
    int_kwargs = _direct_cost_kwargs("polza-tts", integer)
    assert int_kwargs["cost"] == 0.0
    assert int_kwargs["cost_exact"] == "0"
    assert int_kwargs["cost_rub_exact"] == "0"

    string = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": "0.1575"}},
    )
    str_kwargs = _direct_cost_kwargs("polza-tts", string)
    assert str_kwargs["cost"] == 0.1575
    assert str_kwargs["cost_exact"] == "0.1575"
    assert str_kwargs["cost_rub_exact"] == "0.1575"


def test_direct_cost_kwargs_keeps_decimal_exact_value():
    from decimal import Decimal

    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": Decimal("0.1234567890123456789")}},
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost"] == 0.1234567890123456789
    assert kwargs["cost_exact"] == "0.1234567890123456789"


def test_direct_cost_kwargs_projects_decimal_usage_to_json_safe_numbers():
    from decimal import Decimal

    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={
            "usage_direct": {
                "cost_rub": Decimal("0.1"),
                "nested": {"cost": Decimal("0.2")},
                "tokens": 4,
            }
        },
    )
    kwargs = _direct_cost_kwargs("polza-tts", result)
    assert kwargs["cost_exact"] == "0.1"
    assert kwargs["usage"]["nested"]["cost"] == 0.2
    assert not isinstance(kwargs["usage"]["nested"]["cost"], Decimal)
    json.dumps(kwargs, allow_nan=False)


@pytest.mark.parametrize("value", ["bad", "", True, False, [0.5], {"amount": 0.5}])
def test_direct_cost_kwargs_unknown_cost_value_is_empty(value):
    from voiceover_pipeline.cli import _direct_cost_kwargs
    from voiceover_pipeline.models import SynthesisResult

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": value}},
    )
    assert _direct_cost_kwargs("polza-tts", result) == {}


def test_direct_cost_kwargs_cli_wrapper_delegates_to_service():
    from decimal import Decimal

    from voiceover_pipeline.cli import _direct_cost_kwargs, _media_observed_cost
    from voiceover_pipeline.models import SynthesisResult
    from voiceover_pipeline.services import costs

    result = SynthesisResult(
        audio_bytes=b"fake",
        audio_format="mp3",
        raw_metadata={"usage_direct": {"cost_rub": Decimal("0.1"), "tokens": 4}},
    )
    assert _direct_cost_kwargs("polza-tts", result) == costs.direct_cost_kwargs("polza-tts", result)
    assert _media_observed_cost({"cost": 0.3}) == costs.media_observed_cost({"cost": 0.3})


def test_recovery_service_status_alignment_and_cli_wrappers(tmp_path):
    """The recovery service matches the CLI wrappers and ``status --json``.

    A saved raw receipt is resumable with no provider request, so the service
    reports it, the old CLI wrappers delegate to the service, and ``status
    --json`` reports the same run as resumable without echoing the receipt.
    """
    import hashlib

    from voiceover_pipeline.cli import _known_raw_recovery, _recoverable_paid_attempts
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        raw_audio_relative_path,
        record_raw_audio_saved,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import recovery

    run_id = "recovery-service-status"
    run_dir = tmp_path / "out" / run_id
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    script = fixture_path("smoke_test.md")
    chunks = split_markdown_by_delimiter(script, "******")
    model = "openai/gpt-4o-mini-tts"
    voice = "ash"
    state = initial_state(
        provider="polza-tts",
        model=model,
        voice=voice,
        script_path=script,
        chunks=chunks,
        script_format="markdown",
        run_id=run_id,
    )
    first = chunks[0]
    begin_chunk_attempt(state, chunk_id=first.id, number=first.number)
    relative = raw_audio_relative_path(first.id, "mp3")
    raw_path = run_dir / relative
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"paid-mp3")
    record_raw_audio_saved(
        state,
        chunk_id=first.id,
        number=first.number,
        audio_format="mp3",
        relative_path=relative,
        sha256=hashlib.sha256(b"paid-mp3").hexdigest(),
        generation_id="gen-1",
    )
    atomic_write_json(run_dir / "run_state.json", state)

    service_receipt = recovery.known_raw_recovery(state, run_dir, chunks_dir)
    assert service_receipt is not None
    assert _known_raw_recovery(state, run_dir, chunks_dir) == service_receipt
    assert _recoverable_paid_attempts(
        state,
        provider="polza-tts",
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_dir,
    ) == recovery.recoverable_paid_attempts(
        state,
        provider="polza-tts",
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_dir,
    )

    code, data = cli_json(
        "status", "--output-dir", str(tmp_path / "out"), "--run-id", run_id, "--json"
    )

    assert code == 0
    assert data["can_resume"] is True
    assert data["resume_block_reason"] is None
    assert "paid-mp3" not in json.dumps(data)


def test_gemini_prompt_mode_in_manifest_is_none():
    from voiceover_pipeline.tts_prompting import resolve_prompt_mode

    mode = resolve_prompt_mode("openrouter-tts", "google/gemini-3.1-flash-tts-preview")
    assert mode == "none"


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


def test_list_omnivoice_providers_expose_auto_mode():
    code, data = cli_json("list", "providers", "--json")

    assert code == 0
    omnivoice = next(p for p in data["providers"] if p["id"] == "omnivoice-local")
    assert "auto" in omnivoice["modes"]


def test_omnivoice_auto_mode_dry_run_reports_sentence_packed_inference_chunks(tmp_path):
    script = tmp_path / "auto-report.md"
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
        "--mode",
        "auto",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "auto-dry-run",
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 1
    assert data["original_chunks"] == data["chunks"]
    assert not (tmp_path / "out" / "auto-dry-run").exists()


@pytest.mark.parametrize(
    ("mode", "mode_args"),
    [
        ("clone", ("--reference-audio", "{reference_audio}", "--reference-text", "reference")),
        ("design", ("--design-instruction", "female, young adult, moderate pitch")),
    ],
)
def test_omnivoice_clone_and_design_reach_provider_fail_as_single_json_error(
    tmp_path, mode, mode_args
):
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

    assert code == 30
    assert data["status"] == "error"
    assert data["code"] == 30
    assert "not implemented" not in data["error"]
    assert len(data) == 3


@pytest.mark.parametrize(
    ("mode", "mode_args", "message"),
    [
        ("clone", (), "reference audio"),
        ("clone", ("--reference-audio", "{reference_audio}"), "reference-text"),
        ("design", (), "design-instruction"),
        (
            "design",
            ("--reference-audio", "{reference_audio}", "--reference-text", "reference"),
            "rejects",
        ),
    ],
)
def test_omnivoice_invalid_clone_and_design_keep_exit_two_contract(
    tmp_path, mode, mode_args, message
):
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
    assert message in data["error"]
    assert len(data) == 3


def test_omnivoice_design_instruction_invalid_vocabulary_exits_two():
    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--mode",
        "design",
        "--design-instruction",
        "warm and clear",
        "--json",
    )

    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2
    assert "unsupported" in data["error"]
    assert len(data) == 3


def _write_gemini_dialogue(tmp_path, body, extra_meta="", speakers=None):
    if speakers is None:
        speakers = "\n".join(
            [
                "  Speaker1:",
                "    display_name: Первый диктор",
                "    voice: Puck",
                "    profile: calm host",
                "  Speaker2:",
                "    display_name: Второй диктор",
                "    voice: Kore",
                "    profile: energetic co-host",
            ]
        )
    script = tmp_path / "dialogue.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: gemini-dialogue",
                "language: ru",
                "model: google/gemini-3.1-flash-tts-preview",
                "speakers:",
                speakers,
                "allowed_tags:",
                "  - warmly",
                "  - calmly",
                "  - curious",
                "max_chunk_bytes: 3500",
                extra_meta.rstrip(),
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return script


def test_gemini_dialogue_validate_agent_json_is_one_object(tmp_path):
    from conftest import run_cli

    script = _write_gemini_dialogue(
        tmp_path,
        "Speaker1: [warmly] Привет.\nSpeaker2: [curious] Проверяем два голоса.",
    )
    proc = run_cli(
        "validate", "--script", str(script), "--format", "gemini-dialogue", "--agent", "--json"
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["valid"] is True
    assert data["status"] == "success"
    assert data["format"] == "dialogue"
    assert data["speaker_voice_map"] == {"Speaker1": "Puck", "Speaker2": "Kore"}
    json_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    assert len(json_lines) == 1
    assert proc.stderr.strip() == ""


def test_invalid_gemini_dialogue_generate_json_is_single_error_object(tmp_path):
    from conftest import run_cli

    script = _write_gemini_dialogue(
        tmp_path,
        "Speaker1: [warmly] Привет.\nSpeaker2: [curious] Проверяем.",
        speakers="  Speaker1:\n    voice: Puck\n  Speaker2:\n    voice: Puck",
    )
    proc = run_cli(
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "bad-dialogue",
        "--json",
    )
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["status"] == "error"
    assert data["code"] == 2
    assert "DUPLICATE_SPEAKER_VOICE" in data["error"] or "distinct" in data["error"]
    assert "details" in data
    assert data["details"]["valid"] is False
    codes = {item["code"] for item in data["details"]["errors"]}
    assert "DUPLICATE_SPEAKER_VOICE" in codes
    json_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    assert len(json_lines) == 1
    assert proc.stderr.strip() == ""


def test_gemini_dialogue_json_and_json_events_rejected(tmp_path):
    script = _write_gemini_dialogue(
        tmp_path,
        "Speaker1: [warmly] Привет.\nSpeaker2: [curious] Проверяем два голоса.",
    )
    code, data = cli_json(
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "json-events-conflict",
        "--json",
        "--json-events",
    )
    assert code == 2
    assert data["status"] == "error"
    assert data["code"] == 2
    assert "mutually exclusive" in data["error"]
    assert not (tmp_path / "out" / "json-events-conflict").exists()


def _write_reference_wav(path, *, rate=24_000):
    import wave

    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes((b"\x00\x00" * 4000) + (b"\xff\x7f" * 4000))


def _build_voice_bank(tmp_path):
    import hashlib

    bank_root = tmp_path / "bank"
    (bank_root / "voices").mkdir(parents=True)
    reference = bank_root / "voices" / "main.wav"
    _write_reference_wav(reference)
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
                        "description": "Primary narration voice",
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


def test_omnivoice_preset_bank_dry_run_reports_success(tmp_path):
    catalog_path = _build_voice_bank(tmp_path)

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--mode",
        "preset",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "bank-dry-run",
        "--voice-bank",
        str(catalog_path),
        "--voice",
        "main",
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["provider"] == "omnivoice-local"
    assert not (tmp_path / "out" / "bank-dry-run").exists()


def test_list_omnivoice_voices_with_bank_returns_profiles(tmp_path):
    catalog_path = _build_voice_bank(tmp_path)

    code, data = cli_json(
        "list",
        "voices",
        "--provider",
        "omnivoice-local",
        "--voice-bank",
        str(catalog_path),
        "--json",
    )

    assert code == 0
    assert data["voices"] == ["main"]
    assert data["profiles"] == [
        {
            "id": "main",
            "display_name": "Main Narrator",
            "description": "Primary narration voice",
            "language": "ru",
        }
    ]
    assert data["voice_selection"] == {
        "kind": "built-in-style-condition",
        "condition": "female",
        "named_preset": False,
        "voice_cloning": False,
        "voice_design": False,
    }
