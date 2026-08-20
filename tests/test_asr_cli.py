import argparse
import json
import sys
import types

import pytest
from conftest import cli_json, fixture_path

from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider


def test_list_asr_providers_reads_registry_listing(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli

    spec = ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=FixtureASRProvider,
        models=({"id": "fixture-model"},),
        capabilities=ASRCapabilities(
            batch_audio=True, device_modes=("cpu",), compute_modes=("float32",)
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(cli, "list_asr_provider_specs", lambda: [spec])

    with pytest.raises(SystemExit) as exit_info:
        cli.list_cmd(argparse.Namespace(target="asr-providers", json_output=True))

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["status"] == "success"
    assert data["asr_providers"][0]["id"] == "fixture-local"
    assert data["asr_providers"][0]["capabilities"]["batch_audio"] is True


def test_transcribe_unknown_provider_keeps_machine_json_and_invalid_args_exit_code():
    code, data = cli_json(
        "transcribe",
        "--audio",
        str(fixture_path("smoke_test.md")),
        "--provider",
        "missing",
        "--json",
    )

    assert code == 2
    assert data == {
        "status": "error",
        "error": "Unknown ASR provider: missing",
        "code": 2,
    }


def test_transcribe_parser_has_finite_audio_controls_but_no_generic_prompt_flags():
    from voiceover_pipeline.cli import build_parser

    parser = build_parser()
    parsed = parser.parse_args(
        "transcribe --audio audio.wav --provider fixture-local --model fixture-model --language ru --device cpu --compute float32 --json".split()
    )

    assert parsed.audio == "audio.wav"
    assert parsed.provider == "fixture-local"
    assert parsed.language == "ru"
    assert parsed.device == "cpu"
    assert parsed.compute == "float32"
    with pytest.raises(SystemExit):
        parser.parse_args(
            "transcribe --audio audio.wav --provider fixture-local --prompt ignored".split()
        )


def test_transcribe_parser_exposes_context_sources_and_runtime_choice():
    from voiceover_pipeline.cli import build_parser

    parser = build_parser()
    parsed = parser.parse_args(
        "transcribe --audio audio.wav --provider fixture-local --context terms --runtime python".split()
    )

    assert parsed.context == "terms"
    assert parsed.context_file is None
    assert parsed.runtime == "python"
    assert (
        parser.parse_args("transcribe --audio audio.wav --provider fixture-local".split()).runtime
        == "auto"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            "transcribe --audio audio.wav --provider fixture-local --context terms --context-file context.txt".split()
        )


def _fixture_spec(*, available=True):
    return ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=FixtureASRProvider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(
            available=available,
            remediation="Install the approved optional ASR runtime.",
        ),
    )


@pytest.mark.parametrize("context_kind", ["missing", "unreadable", "blank"])
def test_transcribe_context_file_is_validated_before_provider_probe(
    monkeypatch, tmp_path, context_kind
):
    import voiceover_pipeline.cli as cli

    context_path = tmp_path / "context.txt"
    if context_kind == "unreadable":
        context_path.mkdir()
    elif context_kind == "blank":
        context_path.write_text(" \n\t", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "get_asr_provider_spec",
        lambda _provider_id: pytest.fail("provider lookup must follow context-file validation"),
    )
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--context-file",
            str(context_path),
        ]
    )

    with pytest.raises(cli.CliError, match="blank|read") as error:
        cli.transcribe_cmd(args)
    assert error.value.code == 2


@pytest.mark.parametrize("context_source", ["inline", "file"])
def test_transcribe_context_and_runtime_reach_request_without_public_context(
    monkeypatch, capsys, tmp_path, context_source
):
    import voiceover_pipeline.cli as cli

    context = "private context that must not be public"
    context_path = tmp_path / "context.txt"
    context_path.write_text(context, encoding="utf-8")
    requests: list[ASRRequest] = []
    original_transcribe = FixtureASRProvider.transcribe

    def capture_request(provider, request):
        requests.append(request)
        return original_transcribe(provider, request)

    monkeypatch.setattr(FixtureASRProvider, "transcribe", capture_request)
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: _fixture_spec())
    context_args = (
        ["--context", context]
        if context_source == "inline"
        else ["--context-file", str(context_path)]
    )
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--language",
            "ru",
            "--device",
            "cpu",
            "--compute",
            "float32",
            *context_args,
            "--runtime",
            "python",
            "--json",
        ]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert requests[0].hints.context_text == context
    assert requests[0].runtime_choice == "python"
    assert context not in output


def test_transcribe_explicit_audio_cpp_fails_closed_before_probe_or_factory(monkeypatch):
    import voiceover_pipeline.cli as cli

    calls: list[str] = []

    def probe():
        calls.append("probe")
        return ASRDependencyHealth(available=True, remediation="")

    def factory():
        calls.append("factory")
        return FixtureASRProvider()

    spec = ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=factory,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=probe,
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: spec)
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--runtime",
            "audio-cpp",
        ]
    )

    with pytest.raises(cli.CliError, match="audio-cpp") as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 2
    assert calls == []


class FixtureASRProvider(ASRProvider):
    provider_id = "fixture-local"

    def transcribe(self, request: ASRRequest) -> ASRResult:
        assert request.model_id == "fixture-model"
        assert request.language == "ru"
        assert request.device == "cpu"
        assert request.compute == "float32"
        return ASRResult(
            transcript="fixture transcript",
            provider_id=self.provider_id,
            model_id="fixture-model",
            language="ru",
            execution=ASRExecutionReceipt(
                runtime="fixture-runtime",
                runtime_version="1.0",
                resolved_device="cpu",
                resolved_compute="float32",
                measurements={"wall_s": 0.25},
            ),
        )


def test_transcribe_normalizes_a_fixture_provider_without_timestamps(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: _fixture_spec())
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--model",
            "fixture-model",
            "--language",
            "ru",
            "--device",
            "cpu",
            "--compute",
            "float32",
            "--json",
        ]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["transcript"] == "fixture transcript"
    assert data["timestamp_mode"] == "none"
    assert data["segments"] == []
    assert data["words"] == []
    assert data["execution"]["measurements"] == {"wall_s": 0.25}


def test_asr_json_payload_preserves_native_raw_timestamp_entries():
    import voiceover_pipeline.cli as cli

    result = ASRResult(
        transcript="native word",
        provider_id="fixture-local",
        model_id="fixture-model",
        words=(
            ASRWordSpan(text="native ", start_s=0.0, end_s=0.1),
            ASRWordSpan(text="word", start_s=0.2, end_s=0.3),
        ),
        alignment_origin="native",
        execution=ASRExecutionReceipt(
            runtime="fixture-runtime",
            raw_timestamp_entries=(
                {"text": "▁native", "start_s": 0.0, "end_s": 0.1},
                {"text": "▁word", "start_s": 0.2, "end_s": 0.3},
            ),
        ),
    )

    payload = cli._asr_result_payload(result, fixture_path("smoke_test.md"))

    assert payload["execution"]["raw_timestamp_entries"] == [
        {"text": "▁native", "start_s": 0.0, "end_s": 0.1},
        {"text": "▁word", "start_s": 0.2, "end_s": 0.3},
    ]


def test_transcribe_rejects_unrequested_word_timestamps_at_the_provider_boundary(monkeypatch):
    import voiceover_pipeline.cli as cli

    class UnexpectedWordProvider(ASRProvider):
        provider_id = "fixture-local"

        def transcribe(self, request: ASRRequest) -> ASRResult:
            return ASRResult(
                transcript="fixture transcript",
                provider_id=self.provider_id,
                model_id="fixture-model",
                execution=ASRExecutionReceipt(
                    runtime="fixture-runtime",
                    runtime_version="1.0",
                    resolved_device="cpu",
                    resolved_compute="float32",
                ),
                words=(ASRWordSpan(text="fixture transcript", start_s=0.0, end_s=0.25),),
                alignment_origin="native",
            )

    spec = ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=UnexpectedWordProvider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            word_timestamps=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: spec)
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--model",
            "fixture-model",
            "--language",
            "ru",
            "--device",
            "cpu",
            "--compute",
            "float32",
        ]
    )

    with pytest.raises(
        cli.CliError, match="returned word timestamps when timestamp mode is none"
    ) as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 30


def test_transcribe_word_timestamp_flag_reaches_declared_provider(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli

    requests: list[ASRRequest] = []

    class TimedProvider(ASRProvider):
        provider_id = "fixture-local"

        def transcribe(self, request: ASRRequest) -> ASRResult:
            requests.append(request)
            return ASRResult(
                transcript="fixture transcript",
                provider_id=self.provider_id,
                model_id="fixture-model",
                execution=ASRExecutionReceipt(
                    runtime="fixture", resolved_device="cpu", resolved_compute="float32"
                ),
                words=(ASRWordSpan(text="fixture transcript", start_s=0.0, end_s=0.25),),
                alignment_origin="native",
            )

    spec = ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=TimedProvider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            word_timestamps=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: spec)
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "fixture-local",
            "--device",
            "cpu",
            "--compute",
            "float32",
            "--word-timestamps",
            "--json",
        ]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    assert exit_info.value.code == 0
    assert requests[0].timestamp_mode == "word"
    assert json.loads(capsys.readouterr().out)["timestamp_mode"] == "native"


def test_doctor_checks_only_the_selected_asr_dependency_probe(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(
        cli, "get_asr_provider_spec", lambda _provider_id: _fixture_spec(available=False)
    )
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "/fixture/bin")
    monkeypatch.setattr(cli, "read_polza_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_openrouter_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_groq_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_xai_key", lambda: "fixture")

    torch_fixture = types.ModuleType("torch")
    setattr(torch_fixture, "cuda", types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch_fixture)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    args = cli.build_parser().parse_args(
        "doctor --with-asr --asr-provider fixture-local --asr-device cpu --asr-compute float32 --json".split()
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.doctor_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["checks"]["asr_provider"] == {
        "ok": False,
        "provider": "fixture-local",
        "required": True,
    }
    assert data["workflow_ok"] is False
    assert "Install the approved optional ASR runtime." in data["warnings"]
