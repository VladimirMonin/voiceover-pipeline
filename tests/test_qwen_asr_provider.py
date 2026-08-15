import importlib
import json
import sys
import types
from dataclasses import replace

import pytest

from conftest import fixture_path
from voiceover_pipeline.models import ASRContextHints, ASRRequest
from voiceover_pipeline.providers import asr_registry
from voiceover_pipeline.providers.asr_registry import ASRProviderRegistry


def test_qwen_asr_registry_listing_declares_text_only_language_and_context_capabilities():
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    assert QWEN_ASR_PROVIDER_SPEC.provider_id == "qwen-local"
    assert QWEN_ASR_PROVIDER_SPEC.models == (
        {"id": "Qwen/Qwen3-ASR-0.6B", "default": True},
    )
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.forced_language is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.contextual_bias is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.segment_timestamps is False
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.word_timestamps is False
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.forced_alignment is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("de", "German"),
        ("en", "English"),
        ("es", "Spanish"),
        ("ru", "Russian"),
        ("Russian", "Russian"),
        (None, None),
    ],
)
def test_qwen_asr_maps_known_iso_codes_to_runtime_language_names(source, expected):
    from voiceover_pipeline.providers.qwen_asr_local import _qwen_language_name

    assert _qwen_language_name(source) == expected


def test_qwen_asr_factory_and_listing_do_not_import_optional_runtime(monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    imported: list[str] = []

    def forbidden_import(name: str):
        imported.append(name)
        raise AssertionError("registry listing must not import qwen-asr")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    registry = ASRProviderRegistry((QWEN_ASR_PROVIDER_SPEC,))

    assert registry.listing()[0]["id"] == "qwen-local"
    assert imported == []
    assert QWEN_ASR_PROVIDER_SPEC.factory().provider_id == "qwen-local"
    assert imported == []


@pytest.mark.parametrize("missing_module", ("qwen_asr", "torch"))
def test_qwen_asr_dependency_probe_has_one_redacted_install_remedy(monkeypatch, missing_module):
    from voiceover_pipeline.providers import qwen_asr_local

    def missing_runtime(name: str):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return object()

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", missing_runtime)

    health = qwen_asr_local.qwen_asr_dependency_probe()

    assert health.available is False
    assert health.remediation == "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."


def test_qwen_asr_provider_maps_typed_context_and_forced_language_without_timestamps(monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()

    class FakeRuntimeModel:
        def transcribe(self, *, audio, context, language):
            calls["transcribe"] = {
                "audio": audio,
                "context": context,
                "language": language,
            }
            return [types.SimpleNamespace(text="Проверенный текст", language="Russian")]

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["from_pretrained"] = {"model_id": model_id, **kwargs}
            return FakeRuntimeModel()

    fake_qwen_asr = types.ModuleType("qwen_asr")
    fake_qwen_asr.__version__ = "fixture-runtime"
    fake_qwen_asr.Qwen3ASRModel = FakeQwen3ASRModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    request = ASRRequest(
        audio_path="fixture.wav",
        model_id="Qwen/Qwen3-ASR-0.6B",
        language="ru",
        device="cpu",
        compute="auto",
        hints=ASRContextHints(context_text="Термины: Celery, PostgreSQL."),
    )
    result = QwenLocalASRProvider().transcribe(request)

    assert calls["from_pretrained"] == {
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "device_map": "cpu",
        "dtype": fake_torch.float32,
    }
    assert calls["transcribe"] == {
        "audio": "fixture.wav",
        "context": "Термины: Celery, PostgreSQL.",
        "language": "Russian",
    }
    assert result.transcript == "Проверенный текст"
    assert result.language == "Russian"
    assert result.execution.runtime == "qwen-asr"
    assert result.execution.runtime_version == "fixture-runtime"
    assert result.execution.resolved_device == "cpu"
    assert result.execution.resolved_compute == "float32"
    assert result.segments == ()
    assert result.words == ()
    assert result.alignment_origin is None


def test_qwen_asr_cli_fails_closed_when_selected_runtime_is_unavailable(monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers import qwen_asr_local
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    monkeypatch.setattr(
        asr_registry,
        "ASR_PROVIDER_REGISTRY",
        ASRProviderRegistry((QWEN_ASR_PROVIDER_SPEC,)),
    )
    monkeypatch.setattr(
        qwen_asr_local,
        "qwen_asr_dependency_probe",
        lambda: qwen_asr_local.ASRDependencyHealth(
            available=False,
            remediation="qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying.",
        ),
    )
    monkeypatch.setattr(
        asr_registry,
        "ASR_PROVIDER_REGISTRY",
        ASRProviderRegistry((
            replace(QWEN_ASR_PROVIDER_SPEC, dependency_probe=qwen_asr_local.qwen_asr_dependency_probe),
        )),
    )

    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "qwen-local",
            "--json",
        ]
    )
    with pytest.raises(cli.CliError) as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 10
    assert str(error.value) == (
        "qwen-asr runtime is unavailable. "
        "Install an approved qwen-asr runtime before retrying."
    )


def test_qwen_asr_doctor_uses_selected_dependency_probe(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers import qwen_asr_local
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    unavailable_spec = replace(
        QWEN_ASR_PROVIDER_SPEC,
        dependency_probe=lambda: qwen_asr_local.ASRDependencyHealth(
            available=False,
            remediation="qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying.",
        ),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: unavailable_spec)
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "/fixture/bin")
    monkeypatch.setattr(cli, "read_polza_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_openrouter_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_groq_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_xai_key", lambda: "fixture")

    args = cli.build_parser().parse_args(
        "doctor --with-asr --asr-provider qwen-local --asr-device cpu --asr-compute auto --json".split()
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.doctor_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["checks"]["asr_provider"] == {
        "ok": False,
        "provider": "qwen-local",
        "required": True,
    }
    assert "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying." in data["warnings"]
