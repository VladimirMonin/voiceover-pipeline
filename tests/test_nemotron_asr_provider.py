import importlib
import json
import sys
import types
from dataclasses import replace

import pytest
from conftest import fixture_path

from voiceover_pipeline.models import ASRContextHints, ASRPhraseHint, ASRRequest
from voiceover_pipeline.providers import asr_registry
from voiceover_pipeline.providers.asr_registry import ASRProviderRegistry


def test_nemotron_asr_registry_listing_declares_native_word_timestamps_but_not_phrase_boosting():
    from voiceover_pipeline.providers.nemotron_asr_local import NEMOTRON_ASR_PROVIDER_SPEC

    capabilities = NEMOTRON_ASR_PROVIDER_SPEC.capabilities

    assert NEMOTRON_ASR_PROVIDER_SPEC.provider_id == "nemotron-local"
    assert NEMOTRON_ASR_PROVIDER_SPEC.models == (
        {"id": "nvidia/nemotron-3.5-asr-streaming-0.6b", "default": True},
    )
    assert capabilities.batch_audio is True
    assert capabilities.streaming is False
    assert capabilities.contextual_bias is False
    assert capabilities.phrase_boosting is False
    assert capabilities.segment_timestamps is True
    assert capabilities.forced_language is True
    assert capabilities.word_timestamps is True
    assert capabilities.forced_alignment is False
    assert capabilities.confidence is False


def test_nemotron_python_fallback_rejects_word_timestamps_before_loading_a_model():
    from voiceover_pipeline.providers.nemotron_asr_local import (
        NEMOTRON_ASR_MODEL_ID,
        NemotronLocalASRProvider,
    )

    provider = NemotronLocalASRProvider()

    with pytest.raises(ValueError, match="set VOICEOVER_AUDIO_CPP_BINARY"):
        provider.transcribe(
            ASRRequest(
                audio_path="fixture.wav",
                model_id=NEMOTRON_ASR_MODEL_ID,
                timestamp_mode="word",
            )
        )

    assert provider._model is None


def test_nemotron_asr_factory_and_listing_do_not_import_optional_runtime(monkeypatch):
    from voiceover_pipeline.providers.nemotron_asr_local import NEMOTRON_ASR_PROVIDER_SPEC

    imported: list[str] = []

    def forbidden_import(name: str):
        imported.append(name)
        raise AssertionError("registry listing must not import Transformers")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)

    registry = ASRProviderRegistry((NEMOTRON_ASR_PROVIDER_SPEC,))

    assert registry.listing()[0]["id"] == "nemotron-local"
    assert imported == []
    assert NEMOTRON_ASR_PROVIDER_SPEC.factory().provider_id == "nemotron-local"
    assert imported == []


@pytest.mark.parametrize("missing_module", ("transformers", "accelerate", "librosa"))
def test_nemotron_asr_dependency_probe_has_one_redacted_install_remedy(monkeypatch, missing_module):
    from voiceover_pipeline.providers import nemotron_asr_local

    def missing_runtime(name: str):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'")
        if name == "transformers":
            return types.SimpleNamespace(AutoModelForRNNT=object, AutoProcessor=object)
        return object()

    monkeypatch.setattr(nemotron_asr_local.importlib, "import_module", missing_runtime)

    health = nemotron_asr_local.nemotron_asr_dependency_probe()

    assert health.available is False
    assert health.remediation == (
        "Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying."
    )


def test_nemotron_asr_provider_maps_fixture_text_without_hints_or_timestamps(monkeypatch):
    from voiceover_pipeline.providers.nemotron_asr_local import NemotronLocalASRProvider

    calls: dict[str, object] = {}

    class FakeInputs(dict):
        def to(self, device, *, dtype):
            calls["inputs_to"] = {"device": device, "dtype": dtype}
            return self

    class FakeRuntimeModel:
        device = "fixture-device"
        dtype = "fixture-dtype"

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, **inputs):
            calls["generate"] = inputs
            return types.SimpleNamespace(sequences="fixture-token-ids")

    class FakeAutoModelForRNNT:
        @classmethod
        def from_pretrained(cls, model_id, *, device_map):
            calls["model_from_pretrained"] = {"model_id": model_id, "device_map": device_map}
            return FakeRuntimeModel()

    class FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def __call__(self, audio, *, sampling_rate, language):
            calls["processor"] = {
                "audio": audio,
                "sampling_rate": sampling_rate,
                "language": language,
            }
            return FakeInputs(input_features="fixture-features")

        def decode(self, sequences, *, skip_special_tokens):
            calls["decode"] = {"sequences": sequences, "skip_special_tokens": skip_special_tokens}
            return ["Проверенный текст"]

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["processor_from_pretrained"] = {"model_id": model_id}
            return FakeProcessor()

    def load_audio(audio_path, *, sampling_rate):
        calls["load_audio"] = {"audio_path": audio_path, "sampling_rate": sampling_rate}
        return "fixture-audio"

    fake_transformers = types.ModuleType("transformers")
    setattr(fake_transformers, "__version__", "fixture-runtime")
    setattr(fake_transformers, "AutoModelForRNNT", FakeAutoModelForRNNT)
    setattr(fake_transformers, "AutoProcessor", FakeAutoProcessor)
    fake_audio_utils = types.ModuleType("transformers.audio_utils")
    setattr(fake_audio_utils, "load_audio", load_audio)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.audio_utils", fake_audio_utils)

    request = ASRRequest(
        audio_path="fixture.wav",
        model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        language="en",
        device="cpu",
        compute="auto",
        hints=ASRContextHints(phrase_hints=(ASRPhraseHint("PostgreSQL", "strong"),)),
    )
    result = NemotronLocalASRProvider().transcribe(request)

    assert calls == {
        "processor_from_pretrained": {"model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b"},
        "model_from_pretrained": {
            "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "device_map": "cpu",
        },
        "eval": True,
        "load_audio": {"audio_path": "fixture.wav", "sampling_rate": 16000},
        "processor": {"audio": "fixture-audio", "sampling_rate": 16000, "language": "en-US"},
        "inputs_to": {"device": "fixture-device", "dtype": "fixture-dtype"},
        "generate": {"input_features": "fixture-features", "return_dict_in_generate": True},
        "decode": {"sequences": "fixture-token-ids", "skip_special_tokens": True},
    }
    assert result.transcript == "Проверенный текст"
    assert result.language == "en"
    assert result.execution.runtime == "transformers-nemotron-3.5-asr"
    assert result.execution.runtime_version == "fixture-runtime"
    assert result.execution.resolved_device == "cpu"
    assert result.execution.resolved_compute == "auto"
    assert result.segments == ()
    assert result.words == ()
    assert result.alignment_origin is None


def test_nemotron_asr_provider_rejects_a_fixture_without_text(monkeypatch):
    from voiceover_pipeline.providers.nemotron_asr_local import NemotronLocalASRProvider

    class FakeInputs(dict):
        def to(self, _device, *, dtype):
            assert dtype == "fixture-dtype"
            return self

    class FakeRuntimeModel:
        device = "fixture-device"
        dtype = "fixture-dtype"

        def eval(self):
            return self

        def generate(self, **inputs):
            assert inputs == {"input_features": "fixture-features", "return_dict_in_generate": True}
            return types.SimpleNamespace(sequences="fixture-token-ids")

    class FakeAutoModelForRNNT:
        @classmethod
        def from_pretrained(cls, model_id, *, device_map):
            assert model_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
            assert device_map == "cpu"
            return FakeRuntimeModel()

    class FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def __call__(self, _audio, *, sampling_rate, language):
            assert sampling_rate == 16000
            assert language == "auto"
            return FakeInputs(input_features="fixture-features")

        def decode(self, _sequences, *, skip_special_tokens):
            assert skip_special_tokens is True
            return object()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            assert model_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
            return FakeProcessor()

    fake_transformers = types.ModuleType("transformers")
    setattr(fake_transformers, "AutoModelForRNNT", FakeAutoModelForRNNT)
    setattr(fake_transformers, "AutoProcessor", FakeAutoProcessor)
    fake_audio_utils = types.ModuleType("transformers.audio_utils")
    setattr(fake_audio_utils, "load_audio", lambda _path, *, sampling_rate: "fixture-audio")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.audio_utils", fake_audio_utils)

    with pytest.raises(ValueError, match="nemotron ASR response has no text result"):
        NemotronLocalASRProvider().transcribe(ASRRequest(audio_path="fixture.wav"))


def test_nemotron_asr_cli_fails_closed_when_selected_runtime_is_unavailable(monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers import nemotron_asr_local
    from voiceover_pipeline.providers.nemotron_asr_local import NEMOTRON_ASR_PROVIDER_SPEC

    monkeypatch.setattr(
        nemotron_asr_local,
        "nemotron_asr_dependency_probe",
        lambda: nemotron_asr_local.ASRDependencyHealth(
            available=False,
            remediation=(
                "Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying."
            ),
        ),
    )
    unavailable_spec = replace(
        NEMOTRON_ASR_PROVIDER_SPEC,
        dependency_probe=nemotron_asr_local.nemotron_asr_dependency_probe,
    )
    monkeypatch.setattr(
        asr_registry, "ASR_PROVIDER_REGISTRY", ASRProviderRegistry((unavailable_spec,))
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: unavailable_spec)

    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "nemotron-local",
            "--json",
        ]
    )

    with pytest.raises(cli.CliError) as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 10
    assert str(error.value) == (
        "Nemotron ASR runtime is unavailable. "
        "Install an approved Hugging Face Transformers runtime before retrying."
    )


def test_nemotron_asr_doctor_uses_selected_dependency_probe(monkeypatch, capsys):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers import nemotron_asr_local
    from voiceover_pipeline.providers.nemotron_asr_local import NEMOTRON_ASR_PROVIDER_SPEC

    unavailable_spec = replace(
        NEMOTRON_ASR_PROVIDER_SPEC,
        dependency_probe=lambda: nemotron_asr_local.ASRDependencyHealth(
            available=False,
            remediation=(
                "Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying."
            ),
        ),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: unavailable_spec)
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "/fixture/bin")
    monkeypatch.setattr(cli, "read_polza_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_openrouter_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_groq_key", lambda: "fixture")
    monkeypatch.setattr(cli, "read_xai_key", lambda: "fixture")

    args = cli.build_parser().parse_args(
        "doctor --with-asr --asr-provider nemotron-local --asr-device cpu --asr-compute auto --json".split()
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.doctor_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["checks"]["asr_provider"] == {
        "ok": False,
        "provider": "nemotron-local",
        "required": True,
        "reason_code": "unavailable",
    }
    assert (
        "Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying."
        in data["warnings"]
    )
