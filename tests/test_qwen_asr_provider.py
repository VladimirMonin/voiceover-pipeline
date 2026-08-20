import importlib
import importlib.machinery
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import fixture_path

from voiceover_pipeline.models import ASRContextHints, ASRRequest
from voiceover_pipeline.providers import asr_registry
from voiceover_pipeline.providers.asr_registry import ASRProviderRegistry


def _configure_local_qwen_storage(
    monkeypatch, tmp_path: Path, *, with_aligner: bool
) -> dict[str, Path]:
    from voiceover_pipeline.providers import qwen_asr_local

    storage_root = tmp_path / "storage"
    model_path = storage_root / "models" / "qwen-asr"
    aligner_path = storage_root / "models" / "qwen-forced-aligner"
    cache_dir = storage_root / "huggingface-cache"
    model_path.mkdir(parents=True)
    cache_dir.mkdir()
    if with_aligner:
        aligner_path.mkdir(parents=True)
    monkeypatch.setattr(qwen_asr_local, "QWEN_ASR_MODEL_PATH", model_path, raising=False)
    monkeypatch.setattr(
        qwen_asr_local, "QWEN_FORCED_ALIGNER_MODEL_PATH", aligner_path, raising=False
    )
    monkeypatch.setattr(qwen_asr_local, "QWEN_ASR_CACHE_DIR", cache_dir, raising=False)
    return {"model_path": model_path, "aligner_path": aligner_path, "cache_dir": cache_dir}


def test_qwen_asr_registry_listing_keeps_one_runtime_neutral_qwen_family_identity():
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    assert QWEN_ASR_PROVIDER_SPEC.provider_id == "qwen-local"
    assert QWEN_ASR_PROVIDER_SPEC.models == ({"id": "Qwen/Qwen3-ASR-0.6B", "default": True},)
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.forced_language is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.contextual_bias is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.segment_timestamps is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.word_timestamps is True
    assert QWEN_ASR_PROVIDER_SPEC.capabilities.forced_alignment is True
    assert "qwen-audio-cpp" not in asr_registry.ASR_PROVIDER_REGISTRY.provider_ids()


def test_qwen_asr_family_selects_audio_cpp_without_changing_the_public_provider_id(monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")

    provider = QWEN_ASR_PROVIDER_SPEC.factory()

    assert provider.provider_id == "qwen-local"
    assert getattr(provider, "_runtime", None) is not None


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


def test_qwen_asr_defers_dynet_backed_nagisa_until_japanese_tokenization(monkeypatch):
    from voiceover_pipeline.providers import qwen_asr_local

    real_nagisa = types.ModuleType("nagisa")
    setattr(real_nagisa, "tagging", lambda text: f"tagged:{text}")
    imports: list[str] = []

    monkeypatch.delitem(sys.modules, "nagisa", raising=False)
    spec = importlib.machinery.ModuleSpec("nagisa", loader=None, is_package=True)
    monkeypatch.setattr(qwen_asr_local.importlib.util, "find_spec", lambda name: spec)

    def import_module(name: str):
        imports.append(name)
        assert name == "nagisa"
        return real_nagisa

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", import_module)

    qwen_asr_local._prepare_qwen_asr_import()
    proxy = sys.modules["nagisa"]

    assert imports == []
    assert "dynet" not in sys.modules
    assert proxy.tagging("日本語") == "tagged:日本語"
    assert imports == ["nagisa"]
    assert sys.modules["nagisa"] is real_nagisa


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
    assert (
        health.remediation
        == "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
    )


def test_qwen_asr_provider_maps_typed_context_and_forced_language_without_timestamps(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_storage(monkeypatch, tmp_path, with_aligner=False)
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
        "model_id": str(storage["model_path"]),
        "cache_dir": str(storage["cache_dir"]),
        "device_map": "cpu",
        "dtype": fake_torch.float32,
        "local_files_only": True,
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


def test_qwen_asr_provider_uses_official_forced_aligner_for_word_timestamps(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_storage(monkeypatch, tmp_path, with_aligner=True)
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()

    class FakeRuntimeModel:
        def transcribe(self, **kwargs):
            calls["transcribe"] = kwargs
            return [
                types.SimpleNamespace(
                    text="Привет, мир!",
                    language="Russian",
                    time_stamps=types.SimpleNamespace(
                        items=[
                            types.SimpleNamespace(text="Привет", start_time=0.1, end_time=0.6),
                            types.SimpleNamespace(text="мир", start_time=0.7, end_time=1.0),
                        ]
                    ),
                )
            ]

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

    result = QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", language="ru", timestamp_mode="word")
    )

    assert calls["from_pretrained"] == {
        "model_id": str(storage["model_path"]),
        "cache_dir": str(storage["cache_dir"]),
        "device_map": "cpu",
        "dtype": fake_torch.float32,
        "forced_aligner": str(storage["aligner_path"]),
        "forced_aligner_kwargs": {
            "cache_dir": str(storage["cache_dir"]),
            "device_map": "cpu",
            "dtype": fake_torch.float32,
            "local_files_only": True,
        },
        "local_files_only": True,
    }
    assert calls["transcribe"] == {
        "audio": "fixture.wav",
        "context": None,
        "language": "Russian",
        "return_time_stamps": True,
    }
    assert result.alignment_origin == "forced"
    assert [(word.text, word.start_s, word.end_s) for word in result.words] == [
        ("Привет, ", 0.1, 0.6),
        ("мир!", 0.7, 1.0),
    ]
    assert "".join(word.text for word in result.words) == result.transcript


def test_qwen_asr_provider_fails_closed_when_forced_items_cannot_map_to_transcript() -> None:
    from voiceover_pipeline.providers.qwen_asr_local import _forced_words

    response = types.SimpleNamespace(
        time_stamps=types.SimpleNamespace(
            items=[types.SimpleNamespace(text="мир", start_time=0.1, end_time=0.6)]
        )
    )

    with pytest.raises(ValueError, match="cannot be mapped exactly and sequentially"):
        _forced_words(response, transcript="Привет, мир!")


def test_qwen_asr_provider_fails_closed_when_forced_item_mapping_is_ambiguous() -> None:
    from voiceover_pipeline.providers.qwen_asr_local import _forced_words

    response = types.SimpleNamespace(
        time_stamps=types.SimpleNamespace(
            items=[types.SimpleNamespace(text="—", start_time=0.1, end_time=0.6)]
        )
    )

    with pytest.raises(ValueError, match="ambiguous non-speech-only item"):
        _forced_words(response, transcript="— —")


def test_qwen_asr_provider_fails_closed_when_official_aligner_cannot_load(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    _configure_local_qwen_storage(monkeypatch, tmp_path, with_aligner=True)

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, _model_id, **_kwargs):
            raise OSError("aligner weights unavailable")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    fake_qwen_asr.Qwen3ASRModel = FakeQwen3ASRModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError, match="Qwen3-ForcedAligner-0.6B"):
        QwenLocalASRProvider().transcribe(
            ASRRequest(audio_path="fixture.wav", timestamp_mode="word")
        )


def test_qwen_asr_provider_fails_closed_before_loading_when_storage_is_unavailable(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers import qwen_asr_local
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: list[str] = []
    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "float32", object())
    setattr(fake_torch, "bfloat16", object())

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("missing local storage must be rejected before model loading")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)
    monkeypatch.setattr(
        qwen_asr_local, "QWEN_ASR_MODEL_PATH", tmp_path / "missing-model", raising=False
    )
    monkeypatch.setattr(
        qwen_asr_local,
        "QWEN_FORCED_ALIGNER_MODEL_PATH",
        tmp_path / "missing-aligner",
        raising=False,
    )
    monkeypatch.setattr(
        qwen_asr_local, "QWEN_ASR_CACHE_DIR", tmp_path / "missing-cache", raising=False
    )

    with pytest.raises(ModuleNotFoundError, match="under /media/v/storage"):
        QwenLocalASRProvider().transcribe(ASRRequest(audio_path="fixture.wav"))

    assert calls == []


def test_qwen_asr_dependency_probe_fails_closed_without_local_storage(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        qwen_asr_local, "QWEN_ASR_MODEL_PATH", tmp_path / "missing-model", raising=False
    )
    monkeypatch.setattr(
        qwen_asr_local, "QWEN_ASR_CACHE_DIR", tmp_path / "missing-cache", raising=False
    )

    health = qwen_asr_local.qwen_asr_dependency_probe()

    assert health.available is False
    assert "under /media/v/storage" in health.remediation


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
        ASRProviderRegistry(
            (
                replace(
                    QWEN_ASR_PROVIDER_SPEC,
                    dependency_probe=qwen_asr_local.qwen_asr_dependency_probe,
                ),
            )
        ),
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
        "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
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
        "reason_code": "unavailable",
    }
    assert (
        "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
        in data["warnings"]
    )
