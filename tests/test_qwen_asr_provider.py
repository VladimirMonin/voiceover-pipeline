import importlib
import importlib.machinery
import json
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import fixture_path

from voiceover_pipeline.models import ASRContextHints, ASRRequest
from voiceover_pipeline.providers import asr_registry
from voiceover_pipeline.providers.asr_registry import ASRProviderRegistry

QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
QWEN_ASR_LARGE_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"


def _configure_local_qwen_models_root(
    monkeypatch,
    tmp_path: Path,
    *,
    models: tuple[str, ...] = (QWEN_ASR_MODEL_ID,),
    with_aligner: bool = False,
) -> dict[str, Path]:
    """Point the provider at a temporary models root with the named weights."""
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_MODEL_DIRECTORY_NAMES

    models_root = tmp_path / "storage"
    models_dir = models_root / "models"
    cache_dir = models_root / "huggingface-cache"
    models_dir.mkdir(parents=True)
    cache_dir.mkdir()
    created = {"models_root": models_root, "models_dir": models_dir, "cache_dir": cache_dir}
    for model_id in models:
        model_path = models_dir / QWEN_ASR_MODEL_DIRECTORY_NAMES[model_id]
        model_path.mkdir(parents=True)
        created[model_id] = model_path
    aligner_path = models_dir / "Qwen3-ForcedAligner-0.6B"
    if with_aligner:
        aligner_path.mkdir(parents=True)
    created["aligner_path"] = aligner_path
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", str(models_root))
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_CACHE_DIR", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_REVISION", raising=False)
    return created


def _install_fake_qwen_runtime(
    monkeypatch, *, calls: dict[str, object], on_load: Callable[[], None] | None = None
) -> None:
    """Install a fake runtime whose only observable effect is one load call.

    ``on_load`` runs inside ``from_pretrained`` before the model is returned, so a
    test can retarget a selected asset alias at the exact moment of the race.
    """

    class FakeRuntimeModel:
        def transcribe(self, **_kwargs):
            return [types.SimpleNamespace(text="Проверенный текст", language="Russian")]

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            calls["loads"] = int(calls.get("loads", 0)) + 1
            if on_load is not None:
                on_load()
            return FakeRuntimeModel()

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    fake_qwen_asr = types.ModuleType("qwen_asr")
    fake_qwen_asr.__version__ = "fixture-runtime"
    fake_qwen_asr.Qwen3ASRModel = FakeQwen3ASRModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)


def _replace_with_symlink(link_path: Path, target: Path) -> None:
    """Make ``link_path`` a symlink to ``target``, replacing a plain directory."""
    if link_path.is_dir() and not link_path.is_symlink():
        link_path.rmdir()
    else:
        link_path.unlink(missing_ok=True)
    link_path.symlink_to(target)


def _snapshot_path(tmp_path: Path, model_id: str, revision: str) -> Path:
    """Build the Hugging Face snapshot directory a model id and revision describe."""
    repository = "models--" + model_id.replace("/", "--")
    snapshot = tmp_path / "hf" / repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


def test_qwen_asr_registry_listing_exposes_both_local_sizes_with_one_default():
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    assert QWEN_ASR_PROVIDER_SPEC.provider_id == "qwen-local"
    assert QWEN_ASR_PROVIDER_SPEC.models == (
        {"id": "Qwen/Qwen3-ASR-0.6B", "default": True},
        {"id": "Qwen/Qwen3-ASR-1.7B"},
    )
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
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path)
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
        "model_id": str(storage[QWEN_ASR_MODEL_ID]),
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
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path, with_aligner=True)
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
        "model_id": str(storage[QWEN_ASR_MODEL_ID]),
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
    _configure_local_qwen_models_root(monkeypatch, tmp_path, with_aligner=True)

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
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_STORAGE_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: list[str] = []
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", str(tmp_path / "missing-root"))
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_CACHE_DIR", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_REVISION", raising=False)

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("missing local storage must be rejected before model loading")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError) as error:
        QwenLocalASRProvider().transcribe(ASRRequest(audio_path="fixture.wav"))

    assert str(error.value) == QWEN_ASR_STORAGE_REMEDIATION
    assert calls == []


def test_qwen_asr_dependency_probe_fails_closed_without_local_storage(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", lambda _name: object())
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", str(tmp_path / "missing-root"))
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_CACHE_DIR", raising=False)

    health = qwen_asr_local.qwen_asr_dependency_probe()

    assert health.available is False
    assert health.remediation == qwen_asr_local.QWEN_ASR_STORAGE_REMEDIATION


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


def test_qwen_asr_selected_large_model_loads_only_its_own_weights(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    _install_fake_qwen_runtime(monkeypatch, calls=calls)

    result = QwenLocalASRProvider().transcribe(
        ASRRequest(
            audio_path="fixture.wav",
            model_id=QWEN_ASR_LARGE_MODEL_ID,
            language="ru",
        )
    )

    assert calls["model_id"] == str(storage[QWEN_ASR_LARGE_MODEL_ID])
    assert calls["model_id"] != str(storage[QWEN_ASR_MODEL_ID])
    assert result.model_id == QWEN_ASR_LARGE_MODEL_ID
    assert result.execution.model_revision is None


def test_qwen_asr_reused_provider_loads_each_explicitly_selected_size(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    _install_fake_qwen_runtime(monkeypatch, calls=calls)
    provider = QwenLocalASRProvider()

    small = provider.transcribe(ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_MODEL_ID))
    assert calls["model_id"] == str(storage[QWEN_ASR_MODEL_ID])
    large = provider.transcribe(
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
    )

    assert small.model_id == QWEN_ASR_MODEL_ID
    assert large.model_id == QWEN_ASR_LARGE_MODEL_ID
    assert calls["model_id"] == str(storage[QWEN_ASR_LARGE_MODEL_ID])
    assert small.execution.model_path == str(storage[QWEN_ASR_MODEL_ID].resolve())
    assert large.execution.model_path == str(storage[QWEN_ASR_LARGE_MODEL_ID].resolve())


def test_qwen_asr_missing_selected_model_fails_closed_before_runtime_work(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_STORAGE_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: list[str] = []
    _configure_local_qwen_models_root(monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID,))

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("the 0.6B weights must never serve a 1.7B selection")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError) as error:
        QwenLocalASRProvider().transcribe(
            ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
        )

    assert str(error.value) == QWEN_ASR_STORAGE_REMEDIATION
    assert calls == []


def test_qwen_asr_unknown_model_id_fails_closed_instead_of_loading_another_size(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: list[str] = []
    _configure_local_qwen_models_root(monkeypatch, tmp_path)

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("an unknown model id must not resolve to the 0.6B weights")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ValueError, match="Unknown local Qwen ASR model"):
        QwenLocalASRProvider().transcribe(
            ASRRequest(audio_path="fixture.wav", model_id="Qwen/Qwen3-ASR-9.9B")
        )

    assert calls == []


def test_qwen_asr_revision_pin_loads_the_matching_hugging_face_snapshot(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_LARGE_MODEL_ID,)
    )
    snapshot = tmp_path / "hf" / "models--Qwen--Qwen3-ASR-1.7B" / "snapshots" / QWEN_ASR_REVISION
    snapshot.mkdir(parents=True)
    weights_directory = storage[QWEN_ASR_LARGE_MODEL_ID]
    weights_directory.rmdir()
    weights_directory.symlink_to(snapshot)
    _install_fake_qwen_runtime(monkeypatch, calls=calls)
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_REVISION", QWEN_ASR_REVISION)

    result = QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
    )

    assert Path(str(calls["model_id"])).resolve() == snapshot
    assert result.execution.model_revision == QWEN_ASR_REVISION


def test_qwen_asr_revision_pin_fails_closed_without_a_declared_snapshot(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_REVISION_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: list[str] = []
    _configure_local_qwen_models_root(monkeypatch, tmp_path)
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_REVISION", QWEN_ASR_REVISION)

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("an unprovable revision pin must fail before loading")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError) as error:
        QwenLocalASRProvider().transcribe(ASRRequest(audio_path="fixture.wav"))

    assert str(error.value) == QWEN_ASR_REVISION_REMEDIATION
    assert calls == []


def test_qwen_asr_legacy_asset_root_still_resolves_with_one_warning(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    legacy_root = tmp_path / "legacy"
    (legacy_root / "models" / "Qwen3-ASR-0.6B").mkdir(parents=True)
    (legacy_root / "huggingface-cache").mkdir()
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_CACHE_DIR", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_REVISION", raising=False)
    monkeypatch.setattr(qwen_asr_local, "QWEN_ASR_LEGACY_STORAGE_ROOT", legacy_root)
    monkeypatch.setattr(qwen_asr_local, "_legacy_storage_warning_emitted", False)

    with pytest.warns(qwen_asr_local.QwenASRLegacyStorageWarning):
        assets = qwen_asr_local.admit_qwen_asr_local_assets(QWEN_ASR_MODEL_ID)

    assert assets.model_path == legacy_root / "models" / "Qwen3-ASR-0.6B"
    assert assets.cache_dir == legacy_root / "huggingface-cache"
    assert assets.legacy_storage_root is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            qwen_asr_local.admit_qwen_asr_local_assets(QWEN_ASR_MODEL_ID).model_id
            == QWEN_ASR_MODEL_ID
        )


def test_qwen_asr_settings_file_resolves_assets_and_environment_wins(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    configured_root = tmp_path / "configured"
    (configured_root / "models" / "Qwen3-ASR-0.6B").mkdir(parents=True)
    (configured_root / "huggingface-cache").mkdir()
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    (workdir / "settings.toml").write_text(
        f'[asr.qwen_local]\nmodels_root = "{configured_root}"\nrevision = "{QWEN_ASR_REVISION}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_CACHE_DIR", raising=False)
    monkeypatch.delenv("VOICEOVER_QWEN_ASR_REVISION", raising=False)

    configured = qwen_asr_local.resolve_qwen_asr_local_assets(QWEN_ASR_MODEL_ID)

    assert configured.model_path == configured_root / "models" / "Qwen3-ASR-0.6B"
    assert configured.cache_dir == configured_root / "huggingface-cache"
    assert configured.revision == QWEN_ASR_REVISION
    assert configured.legacy_storage_root is False

    environment_root = tmp_path / "environment"
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", str(environment_root))
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_REVISION", "environment-revision")

    overridden = qwen_asr_local.resolve_qwen_asr_local_assets(QWEN_ASR_MODEL_ID)

    assert overridden.model_path == environment_root / "models" / "Qwen3-ASR-0.6B"
    assert overridden.revision == "environment-revision"


def test_qwen_asr_settings_reject_a_non_string_asset_value(tmp_path):
    from voiceover_pipeline import settings

    path = tmp_path / "settings.toml"
    path.write_text("[asr.qwen_local]\nmodels_root = 3\n", encoding="utf-8")

    with pytest.raises(settings.SettingsError, match="models_root must be a non-empty string"):
        settings.load_qwen_asr_local_settings(path)


def test_qwen_asr_dependency_probe_reports_an_installed_size_without_loading_it(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers import qwen_asr_local

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", lambda _name: object())
    _configure_local_qwen_models_root(monkeypatch, tmp_path)

    health = qwen_asr_local.qwen_asr_dependency_probe()

    assert health.available is True
    assert health.remediation == ""


def test_qwen_asr_dependency_probe_accepts_a_large_model_only_install(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    monkeypatch.setattr(qwen_asr_local.importlib, "import_module", lambda _name: object())
    _configure_local_qwen_models_root(monkeypatch, tmp_path, models=(QWEN_ASR_LARGE_MODEL_ID,))

    health = qwen_asr_local.qwen_asr_dependency_probe()

    assert health.available is True
    assert health.remediation == ""


def test_qwen_asr_cli_admits_the_large_model_selection(monkeypatch, tmp_path, capsys):
    import voiceover_pipeline.cli as cli

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    _install_fake_qwen_runtime(monkeypatch, calls=calls)
    monkeypatch.setattr(
        cli,
        "transcribe_prerecorded_long_form",
        lambda provider, request: provider.transcribe(request),
    )

    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "qwen-local",
            "--model",
            QWEN_ASR_LARGE_MODEL_ID,
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["model"] == QWEN_ASR_LARGE_MODEL_ID
    assert calls["model_id"] == str(storage[QWEN_ASR_LARGE_MODEL_ID])


def test_qwen_asr_cli_refuses_the_large_model_on_the_audio_cpp_runtime(monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers.audio_cpp_qwen_asr import (
        AUDIO_CPP_QWEN_MODEL_REMEDIATION,
    )

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE", "fixture-audio-cpp")
    monkeypatch.setattr(
        cli,
        "transcribe_prerecorded_long_form",
        lambda provider, request: provider.transcribe(request),
    )
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "qwen-local",
            "--model",
            QWEN_ASR_LARGE_MODEL_ID,
            "--device",
            "cuda",
            "--runtime",
            "audio-cpp",
            "--json",
        ]
    )

    with pytest.raises(cli.CliError) as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 10
    assert AUDIO_CPP_QWEN_MODEL_REMEDIATION in str(error.value)
    assert QWEN_ASR_LARGE_MODEL_ID in str(error.value)


def test_qwen_asr_refuses_a_selected_model_directory_aliased_to_another_size(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_IDENTITY_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: list[str] = []
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID,))
    _replace_with_symlink(storage["models_dir"] / "Qwen3-ASR-1.7B", storage[QWEN_ASR_MODEL_ID])

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("an aliased 0.6B directory must never serve a 1.7B selection")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError) as error:
        QwenLocalASRProvider().transcribe(
            ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
        )

    assert str(error.value) == QWEN_ASR_IDENTITY_REMEDIATION
    assert calls == []


def test_qwen_asr_refuses_a_selected_model_aliased_to_another_size_snapshot(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_IDENTITY_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: list[str] = []
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID,))
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_MODEL_ID, QWEN_ASR_REVISION)
    _replace_with_symlink(storage["models_dir"] / "Qwen3-ASR-1.7B", snapshot)

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("from_pretrained")
            raise AssertionError("a 0.6B snapshot must never serve a 1.7B selection")

    fake_qwen_asr = types.ModuleType("qwen_asr")
    setattr(fake_qwen_asr, "Qwen3ASRModel", FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    with pytest.raises(ModuleNotFoundError) as error:
        QwenLocalASRProvider().transcribe(
            ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
        )

    assert str(error.value) == QWEN_ASR_IDENTITY_REMEDIATION
    assert calls == []


def test_qwen_asr_accepts_a_correctly_identified_snapshot_and_records_observed_revision(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_LARGE_MODEL_ID,)
    )
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_LARGE_MODEL_ID, QWEN_ASR_REVISION)
    _replace_with_symlink(storage[QWEN_ASR_LARGE_MODEL_ID], snapshot)
    _install_fake_qwen_runtime(monkeypatch, calls=calls)

    result = QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
    )

    assert Path(str(calls["model_id"])).resolve() == snapshot.resolve()
    assert result.execution.model_revision == QWEN_ASR_REVISION
    assert Path(result.execution.model_path or "") == snapshot.resolve()


def test_qwen_asr_revision_remediation_points_at_the_selected_models_directory():
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_REVISION_REMEDIATION

    assert "models/<selected-name>" in QWEN_ASR_REVISION_REMEDIATION


def test_qwen_asr_rechecks_asset_configuration_before_reusing_a_loaded_model(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_ASSET_CHANGE_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path)
    _install_fake_qwen_runtime(monkeypatch, calls=calls)
    provider = QwenLocalASRProvider()
    request = ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_MODEL_ID)

    provider.transcribe(request)
    assert calls["model_id"] == str(storage[QWEN_ASR_MODEL_ID])

    other_root = tmp_path / "other"
    (other_root / "models" / "Qwen3-ASR-0.6B").mkdir(parents=True)
    (other_root / "huggingface-cache").mkdir()
    monkeypatch.setenv("VOICEOVER_QWEN_ASR_MODELS_ROOT", str(other_root))

    with pytest.raises(ModuleNotFoundError) as error:
        provider.transcribe(request)

    assert str(error.value) == QWEN_ASR_ASSET_CHANGE_REMEDIATION
    assert calls["model_id"] == str(storage[QWEN_ASR_MODEL_ID])


def test_qwen_asr_load_pins_the_admitted_model_target_against_an_alias_retarget(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_LARGE_MODEL_ID, QWEN_ASR_REVISION)
    selected_alias = storage["models_dir"] / "Qwen3-ASR-1.7B"
    _replace_with_symlink(selected_alias, snapshot)

    def retarget_selected_alias() -> None:
        _replace_with_symlink(selected_alias, storage[QWEN_ASR_MODEL_ID])

    _install_fake_qwen_runtime(monkeypatch, calls=calls, on_load=retarget_selected_alias)

    result = QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
    )

    assert calls["model_id"] == str(snapshot.resolve())
    assert Path(str(calls["model_id"])).resolve() == snapshot.resolve()
    assert result.execution.model_path == str(snapshot.resolve())
    assert result.execution.model_revision == QWEN_ASR_REVISION


def test_qwen_asr_load_uses_target_verified_before_admission_returns(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_LARGE_MODEL_ID, QWEN_ASR_REVISION)
    selected_alias = storage["models_dir"] / "Qwen3-ASR-1.7B"
    _replace_with_symlink(selected_alias, snapshot)
    actual_admit = qwen_asr_local.admit_qwen_asr_local_assets

    def retarget_after_admission(model_id, **kwargs):
        admitted = actual_admit(model_id, **kwargs)
        _replace_with_symlink(selected_alias, storage[QWEN_ASR_MODEL_ID])
        return admitted

    monkeypatch.setattr(qwen_asr_local, "admit_qwen_asr_local_assets", retarget_after_admission)
    _install_fake_qwen_runtime(monkeypatch, calls=calls)

    result = qwen_asr_local.QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)
    )

    assert calls["model_id"] == str(snapshot.resolve())
    assert result.execution.model_path == str(snapshot.resolve())
    assert result.execution.model_revision == QWEN_ASR_REVISION


def test_qwen_asr_rejects_a_retargeted_alias_before_reusing_the_loaded_model(monkeypatch, tmp_path):
    from voiceover_pipeline.providers.qwen_asr_local import (
        QWEN_ASR_ASSET_CHANGE_REMEDIATION,
        QwenLocalASRProvider,
    )

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_LARGE_MODEL_ID, QWEN_ASR_REVISION)
    selected_alias = storage["models_dir"] / "Qwen3-ASR-1.7B"
    _replace_with_symlink(selected_alias, snapshot)
    _install_fake_qwen_runtime(
        monkeypatch,
        calls=calls,
        on_load=lambda: _replace_with_symlink(selected_alias, storage[QWEN_ASR_MODEL_ID]),
    )
    provider = QwenLocalASRProvider()
    request = ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID)

    provider.transcribe(request)
    assert calls["loads"] == 1

    with pytest.raises(ModuleNotFoundError) as error:
        provider.transcribe(request)

    assert str(error.value) == QWEN_ASR_ASSET_CHANGE_REMEDIATION
    assert calls["loads"] == 1


def test_qwen_asr_word_load_pins_admitted_aligner_and_cache_targets_against_alias_retarget(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.providers.qwen_asr_local import QwenLocalASRProvider

    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(monkeypatch, tmp_path, with_aligner=True)
    aligner_snapshot = _snapshot_path(tmp_path, "Qwen/Qwen3-ForcedAligner-0.6B", QWEN_ASR_REVISION)
    aligner_alias = storage["aligner_path"]
    _replace_with_symlink(aligner_alias, aligner_snapshot)
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    decoy_cache = tmp_path / "decoy-cache"
    decoy_cache.mkdir()
    cache_alias = storage["cache_dir"]
    _replace_with_symlink(cache_alias, real_cache)
    decoy_aligner = tmp_path / "decoy-aligner"
    decoy_aligner.mkdir()

    class FakeWordRuntimeModel:
        def transcribe(self, **_kwargs):
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
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            calls["loads"] = int(calls.get("loads", 0)) + 1
            _replace_with_symlink(aligner_alias, decoy_aligner)
            _replace_with_symlink(cache_alias, decoy_cache)
            return FakeWordRuntimeModel()

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    fake_qwen_asr = types.ModuleType("qwen_asr")
    fake_qwen_asr.__version__ = "fixture-runtime"
    fake_qwen_asr.Qwen3ASRModel = FakeQwen3ASRModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)

    QwenLocalASRProvider().transcribe(
        ASRRequest(audio_path="fixture.wav", language="ru", timestamp_mode="word")
    )

    assert calls["kwargs"]["forced_aligner"] == str(aligner_snapshot.resolve())
    assert calls["kwargs"]["cache_dir"] == str(real_cache.resolve())
    assert calls["kwargs"]["forced_aligner_kwargs"]["cache_dir"] == str(real_cache.resolve())


def test_qwen_asr_cli_explicit_python_runtime_ignores_a_configured_audio_cpp_runtime(
    monkeypatch, tmp_path, capsys
):
    import sqlite3

    import voiceover_pipeline.cli as cli

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")
    calls: dict[str, object] = {}
    storage = _configure_local_qwen_models_root(
        monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID, QWEN_ASR_LARGE_MODEL_ID)
    )
    snapshot = _snapshot_path(tmp_path, QWEN_ASR_LARGE_MODEL_ID, QWEN_ASR_REVISION)
    _replace_with_symlink(storage[QWEN_ASR_LARGE_MODEL_ID], snapshot)
    _install_fake_qwen_runtime(monkeypatch, calls=calls)
    monkeypatch.setattr(
        cli,
        "transcribe_prerecorded_long_form",
        lambda provider, request: provider.transcribe(request),
    )

    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "qwen-local",
            "--model",
            QWEN_ASR_LARGE_MODEL_ID,
            "--language",
            "ru",
            "--runtime",
            "python",
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    data = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == 0
    assert data["model"] == QWEN_ASR_LARGE_MODEL_ID
    assert data["execution"]["runtime"] == "qwen-asr"
    assert Path(data["execution"]["model_path"]) == snapshot.resolve()
    assert Path(str(calls["model_id"])).resolve() == snapshot.resolve()

    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT config_snapshot FROM runs").fetchone()
    finally:
        connection.close()
    snapshot_row = json.loads(row["config_snapshot"])
    assert snapshot_row["model"] == QWEN_ASR_LARGE_MODEL_ID
    assert snapshot_row["model_revision"] == QWEN_ASR_REVISION
    assert Path(snapshot_row["model_path"]) == snapshot.resolve()


def test_qwen_asr_cli_explicit_audio_cpp_runtime_refuses_the_large_model_without_a_package(
    monkeypatch,
):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.providers.audio_cpp_qwen_asr import (
        AUDIO_CPP_QWEN_MODEL_REMEDIATION,
    )

    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE", raising=False)

    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(fixture_path("smoke_test.md")),
            "--provider",
            "qwen-local",
            "--model",
            QWEN_ASR_LARGE_MODEL_ID,
            "--device",
            "cuda",
            "--runtime",
            "audio-cpp",
            "--json",
        ]
    )

    with pytest.raises(cli.CliError) as error:
        cli.transcribe_cmd(args)

    assert error.value.code == 10
    assert AUDIO_CPP_QWEN_MODEL_REMEDIATION in str(error.value)
    assert QWEN_ASR_LARGE_MODEL_ID in str(error.value)


def test_qwen_asr_auto_runtime_keeps_the_environment_based_audio_cpp_selection(monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC
    from voiceover_pipeline.services import transcription

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")

    provider = transcription.resolve_asr_provider(
        QWEN_ASR_PROVIDER_SPEC,
        ASRRequest(audio_path="fixture.wav", model_id=QWEN_ASR_MODEL_ID, runtime_choice="auto"),
    )

    assert provider.provider_id == "qwen-local"
    assert getattr(provider, "_runtime", None) is not None


def test_qwen_asr_auto_refuses_large_model_before_audio_cpp_probe(monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC
    from voiceover_pipeline.services import transcription

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")
    spec = replace(
        QWEN_ASR_PROVIDER_SPEC,
        dependency_probe=lambda: pytest.fail("unsupported model must fail before runtime probe"),
    )

    with pytest.raises(transcription.ASRDependencyUnavailableError, match="pinned"):
        transcription.resolve_asr_provider(
            spec,
            ASRRequest(
                audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID, runtime_choice="auto"
            ),
        )


def test_qwen_asr_auto_preflights_the_selected_python_model(monkeypatch, tmp_path):
    from voiceover_pipeline.providers import qwen_asr_local
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth
    from voiceover_pipeline.services import transcription

    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", raising=False)
    _configure_local_qwen_models_root(monkeypatch, tmp_path, models=(QWEN_ASR_MODEL_ID,))
    monkeypatch.setattr(qwen_asr_local, "_qwen_python_runtime_health", lambda: None)
    spec = replace(
        qwen_asr_local.QWEN_ASR_PROVIDER_SPEC,
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )

    with pytest.raises(transcription.ASRDependencyUnavailableError, match="weights or cache"):
        transcription.resolve_asr_provider(
            spec,
            ASRRequest(
                audio_path="fixture.wav", model_id=QWEN_ASR_LARGE_MODEL_ID, runtime_choice="auto"
            ),
        )
