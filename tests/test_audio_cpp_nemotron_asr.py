from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest

from voiceover_pipeline.local_runtime.contracts import (
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    RuntimeDriverHealth,
    RuntimeExecutionReceipt,
)
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.models import ASRContextHints, ASRPhraseHint, ASRRequest


@dataclass
class FixtureRuntime:
    payload: dict[str, object]
    requests: list[LocalRuntimeRequest] = field(default_factory=list)

    def execute(
        self, request: LocalRuntimeRequest, *, runtime_choice: str = "auto"
    ) -> LocalRuntimeResponse:
        assert runtime_choice == "auto"
        self.requests.append(request)
        return LocalRuntimeResponse(
            request_id=request.request_id,
            payload=self.payload,
            receipt=RuntimeExecutionReceipt(
                driver_id="audio-cpp",
                transport="subprocess-json",
                source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
            ),
        )


def _provider(payload: dict[str, object]):
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import AudioCppNemotronASRProvider

    return AudioCppNemotronASRProvider(FixtureRuntime(payload))


def test_audio_cpp_nemotron_passes_exact_language_and_maps_native_words():
    provider = _provider(
        {
            "transcript": "Привет, мир PostgreSQL",
            "language": "ru-RU",
            "duration_s": 2.0,
            "word_timestamps": [
                {"text": "При", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
                {"text": "вет", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
                {"text": ",", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
                {"text": " мир", "start_s": 0.4, "end_s": 0.6, "frame_index": 4},
                {"text": " Post", "start_s": 0.8, "end_s": 0.9, "frame_index": 8},
                {"text": "gre", "start_s": 0.8, "end_s": 1.0, "frame_index": 8},
                {"text": "SQL", "start_s": 1.0, "end_s": 1.1, "frame_index": 10},
                {"text": "", "start_s": 1.2, "end_s": 1.3, "frame_index": 12},
            ],
        }
    )

    result = provider.transcribe(
        ASRRequest(audio_path="fixture.wav", language="ru-RU", timestamp_mode="word")
    )

    wire = provider._runtime.requests[0].payload
    assert wire == {
        "audio_path": "fixture.wav",
        "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "language": "ru-RU",
        "timestamp_mode": "word",
        "context_text": None,
    }
    assert [(word.text, word.start_s, word.end_s, word.confidence) for word in result.words] == [
        ("Привет, ", 0.0, 0.1, None),
        ("мир ", 0.4, 0.6, None),
        ("PostgreSQL", 0.8, 1.1, None),
    ]
    assert result.alignment_origin == "native"
    assert result.execution.runtime == "audio-cpp"
    assert tuple(dict(entry) for entry in result.execution.raw_timestamp_entries) == tuple(
        provider._runtime.payload["word_timestamps"]
    )


def test_audio_cpp_nemotron_text_route_passes_exact_language_without_timestamp_request():
    provider = _provider({"transcript": "Checked text", "language": "en-US"})

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", language="en-US"))

    wire = provider._runtime.requests[0].payload
    assert wire == {
        "audio_path": "fixture.wav",
        "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "language": "en-US",
        "timestamp_mode": "none",
        "context_text": None,
    }
    assert result.words == ()
    assert result.alignment_origin is None


@pytest.mark.parametrize(
    "hints",
    (
        ASRContextHints(context_text="PostgreSQL Celery"),
        ASRContextHints(initial_prompt="free text is not a Nemotron prompt"),
        ASRContextHints(phrase_hints=(ASRPhraseHint("PostgreSQL", "strong"),)),
    ),
)
def test_audio_cpp_nemotron_rejects_unsupported_free_or_hotword_prompts(hints):
    provider = _provider({"transcript": "unused"})

    with pytest.raises(ValueError, match="does not expose free-text or phrase-boosting"):
        provider.transcribe(ASRRequest(audio_path="fixture.wav", hints=hints))

    assert provider._runtime.requests == []


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"transcript": "speech"}, "word_timestamps"),
        ({"transcript": "speech", "word_timestamps": []}, "returned no words"),
        (
            {
                "transcript": "different",
                "word_timestamps": [{"text": "▁speech", "start_s": 0.0, "end_s": 0.1}],
            },
            "must correspond to the transcript",
        ),
    ),
)
def test_audio_cpp_nemotron_fails_closed_for_missing_or_invalid_native_timestamp_data(
    payload, message
):
    provider = _provider(payload)

    with pytest.raises(ValueError, match=message):
        provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))


def test_nemotron_provider_factory_falls_back_to_python_without_audio_cpp_binary(monkeypatch):
    from voiceover_pipeline.providers.nemotron_asr_local import (
        NemotronLocalASRProvider,
        nemotron_asr_provider_factory,
    )

    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)

    assert isinstance(nemotron_asr_provider_factory(), NemotronLocalASRProvider)


def test_audio_cpp_nemotron_environment_route_cancels_and_releases_gpu_lease(monkeypatch):
    import voiceover_pipeline.providers.audio_cpp_nemotron_asr as audio_cpp_nemotron_asr
    from voiceover_pipeline.local_runtime.lifecycle import GPUSnapshot

    monkeypatch.setattr(audio_cpp_nemotron_asr.sys, "platform", "linux")
    import voiceover_pipeline.local_runtime.gpu_lease as gpu_lease

    @contextmanager
    def _noop_lock(_lock_path: Path) -> Iterator[None]:
        yield

    monkeypatch.setattr(gpu_lease, "_default_lock_backend", lambda: _noop_lock)
    created: list[BlockingDriver] = []

    class BlockingDriver:
        driver_id = "audio-cpp"

        def __init__(self, **_kwargs: object) -> None:
            self.started = threading.Event()
            self.cancelled_event = threading.Event()
            self.cancelled: list[str] = []
            self.request_id = ""
            created.append(self)

        def health(self) -> RuntimeDriverHealth:
            return RuntimeDriverHealth(available=True)

        def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
            self.request_id = request.request_id
            self.started.set()
            assert self.cancelled_event.wait(timeout=2)
            return LocalRuntimeResponse(
                request_id=request.request_id,
                payload={"transcript": ""},
                receipt=RuntimeExecutionReceipt(
                    driver_id=self.driver_id,
                    transport="fixture",
                    source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
                ),
            )

        def cancel(self, request_id: str) -> None:
            self.cancelled.append(request_id)
            self.cancelled_event.set()

        def close(self) -> None:
            pass

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")
    monkeypatch.setattr(audio_cpp_nemotron_asr, "AudioCppRuntimeDriver", BlockingDriver)
    provider = audio_cpp_nemotron_asr.AudioCppNemotronASRProvider.from_environment()
    assert isinstance(provider._runtime, LocalAudioRuntime)
    runtime = provider._runtime
    driver = created[0]
    assert runtime._lifecycle is not None
    lifecycle = runtime._lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_probe",
        lambda: GPUSnapshot(free_vram_mb=None, utilization_percent=None, temperature_c=None),
    )
    failures: list[BaseException] = []

    def transcribe() -> None:
        try:
            provider.transcribe(ASRRequest(audio_path="fixture.wav"))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=transcribe)
    worker.start()
    assert driver.started.wait(timeout=1)

    runtime.cancel(driver.request_id, family="nemotron-3.5-asr")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert failures == []
    assert driver.cancelled == [driver.request_id]
    assert lifecycle._leases.metadata() is None


def test_audio_cpp_nemotron_dependency_probe_checks_the_configured_binary(tmp_path, monkeypatch):
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
        AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION,
        audio_cpp_nemotron_asr_dependency_probe,
    )

    monkeypatch.setattr("voiceover_pipeline.providers.audio_cpp_nemotron_asr.sys.platform", "linux")
    missing = tmp_path / "missing-audio-cpp"
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", str(missing))

    unavailable = audio_cpp_nemotron_asr_dependency_probe()

    assert unavailable == ASRDependencyHealth(
        available=False,
        remediation=(
            f"{AUDIO_CPP_NEMOTRON_INSTALL_REMEDIATION} "
            "The pinned audio.cpp binary is not installed."
        ),
    )

    binary = tmp_path / "audio-cpp"
    binary.touch()
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", str(binary))

    assert audio_cpp_nemotron_asr_dependency_probe() == ASRDependencyHealth(
        available=True, remediation=""
    )


def _native_package(tmp_path: Path) -> tuple[Path, Path]:
    import hashlib
    import json

    package = tmp_path / "nemotron package"
    package.mkdir()
    executable = package / "audiocpp_cli.exe"
    executable.write_bytes(b"native executable")
    runtime_dll = package / "audiocpp_runtime.dll"
    runtime_dll.write_bytes(b"runtime dll")
    model = package / "models" / "nemotron" / "nemotron.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    manifest = package / "audio_cpp_dependency_closure.json"
    files = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*")
        if path.is_file() and path != manifest
    }
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": files}, sort_keys=True), encoding="utf-8"
    )
    receipt = {
        "schema_version": 1,
        "source_revision": "502b5b74bd26e9b4aed267d1776ecf131cae7215",
        "backend": "cuda",
        "compiler": "cl.exe",
        "cmake_version": "3.30.1",
        "cuda_toolkit_version": "12.6.0",
        "architecture": "x86_64",
        "build_flags": "Release",
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "model_families": ["nemotron-3.5-asr"],
    }
    (package / "build_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return executable, model


def test_windows_from_environment_admits_native_package_and_builds_native_transport(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
        NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
        AudioCppNativeCLITransport,
    )
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
        NEMOTRON_AUDIO_CPP_MODEL_ENV,
        AudioCppNemotronASRProvider,
    )

    executable, model = _native_package(tmp_path)
    monkeypatch.setattr("voiceover_pipeline.providers.audio_cpp_nemotron_asr.sys.platform", "win32")
    monkeypatch.setenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, str(executable))
    monkeypatch.setenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, str(model))

    provider = AudioCppNemotronASRProvider.from_environment()

    assert provider._runtime is not None
    driver = next(iter(provider._runtime._registry._drivers.values()))
    assert isinstance(driver._transport, AudioCppNativeCLITransport)
    assert driver._transport_name == "native-cli"
    assert driver._build_hash is not None


def test_windows_from_environment_fails_closed_without_native_executable_or_model(
    monkeypatch, tmp_path
):
    from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
        NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    )
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
        NEMOTRON_AUDIO_CPP_MODEL_ENV,
        AudioCppNemotronASRProvider,
    )

    executable, model = _native_package(tmp_path)
    monkeypatch.setattr("voiceover_pipeline.providers.audio_cpp_nemotron_asr.sys.platform", "win32")
    monkeypatch.setenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, str(executable))
    monkeypatch.delenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, raising=False)

    assert AudioCppNemotronASRProvider.from_environment()._runtime is None

    monkeypatch.setenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, str(model))
    monkeypatch.delenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, raising=False)

    assert AudioCppNemotronASRProvider.from_environment()._runtime is None


def test_windows_from_environment_fails_closed_on_invalid_package(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
        NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    )
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
        NEMOTRON_AUDIO_CPP_MODEL_ENV,
        AudioCppNemotronASRProvider,
    )

    executable, model = _native_package(tmp_path)
    (executable.parent / "audiocpp_runtime.dll").write_bytes(b"tampered")
    monkeypatch.setattr("voiceover_pipeline.providers.audio_cpp_nemotron_asr.sys.platform", "win32")
    monkeypatch.setenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, str(executable))
    monkeypatch.setenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, str(model))

    assert AudioCppNemotronASRProvider.from_environment()._runtime is None


def test_windows_dependency_probe_reports_structured_reason_codes(monkeypatch, tmp_path):
    from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
        NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    )
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
        NEMOTRON_AUDIO_CPP_MODEL_ENV,
        audio_cpp_nemotron_asr_dependency_probe,
    )

    executable, model = _native_package(tmp_path)
    monkeypatch.setattr("voiceover_pipeline.providers.audio_cpp_nemotron_asr.sys.platform", "win32")
    monkeypatch.delenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, raising=False)
    monkeypatch.delenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, raising=False)

    missing = audio_cpp_nemotron_asr_dependency_probe()
    assert missing.available is False
    assert missing.reason_code == "missing_native_executable"

    monkeypatch.setenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, str(executable))
    missing_model = audio_cpp_nemotron_asr_dependency_probe()
    assert missing_model.available is False
    assert missing_model.reason_code == "missing_model_artifact"

    monkeypatch.setenv(NEMOTRON_AUDIO_CPP_MODEL_ENV, str(model))
    healthy = audio_cpp_nemotron_asr_dependency_probe()
    assert healthy.available is True
    assert healthy.reason_code is None

    (executable.parent / "audiocpp_runtime.dll").write_bytes(b"tampered")
    invalid = audio_cpp_nemotron_asr_dependency_probe()
    assert invalid.available is False
    assert invalid.reason_code == "modified_bytes"


def test_execution_receipt_distinguishes_runtime_revision_from_model_revision():
    from voiceover_pipeline.providers.audio_cpp_nemotron_asr import _execution_receipt

    receipt = RuntimeExecutionReceipt(
        driver_id="audio-cpp",
        transport="native-cli",
        source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
    )
    execution = _execution_receipt(
        receipt,
        ASRRequest(audio_path="fixture.wav", device="cuda", compute="auto"),
        (),
    )

    assert execution.runtime == "audio-cpp"
    assert execution.runtime_version == "502b5b74bd26e9b4aed267d1776ecf131cae7215"
    assert execution.model_revision is None
    assert execution.resolved_device == "cuda"
    assert execution.resolved_compute == "auto"


def test_transcribe_uses_staged_wav_duration_from_payload():
    provider = _provider({"transcript": "Checked", "duration_s": 3.5})

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav"))

    assert result.duration_s == 3.5


def test_audio_cpp_nemotron_clamps_trailing_word_to_staged_duration():
    provider = _provider(
        {
            "transcript": "Привет",
            "duration_s": 1.0,
            "word_timestamps": [
                {"text": "▁При", "start_s": 0.0, "end_s": 0.6},
                {"text": "вет", "start_s": 0.6, "end_s": 1.031},
            ],
        }
    )

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))

    assert [(word.text, word.start_s, word.end_s) for word in result.words] == [
        ("Привет", 0.0, 1.0),
    ]
    assert result.duration_s == 1.0


def test_audio_cpp_nemotron_clamp_preserves_monotonicity_and_end_at_least_start():
    provider = _provider(
        {
            "transcript": "Привет мир",
            "duration_s": 1.0,
            "word_timestamps": [
                {"text": "▁При", "start_s": 0.0, "end_s": 0.6},
                {"text": "вет", "start_s": 0.6, "end_s": 0.9},
                {"text": "▁мир", "start_s": 1.02, "end_s": 1.031},
            ],
        }
    )

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))

    assert [(word.text, word.start_s, word.end_s) for word in result.words] == [
        ("Привет ", 0.0, 0.9),
        ("мир", 1.0, 1.0),
    ]
    for previous, current in zip(result.words, result.words[1:]):
        assert current.start_s >= previous.end_s
    assert all(word.end_s >= word.start_s for word in result.words)
