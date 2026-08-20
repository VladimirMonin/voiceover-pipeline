from __future__ import annotations

import threading
from dataclasses import dataclass, field

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
                "transcript": "speech",
                "word_timestamps": [{"text": "▁speech", "start_s": 0.0, "end_s": 2.0}],
                "duration_s": 1.0,
            },
            "must not exceed source duration",
        ),
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
