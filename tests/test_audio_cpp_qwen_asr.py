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
from voiceover_pipeline.models import ASRContextHints, ASRRequest


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
    from voiceover_pipeline.providers.audio_cpp_qwen_asr import AudioCppQwenASRProvider

    return AudioCppQwenASRProvider(FixtureRuntime(payload))


def test_audio_cpp_qwen_text_route_preserves_context_and_forced_language_without_alignment():
    provider = _provider({"transcript": "Проверенный текст", "language": "Russian"})
    request = ASRRequest(
        audio_path="fixture.wav",
        language="ru",
        hints=ASRContextHints(context_text="Celery PostgreSQL"),
    )

    result = provider.transcribe(request)

    wire = provider._runtime.requests[0].payload
    assert wire["context_text"] == "Celery PostgreSQL"
    assert wire["language"] == "Russian"
    assert wire["timestamp_mode"] == "none"
    assert "qwen3_asr" not in wire
    assert result.provider_id == "qwen-local"
    assert result.execution.runtime == "audio-cpp"
    assert result.words == ()
    assert result.alignment_origin is None


def test_audio_cpp_qwen_word_route_requires_forced_aligner_and_normalizes_words():
    provider = _provider(
        {
            "transcript": "Привет мир",
            "language": "Russian",
            "duration_s": 2.0,
            "forced_aligner_available": True,
            "words": [
                {"text": "Привет ", "start_s": 0.1, "end_s": 0.6, "confidence": 0.0},
                {"text": "мир", "start_s": 0.7, "end_s": 1.0, "confidence": 0.9},
            ],
        }
    )

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))

    wire = provider._runtime.requests[0].payload
    assert wire["timestamp_mode"] == "word"
    assert "qwen3_asr" not in wire
    assert result.alignment_origin == "forced"
    assert [(word.text, word.start_s, word.end_s, word.confidence) for word in result.words] == [
        ("Привет ", 0.1, 0.6, None),
        ("мир", 0.7, 1.0, 0.9),
    ]


def test_audio_cpp_qwen_missing_forced_aligner_fails_closed_with_remediation():
    provider = _provider(
        {
            "transcript": "Проверенный текст",
            "forced_aligner_available": False,
        }
    )

    with pytest.raises(ValueError, match="Qwen3-ForcedAligner-0.6B"):
        provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))


def test_audio_cpp_qwen_rejects_malformed_or_empty_alignment_for_speech():
    malformed = _provider(
        {
            "transcript": "Проверенный текст",
            "forced_aligner_available": True,
            "words": [{"text": "Проверенный", "start_s": "not-a-number", "end_s": 0.5}],
        }
    )
    empty = _provider(
        {
            "transcript": "Проверенный текст",
            "forced_aligner_available": True,
            "words": [],
        }
    )

    with pytest.raises(ValueError, match="word 0"):
        malformed.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))
    with pytest.raises(ValueError, match="returned no words"):
        empty.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))


def test_audio_cpp_qwen_accepts_empty_forced_alignment_for_no_speech():
    provider = _provider(
        {
            "transcript": "   ",
            "forced_aligner_available": True,
            "words": [],
        }
    )

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))

    assert result.transcript == "   "
    assert result.words == ()
    assert result.alignment_origin == "forced"


def test_audio_cpp_qwen_environment_route_composes_lifecycle_for_cancellation_and_release(
    monkeypatch,
):
    import voiceover_pipeline.providers.audio_cpp_qwen_asr as audio_cpp_qwen_asr
    from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleBlockedError, GPUSnapshot

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
    monkeypatch.setattr(audio_cpp_qwen_asr, "AudioCppRuntimeDriver", BlockingDriver)
    provider = audio_cpp_qwen_asr.AudioCppQwenASRProvider.from_environment()
    assert isinstance(provider._runtime, LocalAudioRuntime)
    runtime = provider._runtime
    driver = created[0]
    assert runtime._lifecycle is not None
    lifecycle = runtime._lifecycle
    monkeypatch.setattr(
        lifecycle,
        "_probe",
        lambda: GPUSnapshot(
            free_vram_mb=None,
            utilization_percent=None,
            temperature_c=None,
            wvm_active=True,
        ),
    )
    with pytest.raises(GPULifecycleBlockedError, match="WVM owns active GPU work"):
        provider.transcribe(ASRRequest(audio_path="fixture.wav"))
    assert not driver.started.is_set()
    monkeypatch.setattr(
        lifecycle,
        "_probe",
        lambda: GPUSnapshot(
            free_vram_mb=None,
            utilization_percent=None,
            temperature_c=None,
        ),
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

    runtime.cancel(driver.request_id, family="qwen3-asr")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert failures == []
    assert driver.cancelled == [driver.request_id]
    assert lifecycle._leases.metadata() is None


def test_audio_cpp_qwen_environment_route_uses_the_real_local_gpu_probe(monkeypatch):
    import voiceover_pipeline.providers.audio_cpp_qwen_asr as audio_cpp_qwen_asr
    from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleBlockedError, GPUSnapshot

    class FixtureDriver:
        driver_id = "audio-cpp"

        def __init__(self, **_kwargs: object) -> None:
            pass

        def health(self) -> RuntimeDriverHealth:
            return RuntimeDriverHealth(available=True)

        def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
            raise AssertionError(f"unexpected invoke: {request.request_id}")

        def cancel(self, request_id: str) -> None:
            raise AssertionError(f"unexpected cancellation: {request_id}")

        def close(self) -> None:
            pass

    def sentinel_probe() -> None:
        return None

    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_BINARY", "fixture-audio-cpp")
    monkeypatch.setattr(audio_cpp_qwen_asr, "AudioCppRuntimeDriver", FixtureDriver)
    monkeypatch.setattr(audio_cpp_qwen_asr, "probe_local_gpu_state", sentinel_probe)

    provider = audio_cpp_qwen_asr.AudioCppQwenASRProvider.from_environment()

    assert isinstance(provider._runtime, LocalAudioRuntime)
    assert provider._runtime._lifecycle is not None
    assert provider._runtime._lifecycle._probe is sentinel_probe

    lifecycle = provider._runtime._lifecycle
    lifecycle._probe = lambda: GPUSnapshot(free_vram_mb=1, utilization_percent=0, temperature_c=40)
    with pytest.raises(GPULifecycleBlockedError, match="insufficient free GPU memory"):
        provider.transcribe(ASRRequest(audio_path="fixture.wav"))

    lifecycle._probe = lambda: GPUSnapshot(
        free_vram_mb=8192, utilization_percent=100, temperature_c=40
    )
    with pytest.raises(GPULifecycleBlockedError, match="utilization"):
        provider.transcribe(ASRRequest(audio_path="fixture.wav"))


def test_audio_cpp_qwen_applies_chunk_offset_before_contract_validation():
    provider = _provider(
        {
            "transcript": "один два",
            "duration_s": 12.0,
            "forced_aligner_available": True,
            "chunk_offset_s": 10.0,
            "words": [
                {"text": "один ", "start_s": 0.0, "end_s": 0.3},
                {"text": "два", "start_s": 0.4, "end_s": 0.8},
            ],
        }
    )

    result = provider.transcribe(ASRRequest(audio_path="fixture.wav", timestamp_mode="word"))

    assert [(word.start_s, word.end_s) for word in result.words] == [(10.0, 10.3), (10.4, 10.8)]
