from __future__ import annotations

import pytest

from voiceover_pipeline.artifacts import build_srt, build_timing_manifest
from voiceover_pipeline.models import ASRExecutionReceipt, ASRResult, ASRWordSpan


def _result(
    *,
    origin: str | None = "native",
    words: tuple[ASRWordSpan, ...] | None = None,
    duration_s: float | None = 2.0,
) -> ASRResult:
    return ASRResult(
        transcript="Привет мир снова",
        provider_id="fixture-asr",
        model_id="fixture-model",
        language="ru",
        duration_s=duration_s,
        execution=ASRExecutionReceipt(
            runtime="fixture-runtime",
            runtime_version="1",
            resolved_device="cpu",
            resolved_compute="float32",
        ),
        words=words
        if words is not None
        else (
            ASRWordSpan(text="Привет ", start_s=0.0, end_s=0.3),
            ASRWordSpan(text="мир ", start_s=0.4, end_s=0.8),
            ASRWordSpan(text="снова", start_s=1.0, end_s=1.4),
        ),
        alignment_origin=origin,
    )


def test_bridge_converts_native_or_forced_words_to_compatible_manifest_and_srt():
    from voiceover_pipeline.asr_timing_bridge import asr_result_to_timing

    native = asr_result_to_timing(_result(origin="native"), max_words_per_segment=2)
    forced = asr_result_to_timing(_result(origin="forced"), max_words_per_segment=2)

    assert native.provider == "fixture-asr"
    assert native.backend == "fixture-runtime"
    assert [(segment.start_ms, segment.end_ms, segment.text) for segment in native.segments] == [
        (0, 800, "Привет мир"),
        (1000, 1400, "снова"),
    ]
    assert forced.segments == native.segments
    manifest = build_timing_manifest(native, duration_ms=2000)
    assert manifest["artifact_type"] == "voiceover-timings"
    assert manifest["provider"] == "fixture-asr"
    assert manifest["segments"][0]["words"][0] == {"text": "Привет ", "start_s": 0.0, "end_s": 0.3}
    assert (
        build_srt(native)
        == "1\n00:00:00,000 --> 00:00:00,800\nПривет мир\n\n2\n00:00:01,000 --> 00:00:01,400\nснова\n"
    )


def test_bridge_rejects_text_only_missing_origin_and_invalid_word_sequences():
    from voiceover_pipeline.asr_timing_bridge import ASRTimingBridgeError, asr_result_to_timing

    text_only = ASRResult(
        transcript="Проверенный текст",
        provider_id="fixture-asr",
        model_id="fixture-model",
        execution=ASRExecutionReceipt(
            runtime="fixture", resolved_device="cpu", resolved_compute="float32"
        ),
    )
    missing_origin = ASRResult(
        transcript="Проверенный текст",
        provider_id="fixture-asr",
        model_id="fixture-model",
        duration_s=1.0,
        execution=ASRExecutionReceipt(
            runtime="fixture", resolved_device="cpu", resolved_compute="float32"
        ),
        words=(ASRWordSpan(text="Проверенный текст", start_s=0.0, end_s=0.5),),
        alignment_origin="native",
    )

    with pytest.raises(ASRTimingBridgeError, match="word timestamps"):
        asr_result_to_timing(text_only)
    with pytest.raises(ASRTimingBridgeError, match="alignment origin"):
        asr_result_to_timing(object())
    assert asr_result_to_timing(missing_origin).segments[0].text == "Проверенный текст"


def test_bridge_accepts_aligned_no_speech_without_fabricating_words():
    from voiceover_pipeline.asr_timing_bridge import asr_result_to_timing

    no_speech = ASRResult(
        transcript="",
        provider_id="fixture-asr",
        model_id="fixture-model",
        duration_s=2.0,
        execution=ASRExecutionReceipt(
            runtime="fixture", resolved_device="cpu", resolved_compute="float32"
        ),
        alignment_origin="forced",
    )

    timing = asr_result_to_timing(no_speech)

    assert timing.segments == []


def test_bridge_uses_actual_audio_duration_when_provider_omits_duration():
    from voiceover_pipeline.asr_timing_bridge import ASRTimingBridgeError, asr_result_to_timing

    missing_provider_duration = ASRResult(
        transcript="outside",
        provider_id="fixture-asr",
        model_id="fixture-model",
        execution=ASRExecutionReceipt(
            runtime="fixture", resolved_device="cpu", resolved_compute="float32"
        ),
        words=(ASRWordSpan(text="outside", start_s=0.0, end_s=3.0),),
        alignment_origin="native",
    )

    with pytest.raises(ASRTimingBridgeError, match="source duration"):
        asr_result_to_timing(missing_provider_duration, source_duration_s=2.0)


def test_legacy_timing_manifest_remains_byte_compatible_when_provider_is_absent():
    from voiceover_pipeline.asr_timing_bridge import asr_result_to_timing

    timing = asr_result_to_timing(_result())
    legacy = type(timing)(
        segments=timing.segments,
        model=timing.model,
        backend=timing.backend,
        device=timing.device,
        compute_type=timing.compute_type,
        language=timing.language,
        source_audio=timing.source_audio,
    )

    assert "provider" not in build_timing_manifest(legacy, duration_ms=2000)


def test_timing_provider_selection_and_transcribe_belong_to_transcription_service(monkeypatch):
    from pathlib import Path

    from voiceover_pipeline.config import DEFAULT_TIMING_MODEL
    from voiceover_pipeline.providers import faster_whisper, groq_whisper, xai_stt
    from voiceover_pipeline.services import transcription

    calls: list[tuple[str, dict]] = []

    class FakeTimingProvider:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def transcribe(self, **kwargs):
            calls.append(("transcribe", kwargs))
            return "timing-result"

    monkeypatch.setattr(faster_whisper, "FasterWhisperProvider", FakeTimingProvider)
    monkeypatch.setattr(groq_whisper, "GroqWhisperProvider", FakeTimingProvider)
    monkeypatch.setattr(xai_stt, "XAISttProvider", FakeTimingProvider)

    result = transcription.transcribe_timing_audio(
        audio_path=Path("part.mp3"),
        timing_provider="faster-whisper",
        model=None,
        device="cpu",
        compute_type="int8",
        language="ru",
        word_timestamps=True,
        quiet=True,
    )

    assert result == "timing-result"
    assert calls[0] == (
        "init",
        {"model_size": DEFAULT_TIMING_MODEL, "device": "cpu", "compute_type": "int8"},
    )
    assert calls[1] == (
        "transcribe",
        {
            "audio_path": Path("part.mp3"),
            "language": "ru",
            "word_timestamps": True,
            "quiet": True,
            "local_files_only": False,
        },
    )

    calls.clear()
    transcription.transcribe_timing_audio(
        audio_path=Path("part.mp3"),
        timing_provider="groq-whisper",
        model=None,
        device="cpu",
        compute_type="int8",
        language="ru",
    )
    assert calls[0] == ("init", {"model": "whisper-large-v3-turbo"})

    calls.clear()
    transcription.transcribe_timing_audio(
        audio_path=Path("part.mp3"),
        timing_provider="xai-stt",
        model="grok-custom",
        device="cpu",
        compute_type="int8",
        language="ru",
    )
    assert calls[0] == ("init", {"model": "grok-custom"})


def test_generic_asr_timing_route_writes_existing_artifacts_without_replacing_faster_whisper(
    tmp_path, monkeypatch
):
    import json

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.models import ASRCapabilities, ASRRequest
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
    from voiceover_pipeline.providers.base import ASRProvider

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    captured: list[ASRRequest] = []

    class FixtureProvider(ASRProvider):
        provider_id = "fixture-asr"

        def transcribe(self, request: ASRRequest) -> ASRResult:
            captured.append(request)
            return _result(origin="forced", duration_s=None)

    spec = ASRProviderSpec(
        provider_id="fixture-asr",
        description="Fixture ASR",
        factory=FixtureProvider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            word_timestamps=True,
            forced_alignment=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda provider_id: spec)
    monkeypatch.setattr(cli.shutil, "which", lambda command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda ffprobe, source: 2000)

    result = cli._extract_asr_timings(
        audio_path=audio,
        output_dir=tmp_path,
        prefix="fixture",
        provider_id="fixture-asr",
        model=None,
        device="cpu",
        compute="float32",
        language="ru",
    )

    manifest = json.loads((tmp_path / "fixture.timings.json").read_text(encoding="utf-8"))
    assert result == {"segment_count": 1, "total_duration_ms": 2000}
    assert captured[0].timestamp_mode == "word"
    assert manifest["provider"] == "fixture-asr"
    assert (tmp_path / "fixture.srt").exists()


def _fixture_asr_spec(factory):
    from voiceover_pipeline.models import ASRCapabilities
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec

    return ASRProviderSpec(
        provider_id="fixture-asr",
        description="Fixture ASR",
        factory=factory,
        models=(
            {"id": "default-model", "default": True},
            {"id": "other-model"},
        ),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            word_timestamps=True,
            forced_alignment=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )


def test_generic_asr_timing_service_builds_default_word_request_and_duration_in_order(
    tmp_path, monkeypatch
):
    from voiceover_pipeline.models import ASRRequest
    from voiceover_pipeline.providers.base import ASRProvider
    from voiceover_pipeline.services import transcription

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    events: list[str] = []
    captured: list[ASRRequest] = []

    class FixtureProvider(ASRProvider):
        provider_id = "fixture-asr"

        def __init__(self):
            events.append("factory")

        def transcribe(self, request: ASRRequest) -> ASRResult:
            events.append("asr")
            captured.append(request)
            return _result(origin="forced", duration_s=None)

    spec = _fixture_asr_spec(FixtureProvider)
    monkeypatch.setattr(transcription.shutil, "which", lambda command: "ffprobe")

    def record_duration(ffprobe, source):
        events.append("duration")
        assert ffprobe == "ffprobe"
        assert source == audio
        return 2000

    timing = transcription.transcribe_generic_asr_timing(
        spec,
        audio_path=audio,
        model=None,
        device="cpu",
        compute="float32",
        language="ru",
        mp3_duration_ms=record_duration,
    )

    assert events == ["factory", "asr", "duration"]
    assert captured[0].model_id == "default-model"
    assert captured[0].timestamp_mode == "word"
    assert captured[0].audio_path == audio
    assert captured[0].language == "ru"
    assert captured[0].device == "cpu"
    assert captured[0].compute == "float32"
    assert timing.provider == "fixture-asr"
    assert timing.source_audio == str(audio.resolve())


def test_generic_asr_timing_service_requires_ffprobe_after_provider_runs(tmp_path, monkeypatch):
    from voiceover_pipeline.models import ASRRequest
    from voiceover_pipeline.providers.base import ASRProvider
    from voiceover_pipeline.services import transcription

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    events: list[str] = []

    class FixtureProvider(ASRProvider):
        provider_id = "fixture-asr"

        def __init__(self):
            events.append("factory")

        def transcribe(self, request: ASRRequest) -> ASRResult:
            events.append("asr")
            return _result(origin="forced")

    spec = _fixture_asr_spec(FixtureProvider)
    monkeypatch.setattr(transcription.shutil, "which", lambda command: None)

    with pytest.raises(RuntimeError, match="FFprobe is required to validate generic ASR"):
        transcription.transcribe_generic_asr_timing(
            spec,
            audio_path=audio,
            model=None,
            device="cpu",
            compute="float32",
            language="ru",
            mp3_duration_ms=lambda ffprobe, source: 2000,
        )

    assert events == ["factory", "asr"]
