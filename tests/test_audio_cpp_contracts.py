from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_args

import pytest

from voiceover_pipeline.local_runtime.contracts import (
    LocalASRRequest,
    LocalASRResponse,
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    LocalTTSRequest,
    LocalTTSResponse,
    OmniVoiceMode,
    OmniVoiceRequest,
    RuntimeDriverHealth,
    RuntimeProtocolError,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry


@dataclass
class FixtureDriver:
    driver_id: str
    available: bool = True
    requests: list[LocalRuntimeRequest] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    closed: int = 0

    def health(self) -> RuntimeDriverHealth:
        return RuntimeDriverHealth(
            available=self.available, remediation="install fixture" if not self.available else ""
        )

    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
        self.requests.append(request)
        return LocalRuntimeResponse(
            request_id=request.request_id, payload={"driver": self.driver_id}
        )

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)

    def close(self) -> None:
        self.closed += 1


def _request(*, family: str = "qwen3-asr") -> LocalRuntimeRequest:
    return LocalRuntimeRequest(
        request_id="fixture-request",
        operation="asr",
        family=family,
        provider_id="qwen-local",
        payload={"model_id": "Qwen/Qwen3-ASR-0.6B", "timestamp_mode": "word"},
    )


def test_wire_contract_is_runtime_neutral_and_freezes_payload():
    request = _request()

    assert request.to_wire() == {
        "schema_version": 1,
        "request_id": "fixture-request",
        "operation": "asr",
        "family": "qwen3-asr",
        "provider_id": "qwen-local",
        "payload": {"model_id": "Qwen/Qwen3-ASR-0.6B", "timestamp_mode": "word"},
    }
    assert "audio_cpp" not in request.to_wire()
    with pytest.raises(TypeError):
        request.payload["model_id"] = "changed"  # type: ignore[index]


def test_local_asr_and_tts_typed_conversions_accept_valid_payloads_and_reject_malformed_ones():
    asr_request = LocalASRRequest(
        request_id="asr-request",
        family="qwen3-asr",
        provider_id="qwen-local",
        audio_path="fixture.wav",
        model_id="Qwen/Qwen3-ASR-0.6B",
        language="ru",
        timestamp_mode="word",
    )
    tts_request = LocalTTSRequest(
        request_id="tts-request",
        family="qwen3-tts",
        provider_id="qwen-local",
        text="fixture text",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        voice="fixture-voice",
    )

    assert asr_request.to_runtime_request().operation == "asr"
    assert asr_request.to_runtime_request().payload["timestamp_mode"] == "word"
    assert tts_request.to_runtime_request().operation == "tts"
    assert tts_request.to_runtime_request().payload["voice"] == "fixture-voice"
    asr_response = LocalASRResponse.from_runtime_response(
        LocalRuntimeResponse(
            request_id="asr-request", payload={"transcript": "fixture text", "language": "ru"}
        )
    )
    tts_response = LocalTTSResponse.from_runtime_response(
        LocalRuntimeResponse(request_id="tts-request", payload={"audio_path": "fixture.wav"})
    )
    assert asr_response.transcript == "fixture text"
    assert asr_response.language == "ru"
    assert tts_response.audio_path == "fixture.wav"
    with pytest.raises(RuntimeProtocolError, match="no transcript"):
        LocalASRResponse.from_runtime_response(LocalRuntimeResponse(request_id="asr-request"))
    with pytest.raises(RuntimeProtocolError, match="language must be a string"):
        LocalASRResponse.from_runtime_response(
            LocalRuntimeResponse(
                request_id="asr-request", payload={"transcript": "fixture", "language": 1}
            )
        )
    with pytest.raises(RuntimeProtocolError, match="no audio_path"):
        LocalTTSResponse.from_runtime_response(
            LocalRuntimeResponse(request_id="tts-request", payload={"audio_path": ""})
        )


def test_typed_asr_runtime_choice_is_routing_only_and_explicit_audio_cpp_does_not_fallback():
    request = LocalASRRequest(
        request_id="asr-request",
        family="qwen3-asr",
        provider_id="qwen-local",
        audio_path="fixture.wav",
        model_id="Qwen/Qwen3-ASR-0.6B",
        runtime_choice="audio-cpp",
    )
    python = FixtureDriver("python")
    audio_cpp = FixtureDriver("audio-cpp", available=False)
    runtime = LocalAudioRuntime(LocalRuntimeRegistry((python, audio_cpp)))

    assert request.runtime_choice == "audio-cpp"
    assert "runtime_choice" not in request.to_runtime_request().payload
    with pytest.raises(RuntimeUnavailableError, match="audio-cpp"):
        runtime.execute(request.to_runtime_request(), runtime_choice=request.runtime_choice)

    assert python.requests == []
    assert audio_cpp.requests == []


def test_omnivoice_mode_contract_is_family_safe_and_mode_specific():
    assert get_args(OmniVoiceMode) == ("fixed-style", "clone", "design")
    fixed = OmniVoiceRequest(mode="fixed-style", style_condition="female")
    clone = OmniVoiceRequest(
        mode="clone", reference_audio_path="fixture.wav", reference_text="reference transcript"
    )
    design = OmniVoiceRequest(mode="design", instruction="warm and clear")

    assert fixed.mode == "fixed-style"
    assert clone.reference_text == "reference transcript"
    assert design.instruction == "warm and clear"
    with pytest.raises(ValueError, match="OmniVoice mode"):
        OmniVoiceRequest(mode="voice-clone")


@pytest.mark.parametrize(
    ("mode", "kwargs", "message"),
    (
        ("clone", {}, "reference audio"),
        ("clone", {"reference_audio_path": "fixture.wav"}, "reference text"),
        ("design", {}, "instruction"),
        ("fixed-style", {"reference_audio_path": "fixture.wav"}, "fixed-style"),
        ("fixed-style", {"reference_text": "reference transcript"}, "fixed-style"),
        ("fixed-style", {"instruction": "design instruction"}, "fixed-style"),
    ),
)
def test_omnivoice_mode_contract_rejects_missing_or_cross_mode_fields(mode, kwargs, message):
    with pytest.raises(ValueError, match=message):
        OmniVoiceRequest(mode=mode, **kwargs)


def _omnivoice_tts_request(**overrides) -> LocalTTSRequest:
    fields = {
        "request_id": "tts-request",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "text": "fixture text",
        "model_id": "audio-cpp/omnivoice-q8_0",
        "language": "ru",
        "seed": 1,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "text_chunk_size": 420,
    }
    fields.update(overrides)
    return LocalTTSRequest(**fields)


def test_local_tts_request_carries_omnivoice_mode_fields_without_qwen_mode_collision():
    fixed = _omnivoice_tts_request(omnivoice_mode="fixed-style", style_condition="female")
    clone = _omnivoice_tts_request(
        omnivoice_mode="clone",
        reference_audio_path="fixture.wav",
        reference_text="reference transcript",
    )
    design = _omnivoice_tts_request(omnivoice_mode="design", design_instruction="warm and clear")

    assert fixed.to_runtime_request().payload["omnivoice_mode"] == "fixed-style"
    assert fixed.to_runtime_request().payload["style_condition"] == "female"
    assert "mode" not in fixed.to_runtime_request().payload
    assert clone.to_runtime_request().payload["omnivoice_mode"] == "clone"
    assert clone.to_runtime_request().payload["reference_text"] == "reference transcript"
    assert design.to_runtime_request().payload["omnivoice_mode"] == "design"
    assert design.to_runtime_request().payload["design_instruction"] == "warm and clear"


def test_local_tts_request_hides_sensitive_omnivoice_fields_from_repr():
    clone = _omnivoice_tts_request(
        omnivoice_mode="clone",
        reference_audio_path="fixture.wav",
        reference_text="secret reference transcript",
    )
    design = _omnivoice_tts_request(
        omnivoice_mode="design", design_instruction="secret design instruction"
    )

    assert "secret reference transcript" not in repr(clone)
    assert "secret design instruction" not in repr(design)
    assert "fixture.wav" in repr(clone)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"omnivoice_mode": "clone"}, "reference audio"),
        ({"omnivoice_mode": "clone", "reference_audio_path": "fixture.wav"}, "reference text"),
        ({"omnivoice_mode": "design"}, "instruction"),
        (
            {
                "omnivoice_mode": "fixed-style",
                "reference_audio_path": "fixture.wav",
            },
            "fixed-style",
        ),
        (
            {
                "omnivoice_mode": "fixed-style",
                "reference_text": "reference transcript",
            },
            "fixed-style",
        ),
        (
            {
                "omnivoice_mode": "fixed-style",
                "design_instruction": "design instruction",
            },
            "fixed-style",
        ),
        (
            {
                "omnivoice_mode": "clone",
                "reference_audio_path": "fixture.wav",
                "reference_text": "reference",
                "style_condition": "female",
            },
            "style or design",
        ),
        (
            {
                "omnivoice_mode": "design",
                "design_instruction": "warm",
                "reference_audio_path": "fixture.wav",
            },
            "fixed-style or clone",
        ),
        (
            {
                "omnivoice_mode": "fixed-style",
                "style_condition": "female",
                "instruction": "Qwen instruction",
            },
            "Qwen instruction",
        ),
    ),
)
def test_local_tts_request_rejects_missing_or_cross_mode_omnivoice_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _omnivoice_tts_request(**kwargs)


def test_auto_selection_preserves_python_rollback_until_family_promotion():
    python = FixtureDriver("python")
    audio_cpp = FixtureDriver("audio-cpp", available=False)
    runtime = LocalAudioRuntime(LocalRuntimeRegistry((python, audio_cpp)))

    response = runtime.execute(_request())

    assert response.payload == {"driver": "python"}
    assert python.requests == [_request()]
    assert audio_cpp.requests == []
    with pytest.raises(RuntimeUnavailableError, match="audio-cpp"):
        runtime.execute(_request(), runtime_choice="audio-cpp")


def test_promoted_auto_route_can_use_audio_cpp_and_unload_is_family_scoped():
    python = FixtureDriver("python")
    audio_cpp = FixtureDriver("audio-cpp")
    runtime = LocalAudioRuntime(
        LocalRuntimeRegistry((python, audio_cpp)),
        promoted_families=("qwen3-asr",),
    )

    assert runtime.execute(_request()).payload == {"driver": "audio-cpp"}
    runtime.cancel("fixture-request", family="qwen3-asr")
    runtime.unload("qwen3-asr")

    assert audio_cpp.cancelled == ["fixture-request"]
    assert audio_cpp.closed == 1
    assert python.closed == 0


def test_second_driver_can_host_same_provider_contract_without_audio_cpp_fields():
    mlx = FixtureDriver("mlx")
    runtime = LocalAudioRuntime(LocalRuntimeRegistry((mlx,)))

    response = runtime.execute(_request(), runtime_choice="mlx")

    assert response.payload == {"driver": "mlx"}
    assert mlx.requests[0].provider_id == "qwen-local"
    assert mlx.requests[0].family == "qwen3-asr"


def test_runtime_rejects_a_driver_response_bound_to_a_different_request():
    class WrongResponseDriver(FixtureDriver):
        def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
            return LocalRuntimeResponse(request_id="other-request")

    runtime = LocalAudioRuntime(LocalRuntimeRegistry((WrongResponseDriver("python"),)))

    with pytest.raises(RuntimeProtocolError, match="request_id"):
        runtime.execute(_request())


def test_unload_does_not_close_a_driver_still_active_for_another_family():
    python = FixtureDriver("python")
    runtime = LocalAudioRuntime(LocalRuntimeRegistry((python,)))
    nemotron_request = _request(family="nemotron-3.5-asr")

    runtime.execute(_request())
    runtime.execute(nemotron_request)
    runtime.unload("qwen3-asr")

    assert python.closed == 0
    runtime.unload("nemotron-3.5-asr")
    assert python.closed == 1
