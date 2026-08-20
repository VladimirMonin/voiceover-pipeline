from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from voiceover_pipeline.config import (
    OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
    OMNIVOICE_DEFAULT_SEED,
    OMNIVOICE_DEFAULT_STEPS,
    OMNIVOICE_LOCAL_MODEL_ID,
    PROVIDER_DEFAULT_MODELS,
)
from voiceover_pipeline.local_runtime.contracts import (
    LocalRuntimeResponse,
    LocalTTSRequest,
    LocalTTSResponse,
    RuntimeExecutionReceipt,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.transports import audio_cpp_omnivoice
from voiceover_pipeline.local_runtime.transports.audio_cpp_omnivoice import (
    VerifiedOmniVoiceModel,
    admit_omnivoice_model,
)
from voiceover_pipeline.local_tts_text import merge_omnivoice_session_fragments
from voiceover_pipeline.models import ScriptChunk
from voiceover_pipeline.providers import audio_cpp_omnivoice_tts
from voiceover_pipeline.providers.audio_cpp_omnivoice_tts import OmniVoiceLocalTTSProvider


@dataclass
class _Runtime:
    response: LocalTTSResponse
    request: LocalTTSRequest | None = None
    requests: list[LocalTTSRequest] = field(default_factory=list)

    def execute_tts(self, request: LocalTTSRequest, *, runtime_choice: str) -> LocalTTSResponse:
        assert runtime_choice == "audio-cpp"
        self.request = request
        self.requests.append(request)
        return self.response


def _admitted_model(monkeypatch, tmp_path):
    model = tmp_path / "admitted.gguf"
    model.write_bytes(b"fixture bytes")
    inventory = audio_cpp_omnivoice.find_family_inventory("omnivoice")
    assert inventory.model_sha256 is not None
    monkeypatch.setattr(audio_cpp_omnivoice, "_sha256_file", lambda _path: inventory.model_sha256)
    monkeypatch.setenv(
        "VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE", "accept-cc-by-nc-4.0-local-use"
    )
    return admit_omnivoice_model(model_path=model, model_id=OMNIVOICE_LOCAL_MODEL_ID)


def test_typed_tts_contract_carries_omnivoice_controls_and_receipt():
    request = LocalTTSRequest(
        request_id="request-1",
        family="omnivoice",
        provider_id="omnivoice-local",
        text="Проверка",
        model_id=OMNIVOICE_LOCAL_MODEL_ID,
        language="ru",
        omnivoice_mode="fixed-style",
        style_condition="female",
        text_chunk_size=420,
        seed=1234,
        num_inference_steps=32,
        guidance_scale=2.0,
    )
    response = LocalTTSResponse.from_runtime_response(
        LocalRuntimeResponse(
            request_id="request-1",
            payload={"audio_bytes": b"wav", "audio_format": "wav"},
            receipt=RuntimeExecutionReceipt(
                driver_id="audio-cpp",
                transport="container-cli",
                source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
                build_hash="d98b99f10355a018ddaec6d17999725ab7bdbcf5f164ab067c1288a15a4f51dd",
            ),
        )
    )

    assert request.to_runtime_request().payload == {
        "text": "Проверка",
        "model_id": OMNIVOICE_LOCAL_MODEL_ID,
        "voice": None,
        "language": "ru",
        "text_chunk_size": 420,
        "seed": 1234,
        "num_inference_steps": 32,
        "guidance_scale": 2.0,
        "omnivoice_mode": "fixed-style",
        "style_condition": "female",
    }
    assert response.audio_bytes == b"wav"
    assert response.audio_format == "wav"
    assert response.receipt is not None
    assert response.receipt.transport == "container-cli"


def test_provider_returns_wav_bytes_with_exact_admitted_model_receipt(monkeypatch, tmp_path):
    runtime = _Runtime(
        LocalTTSResponse(
            audio_bytes=b"RIFFfixtureWAVE",
            audio_format="wav",
            payload={"sample_rate_hz": 24_000, "channels": 1, "duration_s": 0.01},
            receipt=RuntimeExecutionReceipt(
                driver_id="audio-cpp",
                transport="container-cli",
                source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
                build_hash="d98b99f10355a018ddaec6d17999725ab7bdbcf5f164ab067c1288a15a4f51dd",
            ),
        )
    )
    admitted_model = _admitted_model(monkeypatch, tmp_path)
    provider = OmniVoiceLocalTTSProvider(cast(Any, runtime), admitted_model=admitted_model)

    result = provider.synthesize_chunk("Привет, мир!", "chunk_01")

    assert runtime.request is not None
    assert runtime.request.model_id == OMNIVOICE_LOCAL_MODEL_ID
    assert runtime.request.voice is None
    assert runtime.request.omnivoice_mode == "fixed-style"
    assert runtime.request.style_condition == "female"
    assert runtime.request.text_chunk_size == 420
    assert runtime.request.seed == OMNIVOICE_DEFAULT_SEED
    assert runtime.request.num_inference_steps == OMNIVOICE_DEFAULT_STEPS
    assert runtime.request.guidance_scale == OMNIVOICE_DEFAULT_GUIDANCE_SCALE
    assert result.audio_bytes == b"RIFFfixtureWAVE"
    assert result.audio_format == "wav"
    assert result.client_path == "omnivoice-local"
    assert result.raw_metadata["voice_selection"] == {
        "kind": "built-in-style-condition",
        "condition": "female",
        "named_preset": False,
        "voice_cloning": False,
        "voice_design": False,
    }
    assert result.raw_metadata["voice_session"] == {
        "strategy": "single-native-invocation-internal-text-chunking",
        "seed": OMNIVOICE_DEFAULT_SEED,
        "internal_text_chunk_size": 420,
    }
    assert result.raw_metadata["runtime_receipt"] == admitted_model.public_receipt()
    assert set(result.raw_metadata["runtime_receipt"]) == {
        "model_id",
        "sha256",
        "quantization",
        "license",
        "provenance",
    }
    assert result.raw_metadata["runtime"]["driver_id"] == "audio-cpp"
    assert "/tmp/" not in str(result.raw_metadata)


def test_provider_uses_one_runtime_request_for_multiple_omnivoice_fragments(monkeypatch, tmp_path):
    runtime = _Runtime(LocalTTSResponse(audio_bytes=b"RIFFfixtureWAVE", audio_format="wav"))
    provider = OmniVoiceLocalTTSProvider(
        cast(Any, runtime), admitted_model=_admitted_model(monkeypatch, tmp_path)
    )
    fragments = [
        ScriptChunk(number=1, id="chunk_01_part_01", text="Первая фраза."),
        ScriptChunk(number=2, id="chunk_01_part_02", text="Вторая фраза."),
        ScriptChunk(number=3, id="chunk_02_part_01", text="Третья фраза."),
    ]

    session = merge_omnivoice_session_fragments(fragments)
    results = [provider.synthesize_chunk(chunk.text, chunk.id) for chunk in session]

    assert len(session) == 1
    assert len(results) == 1
    assert len(runtime.requests) == 1
    assert runtime.requests[0].text == "Первая фраза. Вторая фраза. Третья фраза."
    assert runtime.requests[0].omnivoice_mode == "fixed-style"
    assert runtime.requests[0].style_condition == "female"
    assert runtime.requests[0].text_chunk_size == 420


def test_provider_fails_closed_when_an_injected_runtime_has_no_admitted_model():
    runtime = _Runtime(LocalTTSResponse(audio_bytes=b"RIFFfixtureWAVE", audio_format="wav"))
    provider = OmniVoiceLocalTTSProvider(cast(Any, runtime))

    with pytest.raises(RuntimeUnavailableError, match="admitted model"):
        provider.synthesize_chunk("Привет, мир!", "chunk_01")


def test_provider_clone_mode_stages_reference_fields_and_metadata(monkeypatch, tmp_path):
    runtime = _Runtime(
        LocalTTSResponse(
            audio_bytes=b"RIFFfixtureWAVE",
            audio_format="wav",
            payload={"sample_rate_hz": 24_000, "channels": 1, "duration_s": 0.01},
        )
    )
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixture")
    provider = OmniVoiceLocalTTSProvider(
        cast(Any, runtime),
        admitted_model=_admitted_model(monkeypatch, tmp_path),
        mode="clone",
        reference_audio_path=reference,
        reference_text="Текст референса",
    )

    result = provider.synthesize_chunk("Привет, мир!", "chunk_01")

    assert runtime.request is not None
    assert runtime.request.omnivoice_mode == "clone"
    assert runtime.request.reference_audio_path == reference
    assert runtime.request.reference_text == "Текст референса"
    assert runtime.request.style_condition is None
    assert runtime.request.design_instruction is None
    assert result.raw_metadata["voice_selection"] == {
        "kind": "reference-clone",
        "named_preset": False,
        "voice_cloning": True,
        "voice_design": False,
    }
    assert result.raw_metadata["voice_session"]["strategy"] == ("reference-isolated-native-session")
    assert "Текст референса" not in str(result.raw_metadata)
    assert str(reference) not in str(result.raw_metadata)


def test_provider_design_mode_passes_instruction_and_metadata(monkeypatch, tmp_path):
    runtime = _Runtime(
        LocalTTSResponse(
            audio_bytes=b"RIFFfixtureWAVE",
            audio_format="wav",
            payload={"sample_rate_hz": 24_000, "channels": 1, "duration_s": 0.01},
        )
    )
    provider = OmniVoiceLocalTTSProvider(
        cast(Any, runtime),
        admitted_model=_admitted_model(monkeypatch, tmp_path),
        mode="design",
        design_instruction="warm and clear",
    )

    result = provider.synthesize_chunk("Привет, мир!", "chunk_01")

    assert runtime.request is not None
    assert runtime.request.omnivoice_mode == "design"
    assert runtime.request.design_instruction == "warm and clear"
    assert runtime.request.style_condition is None
    assert runtime.request.reference_audio_path is None
    assert runtime.request.reference_text is None
    assert result.raw_metadata["voice_selection"] == {
        "kind": "design-instruction",
        "named_preset": False,
        "voice_cloning": False,
        "voice_design": True,
    }
    assert result.raw_metadata["voice_session"]["strategy"] == ("design-instruction-native-session")
    assert "warm and clear" not in str(result.raw_metadata)


def test_provider_rejects_a_caller_model_id_that_differs_from_admission(monkeypatch, tmp_path):
    runtime = _Runtime(LocalTTSResponse(audio_bytes=b"RIFFfixtureWAVE", audio_format="wav"))
    admitted_model = _admitted_model(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="does not match the admitted model"):
        OmniVoiceLocalTTSProvider(
            cast(Any, runtime), admitted_model=admitted_model, model_id="audio-cpp/not-admitted"
        )


def test_dependency_probe_rejects_forged_caller_supplied_admission(monkeypatch, tmp_path):
    forged = VerifiedOmniVoiceModel(
        model_path=tmp_path / "missing.gguf",
        model_id=OMNIVOICE_LOCAL_MODEL_ID,
        sha256="forged",
        quantization="forged",
        license="forged",
        provenance="forged",
    )
    monkeypatch.setattr(audio_cpp_omnivoice_tts, "which", lambda _command: "/usr/bin/docker")
    monkeypatch.delenv("VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE", raising=False)

    health = audio_cpp_omnivoice_tts.omnivoice_local_dependency_probe(forged)

    assert health.available is False


def test_omnivoice_admission_rejects_arbitrary_existing_bytes_before_provider_construction(
    monkeypatch, tmp_path
):
    model = tmp_path / "arbitrary.gguf"
    model.write_bytes(b"not the approved OmniVoice artifact")
    monkeypatch.setenv("VOICEOVER_OMNIVOICE_MODEL", str(model))
    monkeypatch.setenv(
        "VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE", "accept-cc-by-nc-4.0-local-use"
    )
    monkeypatch.setenv("VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON", '["true"]')

    health = audio_cpp_omnivoice_tts.omnivoice_local_dependency_probe()
    provider = OmniVoiceLocalTTSProvider.from_environment()

    assert health.available is False
    assert provider._runtime is None


def test_omnivoice_admission_requires_explicit_noncommercial_local_use_opt_in(
    monkeypatch, tmp_path
):
    model = tmp_path / "configured.gguf"
    model.write_bytes(b"any bytes are sufficient because acknowledgment fails first")
    monkeypatch.setenv("VOICEOVER_OMNIVOICE_MODEL", str(model))
    monkeypatch.delenv("VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE", raising=False)

    health = audio_cpp_omnivoice_tts.omnivoice_local_dependency_probe()
    provider = OmniVoiceLocalTTSProvider.from_environment()

    assert health.available is False
    assert provider._runtime is None


def test_windows_probe_maps_native_package_failure_to_structured_reason(monkeypatch, tmp_path):
    model = tmp_path / "approved.gguf"
    model.write_bytes(b"approved artifact bytes")
    admitted_model = VerifiedOmniVoiceModel(
        model_path=model,
        model_id=OMNIVOICE_LOCAL_MODEL_ID,
        sha256="approved",
        quantization="q8_0",
        license="non-commercial",
        provenance="fixture",
    )
    monkeypatch.setattr(
        audio_cpp_omnivoice_tts, "_re_admit_omnivoice_model", lambda _m: admitted_model
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE", str(tmp_path / "missing.exe"))

    health = audio_cpp_omnivoice_tts.omnivoice_local_dependency_probe(model)

    assert health.available is False
    assert health.reason_code == "missing_executable"


def test_admitted_omnivoice_model_receipt_has_exact_public_provenance(monkeypatch, tmp_path):
    admitted_model = _admitted_model(monkeypatch, tmp_path)
    runtime = _Runtime(
        LocalTTSResponse(
            audio_bytes=b"RIFFfixtureWAVE",
            audio_format="wav",
            payload={"sample_rate_hz": 24_000, "channels": 1, "duration_s": 0.01},
        )
    )
    provider = OmniVoiceLocalTTSProvider(cast(Any, runtime), admitted_model=admitted_model)

    result = provider.synthesize_chunk("Проверка", "chunk_01")
    receipt = result.raw_metadata["runtime_receipt"]

    assert receipt == {
        "model_id": "audio-cpp/omnivoice-q8_0",
        "sha256": "2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b",
        "quantization": "Q8_0 GGUF",
        "license": "CC-BY-NC-4.0 upstream weights; local noncommercial research only",
        "provenance": (
            "audio-cpp/audio.cpp-gguf@c3857f1ec35cfea8993924e7c2a6f682b5dc060b "
            "OmniVoice-GGUF/omnivoice-q8_0.gguf; converted from k2-fsa/OmniVoice; "
            "runtime audio.cpp@502b5b74bd26e9b4aed267d1776ecf131cae7215"
        ),
    }
    assert str(tmp_path) not in str(receipt)


def test_omnivoice_configuration_is_explicit_and_not_the_global_default():
    assert PROVIDER_DEFAULT_MODELS["omnivoice-local"] == OMNIVOICE_LOCAL_MODEL_ID
    assert PROVIDER_DEFAULT_MODELS["omnivoice-local"] != PROVIDER_DEFAULT_MODELS["polza-chat-audio"]


def test_cli_explicitly_accepts_the_opt_in_provider():
    from voiceover_pipeline.cli import (
        CliError,
        _resolve_provider_style_prompt,
        build_parser,
        build_provider,
    )

    args = build_parser().parse_args(["generate", "--provider", "omnivoice-local"])

    assert args.provider == "omnivoice-local"
    assert _resolve_provider_style_prompt(args) is None
    style_args = build_parser().parse_args(
        ["generate", "--provider", "omnivoice-local", "--style-prompt", "тёплый голос"]
    )
    with pytest.raises(CliError, match="style controls"):
        build_provider(style_args, api_key="", style_prompt=None, prompt_mode="none")
