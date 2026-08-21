from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any, cast

import pytest

from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports import audio_cpp_omnivoice
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    PINNED_AUDIO_CPP_CONTAINER_IMAGE,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_omnivoice import (
    AudioCppOmniVoiceCLITransport,
)


def _request_payload(text: str = "Привет") -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "tts",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "payload": {
            "text": text,
            "model_id": OMNIVOICE_LOCAL_MODEL_ID,
            "voice": None,
            "language": "ru",
            "omnivoice_mode": "fixed-style",
            "style_condition": "female",
            "text_chunk_size": 420,
            "seed": 1234,
            "num_inference_steps": 32,
            "guidance_scale": 2.0,
        },
    }


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\x00\x00" * 240)


class _CompletedProcess:
    returncode = 0

    def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
        return b"", b""

    def poll(self) -> int:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        return self.returncode


def _transport(monkeypatch, tmp_path: Path) -> AudioCppOmniVoiceCLITransport:
    model = tmp_path / "omnivoice-q8_0.gguf"
    model.write_bytes(b"model fixture")
    inventory = audio_cpp_omnivoice.find_family_inventory("omnivoice")
    assert inventory.model_sha256 is not None
    monkeypatch.setattr(audio_cpp_omnivoice, "_sha256_file", lambda _path: inventory.model_sha256)
    admitted_model = audio_cpp_omnivoice.admit_omnivoice_model(
        model_path=model,
        model_id=OMNIVOICE_LOCAL_MODEL_ID,
    )
    return AudioCppOmniVoiceCLITransport(
        model=admitted_model,
        container_command=("docker",),
    )


def test_container_transport_uses_pinned_argv_and_copies_validated_wav(monkeypatch, tmp_path: Path):
    transport = _transport(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        output_directory = Path(output_mount.split("src=", 1)[1].split(",", 1)[0])
        _write_wav(output_directory / "omnivoice.wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    response = cast(dict[str, Any], transport.invoke("request-1", _request_payload()))

    command = cast(tuple[str, ...], captured["command"])
    kwargs = cast(dict[str, Any], captured["kwargs"])
    assert command[:3] == ("docker", "run", "--rm")
    assert PINNED_AUDIO_CPP_CONTAINER_IMAGE in command
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert "--gpus" in command and "all" in command
    assert "--family" in command and "omnivoice" in command
    assert "--instruct" in command and "female" in command
    assert "--text-chunk-size" in command and "420" in command
    assert "--seed" in command and "1234" in command
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == "/"
    assert response["response"]["audio_format"] == "wav"
    assert response["response"]["sample_rate_hz"] == 24_000
    assert response["response"]["channels"] == 1
    assert (
        response["response"]["audio_bytes"][:12]
        == b"RIFF" + response["response"]["audio_bytes"][4:8] + b"WAVE"
    )


def test_container_transport_rejects_bad_output_without_private_path(monkeypatch, tmp_path: Path):
    transport = _transport(monkeypatch, tmp_path)

    def fake_popen(command, **_kwargs):
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        output_directory = Path(output_mount.split("src=", 1)[1].split(",", 1)[0])
        (output_directory / "omnivoice.wav").write_bytes(b"not a wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeProtocolError) as error:
        transport.invoke("request-2", _request_payload())

    assert "WAVE" in str(error.value)
    assert "/tmp/" not in str(error.value)


def test_cancellation_terminates_an_active_process(monkeypatch, tmp_path: Path):
    transport = _transport(monkeypatch, tmp_path)
    active_process = _CompletedProcess()
    terminated: list[object] = []
    with transport._lock:
        transport._processes["active"] = cast(Any, active_process)
    monkeypatch.setattr(transport, "_terminate_process", lambda process: terminated.append(process))

    transport.cancel("active")

    assert terminated == [active_process]


def _clone_payload(reference_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "tts",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "payload": {
            "text": "Привет",
            "model_id": OMNIVOICE_LOCAL_MODEL_ID,
            "voice": None,
            "language": "ru",
            "omnivoice_mode": "clone",
            "text_chunk_size": 420,
            "seed": 1234,
            "num_inference_steps": 32,
            "guidance_scale": 2.0,
            "reference_audio_path": str(reference_path),
            "reference_text": "Текст референса",
        },
    }


def _design_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "tts",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "payload": {
            "text": "Привет",
            "model_id": OMNIVOICE_LOCAL_MODEL_ID,
            "voice": None,
            "language": "ru",
            "omnivoice_mode": "design",
            "design_instruction": "warm and clear",
            "text_chunk_size": 420,
            "seed": 1234,
            "num_inference_steps": 32,
            "guidance_scale": 2.0,
        },
    }


def _auto_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "tts",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "payload": {
            "text": "Привет",
            "model_id": OMNIVOICE_LOCAL_MODEL_ID,
            "voice": None,
            "language": "ru",
            "omnivoice_mode": "auto",
            "text_chunk_size": 420,
            "seed": 1234,
            "num_inference_steps": 32,
            "guidance_scale": 2.0,
        },
    }


def test_container_transport_auto_mode_omits_instruct_and_reference_flags(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(audio_cpp_omnivoice.sys, "platform", "linux")
    transport = _transport(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        output_directory = Path(output_mount.split("src=", 1)[1].split(",", 1)[0])
        _write_wav(output_directory / "omnivoice.wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    response = cast(dict[str, Any], transport.invoke("auto-1", _auto_payload()))

    command = cast(tuple[str, ...], captured["command"])
    assert "--instruct" not in command
    assert "--voice-ref" not in command
    assert "--reference-text" not in command
    assert "--text-chunk-size" in command and "420" in command
    assert "--seed" in command and "1234" in command
    assert "--num-inference-steps" in command and "32" in command
    assert "--guidance-scale" in command and "2.0" in command
    assert response["response"]["audio_format"] == "wav"


def test_container_transport_clone_fails_closed_without_staged_reference(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(audio_cpp_omnivoice.sys, "platform", "linux")
    transport = _transport(monkeypatch, tmp_path)
    reference_path = tmp_path / "reference.wav"
    _write_wav(reference_path)

    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("no container process may start for an unstaged clone")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)

    with pytest.raises(RuntimeTransportError, match="clone reference audio is unavailable"):
        transport.invoke("clone-1", _clone_payload(reference_path))


def test_container_transport_design_passes_instruction_argv(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(audio_cpp_omnivoice.sys, "platform", "linux")
    transport = _transport(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        output_directory = Path(output_mount.split("src=", 1)[1].split(",", 1)[0])
        _write_wav(output_directory / "omnivoice.wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    response = cast(dict[str, Any], transport.invoke("design-1", _design_payload()))

    command = cast(tuple[str, ...], captured["command"])
    assert command[command.index("--instruct") + 1] == "warm and clear"
    assert "--voice-ref" not in command
    assert "--reference-text" not in command
    assert response["response"]["audio_format"] == "wav"
