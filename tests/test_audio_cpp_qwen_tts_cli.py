from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Any, cast

import pytest

from voiceover_pipeline.audio_cpp.inventory import PINNED_AUDIO_CPP_REVISION
from voiceover_pipeline.config import (
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)
from voiceover_pipeline.local_runtime.contracts import LocalTTSRequest, RuntimeProtocolError
from voiceover_pipeline.local_runtime.drivers.audio_cpp import AudioCppRuntimeDriver
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    PINNED_AUDIO_CPP_CONTAINER_IMAGE,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_qwen_tts import (
    AudioCppQwenTTSCLITransport,
)


def _model_package(tmp_path: Path, model_id: str = QWEN_MODEL_CUSTOMVOICE) -> Path:
    package = tmp_path / "arbitrary-package-name"
    package.mkdir()
    for filename in ("model.safetensors", "tokenizer_config.json"):
        (package / filename).write_bytes(b"fixture")
    (package / "config.json").write_text(json.dumps({"_name_or_path": model_id}), encoding="utf-8")
    (package / "speech_tokenizer").mkdir()
    return package


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\x00\x00" * 240)


def _request_payload(
    mode: str,
    model_package: Path,
    reference: Path | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": "Привет",
        "model_id": model_id if model_id is not None else _model_id_for_mode(mode),
        "model_artifact_path": str(model_package),
        "voice": "Sohee" if mode == "custom-voice" else None,
        "language": "ru",
        "mode": mode,
    }
    if mode == "voice-clone":
        assert reference is not None
        payload["reference_audio_path"] = str(reference)
        payload["reference_text"] = "Текст референса"
    else:
        payload["instruction"] = "calm and clear"
    return {
        "schema_version": 1,
        "operation": "tts",
        "family": "qwen3-tts",
        "provider_id": "qwen-local",
        "payload": payload,
    }


def _model_id_for_mode(mode: str) -> str:
    return {
        "custom-voice": QWEN_MODEL_CUSTOMVOICE,
        "voice-clone": QWEN_MODEL_BASE,
        "voice-design": QWEN_MODEL_VOICE_DESIGN,
    }[mode]


class _CompletedProcess:
    returncode = 0

    def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
        return b"", b""

    def poll(self) -> int:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        return self.returncode


def _transport(
    tmp_path: Path, model_id: str = QWEN_MODEL_CUSTOMVOICE
) -> AudioCppQwenTTSCLITransport:
    return AudioCppQwenTTSCLITransport(
        model_package_path=_model_package(tmp_path, model_id),
        container_command=("docker",),
    )


@pytest.mark.parametrize(
    ("mode", "expected_task"),
    [
        ("custom-voice", "tts"),
        ("voice-clone", "clon"),
        ("voice-design", "vdes"),
    ],
)
def test_container_transport_maps_all_qwen_modes_to_pinned_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, expected_task: str
) -> None:
    transport = _transport(tmp_path, _model_id_for_mode(mode))
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
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
        _write_wav(output_directory / "qwen3-tts.wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    response = cast(
        dict[str, Any],
        transport.invoke(
            "request-1", _request_payload(mode, transport._model_package_path, reference)
        ),
    )

    command = cast(tuple[str, ...], captured["command"])
    kwargs = cast(dict[str, Any], captured["kwargs"])
    assert command[:3] == ("docker", "run", "--rm")
    assert PINNED_AUDIO_CPP_CONTAINER_IMAGE in command
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert "/tmp:rw,noexec,nosuid,size=64m" in command
    assert "--gpus" in command and "all" in command
    assert command[command.index("--task") + 1] == expected_task
    assert command[command.index("--family") + 1] == "qwen3_tts"
    assert command[command.index("--model") + 1] == "/models/qwen3-tts"
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == "/"
    assert kwargs["start_new_session"] is True
    if mode == "custom-voice":
        assert command[command.index("--speaker") + 1] == "Sohee"
        assert command[command.index("--instruct") + 1] == "calm and clear"
        assert "--voice-ref" not in command
    elif mode == "voice-clone":
        assert command[command.index("--voice-ref") + 1] == "/input/reference.wav"
        assert command[command.index("--reference-text") + 1] == "Текст референса"
        assert str(reference) not in command
        assert any("dst=/input/reference.wav,readonly" in part for part in command)
    else:
        assert command[command.index("--instruct") + 1] == "calm and clear"
        assert "--voice-ref" not in command
    assert response["response"]["audio_format"] == "wav"
    assert response["response"]["channels"] == 1
    assert response["response"]["sample_rate_hz"] == 24_000
    assert "/tmp/" not in str(response)


def test_typed_qwen_tts_request_reaches_container_cli_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(command, **_kwargs):
        captured["command"] = tuple(command)
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        _write_wav(Path(output_mount.split("src=", 1)[1].split(",", 1)[0]) / "qwen3-tts.wav")
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    driver = AudioCppRuntimeDriver(
        binary_path=None,
        source_revision=PINNED_AUDIO_CPP_REVISION,
        transport=transport,
        transport_name="container-cli",
    )

    response = driver.invoke(
        LocalTTSRequest(
            request_id="typed-request",
            family="qwen3-tts",
            provider_id="qwen-local",
            text="Привет",
            model_id=QWEN_MODEL_CUSTOMVOICE,
            model_artifact_path=transport._model_package_path,
            voice="Sohee",
            language="ru",
            mode="custom-voice",
            instruction="calm and clear",
        ).to_runtime_request()
    )

    assert response.payload["audio_format"] == "wav"
    assert response.receipt is not None
    assert response.receipt.transport == "container-cli"
    assert cast(tuple[str, ...], captured["command"])[-1] == "calm and clear"


@pytest.mark.parametrize(
    "package_kind", ["file", "missing-package-file", "missing-speech-tokenizer"]
)
def test_container_transport_rejects_invalid_safetensors_package(
    tmp_path: Path, package_kind: str
) -> None:
    package = tmp_path / "package"
    if package_kind == "file":
        package.write_bytes(b"not a package")
    else:
        package.mkdir()
        for filename in ("model.safetensors", "config.json", "tokenizer_config.json"):
            (package / filename).write_bytes(b"fixture")
        if package_kind == "missing-package-file":
            (package / "tokenizer_config.json").unlink()
        else:
            (package / "speech_tokenizer").write_bytes(b"not a package")

    with pytest.raises(ValueError, match="Qwen3-TTS model package"):
        AudioCppQwenTTSCLITransport(model_package_path=package)


def test_container_transport_rejects_non_file_clone_reference(tmp_path: Path) -> None:
    transport = _transport(tmp_path, QWEN_MODEL_BASE)
    reference_directory = tmp_path / "reference-directory"
    reference_directory.mkdir()

    with pytest.raises(RuntimeProtocolError, match="reference audio must be a regular file"):
        transport.invoke(
            "request-2",
            _request_payload("voice-clone", transport._model_package_path, reference_directory),
        )


def test_container_transport_rejects_a_model_mode_mismatch(tmp_path: Path) -> None:
    transport = _transport(tmp_path)
    payload = _request_payload("custom-voice", transport._model_package_path)
    cast(dict[str, object], payload["payload"])["model_id"] = QWEN_MODEL_BASE

    with pytest.raises(RuntimeProtocolError, match="does not support the requested mode"):
        transport.invoke("request-model-mismatch", payload)


def test_container_transport_rejects_a_different_model_package(tmp_path: Path) -> None:
    transport = _transport(tmp_path)
    foreign_root = tmp_path / "foreign-model"
    foreign_root.mkdir()
    foreign_package = _model_package(foreign_root)

    with pytest.raises(RuntimeProtocolError, match="does not match the transport"):
        transport.invoke(
            "request-package-mismatch",
            _request_payload("custom-voice", foreign_package),
        )


@pytest.mark.parametrize(
    ("package_model_id", "request_mode"),
    [
        ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "voice-design"),
        (QWEN_MODEL_CUSTOMVOICE, "voice-clone"),
        (QWEN_MODEL_VOICE_DESIGN, "custom-voice"),
    ],
)
def test_container_transport_rejects_package_identity_mismatches_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_model_id: str, request_mode: str
) -> None:
    transport = _transport(tmp_path, package_model_id)
    reference = tmp_path / "reference.wav"
    _write_wav(reference)

    def unexpected_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("container launch must not occur for a package identity mismatch")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)

    with pytest.raises(RuntimeProtocolError, match="identity does not match"):
        transport.invoke(
            "package-identity-mismatch",
            _request_payload(request_mode, transport._model_package_path, reference),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "not json",
        "{}",
        '{"_name_or_path": "Qwen/Qwen3-TTS-12Hz-9B-Unknown"}',
    ],
)
def test_container_transport_rejects_invalid_or_unsupported_package_metadata(
    tmp_path: Path, metadata: str
) -> None:
    package = _model_package(tmp_path)
    (package / "config.json").write_text(metadata, encoding="utf-8")

    with pytest.raises(ValueError, match="model (metadata|identity)"):
        AudioCppQwenTTSCLITransport(model_package_path=package)


@pytest.mark.parametrize(
    ("metadata", "mode", "model_id"),
    [
        (
            {"tts_model_size": "0b6", "tts_model_type": "base"},
            "voice-clone",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ),
        (
            {"tts_model_size": "1b7", "tts_model_type": "custom_voice"},
            "custom-voice",
            QWEN_MODEL_CUSTOMVOICE,
        ),
        ({"tts_model_size": "1b7", "tts_model_type": "base"}, "voice-clone", QWEN_MODEL_BASE),
        (
            {"tts_model_size": "1b7", "tts_model_type": "voice_design"},
            "voice-design",
            QWEN_MODEL_VOICE_DESIGN,
        ),
    ],
)
def test_container_transport_derives_supported_identity_from_qwen_config_metadata(
    tmp_path: Path, metadata: dict[str, str], mode: str, model_id: str
) -> None:
    package = _model_package(tmp_path)
    (package / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    transport = AudioCppQwenTTSCLITransport(model_package_path=package)
    reference = tmp_path / "reference.wav"

    decoded = transport._decode_vop_request(
        _request_payload(mode, package, reference, model_id=model_id)
    )

    assert decoded["model_id"] == model_id
    assert decoded["mode"] == mode


def test_container_transport_rejects_conflicting_package_identity_metadata(tmp_path: Path) -> None:
    package = _model_package(tmp_path)
    (package / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": QWEN_MODEL_BASE,
                "tts_model_size": "1b7",
                "tts_model_type": "voice_design",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting model metadata"):
        AudioCppQwenTTSCLITransport(model_package_path=package)


@pytest.mark.skipif(
    __import__("sys").platform.startswith("win"), reason="Windows symlink creation needs privileges"
)
def test_container_transport_rejects_output_symlink_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(tmp_path)
    outside = tmp_path / "outside.wav"
    _write_wav(outside)

    def fake_popen(command, **_kwargs):
        output_mount = next(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount" and "dst=/output" in command[index + 1]
        )
        output_directory = Path(output_mount.split("src=", 1)[1].split(",", 1)[0])
        (output_directory / "qwen3-tts.wav").symlink_to(outside)
        return _CompletedProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeProtocolError, match="escaped the private workspace"):
        transport.invoke(
            "request-3", _request_payload("custom-voice", transport._model_package_path)
        )


def test_container_transport_cancellation_terminates_an_active_process(tmp_path: Path) -> None:
    transport = _transport(tmp_path)
    active_process = _CompletedProcess()
    terminated: list[object] = []
    with transport._lock:
        transport._processes["active"] = cast(Any, active_process)

    transport._terminate_process = lambda process: terminated.append(process)  # type: ignore[method-assign]
    transport.cancel("active")

    assert terminated == [active_process]
