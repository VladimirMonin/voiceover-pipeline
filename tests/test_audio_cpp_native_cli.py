from __future__ import annotations

import hashlib
import json
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    AudioCppNativeCLITransport,
    build_audio_cpp_cli_arguments,
    decode_audio_cpp_cli_request,
    discover_native_audio_cpp_install,
)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\x00\x00" * 240)


def _package(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "Аудио CPP package"
    package.mkdir()
    executable = package / "audiocpp_cli.exe"
    executable.write_bytes(b"native executable")
    runtime_dll = package / "audiocpp_runtime.dll"
    runtime_dll.write_bytes(b"runtime dll")
    _write_closure(package)
    return executable, runtime_dll


def _write_closure(package: Path) -> None:
    manifest = package / "audio_cpp_dependency_closure.json"
    files = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*")
        if path.is_file() and path != manifest
    }
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": files}, sort_keys=True), encoding="utf-8"
    )
    executable = package / "audiocpp_cli.exe"
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
        "model_families": ["qwen3-asr", "omnivoice"],
    }
    (package / "build_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )


def _qwen_request(audio_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "asr",
        "family": "qwen3-asr",
        "provider_id": "qwen-local",
        "payload": {
            "audio_path": str(audio_path),
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "language": "Russian",
            "timestamp_mode": "word",
            "context_text": "Сохрани PostgreSQL.",
        },
    }


def _nemotron_request(audio_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "asr",
        "family": "nemotron-3.5-asr",
        "provider_id": "nemotron-local",
        "payload": {
            "audio_path": str(audio_path),
            "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "language": "ru-RU",
            "timestamp_mode": "word",
            "context_text": None,
        },
    }


def test_nemotron_codec_passes_model_owned_language_without_a_prompt_dictionary(tmp_path: Path):
    audio_path = tmp_path / "input.wav"
    request = _nemotron_request(audio_path)
    payload = request["payload"]
    assert isinstance(payload, dict)

    decoded = decode_audio_cpp_cli_request(request)

    assert decoded == {"family": "nemotron-3.5-asr", **payload}
    assert "nemotron_asr" not in decoded
    model = tmp_path / "nemotron.gguf"
    model.write_bytes(b"model")
    arguments = build_audio_cpp_cli_arguments(
        family="nemotron-3.5-asr",
        payload=decoded,
        model_paths={"nemotron-3.5-asr": model},
        output_directory=tmp_path / "output",
    )
    assert arguments[arguments.index("--language") + 1] == "ru-RU"

    payload["nemotron_asr"] = {"prompt_dictionary": {"id": "invented"}}
    with pytest.raises(RuntimeProtocolError, match="unsupported runtime fields"):
        decode_audio_cpp_cli_request(request)


def test_native_discovery_requires_a_checksummed_colocated_dependency_closure(tmp_path: Path):
    executable, runtime_dll = _package(tmp_path)

    install = discover_native_audio_cpp_install(executable)

    assert install.executable_path == executable.resolve()
    assert install.closure_manifest_path.name == "audio_cpp_dependency_closure.json"
    model = executable.parent / "models" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"model")
    _write_closure(executable.parent)
    discover_native_audio_cpp_install(executable, required_model_paths=(model,))
    model.write_bytes(b"tampered model")
    with pytest.raises(ValueError, match="checksum"):
        discover_native_audio_cpp_install(executable, required_model_paths=(model,))
    _write_closure(executable.parent)
    runtime_dll.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        discover_native_audio_cpp_install(executable)


def test_native_windows_launcher_preserves_unicode_space_argv_decodes_words_and_cleans_up(
    monkeypatch, tmp_path: Path
):
    from voiceover_pipeline.local_runtime.transports import audio_cpp_cli

    executable, _runtime_dll = _package(tmp_path)
    asr_model = tmp_path / "модели с пробелом" / "qwen asr.gguf"
    asr_model.parent.mkdir()
    asr_model.write_bytes(b"asr")
    aligner = asr_model.with_name("qwen aligner.gguf")
    aligner.write_bytes(b"aligner")
    audio_path = tmp_path / "вход с пробелом.ogg"
    audio_path.write_bytes(b"fixture")
    transport = AudioCppNativeCLITransport(
        executable_path=executable,
        model_paths={"qwen3-asr": asr_model, "qwen3-forced-aligner": aligner},
        host_platform="win32",
        timeout_seconds=2,
    )
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0
        pid = 12345

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 2
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            return self.returncode

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        text_out = Path(command[command.index("--text-out") + 1])
        text_out.write_text("Привет мир\n", encoding="utf-8")
        words_out = Path(command[command.index("--words-out") + 1])
        words_out.write_text(
            '[{"word":"Привет","start_sample":1600,"end_sample":9600},'
            '{"word":"мир","start_sample":11200,"end_sample":14400}]',
            encoding="utf-8",
        )
        segments_out = Path(command[command.index("--segments-out") + 1])
        segments_out.write_text(
            '[{"start_s":0.1,"end_s":0.9,"text":"Привет мир"}]', encoding="utf-8"
        )
        return CompletedProcess()

    monkeypatch.setattr(audio_cpp_cli.subprocess, "Popen", fake_popen)

    response = transport.invoke("request-1", _qwen_request(audio_path))
    response_payload = response["response"]
    assert isinstance(response_payload, dict)

    command = captured["command"]
    assert command[0] == str(executable.resolve())
    assert str(asr_model.resolve()) in command
    assert str(audio_path) in command
    assert "Сохрани PostgreSQL." in command
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["creationflags"] == audio_cpp_cli._windows_creation_flags()
    assert "start_new_session" not in captured["kwargs"]
    assert response_payload["transcript"] == "Привет мир"
    assert response_payload["words"] == [
        {"text": "Привет", "start_s": 0.1, "end_s": 0.6},
        {"text": "мир", "start_s": 0.7, "end_s": 0.9},
    ]
    assert response_payload["segments"] == [{"start_s": 0.1, "end_s": 0.9, "text": "Привет мир"}]
    assert not Path(captured["kwargs"]["cwd"]).exists()


def test_native_windows_launcher_decodes_wav_receipt_and_cleans_up(monkeypatch, tmp_path: Path):
    from voiceover_pipeline.local_runtime.transports import audio_cpp_cli

    executable, _runtime_dll = _package(tmp_path)
    model = tmp_path / "omnivoice.gguf"
    model.write_bytes(b"model")
    transport = AudioCppNativeCLITransport(
        executable_path=executable,
        model_paths={"omnivoice": model},
        host_platform="win32",
        timeout_seconds=2,
    )
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0
        pid = 777

        def poll(self) -> int:
            return self.returncode

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 2
            return "", ""

        def wait(self, *, timeout: float) -> int:
            assert timeout == 2
            return self.returncode

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        _write_wav(Path(command[command.index("--out") + 1]))
        return CompletedProcess()

    monkeypatch.setattr(audio_cpp_cli.subprocess, "Popen", fake_popen)

    response = transport.invoke(
        "voice-1",
        {
            "schema_version": 1,
            "operation": "tts",
            "family": "omnivoice",
            "provider_id": "omnivoice-local",
            "payload": {
                "text": "Привет",
                "model_id": "audio-cpp/omnivoice-q8_0",
                "voice": None,
                "language": "ru",
                "instruction": "female",
                "text_chunk_size": 420,
                "seed": 1,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
            },
        },
    )
    receipt = response["response"]
    assert isinstance(receipt, dict)

    assert response["schema_version"] == 1
    assert response["request_id"] == "voice-1"
    assert receipt["audio_format"] == "wav"
    assert receipt["sample_rate_hz"] == 24_000
    assert receipt["channels"] == 1
    assert receipt["duration_s"] == 0.01
    assert isinstance(receipt["audio_bytes"], bytes)
    assert receipt["audio_bytes"][:4] == b"RIFF"
    assert not Path(captured["kwargs"]["cwd"]).exists()


def test_native_windows_launcher_cancels_before_and_during_launch(monkeypatch, tmp_path: Path):
    from voiceover_pipeline.local_runtime.transports import audio_cpp_cli

    executable, _runtime_dll = _package(tmp_path)
    model = tmp_path / "omnivoice.gguf"
    model.write_bytes(b"model")
    transport = AudioCppNativeCLITransport(
        executable_path=executable,
        model_paths={"omnivoice": model},
        host_platform="win32",
        timeout_seconds=2,
    )
    prelaunch_request = {
        "schema_version": 1,
        "operation": "tts",
        "family": "omnivoice",
        "provider_id": "omnivoice-local",
        "payload": {
            "text": "Привет",
            "model_id": "audio-cpp/omnivoice-q8_0",
            "voice": None,
            "language": "ru",
            "instruction": "female",
            "text_chunk_size": 420,
            "seed": 1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
    }
    transport.cancel("before")
    with pytest.raises(RuntimeTransportError, match="cancelled"):
        transport.invoke("before", prelaunch_request)

    started = threading.Event()
    terminate_called = threading.Event()
    failures: list[Exception] = []

    class BlockingProcess:
        pid = 54321
        returncode = 0

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 2
            started.set()
            assert terminate_called.wait(timeout=1)
            return "", ""

        def poll(self):
            return None

        def wait(self, *, timeout: float) -> int:
            return 0

    monkeypatch.setattr(
        audio_cpp_cli.subprocess, "Popen", lambda *_args, **_kwargs: BlockingProcess()
    )
    monkeypatch.setattr(transport, "_terminate_process", lambda _process: terminate_called.set())

    def invoke() -> None:
        try:
            transport.invoke("during", prelaunch_request)
        except Exception as exc:  # Capture the public cancellation contract.
            failures.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert started.wait(timeout=1)
    transport.cancel("during")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeTransportError)
    assert str(failures[0]) == "audio.cpp native invocation cancelled"


def test_shared_family_codec_uses_no_container_paths_for_all_declared_families(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    output = tmp_path / "output"
    output.mkdir()
    families = {
        "qwen3-asr": {
            "operation": "asr",
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "audio_path": str(tmp_path / "input.wav"),
            "language": None,
            "timestamp_mode": "none",
            "context_text": None,
        },
        "nemotron-3.5-asr": {
            "operation": "asr",
            "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "audio_path": str(tmp_path / "input.wav"),
            "language": "ru-RU",
            "timestamp_mode": "word",
            "context_text": None,
        },
        "qwen3-tts": {
            "operation": "tts",
            "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "text": "Привет",
            "voice": "ryan",
            "language": "ru",
        },
        "omnivoice": {
            "operation": "tts",
            "model_id": "audio-cpp/omnivoice-q8_0",
            "text": "Привет",
            "voice": None,
            "language": "ru",
            "instruction": "female",
            "text_chunk_size": 420,
            "seed": 1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
    }

    for family, payload in families.items():
        command = build_audio_cpp_cli_arguments(
            family=family,
            payload=payload,
            model_paths={family: model},
            output_directory=output,
        )
        assert command[:4] == (
            "--task",
            payload["operation"],
            "--family",
            family.replace("-", "_").replace(".", "_"),
        )
        assert all("docker" not in part.casefold() and "/app" not in part for part in command)


def test_legacy_json_transport_uses_windows_process_options_and_utf8(monkeypatch):
    from voiceover_pipeline.local_runtime.transports import subprocess as json_transport
    from voiceover_pipeline.local_runtime.transports.subprocess import SubprocessJSONTransport

    transport = SubprocessJSONTransport(
        ("audiocpp_cli.exe", "--json"), timeout_seconds=2, host_platform="win32"
    )
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0
        pid = 123

        def communicate(self, text: str, *, timeout: float) -> tuple[str, str]:
            assert "Привет" in text
            assert timeout == 2
            return '{"schema_version":1}', ""

        def poll(self) -> int:
            return self.returncode

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(json_transport.subprocess, "Popen", fake_popen)

    assert transport.invoke("request-1", {"text": "Привет"}) == {"schema_version": 1}

    assert captured["command"] == ("audiocpp_cli.exe", "--json")
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["creationflags"] == getattr(
        json_transport.subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    assert "start_new_session" not in captured["kwargs"]
    assert not Path(captured["kwargs"]["cwd"]).exists()


def test_windows_provider_selection_requires_native_package_and_never_defaults_to_docker(
    monkeypatch, tmp_path: Path
):
    from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
        NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    )
    from voiceover_pipeline.local_runtime.transports.audio_cpp_omnivoice import (
        AudioCppOmniVoiceCLITransport,
        VerifiedOmniVoiceModel,
    )
    from voiceover_pipeline.providers import audio_cpp_omnivoice_tts, audio_cpp_qwen_asr

    executable, _runtime_dll = _package(tmp_path)
    model_directory = executable.parent / "models"
    model_directory.mkdir()
    asr_model = model_directory / "qwen-asr.gguf"
    aligner = model_directory / "qwen-aligner.gguf"
    omnivoice_model = model_directory / "omnivoice.gguf"
    for path in (asr_model, aligner, omnivoice_model):
        path.write_bytes(b"model")
    _write_closure(executable.parent)
    monkeypatch.setattr(audio_cpp_qwen_asr.sys, "platform", "win32")
    monkeypatch.setattr(audio_cpp_omnivoice_tts.sys, "platform", "win32")
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", "must-not-be-selected")
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL", str(asr_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL", str(aligner))
    monkeypatch.setenv("VOICEOVER_OMNIVOICE_MODEL", str(omnivoice_model))
    monkeypatch.setenv(
        "VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE", "accept-cc-by-nc-4.0-local-use"
    )
    monkeypatch.setenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, str(executable))
    monkeypatch.setattr(
        audio_cpp_omnivoice_tts,
        "admit_omnivoice_model",
        lambda *, model_path, model_id: VerifiedOmniVoiceModel(
            model_path=model_path.resolve(),
            model_id=model_id,
            sha256="2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b",
            quantization="Q8_0 GGUF",
            license="CC-BY-NC-4.0 upstream weights; local noncommercial research only",
            provenance="fixture provenance",
        ),
    )

    qwen = audio_cpp_qwen_asr.AudioCppQwenASRProvider.from_environment()
    omnivoice = audio_cpp_omnivoice_tts.OmniVoiceLocalTTSProvider.from_environment()

    assert qwen._runtime is not None
    assert omnivoice._runtime is not None
    qwen_driver = next(iter(qwen._runtime._registry._drivers.values()))
    omnivoice_driver = next(iter(omnivoice._runtime._registry._drivers.values()))
    assert isinstance(qwen_driver._transport, AudioCppNativeCLITransport)
    assert not isinstance(omnivoice_driver._transport, AudioCppOmniVoiceCLITransport)
    assert omnivoice_driver._transport_name == "native-cli"

    monkeypatch.delenv(NATIVE_AUDIO_CPP_EXECUTABLE_ENV)
    assert audio_cpp_qwen_asr.AudioCppQwenASRProvider.from_environment()._runtime is None


def test_model_directory_package_resolves_single_gguf_for_cli_arguments(tmp_path: Path):
    model_directory = tmp_path / "models" / "omnivoice"
    model_directory.mkdir(parents=True)
    model_file = model_directory / "omnivoice-q8_0.gguf"
    model_file.write_bytes(b"model")

    command = build_audio_cpp_cli_arguments(
        family="omnivoice",
        payload={
            "text": "Привет",
            "model_id": "audio-cpp/omnivoice-q8_0",
            "voice": None,
            "language": "ru",
            "instruction": "female",
            "text_chunk_size": 420,
            "seed": 1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
        model_paths={"omnivoice": model_directory},
        output_directory=tmp_path / "output",
    )

    assert command[command.index("--model") + 1] == str(model_file.resolve())


def test_model_directory_without_gguf_artifact_fails_closed(tmp_path: Path):
    empty_directory = tmp_path / "models" / "empty"
    empty_directory.mkdir(parents=True)

    with pytest.raises(RuntimeTransportError, match="no GGUF artifact"):
        build_audio_cpp_cli_arguments(
            family="omnivoice",
            payload={
                "text": "Привет",
                "model_id": "audio-cpp/omnivoice-q8_0",
                "voice": None,
                "language": "ru",
                "instruction": "female",
                "text_chunk_size": 420,
                "seed": 1,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
            },
            model_paths={"omnivoice": empty_directory},
            output_directory=tmp_path / "output",
        )


def test_model_directory_with_multiple_gguf_artifacts_fails_closed(tmp_path: Path):
    model_directory = tmp_path / "models" / "ambiguous"
    model_directory.mkdir(parents=True)
    (model_directory / "a.gguf").write_bytes(b"a")
    (model_directory / "b.gguf").write_bytes(b"b")

    with pytest.raises(RuntimeTransportError, match="exactly one GGUF artifact"):
        build_audio_cpp_cli_arguments(
            family="omnivoice",
            payload={
                "text": "Hello",
                "voice": None,
                "language": "ru",
                "instruction": "instruction",
                "text_chunk_size": 420,
                "seed": 1,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
            },
            model_paths={"omnivoice": model_directory},
            output_directory=tmp_path / "output",
        )


def test_transport_accepts_model_directory_packages(monkeypatch, tmp_path: Path):
    from voiceover_pipeline.local_runtime.transports import audio_cpp_cli

    executable, _runtime_dll = _package(tmp_path)
    model_directory = tmp_path / "models" / "omnivoice"
    model_directory.mkdir(parents=True)
    model_file = model_directory / "omnivoice-q8_0.gguf"
    model_file.write_bytes(b"model")
    transport = AudioCppNativeCLITransport(
        executable_path=executable,
        model_paths={"omnivoice": model_directory},
        host_platform="win32",
        timeout_seconds=2,
    )
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0
        pid = 99

        def poll(self) -> int:
            return self.returncode

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            return "", ""

        def wait(self, *, timeout: float) -> int:
            return self.returncode

    def fake_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["kwargs"] = kwargs
        _write_wav(Path(command[command.index("--out") + 1]))
        return CompletedProcess()

    monkeypatch.setattr(audio_cpp_cli.subprocess, "Popen", fake_popen)

    response = transport.invoke(
        "voice-1",
        {
            "schema_version": 1,
            "operation": "tts",
            "family": "omnivoice",
            "provider_id": "omnivoice-local",
            "payload": {
                "text": "Привет",
                "model_id": "audio-cpp/omnivoice-q8_0",
                "voice": None,
                "language": "ru",
                "instruction": "female",
                "text_chunk_size": 420,
                "seed": 1,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
            },
        },
    )

    assert captured["command"][captured["command"].index("--model") + 1] == str(
        model_file.resolve()
    )
    assert response["ok"] is True


def test_linux_container_launchers_fail_closed_on_windows(monkeypatch, tmp_path: Path):
    from voiceover_pipeline.local_runtime.transports import audio_cpp_container, audio_cpp_omnivoice

    qwen_model = tmp_path / "qwen.gguf"
    aligner_model = tmp_path / "aligner.gguf"
    omnivoice_model = tmp_path / "omnivoice.gguf"
    for path in (qwen_model, aligner_model, omnivoice_model):
        path.write_bytes(b"model")
    monkeypatch.setattr(audio_cpp_container.sys, "platform", "win32")
    monkeypatch.setattr(audio_cpp_omnivoice.sys, "platform", "win32")

    with pytest.raises(ValueError, match="unavailable on Windows"):
        audio_cpp_container.AudioCppContainerCLITransport(
            asr_model_path=qwen_model,
            forced_aligner_model_path=aligner_model,
        )
    with pytest.raises(ValueError, match="unavailable on Windows"):
        audio_cpp_omnivoice.AudioCppOmniVoiceCLITransport(
            model=audio_cpp_omnivoice.VerifiedOmniVoiceModel(
                model_path=omnivoice_model,
                model_id="audio-cpp/omnivoice-q8_0",
                sha256="2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b",
                quantization="Q8_0 GGUF",
                license="CC-BY-NC-4.0 upstream weights; local noncommercial research only",
                provenance="fixture provenance",
            ),
        )
