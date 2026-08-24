from __future__ import annotations

import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import voiceover_pipeline.local_runtime.transports.audio_cpp_container as container
from voiceover_pipeline.local_runtime.contracts import RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.transports.audio_cpp_container import (
    PINNED_AUDIO_CPP_CONTAINER_IMAGE,
    AudioCppContainerCLITransport,
)


def _write_mono_wav(path: Path, *, frames: int = 16_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * frames)


def _transport(tmp_path: Path) -> AudioCppContainerCLITransport:
    asr_model = tmp_path / "qwen-asr.gguf"
    aligner_model = tmp_path / "qwen-aligner.gguf"
    asr_model.write_bytes(b"asr")
    aligner_model.write_bytes(b"aligner")
    return AudioCppContainerCLITransport(
        asr_model_path=asr_model,
        forced_aligner_model_path=aligner_model,
        timeout_seconds=2,
    )


def _word_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": "request-1",
        "operation": "asr",
        "family": "qwen3-asr",
        "provider_id": "qwen-local",
        "payload": {
            "audio_path": "input.ogg",
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "language": "Russian",
            "timestamp_mode": "word",
            "context_text": "Keep PostgreSQL as written.",
        },
    }


def _output_directory(command: tuple[str, ...]) -> Path:
    mount = next(value for value in command if "dst=/output" in value)
    return Path(next(part[4:] for part in mount.split(",") if part.startswith("src=")))


def _configured_container_command(provider: Any) -> tuple[str, ...]:
    driver = next(iter(provider._runtime._registry._drivers.values()))
    assert isinstance(driver._transport, AudioCppContainerCLITransport)
    return driver._transport._container_command


def test_container_cli_transport_constructs_pinned_isolated_command_and_maps_words(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.ogg"
    source.write_bytes(b"fixture")
    transport = _transport(tmp_path)
    ffmpeg_commands: list[tuple[str, ...]] = []
    container_commands: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        ffmpeg_commands.append(tuple(command))
        _write_mono_wav(Path(command[-1]))
        return SimpleNamespace(returncode=0)

    class FixtureProcess:
        pid = 12345
        returncode = 0

        def __init__(self, command, **_kwargs):
            command = tuple(command)
            container_commands.append(command)
            output_directory = _output_directory(command)
            (output_directory / "transcript.txt").write_text("Привет мир\n", encoding="utf-8")
            (output_directory / "words.json").write_text(
                '[{"word":"Привет","start_sample":1600,"end_sample":9600,"confidence":0.0},'
                '{"word":"мир","start_sample":11200,"end_sample":14400,"confidence":0.9}]',
                encoding="utf-8",
            )

        def communicate(self, *, timeout):
            assert timeout == 2
            return b"", b""

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            return self.returncode

    monkeypatch.setattr(container.subprocess, "run", fake_run)
    monkeypatch.setattr(container.subprocess, "Popen", FixtureProcess)
    request = _word_request()
    payload = request["payload"]
    assert isinstance(payload, dict)
    request["payload"] = {**payload, "audio_path": str(source)}

    response = transport.invoke("request-1", request)

    assert ffmpeg_commands == [
        (
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            ffmpeg_commands[0][-1],
        )
    ]
    assert len(container_commands) == 1
    command = container_commands[0]
    assert command[:13] == (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--gpus",
        "all",
        "--entrypoint",
        "/app/audiocpp_cli",
        "--mount",
    )
    assert PINNED_AUDIO_CPP_CONTAINER_IMAGE in command
    assert "--family" in command
    assert command[command.index("--family") + 1] == "qwen3_asr"
    assert "--session-option" in command
    assert (
        command[command.index("--session-option") + 1]
        == "qwen3_asr.forced_aligner_model_path=/models/qwen3-forced-aligner.gguf"
    )
    assert "--words-out" in command
    assert "--request-option" not in command
    assert any("dst=/models/qwen3-asr.gguf,readonly" in value for value in command)
    assert any("dst=/models/qwen3-forced-aligner.gguf,readonly" in value for value in command)
    assert any("dst=/input/audio.wav,readonly" in value for value in command)
    assert response == {
        "schema_version": 1,
        "request_id": "request-1",
        "ok": True,
        "response": {
            "transcript": "Привет мир",
            "duration_s": 1.0,
            "forced_aligner_available": True,
            "words": [
                {
                    "word": "Привет",
                    "start_sample": 1600,
                    "end_sample": 9600,
                    "confidence": 0.0,
                    "text": "Привет",
                    "start_s": 0.1,
                    "end_s": 0.6,
                },
                {
                    "word": "мир",
                    "start_sample": 11200,
                    "end_sample": 14400,
                    "confidence": 0.9,
                    "text": "мир",
                    "start_s": 0.7,
                    "end_s": 0.9,
                },
            ],
        },
    }


def test_container_cli_transport_rejects_malformed_word_boundaries(tmp_path):
    words_path = tmp_path / "words.json"
    words_path.write_text('[{"word":"ошибка","start_sample":20,"end_sample":10}]', encoding="utf-8")

    with pytest.raises(RuntimeProtocolError, match="reversed boundaries"):
        AudioCppContainerCLITransport._decode_words(words_path)


def test_container_cli_transport_cancellation_stops_the_active_container(tmp_path, monkeypatch):
    source = tmp_path / "input.ogg"
    source.write_bytes(b"fixture")
    transport = _transport(tmp_path)
    started = threading.Event()
    terminated = threading.Event()

    def fake_run(command, **_kwargs):
        _write_mono_wav(Path(command[-1]))
        return SimpleNamespace(returncode=0)

    class BlockingProcess:
        pid = 54321
        returncode = 0

        def __init__(self, _command, **_kwargs):
            pass

        def communicate(self, *, timeout):
            assert timeout == 2
            started.set()
            assert terminated.wait(timeout=1)
            return b"", b""

        def poll(self):
            return None

        def wait(self, *, timeout):
            return 0

    monkeypatch.setattr(container.subprocess, "run", fake_run)
    monkeypatch.setattr(container.subprocess, "Popen", BlockingProcess)
    monkeypatch.setattr(transport, "_terminate_process", lambda _process: terminated.set())
    request = _word_request()
    payload = request["payload"]
    assert isinstance(payload, dict)
    request["payload"] = {**payload, "audio_path": str(source)}
    failures: list[Exception] = []

    def invoke() -> None:
        try:
            transport.invoke("request-1", request)
        except Exception as exc:  # Fixture must retain the public cancellation error.
            failures.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert started.wait(timeout=1)

    transport.cancel("request-1")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeTransportError)
    assert str(failures[0]) == "audio.cpp container invocation cancelled"


def test_container_cli_transport_pre_staging_cancellation_never_starts_container(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.ogg"
    source.write_bytes(b"fixture")
    transport = _transport(tmp_path)
    ffmpeg_commands: list[tuple[str, ...]] = []
    container_commands: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        ffmpeg_commands.append(tuple(command))
        raise AssertionError("cancelled invocation must not stage audio")

    class FixtureProcess:
        def __init__(self, command, **_kwargs):
            container_commands.append(tuple(command))
            raise AssertionError("cancelled invocation must not launch a container")

    monkeypatch.setattr(container.subprocess, "run", fake_run)
    monkeypatch.setattr(container.subprocess, "Popen", FixtureProcess)
    request = _word_request()
    payload = request["payload"]
    assert isinstance(payload, dict)
    request["payload"] = {**payload, "audio_path": str(source)}

    transport.cancel("request-1")

    with pytest.raises(RuntimeTransportError, match="invocation cancelled"):
        transport.invoke("request-1", request)

    assert ffmpeg_commands == []
    assert container_commands == []
    assert transport._processes == {}
    assert transport._cancelled == set()


def test_container_cli_transport_cancellation_during_staging_never_starts_container(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.ogg"
    source.write_bytes(b"fixture")
    transport = _transport(tmp_path)
    staging_started = threading.Event()
    allow_staging_to_finish = threading.Event()
    container_commands: list[tuple[str, ...]] = []
    failures: list[Exception] = []

    def fake_run(command, **_kwargs):
        staging_started.set()
        assert allow_staging_to_finish.wait(timeout=1)
        _write_mono_wav(Path(command[-1]))
        return SimpleNamespace(returncode=0)

    class FixtureProcess:
        def __init__(self, command, **_kwargs):
            container_commands.append(tuple(command))
            raise AssertionError("cancelled staging must not launch a container")

    monkeypatch.setattr(container.subprocess, "run", fake_run)
    monkeypatch.setattr(container.subprocess, "Popen", FixtureProcess)
    request = _word_request()
    payload = request["payload"]
    assert isinstance(payload, dict)
    request["payload"] = {**payload, "audio_path": str(source)}

    def invoke() -> None:
        try:
            transport.invoke("request-1", request)
        except Exception as exc:  # Fixture retains the public cancellation error.
            failures.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert staging_started.wait(timeout=1)

    transport.cancel("request-1")
    allow_staging_to_finish.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeTransportError)
    assert str(failures[0]) == "audio.cpp container invocation cancelled"
    assert container_commands == []
    assert transport._processes == {}
    assert transport._cancelled == set()


def test_container_configuration_selects_the_qwen_audio_cpp_provider(tmp_path, monkeypatch):
    from voiceover_pipeline.providers.audio_cpp_qwen_asr import AudioCppQwenASRProvider
    from voiceover_pipeline.providers.qwen_asr_local import qwen_asr_provider_factory

    asr_model = tmp_path / "qwen-asr.gguf"
    aligner_model = tmp_path / "qwen-aligner.gguf"
    asr_model.write_bytes(b"asr")
    aligner_model.write_bytes(b"aligner")
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", raising=False)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", PINNED_AUDIO_CPP_CONTAINER_IMAGE)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL", str(asr_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL", str(aligner_model))

    provider = qwen_asr_provider_factory()

    assert isinstance(provider, AudioCppQwenASRProvider)
    assert provider._runtime is not None
    assert _configured_container_command(provider) == ("docker",)


def test_container_configuration_accepts_reviewed_sudo_argv_prefix(tmp_path, monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import qwen_asr_provider_factory

    asr_model = tmp_path / "qwen-asr.gguf"
    aligner_model = tmp_path / "qwen-aligner.gguf"
    asr_model.write_bytes(b"asr")
    aligner_model.write_bytes(b"aligner")
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", PINNED_AUDIO_CPP_CONTAINER_IMAGE)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL", str(asr_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL", str(aligner_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", '["sudo", "-n", "docker"]')

    provider = qwen_asr_provider_factory()

    assert provider._runtime is not None
    assert _configured_container_command(provider) == ("sudo", "-n", "docker")


def test_container_configuration_rejects_shell_like_command_string(tmp_path, monkeypatch):
    from voiceover_pipeline.providers.qwen_asr_local import qwen_asr_provider_factory

    asr_model = tmp_path / "qwen-asr.gguf"
    aligner_model = tmp_path / "qwen-aligner.gguf"
    asr_model.write_bytes(b"asr")
    aligner_model.write_bytes(b"aligner")
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", PINNED_AUDIO_CPP_CONTAINER_IMAGE)
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL", str(asr_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL", str(aligner_model))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", '"sudo -n docker"')

    with pytest.raises(ValueError, match="JSON array"):
        qwen_asr_provider_factory()
