from __future__ import annotations

import base64
import os
import subprocess
import sys
import threading
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pytest

from voiceover_pipeline.cli import _resolve_qwen_mode_identity
from voiceover_pipeline.config import (
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)
from voiceover_pipeline.local_runtime.contracts import (
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    LocalTTSRequest,
    LocalTTSResponse,
    RuntimeDriverHealth,
    RuntimeExecutionReceipt,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.drivers.audio_cpp import AudioCppRuntimeDriver
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry
from voiceover_pipeline.local_runtime.transports.subprocess import SubprocessJSONTransport
from voiceover_pipeline.providers.audio_cpp_qwen_tts import (
    QWEN_TTS_FAMILY,
    AudioCppQwenTTSProvider,
    qwen_tts_audio_cpp_dependency_probe,
)


def test_qwen_base_model_can_be_loaded_from_configured_local_path(tmp_path: Path) -> None:
    local_model = tmp_path / "Qwen3-TTS-Base"
    env = os.environ.copy()
    env["VOICEOVER_QWEN_TTS_BASE_MODEL"] = str(local_model)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from voiceover_pipeline.config import QWEN_MODEL_BASE; print(QWEN_MODEL_BASE)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == str(local_model)


@pytest.mark.parametrize(
    ("mode", "expected_model", "expected_voice"),
    [
        ("preset", QWEN_MODEL_CUSTOMVOICE, None),
        ("clone", QWEN_MODEL_BASE, "clone"),
        ("design", QWEN_MODEL_VOICE_DESIGN, "design"),
    ],
)
def test_qwen_mode_controls_artifact_identity(
    mode: str, expected_model: str, expected_voice: str | None
) -> None:
    from argparse import Namespace

    args = Namespace(provider="qwen-local", mode=mode, model="wrong-model", voice=None)

    _resolve_qwen_mode_identity(args)

    assert args.model == expected_model
    assert args.voice == expected_voice


def _wav_bytes(*, sample_rate_hz: int = 24_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


def _qwen_model_package(tmp_path: Path) -> Path:
    package = tmp_path / "Qwen3-TTS-package"
    package.mkdir()
    for filename in ("model.safetensors", "tokenizer_config.json"):
        (package / filename).write_bytes(b"fixture model")
    (package / "config.json").write_text(
        '{"model_type": "qwen3_tts", "tts_model_size": "1b7", "tts_model_type": "custom_voice"}',
        encoding="utf-8",
    )
    (package / "speech_tokenizer").mkdir()
    return package


@dataclass
class _Runtime:
    response: LocalTTSResponse
    requests: list[Any] = field(default_factory=list)

    def execute_tts(self, request, *, runtime_choice: str) -> LocalTTSResponse:
        assert runtime_choice == "auto"
        self.requests.append(request)
        return self.response


def _response() -> LocalTTSResponse:
    return LocalTTSResponse(
        audio_bytes=_wav_bytes(),
        audio_format="wav",
        payload={"sample_rate_hz": 24_000, "channels": 1, "duration_s": 0.01},
        receipt=RuntimeExecutionReceipt(
            driver_id="audio-cpp",
            transport="subprocess-json",
            source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
            build_hash="fixture-build",
        ),
    )


@pytest.mark.parametrize(
    ("mode", "expected_model", "expected_runtime_mode"),
    [
        ("preset", QWEN_MODEL_CUSTOMVOICE, "custom-voice"),
        ("clone", QWEN_MODEL_BASE, "voice-clone"),
        ("design", QWEN_MODEL_VOICE_DESIGN, "voice-design"),
    ],
)
def test_qwen_audio_cpp_provider_keeps_mode_specific_runtime_fields(
    tmp_path: Path, mode: str, expected_model: str, expected_runtime_mode: str
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(_wav_bytes())
    model_artifact = tmp_path / f"{mode}-model.gguf"
    model_artifact.write_bytes(b"fixture model")
    runtime = _Runtime(_response())
    provider = AudioCppQwenTTSProvider(
        runtime,
        mode=mode,
        voice="Sohee",
        instruct="calm and clear",
        sample_path=str(reference),
        sample_text="Reference text",
        model_artifact_path=model_artifact,
    )

    result = provider.synthesize_chunk("Привет, мир!", "chunk-01")

    assert len(runtime.requests) == 1
    request = runtime.requests[0].to_runtime_request()
    assert request.family == QWEN_TTS_FAMILY
    assert request.provider_id == "qwen-local"
    assert request.payload["model_id"] == expected_model
    assert request.payload["model_artifact_path"] == str(model_artifact)
    assert request.payload["mode"] == expected_runtime_mode
    assert "audio_cpp" not in request.payload
    if mode == "preset":
        assert request.payload["voice"] == "Sohee"
        assert request.payload["instruction"] == "calm and clear"
        assert "reference_audio_path" not in request.payload
    elif mode == "clone":
        assert request.payload["voice"] is None
        assert request.payload["reference_audio_path"] == str(reference)
        assert request.payload["reference_text"] == "Reference text"
        assert "instruction" not in request.payload
    else:
        assert request.payload["voice"] is None
        assert request.payload["instruction"] == "calm and clear"
        assert "reference_audio_path" not in request.payload
    assert result.audio_format == "wav"
    assert result.audio_bytes == _wav_bytes()
    assert result.raw_metadata["model_id"] == expected_model
    assert result.raw_metadata["mode"] == mode
    assert result.raw_metadata["sample_rate_hz"] == 24_000
    assert result.raw_metadata["runtime"]["driver_id"] == "audio-cpp"
    assert str(tmp_path) not in str(result.raw_metadata)


def test_qwen_audio_cpp_provider_rejects_missing_mode_inputs_and_malformed_wav(
    tmp_path: Path,
) -> None:
    runtime = _Runtime(_response())

    with pytest.raises(FileNotFoundError, match="Reference audio"):
        AudioCppQwenTTSProvider(runtime, mode="clone").synthesize_chunk("text", "chunk")
    assert runtime.requests == []

    reference = tmp_path / "reference.wav"
    reference.write_bytes(_wav_bytes())
    with pytest.raises(ValueError, match="reference text"):
        AudioCppQwenTTSProvider(runtime, mode="clone", sample_path=str(reference)).synthesize_chunk(
            "text", "chunk"
        )
    assert runtime.requests == []

    with pytest.raises(ValueError, match="VoiceDesign"):
        AudioCppQwenTTSProvider(runtime, mode="design", instruct="").synthesize_chunk(
            "text", "chunk"
        )
    assert runtime.requests == []

    malformed = _Runtime(LocalTTSResponse(audio_bytes=b"not a wav", audio_format="wav"))
    with pytest.raises(RuntimeUnavailableError, match="valid WAV"):
        AudioCppQwenTTSProvider(malformed).synthesize_chunk("text", "chunk")


def test_qwen_audio_cpp_dependency_probe_fails_closed_when_resources_are_absent(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", raising=False)

    health = qwen_tts_audio_cpp_dependency_probe()

    assert health.available is False
    assert "VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL" in health.remediation


@pytest.mark.parametrize("command", ["not-json", "[]", '["docker", 1]'])
def test_qwen_audio_cpp_dependency_probe_fails_closed_for_invalid_container_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL", str(_qwen_model_package(tmp_path)))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", command)
    monkeypatch.setattr(
        "voiceover_pipeline.providers.audio_cpp_qwen_tts.which", lambda _command: "/usr/bin/docker"
    )

    health = qwen_tts_audio_cpp_dependency_probe()

    assert health.available is False


def test_qwen_cli_selects_audio_cpp_from_environment_and_carries_the_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voiceover_pipeline.cli import build_parser, build_provider

    model_artifact = _qwen_model_package(tmp_path)
    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "audio-cpp")
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL", str(model_artifact))
    monkeypatch.setenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", '["docker"]')
    monkeypatch.setattr(
        "voiceover_pipeline.providers.audio_cpp_qwen_tts.which", lambda _command: "/usr/bin/docker"
    )

    provider = build_provider(
        build_parser().parse_args("generate --provider qwen-local".split()),
        api_key="",
        style_prompt=None,
        prompt_mode="none",
    )

    assert isinstance(provider, AudioCppQwenTTSProvider)
    assert provider._model_artifact_path == model_artifact


def test_qwen_cli_audio_cpp_selection_fails_closed_without_admitted_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voiceover_pipeline.cli import build_parser, build_provider

    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "audio-cpp")
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_BINARY", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL", raising=False)
    monkeypatch.delenv("VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON", raising=False)

    provider = build_provider(
        build_parser().parse_args("generate --provider qwen-local".split()),
        api_key="",
        style_prompt=None,
        prompt_mode="none",
    )

    assert isinstance(provider, AudioCppQwenTTSProvider)
    with pytest.raises(ModuleNotFoundError, match="audio.cpp Qwen TTS runtime is unavailable"):
        provider.synthesize_chunk("text", "chunk")


def test_qwen_cli_rejects_unknown_runtime_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from voiceover_pipeline.cli import CliError, build_parser, build_provider

    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "automatic")

    with pytest.raises(CliError, match="VOICEOVER_QWEN_TTS_RUNTIME") as error:
        build_provider(
            build_parser().parse_args("generate --provider qwen-local".split()),
            api_key="",
            style_prompt=None,
            prompt_mode="none",
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("mode", "model_id"),
    [
        ("custom-voice", QWEN_MODEL_CUSTOMVOICE),
        ("voice-clone", QWEN_MODEL_BASE),
        ("voice-design", QWEN_MODEL_VOICE_DESIGN),
    ],
)
def test_qwen_audio_cpp_subprocess_consumes_typed_model_artifact(
    tmp_path: Path,
    mode: Literal["custom-voice", "voice-clone", "voice-design"],
    model_id: str,
) -> None:
    model_artifact = tmp_path / f"{mode}.gguf"
    model_artifact.write_bytes(b"fixture model")
    script = tmp_path / "fixture_driver.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['payload']['model_artifact_path'] == sys.argv[1]\n"
        "assert request['payload']['mode'] == sys.argv[2]\n"
        "json.dump({'schema_version': 1, 'request_id': request['request_id'], 'ok': True, "
        "'response': {'consumed_model_artifact_path': request['payload']['model_artifact_path']}}, sys.stdout)\n"
    )
    driver = AudioCppRuntimeDriver(
        binary_path=Path(sys.executable),
        source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
        transport=SubprocessJSONTransport(
            (sys.executable, str(script), str(model_artifact), mode), timeout_seconds=2
        ),
    )
    request = LocalTTSRequest(
        request_id=f"{mode}-request",
        family=QWEN_TTS_FAMILY,
        provider_id="qwen-local",
        text="fixture text",
        model_id=model_id,
        model_artifact_path=model_artifact,
        mode=mode,
    )

    response = driver.invoke(request.to_runtime_request())

    assert response.payload["consumed_model_artifact_path"] == str(model_artifact)


def test_qwen_audio_cpp_subprocess_materializes_private_wav_before_cleanup(tmp_path: Path) -> None:
    model_artifact = tmp_path / "qwen3-tts.gguf"
    model_artifact.write_bytes(b"fixture model")
    workspace_marker = tmp_path / "private-workspace.txt"
    wav_bytes = _wav_bytes()
    script = tmp_path / "fixture_tts_driver.py"
    script.write_text(
        "import base64, json, pathlib, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['operation'] == 'tts'\n"
        "assert request['payload']['model_artifact_path'] == sys.argv[2]\n"
        "workspace = pathlib.Path.cwd()\n"
        "pathlib.Path(sys.argv[1]).write_text(str(workspace), encoding='utf-8')\n"
        f"(workspace / 'output.wav').write_bytes(base64.b64decode({base64.b64encode(wav_bytes)!r}))\n"
        "json.dump({'schema_version': 1, 'request_id': request['request_id'], 'ok': True, "
        "'response': {'audio_path': 'output.wav'}}, sys.stdout)\n"
    )
    driver = AudioCppRuntimeDriver(
        binary_path=Path(sys.executable),
        source_revision="502b5b74bd26e9b4aed267d1776ecf131cae7215",
        transport=SubprocessJSONTransport(
            (sys.executable, str(script), str(workspace_marker), str(model_artifact)),
            timeout_seconds=2,
        ),
    )
    provider = AudioCppQwenTTSProvider(
        LocalAudioRuntime(LocalRuntimeRegistry((driver,)), promoted_families=(QWEN_TTS_FAMILY,)),
        voice="Aiden",
        model_artifact_path=model_artifact,
    )

    result = provider.synthesize_chunk("Привет, мир!", "chunk-01")

    assert result.audio_format == "wav"
    assert result.audio_bytes == wav_bytes
    with wave.open(BytesIO(result.audio_bytes), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24_000
        assert audio.getnframes() > 0
    workspace = Path(workspace_marker.read_text(encoding="utf-8"))
    assert not workspace.exists()
    assert "audio_path" not in result.raw_metadata
    assert result.raw_metadata["runtime"]["transport"] == "subprocess-json"


def test_qwen_audio_cpp_runtime_cancellation_and_unload_release_the_driver() -> None:
    started = threading.Event()
    cancelled = threading.Event()

    class BlockingDriver:
        driver_id = "audio-cpp"

        def __init__(self) -> None:
            self.request: LocalRuntimeRequest | None = None
            self.cancelled: list[str] = []
            self.closed = 0

        def health(self) -> RuntimeDriverHealth:
            return RuntimeDriverHealth(available=True)

        def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
            self.request = request
            started.set()
            assert cancelled.wait(timeout=2)
            return LocalRuntimeResponse(
                request_id=request.request_id,
                payload={
                    "audio_bytes": _wav_bytes(),
                    "audio_format": "wav",
                    "sample_rate_hz": 24_000,
                },
            )

        def cancel(self, request_id: str) -> None:
            self.cancelled.append(request_id)
            cancelled.set()

        def close(self) -> None:
            self.closed += 1

    driver = BlockingDriver()
    runtime = LocalAudioRuntime(
        LocalRuntimeRegistry((driver,)), promoted_families=(QWEN_TTS_FAMILY,)
    )
    provider = AudioCppQwenTTSProvider(runtime, voice="Aiden")
    results: list[Any] = []
    worker = threading.Thread(
        target=lambda: results.append(provider.synthesize_chunk("text", "chunk"))
    )

    worker.start()
    assert started.wait(timeout=1)
    assert driver.request is not None
    runtime.cancel(driver.request.request_id, family=QWEN_TTS_FAMILY)
    worker.join(timeout=2)
    runtime.unload(QWEN_TTS_FAMILY)

    assert not worker.is_alive()
    assert len(results) == 1
    assert driver.cancelled == [driver.request.request_id]
    assert driver.closed == 1


def test_qwen_public_provider_keeps_python_default_and_allows_internal_audio_cpp_selection() -> (
    None
):
    from voiceover_pipeline.providers.qwen_local import QwenLocalTTSProvider

    runtime = _Runtime(_response())
    default_provider = QwenLocalTTSProvider()
    audio_cpp_provider = QwenLocalTTSProvider(
        runtime_choice="audio-cpp", audio_cpp_runtime=runtime, voice="Aiden"
    )

    result = audio_cpp_provider.synthesize_chunk("text", "chunk")

    assert default_provider._runtime_choice == "python"
    assert default_provider._audio_cpp_provider is None
    assert len(runtime.requests) == 1
    assert result.raw_metadata["runtime"]["driver_id"] == "audio-cpp"


def test_qwen_voice_design_is_accepted_without_changing_the_default_cli_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voiceover_pipeline.cli import build_parser, build_provider
    from voiceover_pipeline.providers.qwen_local import QwenLocalTTSProvider

    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)
    parser = build_parser()
    default_args = parser.parse_args("generate --provider qwen-local".split())
    design_args = parser.parse_args(
        "generate --provider qwen-local --mode design --qwen-instruct warm_style".split()
    )

    default_provider = build_provider(
        default_args, api_key="", style_prompt=None, prompt_mode="none"
    )
    design_provider = build_provider(design_args, api_key="", style_prompt=None, prompt_mode="none")

    assert isinstance(default_provider, QwenLocalTTSProvider)
    assert default_provider._runtime_choice == "python"
    assert design_provider._mode == "design"
    assert design_provider._voice is None

    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "python")
    rollback_provider = build_provider(
        default_args, api_key="", style_prompt=None, prompt_mode="none"
    )
    assert isinstance(rollback_provider, QwenLocalTTSProvider)
