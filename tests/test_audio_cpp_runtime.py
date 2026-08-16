from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from voiceover_pipeline.local_runtime.contracts import LocalRuntimeRequest, RuntimeProtocolError, RuntimeTransportError
from voiceover_pipeline.local_runtime.drivers.audio_cpp import AudioCppRuntimeDriver
from voiceover_pipeline.local_runtime.transports.subprocess import SubprocessJSONTransport


PINNED_AUDIO_CPP_REVISION = "502b5b74bd26e9b4aed267d1776ecf131cae7215"


def _request() -> LocalRuntimeRequest:
    return LocalRuntimeRequest(
        request_id="request-1",
        operation="asr",
        family="qwen3-asr",
        provider_id="qwen-local",
        payload={"model_id": "Qwen/Qwen3-ASR-0.6B"},
    )


def _driver(script: Path) -> AudioCppRuntimeDriver:
    return AudioCppRuntimeDriver(
        binary_path=Path(sys.executable),
        source_revision=PINNED_AUDIO_CPP_REVISION,
        transport=SubprocessJSONTransport((sys.executable, str(script)), timeout_seconds=2),
    )


def test_audio_cpp_driver_accepts_only_versioned_json_envelope(tmp_path: Path):
    script = tmp_path / "fixture_driver.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "json.dump({'schema_version': 1, 'request_id': request['request_id'], 'ok': True, 'response': {'text': 'fixture'}}, sys.stdout)\n"
    )

    response = _driver(script).invoke(_request())

    assert response.request_id == "request-1"
    assert response.payload == {"text": "fixture"}
    assert response.receipt.source_revision == PINNED_AUDIO_CPP_REVISION
    assert response.receipt.transport == "subprocess-json"


def test_audio_cpp_driver_rejects_malformed_json_without_parsing_human_cli_output(tmp_path: Path):
    script = tmp_path / "bad_driver.py"
    script.write_text("print('human-readable status only')\n")

    with pytest.raises(RuntimeProtocolError, match="valid JSON"):
        _driver(script).invoke(_request())


def test_audio_cpp_driver_surfaces_nonzero_exit_without_stderr_contents(tmp_path: Path):
    script = tmp_path / "failed_driver.py"
    script.write_text("import sys\nsys.stderr.write('/private/audio.wav secret diagnostic')\nsys.exit(7)\n")

    with pytest.raises(RuntimeTransportError, match="with code 7") as error:
        _driver(script).invoke(_request())

    assert "private" not in str(error.value)
    assert "secret" not in str(error.value)


def test_audio_cpp_driver_health_is_deterministic_when_binary_is_missing(tmp_path: Path):
    driver = AudioCppRuntimeDriver(
        binary_path=tmp_path / "missing-audio-cpp",
        source_revision=PINNED_AUDIO_CPP_REVISION,
    )

    health = driver.health()

    assert health.available is False
    assert health.remediation == "The pinned audio.cpp binary is not installed."


def test_subprocess_transport_times_out_without_exposing_private_workspace(tmp_path: Path):
    script = tmp_path / "slow_driver.py"
    script.write_text("import time\ntime.sleep(10)\n")
    transport = SubprocessJSONTransport((sys.executable, str(script)), timeout_seconds=0.01)

    with pytest.raises(RuntimeTransportError, match="timed out") as error:
        transport.invoke("timeout-request", {"schema_version": 1})

    assert str(tmp_path) not in str(error.value)


def test_subprocess_transport_cancels_the_process_group_for_an_active_request(tmp_path: Path):
    started = tmp_path / "started"
    script = tmp_path / "cancellable_driver.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text('started')\n"
        "time.sleep(10)\n"
    )
    transport = SubprocessJSONTransport((sys.executable, str(script), str(started)), timeout_seconds=20)
    errors: list[Exception] = []

    def invoke() -> None:
        try:
            transport.invoke("cancel-request", {"schema_version": 1})
        except Exception as exc:  # Fixture must retain the exact public transport error.
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    deadline = time.monotonic() + 2
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()

    transport.cancel("cancel-request")
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeTransportError)
    assert str(errors[0]) == "audio.cpp invocation cancelled"
