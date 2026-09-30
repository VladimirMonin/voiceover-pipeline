"""End-to-end contract tests for effective local output processing (S05 batch 2).

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp
reference WAV sample, a fake in-memory Qwen provider, fake local timing and local
ASR quality providers, a stubbed offline availability probe, and patched
FFmpeg/concat/trim seams. No real provider, network call, API key, ``.env``, model,
or download is used. The tests assert that the recorded trim, an integrated local
timing step, and an installed local quality step all run on one native
``qwen-local`` run, that a resume never repeats a completed local TTS or a
persisted verdict, that a changed output option blocks before the model, and that
``history sync`` runs no model -- the shared boundaries rather than a full flag
cartesian product.
"""

import io
import json
import sqlite3
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import QWEN_MODEL_BASE
from voiceover_pipeline.models import (
    ASRExecutionReceipt,
    ASRResult,
    SynthesisResult,
    TimingResult,
    TimingSegment,
)
from voiceover_pipeline.providers import qwen_local
from voiceover_pipeline.providers.qwen_local import QwenLocalTTSProvider
from voiceover_pipeline.services import native_generation, transcription

QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
SCRIPT_TEXT = "Первая часть. Вторая часть."
TIMING_TRANSCRIPT = "Текст распознавания таймингов."


def _mono_wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes([seed]) * 160)
    return buffer.getvalue()


class FakeQwenClone(QwenLocalTTSProvider):
    """Offline stand-in that records every local clone invocation."""

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__(mode="clone", voice="clone", sample_path="unused.wav", sample_text="ref")
        self._state = state

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self._state["calls"].append(chunk_id)
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="qwen-local",
        )


def _asr_result(transcript: str) -> ASRResult:
    return ASRResult(
        transcript=transcript,
        provider_id="qwen-local",
        model_id=QWEN_ASR_MODEL,
        execution=ASRExecutionReceipt(
            runtime="python",
            model_revision="rev-1",
            resolved_device="cpu",
            resolved_compute="auto",
        ),
        language="ru",
    )


def _timing_result() -> TimingResult:
    return TimingResult(
        segments=[
            TimingSegment(
                id=0,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text=TIMING_TRANSCRIPT,
            )
        ],
        model="small",
        backend="faster-whisper",
        provider="faster-whisper",
        device="cpu",
        compute_type="int8",
        language="ru",
    )


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat_audio(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


@pytest.fixture
def trim_calls() -> list[str]:
    return []


@pytest.fixture
def local_env(tmp_path, monkeypatch, trim_calls):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)
    monkeypatch.setattr(
        qwen_local,
        "qwen_local_tts_availability",
        lambda *_args, **_kwargs: qwen_local.QwenLocalTTSAvailability(available=True),
    )
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)

    def recording_trim(_ffmpeg: str, _ffprobe: str, path: Path) -> None:
        trim_calls.append(path.name)
        path.write_bytes(path.read_bytes() + b"-trimmed")

    monkeypatch.setattr(cli, "trim_final_silence", recording_trim)
    # The preflights and the model runs are replaced per test; no-op preflights are
    # the default so only the explicit preflight tests exercise the probes (this
    # environment has no local model install).
    monkeypatch.setattr(native_generation, "_preflight_local_timing", lambda _options: None)
    monkeypatch.setattr(native_generation, "_preflight_local_quality", lambda _options: None)
    return home


def _script(tmp_path: Path, parts: list[str]) -> Path:
    path = tmp_path / "script.md"
    path.write_text("\n******\n".join(parts), encoding="utf-8")
    return path


def _sample(tmp_path: Path, seed: int) -> Path:
    path = tmp_path / f"sample_{seed}.wav"
    path.write_bytes(_mono_wav(seed))
    return path


def _generate_argv(tmp_path: Path, script: Path, sample: Path, run_id: str, *, extra=()):
    return [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "qwen-local",
        "--model",
        QWEN_MODEL_BASE,
        "--mode",
        "clone",
        "--sample",
        str(sample),
        "--sample-text",
        "ref text",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    return code, json.loads(capsys.readouterr().out)


def _install_provider(monkeypatch, state: dict[str, Any]):
    monkeypatch.setattr(cli, "build_provider", lambda *_a, **_k: FakeQwenClone(state))


def _install_timing(monkeypatch, error: Exception | None = None):
    calls: list[dict] = []

    def fake_transcribe_timing(**_kwargs):
        calls.append(dict(_kwargs))
        if error is not None:
            raise error
        return _timing_result()

    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake_transcribe_timing)
    return calls


def _install_quality(monkeypatch, transcript: str, error: Exception | None = None):
    calls: list[dict] = []

    def fake_transcribe(**_kwargs):
        calls.append(dict(_kwargs))
        if error is not None:
            raise error
        return _asr_result(transcript)

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", fake_transcribe)
    return calls


def _history_argv(run_uuid: str, verb: str) -> list[str]:
    return ["voiceover-pipeline", "history", verb, run_uuid, "--json"]


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    runs = list_history()["runs"]
    matches = [run for run in runs if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _rows(home: Path, sql: str, params=()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _children(home: Path, run_uuid: str, operation: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT * FROM runs WHERE parent_uuid = ? AND operation = ?",
        (run_uuid, operation),
    )


def _output_options(home: Path, run_uuid: str) -> dict:
    row = _rows(home, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    return json.loads(row["config_snapshot"])["output"]


def _local_attempts(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT call_type, status, cost, cost_currency FROM attempts "
        "WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )


# ── one native local run with all three requested steps ───────────────────────


def test_native_qwen_local_combined_trim_timing_and_quality_are_linked(
    tmp_path, monkeypatch, capsys, local_env, trim_calls
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    timing_calls = _install_timing(monkeypatch)
    # The observed verification transcript normalizes equal to the script text but
    # is stored with its own casing, so a leak into an export is observable.
    observed_transcript = "первая часть. вторая часть."
    quality_calls = _install_quality(monkeypatch, observed_transcript)
    sample = _sample(tmp_path, 1)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            sample,
            "qwen-combined",
            extra=(
                "--no-trim",
                "--with-timings",
                "--timing-provider",
                "faster-whisper",
                "--tts-quality-provider",
                "qwen-local",
            ),
        ),
    )

    assert code == 0, payload
    # One local TTS invocation per part; no part is skipped or repeated.
    assert state["calls"] == ["chunk_01", "chunk_02"]
    # ``--no-trim`` skipped the trim seam entirely; the default would have trimmed.
    assert trim_calls == []
    # Both local post-audio steps ran exactly once, after the committed audio.
    assert len(timing_calls) == 1
    assert len(quality_calls) == 1
    assert payload["timing"] == {"complete": True}
    assert payload["quality"] == {"complete": True, "passed": True}
    assert payload["files"]["timings_json"].endswith("qwen-combined.timings.json")
    assert payload["files"]["srt"].endswith("qwen-combined.srt")

    run_uuid = _run_uuid("qwen-combined")
    # The private verification transcript is never projected into the machine
    # output or the compatibility JSON exports.
    exports = "".join(
        Path(payload["files"][key]).read_text(encoding="utf-8")
        for key in ("chunks_json", "run_json", "manifest_json")
    )
    assert observed_transcript not in json.dumps(payload, ensure_ascii=False)
    assert observed_transcript not in exports
    options = _output_options(local_env, run_uuid)
    assert options["trim_final_silence"] is False
    assert options["timing_enabled"] is True
    assert options["quality_enabled"] is True
    assert len(_children(local_env, run_uuid, "timings")) == 1
    assert len(_children(local_env, run_uuid, "verify")) == 1
    attempts = _local_attempts(local_env, run_uuid)
    assert [row["call_type"] for row in attempts] == ["local_tts_chunk", "local_tts_chunk"]
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)


def test_native_qwen_local_resume_repeats_no_local_tts_timing_or_quality(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    timing_calls = _install_timing(monkeypatch)
    quality_calls = _install_quality(monkeypatch, SCRIPT_TEXT)
    sample = _sample(tmp_path, 2)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    extra = (
        "--with-timings",
        "--timing-provider",
        "faster-whisper",
        "--tts-quality-provider",
        "qwen-local",
    )

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "qwen-resume", extra=extra)
    )
    assert code == 0, payload
    assert state["calls"] == ["chunk_01", "chunk_02"]
    assert len(timing_calls) == 1 and len(quality_calls) == 1

    # A completed run resumed with the same settings performs no local work at all.
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "qwen-resume", extra=(*extra, "--resume")),
    )
    assert code == 0, payload
    assert state["calls"] == ["chunk_01", "chunk_02"]
    assert len(timing_calls) == 1
    assert len(quality_calls) == 1
    assert payload["timing"] == {"complete": True}
    assert payload["quality"] == {"complete": True, "passed": True}


def test_native_qwen_local_quality_transcription_failure_then_resume_repairs(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    timing_calls = _install_timing(monkeypatch)
    _install_quality(monkeypatch, SCRIPT_TEXT, error=RuntimeError("asr down"))
    sample = _sample(tmp_path, 3)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    extra = (
        "--with-timings",
        "--timing-provider",
        "faster-whisper",
        "--tts-quality-provider",
        "qwen-local",
    )

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "qwen-qa-retry", extra=extra)
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_ASR_FAILED"
    # The completed TTS audio and its linked timing survive the failed check.
    run_uuid = _run_uuid("qwen-qa-retry")
    assert len(_children(local_env, run_uuid, "timings")) == 1
    assert _children(local_env, run_uuid, "verify") == []
    assert list((tmp_path / "out" / "qwen-qa-retry").glob("*voiceover*.mp3"))

    # An explicit resume runs only the pending local check; TTS and timing are kept.
    _install_quality(monkeypatch, SCRIPT_TEXT)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "qwen-qa-retry", extra=(*extra, "--resume")),
    )
    assert code == 0, payload
    assert state["calls"] == ["chunk_01", "chunk_02"]
    assert len(timing_calls) == 1
    assert payload["quality"] == {"complete": True, "passed": True}
    assert len(_children(local_env, run_uuid, "verify")) == 1


def test_native_qwen_local_persisted_quality_fail_is_durable_and_sync_reports_it(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    quality_calls = _install_quality(monkeypatch, "совсем другой текст")
    sample = _sample(tmp_path, 4)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    extra = ("--tts-quality-provider", "qwen-local")

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "qwen-fail", extra=extra)
    )
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    run_uuid = _run_uuid("qwen-fail")
    verdicts = _children(local_env, run_uuid, "verify")
    assert len(verdicts) == 1
    assert json.loads(verdicts[0]["config_snapshot"])["quality_passed"] is False
    assert len(quality_calls) == 1

    # A resume re-reports the durable verdict without running the local ASR again.
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "qwen-fail", extra=(*extra, "--resume")),
    )
    assert code == 60, payload
    assert len(quality_calls) == 1
    assert state["calls"] == ["chunk_01", "chunk_02"]

    # ``history sync`` is a state reader: it reports the recorded failure and runs
    # no local model.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": False}
    assert len(quality_calls) == 1
    assert state["calls"] == ["chunk_01", "chunk_02"]


def test_native_qwen_local_changed_output_options_block_before_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    timing_calls = _install_timing(monkeypatch)
    sample = _sample(tmp_path, 5)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    timing_extra = ("--with-timings", "--timing-provider", "faster-whisper")

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "qwen-opts", extra=timing_extra),
    )
    assert code == 0, payload
    assert len(timing_calls) == 1

    # Resuming with a different recorded output option fails closed before the model.
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, sample, "qwen-opts", extra=(*timing_extra, "--no-trim", "--resume")
        ),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_PROCESSING_UNSUPPORTED"
    assert state["calls"] == ["chunk_01", "chunk_02"]
    assert len(timing_calls) == 1


def test_native_qwen_local_sync_runs_no_model_with_pending_timing(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    timing_calls = _install_timing(monkeypatch, error=RuntimeError("timing down"))
    sample = _sample(tmp_path, 6)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    extra = ("--with-timings", "--timing-provider", "faster-whisper")

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "qwen-timing", extra=extra)
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_TIMING_FAILED"
    run_uuid = _run_uuid("qwen-timing")
    assert _children(local_env, run_uuid, "timings") == []
    assert len(timing_calls) == 1

    # ``history sync`` never runs a local timing model and reports the pending step.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["timing"] == {"complete": False}
    assert len(timing_calls) == 1
    assert state["calls"] == ["chunk_01", "chunk_02"]
