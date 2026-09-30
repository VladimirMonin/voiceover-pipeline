"""End-to-end contract tests for the native local Qwen clone route.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp
reference WAV sample, a fake in-memory Qwen provider, a stubbed offline
availability probe, and patched FFmpeg/concat seams. No real provider, network
call, API key, ``.env``, model, or download is used. The tests assert the committed
clone identity (sample locator/digest/size, reference text, model/mode/runtime,
no sample bytes), the cost-free local attempt per real invocation, local raw
recovery without a second model run, fail-closed reference identity before the
model, and the ``history resume``/``sync`` reconstruction rather than only
statuses.
"""

import hashlib
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
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers import qwen_local
from voiceover_pipeline.providers.qwen_local import QwenLocalTTSProvider
from voiceover_pipeline.services import native_generation


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
        self._state["calls"].append((chunk_id, text))
        if self._state.get("fail_synthesis_on") == chunk_id:
            raise RuntimeError("local clone boom")
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="qwen-local",
        )


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path, state: dict[str, Any]) -> None:
    failing = state.get("fail_conversion_on")
    if failing and data == f"{failing}-audio".encode():
        raise RuntimeError("ffmpeg boom")
    path.write_bytes(data)


def _concat_audio(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    # Force the default python runtime so the committed identity is deterministic
    # regardless of the developer's environment.
    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)
    # The probe is a real dependency check; the tests stub it so no qwen runtime,
    # torch, or cached model is required and no download is attempted.
    monkeypatch.setattr(
        qwen_local,
        "qwen_local_tts_availability",
        lambda *_args, **_kwargs: qwen_local.QwenLocalTTSAvailability(available=True),
    )
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)
    return home


def _script(tmp_path: Path, sections: list[str]) -> Path:
    path = tmp_path / "clone.md"
    path.write_text("\n******\n".join(sections), encoding="utf-8")
    return path


def _sample(tmp_path: Path, seed: int) -> Path:
    path = tmp_path / f"sample_{seed}.wav"
    path.write_bytes(_mono_wav(seed))
    return path


def _generate_argv(
    tmp_path: Path, script: Path, sample: Path, run_id: str, *, extra=(), sample_text="ref text"
):
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
        sample_text,
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


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


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


def _attempt_statuses(home: Path, run_uuid: str) -> list[str | None]:
    return [
        row["status"]
        for row in _rows(
            home,
            "SELECT status FROM attempts WHERE run_uuid = ? ORDER BY rowid",
            (run_uuid,),
        )
    ]


def _snapshot_config(home: Path, run_uuid: str) -> dict:
    row = _rows(home, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    return json.loads(row["config_snapshot"])


# ── fresh local clone ─────────────────────────────────────────────────────────


def test_native_qwen_clone_generates_and_commits_full_clone_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 1)
    sample_bytes = sample.read_bytes()
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-fresh")
    )

    assert code == 0, payload
    assert state["calls"] == [("chunk_01", "Первая часть."), ("chunk_02", "Вторая часть.")]
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["provider"] == "qwen-local"
    assert chunks["model"] == QWEN_MODEL_BASE
    assert chunks["voice"] == "clone"
    assert chunks["script_format"] == "markdown"
    assert [chunk["id"] for chunk in chunks["chunks"]] == ["chunk_01", "chunk_02"]
    assert chunks.get("cost_total") is None
    assert chunks.get("cost_currency") is None
    # The public exports never carry the reference sample or its bytes.
    assert str(sample) not in json.dumps(chunks, ensure_ascii=False)
    assert "ref text" not in json.dumps(chunks, ensure_ascii=False)

    run_uuid = _run_uuid("clone-fresh")
    config = _snapshot_config(local_env, run_uuid)
    clone = config["qwen_clone"]
    assert clone["mode"] == "clone"
    assert clone["model"] == QWEN_MODEL_BASE
    assert clone["sample_path"] == str(sample.resolve())
    assert clone["sample_sha256"] == hashlib.sha256(sample_bytes).hexdigest()
    assert clone["sample_size"] == len(sample_bytes)
    assert clone["sample_text"] == "ref text"
    assert clone["runtime"] == "python"
    assert clone["language"] == "Russian"
    # No sample bytes are copied into the snapshot.
    assert "audio/wav" not in json.dumps(config, ensure_ascii=False)

    # One durable, cost-free local attempt per part; exact NULL cost, never a zero.
    attempts = _rows(
        local_env,
        "SELECT call_type, provider, model, status, cost, cost_currency "
        "FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert len(attempts) == 2
    assert {row["call_type"] for row in attempts} == {"local_tts_chunk"}
    assert {row["provider"] for row in attempts} == {"qwen-local"}
    assert {row["model"] for row in attempts} == {QWEN_MODEL_BASE}
    assert {row["status"] for row in attempts} == {"local_completed"}
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)
    run_root = tmp_path / "out" / "clone-fresh"
    assert (run_root / "raw" / "chunk_01.wav").read_bytes() == b"chunk_01-audio"
    assert not (run_root / "raw" / "chunk_01.wav.receipt.json").exists()


def test_native_qwen_clone_counts_local_attempts_and_costs_without_api_charge(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 3)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-costs")
    )
    assert code == 0, payload

    code, costs = _json_run(
        monkeypatch, capsys, ["voiceover-pipeline", "history", "costs", "--json"]
    )
    assert code == 0, costs
    assert costs["local_attempts_without_api_charge"] == 2
    assert costs["attempts"] == 2
    assert costs["completeness"] == "complete"
    assert costs["totals"] == []


# ── local retry and raw recovery ──────────────────────────────────────────────


def test_native_qwen_clone_resume_retries_only_the_unfinished_part(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 4)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-retry")
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_SYNTHESIS_FAILED"
    assert [call[0] for call in state["calls"]] == ["chunk_01", "chunk_02"]
    run_uuid = _run_uuid("clone-retry")
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", "local_failed"]

    state["fail_synthesis_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "clone-retry", extra=("--resume",)),
    )
    assert code == 0, payload
    assert [call[0] for call in state["calls"]] == ["chunk_01", "chunk_02", "chunk_02"]
    assert _attempt_statuses(local_env, run_uuid) == [
        "local_completed",
        "local_failed",
        "local_completed",
    ]


def test_native_qwen_clone_rebuilds_from_linked_raw_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_conversion_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 5)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    run_root = tmp_path / "out" / "clone-raw"

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-raw")
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert (run_root / "raw" / "chunk_02.wav").read_bytes() == b"chunk_02-audio"
    assert not (run_root / "chunks" / "chunk_02.mp3").exists()

    state["fail_conversion_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "clone-raw", extra=("--resume",)),
    )
    assert code == 0, payload
    assert [call[0] for call in state["calls"]] == ["chunk_01", "chunk_02"]
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"


def test_native_qwen_clone_interrupted_invocation_then_resume_is_truthful(
    tmp_path, monkeypatch, capsys, local_env
):
    from voiceover_pipeline.history.repository import HistoryRepository

    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 6)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    real_link = HistoryRepository.record_local_tts_raw_saved
    links = {"n": 0}

    def interrupting_link(self, *args, **kwargs):
        links["n"] += 1
        if links["n"] == 2:
            raise KeyboardInterrupt
        return real_link(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", interrupting_link)
    monkeypatch.setattr(sys, "argv", _generate_argv(tmp_path, script, sample, "clone-interrupt"))
    with pytest.raises(KeyboardInterrupt):
        cli.main()

    run_uuid = _run_uuid("clone-interrupt")
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", None]

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", real_link)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "clone-interrupt", extra=("--resume",)),
    )
    assert code == 0, payload
    assert [call[0] for call in state["calls"]] == ["chunk_01", "chunk_02", "chunk_02"]
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", None, "local_completed"]


# ── reference identity fail-closed ────────────────────────────────────────────


def test_native_qwen_clone_changed_sample_fails_resume_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 7)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-changed")
    )
    assert code == 30

    # The reference bytes no longer match the committed digest, so the rebuilt
    # candidate identity differs and the resume is refused before any provider.
    sample.write_bytes(_mono_wav(77))
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "clone-changed", extra=("--resume",)),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"


def test_native_qwen_clone_missing_sample_fails_before_provider(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 8)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-missing")
    )
    assert code == 30

    sample.unlink()
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, sample, "clone-missing", extra=("--resume",)),
    )
    # A missing CLI input is a usage error, refused before any provider is built.
    assert code == 2, payload
    assert "sample" in payload["error"]


def test_native_qwen_clone_changed_sample_blocks_history_resume_before_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 9)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-hist-ref")
    )
    assert code == 30
    run_uuid = _run_uuid("clone-hist-ref")
    before = list(state["calls"])

    sample.write_bytes(_mono_wav(99))
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
    assert state["calls"] == before


# ── history resume / sync and legacy roots ────────────────────────────────────


def test_native_qwen_clone_history_resume_after_script_deleted(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 10)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-hist")
    )
    assert code == 0
    run_uuid = _run_uuid("clone-hist")
    before = list(state["calls"])
    script.unlink()
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert Path(payload["files"]["full_mp3"]).is_file()
    assert state["calls"] == before


def test_native_qwen_clone_sync_uncommitted_part_reports_local_error(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 11)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-sync-pending")
    )
    assert code == 30
    run_uuid = _run_uuid("clone-sync-pending")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
    assert state["calls"] == before


def test_native_qwen_clone_history_sync_is_export_only_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    sample = _sample(tmp_path, 12)
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, sample, "clone-sync")
    )
    assert code == 0
    run_uuid = _run_uuid("clone-sync")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert state["calls"] == before


def test_native_qwen_clone_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "legacy-clone"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-clone"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


# ── route gate ────────────────────────────────────────────────────────────────


def test_native_qwen_local_route_gate_rejects_mismatched_model_per_mode(monkeypatch):
    import argparse

    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)

    def _args(**overrides):
        base = dict(
            provider="qwen-local",
            model=QWEN_MODEL_BASE,
            mode="clone",
            no_trim=False,
            with_timings=False,
            tts_quality_provider=None,
            sample="/tmp/sample.wav",
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    assert cli._native_route_eligible(_args(), "markdown") is True
    # The instructed modes are admitted with their own model; a clone model with a
    # preset/design mode (or the unmodeled ``auto``) keeps the legacy executor.
    assert cli._native_route_eligible(_args(mode="preset"), "markdown") is False
    assert cli._native_route_eligible(_args(mode="design"), "markdown") is False
    assert cli._native_route_eligible(_args(mode="auto"), "markdown") is False
    # A missing sample keeps it legacy; the recorded trim, integrated timings
    # (local or paid cloud), and an installed local quality provider are recorded
    # on the route, while a cloud quality provider keeps it legacy.
    assert cli._native_route_eligible(_args(sample=None), "markdown") is False
    assert cli._native_route_eligible(_args(tts_quality_provider="qwen-local"), "markdown") is True
    assert cli._native_route_eligible(_args(with_timings=True), "markdown") is True
    assert cli._native_route_eligible(_args(no_trim=True), "markdown") is True
    assert (
        cli._native_route_eligible(
            _args(with_timings=True, tts_quality_provider="nemotron-local"), "markdown"
        )
        is True
    )
    assert cli._native_route_eligible(_args(tts_quality_provider="xai-stt"), "markdown") is False
    assert (
        cli._native_route_eligible(
            _args(with_timings=True, timing_provider="groq-whisper"), "markdown"
        )
        is True
    )
    assert (
        cli._native_route_eligible(_args(with_timings=True, timing_provider="xai-stt"), "markdown")
        is True
    )
    # A non-markdown format is not the admitted route.
    assert cli._native_route_eligible(_args(), "dialogue") is False
    # An unrecognized runtime keeps it on the legacy executor too.
    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "bogus")
    assert cli._native_route_eligible(_args(), "markdown") is False
