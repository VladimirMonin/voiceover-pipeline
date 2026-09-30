"""End-to-end contract tests for the native local Qwen preset/design routes.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a fake
in-memory Qwen provider, a stubbed offline availability probe, and patched
FFmpeg/concat seams. No real provider, network call, API key, ``.env``, model, or
download is used. The tests assert the committed instructed-mode identity
(mode/model/voice, the exact instruction, the runtime, and the language), a
cost-free local attempt per real invocation, local raw recovery without a second
model run, fail-closed identity before the model, the ``history resume``/``sync``
reconstruction rather than only statuses, and that the private identity knobs
never reach the DB-derived public JSON.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import (
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers import qwen_local
from voiceover_pipeline.providers.qwen_local import QwenLocalTTSProvider
from voiceover_pipeline.services import native_generation


class FakeQwenMode(QwenLocalTTSProvider):
    """Offline stand-in that records every local preset/design invocation."""

    def __init__(self, state: dict[str, Any], *, mode: str, voice: str) -> None:
        super().__init__(mode=mode, voice=voice)
        self._state = state

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self._state["calls"].append({"id": chunk_id, "text": text, "voice": self._voice})
        if self._state.get("fail_synthesis_on") == chunk_id:
            raise RuntimeError("local mode boom")
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
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)
    return home


def _script(tmp_path: Path, sections: list[str], name: str = "mode.md") -> Path:
    path = tmp_path / name
    path.write_text("\n******\n".join(sections), encoding="utf-8")
    return path


def _generate_argv(
    tmp_path: Path,
    script: Path,
    run_id: str,
    *,
    mode: str,
    model: str,
    voice: str | None = None,
    instruct: str | None = None,
    extra=(),
):
    argv = [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "qwen-local",
        "--model",
        model,
        "--mode",
        mode,
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
    ]
    if voice is not None:
        argv += ["--voice", voice]
    if instruct is not None:
        argv += ["--qwen-instruct", instruct]
    argv += list(extra)
    argv += ["--json"]
    return argv


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    return code, json.loads(capsys.readouterr().out)


def _install_provider(monkeypatch, state: dict[str, Any], *, mode: str, voice: str):
    monkeypatch.setattr(
        cli, "build_provider", lambda *_a, **_k: FakeQwenMode(state, mode=mode, voice=voice)
    )


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


# ── fresh preset / design identity ────────────────────────────────────────────


def test_native_qwen_preset_commits_full_mode_identity(tmp_path, monkeypatch, capsys, local_env):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="preset", voice="Serena")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    instruct = "Спокойная, уверенная подача."

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "preset-fresh",
            mode="preset",
            model=QWEN_MODEL_CUSTOMVOICE,
            voice="Serena",
            instruct=instruct,
        ),
    )

    assert code == 0, payload
    assert [call["id"] for call in state["calls"]] == ["chunk_01", "chunk_02"]
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["provider"] == "qwen-local"
    assert chunks["model"] == QWEN_MODEL_CUSTOMVOICE
    assert chunks["voice"] == "Serena"
    assert chunks.get("cost_total") is None
    assert chunks.get("cost_currency") is None
    # The private identity knobs (runtime/language and the raw block) never reach the
    # DB-derived public JSON; the instruction is the existing public ``style_prompt``.
    serialized = json.dumps(chunks, ensure_ascii=False)
    assert "qwen_mode" not in serialized
    assert '"runtime"' not in serialized
    assert '"language"' not in serialized

    run_uuid = _run_uuid("preset-fresh")
    config = _snapshot_config(local_env, run_uuid)
    assert config["qwen_mode"] == {
        "mode": "preset",
        "model": QWEN_MODEL_CUSTOMVOICE,
        "voice": "Serena",
        "instruct": instruct,
        "runtime": "python",
        "language": "Russian",
    }
    assert "qwen_clone" not in config

    attempts = _rows(
        local_env,
        "SELECT call_type, provider, model, status, cost, cost_currency "
        "FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert len(attempts) == 2
    assert {row["call_type"] for row in attempts} == {"local_tts_chunk"}
    assert {row["model"] for row in attempts} == {QWEN_MODEL_CUSTOMVOICE}
    assert {row["status"] for row in attempts} == {"local_completed"}
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)
    run_root = tmp_path / "out" / "preset-fresh"
    assert (run_root / "raw" / "chunk_01.wav").read_bytes() == b"chunk_01-audio"


def test_native_qwen_design_commits_full_mode_identity(tmp_path, monkeypatch, capsys, local_env):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="design", voice="design")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Один блок."], name="design.md")
    instruct = "Молодой женский голос с лёгкой улыбкой."

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "design-fresh",
            mode="design",
            model=QWEN_MODEL_VOICE_DESIGN,
            instruct=instruct,
        ),
    )

    assert code == 0, payload
    assert [call["id"] for call in state["calls"]] == ["chunk_01"]
    run_uuid = _run_uuid("design-fresh")
    config = _snapshot_config(local_env, run_uuid)
    assert config["voice"] == "design"
    assert config["qwen_mode"] == {
        "mode": "design",
        "model": QWEN_MODEL_VOICE_DESIGN,
        "voice": "design",
        "instruct": instruct,
        "runtime": "python",
        "language": "Russian",
    }


def test_native_qwen_mode_counts_local_attempts_without_api_charge(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mode-costs",
            mode="preset",
            model=QWEN_MODEL_CUSTOMVOICE,
            voice="Aiden",
        ),
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


def test_native_qwen_mode_resume_retries_only_the_unfinished_part(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    argv = _generate_argv(
        tmp_path, script, "mode-retry", mode="preset", model=QWEN_MODEL_CUSTOMVOICE, voice="Aiden"
    )

    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_SYNTHESIS_FAILED"
    assert [call["id"] for call in state["calls"]] == ["chunk_01", "chunk_02"]
    run_uuid = _run_uuid("mode-retry")
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", "local_failed"]

    state["fail_synthesis_on"] = None
    code, payload = _json_run(monkeypatch, capsys, argv + ["--resume"])
    assert code == 0, payload
    assert [call["id"] for call in state["calls"]] == ["chunk_01", "chunk_02", "chunk_02"]
    assert _attempt_statuses(local_env, run_uuid) == [
        "local_completed",
        "local_failed",
        "local_completed",
    ]


def test_native_qwen_mode_rebuilds_from_linked_raw_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_conversion_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    run_root = tmp_path / "out" / "mode-raw"
    argv = _generate_argv(
        tmp_path, script, "mode-raw", mode="preset", model=QWEN_MODEL_CUSTOMVOICE, voice="Aiden"
    )

    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert (run_root / "raw" / "chunk_02.wav").read_bytes() == b"chunk_02-audio"
    assert not (run_root / "chunks" / "chunk_02.mp3").exists()

    state["fail_conversion_on"] = None
    code, payload = _json_run(monkeypatch, capsys, argv + ["--resume"])
    assert code == 0, payload
    assert [call["id"] for call in state["calls"]] == ["chunk_01", "chunk_02"]
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"


# ── identity fail-closed before provider ──────────────────────────────────────


def test_native_qwen_mode_changed_design_prompt_blocks_resume_before_provider(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="design", voice="design")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    argv = _generate_argv(
        tmp_path,
        script,
        "design-prompt",
        mode="design",
        model=QWEN_MODEL_VOICE_DESIGN,
        instruct="Спокойный ровный голос.",
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30

    changed = list(argv)
    changed[changed.index("--qwen-instruct") + 1] = "Другой, изменённый голос."
    changed.append("--resume")
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, changed)
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"


def test_native_qwen_mode_changed_voice_blocks_resume_before_provider(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Serena")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    argv = _generate_argv(
        tmp_path,
        script,
        "preset-voice",
        mode="preset",
        model=QWEN_MODEL_CUSTOMVOICE,
        voice="Serena",
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30

    argv[-1:-1] = ["--resume"]
    argv[argv.index("--voice") + 1] = "Ryan"
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"


def test_native_qwen_mode_changed_mode_blocks_resume_before_provider(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Serena")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    argv = _generate_argv(
        tmp_path, script, "preset-mode", mode="preset", model=QWEN_MODEL_CUSTOMVOICE, voice="Serena"
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30

    argv[argv.index("--mode") + 1] = "design"
    argv[argv.index("--model") + 1] = QWEN_MODEL_VOICE_DESIGN
    argv[-1:-1] = ["--resume", "--qwen-instruct", "Другой голос по инструкции."]
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"


def test_native_qwen_mode_changed_runtime_blocks_history_resume_before_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    argv = _generate_argv(
        tmp_path, script, "mode-runtime", mode="preset", model=QWEN_MODEL_CUSTOMVOICE, voice="Aiden"
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30
    run_uuid = _run_uuid("mode-runtime")
    before = list(state["calls"])

    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "audio-cpp")
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
    assert state["calls"] == before


def test_native_qwen_mode_unavailable_model_blocks_before_local_synthesis(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    monkeypatch.setattr(
        qwen_local,
        "qwen_local_tts_availability",
        lambda *_a, **_k: qwen_local.QwenLocalTTSAvailability(
            available=False,
            reason_code="model_not_cached",
            remediation="the local model is not cached; download it explicitly.",
        ),
    )
    script = _script(tmp_path, ["Один блок."], name="design-unavail.md")
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mode-unavail",
            mode="design",
            model=QWEN_MODEL_VOICE_DESIGN,
            instruct="Спокойный голос.",
        ),
    )
    assert code == 10, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_MODEL_UNAVAILABLE"
    assert state["calls"] == []


# ── history resume / sync and legacy roots ────────────────────────────────────


def test_native_qwen_mode_history_resume_after_script_deleted(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="preset", voice="Serena")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mode-hist",
            mode="preset",
            model=QWEN_MODEL_CUSTOMVOICE,
            voice="Serena",
            instruct="Тёплая подача.",
        ),
    )
    assert code == 0
    run_uuid = _run_uuid("mode-hist")
    before = list(state["calls"])
    script.unlink()
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert Path(payload["files"]["full_mp3"]).is_file()
    assert state["calls"] == before


def test_native_qwen_mode_sync_uncommitted_part_reports_local_error(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_02"}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mode-sync-pending",
            mode="preset",
            model=QWEN_MODEL_CUSTOMVOICE,
            voice="Aiden",
        ),
    )
    assert code == 30
    run_uuid = _run_uuid("mode-sync-pending")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
    assert state["calls"] == before


def test_native_qwen_mode_history_sync_is_export_only_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state, mode="preset", voice="Aiden")
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path, ["Первая часть.", "Вторая часть."])
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mode-sync",
            mode="preset",
            model=QWEN_MODEL_CUSTOMVOICE,
            voice="Aiden",
        ),
    )
    assert code == 0
    run_uuid = _run_uuid("mode-sync")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert state["calls"] == before


def test_native_qwen_mode_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "legacy-mode"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-mode"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


# ── route gate ────────────────────────────────────────────────────────────────


def test_native_qwen_mode_route_gate_admits_preset_and_design(monkeypatch):
    import argparse

    monkeypatch.delenv("VOICEOVER_QWEN_TTS_RUNTIME", raising=False)

    def _args(**overrides):
        base = dict(
            provider="qwen-local",
            model=QWEN_MODEL_CUSTOMVOICE,
            mode="preset",
            no_trim=False,
            with_timings=False,
            tts_quality_provider=None,
            qwen_instruct=None,
            sample=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    assert cli._native_route_eligible(_args(), "markdown") is True
    assert (
        cli._native_route_eligible(
            _args(mode="design", model=QWEN_MODEL_VOICE_DESIGN, qwen_instruct="Хриплый голос."),
            "markdown",
        )
        is True
    )
    assert cli._native_route_eligible(_args(qwen_instruct="Спокойная подача."), "markdown") is True
    # A mode/model mismatch, the unmodeled ``auto`` mode, and an empty instruction
    # keep the legacy executor; the recorded trim, integrated local timings, and an
    # installed local quality provider are recorded on the route, while a cloud
    # timing or quality provider keeps it legacy.
    assert (
        cli._native_route_eligible(_args(mode="preset", model="other/model"), "markdown") is False
    )
    assert cli._native_route_eligible(_args(mode="auto"), "markdown") is False
    assert cli._native_route_eligible(_args(qwen_instruct="  "), "markdown") is False
    assert cli._native_route_eligible(_args(qwen_instruct=""), "markdown") is False
    assert cli._native_route_eligible(_args(no_trim=True), "markdown") is True
    assert cli._native_route_eligible(_args(with_timings=True), "markdown") is True
    assert cli._native_route_eligible(_args(tts_quality_provider="qwen-local"), "markdown") is True
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
        is False
    )
    assert cli._native_route_eligible(_args(), "dialogue") is False
    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "bogus")
    assert cli._native_route_eligible(_args(), "markdown") is False
