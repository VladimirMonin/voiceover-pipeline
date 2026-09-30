"""End-to-end contract tests for the native non-preset local OmniVoice modes.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp
reference WAV, a fake in-memory OmniVoice provider installed behind the real
provider factory, and patched FFmpeg/concat seams. No real provider, network
call, API key, ``.env``, model, or download is used. The tests assert the
committed ``auto``/``clone``/``design`` mode identity, the single merged session
call per run, the cost-free local attempt with linked raw recovery, fail-closed
reference identity before the model, and the ``history resume``/``sync``
reconstruction rather than only statuses.
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
from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.services import native_generation, provider_factory

REFERENCE_TEXT = "reference transcript for the clone voice"
DESIGN_INSTRUCTION = "female"


def _mono_wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes([seed]) * 160)
    return buffer.getvalue()


class FakeOmniVoiceMode:
    """Offline stand-in installed behind the real provider factory.

    Replacing the provider class (not the factory) keeps the real CLI option
    validation and the real ``build_tts_provider`` dispatch on the path, so a
    regression that admits a mode/voice mixture the mode rejects, or that rebuilds
    a resume provider without its committed inputs, stays observable.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "FakeOmniVoiceMode":
        _FAKE_STATE["builds"].append(dict(kwargs))
        return cls(_FAKE_STATE["state"])

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self._state["calls"].append((chunk_id, text))
        if self._state.get("fail_synthesis_on") == chunk_id:
            raise RuntimeError("local synthesis boom")
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="omnivoice-local",
        )


class _ExplodingProvider:
    """Fails loudly if an export/recovery path builds a provider."""

    @classmethod
    def from_environment(cls, **_kwargs: Any) -> Any:  # pragma: no cover - asserted never to run
        raise AssertionError("this path must not build a provider")


_FAKE_STATE: dict[str, Any] = {}


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
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)
    return home


def _script(tmp_path: Path) -> Path:
    path = tmp_path / "monologue.md"
    path.write_text(
        "Первый фрагмент текста.\n\n******\n\nВторой фрагмент текста.\n", encoding="utf-8"
    )
    return path


def _reference(tmp_path: Path, seed: int = 1) -> Path:
    path = tmp_path / f"reference_{seed}.wav"
    path.write_bytes(_mono_wav(seed))
    return path


def _generate_argv(
    tmp_path: Path,
    script: Path,
    run_id: str,
    *,
    mode: str = "auto",
    extra=(),
    reference: Path | None = None,
    reference_text: str = REFERENCE_TEXT,
    design_instruction: str | None = None,
):
    argv = [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        OMNIVOICE_LOCAL_MODEL_ID,
        "--mode",
        mode,
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
    ]
    if reference is not None:
        argv += ["--reference-audio", str(reference), "--reference-text", reference_text]
    if design_instruction is not None:
        argv += ["--design-instruction", design_instruction]
    return [*argv, *extra, "--json"]


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    return code, json.loads(capsys.readouterr().out)


def _install_provider(monkeypatch, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Install the fake provider class and record every real factory build."""
    builds: list[dict[str, Any]] = []
    _FAKE_STATE["state"] = state
    _FAKE_STATE["builds"] = builds
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", FakeOmniVoiceMode)
    return builds


def _install_exploding_provider(monkeypatch) -> None:
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", _ExplodingProvider)


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


# ── fresh mode runs ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["auto", "clone", "design"])
def test_native_local_omnivoice_modes_generate_and_commit_the_mode_identity(
    tmp_path, monkeypatch, capsys, local_env, mode
):
    state: dict[str, Any] = {"calls": []}
    builds = _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path) if mode == "clone" else None
    design = DESIGN_INSTRUCTION if mode == "design" else None
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            f"omni-{mode}",
            mode=mode,
            reference=reference,
            design_instruction=design,
        ),
    )

    assert code == 0, payload
    # The whole script merges into one local session request per run.
    assert state["calls"] == [
        ("chunk_01_omnivoice_session", "Первый фрагмент текста. Второй фрагмент текста.")
    ]
    # The real factory built the provider for exactly this mode and its inputs.
    assert [build["mode"] for build in builds] == [mode]
    if mode == "clone":
        assert Path(builds[0]["reference_audio_path"]) == reference
        assert builds[0]["reference_text"] == REFERENCE_TEXT
    elif mode == "design":
        assert builds[0]["design_instruction"] == DESIGN_INSTRUCTION
    assert Path(payload["files"]["full_mp3"]).is_file()
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["provider"] == "omnivoice-local"
    assert chunks["model"] == OMNIVOICE_LOCAL_MODEL_ID
    assert chunks["script_format"] == "markdown"
    # The mode is the run's effective voice: the mode rejects any CLI voice.
    assert chunks["voice"] == mode
    assert [chunk["id"] for chunk in chunks["chunks"]] == ["chunk_01_omnivoice_session"]

    run_uuid = _run_uuid(f"omni-{mode}")
    identity = _snapshot_config(local_env, run_uuid)["omnivoice_mode"]
    assert identity["mode"] == mode
    assert identity["model"] == OMNIVOICE_LOCAL_MODEL_ID
    assert identity["voice"] == mode
    if mode == "clone":
        assert reference is not None
        assert identity["reference_audio"] == str(reference.resolve())
        assert identity["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
        assert identity["reference_size"] == reference.stat().st_size
        assert identity["reference_text"] == REFERENCE_TEXT
        assert identity["design_instruction"] is None
        # No reference bytes are copied into the snapshot.
        assert "audio/wav" not in json.dumps(identity, ensure_ascii=False)
    elif mode == "design":
        assert identity["design_instruction"] == DESIGN_INSTRUCTION
        assert identity["reference_audio"] is None
        assert identity["reference_sha256"] is None
        assert identity["reference_size"] is None
        assert identity["reference_text"] is None
    else:
        assert identity["design_instruction"] is None
        assert identity["reference_audio"] is None
        assert identity["reference_text"] is None

    attempts = _rows(
        local_env,
        "SELECT call_type, provider, model, status, cost, cost_currency "
        "FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert len(attempts) == 1
    assert attempts[0]["call_type"] == "local_tts_chunk"
    assert attempts[0]["provider"] == "omnivoice-local"
    assert attempts[0]["model"] == OMNIVOICE_LOCAL_MODEL_ID
    assert attempts[0]["status"] == "local_completed"
    # Exact NULL cost, never a fabricated zero.
    assert attempts[0]["cost"] is None and attempts[0]["cost_currency"] is None
    run_root = tmp_path / "out" / f"omni-{mode}"
    assert (run_root / "raw" / "chunk_01_omnivoice_session.wav").read_bytes() == (
        b"chunk_01_omnivoice_session-audio"
    )
    assert not (run_root / "raw" / "chunk_01_omnivoice_session.wav.receipt.json").exists()
    # A local mode route never touches the paid seam: no paid receipt, no paid marker.
    roles = {
        row["role"]
        for row in _rows(
            local_env,
            "SELECT role FROM artifacts WHERE run_uuid = ?",
            (run_uuid,),
        )
    }
    assert roles == {"local_raw_audio", "chunk_audio", "final_audio"}


def test_native_local_omnivoice_auto_counts_one_local_attempt_without_api_charge(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "omni-auto-cost", mode="auto")
    )
    assert code == 0, payload

    code, costs = _json_run(
        monkeypatch, capsys, ["voiceover-pipeline", "history", "costs", "--json"]
    )
    assert code == 0, costs
    assert costs["local_attempts_without_api_charge"] == 1
    assert costs["attempts"] == 1
    assert costs["completeness"] == "complete"
    assert costs["totals"] == []


def test_native_local_omnivoice_clone_never_publishes_the_reference_text(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-priv", mode="clone", reference=reference),
    )
    assert code == 0, payload

    run_root = tmp_path / "out" / "omni-clone-priv"
    exported = sorted(run_root.rglob("*.json"))
    assert exported, "expected committed compatibility JSON exports"
    for path in exported:
        assert REFERENCE_TEXT not in path.read_text(encoding="utf-8"), path.name

    run_uuid = _run_uuid("omni-clone-priv")
    code, detail = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "show"))
    assert code == 0, detail
    assert REFERENCE_TEXT not in json.dumps(detail, ensure_ascii=False)


def test_native_local_omnivoice_design_never_publishes_the_instruction(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-design-priv",
            mode="design",
            design_instruction=DESIGN_INSTRUCTION,
        ),
    )
    assert code == 0, payload

    run_root = tmp_path / "out" / "omni-design-priv"
    for path in sorted(run_root.rglob("*.json")):
        assert DESIGN_INSTRUCTION not in path.read_text(encoding="utf-8"), path.name


# ── local retry and raw recovery ──────────────────────────────────────────────


def test_native_local_omnivoice_clone_resume_retries_only_the_unfinished_part(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-retry", mode="clone", reference=reference),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_SYNTHESIS_FAILED"
    run_uuid = _run_uuid("omni-clone-retry")
    assert _attempt_statuses(local_env, run_uuid) == ["local_failed"]

    state["fail_synthesis_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-clone-retry",
            mode="clone",
            reference=reference,
            extra=("--resume",),
        ),
    )
    assert code == 0, payload
    assert len(state["calls"]) == 2
    assert _attempt_statuses(local_env, run_uuid) == ["local_failed", "local_completed"]


def test_native_local_omnivoice_clone_rebuilds_from_linked_raw_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_conversion_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    run_root = tmp_path / "out" / "omni-clone-raw"

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-raw", mode="clone", reference=reference),
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert (run_root / "raw" / "chunk_01_omnivoice_session.wav").read_bytes() == (
        b"chunk_01_omnivoice_session-audio"
    )
    assert not (run_root / "chunks" / "chunk_01_omnivoice_session.mp3").exists()

    state["fail_conversion_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-clone-raw",
            mode="clone",
            reference=reference,
            extra=("--resume",),
        ),
    )
    assert code == 0, payload
    assert len(state["calls"]) == 1
    assert (run_root / "chunks" / "chunk_01_omnivoice_session.mp3").read_bytes() == (
        b"chunk_01_omnivoice_session-audio"
    )


# ── reference and instruction identity fail-closed ────────────────────────────


def test_native_local_omnivoice_clone_changed_reference_blocks_resume_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-changed", mode="clone", reference=reference),
    )
    assert code == 30

    # The reference bytes no longer match the committed digest, so the rebuilt
    # candidate identity differs and the resume is refused before any provider.
    reference.write_bytes(_mono_wav(77))
    state["fail_synthesis_on"] = None
    _install_exploding_provider(monkeypatch)
    before = list(state["calls"])
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-clone-changed",
            mode="clone",
            reference=reference,
            extra=("--resume",),
        ),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert state["calls"] == before


def test_native_local_omnivoice_clone_missing_reference_fails_before_provider(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-missing", mode="clone", reference=reference),
    )
    assert code == 30

    reference.unlink()
    state["fail_synthesis_on"] = None
    _install_exploding_provider(monkeypatch)
    before = list(state["calls"])
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-clone-missing",
            mode="clone",
            reference=reference,
            extra=("--resume",),
        ),
    )
    # A missing CLI input is a usage error, refused before any provider is built.
    assert code == 2, payload
    assert "reference" in payload["error"]
    assert state["calls"] == before


def test_native_local_omnivoice_design_changed_instruction_blocks_resume_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-design-changed",
            mode="design",
            design_instruction=DESIGN_INSTRUCTION,
        ),
    )
    assert code == 30

    state["fail_synthesis_on"] = None
    _install_exploding_provider(monkeypatch)
    before = list(state["calls"])
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-design-changed",
            mode="design",
            design_instruction="male",
            extra=("--resume",),
        ),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert state["calls"] == before


# ── history resume / sync ─────────────────────────────────────────────────────


def test_native_local_omnivoice_clone_history_resume_after_script_deleted(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-hist", mode="clone", reference=reference),
    )
    assert code == 0
    run_uuid = _run_uuid("omni-clone-hist")
    before = list(state["calls"])
    script.unlink()
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert Path(payload["files"]["full_mp3"]).is_file()
    # A completed run only repairs its exports, so no provider is built at all.
    assert state["calls"] == before


def test_native_local_omnivoice_clone_history_resume_rebuilds_the_committed_reference(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, "omni-clone-hist-build", mode="clone", reference=reference
        ),
    )
    assert code == 30
    run_uuid = _run_uuid("omni-clone-hist-build")
    script.unlink()

    # The rebuilt provider takes the reference locator and text from the committed
    # snapshot, never from the deleted script or the command line.
    state["fail_synthesis_on"] = None
    builds = _install_provider(monkeypatch, state)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert len(state["calls"]) == 2
    assert Path(builds[-1]["reference_audio_path"]) == reference.resolve()
    assert builds[-1]["reference_text"] == REFERENCE_TEXT


def test_native_local_omnivoice_clone_changed_reference_blocks_history_resume_before_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-hist-ref", mode="clone", reference=reference),
    )
    assert code == 30
    run_uuid = _run_uuid("omni-clone-hist-ref")
    before = list(state["calls"])

    reference.write_bytes(_mono_wav(99))
    _install_exploding_provider(monkeypatch)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
    assert state["calls"] == before


def test_native_local_omnivoice_design_history_resume_rebuilds_from_the_snapshot(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    builds = _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "omni-design-hist",
            mode="design",
            design_instruction=DESIGN_INSTRUCTION,
        ),
    )
    assert code == 0
    run_uuid = _run_uuid("omni-design-hist")
    before = list(state["calls"])
    script.unlink()

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert state["calls"] == before
    # The reconstructed provider repeats the committed mode, reference locator,
    # and instruction.
    assert builds[-1]["mode"] == "design"
    assert builds[-1]["design_instruction"] == DESIGN_INSTRUCTION


def test_native_local_omnivoice_auto_history_sync_is_export_only_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "omni-auto-synced", mode="auto")
    )
    assert code == 0
    run_uuid = _run_uuid("omni-auto-synced")
    before = list(state["calls"])
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert state["calls"] == before


def test_native_local_omnivoice_clone_sync_uncommitted_part_reports_local_error(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {
        "calls": [],
        "fail_synthesis_on": "chunk_01_omnivoice_session",
    }
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    reference = _reference(tmp_path)
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "omni-clone-sync", mode="clone", reference=reference),
    )
    assert code == 30
    run_uuid = _run_uuid("omni-clone-sync")
    before = list(state["calls"])
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
    assert state["calls"] == before


# ── route gate and legacy roots ───────────────────────────────────────────────


def test_native_local_omnivoice_mode_route_gate(tmp_path):
    import argparse

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixture")

    def _args(**overrides):
        base = dict(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="auto",
            no_trim=False,
            with_timings=False,
            tts_quality_provider=None,
            voice_bank_catalog=None,
            voice_bank_profile=None,
            reference_audio=None,
            reference_text=None,
            design_instruction=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    # The three non-preset modes are native with their own required inputs.
    assert cli._native_route_eligible(_args(), "markdown") is True
    assert cli._native_route_eligible(_args(mode="clone"), "markdown") is False
    assert (
        cli._native_route_eligible(
            _args(mode="clone", reference_audio=reference, reference_text="reference"),
            "markdown",
        )
        is True
    )
    assert cli._native_route_eligible(_args(mode="design"), "markdown") is False
    assert (
        cli._native_route_eligible(_args(mode="design", design_instruction="female"), "markdown")
        is True
    )
    # The admitted preset bank route is unchanged.
    assert cli._native_route_eligible(_args(), "markdown") is True
    assert (
        cli._native_route_eligible(
            _args(mode="preset", voice_bank_catalog=object(), voice_bank_profile=object()),
            "markdown",
        )
        is True
    )
    assert (
        cli._native_route_eligible(_args(mode="preset", voice_bank_catalog=object()), "markdown")
        is False
    )
    # ``--no-trim``, timings, a quality provider, another model, and a voiceover
    # script keep every OmniVoice route on the legacy executor.
    assert cli._native_route_eligible(_args(no_trim=True), "markdown") is False
    assert cli._native_route_eligible(_args(with_timings=True), "markdown") is False
    assert cli._native_route_eligible(_args(tts_quality_provider="qwen-local"), "markdown") is False
    assert cli._native_route_eligible(_args(model="another/model"), "markdown") is False
    assert cli._native_route_eligible(_args(), "voiceover") is False
    assert (
        cli._native_route_eligible(
            _args(mode="clone", reference_audio=reference, reference_text="reference"),
            "voiceover",
        )
        is False
    )


def test_native_local_omnivoice_mode_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "legacy-mode"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-mode"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()
