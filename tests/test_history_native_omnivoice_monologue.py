"""End-to-end contract tests for the native local OmniVoice monologue bank route.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp
voice bank with a distinct mono-WAV reference profile, a fake in-memory
OmniVoice provider installed behind the real provider factory, and patched
FFmpeg/concat seams. No real provider, network call, API key, ``.env``, model, or
download is used. The tests assert the single merged session call with the
selected bank profile, local raw recovery without a second model run, fail-closed
reference identity, no paid attempt or invented cost, the ``history
resume``/``sync`` reconstruction rather than only statuses, and that the private
voice-bank reference text never reaches an exported artifact.
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

PROFILE_A = "voice_a"
REFERENCE_TEXT = "reference for voice_a"


def _mono_wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes([seed]) * 160)
    return buffer.getvalue()


def _write_bank(root: Path) -> Path:
    """Write a one-profile voice bank with a distinct reference digest."""
    voices_dir = root / "voices"
    voices_dir.mkdir(parents=True)
    data = _mono_wav(1)
    (voices_dir / f"{PROFILE_A}.wav").write_bytes(data)
    entry = {
        "id": PROFILE_A,
        "display_name": PROFILE_A,
        "description": "synthetic",
        "language": "ru",
        "reference_audio": f"voices/{PROFILE_A}.wav",
        "reference_text": REFERENCE_TEXT,
        "reference_sha256": hashlib.sha256(data).hexdigest(),
        "origin": {"mode": "owner-reference", "instruction": None, "seed": None},
    }
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {"schema_version": 1, "default_voice": PROFILE_A, "voices": [entry]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return catalog


class FakeOmniVoice:
    """Offline stand-in installed behind the real provider factory.

    Replacing the provider class (not the factory) keeps the real bank-catalog
    admission and ``resolve_bank_profile`` reference verification on the path, so
    a regression that stops verifying the committed reference stays observable.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "FakeOmniVoice":
        # ``build_tts_provider`` passes the admitted bank tuple as ``voice_bank``.
        reference = kwargs.get("voice_bank")
        if reference is not None:
            assert reference[0].reference_text == REFERENCE_TEXT
        return cls(_FAKE_STATE["state"])

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        state = self._state
        state["calls"].append((chunk_id, text))
        if state.get("fail_synthesis_on") == chunk_id:
            raise RuntimeError("local synthesis boom")
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="omnivoice-local",
        )


class _ExplodingProvider:
    """A provider class that fails loudly if an export/recovery path builds one."""

    @classmethod
    def from_environment(cls, **_kwargs: Any) -> Any:  # pragma: no cover - asserted never to run
        raise AssertionError("this path must not build a provider")


_FAKE_STATE: dict[str, Any] = {}


def _install_provider(monkeypatch, state: dict[str, Any]) -> None:
    _FAKE_STATE["state"] = state
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", FakeOmniVoice)


def _install_exploding_provider(monkeypatch) -> None:
    monkeypatch.setattr(provider_factory, "OmniVoiceLocalTTSProvider", _ExplodingProvider)


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
    path.write_text("Первый фрагмент.\n\n******\n\nВторой фрагмент.\n", encoding="utf-8")
    return path


def _generate_argv(tmp_path: Path, script: Path, catalog: Path, run_id: str, *, extra=()):
    return [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        OMNIVOICE_LOCAL_MODEL_ID,
        "--mode",
        "preset",
        "--voice-bank",
        str(catalog),
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


# ── fresh monologue bank run ──────────────────────────────────────────────────


def test_native_local_monologue_clones_the_selected_bank_profile_once(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono")
    )

    assert code == 0, payload
    # The whole script merges into one OmniVoice session part, synthesized once.
    assert state["calls"] == [
        ("chunk_01_omnivoice_session", "Первый фрагмент. Второй фрагмент."),
    ]
    assert Path(payload["files"]["full_mp3"]).is_file()
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "markdown"
    assert chunks["voice"] == PROFILE_A
    assert [chunk["id"] for chunk in chunks["chunks"]] == ["chunk_01_omnivoice_session"]

    run_uuid = _run_uuid("local-mono")
    home = local_env
    attempts = _rows(
        home,
        "SELECT part_uuid, call_type, provider, model, status, cost, cost_currency "
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
    run_root = tmp_path / "out" / "local-mono"
    # The raw bytes are linked with no paid receipt on disk.
    assert (run_root / "raw" / "chunk_01_omnivoice_session.wav").read_bytes() == (
        b"chunk_01_omnivoice_session-audio"
    )
    assert not (run_root / "raw" / "chunk_01_omnivoice_session.wav.receipt.json").exists()
    linked = _rows(
        home,
        "SELECT role, attempt_uuid, part_uuid FROM artifacts "
        "WHERE run_uuid = ? AND role IN ('local_raw_audio', 'chunk_audio')",
        (run_uuid,),
    )
    assert {row["role"] for row in linked} == {"local_raw_audio", "chunk_audio"}
    assert len({row["attempt_uuid"] for row in linked}) == 1
    assert all(row["attempt_uuid"] is not None for row in linked)
    assert all(row["part_uuid"] == attempts[0]["part_uuid"] for row in linked)


# ── local retry and raw recovery ──────────────────────────────────────────────


def test_native_local_monologue_resume_retries_only_the_unfinished_part(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_01_omnivoice_session"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-retry")
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_SYNTHESIS_FAILED"
    run_uuid = _run_uuid("local-mono-retry")
    assert _attempt_statuses(local_env, run_uuid) == ["local_failed"]

    state["fail_synthesis_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-mono-retry", extra=("--resume",)),
    )
    assert code == 0, payload
    # The failed invocation is repeatable, and the retry records its own attempt.
    assert len(state["calls"]) == 2
    assert _attempt_statuses(local_env, run_uuid) == ["local_failed", "local_completed"]


def test_native_local_monologue_rebuilds_from_linked_raw_without_a_model(
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
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)
    run_root = tmp_path / "out" / "local-mono-raw"

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-raw")
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
        _generate_argv(tmp_path, script, catalog, "local-mono-raw", extra=("--resume",)),
    )
    assert code == 0, payload
    # The part is rebuilt from its linked raw bytes; no second model run.
    assert len(state["calls"]) == 1
    assert (run_root / "chunks" / "chunk_01_omnivoice_session.mp3").read_bytes() == (
        b"chunk_01_omnivoice_session-audio"
    )


def test_native_local_monologue_interrupted_invocation_then_resume_is_truthful(
    tmp_path, monkeypatch, capsys, local_env
):
    from voiceover_pipeline.history.repository import HistoryRepository

    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)

    real_link = HistoryRepository.record_local_tts_raw_saved

    def interrupting_link(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", interrupting_link)
    monkeypatch.setattr(sys, "argv", _generate_argv(tmp_path, script, catalog, "local-mono-die"))
    with pytest.raises(KeyboardInterrupt):
        cli.main()

    run_uuid = _run_uuid("local-mono-die")
    # The interrupted local invocation left one pending attempt with no raw link.
    assert _attempt_statuses(local_env, run_uuid) == [None]

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", real_link)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-mono-die", extra=("--resume",)),
    )
    assert code == 0, payload
    assert _attempt_statuses(local_env, run_uuid) == [None, "local_completed"]


# ── local attempt accounting and sync ────────────────────────────────────────


def test_native_local_monologue_counts_one_local_attempt_and_cost(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-cost")
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


def test_native_local_monologue_sync_uncommitted_part_reports_local_error(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "chunk_01_omnivoice_session"}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-sync")
    )
    assert code == 30
    run_uuid = _run_uuid("local-mono-sync")
    before = list(state["calls"])
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
    assert state["calls"] == before


# ── reference identity fail-closed ────────────────────────────────────────────


def test_native_local_monologue_missing_reference_fails_before_synthesis(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    catalog = _write_bank(tmp_path / "bank")
    (tmp_path / "bank" / "voices" / f"{PROFILE_A}.wav").unlink()
    script = _script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-missing")
    )

    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
    # The reference is refused before the local model is ever called.
    assert state["calls"] == []


def test_native_local_monologue_changed_reference_digest_fails_resume_identity(
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
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-digest")
    )
    assert code == 30

    payload_json = json.loads(catalog.read_text(encoding="utf-8"))
    payload_json["voices"][0]["reference_sha256"] = "f" * 64
    catalog.write_text(json.dumps(payload_json, ensure_ascii=False), encoding="utf-8")
    state["fail_synthesis_on"] = None
    _install_exploding_provider(monkeypatch)
    before = list(state["calls"])
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-mono-digest", extra=("--resume",)),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert state["calls"] == before


# ── history resume / sync and legacy roots ────────────────────────────────────


def test_native_local_monologue_history_resume_after_script_deleted(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-hist")
    )
    assert code == 0
    run_uuid = _run_uuid("local-mono-hist")
    before = list(state["calls"])
    script.unlink()
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert Path(payload["files"]["full_mp3"]).is_file()
    assert state["calls"] == before


def test_native_local_monologue_history_sync_is_export_only_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-synced")
    )
    assert code == 0
    run_uuid = _run_uuid("local-mono-synced")
    before = list(state["calls"])
    _install_exploding_provider(monkeypatch)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert state["calls"] == before


def test_native_local_monologue_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "legacy-mono"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-mono"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


# ── output privacy ────────────────────────────────────────────────────────────


def test_native_local_monologue_outputs_never_publish_the_reference_text(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-mono-priv")
    )
    assert code == 0, payload

    run_root = tmp_path / "out" / "local-mono-priv"
    exported = sorted(run_root.rglob("*.json"))
    assert exported, "expected committed compatibility JSON exports"
    for path in exported:
        assert REFERENCE_TEXT not in path.read_text(encoding="utf-8"), path.name

    run_uuid = _run_uuid("local-mono-priv")
    code, detail = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "show"))
    assert code == 0, detail
    assert REFERENCE_TEXT not in json.dumps(detail, ensure_ascii=False)

    # The private script is still committed as a native text source.
    kinds = _rows(
        local_env,
        "SELECT DISTINCT kind FROM text_sources WHERE run_uuid = ?",
        (run_uuid,),
    )
    assert {row["kind"] for row in kinds} == {"tts_script"}


# ── route gate and route regressions ─────────────────────────────────────────


def test_native_local_monologue_route_gate():
    import argparse

    def _args(**overrides):
        base = dict(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="preset",
            voice_bank_catalog=object(),
            voice_bank_profile=object(),
            no_trim=False,
            with_timings=False,
            tts_quality_provider=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    # The admitted bank preset run is native.
    assert cli._native_route_eligible(_args(), "markdown") is True
    # ``auto``/``clone``/``design`` carry no bank profile and stay legacy.
    assert cli._native_route_eligible(_args(mode="auto"), "markdown") is False
    assert cli._native_route_eligible(_args(mode="clone"), "markdown") is False
    assert cli._native_route_eligible(_args(mode="design"), "markdown") is False
    # A missing catalog or unresolved profile stays legacy.
    assert cli._native_route_eligible(_args(voice_bank_catalog=None), "markdown") is False
    assert cli._native_route_eligible(_args(voice_bank_profile=None), "markdown") is False
    # ``--no-trim``, integrated timings, or a quality provider keep it legacy.
    assert cli._native_route_eligible(_args(no_trim=True), "markdown") is False
    assert cli._native_route_eligible(_args(with_timings=True), "markdown") is False
    assert cli._native_route_eligible(_args(tts_quality_provider="qwen-local"), "markdown") is False
    # A non-markdown script stays legacy, and the paid/openrouter route is unchanged.
    assert cli._native_route_eligible(_args(), "voiceover") is False
    paid = argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        tts_quality_provider=None,
        with_timings=False,
    )
    assert cli._native_route_eligible(paid, "markdown") is True
