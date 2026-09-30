"""End-to-end contract tests for the native local OmniVoice dialogue route.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp
voice bank with two distinct mono-WAV reference profiles, a fake in-memory
OmniVoice provider, and patched FFmpeg/concat seams. No real provider, network
call, API key, ``.env``, model, or download is used. The tests assert per-cast
profile selection, the 250/600/0 ms pause plan, local raw recovery without a
second model run, fail-closed reference identity, no paid attempt or invented
cost, and the ``history resume``/``sync`` reconstruction rather than only
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
from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.models import ASRExecutionReceipt, ASRResult, SynthesisResult
from voiceover_pipeline.providers import OmniVoiceLocalTTSProvider
from voiceover_pipeline.services import native_generation, transcription

PROFILE_A = "voice_a"
PROFILE_B = "voice_b"
QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"


def _mono_wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes([seed]) * 160)
    return buffer.getvalue()


def _write_bank(root: Path) -> Path:
    """Write a two-profile voice bank whose references have distinct digests."""
    voices_dir = root / "voices"
    voices_dir.mkdir(parents=True)
    entries = []
    for profile_id, seed in ((PROFILE_A, 1), (PROFILE_B, 2)):
        data = _mono_wav(seed)
        (voices_dir / f"{profile_id}.wav").write_bytes(data)
        entries.append(
            {
                "id": profile_id,
                "display_name": profile_id,
                "description": "synthetic",
                "language": "ru",
                "reference_audio": f"voices/{profile_id}.wav",
                "reference_text": f"reference for {profile_id}",
                "reference_sha256": hashlib.sha256(data).hexdigest(),
                "origin": {"mode": "owner-reference", "instruction": None, "seed": None},
            }
        )
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {"schema_version": 1, "default_voice": PROFILE_A, "voices": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return catalog


class FakeOmniVoice(OmniVoiceLocalTTSProvider):
    """Offline stand-in that records the cast profile of every local turn.

    Subclassing the real provider keeps ``bind_dialogue_voice_bank_providers``'s
    ``isinstance`` check and the per-cast clone real, so a regression that drops
    the cast profile selection is observable here.
    """

    def __init__(self, state: dict[str, Any], profile_id: str | None = None) -> None:
        super().__init__(
            runtime=None,
            admitted_model=None,
            model_id=OMNIVOICE_LOCAL_MODEL_ID,
            mode="preset",
        )
        self._state = state
        self._profile_id = profile_id

    def for_voice_bank_profile(self, profile, reference_path) -> "FakeOmniVoice":
        return FakeOmniVoice(self._state, profile.id)

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self._state["calls"].append((self._profile_id, chunk_id, text))
        if self._state.get("fail_synthesis_on") == chunk_id:
            raise RuntimeError("local synthesis boom")
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="wav",
            transcript=text,
            client_path="omnivoice-local",
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
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)
    # The local quality preflight is a real dependency probe; the tests stub it so
    # no local ASR install is required. The per-turn model run is replaced per test.
    monkeypatch.setattr(native_generation, "_preflight_local_quality", lambda _options: None)
    return home


def _script(tmp_path: Path, sections: list[list[str]]) -> Path:
    path = tmp_path / "dialogue.md"
    lines = [
        "---",
        "format: dialogue",
        "language: ru",
        f"model: {OMNIVOICE_LOCAL_MODEL_ID}",
        "speakers:",
        "  Host:",
        f"    voice: {PROFILE_A}",
        "  Guest:",
        f"    voice: {PROFILE_B}",
        "---",
    ]
    for index, section in enumerate(sections):
        lines.extend(section)
        if index < len(sections) - 1:
            lines.append("******")
    path.write_text("\n".join(lines), encoding="utf-8")
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
        "--format",
        "dialogue",
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


def _install_provider(monkeypatch, state: dict[str, Any]):
    monkeypatch.setattr(cli, "build_provider", lambda *_a, **_k: FakeOmniVoice(state))


def _install_concat(monkeypatch):
    calls: list[list[tuple[str, int]]] = []

    def fake_concat(_ffmpeg: str, turns, output: Path) -> None:
        calls.append([(Path(path).name, pause) for path, pause in turns])
        output.write_bytes(b"-".join(Path(path).read_bytes() for path, _pause in turns))

    monkeypatch.setattr(cli, "concat_dialogue_turns", fake_concat)
    return calls


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


def _install_quality(monkeypatch, transcripts: dict[str, str]):
    """Replace the local ASR with a lookup keyed by the checked turn's file name."""
    calls: list[dict[str, Any]] = []

    def fake_transcribe(**kwargs):
        calls.append(dict(kwargs))
        return _asr_result(transcripts[Path(kwargs["audio_path"]).name])

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", fake_transcribe)
    return calls


def _turn_quality(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT parts.position AS position, artifacts.media_metadata_json AS metadata, "
        "text_sources.content AS transcript FROM artifacts JOIN text_sources "
        "ON text_sources.artifact_uuid = artifacts.artifact_uuid "
        "JOIN parts ON parts.part_uuid = artifacts.part_uuid "
        "WHERE artifacts.run_uuid = ? AND artifacts.role = 'tts_turn_quality_receipt' "
        "ORDER BY parts.position",
        (run_uuid,),
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
    """Return every local attempt's status for a run in durable insertion order."""
    return [
        row["status"]
        for row in _rows(
            home,
            "SELECT status FROM attempts WHERE run_uuid = ? ORDER BY rowid",
            (run_uuid,),
        )
    ]


# ── fresh two-speaker local dialogue ──────────────────────────────────────────


def test_native_local_dialogue_selects_each_cast_profile_and_pause_plan(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    concat_calls = _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(
        tmp_path,
        [
            ["Host: Первая реплика.", "Guest: Вторая реплика."],
            ["Host: Третья реплика.", "Guest: Четвёртая реплика."],
        ],
    )

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-dialogue")
    )

    assert code == 0, payload
    # Each turn is generated once, with its own cast voice-bank profile.
    assert state["calls"] == [
        (PROFILE_A, "turn_0001", "Первая реплика."),
        (PROFILE_B, "turn_0002", "Вторая реплика."),
        (PROFILE_A, "turn_0003", "Третья реплика."),
        (PROFILE_B, "turn_0004", "Четвёртая реплика."),
    ]
    # The final concat preserves the recorded 250/600/0 pause plan and turn order.
    assert concat_calls == [
        [
            ("turn_0001.mp3", 250),
            ("turn_0002.mp3", 600),
            ("turn_0003.mp3", 250),
            ("turn_0004.mp3", 0),
        ]
    ]
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "dialogue"
    assert chunks["speaker_voice_map"] == {"Host": PROFILE_A, "Guest": PROFILE_B}
    assert [chunk["voice"] for chunk in chunks["chunks"]] == [
        PROFILE_A,
        PROFILE_B,
        PROFILE_A,
        PROFILE_B,
    ]
    assert [chunk["pause_after_ms"] for chunk in chunks["chunks"]] == [250, 600, 250, 0]
    assert [chunk["turn_index"] for chunk in chunks["chunks"]] == [1, 2, 3, 4]
    assert all(chunk["audio_sha256"] for chunk in chunks["chunks"])
    # A local run has no linked timing or quality gate in its payload.
    assert "timing" not in payload
    assert "quality" not in payload

    run_uuid = _run_uuid("local-dialogue")
    home = local_env
    # Every turn records exactly one durable, cost-free local attempt; no cloud
    # unknown and no invented zero is written.
    attempts = _rows(
        home,
        "SELECT attempt_uuid, part_uuid, call_type, provider, model, status, cost, "
        "cost_currency FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert len(attempts) == 4
    assert {row["call_type"] for row in attempts} == {"local_tts_chunk"}
    assert {row["provider"] for row in attempts} == {"omnivoice-local"}
    assert {row["model"] for row in attempts} == {OMNIVOICE_LOCAL_MODEL_ID}
    assert {row["status"] for row in attempts} == {"local_completed"}
    assert all(row["part_uuid"] for row in attempts)
    # Exact NULL costs, never a fabricated zero.
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)
    assert chunks.get("cost_total") is None
    assert chunks.get("cost_currency") is None
    attempt_by_part = {row["part_uuid"]: row["attempt_uuid"] for row in attempts}
    run_root = tmp_path / "out" / "local-dialogue"
    # Each turn's local raw bytes are linked with no paid receipt on disk, and both
    # the raw and the converted chunk artifacts point at that turn's local attempt.
    assert (run_root / "raw" / "turn_0001.wav").read_bytes() == b"turn_0001-audio"
    assert not (run_root / "raw" / "turn_0001.wav.receipt.json").exists()
    linked = _rows(
        home,
        "SELECT part_uuid, role, attempt_uuid FROM artifacts "
        "WHERE run_uuid = ? AND role IN ('local_raw_audio', 'chunk_audio')",
        (run_uuid,),
    )
    assert len(linked) == 8
    assert {row["role"] for row in linked} == {"local_raw_audio", "chunk_audio"}
    assert all(row["attempt_uuid"] == attempt_by_part[row["part_uuid"]] for row in linked)


# ── local retry and raw recovery ──────────────────────────────────────────────


def test_native_local_dialogue_resume_retries_only_the_unfinished_turn(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "turn_0002"}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-retry")
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_SYNTHESIS_FAILED"
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002"]
    run_uuid = _run_uuid("local-retry")
    # The finished first turn records a completed attempt and the failed second turn
    # records its own failed attempt; neither is a paid marker.
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", "local_failed"]

    # An unconfirmed local attempt may be retried on an explicit resume.
    state["fail_synthesis_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-retry", extra=("--resume",)),
    )
    assert code == 0, payload
    # The finished first turn is never re-synthesized; only the remaining turn runs.
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002", "turn_0002"]
    # The resumed invocation records its own attempt, so every real local call is
    # counted and the failed one keeps its truthful outcome.
    assert _attempt_statuses(local_env, run_uuid) == [
        "local_completed",
        "local_failed",
        "local_completed",
    ]


def test_native_local_dialogue_rebuilds_from_linked_raw_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_conversion_on": "turn_0002"}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    run_root = tmp_path / "out" / "local-raw"

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-raw")
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    # The second turn's raw bytes are linked even though its conversion failed.
    assert (run_root / "raw" / "turn_0002.wav").read_bytes() == b"turn_0002-audio"
    assert not (run_root / "chunks" / "turn_0002.mp3").exists()

    state["fail_conversion_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-raw", extra=("--resume",)),
    )
    assert code == 0, payload
    # The pending turn is rebuilt from its linked raw bytes; no second model run.
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002"]
    assert (run_root / "chunks" / "turn_0002.mp3").read_bytes() == b"turn_0002-audio"


# ── local attempt accounting and sync ────────────────────────────────────────


def test_native_local_dialogue_counts_two_local_attempts_and_costs(
    tmp_path, monkeypatch, capsys, local_env
):
    """A two-turn local run persists two attempts and counts them in history costs."""
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-two")
    )
    assert code == 0, payload
    run_uuid = _run_uuid("local-two")

    attempts = _rows(
        local_env,
        "SELECT call_type, provider, model, status, cost, cost_currency "
        "FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert len(attempts) == 2
    assert [row["call_type"] for row in attempts] == ["local_tts_chunk", "local_tts_chunk"]
    assert {row["provider"] for row in attempts} == {"omnivoice-local"}
    assert {row["model"] for row in attempts} == {OMNIVOICE_LOCAL_MODEL_ID}
    assert {row["status"] for row in attempts} == {"local_completed"}
    assert all(row["cost"] is None and row["cost_currency"] is None for row in attempts)

    code, detail = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "show"))
    assert code == 0, detail
    assert len(detail["attempts"]) == 2
    assert {attempt["call_type"] for attempt in detail["attempts"]} == {"local_tts_chunk"}

    code, costs = _json_run(
        monkeypatch, capsys, ["voiceover-pipeline", "history", "costs", "--json"]
    )
    assert code == 0, costs
    # The two real local attempts are counted as local work without an API charge
    # and neither as a cloud unknown nor as an invented zero.
    assert costs["local_attempts_without_api_charge"] == 2
    assert costs["attempts"] == 2
    assert costs["completeness"] == "complete"
    assert all(total["unknown_attempts"] == 0 for total in costs["totals"])
    assert costs["totals"] == []


def test_native_local_dialogue_interrupted_invocation_then_resume_is_truthful(
    tmp_path, monkeypatch, capsys, local_env
):
    """An interrupted second turn leaves a pending attempt and resumes truthfully."""
    from voiceover_pipeline.history.repository import HistoryRepository

    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    real_link = HistoryRepository.record_local_tts_raw_saved
    links = {"n": 0}

    def interrupting_link(self, *args, **kwargs):
        links["n"] += 1
        if links["n"] == 2:
            # Simulate the process dying between the reserved local attempt and its
            # raw link for the second turn.
            raise KeyboardInterrupt
        return real_link(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", interrupting_link)
    monkeypatch.setattr(sys, "argv", _generate_argv(tmp_path, script, catalog, "local-interrupt"))
    with pytest.raises(KeyboardInterrupt):
        cli.main()

    run_uuid = _run_uuid("local-interrupt")
    # The finished first turn recorded a completed attempt; the interrupted second
    # turn left one pending attempt with no raw link.
    assert _attempt_statuses(local_env, run_uuid) == ["local_completed", None]

    monkeypatch.setattr(HistoryRepository, "record_local_tts_raw_saved", real_link)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-interrupt", extra=("--resume",)),
    )
    assert code == 0, payload
    # The completed first turn is never re-synthesized; only the interrupted turn runs.
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002", "turn_0002"]
    assert _attempt_statuses(local_env, run_uuid) == [
        "local_completed",
        None,
        "local_completed",
    ]

    code, detail = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "show"))
    assert code == 0, detail
    assert len(detail["attempts"]) == 3
    code, costs = _json_run(
        monkeypatch, capsys, ["voiceover-pipeline", "history", "costs", "--json"]
    )
    assert code == 0, costs
    assert costs["local_attempts_without_api_charge"] == 3


def test_native_local_dialogue_sync_uncommitted_part_reports_local_error(
    tmp_path, monkeypatch, capsys, local_env
):
    """history sync never runs the local model and reports its own local error."""
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "turn_0002"}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-sync-pending")
    )
    assert code == 30
    run_uuid = _run_uuid("local-sync-pending")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30
    # The error names the local no-inference rule, never a paid submit refusal.
    assert payload["details"]["error_code"] == "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
    assert state["calls"] == before


# ── reference identity fail-closed ────────────────────────────────────────────


def test_native_local_dialogue_changed_reference_file_fails_before_synthesis(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "turn_0002"}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-ref")
    )
    assert code == 30
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002"]

    # A reference file whose bytes no longer match the committed catalog digest is
    # refused before the pending turn is synthesized again.
    (tmp_path / "bank" / "voices" / f"{PROFILE_B}.wav").write_bytes(_mono_wav(9))
    state["fail_synthesis_on"] = None
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-ref", extra=("--resume",)),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
    assert [call[1] for call in state["calls"]] == ["turn_0001", "turn_0002"]


def test_native_local_dialogue_changed_reference_digest_fails_resume_identity(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": [], "fail_synthesis_on": "turn_0002"}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-digest")
    )
    assert code == 30

    # A catalog whose reference digest changed is a changed synthesis identity.
    payload_json = json.loads(catalog.read_text(encoding="utf-8"))
    payload_json["voices"][1]["reference_sha256"] = "f" * 64
    catalog.write_text(json.dumps(payload_json, ensure_ascii=False), encoding="utf-8")
    state["fail_synthesis_on"] = None
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-digest", extra=("--resume",)),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"


# ── history resume / sync and legacy roots ────────────────────────────────────


def test_native_local_dialogue_history_resume_after_script_deleted(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-hist")
    )
    assert code == 0
    run_uuid = _run_uuid("local-hist")
    before = list(state["calls"])
    script.unlink()
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert Path(payload["files"]["full_mp3"]).is_file()
    assert state["calls"] == before


def test_native_local_dialogue_history_sync_is_export_only_without_a_model(
    tmp_path, monkeypatch, capsys, local_env
):
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, catalog, "local-sync")
    )
    assert code == 0
    run_uuid = _run_uuid("local-sync")
    before = list(state["calls"])
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert state["calls"] == before


def test_native_local_dialogue_optional_quality_gate_runs_before_concat(
    tmp_path, monkeypatch, capsys, local_env
):
    """The optional OmniVoice dialogue QA runs per turn and links its verdicts."""
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    concat_calls = _install_concat(monkeypatch)
    quality_calls = _install_quality(
        monkeypatch,
        {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."},
    )
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            catalog,
            "local-dialogue-qa",
            extra=("--tts-quality-provider", "qwen-local"),
        ),
    )

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert len(quality_calls) == 2
    assert len(concat_calls) == 1
    run_uuid = _run_uuid("local-dialogue-qa")
    verdicts = _turn_quality(local_env, run_uuid)
    assert [row["position"] for row in verdicts] == [1, 2]
    assert all(json.loads(row["metadata"])["quality_passed"] is True for row in verdicts)
    assert [row["transcript"] for row in verdicts] == ["Первая реплика.", "Вторая реплика."]
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert "verification_transcript" not in json.dumps(chunks, ensure_ascii=False)


def test_native_local_dialogue_quality_fail_stops_concat_and_is_never_repeated(
    tmp_path, monkeypatch, capsys, local_env
):
    """A recorded OmniVoice dialogue FAIL keeps the local audio and never reruns."""
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    concat_calls = _install_concat(monkeypatch)
    quality_calls = _install_quality(
        monkeypatch,
        {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Совсем не то."},
    )
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    extra = ("--tts-quality-provider", "qwen-local")

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-dialogue-fail", extra=extra),
    )
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    # The failed turn stops the final concat; the local audio and attempts stay.
    assert concat_calls == []
    run_uuid = _run_uuid("local-dialogue-fail")
    verdicts = _turn_quality(local_env, run_uuid)
    assert {json.loads(row["metadata"])["quality_passed"] for row in verdicts} == {True, False}
    calls_after_fail = len(quality_calls)
    calls_after_tts = list(state["calls"])

    # A resume re-reports the durable verdict without running the local model.
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, catalog, "local-dialogue-fail", extra=(*extra, "--resume")
        ),
    )
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"

    # ``history sync`` re-reports the recorded failure and runs no local model.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    assert len(quality_calls) == calls_after_fail
    assert state["calls"] == calls_after_tts
    assert concat_calls == []


def test_native_local_dialogue_no_trim_skips_the_trim_seam_and_is_recorded(
    tmp_path, monkeypatch, capsys, local_env
):
    """``--no-trim`` is recorded and skips the per-turn trim on the local route."""
    state: dict[str, Any] = {"calls": []}
    _install_provider(monkeypatch, state)
    _install_concat(monkeypatch)
    trims: list[str] = []
    monkeypatch.setattr(cli, "trim_final_silence", lambda _f, _p, path: trims.append(path.name))
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda f, d, fmt, p: _write_mp3(f, d, fmt, p, state)
    )
    catalog = _write_bank(tmp_path / "bank")
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, catalog, "local-notrim", extra=("--no-trim",)),
    )
    assert code == 0, payload
    assert trims == []
    run_uuid = _run_uuid("local-notrim")
    snapshot = _rows(local_env, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,))[
        0
    ]
    assert json.loads(snapshot["config_snapshot"])["output"]["trim_final_silence"] is False


def test_native_local_dialogue_legacy_root_stays_legacy(tmp_path, monkeypatch, local_env):
    run_root = tmp_path / "out" / "legacy-local"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-local"}), encoding="utf-8"
    )
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


def test_native_local_dialogue_route_gate_requires_the_admitted_bank():
    import argparse

    args = argparse.Namespace(
        provider="omnivoice-local",
        model=OMNIVOICE_LOCAL_MODEL_ID,
        mode="preset",
        no_trim=False,
        with_timings=False,
        tts_quality_provider=None,
        voice_bank_catalog=None,
    )
    from voiceover_pipeline.gemini_dialogue import DIALOGUE_FORMAT

    # Without an admitted voice bank the route stays on the legacy executor.
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.voice_bank_catalog = object()
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    # ``--no-trim``, integrated local timings, and an installed local quality
    # provider are recorded on the route; a cloud timing or quality provider keeps
    # it legacy.
    args.no_trim = True
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.no_trim = False
    args.with_timings = True
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.timing_provider = "xai-stt"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.timing_provider = "faster-whisper"
    args.with_timings = False
    args.tts_quality_provider = "qwen-local"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.tts_quality_provider = "xai-stt"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.tts_quality_provider = None
    args.mode = "clone"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
