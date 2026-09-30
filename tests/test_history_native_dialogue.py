"""End-to-end contract tests for the native DB-first OpenRouter Gemini dialogue slice.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, a fake in-memory OpenRouter TTS provider, a fake installed local ASR,
and patched FFmpeg/concat seams. No real provider, network call, API key, ``.env``,
model, or paid request is used. The tests assert paid-submit counts, per-turn
voices, the 250/600/0 ms pause plan, the durable per-turn PASS/FAIL verdicts, and
the emitted JSON envelope rather than only statuses, so a slice that repeats a
paid submit, drops a recorded failure, reorders turns, or leaks the private
verification transcript fails here.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.gemini_dialogue import DIALOGUE_FORMAT, GEMINI_TTS_MODEL
from voiceover_pipeline.models import (
    ASRExecutionReceipt,
    ASRResult,
    SynthesisResult,
    TimingResult,
    TimingSegment,
)
from voiceover_pipeline.providers import OpenRouterTTSProvider
from voiceover_pipeline.services import native_generation, transcription

QWEN_MODEL = "Qwen/Qwen3-ASR-0.6B"
QUALITY_ARGS = ("--tts-quality-provider", "qwen-local")


class FakeDialogueProvider(OpenRouterTTSProvider):
    """Offline stand-in that records every per-turn voice override.

    Subclassing the real provider keeps ``services.synthesis.synthesize_part``
    passing the per-turn ``voice`` keyword exactly as it does in production, so a
    regression that drops the cast voice is observable here. ``fail_on`` makes one
    turn raise before it returns, modelling an interrupted or unconfirmed submit.
    """

    def __init__(self) -> None:
        super().__init__(api_key="sk-test", model=GEMINI_TTS_MODEL, voice="Kore")
        self.calls: list[tuple[str, str | None]] = []
        self.fail_on: str | None = None

    def synthesize_chunk(self, text: str, chunk_id: str, voice: str | None = None):
        self.calls.append((chunk_id, voice))
        if self.fail_on == chunk_id:
            raise requests.Timeout("read timed out")
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
        )


def _asr_result(transcript: str) -> ASRResult:
    return ASRResult(
        transcript=transcript,
        provider_id="qwen-local",
        model_id=QWEN_MODEL,
        execution=ASRExecutionReceipt(
            runtime="python",
            model_revision="rev-1",
            resolved_device="cpu",
            resolved_compute="auto",
        ),
        language="ru",
    )


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat_audio(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


@pytest.fixture
def dialogue_env(tmp_path, monkeypatch):
    """Point history at a temp home and replace every local media seam."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat_audio)
    # The preflight and the model run are replaced per test; a no-op preflight is
    # the default so only the explicit preflight test exercises the installer probe.
    monkeypatch.setattr(native_generation, "_preflight_local_quality", lambda _options: None)
    monkeypatch.setattr(native_generation, "_preflight_local_timing", lambda _options: None)
    return home


def _script(tmp_path: Path, sections: list[list[str]]) -> Path:
    """Write a validated two-speaker dialogue script from per-section turn lines."""
    path = tmp_path / "dialogue.md"
    lines = [
        "---",
        "format: gemini-dialogue",
        "language: ru",
        f"model: {GEMINI_TTS_MODEL}",
        "speakers:",
        "  Host:",
        "    voice: Kore",
        "  Guest:",
        "    voice: Puck",
        "---",
    ]
    for index, section in enumerate(sections):
        lines.extend(section)
        if index < len(sections) - 1:
            lines.append("******")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _generate_argv(tmp_path: Path, script: Path, run_id: str, *, extra=()):
    return [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "openrouter-tts",
        "--model",
        GEMINI_TTS_MODEL,
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


def _install_provider(monkeypatch, provider):
    builds: list[str] = []

    def fake_build(*_args, **_kwargs):
        builds.append("built")
        return provider

    monkeypatch.setattr(cli, "build_provider", fake_build)
    return builds


def _install_quality(monkeypatch, transcripts: dict[str, str], error: Exception | None = None):
    """Replace the local ASR with a lookup keyed by the checked turn's file name."""
    calls: list[dict[str, Any]] = []

    def fake_transcribe(**kwargs):
        calls.append(dict(kwargs))
        if error is not None:
            raise error
        audio_name = Path(kwargs["audio_path"]).name
        return _asr_result(transcripts[audio_name])

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", fake_transcribe)
    return calls


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
                text="Текст распознавания таймингов.",
            )
        ],
        model="small",
        backend="faster-whisper",
        provider="faster-whisper",
        device="cpu",
        compute_type="int8",
        language="ru",
    )


def _install_timing(monkeypatch, error: Exception | None = None):
    calls: list[dict[str, Any]] = []

    def fake_transcribe_timing(**kwargs):
        calls.append(dict(kwargs))
        if error is not None:
            raise error
        return _timing_result()

    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake_transcribe_timing)
    return calls


def _install_concat(monkeypatch):
    calls: list[list[tuple[str, int]]] = []

    def fake_concat(_ffmpeg: str, turns, output: Path) -> None:
        calls.append([(Path(path).name, pause) for path, pause in turns])
        output.write_bytes(b"-".join(Path(path).read_bytes() for path, _pause in turns))

    monkeypatch.setattr(cli, "concat_dialogue_turns", fake_concat)
    return calls


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


def _turn_quality(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT artifacts.media_metadata_json AS metadata, text_sources.content AS transcript "
        "FROM artifacts JOIN text_sources "
        "ON text_sources.artifact_uuid = artifacts.artifact_uuid "
        "JOIN parts ON parts.part_uuid = artifacts.part_uuid "
        "WHERE artifacts.run_uuid = ? AND artifacts.role = 'tts_turn_quality_receipt' "
        "ORDER BY parts.position",
        (run_uuid,),
    )


def _timing_children(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT * FROM runs WHERE parent_uuid = ? AND operation = 'timings'",
        (run_uuid,),
    )


def _paid_attempts(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    """The run's paid TTS attempts in turn order with their paid raw artifact link.

    One row per durable attempt: its identity, status, remote id, cost and the
    ``paid_raw_audio`` artifact linked to it by ``attempt_uuid`` (``NULL`` when
    that link is missing). This lets a test prove the paid evidence survives a
    post-audio failure without being rewritten by a later resume.
    """
    return _rows(
        home,
        "SELECT a.attempt_uuid, a.call_type, a.status, a.remote_id, a.cost, "
        "a.cost_exact_available, a.part_uuid, art.part_uuid AS raw_part_uuid "
        "FROM attempts a JOIN parts p ON p.part_uuid = a.part_uuid "
        "LEFT JOIN artifacts art ON art.attempt_uuid = a.attempt_uuid "
        "AND art.role = 'paid_raw_audio' "
        "WHERE a.run_uuid = ? ORDER BY p.position",
        (run_uuid,),
    )


# ── fresh two-speaker dialogue ────────────────────────────────────────────────


def test_native_dialogue_fresh_two_speakers_pass_concat_and_export(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """Distinct cast voices, the 250/600/0 pause plan, and per-turn PASS proofs."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    concat_calls = _install_concat(monkeypatch)
    script = _script(
        tmp_path,
        [
            ["Host: Первая реплика.", "Guest: Вторая реплика."],
            ["Host: Третья реплика.", "Guest: Четвёртая реплика."],
        ],
    )
    transcripts = {
        "turn_0001.mp3": "Первая реплика.",
        "turn_0002.mp3": "Вторая реплика.",
        "turn_0003.mp3": "Третья реплика.",
        "turn_0004.mp3": "Четвёртая реплика.",
    }
    quality_calls = _install_quality(monkeypatch, transcripts)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-dialogue", extra=QUALITY_ARGS)
    )

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert payload["files"]["full_mp3"].endswith(".mp3")
    # Each turn is submitted once with its own cast voice, in script order.
    assert provider.calls == [
        ("turn_0001", "Kore"),
        ("turn_0002", "Puck"),
        ("turn_0003", "Kore"),
        ("turn_0004", "Puck"),
    ]
    assert len(quality_calls) == 4
    # The final concat preserves the exact recorded pause plan and turn order.
    assert concat_calls == [
        [
            ("turn_0001.mp3", 250),
            ("turn_0002.mp3", 600),
            ("turn_0003.mp3", 250),
            ("turn_0004.mp3", 0),
        ]
    ]

    run_uuid = _run_uuid("native-dialogue")
    # Each paid turn keeps its raw bytes and a bounded receipt with no remote id and
    # only the bounded generation id the synchronous OpenRouter response carried.
    run_root = tmp_path / "out" / "native-dialogue"
    receipt = json.loads(
        (run_root / "raw" / "turn_0001.mp3.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["remote_task_id"] is None
    assert receipt["generation_id"] == "gen-turn_0001"
    # The TTS run itself carries one private transcript and one receipt per turn,
    # both linked to the exact part.
    per_turn = _turn_quality(dialogue_env, run_uuid)
    assert [json.loads(row["metadata"])["passed"] for row in per_turn] == [True, True, True, True]
    assert [row["transcript"] for row in per_turn] == [
        transcripts[name] for name in sorted(transcripts)
    ]
    linked_parts = _rows(
        dialogue_env,
        "SELECT DISTINCT part_uuid FROM artifacts WHERE run_uuid = ? "
        "AND role = 'tts_turn_quality_receipt'",
        (run_uuid,),
    )
    assert len(linked_parts) == 4 and all(row["part_uuid"] for row in linked_parts)

    # The DB-derived exports are dialogue-compatible: ordered turn fields, the cast
    # map, and a content-free quality receipt, all carrying UUID/revision.
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "dialogue"
    assert chunks["speaker_voice_map"] == {"Host": "Kore", "Guest": "Puck"}
    assert [chunk["turn_index"] for chunk in chunks["chunks"]] == [1, 2, 3, 4]
    assert [chunk["voice"] for chunk in chunks["chunks"]] == ["Kore", "Puck", "Kore", "Puck"]
    assert [chunk["pause_after_ms"] for chunk in chunks["chunks"]] == [250, 600, 250, 0]
    assert all(chunk["audio_sha256"] for chunk in chunks["chunks"])
    quality_receipt = chunks["tts_quality"]
    assert quality_receipt["passed"] is True
    assert quality_receipt["turn_count"] == 4
    # The aggregate receipt is content-free: only the observed-word hashes and
    # counts, never the private transcript text.
    receipt_text = json.dumps(quality_receipt, ensure_ascii=False)
    assert all(text not in receipt_text for text in transcripts.values())
    for name in ("chunks_json", "run_json", "manifest_json"):
        document = json.loads(Path(payload["files"][name]).read_text(encoding="utf-8"))
        assert document["history_run_uuid"] == run_uuid
        assert isinstance(document["history_revision"], int)


def test_native_dialogue_requires_two_distinct_validated_voices(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """The existing validator refuses a duplicate voice before any native work."""
    _install_provider(monkeypatch, FakeDialogueProvider())
    path = tmp_path / "dialogue.md"
    path.write_text(
        "\n".join(
            [
                "---",
                "format: dialogue",
                f"model: {GEMINI_TTS_MODEL}",
                "speakers:",
                "  Host:",
                "    voice: Kore",
                "  Guest:",
                "    voice: Kore",
                "---",
                "Host: Привет.",
                "Guest: Здравствуйте.",
            ]
        ),
        encoding="utf-8",
    )
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, path, "native-dialogue-dup", extra=QUALITY_ARGS),
    )
    assert code == 2
    assert any(item["code"] == "DUPLICATE_SPEAKER_VOICE" for item in payload["details"]["errors"])
    assert not (tmp_path / "out" / "native-dialogue-dup").exists()


# ── per-turn quality failures ─────────────────────────────────────────────────


def test_native_dialogue_quality_failure_is_durable_and_never_repeats_the_post(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """A FAIL keeps the paid raw/audio/cost, and resume/sync re-report it only."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Ожидаемая реплика.", "Guest: Вторая реплика."]])
    _install_quality(
        monkeypatch,
        {"turn_0001.mp3": "Совсем другой текст.", "turn_0002.mp3": "Вторая реплика."},
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-fail", extra=QUALITY_ARGS),
    )
    assert code == 60
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    # Only the first turn was submitted; the later turn is never paid for after a
    # recorded failure.
    assert provider.calls == [("turn_0001", "Kore")]

    run_uuid = _run_uuid("native-dialogue-fail")
    run_root = tmp_path / "out" / "native-dialogue-fail"
    assert (run_root / "chunks" / "turn_0001.mp3").read_bytes() == b"turn_0001-audio"
    assert (run_root / "raw" / "turn_0001.mp3").read_bytes() == b"turn_0001-audio"
    assert (run_root / "raw" / "turn_0001.mp3.receipt.json").is_file()
    # The observed FAIL is durable, linked to the first turn only.
    per_turn = _turn_quality(dialogue_env, run_uuid)
    assert [json.loads(row["metadata"])["passed"] for row in per_turn] == [False]

    def explode(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a recorded failure must not rerun the local model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-fail", extra=(*QUALITY_ARGS, "--resume")),
    )
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"

    assert provider.calls == [("turn_0001", "Kore")]
    assert len(_turn_quality(dialogue_env, run_uuid)) == 1


def test_native_dialogue_resume_after_first_turn_continues_only_the_rest(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """An interrupted run resumes the remaining turn with no repeated POST."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    # The first turn's local check fails transiently, so the run stops with the
    # first turn's paid chunk committed and the second turn never attempted.
    _install_quality(monkeypatch, {}, error=RuntimeError("qwen boom"))

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-cut", extra=QUALITY_ARGS),
    )
    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_ASR_FAILED"
    assert provider.calls == [("turn_0001", "Kore")]

    _install_quality(
        monkeypatch, {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."}
    )
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-cut", extra=(*QUALITY_ARGS, "--resume")),
    )
    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    # The first turn is not re-submitted; only the remaining turn is paid for.
    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]


def test_native_dialogue_crash_after_first_verified_turn_continues_only_the_rest(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """A crash between turns leaves no marker; resume pays only the missing turn."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    _install_quality(
        monkeypatch, {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."}
    )
    # Interrupt at the loop boundary before the second turn is even reserved, which
    # is exactly the crash window a real process kill between turns leaves behind.
    interrupted = {"done": False}
    original = native_generation._Executor._submit_fresh_part

    def maybe_interrupt(self, evidence):
        if evidence.part.chunk_id == "turn_0002" and not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt("simulated crash at the loop boundary")
        return original(self, evidence)

    monkeypatch.setattr(native_generation._Executor, "_submit_fresh_part", maybe_interrupt)
    monkeypatch.setattr(
        sys,
        "argv",
        _generate_argv(tmp_path, script, "native-dialogue-crash", extra=QUALITY_ARGS),
    )
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    assert provider.calls == [("turn_0001", "Kore")]

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, "native-dialogue-crash", extra=(*QUALITY_ARGS, "--resume")
        ),
    )
    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    # The already-verified first turn is not re-submitted nor re-checked.
    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]


def test_native_dialogue_sync_reports_pending_quality_without_running_a_model(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """Sync never runs the local model; a pending turn check stays incomplete."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Единственная реплика."]])
    _install_quality(monkeypatch, {}, error=RuntimeError("qwen boom"))
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-sync-pending", extra=QUALITY_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("native-dialogue-sync-pending")

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("history sync must not run the local model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_INCOMPLETE"
    assert provider.calls == [("turn_0001", "Kore")]
    assert _turn_quality(dialogue_env, run_uuid) == []


# ── unconfirmed paid submit protection ────────────────────────────────────────


def test_native_dialogue_unknown_sync_submit_blocks_retry_and_overwrite(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """An unconfirmed dialogue submit blocks resume, sync, and --overwrite."""
    provider = FakeDialogueProvider()
    provider.fail_on = "turn_0001"
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Неопределённый платный исход."]])
    quality_calls = _install_quality(monkeypatch, {"turn_0001.mp3": "Не должно выполниться."})

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-unconfirmed", extra=QUALITY_ARGS),
    )
    assert code == 30
    assert provider.calls == [("turn_0001", "Kore")]

    run_uuid = _run_uuid("native-dialogue-unconfirmed")
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    for argv in (
        _generate_argv(
            tmp_path, script, "native-dialogue-unconfirmed", extra=(*QUALITY_ARGS, "--resume")
        ),
        _history_argv(run_uuid, "resume"),
        _history_argv(run_uuid, "sync"),
    ):
        code, payload = _json_run(monkeypatch, capsys, argv)
        assert code == 30, payload
        assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "native-dialogue-unconfirmed",
            extra=(*QUALITY_ARGS, "--overwrite"),
        ),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_OVERWRITE_UNSUPPORTED"

    assert provider.calls == [("turn_0001", "Kore")]
    assert quality_calls == []


# ── resume identity ───────────────────────────────────────────────────────────


def test_native_dialogue_resume_rejects_changed_cast_and_text_before_provider(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """A changed cast or spoken text fails the committed identity before any key."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    _install_quality(
        monkeypatch, {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."}
    )
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-identity", extra=QUALITY_ARGS),
    )
    assert code == 0
    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    before = list(provider.calls)

    # A changed cast voice is rejected by the committed synthesis identity.
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "native-dialogue-identity",
            extra=(*QUALITY_ARGS, "--resume", "--speaker-voice", "Guest=Zephyr"),
        ),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"

    # A changed spoken text is rejected the same way.
    script.write_text(
        script.read_text(encoding="utf-8").replace("Вторая реплика.", "Иной текст."),
        encoding="utf-8",
    )
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, "native-dialogue-identity", extra=(*QUALITY_ARGS, "--resume")
        ),
    )
    assert code == 30, payload
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert provider.calls == before


# ── history resume / sync from the snapshot alone ─────────────────────────────


def test_native_dialogue_history_resume_after_script_deleted_reuses_committed_state(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """A completed dialogue run resumes from its snapshot with the YAML deleted."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    _install_quality(
        monkeypatch, {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."}
    )
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-hist", extra=QUALITY_ARGS),
    )
    assert code == 0
    run_uuid = _run_uuid("native-dialogue-hist")
    script.unlink()

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a completed run must not run the local model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    for verb in ("resume", "sync"):
        code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, verb))
        assert code == 0, payload
        assert payload["mode"] == verb
        assert payload["quality"] == {"complete": True, "passed": True}
        assert Path(payload["files"]["full_mp3"]).is_file()

    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]


def test_native_dialogue_legacy_root_stays_legacy(tmp_path, monkeypatch, dialogue_env):
    """An existing legacy dialogue root is never claimed by the native writer."""
    run_root = tmp_path / "out" / "legacy-dialogue"
    (run_root / "chunks").mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "completed", "run_id": "legacy-dialogue"}), encoding="utf-8"
    )
    (run_root / "chunks" / "turn_0001.mp3").write_bytes(b"legacy")
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "legacy"
    assert not (run_root / ".voiceover-native-history.json").exists()


# ── local model availability preflight ────────────────────────────────────────


def test_native_dialogue_local_quality_model_unavailable_blocks_before_payment(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """An unavailable local model stops the command before the first paid POST."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    quality_calls = _install_quality(monkeypatch, {"turn_0001.mp3": "не должно выполниться"})

    def blocked(_options):
        raise native_generation.NativeGenerationError(
            "the local Qwen ASR model is not installed.",
            code=10,
            error_code="NATIVE_QUALITY_MODEL_UNAVAILABLE",
        )

    monkeypatch.setattr(native_generation, "_preflight_local_quality", blocked)
    script = _script(tmp_path, [["Host: Платный POST не должен случиться."]])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-preflight", extra=QUALITY_ARGS),
    )
    assert code == 10
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_MODEL_UNAVAILABLE"
    assert provider.calls == []
    assert quality_calls == []
    assert not list((tmp_path / "out" / "native-dialogue-preflight" / "chunks").glob("*.mp3"))


# ── privacy and the CLI envelope ──────────────────────────────────────────────


def test_native_dialogue_private_verification_transcript_never_reaches_exports(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """The machine payload and exports carry only the content-free verdicts."""
    marker = "СЕКРЕТНАЯ_ПРОВЕРЕННАЯ_ФРАЗА"
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    text = f"{marker} первая реплика."
    script = _script(tmp_path, [[f"Host: {text}", "Guest: Вторая реплика."]])
    transcripts = {"turn_0001.mp3": text, "turn_0002.mp3": "Вторая реплика."}
    _install_quality(monkeypatch, transcripts)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-dialogue-privacy", extra=QUALITY_ARGS),
    )
    assert code == 0, payload
    serialized = json.dumps(payload, ensure_ascii=False)
    assert marker not in serialized
    assert "verification_transcript" not in serialized

    run_root = tmp_path / "out" / "native-dialogue-privacy"
    for name in ("run_state.json",):
        exported = (run_root / name).read_text(encoding="utf-8")
        assert marker not in exported
        assert "verification_transcript" not in exported
    for key in ("chunks_json", "run_json", "manifest_json"):
        exported = Path(payload["files"][key]).read_text(encoding="utf-8")
        assert marker not in exported
        assert "verification_transcript" not in exported


def test_native_dialogue_timing_failure_preserves_paid_turns_and_resume_finishes(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """A local timing failure after concat never re-submits a paid turn."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    concat = _install_concat(monkeypatch)
    _install_quality(
        monkeypatch,
        {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."},
    )
    timing_calls = _install_timing(monkeypatch, error=RuntimeError("timing boom"))
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])
    extra = (*QUALITY_ARGS, "--with-timings", "--timing-provider", "faster-whisper")

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "dialogue-timing", extra=extra),
    )
    assert code == 50, payload
    assert payload["details"]["error_code"] == "NATIVE_TIMING_FAILED"
    # Both turns were submitted exactly once and the final concat already ran.
    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]
    assert len(concat) == 1
    run_uuid = _run_uuid("dialogue-timing")
    assert _timing_children(dialogue_env, run_uuid) == []
    assert len(timing_calls) == 1

    # The post-audio timing failure must leave both paid turns as valid durable
    # evidence: a completed attempt each, linked to its own paid raw artifact, with
    # the synchronous OpenRouter cost still unknown (never an invented zero).
    run_root = tmp_path / "out" / "dialogue-timing"
    attempts = _paid_attempts(dialogue_env, run_uuid)
    assert [row["call_type"] for row in attempts] == ["tts_chunk", "tts_chunk"]
    assert all(row["status"] == "completed" for row in attempts)
    assert all(row["remote_id"] is None for row in attempts)
    assert all(row["cost"] is None for row in attempts)
    assert all(row["cost_exact_available"] == 0 for row in attempts)
    assert all(row["raw_part_uuid"] == row["part_uuid"] for row in attempts)
    assert all(row["raw_part_uuid"] is not None for row in attempts)
    paid_attempt_uuids = [row["attempt_uuid"] for row in attempts]
    for turn in ("turn_0001", "turn_0002"):
        assert (run_root / "raw" / f"{turn}.mp3").is_file()
        assert (run_root / "raw" / f"{turn}.mp3.receipt.json").is_file()

    # Resume finishes only the pending timing: no second POST and no re-concat.
    _install_timing(monkeypatch)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "dialogue-timing", extra=(*extra, "--resume")),
    )
    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    assert payload["quality"] == {"complete": True, "passed": True}
    assert payload["cost"]["total"] is None
    assert provider.calls == [("turn_0001", "Kore"), ("turn_0002", "Puck")]
    assert len(concat) == 1
    assert len(_timing_children(dialogue_env, run_uuid)) == 1

    # The resume rewrote no paid evidence: the same two attempts, still completed,
    # still linked to their paid raw bytes, and their cost is still unknown.
    resumed = _paid_attempts(dialogue_env, run_uuid)
    assert [row["attempt_uuid"] for row in resumed] == paid_attempt_uuids
    assert all(row["status"] == "completed" for row in resumed)
    assert all(row["cost"] is None for row in resumed)
    assert all(row["cost_exact_available"] == 0 for row in resumed)
    assert all(row["raw_part_uuid"] == row["part_uuid"] for row in resumed)
    for turn in ("turn_0001", "turn_0002"):
        assert (run_root / "raw" / f"{turn}.mp3.receipt.json").is_file()


def test_native_dialogue_no_trim_skips_the_trim_seam_and_is_recorded(
    tmp_path, monkeypatch, capsys, dialogue_env
):
    """``--no-trim`` is part of the dialogue route, not a reason to go legacy."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_concat(monkeypatch)
    _install_quality(
        monkeypatch,
        {"turn_0001.mp3": "Первая реплика.", "turn_0002.mp3": "Вторая реплика."},
    )
    trims: list[str] = []
    monkeypatch.setattr(cli, "trim_final_silence", lambda _f, _p, path: trims.append(path.name))
    script = _script(tmp_path, [["Host: Первая реплика.", "Guest: Вторая реплика."]])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "dialogue-notrim", extra=(*QUALITY_ARGS, "--no-trim")),
    )
    assert code == 0, payload
    assert trims == []
    run_uuid = _run_uuid("dialogue-notrim")
    snapshot = _rows(
        dialogue_env, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,)
    )[0]
    assert json.loads(snapshot["config_snapshot"])["output"]["trim_final_silence"] is False


def test_native_dialogue_route_gate_admits_only_the_validated_openrouter_route():
    """The gate admits both dialogue routes with their recorded local steps."""
    import argparse

    args = argparse.Namespace(
        provider="openrouter-tts",
        model=GEMINI_TTS_MODEL,
        tts_quality_provider="qwen-local",
        no_trim=False,
        with_timings=False,
    )
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    # The required local quality provider is not optional on this route.
    args.tts_quality_provider = None
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.tts_quality_provider = "xai-stt"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.tts_quality_provider = "qwen-local"
    # The recorded trim and an integrated local timing step are admitted; a cloud
    # timing provider keeps the legacy executor.
    args.no_trim = True
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.no_trim = False
    args.with_timings = True
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.timing_provider = "groq-whisper"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.timing_provider = "faster-whisper"
    args.with_timings = False
    # OmniVoice is admitted with its own voice bank and an optional installed local
    # quality provider; every Polza dialogue stays legacy.
    args.provider = "omnivoice-local"
    args.model = "audio-cpp/omnivoice-q8_0"
    args.mode = "preset"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.voice_bank_catalog = object()
    args.tts_quality_provider = None
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.tts_quality_provider = "qwen-local"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is True
    args.tts_quality_provider = "xai-stt"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
    args.tts_quality_provider = None
    args.provider = "polza-tts"
    assert cli._native_route_eligible(args, DIALOGUE_FORMAT) is False
