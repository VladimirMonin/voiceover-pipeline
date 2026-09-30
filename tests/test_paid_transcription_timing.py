"""Offline tests for the paid cloud-transcription boundary (S05 standalone).

The single functioning route admitted here is standalone ``timings`` with the
Groq Whisper or xAI STT provider. Every fixture is synthetic and lives under
``tmp_path``: a temporary private ``VOICEOVER_HOME``, fabricated audio bytes, and
a fake ``transcribe_timing_audio`` seam that mimics the real adapter by handing
its raw response body to ``on_raw_response`` before returning a parsed result. No
model is loaded, no network call is made, and no ``.env`` is read.

The tests are fail-first tripwires for the paid-safety boundary: a repeated or
concurrent invoke must never issue a second paid POST for one output root, a
read-only dispatch must not migrate or write, a crash window must be reconciled
from a validated receipt, and a changed source or unsafe destination must fail
closed before publishing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest

import voiceover_pipeline.cli as cli
import voiceover_pipeline.history.paid_transcription as paid
import voiceover_pipeline.services.native_generation as native_generation
from voiceover_pipeline.history.database import MIGRATIONS, Migration
from voiceover_pipeline.history.native_snapshot import NATIVE_SNAPSHOT_ORIGIN
from voiceover_pipeline.history.paths import history_database_path
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.groq_whisper import GroqWhisperProvider
from voiceover_pipeline.providers.xai_stt import XAISttProvider
from voiceover_pipeline.services import transcription
from voiceover_pipeline.settings import HistorySettings

AUDIO_BYTES = b"synthetic-audio-bytes"
TRANSCRIPT = "привет мир"


@pytest.fixture
def paid_home(tmp_path, monkeypatch):
    """Point ``VOICEOVER_HOME`` at a private temporary home."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    return home


def _audio(tmp_path: Path, name: str = "audio.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(AUDIO_BYTES)
    return path


def _groq_body(text: str = TRANSCRIPT) -> bytes:
    return json.dumps(
        {
            "text": text,
            "language": "ru",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": text}],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _xai_body(text: str = TRANSCRIPT) -> bytes:
    return json.dumps(
        {
            "text": text,
            "language": "ru",
            "words": [{"text": text, "start": 0.0, "end": 1.0}],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _timing(provider: str = "groq-whisper") -> TimingResult:
    return TimingResult(
        segments=[
            TimingSegment(
                id=0,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text=TRANSCRIPT,
            )
        ],
        model="whisper-large-v3-turbo" if provider == "groq-whisper" else "grok-stt",
        backend=provider,
        provider=provider,
        language="ru",
        timestamp_basis="provider_segment_timestamps",
    )


class _CloudTiming:
    """Fake cloud timing adapter that emits its raw body before returning."""

    def __init__(self, *, body: bytes, timing: TimingResult, calls: list[str]) -> None:
        self.body = body
        self.timing = timing
        self.calls = calls
        self.fail: Exception | None = None
        self.emit_raw = True

    def __call__(self, **kwargs):
        self.calls.append("post")
        on_raw = kwargs.get("on_raw_response")
        if self.fail is not None and not self.emit_raw:
            raise self.fail
        if on_raw is not None:
            on_raw(self.body, "application/json")
        if self.fail is not None:
            raise self.fail
        return self.timing


def _install_cli_seams(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _source: 1000)


def _fork_context():
    """Return the ``fork`` start method, skipping on a platform that lacks it.

    The cross-process barrier tests coordinate real forked writers, so a platform
    without ``fork`` (for example a spawn-only Windows host) skips them instead of
    weakening the POSIX two-process assertion with a thread surrogate.
    """
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - exercised only on a forkless platform
        pytest.skip("the cross-process barrier harness requires the fork start method")


def _table_names(home: Path) -> list[str]:
    """Return the sorted table names of the history database, or an empty list."""
    database_path = home / "history.sqlite3"
    if not database_path.exists():
        return []
    connection = sqlite3.connect(database_path)
    try:
        return sorted(
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        )
    finally:
        connection.close()


def _provision_empty_database(home: Path) -> Path:
    """Create a validated empty history database (zero tables, ``user_version`` 0)."""
    database_path = home / "history.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE _bootstrap (x INTEGER)")
        connection.execute("DROP TABLE _bootstrap")
        connection.commit()
    finally:
        connection.close()
    return database_path


def _timings_args(
    audio: Path,
    tmp_path: Path,
    *,
    provider: str = "groq-whisper",
    run_id="timing-run",
    output_dir=None,
    extra=(),
):
    return cli.build_parser().parse_args(
        [
            "timings",
            "--audio",
            str(audio),
            "--output-dir",
            str(output_dir if output_dir is not None else tmp_path / "out"),
            "--run-id",
            run_id,
            "--timing-provider",
            provider,
            "--json",
            *extra,
        ]
    )


def _run_timings(
    monkeypatch, audio: Path, tmp_path: Path, **kwargs
) -> tuple[int, cli.CliError | None]:
    _install_cli_seams(monkeypatch, tmp_path)
    args = _timings_args(audio, tmp_path, **kwargs)
    try:
        cli.run_timings(args)
    except SystemExit as exc:
        return int(exc.code or 0), None
    except cli.CliError as exc:
        return exc.code, exc
    raise AssertionError("run_timings did not exit")


def _run_history(monkeypatch, run_uuid: str, mode: str) -> tuple[int, str, cli.CliError | None]:
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _source: 1000)
    args = cli.build_parser().parse_args(["history", mode, run_uuid, "--json"])
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            cli.history_cmd(args)
        except SystemExit as exc:
            return int(exc.code or 0), buffer.getvalue(), None
        except cli.CliError as exc:
            return exc.code, buffer.getvalue(), exc
    raise AssertionError("history_cmd did not exit")


def _rows(home: Path, sql: str, params=()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return list(connection.execute(sql, params))
    finally:
        connection.close()


def _inventory(home: Path) -> list[str]:
    if not home.exists():
        return []
    return sorted(str(path.relative_to(home)) for path in home.rglob("*"))


def _output_root(tmp_path: Path, run_id: str = "timing-run") -> Path:
    return tmp_path / "out" / run_id


def _raw_body(home: Path) -> Path:
    artifact = next(
        row
        for row in _rows(home, "SELECT * FROM artifacts")
        if row["role"] == "provider_raw_response"
    )
    run = _rows(home, "SELECT run_uuid, run_root FROM runs")[0]
    return Path(run["run_root"]) / artifact["path"]


def _single_run_uuid(home: Path) -> str:
    return _rows(home, "SELECT run_uuid FROM runs")[0]["run_uuid"]


def _raw_saved_attempt_uuid(home: Path) -> str:
    return _rows(home, "SELECT attempt_uuid FROM attempts")[0]["attempt_uuid"]


def _seed_native_owner(root: Path) -> str:
    """Commit a native-history run bound to ``root`` and write its descriptor."""
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import HistoryRepository

    root.mkdir(parents=True, exist_ok=True)
    database = HistoryDatabase(history_database_path())
    database.connect()
    database.migrate()
    try:
        repository = HistoryRepository(database)
        run = repository.create_run(
            operation="tts",
            run_root=str(root.resolve()),
            status="completed",
            config_snapshot={"operation_origin": NATIVE_SNAPSHOT_ORIGIN},
        )
    finally:
        database.close()
    native_generation._write_descriptor(root, run.run_uuid)
    return run.run_uuid


def _run_generate(
    monkeypatch, tmp_path: Path, *, script: Path, output_dir: Path, run_id: str, extra=()
) -> tuple[int, cli.CliError | None]:
    _install_cli_seams(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "polza-chat-audio",
            "--script",
            str(script),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--json",
            *extra,
        ]
    )
    try:
        cli.generate(args)
    except SystemExit as exc:
        return int(exc.code or 0), None
    except cli.CliError as exc:
        return exc.code, exc
    raise AssertionError("generate did not exit")


# -- adapter raw-before-parse contract ----------------------------------------


class _FakeResponse:
    def __init__(self, *, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = content.decode("utf-8", "replace")
        self.parsed = False

    def json(self):
        self.parsed = True
        return json.loads(self.content)


def test_groq_adapter_hands_raw_body_before_parsing(monkeypatch, tmp_path):
    audio = _audio(tmp_path)
    response = _FakeResponse(content=_groq_body())
    monkeypatch.setattr(
        "voiceover_pipeline.providers.groq_whisper.requests.post",
        lambda *args, **kwargs: response,
    )
    seen: list[tuple[bool, bytes]] = []

    def sink(body: bytes, content_type: str) -> None:
        seen.append((response.parsed, body))

    provider = GroqWhisperProvider(model="whisper-large-v3-turbo", api_key="test")
    result = provider.transcribe(audio, language="ru", on_raw_response=sink)

    assert result.segments[0].text == TRANSCRIPT
    assert len(seen) == 1
    observed_before_parse, body = seen[0]
    assert observed_before_parse is False
    assert body == response.content


def test_groq_adapter_callback_failure_stops_before_parse(monkeypatch, tmp_path):
    audio = _audio(tmp_path)
    response = _FakeResponse(content=_groq_body())
    monkeypatch.setattr(
        "voiceover_pipeline.providers.groq_whisper.requests.post",
        lambda *args, **kwargs: response,
    )

    def sink(body: bytes, content_type: str) -> None:
        raise RuntimeError("storage failed")

    provider = GroqWhisperProvider(model="whisper-large-v3-turbo", api_key="test")
    with pytest.raises(RuntimeError, match="storage failed"):
        provider.transcribe(audio, language="ru", on_raw_response=sink)
    assert response.parsed is False


def test_xai_adapter_hands_raw_body_before_parsing(monkeypatch, tmp_path):
    audio = _audio(tmp_path)
    response = _FakeResponse(content=_xai_body())
    monkeypatch.setattr(
        "voiceover_pipeline.providers.xai_stt.requests.post",
        lambda *args, **kwargs: response,
    )
    seen: list[bool] = []

    def sink(raw: bytes, content_type: str) -> None:
        seen.append(response.parsed)

    provider = XAISttProvider(model="grok-stt", api_key="test")
    result = provider.transcribe(audio, language="ru", on_raw_response=sink)

    assert result.timestamp_basis == "derived_from_provider_words"
    assert seen == [False]


# -- standalone paid timing lifecycle -----------------------------------------


def test_standalone_cloud_timing_persists_marker_raw_and_transcript(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    fake = _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls)
    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["history"]["saved"] is True
    run_uuid = payload["history"]["run_uuid"]
    assert calls == ["post"]

    runs = _rows(paid_home, "SELECT * FROM runs")
    assert len(runs) == 1
    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["operation_origin"] == paid.PAID_TRANSCRIPTION_ORIGIN
    assert snapshot["provider"] == "groq-whisper"
    assert snapshot["request_options"]["timestamp_granularity"] == "segment"
    assert snapshot["source_audio_sha256"] == hashlib.sha256(AUDIO_BYTES).hexdigest()
    # Ownership is bound to the canonical output root, not the history UUID.
    assert runs[0]["run_root"] == str(_output_root(tmp_path))
    assert snapshot["output_root"] == str(_output_root(tmp_path))
    assert runs[0]["status"] == "completed"

    attempts = _rows(paid_home, "SELECT * FROM attempts")
    assert len(attempts) == 1
    assert attempts[0]["call_type"] == paid.ATTEMPT_CALL_TYPE_PAID_TIMING
    assert attempts[0]["status"] == "completed"
    # Cost unknown is a NULL amount, never an invented zero.
    assert attempts[0]["cost"] is None
    assert attempts[0]["cost_source"] == "unknown"
    assert attempts[0]["cost_exact_available"] == 0
    usage = json.loads(attempts[0]["usage_json"])
    assert usage["cost_known"] is False

    artifacts = _rows(paid_home, "SELECT * FROM artifacts")
    roles = sorted(row["role"] for row in artifacts)
    assert roles == ["provider_raw_response", "srt", "timings_json"]
    raw = next(row for row in artifacts if row["role"] == "provider_raw_response")
    assert raw["path"] == f"raw/{attempts[0]['attempt_uuid']}.body"
    assert raw["sha256"] == hashlib.sha256(_groq_body()).hexdigest()

    body_path = _output_root(tmp_path) / raw["path"]
    assert body_path.is_file()
    assert body_path.read_bytes() == _groq_body()
    assert run_uuid

    texts = _rows(paid_home, "SELECT kind, content FROM text_sources")
    assert [(row["kind"], row["content"]) for row in texts] == [("asr_transcript", TRANSCRIPT)]


def test_reservation_failure_makes_zero_requests(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)

    def fail_reserve(_request):
        raise paid.PaidTranscriptionError("boom", error_code="HISTORY_UNAVAILABLE")

    monkeypatch.setattr(paid, "reserve_paid_transcription", fail_reserve)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )

    code, error = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 50
    assert error is not None
    assert calls == []
    assert not (paid_home / "history.sqlite3").exists()


def test_first_post_failure_leaves_unconfirmed_and_resume_sync_do_not_retry(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    fake = _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls)
    fake.fail = RuntimeError("timed out")
    fake.emit_raw = False
    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)
    assert code == 40
    assert calls == ["post"]

    attempts = _rows(paid_home, "SELECT status FROM attempts")
    assert attempts[0]["status"] == "submitting"
    run_uuid = _single_run_uuid(paid_home)

    sync_code, sync_stdout, _sync_error = _run_history(monkeypatch, run_uuid, "sync")
    assert sync_code == 0
    sync_payload = json.loads(sync_stdout)
    assert sync_payload["complete"] is False
    assert sync_payload["status"] == "running"
    assert calls == ["post"]

    resume_code, _resume_stdout, resume_error = _run_history(monkeypatch, run_uuid, "resume")
    assert resume_code == 30
    assert resume_error is not None
    assert resume_error.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert calls == ["post"]


def test_repeat_invocations_and_overwrite_never_issue_second_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    fake = _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls)
    fake.fail = RuntimeError("timed out")
    fake.emit_raw = False
    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake)

    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    assert calls == ["post"]
    # A second plain invocation, a second one with --overwrite, one with changed
    # options, and one through a canonical path alias all refuse the owned root.
    for kwargs in (
        {},
        {"extra": ("--overwrite",)},
        {"extra": ("--model", "whisper-large-v3")},
        {"output_dir": str((tmp_path / "out" / ".").resolve())},
    ):
        code, error = _run_timings(monkeypatch, audio, tmp_path, **kwargs)
        assert code == 50, kwargs
        assert error is not None
        assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert calls == ["post"]
    assert len(_rows(paid_home, "SELECT * FROM runs")) == 1
    assert len(_rows(paid_home, "SELECT * FROM attempts")) == 1


def test_deliberately_different_run_id_is_a_separate_decision(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    assert _run_timings(monkeypatch, audio, tmp_path, run_id="other-run")[0] == 0
    assert calls == ["post", "post"]
    assert len(_rows(paid_home, "SELECT * FROM runs")) == 2


def test_completed_owner_refuses_fresh_overwrite(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    code, error = _run_timings(monkeypatch, audio, tmp_path, extra=("--overwrite",))
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert calls == ["post"]


def test_cloud_timing_requires_history_enabled_before_any_mutation(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    monkeypatch.setattr(
        cli.settings_module, "load_history_settings", lambda *a, **k: HistorySettings(enabled=False)
    )
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )

    code, error = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_HISTORY_REQUIRED"
    assert calls == []
    assert not _output_root(tmp_path).exists()


def test_disabled_history_overwrite_preserves_sentinel_and_makes_zero_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    existing = _output_root(tmp_path)
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep me")
    before = _inventory(paid_home)
    monkeypatch.setattr(
        cli.settings_module, "load_history_settings", lambda *a, **k: HistorySettings(enabled=False)
    )
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )

    code, error = _run_timings(monkeypatch, audio, tmp_path, extra=("--overwrite",))

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_HISTORY_REQUIRED"
    assert calls == []
    assert sentinel.read_text() == "keep me"
    assert _inventory(paid_home) == before


def test_raw_saved_after_artifact_failure_replays_without_second_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    fake = _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls)
    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake)

    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)
    assert code == 50
    assert calls == ["post"]
    attempts = _rows(paid_home, "SELECT status FROM attempts")
    assert attempts[0]["status"] == "raw_saved"
    run_uuid = _single_run_uuid(paid_home)

    resume_code, resume_stdout, _resume_error = _run_history(monkeypatch, run_uuid, "resume")
    assert resume_code == 0
    payload = json.loads(resume_stdout)
    assert payload["complete"] is True
    assert payload["status"] == "completed"
    # No second POST: only the local parse/replay ran.
    assert calls == ["post"]
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "completed"
    timings_json = Path(payload["files"]["timings_json"])
    assert timings_json.is_file()
    assert json.loads(timings_json.read_text(encoding="utf-8"))["segment_count"] == 1


def test_completed_run_sync_and_resume_are_read_only(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path) == (0, None)
    run_uuid = json.loads(capsys.readouterr().out)["history"]["run_uuid"]

    for mode in ("sync", "resume"):
        code, stdout, _error = _run_history(monkeypatch, run_uuid, mode)
        assert code == 0
        payload = json.loads(stdout)
        assert payload["complete"] is True
        assert payload["status"] == "completed"
    assert calls == ["post"]


# -- crash window and receipt reconciliation ----------------------------------


def _crash_after_body(monkeypatch):
    """Make the raw-phase database link fail so receipt+body survive unlinked."""
    original = paid._open_database
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 2:
            raise paid.PaidTranscriptionError("down", error_code="HISTORY_UNAVAILABLE")
        return original()

    monkeypatch.setattr(paid, "_open_database", flaky)
    return original


def test_crash_after_body_reconciles_same_attempt_without_second_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original = _crash_after_body(monkeypatch)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)
    assert code == 40
    assert calls == ["post"]
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "submitting"
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _rows(paid_home, "SELECT attempt_uuid FROM attempts")[0]["attempt_uuid"]
    raw_dir = _output_root(tmp_path) / "raw"
    assert (raw_dir / f"{attempt_uuid}.body").is_file()
    assert (raw_dir / f"{attempt_uuid}.body.receipt").is_file()
    assert _rows(paid_home, "SELECT * FROM artifacts") == []

    monkeypatch.setattr(paid, "_open_database", original)
    resume_code, resume_stdout, _resume_error = _run_history(monkeypatch, run_uuid, "resume")
    assert resume_code == 0
    assert calls == ["post"]
    payload = json.loads(resume_stdout)
    assert payload["complete"] is True
    assert payload["run_uuid"] == run_uuid
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "completed"
    assert _rows(paid_home, "SELECT * FROM attempts")[0]["attempt_uuid"] == attempt_uuid

    # A repeat resume is idempotent: still one attempt and one POST.
    again_code, _stdout, _error = _run_history(monkeypatch, run_uuid, "resume")
    assert again_code == 0
    assert calls == ["post"]
    assert len(_rows(paid_home, "SELECT * FROM attempts")) == 1


def test_resume_does_not_reconcile_missing_receipt(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original = _crash_after_body(monkeypatch)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _rows(paid_home, "SELECT attempt_uuid FROM attempts")[0]["attempt_uuid"]
    (_output_root(tmp_path) / "raw" / f"{attempt_uuid}.body.receipt").unlink()
    monkeypatch.setattr(paid, "_open_database", original)

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert calls == ["post"]
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "submitting"


def test_resume_does_not_reconcile_tampered_body(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original = _crash_after_body(monkeypatch)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _rows(paid_home, "SELECT attempt_uuid FROM attempts")[0]["attempt_uuid"]
    (_output_root(tmp_path) / "raw" / f"{attempt_uuid}.body").write_bytes(b'{"tampered":true}')
    monkeypatch.setattr(paid, "_open_database", original)

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert calls == ["post"]


def test_resume_does_not_reconcile_wrong_fingerprint(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original = _crash_after_body(monkeypatch)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _rows(paid_home, "SELECT attempt_uuid FROM attempts")[0]["attempt_uuid"]
    receipt = _output_root(tmp_path) / "raw" / f"{attempt_uuid}.body.receipt"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["operation_fingerprint"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(paid, "_open_database", original)

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert calls == ["post"]


def test_tampered_saved_body_blocks_replay_without_second_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 50
    run_uuid = _single_run_uuid(paid_home)
    _raw_body(paid_home).write_bytes(b'{"tampered":true}')

    resume_code, _stdout, resume_error = _run_history(monkeypatch, run_uuid, "resume")
    assert resume_code == 50
    assert resume_error is not None
    # The replay never issues a second paid POST and never claims a save.
    assert calls == ["post"]
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "raw_saved"


def test_oversized_body_is_refused_before_parse(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    monkeypatch.setattr(paid, "_MAX_RAW_BODY_BYTES", 4)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 40
    assert calls == ["post"]
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "submitting"
    assert _rows(paid_home, "SELECT * FROM artifacts") == []


# -- source and output identity -----------------------------------------------


def test_changed_same_size_source_refuses_replay(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 50
    run_uuid = _single_run_uuid(paid_home)
    audio.write_bytes(b"X" * len(AUDIO_BYTES))

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_SOURCE_CHANGED"
    assert calls == ["post"]


def test_missing_source_refuses_replay(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 50
    run_uuid = _single_run_uuid(paid_home)
    audio.unlink()

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_SOURCE_UNAVAILABLE"
    assert calls == ["post"]


def test_symlinked_artifact_leaf_is_not_followed(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 50
    run_uuid = _single_run_uuid(paid_home)
    attacker = tmp_path / "attacker.srt"
    attacker.write_text("untouched")
    (_output_root(tmp_path) / "timing-run.srt").symlink_to(attacker)

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_OUTPUT_UNSAFE"
    assert attacker.read_text() == "untouched"
    assert calls == ["post"]


def test_source_inside_output_is_not_deleted(paid_home, tmp_path, monkeypatch, capsys):
    output_root = _output_root(tmp_path)
    output_root.mkdir(parents=True)
    audio = output_root / "audio.wav"
    audio.write_bytes(AUDIO_BYTES)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )

    code, error = _run_timings(monkeypatch, audio, tmp_path, extra=("--overwrite",))

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_SOURCE_INSIDE_OUTPUT"
    assert audio.read_bytes() == AUDIO_BYTES
    assert calls == []


def test_atomic_write_artifact_rejects_unsafe_leaf(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for leaf in ("../escape.json", "a/b.json", ".", "..", ""):
        with pytest.raises(paid.PaidTranscriptionError) as excinfo:
            paid.atomic_write_artifact(root, leaf, b"x")
        assert excinfo.value.error_code == "PAID_OUTPUT_UNSAFE"


# -- read-only dispatch -------------------------------------------------------


def test_unknown_and_non_paid_uuid_dispatch_without_writing(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=[]),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    paid_run = _single_run_uuid(paid_home)

    # A non-paid run row dispatched onward, never treated as this handler's run.
    database_path = history_database_path()
    before = _inventory(paid_home)
    import sqlite3 as _sqlite3

    connection = _sqlite3.connect(database_path)
    connection.execute(
        "INSERT INTO runs (run_uuid, operation, status, run_root, legacy_source_root, "
        "user_label, parent_uuid, config_snapshot, record_version, revision, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "00000000-0000-4000-8000-0000000000aa",
            "asr",
            "completed",
            str(tmp_path / "legacy"),
            None,
            None,
            None,
            json.dumps({"operation_origin": "legacy_import"}),
            1,
            1,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()

    assert (
        cli._paid_transcription_history_command(
            cli.build_parser().parse_args(
                ["history", "sync", "00000000-0000-4000-8000-0000000000aa"]
            ),
            "sync",
        )
        is None
    )
    assert (
        cli._paid_transcription_history_command(
            cli.build_parser().parse_args(
                ["history", "sync", "00000000-0000-4000-8000-00000000dead"]
            ),
            "sync",
        )
        is None
    )
    # The paid run itself is recognized.
    state = cli._paid_transcription_history_command(
        cli.build_parser().parse_args(["history", "sync", paid_run]), "sync"
    )
    assert state is not None
    assert state["complete"] is True
    assert _inventory(paid_home) == before


def test_read_only_dispatch_creates_no_sidecar_or_migration(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=[]),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    run_uuid = _single_run_uuid(paid_home)

    before_inventory = _inventory(paid_home)
    before_schema = [
        tuple(row) for row in _rows(paid_home, "SELECT version, name FROM schema_migrations")
    ]

    state = paid.load_paid_transcription_state(run_uuid)
    assert state.run_uuid == run_uuid
    _run_history(monkeypatch, run_uuid, "sync")

    assert _inventory(paid_home) == before_inventory
    after_schema = [
        tuple(row) for row in _rows(paid_home, "SELECT version, name FROM schema_migrations")
    ]
    assert after_schema == before_schema


def test_absent_home_read_creates_nothing(tmp_path, monkeypatch):
    home = tmp_path / "absent-home"
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    with pytest.raises(paid.PaidTranscriptionError) as excinfo:
        paid.load_paid_transcription_state("00000000-0000-4000-8000-0000000000cc")
    assert excinfo.value.error_code == "PAID_TRANSCRIPTION_NOT_FOUND"
    assert not home.exists()


def test_pending_migration_is_not_applied_by_a_paid_read(paid_home, tmp_path, monkeypatch):
    from voiceover_pipeline.history.database import HistoryDatabase

    home = paid_home
    database_path = history_database_path()
    older: tuple[Migration, ...] = tuple(MIGRATIONS[:-1])
    database = HistoryDatabase(database_path, migrations=older)
    database.connect()
    database.migrate()
    database.close()
    before = _rows(paid_home, "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations")[0][
        "v"
    ]

    with pytest.raises(paid.PaidTranscriptionError):
        paid.load_paid_transcription_state("00000000-0000-4000-8000-0000000000bb")

    after = _rows(paid_home, "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations")[0]["v"]
    assert after == before
    assert not (home / "history.sqlite3-wal").exists()
    assert not (home / "history.sqlite3-shm").exists()


def test_active_committed_wal_is_read_consistently(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=[]),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    run_uuid = _single_run_uuid(paid_home)
    database_path = history_database_path()

    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("UPDATE runs SET user_label = 'wal-committed' WHERE run_uuid = ?", (run_uuid,))
    writer.commit()
    try:
        assert (paid_home / "history.sqlite3-wal").exists()
        state = paid.load_paid_transcription_state(run_uuid)
        assert state.run_uuid == run_uuid
        # The reader never deletes the live writer's sidecars.
        assert (paid_home / "history.sqlite3-wal").exists()
    finally:
        writer.close()


# -- concurrency --------------------------------------------------------------


def test_two_process_invocations_issue_at_most_one_post(paid_home, tmp_path, monkeypatch):
    """Two real processes contending for one output root issue exactly one POST.

    Each child runs the real ``cli.run_timings`` in its own process against the
    same canonical output root. A shared filesystem log and a ``multiprocessing``
    event prove that whoever wins the cross-process run lock reaches the fake
    transport once, while the other fails closed on the lock without a POST.
    """
    audio = _audio(tmp_path)
    _install_cli_seams(monkeypatch, tmp_path)
    post_log = tmp_path / "posts.log"
    release = multiprocessing.Event()
    started = multiprocessing.Event()
    results = multiprocessing.Queue()

    class _GatedTransport:
        def __call__(self, **kwargs):
            with open(post_log, "a", encoding="utf-8") as handle:
                handle.write("post\n")
                handle.flush()
                os.fsync(handle.fileno())
            started.set()
            assert release.wait(timeout=30), "gate was never released"
            on_raw = kwargs.get("on_raw_response")
            if on_raw is not None:
                on_raw(_groq_body(), "application/json")
            return _timing()

    monkeypatch.setattr(transcription, "transcribe_timing_audio", _GatedTransport())

    def worker():
        args = _timings_args(audio, tmp_path)
        try:
            cli.run_timings(args)
        except SystemExit as exc:
            results.put((int(exc.code or 0), None))
        except cli.CliError as exc:
            error_code = exc.details.get("error_code") if exc.details else None
            results.put((exc.code, error_code))
        else:
            results.put((0, None))

    context = _fork_context()
    processes = [context.Process(target=worker) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert started.wait(timeout=30), "no invocation reached the POST"
        # The losing process exits without a POST; wait for it before releasing
        # the winner so a released lock cannot admit a second POST.
        losing = results.get(timeout=30)
        release.set()
        winning = results.get(timeout=30)
    finally:
        release.set()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert sorted(code for code, _error_code in (losing, winning)) == [0, 30]
    assert losing[1] == "PAID_TIMING_RUN_LOCKED"
    assert post_log.read_text(encoding="utf-8").count("post") == 1
    assert len(_rows(paid_home, "SELECT * FROM runs")) == 1
    assert len(_rows(paid_home, "SELECT * FROM attempts")) == 1


# -- transaction atomicity ----------------------------------------------------


def test_reservation_rollback_leaves_no_partial_run(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_add_attempt = paid.HistoryRepository.add_attempt

    def boom(self, *args, **kwargs):
        raise sqlite3.Error("injected failure")

    monkeypatch.setattr(paid.HistoryRepository, "add_attempt", boom)

    code, _error = _run_timings(monkeypatch, audio, tmp_path)
    assert code == 50
    assert calls == []
    assert _rows(paid_home, "SELECT * FROM runs") == []
    assert not (_output_root(tmp_path) / paid.PAID_OWNERSHIP_FILE_NAME).exists()
    monkeypatch.setattr(paid.HistoryRepository, "add_attempt", original_add_attempt)


# -- private transcript -------------------------------------------------------


def test_private_transcript_is_absent_from_show_and_costs(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=[]),
    )
    assert _run_timings(monkeypatch, audio, tmp_path) == (0, None)
    run_uuid = json.loads(capsys.readouterr().out)["history"]["run_uuid"]

    from voiceover_pipeline.commands.history import costs_history, show_history

    shown = show_history(run_uuid)
    serialized = json.dumps(shown, ensure_ascii=False)
    assert TRANSCRIPT not in serialized
    assert str(_output_root(tmp_path)) == shown["run"]["run_root"]

    costs = costs_history()
    assert costs["completeness"] == "partial"
    assert costs["local_attempts_without_api_charge"] == 0

    _code, stdout, _error = _run_history(monkeypatch, run_uuid, "sync")
    assert TRANSCRIPT not in stdout


# -- cross-route root ownership -----------------------------------------------


def _leave_raw_saved(monkeypatch, audio: Path, tmp_path: Path) -> list[str]:
    """Run one paid timing whose artifact publish fails, leaving a ``raw_saved`` run."""
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original_write = cli._write_paid_timing_artifacts
    remaining = {"failures": 1}

    def flaky_write(*args, **kwargs):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(cli, "_write_paid_timing_artifacts", flaky_write)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 50
    return calls


def _insert_raw_artifact_row(
    run_uuid: str, attempt_uuid: str, *, path: str, sha256: str, size: int
) -> None:
    connection = sqlite3.connect(history_database_path())
    try:
        connection.execute(
            "INSERT INTO artifacts (artifact_uuid, run_uuid, part_uuid, attempt_uuid, role, "
            "path_kind, path, mime, size_bytes, sha256, media_metadata_json, availability, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                run_uuid,
                None,
                attempt_uuid,
                "provider_raw_response",
                "managed_relative",
                path,
                "application/json",
                size,
                sha256,
                None,
                "present",
                "2024-01-01T00:00:00Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_generate_overwrite_refuses_paid_owned_root(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    root = _output_root(tmp_path)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me")
    body = _raw_body(paid_home)
    body_bytes = body.read_bytes()

    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n")
    code, error = _run_generate(
        monkeypatch,
        tmp_path,
        script=script,
        output_dir=tmp_path / "out",
        run_id="timing-run",
        extra=("--overwrite", "--confirm-delete-paid-audio"),
    )

    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert sentinel.read_text() == "keep me"
    assert body.read_bytes() == body_bytes
    assert calls == ["post"]


def test_generate_refuses_db_only_paid_owner_after_descriptor_loss(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    root = _output_root(tmp_path)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me")
    body = _raw_body(paid_home)
    body_bytes = body.read_bytes()
    # Simulate a descriptor write that failed after the run committed: the root
    # carries no descriptor, but the committed database row still owns it.
    (root / paid.PAID_OWNERSHIP_FILE_NAME).unlink()

    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n")
    code, error = _run_generate(
        monkeypatch,
        tmp_path,
        script=script,
        output_dir=tmp_path / "out",
        run_id="timing-run",
        extra=("--overwrite", "--confirm-delete-paid-audio"),
    )

    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert sentinel.read_text() == "keep me"
    assert body.read_bytes() == body_bytes
    assert calls == ["post"]


def test_timings_overwrite_refuses_native_owned_root(paid_home, tmp_path, monkeypatch, capsys):
    root = tmp_path / "out" / "timing-run"
    root.mkdir(parents=True)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me")
    _seed_native_owner(root)

    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )

    code, error = _run_timings(monkeypatch, audio, tmp_path, extra=("--overwrite",))

    assert code == 50
    assert error is not None
    assert error.details["error_code"] in (
        "NATIVE_TIMING_OUTPUT_OWNED",
        "NATIVE_OWNERSHIP_RUN_MISSING",
        "NATIVE_OWNERSHIP_UNVERIFIABLE",
    )
    assert sentinel.read_text() == "keep me"
    assert calls == []


def test_paid_ownership_lookup_not_truncated_by_newer_runs(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    fake = _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls)
    fake.fail = RuntimeError("timed out")
    fake.emit_raw = False
    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    root = str(_output_root(tmp_path))
    # A failed post-commit descriptor write leaves the committed submitting row as
    # the only owner evidence, so a truncated root lookup would miss it.
    (_output_root(tmp_path) / paid.PAID_OWNERSHIP_FILE_NAME).unlink()

    connection = sqlite3.connect(history_database_path())
    try:
        for index in range(550):
            connection.execute(
                "INSERT INTO runs (run_uuid, operation, status, run_root, legacy_source_root, "
                "user_label, parent_uuid, config_snapshot, record_version, revision, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    "asr",
                    "completed",
                    root,
                    None,
                    None,
                    None,
                    json.dumps({"operation_origin": "local_asr"}),
                    1,
                    1,
                    f"2099-01-01T00:00:{index % 60:02d}Z",
                    "2099-01-01T00:00:00Z",
                ),
            )
        connection.commit()
    finally:
        connection.close()
    assert len(_rows(paid_home, "SELECT * FROM runs")) == 551

    # The exact origin lookup still resolves the older submitting paid owner.
    ownership = paid.resolve_paid_timing_ownership(root)
    assert ownership.owned is True

    code, error = _run_timings(monkeypatch, audio, tmp_path)
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert calls == ["post"]


def test_unusable_history_database_fails_closed_before_overwrite(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    root = _output_root(tmp_path)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me")
    # Remove every local marker so the committed database row is the only
    # evidence: an unreadable database must still fail closed rather than let the
    # overwrite delete the root.
    (root / paid.PAID_OWNERSHIP_FILE_NAME).unlink()
    shutil.rmtree(root / paid.RAW_RESPONSE_DIRECTORY_NAME)

    database_path = history_database_path()
    for suffix in ("-wal", "-shm"):
        sidecar = database_path.with_name(database_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    database_path.write_bytes(b"not a database at all")

    code, error = _run_timings(monkeypatch, audio, tmp_path, extra=("--overwrite",))

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "HISTORY_UNAVAILABLE"
    assert sentinel.read_text() == "keep me"
    assert calls == ["post"]


# -- read-only probe and one consistent snapshot ------------------------------


def test_read_only_probe_refuses_wal_without_usable_shm(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    _leave_raw_saved(monkeypatch, audio, tmp_path)
    run_uuid = _single_run_uuid(paid_home)
    database_path = history_database_path()
    shm = database_path.with_name(database_path.name + "-shm")
    wal = database_path.with_name(database_path.name + "-wal")
    if shm.exists():
        shm.unlink()
    if not wal.exists():
        wal.write_bytes(b"")

    before = _inventory(paid_home)
    with pytest.raises(paid.PaidTranscriptionError) as excinfo:
        paid.load_paid_transcription_state(run_uuid)
    assert excinfo.value.error_code == "HISTORY_UNAVAILABLE"
    assert _inventory(paid_home) == before
    assert not shm.exists()

    shm.write_bytes(b"\x00" * 16)
    with pytest.raises(paid.PaidTranscriptionError) as excinfo:
        paid.load_paid_transcription_state(run_uuid)
    assert excinfo.value.error_code == "HISTORY_UNAVAILABLE"
    assert shm.read_bytes() == b"\x00" * 16


def test_state_read_is_one_consistent_snapshot(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls = _leave_raw_saved(monkeypatch, audio, tmp_path)
    run_uuid = _single_run_uuid(paid_home)
    database_path = history_database_path()

    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("UPDATE runs SET user_label = 'wal-open' WHERE run_uuid = ?", (run_uuid,))
    writer.commit()
    assert (paid_home / "history.sqlite3-wal").exists()
    assert (paid_home / "history.sqlite3-shm").exists()

    original_get_artifacts = paid.HistoryRepository.get_artifacts

    def interleaved(self, arg_run_uuid):
        other = sqlite3.connect(database_path)
        try:
            other.execute("DELETE FROM artifacts WHERE run_uuid = ?", (arg_run_uuid,))
            other.commit()
        finally:
            other.close()
        return original_get_artifacts(self, arg_run_uuid)

    monkeypatch.setattr(paid.HistoryRepository, "get_artifacts", interleaved)
    try:
        state = paid.load_paid_transcription_state(run_uuid)
        # The single read transaction keeps the snapshot taken at the run read, so
        # the interleaved delete is not observed mid-read.
        assert state.raw_saved is True
        assert state.raw_missing is False
    finally:
        monkeypatch.setattr(paid.HistoryRepository, "get_artifacts", original_get_artifacts)
        writer.close()

    fresh = paid.load_paid_transcription_state(run_uuid)
    assert fresh.raw_missing is True
    assert calls == ["post"]


# -- exact raw artifact identity ----------------------------------------------


def test_replay_rejects_duplicate_identical_raw_artifact_row(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls = _leave_raw_saved(monkeypatch, audio, tmp_path)
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _raw_saved_attempt_uuid(paid_home)
    original = _rows(
        paid_home,
        "SELECT path, size_bytes, sha256 FROM artifacts WHERE role = 'provider_raw_response'",
    )[0]
    _insert_raw_artifact_row(
        run_uuid,
        attempt_uuid,
        path=original["path"],
        sha256=original["sha256"],
        size=original["size_bytes"],
    )

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TRANSCRIPTION_UNSUPPORTED"
    assert calls == ["post"]
    assert not (_output_root(tmp_path) / "timing-run.timings.json").exists()


def test_replay_rejects_conflicting_raw_artifact_row(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls = _leave_raw_saved(monkeypatch, audio, tmp_path)
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _raw_saved_attempt_uuid(paid_home)
    # A conflicting raw row whose own path/size/digest are self-consistent, so a
    # first-row selection would replay this body instead of the validated one.
    other = _output_root(tmp_path) / "raw" / "other.body"
    other.write_bytes(_groq_body("conflicting transcript"))
    _insert_raw_artifact_row(
        run_uuid,
        attempt_uuid,
        path="raw/other.body",
        sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
        size=other.stat().st_size,
    )

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TRANSCRIPTION_UNSUPPORTED"
    assert calls == ["post"]
    assert not (_output_root(tmp_path) / "timing-run.timings.json").exists()


def test_replay_rejects_missing_or_tampered_raw_metadata(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls = _leave_raw_saved(monkeypatch, audio, tmp_path)
    run_uuid = _single_run_uuid(paid_home)
    database_path = history_database_path()
    # A row with no mandatory size or digest is not a validated body.
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE artifacts SET size_bytes = NULL, sha256 = NULL "
            "WHERE role = 'provider_raw_response'"
        )
        connection.commit()
    finally:
        connection.close()

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TRANSCRIPTION_UNSUPPORTED"
    assert calls == ["post"]
    assert not (_output_root(tmp_path) / "timing-run.timings.json").exists()

    # A present but wrong digest is refused as well.
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE artifacts SET size_bytes = ?, sha256 = ? WHERE role = 'provider_raw_response'",
            (len(_groq_body()), "0" * 64),
        )
        connection.commit()
    finally:
        connection.close()

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TRANSCRIPTION_UNSUPPORTED"
    assert calls == ["post"]


def test_recovery_rejects_conflicting_raw_row_without_second_post(
    paid_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    original = _crash_after_body(monkeypatch)
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 40
    run_uuid = _single_run_uuid(paid_home)
    attempt_uuid = _raw_saved_attempt_uuid(paid_home)
    _insert_raw_artifact_row(
        run_uuid,
        attempt_uuid,
        path=f"raw/{attempt_uuid}.body",
        sha256="0" * 64,
        size=len(_groq_body()),
    )
    monkeypatch.setattr(paid, "_open_database", original)

    code, _stdout, error = _run_history(monkeypatch, run_uuid, "resume")

    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert calls == ["post"]
    assert not (_output_root(tmp_path) / "timing-run.timings.json").exists()
    assert _rows(paid_home, "SELECT status FROM attempts")[0]["status"] == "submitting"


# -- symlinked output leaf ----------------------------------------------------


def test_swapped_symlinked_output_leaf_is_refused(paid_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing(), calls=calls),
    )
    assert _run_timings(monkeypatch, audio, tmp_path)[0] == 0
    root = _output_root(tmp_path)
    moved = root.with_name("timing-run-real")
    root.rename(moved)
    target = tmp_path / "out" / "fresh-target"
    root.symlink_to(target, target_is_directory=True)

    code, error = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OWNERSHIP_UNVERIFIABLE"
    assert calls == ["post"]
    assert not target.exists()


# -- empty validated schema ownership recognizer ------------------------------


def test_empty_validated_database_is_unowned_and_probe_writes_nothing(paid_home, tmp_path):
    """A validated empty (v0, zero-table) database means no paid owner, read-only.

    The read-only ownership probe must treat the genuinely empty schema as unowned
    instead of misreading the missing ``runs`` table as an unusable database, and
    it must not create the migration schema or a WAL sidecar.
    """
    _provision_empty_database(paid_home)
    root = _output_root(tmp_path)

    ownership = paid.resolve_paid_timing_ownership(root)

    assert ownership.owned is False
    assert _table_names(paid_home) == []
    assert not (paid_home / "history.sqlite3-wal").exists()
    assert not (paid_home / "history.sqlite3-shm").exists()


def test_empty_validated_database_does_not_block_generate_or_local_timings(
    paid_home, tmp_path, monkeypatch, capsys
):
    """A fresh empty database must not fail-close the generate guard or local timings."""
    _provision_empty_database(paid_home)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n", encoding="utf-8")

    reached: list[str] = []

    def stub_native_route(*_args, **_kwargs):
        reached.append("native")
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_run_native_route", stub_native_route)
    _install_cli_seams(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "polza-chat-audio",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-run",
            "--json",
        ]
    )
    with pytest.raises(SystemExit):
        cli.generate(args)
    # The paid guard let the fresh command through to the native route, and its
    # read-only probe created no schema.
    assert reached == ["native"]
    assert _table_names(paid_home) == []

    audio = _audio(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing("faster-whisper"), calls=calls),
    )
    code, error = _run_timings(monkeypatch, audio, tmp_path, provider="faster-whisper")
    assert code == 0
    assert error is None
    assert calls == ["post"]


def test_corrupt_or_foreign_database_fails_closed_without_mutation(
    paid_home, tmp_path, monkeypatch, capsys
):
    """A corrupt database fails the generate guard and local timings before deletion."""
    (paid_home / "history.sqlite3").write_bytes(b"not a database at all")
    root = _output_root(tmp_path)
    root.mkdir(parents=True)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    reached: list[str] = []
    monkeypatch.setattr(cli, "_run_native_route", lambda *_a, **_k: reached.append("native"))
    _install_cli_seams(monkeypatch, tmp_path)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "polza-chat-audio",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-run",
            "--json",
        ]
    )
    with pytest.raises(cli.CliError) as excinfo:
        cli.generate(args)
    assert excinfo.value.code == 30
    assert excinfo.value.details["error_code"] == "HISTORY_UNAVAILABLE"
    assert reached == []

    calls: list[str] = []
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _CloudTiming(body=_groq_body(), timing=_timing("faster-whisper"), calls=calls),
    )
    audio = _audio(tmp_path)
    code, error = _run_timings(
        monkeypatch, audio, tmp_path, provider="faster-whisper", extra=("--overwrite",)
    )
    assert code == 50
    assert error is not None
    assert error.details["error_code"] == "HISTORY_UNAVAILABLE"
    assert sentinel.read_text() == "keep me"
    assert calls == []


# -- native admission vs a DB-only paid owner ---------------------------------


class _NoTtsProvider:
    """A provider that records any submit; a refused native run must never build one."""

    def __init__(self, log: Path) -> None:
        self.log = log

    def synthesize_chunk(self, text: str, chunk_id: str):
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write(f"submit {chunk_id}\n")
        raise AssertionError("a refused native run must not submit a TTS chunk")


def _install_native_seams(monkeypatch, tts_log: Path) -> None:
    """Replace every offline seam a fresh native generate touches before a submit."""
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _source: 1000)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda _ffmpeg, data, _fmt, path: path.write_bytes(data)
    )
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli, "concat_audio_files", lambda _ffmpeg, _paths, output: output.write_bytes(b"")
    )

    def fake_build_provider(*_args, **_kwargs):
        tts_log.write_text("built\n", encoding="utf-8")
        return _NoTtsProvider(tts_log)

    monkeypatch.setattr(cli, "build_provider", fake_build_provider)


def test_native_generate_refuses_paid_db_row_committed_after_early_check(
    paid_home, tmp_path, monkeypatch
):
    """A DB-only paid owner committed after the early check is refused under the lock.

    The native child passes its early paid-root check, then pauses just before its
    run lock. The parent then commits a paid reservation row and fails its
    descriptor write (a real post-commit descriptor failure), so the root carries a
    committed paid owner but no local marker. The native child resumes its locked
    admission and must refuse with zero TTS submits and the paid evidence intact.
    """
    audio = _audio(tmp_path)
    tts_log = tmp_path / "tts_posts.log"
    _install_native_seams(monkeypatch, tts_log)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n", encoding="utf-8")
    argv = [
        "generate",
        "--provider",
        "polza-chat-audio",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "timing-run",
        "--json",
    ]

    native_ready = multiprocessing.Event()
    paid_done = multiprocessing.Event()
    results = multiprocessing.Queue()

    def native_worker():
        real_lock = native_generation.acquire_run_lock

        @contextlib.contextmanager
        def gated_lock(run_root, *, home=None):
            native_ready.set()
            if not paid_done.wait(timeout=30):
                raise RuntimeError("paid reservation never completed")
            with real_lock(run_root, home=home):
                yield

        native_generation.acquire_run_lock = gated_lock
        try:
            cli.generate(cli.build_parser().parse_args(argv))
        except SystemExit as exc:
            results.put((int(exc.code or 0), None))
        except cli.CliError as exc:
            error_code = exc.details.get("error_code") if exc.details else None
            results.put((exc.code, error_code))
        else:
            results.put((0, None))

    context = _fork_context()
    process = context.Process(target=native_worker)
    process.start()
    original_descriptor = paid._write_ownership_descriptor
    try:
        assert native_ready.wait(timeout=30), "the native child never reached its run lock"
        root = _output_root(tmp_path)
        root.mkdir(parents=True, exist_ok=True)

        def fail_descriptor(*_args, **_kwargs):
            raise paid.PaidTranscriptionError(
                "descriptor write failed", error_code="PAID_TIMING_OWNERSHIP_UNVERIFIABLE"
            )

        monkeypatch.setattr(paid, "_write_ownership_descriptor", fail_descriptor)
        with pytest.raises(paid.PaidTranscriptionError):
            paid.reserve_paid_transcription(
                paid.PaidTranscriptionRequest(
                    call_type=paid.ATTEMPT_CALL_TYPE_PAID_TIMING,
                    provider="groq-whisper",
                    model="whisper-large-v3-turbo",
                    language="ru",
                    word_timestamps=False,
                    source_audio=audio,
                    output_root=root,
                    request_options={"response_format": "verbose_json"},
                )
            )
        monkeypatch.setattr(paid, "_write_ownership_descriptor", original_descriptor)
        # The committed row is the only owner evidence: no local descriptor exists.
        assert not (root / paid.PAID_OWNERSHIP_FILE_NAME).exists()
        assert len(_rows(paid_home, "SELECT * FROM runs")) == 1
        paid_done.set()
        code, error_code = results.get(timeout=30)
    finally:
        paid_done.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        monkeypatch.setattr(paid, "_write_ownership_descriptor", original_descriptor)

    assert code == 30
    assert error_code == "PAID_TIMING_OUTPUT_OWNED"
    # Zero TTS submits: the provider was neither built nor invoked.
    assert not tts_log.exists()
    # The committed paid evidence is intact and its output root is untouched.
    assert [row["status"] for row in _rows(paid_home, "SELECT status FROM attempts")] == [
        "submitting"
    ]
    assert root.exists()


def test_native_generate_refuses_db_only_paid_owner_before_creating_root(
    paid_home, tmp_path, monkeypatch
):
    """A raced DB-only paid owner is refused before the native logger creates the root.

    A paid reservation whose post-commit descriptor write failed binds an output
    root that does not exist yet (its descriptor temp file cannot be created). The
    early generate guard can run before that commit is visible, so the locked
    native admission must re-read the committed paid owner before the generation
    logger -- whose constructor creates the run directory -- the chunks, or any
    provider. The early guard is disabled here to reproduce exactly that race.
    """
    audio = _audio(tmp_path)
    tts_log = tmp_path / "tts_posts.log"
    _install_native_seams(monkeypatch, tts_log)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    run_root = output_dir / "timing-run"
    assert not output_dir.exists()

    with pytest.raises(paid.PaidTranscriptionError):
        paid.reserve_paid_transcription(
            paid.PaidTranscriptionRequest(
                call_type=paid.ATTEMPT_CALL_TYPE_PAID_TIMING,
                provider="groq-whisper",
                model="whisper-large-v3-turbo",
                language="ru",
                word_timestamps=False,
                source_audio=audio,
                output_root=run_root,
                request_options={"response_format": "verbose_json"},
            )
        )
    # The row committed; the descriptor write failed because the root is absent.
    assert len(_rows(paid_home, "SELECT run_uuid FROM runs")) == 1
    assert not run_root.exists()

    # Reproduce the race: the early paid guard already saw this root as unowned.
    monkeypatch.setattr(cli, "_reject_paid_owned_root_for_generate", lambda _paths: None)

    constructed: list[Path] = []
    real_logger = native_generation.GenerationLogger

    def recording_logger(path):
        constructed.append(path)
        return real_logger(path)

    monkeypatch.setattr(native_generation, "GenerationLogger", recording_logger)

    code, error = _run_generate(
        monkeypatch, tmp_path, script=script, output_dir=output_dir, run_id="timing-run"
    )

    assert code == 30
    assert error is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    # No native writer touched the paid root: no logger, no run directory, no log
    # file, no chunks, and no provider.
    assert constructed == []
    assert not run_root.exists()
    assert not output_dir.exists()
    assert not (run_root / native_generation.LOG_FILE).exists()
    assert not tts_log.exists()


# -- local timings vs a concurrent paid reservation ---------------------------


def _post_logging_transport(post_log: Path, timing: TimingResult):
    """A fake transport that appends one shared-log line per real POST."""

    def transport(**kwargs):
        with open(post_log, "a", encoding="utf-8") as handle:
            handle.write("post\n")
            handle.flush()
            os.fsync(handle.fileno())
        on_raw = kwargs.get("on_raw_response")
        if on_raw is not None:
            on_raw(_groq_body(), "application/json")
        return timing

    return transport


def test_local_overwrite_cannot_delete_a_concurrent_paid_reservation(
    paid_home, tmp_path, monkeypatch
):
    """A local ``--overwrite`` paused after its early check must not delete paid output.

    The local child holds the same canonical-root run lock. It pauses just before
    acquiring it, the paid route reserves and POSTs once on that root, and the
    resumed local route re-resolves ownership under the lock and refuses instead of
    deleting the paid evidence. Exactly one paid POST happens.
    """
    audio = _audio(tmp_path)
    post_log = tmp_path / "posts.log"
    monkeypatch.setattr(
        transcription,
        "transcribe_timing_audio",
        _post_logging_transport(post_log, _timing()),
    )
    _install_cli_seams(monkeypatch, tmp_path)

    local_ready = multiprocessing.Event()
    paid_done = multiprocessing.Event()
    results = multiprocessing.Queue()

    def local_worker():
        real_lock = cli.acquire_run_lock

        @contextlib.contextmanager
        def gated_lock(run_root, *, home=None):
            local_ready.set()
            if not paid_done.wait(timeout=30):
                raise RuntimeError("paid reservation never completed")
            with real_lock(run_root, home=home):
                yield

        cli.acquire_run_lock = gated_lock
        args = _timings_args(audio, tmp_path, provider="faster-whisper", extra=("--overwrite",))
        try:
            cli.run_timings(args)
        except SystemExit as exc:
            results.put((int(exc.code or 0), None))
        except cli.CliError as exc:
            error_code = exc.details.get("error_code") if exc.details else None
            results.put((exc.code, error_code))
        else:
            results.put((0, None))

    context = _fork_context()
    process = context.Process(target=local_worker)
    process.start()
    try:
        assert local_ready.wait(timeout=30), "the local child never reached its run lock"
        paid_code, paid_error = _run_timings(monkeypatch, audio, tmp_path)
        assert paid_code == 0
        assert paid_error is None
        paid_done.set()
        code, error_code = results.get(timeout=30)
    finally:
        paid_done.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    # One paid POST total: the local child never reached its extraction.
    assert post_log.read_text(encoding="utf-8").count("post") == 1
    assert code == 50
    assert error_code == "PAID_TIMING_OUTPUT_OWNED"
    root = _output_root(tmp_path)
    assert (root / paid.PAID_OWNERSHIP_FILE_NAME).exists()
    assert _raw_body(paid_home).is_file()


# -- one root, two committed owners -------------------------------------------


def _root_inventory(root: Path) -> list[str]:
    """Return every path under ``root`` as a sorted relative name."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _native_tts_run(root: Path, script: Path) -> str:
    """Commit one valid native TTS run at ``root`` and write its ownership descriptor."""
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.native_snapshot import persist_prepared_tts_snapshot
    from voiceover_pipeline.history.repository import HistoryRepository
    from voiceover_pipeline.models import ScriptChunk
    from voiceover_pipeline.services.prepare import PreparedPart, PreparedRun

    root.mkdir(parents=True, exist_ok=True)
    chunk = ScriptChunk(number=1, id="chunk_01", text="первый абзац", speaker="host", voice="alloy")
    prepared = PreparedRun(
        provider="polza-chat-audio",
        model="openai/gpt-audio-mini",
        voice="alloy",
        style_prompt="Read warmly.",
        prompt_mode="plain",
        parts=(PreparedPart(chunk=chunk, voice=chunk.voice),),
        fallback_voice="alloy",
    )
    database = HistoryDatabase(history_database_path())
    database.connect()
    database.migrate()
    try:
        snapshot = persist_prepared_tts_snapshot(
            HistoryRepository(database),
            prepared=prepared,
            run_root=root,
            user_label="native-run",
            script_format="markdown",
            script_text=chunk.text,
            script_path=script,
        )
    finally:
        database.close()
    native_generation._write_descriptor(root, snapshot.run.run_uuid)
    return snapshot.run.run_uuid


def _commit_paid_run_at_native_root(root: Path, *, descriptor: bool, raw: bool) -> tuple[str, str]:
    """Write one committed paid timing run over ``root`` the way a reservation does.

    ``reserve_paid_transcription`` refuses a native-owned root, so this writes the
    conflicting state directly to reproduce a database that already carries both
    owners (committed before that guard, or by another writer). ``descriptor`` adds
    the run-local paid ownership descriptor, and ``raw`` adds the saved response
    receipt and body under ``raw/`` and leaves the attempt ``raw_saved`` instead of
    the ``submitting`` marker a fresh reservation commits. No POST happens here.
    """
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import Cost, HistoryRepository

    root = root.resolve()
    database = HistoryDatabase(history_database_path())
    database.connect()
    database.migrate()
    try:
        repository = HistoryRepository(database)
        with repository.transaction():
            run = repository.create_run(
                operation="timings",
                run_root=str(root),
                config_snapshot={
                    "operation_origin": paid.PAID_TRANSCRIPTION_ORIGIN,
                    "output_root": str(root),
                    "provider": "groq-whisper",
                    "model": "whisper-large-v3-turbo",
                    "language": "ru",
                    "word_timestamps_requested": False,
                    "cost_known": False,
                },
            )
            attempt = repository.add_attempt(
                run.run_uuid,
                call_type=paid.ATTEMPT_CALL_TYPE_PAID_TIMING,
                provider="groq-whisper",
                model="whisper-large-v3-turbo",
                status=(paid.ATTEMPT_STATUS_RAW_SAVED if raw else paid.ATTEMPT_STATUS_SUBMITTING),
                cost=Cost.unknown(),
            )
    finally:
        database.close()
    if descriptor:
        paid._write_ownership_descriptor(root, run.run_uuid)
    if raw:
        body = _groq_body()
        relative_path = paid._raw_relative_path(attempt.attempt_uuid)
        receipt = paid._receipt_payload(
            run_uuid=run.run_uuid,
            attempt_uuid=attempt.attempt_uuid,
            operation_fingerprint=hashlib.sha256(b"offline").hexdigest(),
            relative_path=relative_path,
            size=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            content_type="application/json",
        )
        receipt_path = paid._receipt_path(root, attempt.attempt_uuid)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        paid._raw_body_path(root, attempt.attempt_uuid).write_bytes(body)
    return run.run_uuid, attempt.attempt_uuid


def _recording_logger(constructed: list[Path]):
    """Return a ``GenerationLogger`` stand-in that records its construction."""
    real_logger = native_generation.GenerationLogger

    def logger(path):
        constructed.append(path)
        return real_logger(path)

    return logger


def test_paid_reserve_refuses_native_owned_root(paid_home, tmp_path):
    """A paid reservation must not commit a second owner over a native TTS run.

    The paid timings route binds the same canonical output root the native TTS route
    owns, and no reader reconciles one directory holding both. A direct reservation
    against a native-owned root therefore refuses before the paid run row, its
    attempt marker, or the ownership descriptor exist, and leaves the native root
    exactly as it was.
    """
    root = _output_root(tmp_path)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n", encoding="utf-8")
    native_uuid = _native_tts_run(root, script)
    audio = _audio(tmp_path)
    runs_before = [row["run_uuid"] for row in _rows(paid_home, "SELECT run_uuid FROM runs")]
    inventory_before = _root_inventory(root)

    with pytest.raises(paid.PaidTranscriptionError) as excinfo:
        paid.reserve_paid_transcription(
            paid.PaidTranscriptionRequest(
                call_type=paid.ATTEMPT_CALL_TYPE_PAID_TIMING,
                provider="groq-whisper",
                model="whisper-large-v3-turbo",
                language="ru",
                word_timestamps=False,
                source_audio=audio,
                output_root=root,
                request_options={"response_format": "verbose_json"},
            )
        )

    assert excinfo.value.error_code == "NATIVE_TIMING_OUTPUT_OWNED"
    # Only the seeded native run exists: no paid run, no attempt, no descriptor.
    assert [row["run_uuid"] for row in _rows(paid_home, "SELECT run_uuid FROM runs")] == (
        runs_before
    )
    assert runs_before == [native_uuid]
    assert _rows(paid_home, "SELECT attempt_uuid FROM attempts") == []
    assert not (root / paid.PAID_OWNERSHIP_FILE_NAME).exists()
    assert _root_inventory(root) == inventory_before


@pytest.mark.parametrize(
    "paid_state",
    [
        pytest.param("descriptor", id="committed-descriptor"),
        pytest.param("db_only", id="db-only-descriptor-failed"),
        pytest.param("raw_evidence", id="raw-saved-response"),
    ],
)
def test_native_history_resume_refuses_conflicting_paid_owner(
    paid_home, tmp_path, monkeypatch, paid_state
):
    """``history resume`` refuses a native run whose root also carries a paid owner.

    The conflicting database is written directly, because the reservation path now
    refuses a native-owned root (see ``test_paid_reserve_refuses_native_owned_root``)
    and the native snapshot writer refuses a root any run already owns. A root bound
    to both owners must fail closed with the fixed paid-ownership error before the
    generation logger, the provider, or any file in the root is touched, and the
    paid attempt and its saved raw response must stay intact.
    """
    root = _output_root(tmp_path)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n", encoding="utf-8")
    native_uuid = _native_tts_run(root, script)
    paid_uuid, attempt_uuid = _commit_paid_run_at_native_root(
        root, descriptor=paid_state != "db_only", raw=paid_state == "raw_evidence"
    )
    raw_body = paid._raw_body_path(root, attempt_uuid)
    inventory_before = _root_inventory(root)
    body_before = raw_body.read_bytes() if paid_state == "raw_evidence" else None

    constructed: list[Path] = []
    monkeypatch.setattr(native_generation, "GenerationLogger", _recording_logger(constructed))
    builds: list[str] = []

    def fake_provider_builder(_prepared):
        builds.append("built")
        raise AssertionError("a refused native resume must not build a provider")

    monkeypatch.setattr(cli, "_native_history_provider_builder", fake_provider_builder)

    code, _stdout, error = _run_history(monkeypatch, native_uuid, "resume")

    assert code == 30
    assert error is not None
    assert error.details is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    # Nothing native ran: no logger, no provider, and no file in the root changed.
    assert constructed == []
    assert builds == []
    assert _root_inventory(root) == inventory_before
    # The paid owner's committed attempt and raw response are preserved verbatim.
    assert [
        (row["run_uuid"], row["status"])
        for row in _rows(paid_home, "SELECT run_uuid, status FROM attempts")
    ] == [
        (
            paid_uuid,
            "raw_saved" if paid_state == "raw_evidence" else "submitting",
        )
    ]
    if body_before is not None:
        assert raw_body.read_bytes() == body_before
    assert (root / paid.PAID_OWNERSHIP_FILE_NAME).exists() is (paid_state != "db_only")


def test_native_generate_resume_refuses_conflicting_paid_owner(paid_home, tmp_path, monkeypatch):
    """``generate --resume`` refuses a native-owned root that also holds a paid owner.

    The early generate paid-root guard is disabled to reproduce the race the locked
    native admission must still catch. The root carries a committed native run and a
    committed DB-only paid run whose descriptor write failed, so the paid ownership
    is invisible locally: the locked admission must refuse before the generation
    logger, the chunks directory, or any provider.
    """
    root = _output_root(tmp_path)
    tts_log = tmp_path / "tts_posts.log"
    _install_native_seams(monkeypatch, tts_log)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n******\nвторой абзац\n", encoding="utf-8")
    native_uuid = _seed_native_owner(root)
    paid_uuid, _attempt_uuid = _commit_paid_run_at_native_root(root, descriptor=False, raw=False)
    assert not (root / paid.PAID_OWNERSHIP_FILE_NAME).exists()
    inventory_before = _root_inventory(root)

    # Reproduce the race: the early paid guard already saw this root as unowned.
    monkeypatch.setattr(cli, "_reject_paid_owned_root_for_generate", lambda _paths: None)
    constructed: list[Path] = []
    monkeypatch.setattr(native_generation, "GenerationLogger", _recording_logger(constructed))

    code, error = _run_generate(
        monkeypatch,
        tmp_path,
        script=script,
        output_dir=tmp_path / "out",
        run_id="timing-run",
        extra=("--resume",),
    )

    assert code == 30
    assert error is not None
    assert error.details is not None
    assert error.details["error_code"] == "PAID_TIMING_OUTPUT_OWNED"
    assert constructed == []
    assert not tts_log.exists()
    assert _root_inventory(root) == inventory_before
    assert [
        (row["run_uuid"], row["status"])
        for row in _rows(paid_home, "SELECT run_uuid, status FROM attempts")
    ] == [(paid_uuid, "submitting")]
    assert native_generation._read_descriptor(root) == native_uuid


class _AdmissionReached(Exception):
    """Sentinel from the spy logger: the locked native admission passed its guards."""


def test_clean_native_history_resume_reaches_the_logger(paid_home, tmp_path, monkeypatch):
    """A native run with no paid owner still resumes: the guard only refuses conflicts.

    The generation logger is the first step after the locked ownership guard and the
    step that creates the run directory, so stopping it there proves the paid-first
    guard leaves a clean native resume on its normal path and wrote nothing itself.
    """
    root = _output_root(tmp_path)
    script = tmp_path / "script.md"
    script.write_text("первый абзац\n", encoding="utf-8")
    native_uuid = _native_tts_run(root, script)
    inventory_before = _root_inventory(root)

    reached: list[Path] = []

    def stop_logger(path):
        reached.append(path)
        raise _AdmissionReached

    monkeypatch.setattr(native_generation, "GenerationLogger", stop_logger)
    _install_cli_seams(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(["history", "resume", native_uuid, "--json"])
    with pytest.raises(_AdmissionReached):
        cli.history_cmd(args)

    assert reached == [root / native_generation.LOG_FILE]
    assert _root_inventory(root) == inventory_before
