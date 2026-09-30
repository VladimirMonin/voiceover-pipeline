"""Offline tests for the local ASR / timings / verify canonical-history vertical.

Every fixture is synthetic and lives under ``tmp_path``: a temporary private
``VOICEOVER_HOME``, fabricated audio bytes, and a fake ASR provider factory. No
model is loaded, no network call is made, and no ``.env`` is read. The tests
assert what the commands actually store in the canonical SQLite history and that
the public machine output stays content-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.commands.history import costs_history, list_history, show_history
from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
    TimingResult,
    TimingSegment,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider
from voiceover_pipeline.services import transcription

AUDIO_BYTES = b"synthetic-audio-bytes"
TRANSCRIPT = "привет мир"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"


@pytest.fixture
def asr_home(tmp_path, monkeypatch):
    """Point ``VOICEOVER_HOME`` at a private temporary home."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    return home


def _audio(tmp_path: Path, name: str = "audio.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(AUDIO_BYTES)
    return path


def _result(
    *, transcript: str = TRANSCRIPT, timed: bool = True, words: bool = False, duration_s=2.0
) -> ASRResult:
    """Build one fixture result: text-only, span-timed, or word-timed.

    The word spans are only used by the word-timestamp timing route: the plain
    ``transcribe`` and ``verify-tts`` routes request ``timestamp_mode="none"``,
    and a result that returned words there would be rejected by the model layer.
    """
    execution = ASRExecutionReceipt(
        runtime="fixture-runtime",
        runtime_version="1.0",
        resolved_device="cpu",
        resolved_compute="float32",
        measurements={"wall_s": 0.25},
    )
    if not timed:
        return ASRResult(
            transcript=transcript,
            provider_id="qwen-local",
            model_id=MODEL_ID,
            language="ru",
            execution=execution,
        )
    return ASRResult(
        transcript=transcript,
        provider_id="qwen-local",
        model_id=MODEL_ID,
        language="ru",
        duration_s=duration_s,
        segments=(ASRSegment(text=transcript, start_s=0.0, end_s=1.0),),
        words=(
            (
                ASRWordSpan(text="привет", start_s=0.0, end_s=0.5),
                ASRWordSpan(text="мир", start_s=0.5, end_s=1.0),
            )
            if words
            else ()
        ),
        alignment_origin="native",
        execution=execution,
    )


class _FixtureProvider(ASRProvider):
    provider_id = "qwen-local"

    def __init__(self, result: ASRResult, calls: list[str] | None = None) -> None:
        self._result = result
        self._calls = calls

    def transcribe(self, request: ASRRequest) -> ASRResult:
        if self._calls is not None:
            self._calls.append("transcribe")
        return self._result


class _DeletingProvider(_FixtureProvider):
    """Delete the source audio during the provider call to prove honest absence."""

    def transcribe(self, request: ASRRequest) -> ASRResult:
        Path(request.audio_path).unlink()
        return super().transcribe(request)


def _spec(factory, *, timed: bool = True) -> ASRProviderSpec:
    return ASRProviderSpec(
        provider_id="qwen-local",
        description="Offline fixture provider",
        factory=factory,
        models=({"id": MODEL_ID, "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            segment_timestamps=timed,
            word_timestamps=timed,
            device_modes=("cpu",),
            compute_modes=("auto", "float32"),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )


def _install_spec(monkeypatch, result: ASRResult, factory=None, *, timed: bool = True):
    """Install the fake provider spec and skip the real long-form orchestrator."""
    monkeypatch.setattr(
        cli,
        "get_asr_provider_spec",
        lambda _provider_id: _spec(
            factory if factory is not None else (lambda: _FixtureProvider(result)),
            timed=timed,
        ),
    )
    # ``qwen-local`` declares long-form orchestration, which would require a real
    # FFmpeg/FFprobe pass over the fabricated bytes; the persistence contract under
    # test is independent of it, so the provider call is used directly.
    monkeypatch.setattr(
        cli,
        "transcribe_prerecorded_long_form",
        lambda provider, request: provider.transcribe(request),
    )


def _run_transcribe(monkeypatch, audio: Path, *, extra=()) -> int:
    args = cli.build_parser().parse_args(
        [
            "transcribe",
            "--audio",
            str(audio),
            "--provider",
            "qwen-local",
            "--model",
            MODEL_ID,
            "--language",
            "ru",
            "--device",
            "cpu",
            "--compute",
            "auto",
            *extra,
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)
    return exit_info.value.code


def _run_verify(monkeypatch, audio: Path, expected: str, *, extra=()) -> int:
    args = cli.build_parser().parse_args(
        [
            "verify-tts",
            "--audio",
            str(audio),
            "--expected-text",
            expected,
            "--provider",
            "qwen-local",
            "--model",
            MODEL_ID,
            "--language",
            "ru",
            "--device",
            "cpu",
            "--compute",
            "auto",
            *extra,
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.verify_tts_cmd(args)
    return exit_info.value.code


def _run_timings(monkeypatch, audio: Path, tmp_path: Path, *, extra=()) -> int:
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _source: 1000)
    args = cli.build_parser().parse_args(
        [
            "timings",
            "--audio",
            str(audio),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-run",
            *extra,
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.run_timings(args)
    return exit_info.value.code


def _rows(home: Path, sql: str, params=()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _kinds(home: Path) -> list[str]:
    return [row["kind"] for row in _rows(home, "SELECT kind FROM text_sources ORDER BY kind")]


# ── transcribe ────────────────────────────────────────────────────────────────


def test_transcribe_persists_run_attempt_source_audio_and_private_transcript(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_transcribe(monkeypatch, audio)

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert data["transcript"] == TRANSCRIPT
    assert data["history"]["saved"] is True
    run_uuid = data["history"]["run_uuid"]

    runs = _rows(asr_home, "SELECT * FROM runs")
    assert len(runs) == 1
    assert runs[0]["run_uuid"] == run_uuid
    assert runs[0]["operation"] == "asr"
    assert runs[0]["status"] == "completed"
    assert Path(runs[0]["run_root"]).is_dir()
    assert Path(runs[0]["run_root"]).parent == asr_home / "runs"

    attempts = _rows(asr_home, "SELECT * FROM attempts")
    assert len(attempts) == 1
    assert attempts[0]["call_type"] == "asr_transcription"
    assert attempts[0]["provider"] == "qwen-local"
    assert attempts[0]["model"] == MODEL_ID
    assert attempts[0]["status"] == "completed"
    # A local, keyless route has no external charge; the amount stays unknown.
    assert attempts[0]["cost"] is None
    assert attempts[0]["cost_exact_available"] == 0

    artifacts = _rows(asr_home, "SELECT * FROM artifacts")
    assert [row["role"] for row in artifacts] == ["asr_source_audio"]
    assert artifacts[0]["path_kind"] == "external_absolute"
    assert artifacts[0]["path"] == str(audio.resolve())
    assert artifacts[0]["size_bytes"] == len(AUDIO_BYTES)
    assert artifacts[0]["sha256"] == hashlib.sha256(AUDIO_BYTES).hexdigest()
    assert artifacts[0]["availability"] == "present"

    texts = _rows(asr_home, "SELECT * FROM text_sources")
    assert [row["kind"] for row in texts] == ["asr_transcript"]
    assert texts[0]["content"] == TRANSCRIPT
    assert texts[0]["origin"] == "native_asr"
    assert texts[0]["text_completeness"] == "complete"
    assert texts[0]["language"] == "ru"

    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["operation_origin"] == "native_asr"
    assert snapshot["timestamp_mode"] == "native"
    assert snapshot["segments"] == [
        {"text": TRANSCRIPT, "start_s": 0.0, "end_s": 1.0},
    ]
    assert snapshot["segments_with_timestamps"] == 1


def test_transcribe_run_is_visible_through_history_metadata_only(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())
    _run_transcribe(monkeypatch, audio)
    capsys.readouterr()
    run_uuid = list_history(operation="asr")["runs"][0]["run_uuid"]

    detail = show_history(run_uuid)

    assert detail["run"]["operation"] == "asr"
    assert detail["run"]["status"] == "completed"
    assert detail["text_sources"][0]["kind"] == "asr_transcript"
    assert detail["text_sources"][0]["has_content"] is True
    assert detail["artifacts"][0]["role"] == "asr_source_audio"
    assert detail["artifacts"][0]["availability"] == "present"
    # The generic read never prints the private transcript itself.
    rendered = json.dumps(detail, ensure_ascii=False)
    assert TRANSCRIPT not in rendered


def test_user_flow_transcribe_then_history_list_show_costs(asr_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())
    _run_transcribe(monkeypatch, audio)
    capsys.readouterr()

    listed = list_history(operation="asr")
    assert listed["count"] == 1
    assert listed["database"]["exists"] is True
    run_uuid = listed["runs"][0]["run_uuid"]

    shown = show_history(run_uuid)
    assert shown["run"]["run_uuid"] == run_uuid

    # ``history costs`` uses the WAL-consistent reader, so a just-committed local
    # attempt is visible and counted as local work without an API charge.
    costs = costs_history()
    assert costs["attempts"] == 1
    assert costs["local_attempts_without_api_charge"] == 1
    assert costs["totals"] == []
    assert costs["completeness"] == "complete"


def test_transcribe_text_only_result_stores_no_invented_spans(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result(timed=False), timed=False)

    code = _run_transcribe(monkeypatch, audio)

    assert code == 0
    assert json.loads(capsys.readouterr().out)["history"]["saved"] is True
    runs = _rows(asr_home, "SELECT * FROM runs")
    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["timestamp_mode"] == "none"
    assert snapshot["segments"] == []
    assert snapshot["segments_with_timestamps"] == 0
    assert snapshot["word_count"] == 0
    assert _kinds(asr_home) == ["asr_transcript"]
    assert _rows(asr_home, "SELECT content FROM text_sources")[0]["content"] == TRANSCRIPT


def test_transcribe_stores_context_prompt_and_its_provenance(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_transcribe(monkeypatch, audio, extra=("--context", "private context hint"))

    assert code == 0
    run_uuid = json.loads(capsys.readouterr().out)["history"]["run_uuid"]
    texts = {
        row["kind"]: row["content"]
        for row in _rows(asr_home, "SELECT kind, content FROM text_sources")
    }
    assert texts["asr_transcript"] == TRANSCRIPT
    assert texts["asr_context"] == "private context hint"
    snapshot = json.loads(_rows(asr_home, "SELECT config_snapshot FROM runs")[0][0])
    assert snapshot["context_source"] == "inline"
    assert snapshot["context_prompt_present"] is True
    # ``history show`` reports presence only; the prompt text is never printed.
    rendered = json.dumps(show_history(run_uuid), ensure_ascii=False)
    assert "private context hint" not in rendered


def test_transcribe_records_absent_source_audio_without_invented_hash(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    calls: list[str] = []
    _install_spec(
        monkeypatch,
        _result(),
        factory=lambda: _DeletingProvider(_result(), calls),
    )

    code = _run_transcribe(monkeypatch, audio)

    assert code == 0
    assert calls == ["transcribe"]
    artifacts = _rows(asr_home, "SELECT * FROM artifacts")
    assert artifacts[0]["role"] == "asr_source_audio"
    assert artifacts[0]["availability"] == "missing"
    assert artifacts[0]["sha256"] is None
    assert artifacts[0]["size_bytes"] is None
    # The transcript is still stored even though the audio disappeared.
    assert _kinds(asr_home) == ["asr_transcript"]


@pytest.mark.skipif(os.name != "posix", reason="private-mode check is POSIX-only")
def test_transcribe_persistence_failure_keeps_transcript_and_reports_fixed_code(
    tmp_path, monkeypatch, capsys
):
    insecure_home = tmp_path / "insecure-home"
    insecure_home.mkdir()
    os.chmod(insecure_home, 0o755)
    monkeypatch.setenv("VOICEOVER_HOME", str(insecure_home))
    audio = _audio(tmp_path)
    calls: list[str] = []
    _install_spec(monkeypatch, _result(), factory=lambda: _FixtureProvider(_result(), calls))

    code = _run_transcribe(monkeypatch, audio)

    assert code == 50
    data = json.loads(capsys.readouterr().out)
    assert data["transcript"] == TRANSCRIPT
    assert data["status"] == "partial"
    assert data["history"] == {"saved": False, "error_code": "HISTORY_PERSISTENCE_FAILED"}
    # The provider ran exactly once: a persistence failure never retries it.
    assert calls == ["transcribe"]
    assert not (insecure_home / "history.sqlite3").exists()


def test_disabled_history_creates_nothing_and_keeps_the_old_output(tmp_path, monkeypatch, capsys):
    home = tmp_path / "disabled-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "settings.toml").write_text("[history]\nenabled = false\n", encoding="utf-8")
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_transcribe(monkeypatch, audio)

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert "history" not in data
    assert list(home.iterdir()) == []


def test_transcribe_persists_long_form_provenance_without_changing_coverage(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    base = _result()
    long_form_result = ASRResult(
        transcript=base.transcript,
        provider_id=base.provider_id,
        model_id=base.model_id,
        language=base.language,
        duration_s=200.0,
        segments=(ASRSegment(text=TRANSCRIPT, start_s=0.0, end_s=1.0),),
        alignment_origin="chunked",
        execution=ASRExecutionReceipt(
            runtime="fixture-runtime",
            resolved_device="cpu",
            resolved_compute="auto",
            long_form={
                "source_duration_s": 200.0,
                "coverage_verified": True,
                "chunks": [{"index": 0, "output_duration_s": 110.0}],
            },
        ),
    )
    _install_spec(monkeypatch, long_form_result)

    code = _run_transcribe(monkeypatch, audio)

    assert code == 0
    assert json.loads(capsys.readouterr().out)["history"]["saved"] is True
    snapshot = json.loads(_rows(asr_home, "SELECT config_snapshot FROM runs")[0][0])
    assert snapshot["timestamp_mode"] == "chunked"
    assert snapshot["long_form"]["coverage_verified"] is True
    assert snapshot["long_form"]["chunks"][0]["index"] == 0


def test_unreadable_settings_fails_closed_instead_of_silently_saving(
    asr_home, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "settings.toml").write_text('[history]\nenabled = "yes"\n', encoding="utf-8")
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_transcribe(monkeypatch, audio)

    assert code == 50
    data = json.loads(capsys.readouterr().out)
    assert data["transcript"] == TRANSCRIPT
    assert data["history"] == {"saved": False, "error_code": "HISTORY_PERSISTENCE_FAILED"}
    assert not (asr_home / "history.sqlite3").exists()


def test_read_commands_create_nothing_for_an_absent_database(asr_home):
    assert list_history()["count"] == 0
    assert costs_history()["attempts"] == 0
    assert not (asr_home / "history.sqlite3").exists()
    assert list(asr_home.iterdir()) == []


# ── timings ───────────────────────────────────────────────────────────────────


def test_timings_generic_local_asr_persists_json_srt_and_provenance(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result(words=True))

    code = _run_timings(monkeypatch, audio, tmp_path, extra=("--asr-provider", "qwen-local"))

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert data["history"]["saved"] is True
    assert Path(data["files"]["timings_json"]).is_file()
    assert Path(data["files"]["srt"]).is_file()

    runs = _rows(asr_home, "SELECT * FROM runs")
    assert runs[0]["operation"] == "timings"
    attempts = _rows(asr_home, "SELECT * FROM attempts")
    assert attempts[0]["call_type"] == "asr_timing"
    assert attempts[0]["provider"] == "qwen-local"
    assert attempts[0]["status"] == "completed"
    assert attempts[0]["cost"] is None

    artifacts = {row["role"]: row for row in _rows(asr_home, "SELECT * FROM artifacts")}
    assert set(artifacts) == {"asr_source_audio", "timings_json", "srt"}
    timings_json = Path(data["files"]["timings_json"])
    assert artifacts["timings_json"]["path"] == str(timings_json.resolve())
    assert (
        artifacts["timings_json"]["sha256"] == hashlib.sha256(timings_json.read_bytes()).hexdigest()
    )
    assert artifacts["timings_json"]["availability"] == "present"
    assert artifacts["srt"]["availability"] == "present"

    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["timestamp_basis"] == "asr_word_spans"
    assert snapshot["segment_count"] == 1
    assert snapshot["total_duration_ms"] == 1000
    assert snapshot["timings_json"] == str(timings_json)
    assert _kinds(asr_home) == ["asr_transcript"]
    assert _rows(asr_home, "SELECT content FROM text_sources")[0]["content"] == TRANSCRIPT


def test_timings_faster_whisper_persists_provider_segment_provenance(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    timing = TimingResult(
        segments=[
            TimingSegment(
                id=1,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text=TRANSCRIPT,
            )
        ],
        model="small",
        backend="faster-whisper",
        provider="faster-whisper",
        device="cpu",
        compute_type="int8",
        language="ru",
    )
    monkeypatch.setattr(transcription, "transcribe_timing_audio", lambda **_kwargs: timing)

    code = _run_timings(monkeypatch, audio, tmp_path)

    assert code == 0
    assert json.loads(capsys.readouterr().out)["history"]["saved"] is True
    runs = _rows(asr_home, "SELECT * FROM runs")
    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["timestamp_basis"] == "provider_segment_timestamps"
    assert snapshot["provider"] == "faster-whisper"
    assert snapshot["model"] == "small"
    assert _rows(asr_home, "SELECT content FROM text_sources")[0]["content"] == TRANSCRIPT


def test_timings_cloud_route_stays_legacy_and_persists_nothing(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    audio = _audio(tmp_path)
    timing = TimingResult(
        segments=[
            TimingSegment(
                id=1,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text=TRANSCRIPT,
            )
        ],
        model="grok-stt",
        backend="xai-stt",
        provider="xai-stt",
        device="",
        compute_type="",
        language="ru",
    )
    monkeypatch.setattr(transcription, "transcribe_timing_audio", lambda **_kwargs: timing)

    code = _run_timings(monkeypatch, audio, tmp_path, extra=("--timing-provider", "xai-stt"))

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "success"
    assert "history" not in data
    assert Path(data["files"]["timings_json"]).is_file()
    assert not (home / "history.sqlite3").exists()


def test_timings_text_only_asr_is_refused_before_any_timing_artifact(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result(timed=False))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    args = cli.build_parser().parse_args(
        [
            "timings",
            "--audio",
            str(audio),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-run",
            "--asr-provider",
            "qwen-local",
            "--json",
        ]
    )

    with pytest.raises(cli.CliError) as error:
        cli.run_timings(args)

    assert error.value.code == 40
    assert not (asr_home / "history.sqlite3").exists()
    out_dir = tmp_path / "out" / "timing-run"
    assert not (out_dir / "timing-run.srt").exists()


# ── verify-tts ────────────────────────────────────────────────────────────────


def test_verify_tts_pass_stores_private_transcript_and_no_expected_text(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    _install_spec(monkeypatch, _result())

    code = _run_verify(monkeypatch, audio, TRANSCRIPT, extra=("--receipt", str(receipt_path)))

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["passed"] is True
    assert data["history"]["saved"] is True
    assert "transcript" not in data
    assert TRANSCRIPT not in json.dumps(data, ensure_ascii=False)

    runs = _rows(asr_home, "SELECT * FROM runs")
    assert runs[0]["operation"] == "verify"
    assert runs[0]["parent_uuid"] is None
    assert _kinds(asr_home) == ["verification_transcript"]
    assert _rows(asr_home, "SELECT content FROM text_sources")[0]["content"] == TRANSCRIPT
    roles = {row["role"] for row in _rows(asr_home, "SELECT role FROM artifacts")}
    assert roles == {"asr_source_audio", "tts_quality_receipt"}
    assert _rows(asr_home, "SELECT path FROM artifacts WHERE role = 'tts_quality_receipt'")[0][
        "path"
    ] == str(receipt_path.resolve())
    snapshot = json.loads(runs[0]["config_snapshot"])
    assert snapshot["expected_text_persisted"] is False
    assert snapshot["quality_passed"] is True


def test_verify_tts_fail_keeps_exit_code_and_stores_no_expected_text(
    asr_home, tmp_path, monkeypatch, capsys
):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_verify(monkeypatch, audio, "совершенно другой ожидаемый текст")

    assert code == 60
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "quality_failed"
    assert data["history"]["saved"] is True
    stored = [row["content"] for row in _rows(asr_home, "SELECT content FROM text_sources")]
    assert stored == [TRANSCRIPT]
    assert all("совершенно другой" not in (content or "") for content in stored)
    snapshot = json.loads(_rows(asr_home, "SELECT config_snapshot FROM runs")[0][0])
    assert snapshot["quality_passed"] is False
    assert snapshot["failure_reasons"]


def test_verify_tts_links_a_parent_native_tts_run_for_its_audio(asr_home, tmp_path, monkeypatch):
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import HistoryRepository

    run_dir = tmp_path / "out" / "tts-run"
    run_dir.mkdir(parents=True)
    database = HistoryDatabase(asr_home / "history.sqlite3")
    database.connect()
    database.migrate()
    try:
        parent = HistoryRepository(database).create_run(
            operation="tts",
            run_root=str(run_dir.resolve()),
            status="completed",
            config_snapshot={"operation_origin": "native_snapshot"},
        )
    finally:
        database.close()
    audio = run_dir / "full.mp3"
    audio.write_bytes(AUDIO_BYTES)
    _install_spec(monkeypatch, _result())

    _run_verify(monkeypatch, audio, TRANSCRIPT)

    verify_runs = _rows(asr_home, "SELECT * FROM runs WHERE operation = 'verify'")
    assert verify_runs[0]["parent_uuid"] == parent.run_uuid


# ── privacy and reader safety ─────────────────────────────────────────────────


def test_private_transcript_never_reaches_history_stdout(asr_home, tmp_path, monkeypatch, capsys):
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())
    _run_transcribe(monkeypatch, audio)
    capsys.readouterr()
    run_uuid = list_history(operation="asr")["runs"][0]["run_uuid"]

    for payload in (list_history(operation="asr"), show_history(run_uuid), costs_history()):
        assert TRANSCRIPT not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.skipif(os.name != "posix", reason="private-mode check is POSIX-only")
def test_verify_tts_persistence_failure_keeps_fail_closed_semantics(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    os.chmod(home, 0o755)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    audio = _audio(tmp_path)
    _install_spec(monkeypatch, _result())

    code = _run_verify(monkeypatch, audio, "несовпадающий текст")

    assert code == 50
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "quality_failed"
    assert data["history"] == {"saved": False, "error_code": "HISTORY_PERSISTENCE_FAILED"}
