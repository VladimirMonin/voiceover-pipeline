"""End-to-end contract tests for the native local-timing vertical (S05 slice).

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, fake in-memory TTS and timing providers, and patched FFmpeg/concat
seams. No real provider, network call, API key, ``.env``, model, or download is
used. The tests assert paid-submit counts, linked timing history, and the
partial-result behavior rather than only statuses, so a slice that silently
re-submits TTS, loses the paid audio on a timing failure, or duplicates timing
rows fails here.
"""

import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.models import ScriptChunk, SynthesisResult, TimingResult, TimingSegment
from voiceover_pipeline.services import native_generation, transcription

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
POLZA_SYNC_MODEL = "openai/gpt-4o-mini-tts"
TRANSCRIPT = "Текст распознавания таймингов."


class FakeMediaProvider:
    """Offline stand-in for the Polza ElevenLabs ``/media`` provider."""

    def __init__(self) -> None:
        self.on_media_task_accepted: Any = None
        self.on_media_completed: Any = None
        self.submits: list[str] = []
        self.recovers: list[str] = []
        self.script: Any = None

    @staticmethod
    def _audio(text: str, chunk_id: str) -> SynthesisResult:
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.submits.append(chunk_id)
        if self.script is not None:
            return self.script(self, text, chunk_id)
        self.on_media_task_accepted(f"task-{chunk_id}")
        self.on_media_completed(f"task-{chunk_id}", {"cost_rub": Decimal("0.3")}, f"gen-{chunk_id}")
        return self._audio(text, chunk_id)

    def recover_media_task(self, task_id: str, text: str, chunk_id: str) -> SynthesisResult:
        self.recovers.append(task_id)
        self.on_media_completed(task_id, {"cost_rub": Decimal("0.3")}, f"gen-{chunk_id}")
        return self._audio(text, chunk_id)


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


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


@pytest.fixture
def native_env(tmp_path, monkeypatch):
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
    monkeypatch.setattr(cli, "concat_audio_files", _concat)
    # The preflight and the model run are replaced per test; a no-op preflight is
    # the default so only the explicit preflight test exercises the dependency
    # probe (this environment has no local faster-whisper install).
    monkeypatch.setattr(native_generation, "_preflight_local_timing", lambda _options: None)
    return home


def _script(tmp_path: Path, parts: list[str]) -> Path:
    path = tmp_path / "script.md"
    path.write_text("\n******\n".join(parts), encoding="utf-8")
    return path


def _generate_argv(
    tmp_path: Path,
    script: Path,
    run_id: str,
    *,
    model: str = POLZA_MEDIA_MODEL,
    voice: str = "Rachel",
    provider: str = "polza-tts",
    extra=(),
):
    return [
        "voiceover-pipeline",
        "generate",
        "--provider",
        provider,
        "--model",
        model,
        "--voice",
        voice,
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]


def _timing_args() -> tuple[str, ...]:
    return ("--with-timings", "--timing-provider", "faster-whisper")


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


def _install_timing(monkeypatch, result=None, error=None):
    calls: list[dict] = []

    def fake_import(**_kwargs):
        calls.append(dict(_kwargs))
        if error is not None:
            raise error
        return result if result is not None else _timing_result()

    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake_import)
    return calls


def _history_argv(run_uuid: str, verb: str, *, extra=()) -> list[str]:
    return ["voiceover-pipeline", "history", verb, run_uuid, *extra, "--json"]


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


def _timing_children(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT * FROM runs WHERE parent_uuid = ? AND operation = 'timings'",
        (run_uuid,),
    )


def _tts_attempt_costs(home: Path, run_uuid: str) -> list[str | None]:
    return [
        row["cost"]
        for row in _rows(
            home,
            "SELECT cost FROM attempts WHERE run_uuid = ? ORDER BY attempt_uuid",
            (run_uuid,),
        )
    ]


# ── fresh run: TTS + local timings in one command ─────────────────────────────


def test_native_fresh_with_timings_produces_audio_captions_and_linked_history(
    tmp_path, monkeypatch, capsys, native_env
):
    """One command writes audio, local caption files, and a linked timing run."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    timing_calls = _install_timing(monkeypatch)
    script = _script(tmp_path, ["Первый фрагмент текста.", "Второй фрагмент текста."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-timing", extra=_timing_args())
    )

    assert code == 0, payload
    assert payload["status"] == "success"
    assert payload["timing"] == {"complete": True}
    assert payload["segment_count"] == 1
    assert provider.submits == ["chunk_01", "chunk_02"]
    # The local model ran exactly once, offline, and never allowed a download.
    assert len(timing_calls) == 1
    assert timing_calls[0]["local_files_only"] is True

    run_root = tmp_path / "out" / "native-timing"
    timings_json = run_root / "native-timing.timings.json"
    srt = run_root / "native-timing.srt"
    assert timings_json.is_file() and srt.is_file()
    assert Path(payload["files"]["timings_json"]) == timings_json
    assert Path(payload["files"]["srt"]) == srt
    assert "00:00:00,000 --> 00:00:01,000" in srt.read_text(encoding="utf-8")
    # The generated audio and its projection exist too.
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["timings_json"] == str(timings_json)
    assert manifest["srt"] == str(srt)

    run_uuid = _run_uuid("native-timing")
    children = _timing_children(native_env, run_uuid)
    assert len(children) == 1
    child = children[0]
    assert child["status"] == "completed"
    # The linked timing run carries the observed transcript privately and the
    # TTS run's script text is not replaced.
    texts = _rows(
        native_env,
        "SELECT kind, content FROM text_sources WHERE run_uuid = ?",
        (child["run_uuid"],),
    )
    assert [(row["kind"], row["content"]) for row in texts] == [("asr_transcript", TRANSCRIPT)]
    artifacts = _rows(
        native_env, "SELECT role, path FROM artifacts WHERE run_uuid = ?", (child["run_uuid"],)
    )
    roles = {row["role"] for row in artifacts}
    assert {"timings_json", "srt", "asr_source_audio"} <= roles
    # The transcript is private: it never reaches the machine output or the JSON.
    assert TRANSCRIPT not in json.dumps(payload)


# ── timing failure preserves the paid audio, then resume repairs it ──────────


def test_native_timing_failure_preserves_paid_audio_then_resume_repairs(
    tmp_path, monkeypatch, capsys, native_env
):
    """A timing failure keeps the paid TTS evidence; resume finishes only timing."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch, error=RuntimeError("whisper boom"))
    script = _script(tmp_path, ["Оплаченный фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-fail", extra=_timing_args()),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_TIMING_FAILED"
    assert provider.submits == ["chunk_01"]
    run_uuid = _run_uuid("native-timing-fail")
    run_root = tmp_path / "out" / "native-timing-fail"
    # Completed TTS audio and cost stay; no timing artifacts or linked run exist.
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    status = _rows(native_env, "SELECT status FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    assert status["status"] == "completed"
    costs_before = _tts_attempt_costs(native_env, run_uuid)
    assert costs_before == ["0.3"]
    assert not (run_root / "native-timing-fail.timings.json").exists()
    assert not (run_root / "native-timing-fail.srt").exists()
    assert _timing_children(native_env, run_uuid) == []

    # Resume repairs only the missing local timing: no TTS POST or GET.
    _install_timing(monkeypatch)
    monkeypatch.setattr(cli, "build_provider", _explode)
    resume_argv = _generate_argv(
        tmp_path, script, "native-timing-fail", extra=(*_timing_args(), "--resume")
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    assert provider.submits == ["chunk_01"]
    assert (run_root / "native-timing-fail.timings.json").is_file()
    assert len(_timing_children(native_env, run_uuid)) == 1
    assert _tts_attempt_costs(native_env, run_uuid) == costs_before


def test_native_timing_history_failure_keeps_audio_and_reports_partial(
    tmp_path, monkeypatch, capsys, native_env
):
    """A linked-history write failure keeps audio/cost, drops artifacts, exit 50."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch)

    def boom(_save):
        raise RuntimeError("history database unavailable")

    monkeypatch.setattr(native_generation, "persist_asr_history", boom)
    script = _script(tmp_path, ["Сбой записи таймингов."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-db", extra=_timing_args()),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_TIMING_HISTORY_FAILED"
    run_uuid = _run_uuid("native-timing-db")
    run_root = tmp_path / "out" / "native-timing-db"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    status = _rows(native_env, "SELECT status FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    assert status["status"] == "completed"
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]
    # The unlinked timing artifacts are removed so no export can reference them.
    assert not (run_root / "native-timing-db.timings.json").exists()
    assert not (run_root / "native-timing-db.srt").exists()
    assert _timing_children(native_env, run_uuid) == []


# ── completed resume / export reuse verified timing, never duplicate it ──────


def test_native_completed_timing_resume_and_sync_do_not_duplicate_or_call_model(
    tmp_path, monkeypatch, capsys, native_env
):
    """A completed timed run re-exports without a model call or a second timing row."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch)
    script = _script(tmp_path, ["Готовый тайминговый прогон."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-done", extra=_timing_args()),
    )
    assert code == 0
    run_uuid = _run_uuid("native-timing-done")
    assert len(_timing_children(native_env, run_uuid)) == 1

    def explode_timing(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a completed timed run must not run the local model")

    monkeypatch.setattr(transcription, "transcribe_timing_audio", explode_timing)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    for verb in ("resume", "sync"):
        code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, verb))
        assert code == 0, payload
        assert payload["timing"] == {"complete": True}
        assert Path(payload["files"]["srt"]).is_file()

    assert len(_timing_children(native_env, run_uuid)) == 1
    assert provider.submits == ["chunk_01"]


# ── changed options and unconfirmed submits block before any provider ────────


def test_native_changed_timing_options_block_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A changed timing option fails the recorded-output check before any provider."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch)
    script = _script(tmp_path, ["Проверка опций таймингов."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-opts", extra=_timing_args()),
    )
    assert code == 0

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(transcription, "transcribe_timing_audio", _explode)
    changed = _generate_argv(
        tmp_path,
        script,
        "native-timing-opts",
        extra=(*_timing_args(), "--timing-model", "medium", "--resume"),
    )
    code, payload = _json_run(monkeypatch, capsys, changed)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_PROCESSING_UNSUPPORTED"
    assert provider.submits == ["chunk_01"]


def test_native_timing_run_blocks_unconfirmed_paid_submit_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """An unconfirmed paid submit still blocks a timing run, with zero requests."""
    provider = FakeMediaProvider()

    def fail_before_accept(_inner, _text, _chunk_id):
        raise requests.Timeout("read timed out")

    provider.script = fail_before_accept
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch)
    script = _script(tmp_path, ["Неопределённый платный исход."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-unconfirmed", extra=_timing_args()),
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]

    monkeypatch.setattr(cli, "build_provider", _explode)
    resume_argv = _generate_argv(
        tmp_path,
        script,
        "native-timing-unconfirmed",
        extra=(*_timing_args(), "--resume"),
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert provider.submits == ["chunk_01"]


# ── history resume without the source script completes pending timing ────────


def test_history_resume_without_script_completes_pending_timing(
    tmp_path, monkeypatch, capsys, native_env
):
    """A timed run with a failed timing is repaired from its snapshot, no script."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch, error=RuntimeError("first timing boom"))
    script = _script(tmp_path, ["Первый.", "Второй."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-hist", extra=_timing_args()),
    )
    assert code == 50
    run_uuid = _run_uuid("native-timing-hist")
    run_root = tmp_path / "out" / "native-timing-hist"
    script.unlink()

    _install_timing(monkeypatch)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert payload["timing"] == {"complete": True}
    assert (run_root / "native-timing-hist.srt").is_file()
    assert len(_timing_children(native_env, run_uuid)) == 1
    assert provider.submits == ["chunk_01", "chunk_02"]


# ── history sync never runs the model and never re-POSTs ─────────────────────


def test_history_sync_with_timings_runs_no_model_and_stays_incomplete(
    tmp_path, monkeypatch, capsys, native_env
):
    """Sync repairs exports and a known Media id, never the pending timing."""
    provider = FakeMediaProvider()

    def accept_then_fail(inner, text, chunk_id):
        inner.on_media_task_accepted(f"task-{chunk_id}")
        raise requests.Timeout("poll timed out")

    provider.script = accept_then_fail
    _install_provider(monkeypatch, provider)
    _install_timing(monkeypatch)
    script = _script(tmp_path, ["Фрагмент для GET и таймингов."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-sync", extra=_timing_args()),
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]

    run_uuid = _run_uuid("native-timing-sync")
    sync_provider = FakeMediaProvider()
    _install_provider(monkeypatch, sync_provider)

    def explode_timing(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("history sync must not run the local timing model")

    monkeypatch.setattr(transcription, "transcribe_timing_audio", explode_timing)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert sync_provider.submits == []
    assert sync_provider.recovers == ["task-chunk_01"]
    assert payload["timing"] == {"complete": False}
    assert "timings_json" not in payload["files"]
    assert _timing_children(native_env, run_uuid) == []


# ── local model availability preflight ───────────────────────────────────────


def test_native_timing_model_unavailable_blocks_before_paid_submit(
    tmp_path, monkeypatch, capsys, native_env
):
    """An unavailable local model stops the command before the paid TTS POST."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    timing_calls = _install_timing(monkeypatch)

    def blocked(_options):
        raise native_generation.NativeGenerationError(
            "the local Faster-Whisper model is not cached.",
            code=10,
            error_code="NATIVE_TIMING_MODEL_UNAVAILABLE",
        )

    monkeypatch.setattr(native_generation, "_preflight_local_timing", blocked)
    script = _script(tmp_path, ["Платный POST не должен случиться."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-timing-preflight", extra=_timing_args()),
    )

    assert code == 10
    assert payload["details"]["error_code"] == "NATIVE_TIMING_MODEL_UNAVAILABLE"
    assert provider.submits == []
    assert timing_calls == []
    # No paid chunk audio was written for the blocked run.
    chunks_dir = tmp_path / "out" / "native-timing-preflight" / "chunks"
    assert not list(chunks_dir.glob("chunk_*.mp3"))


def test_preflight_local_timing_rejects_cloud_provider_and_reports_unavailable(monkeypatch):
    """The preflight only admits faster-whisper and reports an unavailable model."""
    options = native_generation.NativeTimingOptions(
        provider="groq-whisper",
        model=None,
        device="cpu",
        compute="int8",
        language="ru",
        word_timestamps=False,
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_local_timing(options)
    assert excinfo.value.error_code == "NATIVE_TIMING_PROVIDER_UNSUPPORTED"
    assert excinfo.value.code == 30

    from voiceover_pipeline.providers import faster_whisper as faster_whisper_module

    monkeypatch.setattr(
        faster_whisper_module,
        "faster_whisper_availability",
        lambda _model: faster_whisper_module.FasterWhisperAvailability(
            available=False, reason_code="model_not_cached", remediation="not cached"
        ),
    )
    local = native_generation.NativeTimingOptions(
        provider="faster-whisper",
        model=None,
        device="cpu",
        compute="int8",
        language="ru",
        word_timestamps=False,
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_local_timing(local)
    assert excinfo.value.error_code == "NATIVE_TIMING_MODEL_UNAVAILABLE"
    assert excinfo.value.code == 10


def test_faster_whisper_availability_reports_missing_package() -> None:
    """The availability probe reports an unimportable package without raising."""
    from voiceover_pipeline.providers.faster_whisper import faster_whisper_availability

    availability = faster_whisper_availability("small")
    # This environment has no local faster-whisper install; the probe must report
    # an unavailable dependency rather than raising or downloading.
    assert availability.available is False
    assert availability.reason_code is not None
    assert availability.remediation


def test_native_timing_options_are_recorded_and_validated() -> None:
    """The recorded timing options round-trip and a malformed block fails closed."""
    options = native_generation.NativeTimingOptions(
        provider="faster-whisper",
        model=None,
        device="cpu",
        compute="int8",
        language="ru",
        word_timestamps=True,
    )
    recorded = native_generation.build_output_options(False, timing=options)
    assert native_generation.timing_options_from_output_options(recorded) == options
    # A run without timing keeps its original output options.
    base = native_generation.build_output_options(False)
    assert "timing_enabled" not in base
    assert native_generation.timing_options_from_output_options(base) is None
    # A malformed timing block is refused rather than silently dropped.
    malformed = dict(recorded)
    malformed["timing_model"] = 5
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation.timing_options_from_output_options(malformed)
    assert excinfo.value.error_code == "NATIVE_TIMING_OPTIONS_INVALID"


def test_script_chunk_helpers_are_stable_for_timing() -> None:
    """Sanity check that the timing slice consumes plain generated chunk ids."""
    chunk = ScriptChunk(number=1, id="chunk_01", text="x")
    assert chunk.id == "chunk_01"
