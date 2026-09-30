"""Offline contract tests for the two integrated paid cloud post-audio routes.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, fake in-memory TTS/cloud-STT adapters, and patched FFmpeg/concat seams.
No real provider, network call, API key, ``.env``, model, or paid request is used.

Two functioning routes are covered, both riding the same committed
paid-transcription boundary the standalone ``timings`` command uses:

* the integrated cloud timing step (``--with-timings --timing-provider
  groq-whisper|xai-stt``) after a completed native TTS run; and
* the per-turn paid xAI quality gate (``--tts-quality-provider xai-stt``) on the
  OpenRouter Gemini dialogue route, before the next paid turn and before concat.

The tests assert POST counts, child-run identity, private raw evidence, durable
verdicts, parent cost preservation, and the machine envelope rather than only
statuses, so a slice that reissues a paid POST, loses the paid audio/cost, leaks a
private transcript, or overwrites the parent native root fails here.
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
import voiceover_pipeline.config as config
from voiceover_pipeline.gemini_dialogue import GEMINI_TTS_MODEL
from voiceover_pipeline.models import SynthesisResult, TimingResult, TimingSegment
from voiceover_pipeline.providers import xai_stt as xai_stt_module
from voiceover_pipeline.providers.xai_stt import parse_xai_timing_result
from voiceover_pipeline.services import native_generation, transcription

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
TRANSCRIPT = "Текст распознавания таймингов."
_TIMING_ARGS = ("--with-timings", "--timing-provider", "groq-whisper")
_TIMING_ARGS_XAI = ("--with-timings", "--timing-provider", "xai-stt")
_QUALITY_ARGS = ("--tts-quality-provider", "xai-stt")
# Captured before the fixture replaces it, so one test can exercise the real key
# preflight while every other test keeps the offline no-op.
_REAL_PREFLIGHT_PAID_TIMING = native_generation._preflight_paid_timing


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

    def synthesize_chunk(
        self, text: str, chunk_id: str, voice: str | None = None
    ) -> SynthesisResult:
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


class FakeDialogueProvider:
    """Offline stand-in for the OpenRouter Gemini per-turn dialogue provider."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.fail_on: str | None = None

    def synthesize_chunk(
        self, text: str, chunk_id: str, voice: str | None = None
    ) -> SynthesisResult:
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


class FakeXaiStt:
    """Offline stand-in for the xAI STT adapter used by the per-turn quality gate."""

    posts: list[str] = []
    texts: dict[str, str] = {}
    fail: Exception | None = None

    @classmethod
    def reset(cls) -> None:
        cls.posts = []
        cls.texts = {}
        cls.fail = None

    def __init__(self, model: str = "grok-stt", api_key: str | None = None) -> None:
        self.model = model

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
        on_raw_response: Any = None,
    ) -> TimingResult:
        name = Path(audio_path).name
        FakeXaiStt.posts.append(name)
        if FakeXaiStt.fail is not None:
            raise FakeXaiStt.fail
        text = FakeXaiStt.texts[name]
        body = json.dumps(
            {
                "text": text,
                "language": language,
                "duration": 1.0,
                "words": [{"text": text, "start": 0.0, "end": 1.0, "confidence": 0.99}],
            }
        ).encode()
        if on_raw_response is not None:
            on_raw_response(body, "application/json")
        return parse_xai_timing_result(
            json.loads(body),
            model=self.model,
            language=language,
            word_timestamps=word_timestamps,
            audio_path=Path(audio_path),
        )


def _timing_result(provider: str) -> TimingResult:
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
        model="whisper-large-v3-turbo",
        backend=provider,
        provider=provider,
        language="ru",
        timestamp_basis="provider_segment_timestamps",
    )


def _groq_body() -> bytes:
    return json.dumps(
        {
            "text": TRANSCRIPT,
            "language": "ru",
            "duration": 1.0,
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": TRANSCRIPT}],
        }
    ).encode()


def _xai_body() -> bytes:
    return json.dumps(
        {
            "text": TRANSCRIPT,
            "language": "ru",
            "duration": 1.0,
            "words": [{"text": TRANSCRIPT, "start": 0.0, "end": 1.0, "confidence": 0.9}],
        }
    ).encode()


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


def _concat_dialogue(_ffmpeg: str, turns, output: Path) -> None:
    output.write_bytes(b"-".join(Path(path).read_bytes() for path, _pause in turns))


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


@pytest.fixture
def paid_env(tmp_path, monkeypatch):
    """Point history at a temp home, replace media seams, and stub key reads."""
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
    monkeypatch.setattr(cli, "concat_dialogue_turns", _concat_dialogue)
    monkeypatch.setattr(native_generation, "_preflight_local_timing", lambda _options: None)
    monkeypatch.setattr(native_generation, "_preflight_local_quality", lambda _options: None)
    monkeypatch.setattr(native_generation, "_preflight_paid_timing", lambda _options: None)
    monkeypatch.setattr(native_generation, "_preflight_paid_quality", lambda _options: None)
    monkeypatch.setattr(config, "read_groq_key", lambda: "gsk-test")
    monkeypatch.setattr(config, "read_xai_key", lambda: "xai-test")
    FakeXaiStt.reset()
    return home


def _markdown_script(tmp_path: Path, parts: list[str]) -> Path:
    path = tmp_path / "script.md"
    path.write_text("\n******\n".join(parts), encoding="utf-8")
    return path


def _dialogue_script(tmp_path: Path, sections: list[list[str]]) -> Path:
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


def _generate_argv(
    tmp_path: Path,
    script: Path,
    run_id: str,
    *,
    provider="polza-tts",
    model=POLZA_MEDIA_MODEL,
    voice="Rachel",
    extra=(),
):
    argv = [
        "voiceover-pipeline",
        "generate",
        "--provider",
        provider,
        "--model",
        model,
    ]
    if voice is not None:
        argv += ["--voice", voice]
    argv += [
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]
    return argv


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    out = capsys.readouterr().out
    return code, json.loads(out)


def _install_provider(monkeypatch, provider):
    builds: list[str] = []

    def fake_build(*_args, **_kwargs):
        builds.append("built")
        return provider

    monkeypatch.setattr(cli, "build_provider", fake_build)
    return builds


def _install_cloud_timing(
    monkeypatch,
    provider: str,
    *,
    body: bytes | None = None,
    post_fail: Exception | None = None,
    crash_after_body: bool = False,
):
    """Replace the cloud timing adapter with one that records and never networks."""
    calls: list[dict] = []

    def fake_transcribe(**kwargs):
        calls.append(dict(kwargs))
        if post_fail is not None:
            raise post_fail
        payload = body if body is not None else _groq_body()
        kwargs["on_raw_response"](payload, "application/json")
        if crash_after_body:
            raise requests.Timeout("read timed out")
        return _timing_result(provider)

    monkeypatch.setattr(transcription, "transcribe_timing_audio", fake_transcribe)
    return calls


def _install_xai(monkeypatch, texts: dict[str, str], fail: Exception | None = None):
    FakeXaiStt.posts = []
    FakeXaiStt.texts = dict(texts)
    FakeXaiStt.fail = fail
    monkeypatch.setattr(xai_stt_module, "XAISttProvider", FakeXaiStt)
    return FakeXaiStt.posts


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


def _timing_children(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT * FROM runs WHERE parent_uuid = ? AND operation = 'timings'",
        (run_uuid,),
    )


def _quality_children(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    return _rows(
        home,
        "SELECT * FROM runs WHERE parent_uuid = ? AND operation = 'verify'",
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


def _child_raw_bodies(child_root: Path) -> list[Path]:
    raw = child_root / "raw"
    return sorted(raw.glob("*.body")) if raw.is_dir() else []


def _set_attempt_status(home: Path, run_uuid: str, status: str) -> None:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        connection.execute("UPDATE attempts SET status = ? WHERE run_uuid = ?", (status, run_uuid))
        connection.commit()
    finally:
        connection.close()


def _drop_raw_artifact(home: Path, run_uuid: str) -> None:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        connection.execute(
            "DELETE FROM artifacts WHERE run_uuid = ? AND role = 'provider_raw_response'",
            (run_uuid,),
        )
        connection.commit()
    finally:
        connection.close()


# ── integrated cloud timing: fresh run ───────────────────────────────────────


def test_native_cloud_timing_fresh_persists_boundary_and_links_artifacts(
    tmp_path, monkeypatch, capsys, paid_env
):
    """One command pays the TTS once, the timing once, and links private evidence."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    script = _markdown_script(tmp_path, ["Первый фрагмент текста.", "Второй фрагмент текста."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "cloud-timing", extra=_TIMING_ARGS)
    )

    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    assert provider.submits == ["chunk_01", "chunk_02"]
    assert len(calls) == 1
    run_uuid = _run_uuid("cloud-timing")
    run_root = tmp_path / "out" / "cloud-timing"
    children = _timing_children(paid_env, run_uuid)
    assert len(children) == 1
    child = children[0]
    assert child["status"] == "completed"
    snapshot = json.loads(child["config_snapshot"])
    assert snapshot["operation_origin"] == "native_paid_transcription"
    assert snapshot["provider"] == "groq-whisper"
    # The private raw response body lives under the child's own root, never the
    # parent native root.
    child_root = Path(snapshot["output_root"])
    assert child_root != run_root
    assert child_root.is_relative_to(run_root)
    bodies = _child_raw_bodies(child_root)
    assert len(bodies) == 1
    assert bodies[0].read_bytes() == _groq_body()
    # The published timing artifacts live in the parent root, as for a local step.
    timings_json = run_root / "cloud-timing.timings.json"
    srt = run_root / "cloud-timing.srt"
    assert timings_json.is_file() and srt.is_file()
    assert Path(payload["files"]["timings_json"]) == timings_json
    assert "00:00:00,000 --> 00:00:01,000" in srt.read_text(encoding="utf-8")
    manifest = json.loads(timings_json.read_text(encoding="utf-8"))
    assert manifest["timestamp_basis"] == "provider_segment_timestamps"
    # The observed transcript is stored privately on the child, not in the payload.
    texts = _rows(
        paid_env,
        "SELECT kind, content FROM text_sources WHERE run_uuid = ?",
        (child["run_uuid"],),
    )
    assert [(row["kind"], row["content"]) for row in texts] == [("asr_transcript", TRANSCRIPT)]
    assert TRANSCRIPT not in json.dumps(payload)
    # The parent's own TTS raw evidence is untouched.
    assert (run_root / "raw" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"


def test_native_cloud_timing_unknown_submit_blocks_resume_without_second_post(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A lost/pending paid timing POST leaves a marker that no resume may retry."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "groq-whisper", post_fail=requests.Timeout("read timed out"))
    script = _markdown_script(tmp_path, ["Оплаченный фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-timing-lost", extra=_TIMING_ARGS),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_TIMING_FAILED"
    assert provider.submits == ["chunk_01"]
    run_uuid = _run_uuid("cloud-timing-lost")
    run_root = tmp_path / "out" / "cloud-timing-lost"
    # The completed paid TTS audio, raw bytes, and cost survive the timing failure.
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert _tts_attempt_costs(paid_env, run_uuid) == ["0.3"]
    assert not (run_root / "cloud-timing-lost.timings.json").exists()
    children = _timing_children(paid_env, run_uuid)
    assert len(children) == 1
    assert children[0]["status"] != "completed"

    # Resume: the unconfirmed marker blocks with no second provider call.
    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert retry_calls == []
    assert provider.submits == ["chunk_01"]
    assert _timing_children(paid_env, run_uuid)[0]["status"] != "completed"


def test_native_cloud_timing_saved_body_replays_locally_with_no_second_post(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A response saved before a downstream crash is replayed locally on resume."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "groq-whisper", crash_after_body=True)
    script = _markdown_script(tmp_path, ["Фрагмент для повтора."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-timing-replay", extra=_TIMING_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("cloud-timing-replay")
    run_root = tmp_path / "out" / "cloud-timing-replay"
    child = _timing_children(paid_env, run_uuid)[0]
    child_root = Path(json.loads(child["config_snapshot"])["output_root"])
    assert len(_child_raw_bodies(child_root)) == 1

    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    assert retry_calls == []
    assert provider.submits == ["chunk_01"]
    assert (run_root / "cloud-timing-replay.srt").is_file()
    assert _timing_children(paid_env, run_uuid)[0]["status"] == "completed"


def test_native_cloud_timing_recovers_linked_crash_window_with_no_second_post(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A validated receipt+body pair left unlinked is reconciled on resume."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "groq-whisper", crash_after_body=True)
    script = _markdown_script(tmp_path, ["Окно падения."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-timing-crash", extra=_TIMING_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("cloud-timing-crash")
    child_uuid = _timing_children(paid_env, run_uuid)[0]["run_uuid"]
    # Recreate the crash window: the body was written before the link committed.
    _set_attempt_status(paid_env, child_uuid, "submitting")
    _drop_raw_artifact(paid_env, child_uuid)

    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    assert retry_calls == []
    assert provider.submits == ["chunk_01"]
    assert _timing_children(paid_env, run_uuid)[0]["status"] == "completed"


def test_native_cloud_timing_sync_is_read_only_and_never_posts(
    tmp_path, monkeypatch, capsys, paid_env
):
    """Sync reports the pending paid timing incomplete and issues no request."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "groq-whisper", crash_after_body=True)
    script = _markdown_script(tmp_path, ["Синхронизация."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-timing-sync", extra=_TIMING_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("cloud-timing-sync")

    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert payload["timing"] == {"complete": False}
    assert retry_calls == []
    assert _timing_children(paid_env, run_uuid)[0]["status"] != "completed"


def test_native_cloud_xai_timing_records_derived_basis(tmp_path, monkeypatch, capsys, paid_env):
    """The xai-stt integrated timing step records its honest derived basis."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "xai-stt", body=_xai_body())
    script = _markdown_script(tmp_path, ["Фрагмент для xai таймингов."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-xai-timing", extra=_TIMING_ARGS_XAI),
    )

    assert code == 0, payload
    assert payload["timing"] == {"complete": True}
    run_uuid = _run_uuid("cloud-xai-timing")
    child = _timing_children(paid_env, run_uuid)[0]
    assert json.loads(child["config_snapshot"])["provider"] == "xai-stt"
    manifest = json.loads(
        (tmp_path / "out" / "cloud-xai-timing" / "cloud-xai-timing.timings.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["timestamp_basis"] == "derived_from_provider_words"


def test_child_paid_run_read_only_dispatch_is_sidecar_safe(tmp_path, monkeypatch, capsys, paid_env):
    """A child paid run is read-only dispatched without starting new work."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_cloud_timing(monkeypatch, "groq-whisper")
    script = _markdown_script(tmp_path, ["Готовый фрагмент."])

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "child-dispatch", extra=_TIMING_ARGS)
    )
    assert code == 0
    parent_uuid = _run_uuid("child-dispatch")
    child_uuid = _timing_children(paid_env, parent_uuid)[0]["run_uuid"]
    runs_before = len(_rows(paid_env, "SELECT run_uuid FROM runs"))

    # ``history sync <timing-child>`` reports the durable state with no provider.
    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_calls = _install_cloud_timing(monkeypatch, "groq-whisper")
    code, payload = _json_run(monkeypatch, capsys, _history_argv(child_uuid, "sync"))
    assert code == 0, payload
    assert payload["operation"] == "timings"
    assert payload["complete"] is True
    assert retry_calls == []
    assert len(_rows(paid_env, "SELECT run_uuid FROM runs")) == runs_before

    # ``history sync <quality-child>`` fails closed read-only (no paid route).
    posts = _install_xai(monkeypatch, _dialogue_turn_texts())
    _install_provider(monkeypatch, FakeDialogueProvider())
    dialogue = _dialogue_script(tmp_path, [["Host: Первая реплика."]])
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            dialogue,
            "child-dispatch-quality",
            provider="openrouter-tts",
            model=GEMINI_TTS_MODEL,
            voice=None,
            extra=_QUALITY_ARGS,
        ),
    )
    assert code == 0
    quality_parent = _run_uuid("child-dispatch-quality")
    quality_child = _quality_children(paid_env, quality_parent)[0]["run_uuid"]
    runs_before = len(_rows(paid_env, "SELECT run_uuid FROM runs"))
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, _payload = _json_run(monkeypatch, capsys, _history_argv(quality_child, "sync"))
    assert code != 0
    assert len(_rows(paid_env, "SELECT run_uuid FROM runs")) == runs_before
    assert posts == ["turn_0001.mp3"]


def test_native_cloud_timing_requires_key_before_any_paid_submit(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A missing cloud timing key stops the command before the paid TTS POST."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    calls = _install_cloud_timing(monkeypatch, "groq-whisper")

    def missing() -> str:
        raise RuntimeError("GROQ_API_KEY is required for timing-provider=groq-whisper.")

    monkeypatch.setattr(config, "read_groq_key", missing)
    monkeypatch.setattr(native_generation, "_preflight_paid_timing", _REAL_PREFLIGHT_PAID_TIMING)
    script = _markdown_script(tmp_path, ["Платный POST не должен случиться."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "cloud-timing-key", extra=_TIMING_ARGS),
    )

    assert code == 20
    assert payload["details"]["error_code"] == "NATIVE_PAID_TIMING_KEY_MISSING"
    assert provider.submits == []
    assert calls == []


def test_preflight_paid_timing_and_quality_require_the_provider_key(monkeypatch):
    """The paid preflights fail closed on an absent key and pass on a present one."""

    def missing() -> str:
        raise RuntimeError("missing key")

    monkeypatch.setattr(config, "read_groq_key", missing)
    monkeypatch.setattr(config, "read_xai_key", missing)
    timing = native_generation.NativeTimingOptions(
        provider="groq-whisper",
        model=None,
        device="cpu",
        compute="int8",
        language="ru",
        word_timestamps=False,
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_paid_timing(timing)
    assert excinfo.value.error_code == "NATIVE_PAID_TIMING_KEY_MISSING"
    assert excinfo.value.code == 20

    quality = native_generation.NativeQualityOptions(
        provider="xai-stt",
        model=None,
        device="cpu",
        compute="auto",
        runtime="auto",
        language=None,
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_paid_quality(quality)
    assert excinfo.value.error_code == "NATIVE_PAID_QUALITY_KEY_MISSING"
    assert excinfo.value.code == 20

    monkeypatch.setattr(config, "read_groq_key", lambda: "gsk")
    monkeypatch.setattr(config, "read_xai_key", lambda: "xai")
    native_generation._preflight_paid_timing(timing)
    native_generation._preflight_paid_quality(quality)


# ── per-turn paid xAI dialogue quality ───────────────────────────────────────


def _dialogue_turn_texts() -> dict[str, str]:
    return {
        "turn_0001.mp3": "Первая реплика.",
        "turn_0002.mp3": "Вторая реплика.",
        "turn_0003.mp3": "Третья реплика.",
        "turn_0004.mp3": "Четвёртая реплика.",
    }


def test_native_dialogue_paid_quality_pass_links_each_turn_verdict(
    tmp_path, monkeypatch, capsys, paid_env
):
    """One paid TTS per turn plus one paid quality POST per turn, linked privately."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    posts = _install_xai(monkeypatch, _dialogue_turn_texts())
    script = _dialogue_script(
        tmp_path,
        [
            ["Host: Первая реплика.", "Guest: Вторая реплика."],
            ["Host: Третья реплика.", "Guest: Четвёртая реплика."],
        ],
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "dialogue-paid-quality",
            provider="openrouter-tts",
            model=GEMINI_TTS_MODEL,
            voice=None,
            extra=_QUALITY_ARGS,
        ),
    )

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert [chunk for chunk, _voice in provider.calls] == [
        "turn_0001",
        "turn_0002",
        "turn_0003",
        "turn_0004",
    ]
    assert posts == ["turn_0001.mp3", "turn_0002.mp3", "turn_0003.mp3", "turn_0004.mp3"]
    run_uuid = _run_uuid("dialogue-paid-quality")
    children = _quality_children(paid_env, run_uuid)
    assert len(children) == 4
    for child in children:
        assert child["status"] == "completed"
    # The transcript never reaches the machine output.
    assert "Первая реплика." not in json.dumps(payload)
    # Every turn verdict is linked to the parent turn with a private transcript.
    verdicts = _rows(
        paid_env,
        "SELECT artifacts.media_metadata_json AS metadata, text_sources.content AS transcript "
        "FROM artifacts JOIN text_sources ON text_sources.artifact_uuid = artifacts.artifact_uuid "
        "WHERE artifacts.run_uuid = ? AND artifacts.role = 'tts_turn_quality_receipt'",
        (run_uuid,),
    )
    assert len(verdicts) == 4
    assert all(json.loads(row["metadata"])["quality_passed"] is True for row in verdicts)
    assert sorted(row["transcript"] for row in verdicts) == sorted(_dialogue_turn_texts().values())


def test_native_dialogue_paid_quality_failure_blocks_next_turn_and_is_durable(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A failed turn blocks the next paid TTS turn and is re-reported with no POST."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    texts = _dialogue_turn_texts()
    texts["turn_0002.mp3"] = "Совсем другой текст."
    posts = _install_xai(monkeypatch, texts)
    script = _dialogue_script(
        tmp_path,
        [
            ["Host: Первая реплика.", "Guest: Вторая реплика."],
            ["Host: Третья реплика.", "Guest: Четвёртая реплика."],
        ],
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "dialogue-paid-fail",
            provider="openrouter-tts",
            model=GEMINI_TTS_MODEL,
            voice=None,
            extra=_QUALITY_ARGS,
        ),
    )

    assert code == 60
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    # The next paid TTS turn never ran; only turns 1 and 2 were synthesized.
    assert [chunk for chunk, _voice in provider.calls] == ["turn_0001", "turn_0002"]
    assert posts == ["turn_0001.mp3", "turn_0002.mp3"]
    run_uuid = _run_uuid("dialogue-paid-fail")
    failing = _rows(
        paid_env,
        "SELECT media_metadata_json FROM artifacts WHERE run_uuid = ? "
        "AND role = 'tts_turn_quality_receipt'",
        (run_uuid,),
    )
    assert len(failing) == 2
    assert any(json.loads(row["media_metadata_json"])["quality_passed"] is False for row in failing)

    # Resume re-reports the durable FAIL with no provider call at all.
    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_posts = _install_xai(monkeypatch, texts)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 60
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    assert retry_posts == []

    # Sync re-reports the same recorded failure without running any model.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    assert retry_posts == []


def test_native_dialogue_paid_quality_unknown_submit_blocks_resume(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A lost paid quality POST leaves a marker that no resume may retry."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)
    _install_xai(
        monkeypatch,
        _dialogue_turn_texts(),
        fail=requests.Timeout("read timed out"),
    )
    script = _dialogue_script(tmp_path, [["Host: Первая реплика."]])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "dialogue-paid-lost",
            provider="openrouter-tts",
            model=GEMINI_TTS_MODEL,
            voice=None,
            extra=_QUALITY_ARGS,
        ),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_ASR_FAILED"
    assert [chunk for chunk, _voice in provider.calls] == ["turn_0001"]
    run_uuid = _run_uuid("dialogue-paid-lost")
    children = _quality_children(paid_env, run_uuid)
    assert len(children) == 1
    assert children[0]["status"] != "completed"

    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_posts = _install_xai(monkeypatch, _dialogue_turn_texts())
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert retry_posts == []
    assert [chunk for chunk, _voice in provider.calls] == ["turn_0001"]


def test_native_dialogue_paid_quality_replays_saved_body_without_second_post(
    tmp_path, monkeypatch, capsys, paid_env
):
    """A per-turn response saved before a crash is replayed locally on resume."""
    provider = FakeDialogueProvider()
    _install_provider(monkeypatch, provider)

    class CrashAfterBody(FakeXaiStt):
        def transcribe(
            self,
            audio_path,
            language="ru",
            word_timestamps=False,
            quiet=False,
            on_raw_response=None,
        ):
            name = Path(audio_path).name
            FakeXaiStt.posts.append(name)
            text = FakeXaiStt.texts[name]
            body = json.dumps(
                {
                    "text": text,
                    "language": language,
                    "duration": 1.0,
                    "words": [{"text": text, "start": 0.0, "end": 1.0, "confidence": 0.9}],
                }
            ).encode()
            on_raw_response(body, "application/json")
            raise requests.Timeout("read timed out")

    FakeXaiStt.reset()
    FakeXaiStt.texts = _dialogue_turn_texts()
    monkeypatch.setattr(xai_stt_module, "XAISttProvider", CrashAfterBody)
    script = _dialogue_script(tmp_path, [["Host: Первая реплика."]])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "dialogue-paid-replay",
            provider="openrouter-tts",
            model=GEMINI_TTS_MODEL,
            voice=None,
            extra=_QUALITY_ARGS,
        ),
    )
    assert code == 50

    run_uuid = _run_uuid("dialogue-paid-replay")
    posts_before = list(FakeXaiStt.posts)
    monkeypatch.setattr(cli, "build_provider", _explode)
    retry_posts = _install_xai(monkeypatch, _dialogue_turn_texts())
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert retry_posts == []
    assert posts_before == ["turn_0001.mp3"]
    assert _quality_children(paid_env, run_uuid)[0]["status"] == "completed"
