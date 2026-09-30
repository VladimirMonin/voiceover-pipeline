"""End-to-end contract tests for the native ``--no-trim`` / local-quality slice.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, fake in-memory TTS and local ASR providers, and patched FFmpeg/concat
seams. No real provider, network call, API key, ``.env``, model, or download is
used. The tests assert paid-submit counts, the recorded trimming semantics, linked
verification history, and the partial-result behavior rather than only statuses,
so a slice that silently re-submits TTS, ignores ``--no-trim`` on resume, trims
against the recorded setting, loses the paid audio on a quality failure, or leaks
the private verification transcript fails here.
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
from voiceover_pipeline.models import (
    ASRExecutionReceipt,
    ASRResult,
    SynthesisResult,
)
from voiceover_pipeline.services import native_generation, transcription

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
POLZA_SYNC_MODEL = "openai/gpt-4o-mini-tts"
QWEN_MODEL = "Qwen/Qwen3-ASR-0.6B"
QUALITY_ARGS = ("--tts-quality-provider", "qwen-local")


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


class FakeSyncProvider:
    """Offline stand-in for a synchronous (non-media) TTS submit."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.calls.append(chunk_id)
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


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


@pytest.fixture
def trim_calls() -> list[str]:
    """Names of the files each run asked the silence-trim seam to trim."""
    return []


@pytest.fixture
def native_env(tmp_path, monkeypatch, trim_calls):
    """Point history at a temp home and record every local media seam call."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat)

    def recording_trim(_ffmpeg: str, _ffprobe: str, path: Path) -> None:
        # The trim is observable in the bytes, so the default and ``--no-trim``
        # runs cannot look identical through a no-op seam.
        trim_calls.append(path.name)
        path.write_bytes(path.read_bytes() + b"-trimmed")

    monkeypatch.setattr(cli, "trim_final_silence", recording_trim)
    # The preflight and the model run are replaced per test; a no-op preflight is
    # the default so only the explicit preflight test exercises the dependency
    # probe (this environment has no local ASR install).
    monkeypatch.setattr(native_generation, "_preflight_local_quality", lambda _options: None)
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


def _install_quality(monkeypatch, transcript: str, error: Exception | None = None):
    calls: list[dict] = []

    def fake_transcribe(**_kwargs):
        calls.append(dict(_kwargs))
        if error is not None:
            raise error
        return _asr_result(transcript)

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", fake_transcribe)
    return calls


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


# ── recorded trimming semantics ───────────────────────────────────────────────


def test_native_no_trim_skips_trim_and_default_trim_still_runs(
    tmp_path, monkeypatch, capsys, native_env, trim_calls
):
    """``--no-trim`` keeps the provider silence; the default still trims."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Обрезка по умолчанию.", "Без обрезки."])

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-trim-default")
    )
    assert code == 0
    assert len(trim_calls) == 2
    default_chunk = tmp_path / "out" / "native-trim-default" / "chunks" / "chunk_01.mp3"
    assert default_chunk.read_bytes() == b"chunk_01-audio-trimmed"

    trim_calls.clear()
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-no-trim", extra=["--no-trim"]),
    )
    assert code == 0
    kept_chunk = tmp_path / "out" / "native-no-trim" / "chunks" / "chunk_01.mp3"
    assert kept_chunk.read_bytes() == b"chunk_01-audio"
    assert trim_calls == []
    # ``--no-trim`` must not silently fall out of canonical history: the run is
    # native-owned and records the trimming semantics a resume has to repeat.
    run_id_root = tmp_path / "out" / "native-no-trim"
    assert (run_id_root / ".voiceover-native-history.json").is_file()
    run_uuid = _run_uuid("native-no-trim")
    snapshot = json.loads(
        _rows(
            native_env,
            "SELECT config_snapshot FROM runs WHERE run_uuid = ?",
            (run_uuid,),
        )[0]["config_snapshot"]
    )
    assert snapshot["output"]["trim_final_silence"] is False


def test_native_resume_rejects_changed_trim_flag_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A resume must repeat the recorded trimming flag, not the current default."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Прогон без обрезки."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-trim-resume", extra=["--no-trim"]),
    )
    assert code == 0
    assert provider.submits == ["chunk_01"]

    monkeypatch.setattr(cli, "build_provider", _explode)
    for extra in (["--resume"], ["--resume", "--no-trim"]):
        code, payload = _json_run(
            monkeypatch,
            capsys,
            _generate_argv(tmp_path, script, "native-trim-resume", extra=extra),
        )
        assert code == (30 if extra == ["--resume"] else 0), payload
        if code == 30:
            assert payload["details"]["error_code"] == "NATIVE_PROCESSING_UNSUPPORTED"
    assert provider.submits == ["chunk_01"]


def test_output_options_record_trim_and_quality_settings():
    """The recorded output options round-trip and a malformed block fails closed."""
    quality = native_generation.NativeQualityOptions(
        provider="qwen-local",
        model=None,
        device="cpu",
        compute="auto",
        runtime="auto",
        language=None,
    )
    recorded = native_generation.build_output_options(True, quality=quality)
    assert recorded["trim_final_silence"] is False
    assert native_generation.quality_options_from_output_options(recorded) == quality
    # A run without quality keeps its original output options.
    base = native_generation.build_output_options(False)
    assert "quality_enabled" not in base
    assert native_generation.quality_options_from_output_options(base) is None
    # A malformed quality block is refused rather than silently dropped.
    malformed = dict(recorded)
    malformed["quality_runtime"] = 5
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation.quality_options_from_output_options(malformed)
    assert excinfo.value.error_code == "NATIVE_QUALITY_OPTIONS_INVALID"


def test_transcribe_local_asr_quality_reuses_the_registered_asr_path(monkeypatch, tmp_path):
    """The adapter uses the existing spec/provider/validation path, not a new one."""
    from voiceover_pipeline.models import ASRCapabilities
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec

    class FixtureQwenProvider:
        def transcribe(self, request):
            assert request.timestamp_mode == "none"
            assert request.model_id == QWEN_MODEL
            return _asr_result("Проверенный текст.")

    spec = ASRProviderSpec(
        provider_id="qwen-local",
        description="Offline fixture provider",
        factory=FixtureQwenProvider,
        models=({"id": QWEN_MODEL, "default": True},),
        capabilities=ASRCapabilities(batch_audio=True),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(transcription, "get_asr_provider_spec", lambda _provider_id: spec)
    monkeypatch.setattr(
        "voiceover_pipeline.providers.qwen_asr_local.qwen_asr_python_dependency_probe",
        lambda _model_id: ASRDependencyHealth(available=True, remediation=""),
    )
    # The adapter reuses the long-form orchestration of ``transcribe``, which needs
    # a real duration probe for this provider; the passthrough keeps the unit test
    # offline while still exercising the spec/provider/validation path.
    from voiceover_pipeline import asr_longform

    monkeypatch.setattr(
        asr_longform,
        "transcribe_prerecorded_long_form",
        lambda provider, request: provider.transcribe(request),
    )
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")

    result = transcription.transcribe_local_asr_quality(
        provider_id="qwen-local",
        audio_path=audio_path,
        model=None,
        device="cpu",
        compute="auto",
        runtime="auto",
        language="ru",
    )

    assert result.transcript == "Проверенный текст."
    assert result.model_id == QWEN_MODEL


# ── fresh run: local quality verification after the paid audio ───────────────


def test_native_local_quality_pass_links_a_private_verification(
    tmp_path, monkeypatch, capsys, native_env
):
    """One command pays once, verifies locally, and links a private transcript."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    transcript = "Первый фрагмент текста. Второй фрагмент текста."
    quality_calls = _install_quality(monkeypatch, transcript)
    script = _script(tmp_path, ["Первый фрагмент текста.", "Второй фрагмент текста."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-quality", extra=QUALITY_ARGS)
    )

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert provider.submits == ["chunk_01", "chunk_02"]
    assert len(quality_calls) == 1
    # The local check ran against the committed final audio, without timestamps.
    assert quality_calls[0]["provider_id"] == "qwen-local"
    checked_audio = quality_calls[0]["audio_path"]
    assert checked_audio.parent == tmp_path / "out" / "native-quality"
    assert checked_audio.is_file()

    run_uuid = _run_uuid("native-quality")
    children = _quality_children(native_env, run_uuid)
    assert len(children) == 1
    child = children[0]
    assert child["status"] == "completed"
    texts = _rows(
        native_env,
        "SELECT kind, content FROM text_sources WHERE run_uuid = ?",
        (child["run_uuid"],),
    )
    assert [(row["kind"], row["content"]) for row in texts] == [
        ("verification_transcript", transcript)
    ]
    artifacts = _rows(
        native_env, "SELECT role FROM artifacts WHERE run_uuid = ?", (child["run_uuid"],)
    )
    assert {row["role"] for row in artifacts} == {"asr_source_audio"}
    snapshot = json.loads(child["config_snapshot"])
    assert snapshot["quality_passed"] is True
    assert snapshot["expected_text_persisted"] is False
    # The TTS run keeps its own script text and gains no verification transcript.
    tts_texts = _rows(
        native_env,
        "SELECT kind FROM text_sources WHERE run_uuid = ? AND part_uuid IS NULL",
        (run_uuid,),
    )
    assert "verification_transcript" not in {row["kind"] for row in tts_texts}
    # The private transcript reaches neither the machine output nor the exports.
    assert transcript not in json.dumps(payload)
    run_root = tmp_path / "out" / "native-quality"
    exported = [
        run_root / "run_state.json",
        Path(payload["files"]["run_json"]),
        Path(payload["files"]["manifest_json"]),
        Path(payload["files"]["chunks_json"]),
    ]
    for path in exported:
        assert path.is_file(), path
        exported_text = path.read_text(encoding="utf-8")
        assert "verification_transcript" not in exported_text
        assert transcript not in exported_text


def test_native_sync_route_with_local_quality_pays_once_and_links_verification(
    tmp_path, monkeypatch, capsys, native_env
):
    """The synchronous polza route verifies its inline audio with one paid POST."""
    provider = FakeSyncProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "Синхронный проверенный фрагмент.")
    script = _script(tmp_path, ["Синхронный проверенный фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "native-quality-sync-route",
            model=POLZA_SYNC_MODEL,
            extra=QUALITY_ARGS,
        ),
    )

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert provider.calls == ["chunk_01"]
    run_uuid = _run_uuid("native-quality-sync-route")
    assert len(_quality_children(native_env, run_uuid)) == 1


def test_native_local_quality_mismatch_keeps_paid_audio_and_cost(
    tmp_path, monkeypatch, capsys, native_env
):
    """A quality mismatch reports exit 60 and records the FAIL durably."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    observed = "Совсем другой произнесённый текст."
    _install_quality(monkeypatch, observed)
    script = _script(tmp_path, ["Ожидаемый текст фрагмента."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-fail", extra=QUALITY_ARGS),
    )

    assert code == 60
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    assert provider.submits == ["chunk_01"]
    run_uuid = _run_uuid("native-quality-fail")
    run_root = tmp_path / "out" / "native-quality-fail"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio-trimmed"
    assert list(run_root.glob("*.mp3"))
    # The paid raw bytes and their bounded receipt survive the failed check.
    raw_dir = run_root / "raw"
    assert (raw_dir / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert (raw_dir / "chunk_01.mp3.receipt.json").is_file()
    status = _rows(native_env, "SELECT status FROM runs WHERE run_uuid = ?", (run_uuid,))[0]
    assert status["status"] == "completed"
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]
    # The observed FAIL is durable: exactly one linked verify child records the
    # content-free verdict beside the private transcript, never the expected text.
    children = _quality_children(native_env, run_uuid)
    assert len(children) == 1
    child = children[0]
    assert child["status"] == "completed"
    snapshot = json.loads(child["config_snapshot"])
    assert snapshot["quality_passed"] is False
    assert isinstance(snapshot["similarity"], float)
    assert snapshot["failure_reasons"]
    assert set(snapshot["failure_reasons"]) <= {
        "similarity",
        "missing_words",
        "unexpected_words",
        "repeated_ngram",
    }
    assert snapshot["expected_text_persisted"] is False
    texts = _rows(
        native_env,
        "SELECT kind, content FROM text_sources WHERE run_uuid = ?",
        (child["run_uuid"],),
    )
    assert [(row["kind"], row["content"]) for row in texts] == [
        ("verification_transcript", observed)
    ]
    artifacts = _rows(
        native_env, "SELECT role FROM artifacts WHERE run_uuid = ?", (child["run_uuid"],)
    )
    assert {row["role"] for row in artifacts} == {"asr_source_audio"}
    # The transcript and the expected script stay out of the machine error payload.
    serialized = json.dumps(payload)
    assert observed not in serialized
    assert "Ожидаемый текст фрагмента." not in serialized


def test_persisted_quality_failure_is_reused_without_another_run(
    tmp_path, monkeypatch, capsys, native_env
):
    """A stored FAIL is re-reported by resume/sync with no TTS POST or model run."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    observed = "Совсем другой произнесённый текст."
    quality_calls = _install_quality(monkeypatch, observed)
    script = _script(tmp_path, ["Ожидаемый текст фрагмента."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-durable", extra=QUALITY_ARGS),
    )
    assert code == 60
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"
    assert len(quality_calls) == 1
    run_uuid = _run_uuid("native-quality-durable")
    assert len(_quality_children(native_env, run_uuid)) == 1

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a persisted failure must not rerun the local model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    # ``generate --resume`` repeats the recorded failure: no second TTS POST.
    resume_argv = _generate_argv(
        tmp_path, script, "native-quality-durable", extra=(*QUALITY_ARGS, "--resume")
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"

    # ``history resume ID`` re-reports the same failure from the snapshot alone.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 60, payload
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_FAILED"

    # ``history sync ID`` runs no model and reports the persisted verdict instead.
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": False}

    assert provider.submits == ["chunk_01"]
    assert len(quality_calls) == 1
    assert len(_quality_children(native_env, run_uuid)) == 1
    # The private transcript and the expected script never reach the exports.
    run_root = tmp_path / "out" / "native-quality-durable"
    exported = [
        run_root / "run_state.json",
        Path(payload["files"]["run_json"]),
        Path(payload["files"]["chunks_json"]),
        Path(payload["files"]["manifest_json"]),
    ]
    for path in exported:
        assert path.is_file(), path
        exported_text = path.read_text(encoding="utf-8")
        assert observed not in exported_text
        assert "verification_transcript" not in exported_text


def test_quality_failure_persistence_failure_reports_partial_not_a_durable_fail(
    tmp_path, monkeypatch, capsys, native_env
):
    """A failed write after an observed FAIL reports exit 50, claiming no verdict."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    observed = "Совсем другой произнесённый текст."
    _install_quality(monkeypatch, observed)

    def boom(_save):
        raise RuntimeError("history database unavailable")

    monkeypatch.setattr(native_generation, "persist_asr_history", boom)
    script = _script(tmp_path, ["Ожидаемый текст фрагмента."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-fail-db", extra=QUALITY_ARGS),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_HISTORY_FAILED"
    serialized = json.dumps(payload)
    assert observed not in serialized
    assert "Ожидаемый текст фрагмента." not in serialized
    run_uuid = _run_uuid("native-quality-fail-db")
    run_root = tmp_path / "out" / "native-quality-fail-db"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio-trimmed"
    assert (run_root / "raw" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]
    assert _quality_children(native_env, run_uuid) == []


def test_native_local_quality_transcription_failure_then_resume_repairs(
    tmp_path, monkeypatch, capsys, native_env
):
    """A local transcription failure keeps the paid audio; resume repairs it."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "", error=RuntimeError("qwen boom"))
    script = _script(tmp_path, ["Оплаченный фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-asr", extra=QUALITY_ARGS),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_ASR_FAILED"
    run_uuid = _run_uuid("native-quality-asr")
    run_root = tmp_path / "out" / "native-quality-asr"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio-trimmed"
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]
    assert _quality_children(native_env, run_uuid) == []

    # Resume repairs only the missing local verification: zero additional TTS POSTs.
    _install_quality(monkeypatch, "Оплаченный фрагмент.")
    monkeypatch.setattr(cli, "build_provider", _explode)
    resume_argv = _generate_argv(
        tmp_path, script, "native-quality-asr", extra=(*QUALITY_ARGS, "--resume")
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0, payload
    assert payload["quality"] == {"complete": True, "passed": True}
    assert provider.submits == ["chunk_01"]
    assert len(_quality_children(native_env, run_uuid)) == 1
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]


def test_native_local_quality_history_failure_reports_partial(
    tmp_path, monkeypatch, capsys, native_env
):
    """A linked-history write failure keeps the audio and cost, exit 50."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "Сбой записи проверки.")

    def boom(_save):
        raise RuntimeError("history database unavailable")

    monkeypatch.setattr(native_generation, "persist_asr_history", boom)
    script = _script(tmp_path, ["Сбой записи проверки."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-db", extra=QUALITY_ARGS),
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_HISTORY_FAILED"
    run_uuid = _run_uuid("native-quality-db")
    run_root = tmp_path / "out" / "native-quality-db"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio-trimmed"
    assert list(run_root.glob("*.mp3"))
    assert _tts_attempt_costs(native_env, run_uuid) == ["0.3"]
    assert _quality_children(native_env, run_uuid) == []


# ── completed resume / sync reuse the linked verification ────────────────────


def test_native_completed_quality_resume_and_sync_do_not_repeat_the_check(
    tmp_path, monkeypatch, capsys, native_env
):
    """A completed verified run re-exports without a model call or a second row."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "Готовый проверенный прогон.")
    script = _script(tmp_path, ["Готовый проверенный прогон."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-done", extra=QUALITY_ARGS),
    )
    assert code == 0
    run_uuid = _run_uuid("native-quality-done")
    assert len(_quality_children(native_env, run_uuid)) == 1

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a completed verified run must not run the local model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)

    for verb in ("resume", "sync"):
        code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, verb))
        assert code == 0, payload
        assert payload["quality"] == {"complete": True, "passed": True}

    assert len(_quality_children(native_env, run_uuid)) == 1
    assert provider.submits == ["chunk_01"]


# ── history resume / sync without the source script ──────────────────────────


def test_history_resume_without_script_completes_pending_quality(
    tmp_path, monkeypatch, capsys, native_env
):
    """A verified run with a failed check is repaired from its snapshot, no script."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "", error=RuntimeError("first quality boom"))
    script = _script(tmp_path, ["Первый.", "Второй."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-hist", extra=QUALITY_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("native-quality-hist")
    script.unlink()

    _install_quality(monkeypatch, "Первый. Второй.")
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 0, payload
    assert payload["mode"] == "resume"
    assert payload["quality"] == {"complete": True, "passed": True}
    assert len(_quality_children(native_env, run_uuid)) == 1
    assert provider.submits == ["chunk_01", "chunk_02"]


def test_history_sync_reports_quality_pending_and_never_runs_asr(
    tmp_path, monkeypatch, capsys, native_env
):
    """Sync repairs exports only; a pending verification stays incomplete."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "", error=RuntimeError("first quality boom"))
    script = _script(tmp_path, ["Фрагмент для синхронизации."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-sync", extra=QUALITY_ARGS),
    )
    assert code == 50
    run_uuid = _run_uuid("native-quality-sync")

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("history sync must not run the local quality model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert payload["quality"] == {"complete": False, "passed": None}
    assert _quality_children(native_env, run_uuid) == []
    assert provider.submits == ["chunk_01"]


# ── local model availability preflight ───────────────────────────────────────


def test_native_quality_model_unavailable_blocks_before_paid_submit(
    tmp_path, monkeypatch, capsys, native_env
):
    """An unavailable local model stops the command before the paid TTS POST."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    quality_calls = _install_quality(monkeypatch, "не должно выполниться")

    def blocked(_options):
        raise native_generation.NativeGenerationError(
            "the local Qwen ASR model is not installed.",
            code=10,
            error_code="NATIVE_QUALITY_MODEL_UNAVAILABLE",
        )

    monkeypatch.setattr(native_generation, "_preflight_local_quality", blocked)
    script = _script(tmp_path, ["Платный POST не должен случиться."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-preflight", extra=QUALITY_ARGS),
    )

    assert code == 10
    assert payload["details"]["error_code"] == "NATIVE_QUALITY_MODEL_UNAVAILABLE"
    assert provider.submits == []
    assert quality_calls == []
    chunks_dir = tmp_path / "out" / "native-quality-preflight" / "chunks"
    assert not list(chunks_dir.glob("chunk_*.mp3"))


def _local_quality_options() -> native_generation.NativeQualityOptions:
    return native_generation.NativeQualityOptions(
        provider="qwen-local",
        model=None,
        device="cpu",
        compute="auto",
        runtime="auto",
        language=None,
    )


def _fixture_spec(*, available: bool):
    from voiceover_pipeline.models import ASRCapabilities
    from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec

    return ASRProviderSpec(
        provider_id="qwen-local",
        description="Offline fixture provider",
        factory=lambda: None,
        models=({"id": QWEN_MODEL, "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True, device_modes=("cpu",), compute_modes=("auto",)
        ),
        dependency_probe=lambda: ASRDependencyHealth(
            available=available,
            remediation="" if available else "Qwen local assets are unavailable.",
        ),
    )


def test_preflight_local_quality_rejects_cloud_provider(monkeypatch):
    """A cloud ASR provider keeps the legacy executor instead of the local check."""
    cloud = native_generation.NativeQualityOptions(
        provider="xai-stt",
        model=None,
        device="cpu",
        compute="auto",
        runtime="auto",
        language=None,
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_local_quality(cloud)
    assert excinfo.value.error_code == "NATIVE_QUALITY_PROVIDER_UNSUPPORTED"
    assert excinfo.value.code == 30


def test_preflight_local_quality_reports_unavailable_local_model(monkeypatch):
    """An absent or uninstalled local model fails closed without a download."""
    from voiceover_pipeline.providers import asr_registry

    monkeypatch.setattr(
        asr_registry, "get_asr_provider_spec", lambda _provider_id: _fixture_spec(available=False)
    )
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_local_quality(_local_quality_options())
    assert excinfo.value.error_code == "NATIVE_QUALITY_MODEL_UNAVAILABLE"
    assert excinfo.value.code == 10


def test_preflight_local_quality_accepts_an_installed_local_provider(monkeypatch):
    """An installed local provider passes the offline preflight without a call."""
    from voiceover_pipeline.providers import asr_registry

    monkeypatch.setattr(
        asr_registry, "get_asr_provider_spec", lambda _provider_id: _fixture_spec(available=True)
    )
    native_generation._preflight_local_quality(_local_quality_options())


def test_preflight_local_quality_rejects_an_unregistered_provider(monkeypatch):
    """A provider the registry cannot resolve is refused before any paid submit."""
    from voiceover_pipeline.providers import asr_registry

    def missing(provider_id: str):
        raise asr_registry.ASRProviderNotFoundError(provider_id)

    monkeypatch.setattr(asr_registry, "get_asr_provider_spec", missing)
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation._preflight_local_quality(_local_quality_options())
    assert excinfo.value.error_code == "NATIVE_QUALITY_PROVIDER_UNSUPPORTED"
    assert excinfo.value.code == 2


# ── paid evidence stays protected ────────────────────────────────────────────


def test_native_quality_run_blocks_unconfirmed_paid_submit_without_asr(
    tmp_path, monkeypatch, capsys, native_env
):
    """An unconfirmed paid submit still blocks, with zero TTS or ASR requests."""
    provider = FakeMediaProvider()

    def fail_before_accept(_inner, _text, _chunk_id):
        raise requests.Timeout("read timed out")

    provider.script = fail_before_accept
    _install_provider(monkeypatch, provider)
    quality_calls = _install_quality(monkeypatch, "не должно выполниться")
    script = _script(tmp_path, ["Неопределённый платный исход."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-unconfirmed", extra=QUALITY_ARGS),
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]
    assert quality_calls == []

    monkeypatch.setattr(cli, "build_provider", _explode)
    resume_argv = _generate_argv(
        tmp_path, script, "native-quality-unconfirmed", extra=(*QUALITY_ARGS, "--resume")
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert provider.submits == ["chunk_01"]
    assert quality_calls == []


def test_history_sync_with_quality_recovers_media_by_get_only(
    tmp_path, monkeypatch, capsys, native_env
):
    """Sync finishes a known Media id with GET calls and never runs the local model."""
    provider = FakeMediaProvider()

    def accept_then_fail(inner, text, chunk_id):
        inner.on_media_task_accepted(f"task-{chunk_id}")
        raise requests.Timeout("poll timed out")

    provider.script = accept_then_fail
    _install_provider(monkeypatch, provider)
    _install_quality(monkeypatch, "Фрагмент для GET.")
    script = _script(tmp_path, ["Фрагмент для GET."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-quality-get", extra=QUALITY_ARGS),
    )
    assert code == 30
    run_uuid = _run_uuid("native-quality-get")

    sync_provider = FakeMediaProvider()
    _install_provider(monkeypatch, sync_provider)

    def explode_quality(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("history sync must not run the local quality model")

    monkeypatch.setattr(transcription, "transcribe_local_asr_quality", explode_quality)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert sync_provider.submits == []
    assert sync_provider.recovers == ["task-chunk_01"]
    assert payload["quality"] == {"complete": False, "passed": None}
    assert _quality_children(native_env, run_uuid) == []
