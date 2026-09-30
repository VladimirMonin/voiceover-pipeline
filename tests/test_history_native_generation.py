"""End-to-end contract tests for the native DB-first Polza Media TTS slice.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, a fake in-memory media provider, and patched FFmpeg/concat seams. No
real provider, network call, API key, ``.env``, model, or paid request is used.
The tests assert paid-submit counts and committed history rather than only
statuses, so a slice that silently re-submits or loses evidence fails here.
"""

import hashlib
import json
import sqlite3
import sys
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.models import ScriptChunk, SynthesisResult
from voiceover_pipeline.providers import PolzaTTSProvider

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
POLZA_SYNC_MODEL = "openai/gpt-4o-mini-tts"


class FakeMediaProvider:
    """Offline stand-in for the Polza ElevenLabs ``/media`` provider.

    ``on_media_task_accepted`` and ``on_media_completed`` are bound by the native
    executor before any submit, so the default path exercises the real DB
    transitions. ``script`` lets a test replace one call, for example to raise
    after acceptance or before it.
    """

    def __init__(self) -> None:
        self.on_media_task_accepted = None
        self.on_media_completed = None
        self.submits: list[str] = []
        self.recovers: list[str] = []
        self.script = None

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
    """Offline stand-in for a synchronous (non-media) TTS submit.

    ``usage`` seeds the Polza ``/audio/speech`` ``usage_direct`` metadata so the
    exact-cost path is exercised; ``script`` lets a test replace one call, for
    example to raise before or after a response.
    """

    def __init__(self, usage: dict | None = None) -> None:
        self.calls: list[str] = []
        self.usage = usage
        self.script = None

    def _result(self, text: str, chunk_id: str) -> SynthesisResult:
        raw_metadata: dict[str, object] = {}
        if self.usage is not None:
            raw_metadata["usage_direct"] = self.usage
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
            raw_metadata=raw_metadata,
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.calls.append(chunk_id)
        if self.script is not None:
            return self.script(self, text, chunk_id)
        return self._result(text, chunk_id)


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


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


def _run(monkeypatch, argv) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, ""


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    code, _ = _run(monkeypatch, argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def _install_provider(monkeypatch, provider):
    builds: list[str] = []

    def fake_build(*_args, **_kwargs):
        builds.append("built")
        return provider

    monkeypatch.setattr(cli, "build_provider", fake_build)
    return builds


def _patch_legacy_finalization(monkeypatch) -> None:
    """Replace the legacy-only cost enrichment and concat seams."""
    monkeypatch.setattr(
        cli,
        "attach_costs",
        lambda _provider, _key, _model, _started, chunks: chunks,
    )
    monkeypatch.setattr(
        cli,
        "concat_mp3_chunks",
        lambda ffmpeg, chunks_dir, output: _concat(
            ffmpeg, sorted(chunks_dir.glob("chunk_*.mp3")), output
        ),
    )


def _history_show(run_uuid: str) -> dict:
    from voiceover_pipeline.commands.history import show_history

    return show_history(run_uuid)


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    runs = list_history()["runs"]
    matches = [run for run in runs if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _history_argv(run_uuid: str, verb: str, *, extra=()) -> list[str]:
    return ["voiceover-pipeline", "history", verb, run_uuid, *extra, "--json"]


def _run_row(home: Path, run_uuid: str) -> tuple[str, int]:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        row = connection.execute(
            "SELECT status, revision FROM runs WHERE run_uuid = ?", (run_uuid,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return row[0], row[1]


def _evidence_counts(home: Path, run_uuid: str) -> dict[str, int]:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        return {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_uuid = ?", (run_uuid,)
            ).fetchone()[0]
            for table in ("parts", "attempts", "artifacts")
        }
    finally:
        connection.close()


def _write_legacy_tree(root: Path, run_id: str = "legacy-run") -> Path:
    """Write one minimal legacy run tree so ``history import`` can adopt it."""
    slug = "openai-gpt-4o-mini-tts"
    source = root / run_id
    chunks_dir = source / "chunks"
    chunks_dir.mkdir(parents=True)
    text = "Legacy text."
    entry = {
        "status": "completed",
        "number": 1,
        "id": "chunk_01",
        "file": "chunk_01.mp3",
        "duration_ms": 1200,
        "text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "cost": 0.1,
        "cost_currency": "RUB",
    }
    (chunks_dir / "chunk_01.mp3").write_bytes(b"ID3chunk")
    full_mp3 = source / f"{run_id}-voiceover-{slug}.mp3"
    full_mp3.write_bytes(b"ID3full")
    run_json = source / f"{run_id}-voiceover-{slug}.json"
    state = {
        "artifact_type": "voiceover-run-state",
        "status": "completed",
        "run_id": run_id,
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
        "voice": "alloy",
        "script": str(source / "missing-script.md"),
        "script_hash": "a" * 64,
        "chunk_count": 1,
        "completed_count": 1,
        "chunks": [entry],
    }
    (source / "run_state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    run_json.write_text(
        json.dumps({**state, "artifact_type": "voiceover-run", "full_mp3": str(full_mp3)}),
        encoding="utf-8",
    )
    (chunks_dir / "chunks.json").write_text(
        json.dumps(
            {
                "artifact_type": "voiceover-chunks",
                "provider": "polza-tts",
                "model": "openai/gpt-4o-mini-tts",
                "voice": "alloy",
                "chunk_count": 1,
                "chunks": [entry],
            }
        ),
        encoding="utf-8",
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "voiceover-production-bundle",
                "run_id": run_id,
                "full_mp3": str(full_mp3),
                "run_json": str(run_json),
                "chunks_json": str(chunks_dir / "chunks.json"),
                "duration_ms": 1200,
            }
        ),
        encoding="utf-8",
    )
    return source


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider or read an API key")


# ── fresh generation ──────────────────────────────────────────────────────────


def test_native_fresh_two_parts_write_history_audio_and_exports(
    tmp_path, monkeypatch, capsys, native_env
):
    """A fresh two-part media run produces audio, canonical history, and exports."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент текста.", "Второй фрагмент текста."])

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-fresh"))

    assert code == 0
    assert payload["status"] == "success"
    assert payload["duration_ms"] == 1000
    assert provider.submits == ["chunk_01", "chunk_02"]

    run_root = tmp_path / "out" / "native-fresh"
    prefix = "native-fresh"
    slug = POLZA_MEDIA_MODEL.replace("/", "-")
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"
    assert (run_root / f"{prefix}-voiceover-{slug}.mp3").exists()
    for name in ("run_state.json", "manifest.json", f"{prefix}-voiceover-{slug}.json"):
        assert (run_root / name).exists(), name
    assert (run_root / "chunks" / "chunks.json").exists()
    assert (run_root / ".voiceover-native-history.json").exists()

    # The compatibility state is a projection: completed, no paid permission.
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert "pending_attempt" not in state
    assert state["native_history"]["history_run_uuid"]
    assert state["history_run_uuid"] == state["native_history"]["history_run_uuid"]

    run_uuid = _run_uuid("native-fresh")
    detail = _history_show(run_uuid)
    assert detail["run"]["status"] == "completed"
    assert detail["run"]["operation"] == "tts"
    assert len(detail["parts"]) == 2
    assert len(detail["attempts"]) == 2
    roles = {artifact["role"] for artifact in detail["artifacts"]}
    assert {"paid_raw_audio", "chunk_audio", "final_audio"} <= roles
    assert sum(1 for art in detail["artifacts"] if art["role"] == "chunk_audio") == 2
    assert all(attempt["status"] == "completed" for attempt in detail["attempts"])
    assert all(attempt["cost"]["amount"] == "0.3" for attempt in detail["attempts"])
    # No signed URL or secret reached the machine output.
    assert "cdn.example.com" not in capsys.readouterr().out


def test_native_fresh_route_never_calls_legacy_pricing_or_key_early(
    tmp_path, monkeypatch, capsys, native_env
):
    """The provider is built only when a paid submit is actually needed."""
    provider = FakeMediaProvider()
    builds = _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Один фрагмент."])

    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-lazy"))

    assert code == 0
    assert builds == ["built"]
    assert provider.submits == ["chunk_01"]


# ── synchronous Polza /audio/speech and OpenRouter routes ─────────────────────


def _chunk_costs(history_db: Path) -> dict[int, str | None]:
    """Read the committed per-part cost column from the open history database."""
    connection = sqlite3.connect(history_db)
    try:
        rows = connection.execute(
            "SELECT p.position, a.cost FROM attempts a "
            "JOIN parts p ON p.part_uuid = a.part_uuid ORDER BY p.position"
        ).fetchall()
    finally:
        connection.close()
    return {position: cost for position, cost in rows}


def _artifact_media_metadata(history_db: Path) -> dict[str, dict]:
    """Read each linked artifact's metadata JSON from the open history database."""
    connection = sqlite3.connect(history_db)
    try:
        rows = connection.execute("SELECT role, media_metadata_json FROM artifacts").fetchall()
    finally:
        connection.close()
    return {role: json.loads(metadata) for role, metadata in rows if metadata is not None}


def test_native_sync_polza_two_parts_persist_exact_cost_before_conversion(
    tmp_path, monkeypatch, capsys, native_env
):
    """A fresh sync Polza run writes the exact observed cost before every FFmpeg step."""
    provider = FakeSyncProvider(usage={"cost_rub": Decimal("0.3")})
    _install_provider(monkeypatch, provider)
    monkeypatch.setattr(
        cli,
        "attach_costs",
        lambda *_args, **_kwargs: pytest.fail("the native route must not GET an enriched cost"),
    )
    history_db = native_env / "history.sqlite3"
    seen: list[dict[int, str | None]] = []

    def record_then_write(_ffmpeg, data, _fmt, path):
        # Snapshot the committed cost column right before each chunk is converted,
        # so the amount must already be durable when FFmpeg starts.
        seen.append(_chunk_costs(history_db))
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", record_then_write)
    script = _script(tmp_path, ["Первый фрагмент.", "Второй фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "native-sync-cost", model=POLZA_SYNC_MODEL, voice="alloy"),
    )

    assert code == 0
    assert payload["status"] == "success"
    assert provider.calls == ["chunk_01", "chunk_02"]
    assert seen[0] == {1: "0.3"}
    assert seen[1] == {1: "0.3", 2: "0.3"}

    run_root = tmp_path / "out" / "native-sync-cost"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"
    detail = _history_show(_run_uuid("native-sync-cost"))
    assert detail["run"]["status"] == "completed"
    assert all(attempt["status"] == "completed" for attempt in detail["attempts"])
    assert all(attempt["remote_id"] is None for attempt in detail["attempts"])
    assert all(attempt["cost"]["amount"] == "0.3" for attempt in detail["attempts"])
    assert all(attempt["cost"]["exact_available"] is True for attempt in detail["attempts"])
    roles = {artifact["role"] for artifact in detail["artifacts"]}
    assert {"paid_raw_audio", "chunk_audio", "final_audio"} <= roles


def test_native_sync_openrouter_links_raw_without_cost_or_generation_get(
    tmp_path, monkeypatch, capsys, native_env
):
    """An OpenRouter sync run links raw with no remote id and an unknown cost."""
    provider = FakeSyncProvider()  # the inline audio carries no synchronous usage
    _install_provider(monkeypatch, provider)
    monkeypatch.setattr(
        cli,
        "attach_costs",
        lambda *_args, **_kwargs: pytest.fail("a sync run must not fetch a generation cost"),
    )
    script = _script(tmp_path, ["Один фрагмент."])

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "native-openrouter",
            provider="openrouter-tts",
            model="google/gemini-3.1-flash-tts-preview",
            voice="Puck",
        ),
    )

    assert code == 0
    assert provider.calls == ["chunk_01"]
    run_root = tmp_path / "out" / "native-openrouter"
    receipt = json.loads(
        (run_root / "raw" / "chunk_01.mp3.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["remote_task_id"] is None
    assert receipt["generation_id"] == "gen-chunk_01"
    attempt = _history_show(_run_uuid("native-openrouter"))["attempts"][0]
    assert attempt["provider"] == "openrouter-tts"
    assert attempt["remote_id"] is None
    assert attempt["status"] == "completed"
    assert attempt["cost"]["amount"] is None
    assert attempt["cost"]["source"] == "unknown"


@pytest.mark.parametrize(
    ("provider_name", "model", "voice", "run_id", "invalid_id", "usage", "expected_cost"),
    [
        (
            "polza-tts",
            POLZA_SYNC_MODEL,
            "alloy",
            "native-sync-id-dotted",
            "generation.123",
            {"cost_rub": Decimal("0.3")},
            "0.3",
        ),
        (
            "polza-tts",
            POLZA_SYNC_MODEL,
            "alloy",
            "native-sync-id-overlong",
            "g" * 129,
            {"cost_rub": Decimal("0.3")},
            "0.3",
        ),
        (
            "openrouter-tts",
            "google/gemini-3.1-flash-tts-preview",
            "Puck",
            "native-sync-id-unicode",
            "gen-\u03a9-123",
            None,
            None,
        ),
    ],
)
def test_native_sync_drops_invalid_optional_generation_id(
    tmp_path,
    monkeypatch,
    capsys,
    native_env,
    provider_name,
    model,
    voice,
    run_id,
    invalid_id,
    usage,
    expected_cost,
):
    """A dotted, overlong, or non-ASCII sync generation id is dropped, not fatal.

    The optional id a synchronous provider reports is untrusted and not required
    to be a bounded opaque token. It must be discarded to ``None`` before the
    bounded receipt is written: the accepted paid audio, its raw receipt, the
    database link, the local conversion, and the export all survive, and the
    invalid value never reaches the receipt, the database, or the export.
    """
    provider = FakeSyncProvider(usage=usage)

    def invalid_result(_inner, text, chunk_id):
        raw_metadata = {} if usage is None else {"usage_direct": usage}
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="mp3",
            transcript=text,
            generation_id=invalid_id,
            client_path="requests",
            raw_metadata=raw_metadata,
        )

    provider.script = invalid_result
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Один фрагмент."])
    argv = _generate_argv(
        tmp_path,
        script,
        run_id,
        provider=provider_name,
        model=model,
        voice=voice,
    )

    code, payload = _json_run(monkeypatch, capsys, argv)

    # Exactly one paid POST, no second request, and the run completes locally.
    assert code == 0, payload
    assert payload["status"] == "success"
    assert provider.calls == ["chunk_01"]

    run_root = tmp_path / "out" / run_id
    raw_dir = run_root / "raw"
    assert (raw_dir / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    receipt_text = (raw_dir / "chunk_01.mp3.receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["remote_task_id"] is None
    assert receipt["generation_id"] is None
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"

    attempt = _history_show(_run_uuid(run_id))["attempts"][0]
    assert attempt["status"] == "completed"
    assert attempt["remote_id"] is None
    assert attempt["cost"]["amount"] == expected_cost

    # The receipt, the raw artifact, and the converted chunk metadata all carry
    # the same dropped (``None``) id, so no verification path can mismatch.
    artifacts = _artifact_media_metadata(native_env / "history.sqlite3")
    assert artifacts["paid_raw_audio"]["generation_id"] is None
    assert artifacts["chunk_audio"]["generation_id"] is None

    chunks_manifest = json.loads((run_root / "chunks" / "chunks.json").read_text(encoding="utf-8"))
    export_text = json.dumps(chunks_manifest)
    assert chunks_manifest["chunks"][0].get("generation_id") is None
    assert invalid_id not in receipt_text
    assert invalid_id not in json.dumps(payload)
    assert invalid_id not in export_text


def test_native_sync_unknown_submit_blocks_without_repeat(
    tmp_path, monkeypatch, capsys, native_env
):
    """A synchronous submit whose outcome is unknown blocks resume with no repeat."""
    provider = FakeSyncProvider()

    def fail_after_reserve(_inner, _text, _chunk_id):
        raise requests.Timeout("response lost")

    provider.script = fail_after_reserve
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Неопределённый исход."])
    argv = _generate_argv(
        tmp_path, script, "native-sync-unknown", model=POLZA_SYNC_MODEL, voice="alloy"
    )

    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert provider.calls == ["chunk_01"]

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("an unconfirmed sync submit must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(
        tmp_path,
        script,
        "native-sync-unknown",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--resume"],
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert provider.calls == ["chunk_01"]


def test_native_sync_resume_reconciles_raw_receipt_written_before_db_link(
    tmp_path, monkeypatch, capsys, native_env
):
    """A crash between sync raw bytes and their DB link rebuilds locally, no POST/GET."""
    from voiceover_pipeline.history.repository import HistoryRepository

    provider = FakeSyncProvider(usage={"cost_rub": Decimal("0.3")})
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Окно между raw и БД."])

    real_link = HistoryRepository.record_polza_sync_raw_saved
    calls = {"n": 0}

    def flaky_link(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash between raw bytes and the database row")
        return real_link(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_polza_sync_raw_saved", flaky_link)
    argv = _generate_argv(
        tmp_path, script, "native-sync-reconcile", model=POLZA_SYNC_MODEL, voice="alloy"
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30
    assert provider.calls == ["chunk_01"]
    raw_dir = tmp_path / "out" / "native-sync-reconcile" / "raw"
    assert (raw_dir / "chunk_01.mp3").exists()
    assert (raw_dir / "chunk_01.mp3.receipt.json").exists()

    monkeypatch.setattr(HistoryRepository, "record_polza_sync_raw_saved", real_link)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("reconciling an on-disk sync receipt must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(
        tmp_path,
        script,
        "native-sync-reconcile",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--resume"],
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0, payload
    assert provider.calls == ["chunk_01"]
    run_root = tmp_path / "out" / "native-sync-reconcile"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    # The cost observed in the crash window is unrecoverable: the receipt carries
    # no cost, so the rebuilt attempt keeps an unknown cost rather than a false one.
    assert (
        _history_show(_run_uuid("native-sync-reconcile"))["attempts"][0]["cost"]["amount"] is None
    )


def test_native_sync_resume_rebuilds_from_raw_artifact_without_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A committed sync raw artifact is rebuilt locally with no provider construction."""
    provider = FakeSyncProvider(usage={"cost_rub": Decimal("0.3")})
    _install_provider(monkeypatch, provider)
    calls = {"n": 0}

    def flaky_write(_ffmpeg, data, _fmt, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("conversion boom")
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)
    script = _script(tmp_path, ["Сохранённый raw."])
    argv = _generate_argv(
        tmp_path, script, "native-sync-raw", model=POLZA_SYNC_MODEL, voice="alloy"
    )
    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert provider.calls == ["chunk_01"]

    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a local sync raw rebuild must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(
        tmp_path,
        script,
        "native-sync-raw",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--resume"],
    )
    code, _payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0
    assert provider.calls == ["chunk_01"]
    assert (
        tmp_path / "out" / "native-sync-raw" / "chunks" / "chunk_01.mp3"
    ).read_bytes() == b"chunk_01-audio"
    detail = _history_show(_run_uuid("native-sync-raw"))
    assert detail["run"]["status"] == "completed"
    # The exact cost linked before conversion survives the failed conversion.
    assert detail["attempts"][0]["cost"]["amount"] == "0.3"


def test_native_sync_resume_corrupted_raw_fails_closed(tmp_path, monkeypatch, capsys, native_env):
    """A tampered synchronous raw file never permits a second paid submit."""
    provider = FakeSyncProvider(usage={"cost_rub": Decimal("0.3")})
    _install_provider(monkeypatch, provider)
    calls = {"n": 0}

    def flaky_write(_ffmpeg, data, _fmt, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("conversion boom")
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)
    script = _script(tmp_path, ["Повреждённый raw."])
    argv = _generate_argv(
        tmp_path, script, "native-sync-corrupt", model=POLZA_SYNC_MODEL, voice="alloy"
    )
    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert provider.calls == ["chunk_01"]

    raw = tmp_path / "out" / "native-sync-corrupt" / "raw" / "chunk_01.mp3"
    assert raw.exists()
    raw.write_bytes(b"tampered-paid-bytes")

    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a corrupt raw file must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(
        tmp_path,
        script,
        "native-sync-corrupt",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--resume"],
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_RAW_EVIDENCE_INVALID"
    assert provider.calls == ["chunk_01"]


def test_native_sync_resume_changed_text_blocks_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """Changed spoken text fails the sync identity preflight before a provider exists."""
    provider = FakeSyncProvider(usage={"cost_rub": Decimal("0.3")})
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Исходный текст."])
    argv = _generate_argv(
        tmp_path, script, "native-sync-text", model=POLZA_SYNC_MODEL, voice="alloy"
    )
    code, _payload = _json_run(monkeypatch, capsys, argv)
    assert code == 0

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a blocked resume must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    changed = _script(tmp_path, ["Совсем другой текст."])
    resume_argv = _generate_argv(
        tmp_path,
        changed,
        "native-sync-text",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--resume"],
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert provider.calls == ["chunk_01"]


# ── resume: identity and evidence ─────────────────────────────────────────────


def test_native_resume_changed_text_blocks_before_any_network(
    tmp_path, monkeypatch, capsys, native_env
):
    """Changed spoken text fails the identity preflight before a provider exists."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент текста."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-identity")
    )
    assert code == 0
    assert provider.submits == ["chunk_01"]

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("provider must not be constructed on a blocked resume")

    monkeypatch.setattr(cli, "build_provider", explode)
    changed = _script(tmp_path, ["Совершенно другой текст."])
    resume_argv = _generate_argv(tmp_path, changed, "native-identity", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["status"] == "error"
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert provider.submits == ["chunk_01"]


def test_native_resume_known_id_uses_get_only_and_never_resubmits(
    tmp_path, monkeypatch, capsys, native_env
):
    """A stored accepted task id is finished with recover-only calls."""
    provider = FakeMediaProvider()

    def accept_then_fail(inner, text, chunk_id):
        inner.on_media_task_accepted(f"task-{chunk_id}")
        raise requests.Timeout("poll timed out")

    provider.script = accept_then_fail
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Фрагмент для GET-восстановления."])

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-get"))
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert provider.submits == ["chunk_01"]
    assert provider.recovers == []

    resume_provider = FakeMediaProvider()
    _install_provider(monkeypatch, resume_provider)
    resume_argv = _generate_argv(tmp_path, script, "native-get", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0
    assert resume_provider.submits == []
    assert resume_provider.recovers == ["task-chunk_01"]
    assert (tmp_path / "out" / "native-get" / "chunks" / "chunk_01.mp3").exists()


def test_native_resume_rebuilds_from_raw_receipt_without_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A verified raw receipt is converted locally with no provider construction."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    calls = {"n": 0}

    def flaky_write(_ffmpeg, data, _fmt, path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("conversion boom")
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)
    script = _script(tmp_path, ["Первый.", "Второй."])

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-raw"))
    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_CONVERSION_FAILED"
    assert provider.submits == ["chunk_01", "chunk_02"]
    # The accepted paid bytes survive the conversion failure.
    assert (tmp_path / "out" / "native-raw" / "raw" / "chunk_02.mp3").exists()

    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_mp3)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a local raw rebuild must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(tmp_path, script, "native-raw", extra=["--resume"])
    code, _payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0
    assert provider.submits == ["chunk_01", "chunk_02"]
    assert (tmp_path / "out" / "native-raw" / "chunks" / "chunk_02.mp3").read_bytes() == (
        b"chunk_02-audio"
    )
    run_uuid = _run_uuid("native-raw")
    detail = _history_show(run_uuid)
    assert detail["run"]["status"] == "completed"


def test_native_resume_unconfirmed_submit_blocks_without_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A submit that was never confirmed blocks resume and sends no request."""
    provider = FakeMediaProvider()

    def fail_before_accept(_inner, _text, _chunk_id):
        raise requests.Timeout("read timed out")

    provider.script = fail_before_accept
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Неопределённый платный исход."])

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-unconfirmed")
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("an unconfirmed paid submit must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(tmp_path, script, "native-unconfirmed", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert provider.submits == ["chunk_01"]


def test_native_resume_reconciles_raw_receipt_written_before_db_link(
    tmp_path, monkeypatch, capsys, native_env
):
    """A crash between raw bytes and their database row is reconciled locally."""
    from voiceover_pipeline.history.repository import HistoryRepository

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Окно между raw и БД."])

    real_link = HistoryRepository.record_polza_media_raw_saved
    calls = {"n": 0}

    def flaky_link(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash between raw bytes and the database row")
        return real_link(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_polza_media_raw_saved", flaky_link)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-reconcile")
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]
    raw_dir = tmp_path / "out" / "native-reconcile" / "raw"
    assert (raw_dir / "chunk_01.mp3").exists()
    assert (raw_dir / "chunk_01.mp3.receipt.json").exists()

    monkeypatch.setattr(HistoryRepository, "record_polza_media_raw_saved", real_link)

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("reconciling an on-disk receipt must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(tmp_path, script, "native-reconcile", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0, payload
    assert provider.submits == ["chunk_01"]
    assert (
        tmp_path / "out" / "native-reconcile" / "chunks" / "chunk_01.mp3"
    ).read_bytes() == b"chunk_01-audio"


@pytest.mark.parametrize(
    ("overrides"),
    [{"voice": "Bella"}, {"model": "elevenlabs/text-to-speech-multilingual-v2"}],
)
def test_native_resume_changed_synthesis_identity_blocks_before_network(
    tmp_path, monkeypatch, capsys, native_env, overrides
):
    """A changed voice or model fails the identity preflight with no provider."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Проверка identity."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-synth-identity")
    )
    assert code == 0

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("provider must not be constructed on a blocked resume")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(
        tmp_path, script, "native-synth-identity", extra=["--resume"], **overrides
    )
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_RESUME_IDENTITY_CHANGED"
    assert provider.submits == ["chunk_01"]


# ── completed-run export repair ───────────────────────────────────────────────


def test_native_completed_run_resume_only_repairs_exports(
    tmp_path, monkeypatch, capsys, native_env
):
    """A completed run re-exports its JSON without a provider or paid action."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Готовый прогон."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-export")
    )
    assert code == 0
    run_root = tmp_path / "out" / "native-export"
    (run_root / "run_state.json").unlink()
    (run_root / "chunks" / "chunks.json").unlink()

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("export repair must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(tmp_path, script, "native-export", extra=["--resume"])
    code, _payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 0
    assert (run_root / "run_state.json").exists()
    assert (run_root / "chunks" / "chunks.json").exists()
    assert provider.submits == ["chunk_01"]


def test_native_export_failure_keeps_history_and_audio_then_repairs(
    tmp_path, monkeypatch, capsys, native_env
):
    """A failed export leaves audio and database intact and returns code 50."""
    from voiceover_pipeline.services import native_generation

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Экспорт ломается."])

    def broken_export(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(native_generation, "write_native_export", broken_export)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-export-fail")
    )

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_EXPORT_FAILED"
    run_root = tmp_path / "out" / "native-export-fail"
    assert (run_root / "chunks" / "chunk_01.mp3").exists()
    run_uuid = _run_uuid("native-export-fail")
    assert _history_show(run_uuid)["run"]["status"] == "completed"

    # A later resume repairs the projection with no provider or paid action.
    monkeypatch.setattr(native_generation, "write_native_export", _real_write_export())

    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("projection repair must not build a provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    resume_argv = _generate_argv(tmp_path, script, "native-export-fail", extra=["--resume"])
    code, _payload = _json_run(monkeypatch, capsys, resume_argv)
    assert code == 0
    assert (run_root / "run_state.json").exists()
    assert provider.submits == ["chunk_01"]


def _real_write_export():
    from voiceover_pipeline.history.native_export import write_native_export

    return write_native_export


# ── safety guards ─────────────────────────────────────────────────────────────


def test_native_overwrite_is_rejected_and_keeps_accepted_evidence(
    tmp_path, monkeypatch, capsys, native_env
):
    """Overwrite is refused for a native run; the accepted evidence stays."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Перезапись запрещена."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-overwrite")
    )
    assert code == 0

    argv = _generate_argv(
        tmp_path,
        script,
        "native-overwrite",
        extra=["--overwrite", "--confirm-delete-paid-audio"],
    )
    code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_OVERWRITE_UNSUPPORTED"
    assert (tmp_path / "out" / "native-overwrite" / "chunks" / "chunk_01.mp3").exists()


def test_native_run_held_lock_blocks_a_second_writer(tmp_path, monkeypatch, capsys, native_env):
    """A held run lock fails the second writer closed instead of double-writing."""
    from voiceover_pipeline.history.locking import acquire_run_lock

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Конкурентный писатель."])
    run_root = tmp_path / "out" / "native-lock"

    with acquire_run_lock(run_root):
        argv = _generate_argv(tmp_path, script, "native-lock")
        code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_RUN_LOCKED"
    assert provider.submits == []


def test_native_run_with_missing_history_database_fails_closed(
    tmp_path, monkeypatch, capsys, native_env
):
    """Native local evidence without its database row refuses to run as legacy."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Потерянная история."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-lost"))
    assert code == 0
    run_root = tmp_path / "out" / "native-lost"

    # Remove the committed history database but keep the run-local evidence.
    database = native_env / "history.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(database) + suffix)
        if candidate.exists():
            candidate.unlink()

    argv = _generate_argv(tmp_path, script, "native-lost", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_OWNERSHIP_RUN_MISSING"
    assert (run_root / "chunks" / "chunk_01.mp3").exists()

    # A present but unreadable database next to native evidence fails closed too.
    root_db = native_env / "history.sqlite3"
    root_db.write_bytes(b"not a database")
    code, payload = _json_run(monkeypatch, capsys, argv)
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_OWNERSHIP_UNVERIFIABLE"
    assert root_db.exists()


def test_legacy_existing_sync_directory_stays_on_the_json_writer(
    tmp_path, monkeypatch, capsys, native_env
):
    """An on-disk legacy sync run is never captured by the native executor.

    A fresh eligible sync run is native, but a directory that already carries
    legacy JSON state with no native trace must stay on the legacy writer: the
    fresh-native branch needs a run root that does not exist yet, and ownership
    resolves the existing root to legacy before any deletion.
    """
    provider = FakeSyncProvider()
    _install_provider(monkeypatch, provider)
    _patch_legacy_finalization(monkeypatch)
    script = _script(tmp_path, ["Легаси-прогон."])

    # A pre-existing legacy run directory with only its JSON state and no native
    # descriptor, native history row, or paid raw receipt.
    run_root = tmp_path / "out" / "legacy-sync"
    run_root.mkdir(parents=True)
    (run_root / "run_state.json").write_text(
        json.dumps({"status": "running", "chunks": [], "run_id": "legacy-sync"}),
        encoding="utf-8",
    )

    argv = _generate_argv(
        tmp_path,
        script,
        "legacy-sync",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--overwrite"],
    )
    code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 0
    assert payload["status"] == "success"
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert "native_history" not in state
    assert not (run_root / ".voiceover-native-history.json").exists()

    from voiceover_pipeline.commands.history import list_history

    assert list_history()["count"] == 0


def test_native_descriptor_is_required_for_ownership(monkeypatch, tmp_path, native_env):
    """A native run row alone makes a root native-owned; a descriptor alone blocks."""
    from voiceover_pipeline.services import native_generation

    run_root = tmp_path / "out" / "owned"
    run_root.mkdir(parents=True)
    assert native_generation.resolve_native_ownership(run_root).route == "legacy"

    native_generation._write_descriptor(run_root, "8e2f1c8e-0000-4000-8000-000000000001")
    decision = native_generation.resolve_native_ownership(run_root)
    assert decision.route == "blocked"
    assert decision.error_code == "NATIVE_OWNERSHIP_RUN_MISSING"


def test_native_run_parts_shape_is_valid() -> None:
    """The native route is admitted for the ordinary runs and the one dialogue route."""
    import argparse

    args = argparse.Namespace(
        provider="polza-tts",
        model=POLZA_MEDIA_MODEL,
        with_timings=False,
        timing_provider="faster-whisper",
        tts_quality_provider=None,
        no_trim=False,
    )
    assert cli._native_route_eligible(args, "markdown") is True
    # A synchronous polza-tts model and an openrouter-tts run are admitted too.
    args.model = POLZA_SYNC_MODEL
    assert cli._native_route_eligible(args, "markdown") is True
    args.provider = "openrouter-tts"
    args.model = "google/gemini-3.1-flash-tts-preview"
    assert cli._native_route_eligible(args, "markdown") is True
    args.provider = "polza-tts"
    args.model = POLZA_MEDIA_MODEL
    # The one admitted integrated step is a local faster-whisper timing request.
    args.with_timings = True
    assert cli._native_route_eligible(args, "markdown") is True
    # A cloud timing provider stays on the legacy executor.
    args.timing_provider = "groq-whisper"
    assert cli._native_route_eligible(args, "markdown") is False
    args.timing_provider = "openrouter-whisper"
    assert cli._native_route_eligible(args, "markdown") is False
    args.with_timings = False
    args.timing_provider = "faster-whisper"
    # ``--no-trim`` is recorded as the run's own trimming semantics.
    args.no_trim = True
    assert cli._native_route_eligible(args, "markdown") is True
    args.no_trim = False
    # An installed local ASR quality provider is admitted for the same route.
    args.tts_quality_provider = "qwen-local"
    assert cli._native_route_eligible(args, "markdown") is True
    args.tts_quality_provider = "nemotron-local"
    assert cli._native_route_eligible(args, "markdown") is True
    # A cloud ASR quality provider keeps the legacy executor.
    args.tts_quality_provider = "xai-stt"
    assert cli._native_route_eligible(args, "markdown") is False
    # Asking for local timings and local quality at once is not part of the slice.
    args.tts_quality_provider = "qwen-local"
    args.with_timings = True
    assert cli._native_route_eligible(args, "markdown") is False
    args.with_timings = False
    args.tts_quality_provider = None
    assert cli._native_route_eligible(args, "dialogue") is False
    # The one admitted dialogue route is the OpenRouter Gemini two-speaker script
    # with the required installed local quality provider and the default trim.
    args.provider = "openrouter-tts"
    args.model = "google/gemini-3.1-flash-tts-preview"
    args.tts_quality_provider = "qwen-local"
    assert cli._native_route_eligible(args, "dialogue") is True
    args.no_trim = True
    assert cli._native_route_eligible(args, "dialogue") is False
    args.no_trim = False
    args.tts_quality_provider = "xai-stt"
    assert cli._native_route_eligible(args, "dialogue") is False
    args.tts_quality_provider = "qwen-local"
    args.provider = "polza-tts"
    assert cli._native_route_eligible(args, "dialogue") is False
    # A provider outside the admitted set keeps the legacy executor.
    args.provider = "polza-chat-audio"
    assert cli._native_route_eligible(args, "markdown") is False


def test_polza_provider_type_is_referenceable() -> None:
    """Guard that the fake mirrors the real provider's callback surface."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_MEDIA_MODEL, voice="Rachel")
    assert hasattr(provider, "on_media_task_accepted")
    assert hasattr(provider, "on_media_completed")
    assert hasattr(provider, "recover_media_task")


def test_native_ownership_reads_a_live_wal_database(tmp_path, monkeypatch, capsys, native_env):
    """A different run's active WAL must not hide a committed native owner."""
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.services.native_generation import resolve_native_ownership

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Активная история."])
    code, _ = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-wal"))
    assert code == 0
    run_root = tmp_path / "out" / "native-wal"
    with HistoryDatabase(native_env / "history.sqlite3") as database:
        database.connect()
        database.migrate()
        assert resolve_native_ownership(run_root).route == "native_existing"


@pytest.mark.parametrize("artifact_kind", ["chunk", "final"])
def test_native_resume_never_writes_through_artifact_symlink(
    tmp_path, monkeypatch, capsys, native_env, artifact_kind
):
    """A recovered output must not overwrite an external file via an alias."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Не перезаписывать чужое."])
    code, _ = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-alias"))
    assert code == 0
    run_root = tmp_path / "out" / "native-alias"
    if artifact_kind == "chunk":
        artifact = run_root / "chunks" / "chunk_01.mp3"
    else:
        slug = POLZA_MEDIA_MODEL.replace("/", "-")
        artifact = run_root / f"native-alias-voiceover-{slug}.mp3"
    external = tmp_path / "synthetic-external.mp3"
    external.write_bytes(b"protected-external-bytes")
    artifact.unlink()
    artifact.symlink_to(external)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider on repair")
    )

    code, _ = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-alias", extra=["--resume"])
    )
    assert code != 0
    assert external.read_bytes() == b"protected-external-bytes"


@pytest.mark.parametrize("artifact_kind", ["chunk", "final"])
def test_native_resume_never_clobbers_conflicting_regular_artifact(
    tmp_path, monkeypatch, capsys, native_env, artifact_kind
):
    """Local recovery must not replace a pre-existing conflicting output name."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Сохраняем существующий файл."])
    code, _ = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-conflict"))
    assert code == 0
    root = tmp_path / "out" / "native-conflict"
    if artifact_kind == "chunk":
        artifact = root / "chunks" / "chunk_01.mp3"
    else:
        slug = POLZA_MEDIA_MODEL.replace("/", "-")
        artifact = root / f"native-conflict-voiceover-{slug}.mp3"
    artifact.write_bytes(b"protected-existing-content")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider on repair")
    )

    code, _ = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "native-conflict", extra=["--resume"])
    )
    assert code != 0
    assert artifact.read_bytes() == b"protected-existing-content"


def test_native_provider_error_does_not_echo_untrusted_secret(
    tmp_path, monkeypatch, capsys, native_env
):
    """A remote error body is untrusted and cannot become the public JSON error."""
    provider = FakeMediaProvider()

    def fail_with_secret(_provider, _text, _chunk_id):
        raise RuntimeError("Authorization: Bearer sk-synthetic-private")

    provider.script = fail_with_secret
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Секрет не печатать."])
    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-error"))
    assert code == 30
    assert "sk-synthetic-private" not in json.dumps(payload)
    assert provider.submits == ["chunk_01"]


def test_native_descriptor_symlink_is_rejected_without_reading_target(
    tmp_path, native_env, monkeypatch
):
    """The ownership probe must not follow a user-controlled descriptor link."""
    from voiceover_pipeline.services import native_generation

    run_root = tmp_path / "out" / "native-descriptor-link"
    run_root.mkdir(parents=True)
    external = tmp_path / "synthetic-descriptor-target.json"
    external.write_text(
        json.dumps(
            {
                "artifact_type": native_generation.OWNERSHIP_ARTIFACT_TYPE,
                "ownership_version": native_generation.OWNERSHIP_VERSION,
                "run_uuid": "8e2f1c8e-0000-4000-8000-000000000001",
            }
        ),
        encoding="utf-8",
    )
    (run_root / native_generation.OWNERSHIP_FILE_NAME).symlink_to(external)
    with pytest.raises(native_generation.NativeGenerationError) as excinfo:
        native_generation.resolve_native_ownership(run_root)
    assert excinfo.value.error_code == "NATIVE_OWNERSHIP_UNVERIFIABLE"


def test_native_duplicate_part_evidence_fails_closed_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """A part with two attempts is refused before any recovery or provider work."""
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import (
        ATTEMPT_CALL_TYPE_TTS_CHUNK,
        ATTEMPT_STATUS_REMOTE_ACCEPTED,
        HistoryRepository,
    )

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Дубль доказательств."])
    code, _ = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-duplicate"))
    assert code == 0
    run_uuid = _run_uuid("native-duplicate")

    database = HistoryDatabase(native_env / "history.sqlite3")
    database.connect()
    database.migrate()
    try:
        repository = HistoryRepository(database)
        part = repository.get_parts(run_uuid)[0]
        repository.add_attempt(
            run_uuid,
            call_type=ATTEMPT_CALL_TYPE_TTS_CHUNK,
            part_uuid=part.part_uuid,
            provider="polza-tts",
            model=POLZA_MEDIA_MODEL,
            remote_id="task-duplicate",
            status=ATTEMPT_STATUS_REMOTE_ACCEPTED,
        )
    finally:
        database.close()

    monkeypatch.setattr(
        cli,
        "build_provider",
        lambda *_args, **_kwargs: pytest.fail("duplicate evidence must not build a provider"),
    )
    resume_argv = _generate_argv(tmp_path, script, "native-duplicate", extra=["--resume"])
    code, payload = _json_run(monkeypatch, capsys, resume_argv)

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_EVIDENCE_INCONSISTENT"
    assert provider.submits == ["chunk_01"]


def test_legacy_stale_selection_rechecks_native_ownership_before_writing(
    tmp_path, monkeypatch, capsys, native_env
):
    """A stale legacy selection must not delete or overwrite a native-owned root."""
    from voiceover_pipeline.services import native_generation

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Нативный владелец каталога."])
    code, _ = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "native-recheck"))
    assert code == 0
    run_root = tmp_path / "out" / "native-recheck"
    chunk = run_root / "chunks" / "chunk_01.mp3"
    descriptor = run_root / ".voiceover-native-history.json"
    assert chunk.exists() and descriptor.exists()

    real_resolve = native_generation.resolve_native_ownership
    calls = {"n": 0}

    def stale_first_read(output_root, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # The early read raced before the native commit was visible.
            return native_generation.NativeOwnership(route="legacy")
        return real_resolve(output_root, **kwargs)

    monkeypatch.setattr(cli.native_generation, "resolve_native_ownership", stale_first_read)
    argv = _generate_argv(
        tmp_path,
        script,
        "native-recheck",
        extra=["--overwrite", "--confirm-delete-paid-audio"],
    )
    code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 30
    assert payload["details"]["error_code"]
    assert chunk.exists()
    assert descriptor.exists()


def test_native_fresh_run_refuses_a_root_with_existing_local_state(tmp_path):
    """A fresh native run must not claim a root another writer already owns."""
    from voiceover_pipeline.history.database import HistoryDatabase
    from voiceover_pipeline.history.repository import HistoryRepository
    from voiceover_pipeline.services import native_generation
    from voiceover_pipeline.services.prepare import PreparedPart, PreparedRun

    run_root = tmp_path / "out" / "native-race"
    run_root.mkdir(parents=True)
    legacy_state = run_root / "run_state.json"
    legacy_bytes = json.dumps({"status": "completed", "chunks": []})
    legacy_state.write_text(legacy_bytes, encoding="utf-8")

    chunk = ScriptChunk(number=1, id="chunk_01", text="race", voice="Rachel")
    prepared = PreparedRun(
        provider="polza-tts",
        model=POLZA_MEDIA_MODEL,
        voice="Rachel",
        style_prompt=None,
        prompt_mode="none",
        parts=(PreparedPart(chunk=chunk, voice="Rachel"),),
    )

    database = HistoryDatabase(tmp_path / "history.sqlite3")
    database.connect()
    database.migrate()
    try:
        repository = HistoryRepository(database)
        with pytest.raises(native_generation.NativeGenerationError) as excinfo:
            native_generation.execute_native_tts(
                repository=repository,
                run_root=run_root,
                paths=object(),
                prepared=prepared,
                chunks=[chunk],
                script_format="markdown",
                script_path=tmp_path / "script.md",
                output_options={"trim_final_silence": True, "processing_version": 1},
                ffmpeg_path="ffmpeg",
                ffprobe_path="ffprobe",
                user_label="native-race",
                resume=False,
                provider_factory=lambda: pytest.fail("no provider for a refused fresh run"),
                hooks=object(),
                logger=object(),
            )
        assert excinfo.value.error_code == "NATIVE_OWNERSHIP_RUN_MISSING"
        assert repository.find_runs_by_root(str(run_root.resolve()), limit=1) == []
    finally:
        database.close()
    assert legacy_state.read_text(encoding="utf-8") == legacy_bytes


@pytest.mark.parametrize("writer", ["descriptor", "export"])
def test_native_json_writer_never_follows_predictable_temporary_symlink(tmp_path, writer):
    """A pre-existing temp-name alias cannot overwrite an external file."""
    from voiceover_pipeline.history.native_export import _atomic_write_json
    from voiceover_pipeline.services import native_generation

    run_root = tmp_path / "out" / "native-temp-link"
    run_root.mkdir(parents=True)
    external = tmp_path / "synthetic-protected.txt"
    external.write_text("protected", encoding="utf-8")
    if writer == "descriptor":
        target = run_root / native_generation.OWNERSHIP_FILE_NAME
        (run_root / f"{native_generation.OWNERSHIP_FILE_NAME}.tmp").symlink_to(external)
        native_generation._write_descriptor(run_root, "8e2f1c8e-0000-4000-8000-000000000001")
    else:
        target = run_root / "run_state.json"
        (run_root / "run_state.json.tmp").symlink_to(external)
        _atomic_write_json(target, {"native_history": "synthetic"})
    assert target.exists()
    assert external.read_text(encoding="utf-8") == "protected"


def test_native_locked_fresh_run_creates_no_run_directory_before_lock(
    tmp_path, monkeypatch, capsys, native_env
):
    """A fresh native attempt refused by a held lock leaves no run directory behind."""
    from voiceover_pipeline.history.locking import acquire_run_lock

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Никаких каталогов до лока."])
    run_root = tmp_path / "out" / "native-no-dirs"

    with acquire_run_lock(run_root):
        code, payload = _json_run(
            monkeypatch, capsys, _generate_argv(tmp_path, script, "native-no-dirs")
        )

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_RUN_LOCKED"
    # The lock is taken before any directory or descriptor write, so a refused
    # fresh attempt must not create even the output directory.
    assert not (tmp_path / "out").exists()
    assert provider.submits == []


def test_legacy_writer_holds_run_lock_across_ownership_recheck(
    tmp_path, monkeypatch, capsys, native_env
):
    """After the legacy ownership recheck, no native claim can acquire the run lock."""
    from voiceover_pipeline.history.locking import HistoryRunLockedError, acquire_run_lock
    from voiceover_pipeline.services import native_generation

    provider = FakeSyncProvider()
    _install_provider(monkeypatch, provider)
    _patch_legacy_finalization(monkeypatch)
    script = _script(tmp_path, ["Легаси под локом."])
    # ``--tts-quality-provider xai-stt`` is a cloud ASR provider, so this sync
    # non-dialogue run keeps the legacy writer; the lock interval around the legacy
    # ownership recheck is still exercised.
    argv = _generate_argv(
        tmp_path,
        script,
        "legacy-lock",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        extra=["--tts-quality-provider", "xai-stt"],
    )

    observed: dict[str, object] = {}
    real_reject = cli._reject_native_owned_root_for_legacy

    def spy_reject(paths):
        real_reject(paths)
        run_root = paths.output_root
        # The interval the reviewer flagged: right after the ownership recheck and
        # before the legacy JSON writer runs. A native claim must not acquire the
        # run lock here, so it can neither commit a snapshot nor submit.
        try:
            with acquire_run_lock(run_root):
                observed["second_writer_acquired"] = True
        except HistoryRunLockedError:
            observed["second_writer_acquired"] = False
        observed["ownership_route"] = native_generation.resolve_native_ownership(run_root).route
        observed["json_written"] = (run_root / "run_state.json").exists()

    monkeypatch.setattr(cli, "_reject_native_owned_root_for_legacy", spy_reject)

    code, payload = _json_run(monkeypatch, capsys, argv)

    assert code == 0
    assert payload["status"] == "success"
    run_root = tmp_path / "out" / "legacy-lock"
    assert observed["second_writer_acquired"] is False
    assert observed["ownership_route"] == "legacy"
    assert observed["json_written"] is False
    assert provider.calls == ["chunk_01"]
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert "native_history" not in state
    assert not (run_root / ".voiceover-native-history.json").exists()

    from voiceover_pipeline.commands.history import list_history

    assert list_history()["count"] == 0


def test_script_chunk_helpers_are_stable() -> None:
    """Sanity check that the native snapshot consumes plain generated chunk ids."""
    chunk = ScriptChunk(number=1, id="chunk_01", text="x")
    assert chunk.id == "chunk_01"


# ── history resume / history sync (DB-first user journey) ─────────────────────


def test_history_resume_and_sync_parser_and_help_envelopes(monkeypatch, capsys, native_env):
    """Both verbs parse, state their paid semantics, and reject unknown flags."""
    import voiceover_pipeline.cli as cli_module

    for verb in ("resume", "sync"):
        monkeypatch.setattr(sys, "argv", ["voiceover-pipeline", "history", verb, "--help"])
        with pytest.raises(SystemExit) as excinfo:
            cli_module.main()
        assert excinfo.value.code == 0
        help_text = capsys.readouterr().out
        assert "--json" in help_text

    monkeypatch.setattr(sys, "argv", ["voiceover-pipeline", "history", "resume", "--help"])
    with pytest.raises(SystemExit):
        cli_module.main()
    assert "potentially paid" in " ".join(capsys.readouterr().out.split())

    monkeypatch.setattr(sys, "argv", ["voiceover-pipeline", "history", "sync", "--help"])
    with pytest.raises(SystemExit):
        cli_module.main()
    assert "without a new paid submit" in " ".join(capsys.readouterr().out.split())

    # A missing run id and an unknown flag are ordinary argparse usage errors.
    code, payload = _json_run(
        monkeypatch, capsys, ["voiceover-pipeline", "history", "resume", "--json"]
    )
    assert code == 2
    assert payload == {"status": "error", "error": "Invalid command-line arguments", "code": 2}

    code, payload = _json_run(
        monkeypatch,
        capsys,
        ["voiceover-pipeline", "history", "resume", str(uuid.uuid4()), "--overwrite", "--json"],
    )
    assert code == 2
    assert payload["code"] == 2


def test_history_sync_completed_run_repairs_exports_without_provider_or_script(
    tmp_path, monkeypatch, capsys, native_env
):
    """A completed run syncs by rewriting the four JSON exports with no provider."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент.", "Второй фрагмент."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-done")
    )
    assert code == 0
    assert provider.submits == ["chunk_01", "chunk_02"]

    run_root = tmp_path / "out" / "hist-sync-done"
    run_uuid = _run_uuid("hist-sync-done")
    _, revision_before = _run_row(native_env, run_uuid)
    evidence_before = _evidence_counts(native_env, run_uuid)
    # The original script is gone, and the exports are deleted to force a rebuild.
    script.unlink()
    (run_root / "run_state.json").unlink()
    (run_root / "chunks" / "chunks.json").unlink()

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0
    assert payload["status"] == "success"
    assert payload["run_uuid"] == run_uuid
    assert payload["mode"] == "sync"
    assert payload["revision"] == revision_before
    files = payload["files"]
    assert set(files) == {"full_mp3", "run_json", "chunks_json", "manifest_json"}
    first = {name: Path(path).read_bytes() for name, path in files.items()}
    assert all(data for data in first.values())

    # A repeated sync is byte-identical, adds no paid evidence, and bumps no revision.
    code, second_payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))
    assert code == 0
    assert second_payload["revision"] == revision_before
    second = {name: Path(path).read_bytes() for name, path in files.items()}
    assert second == first
    assert provider.submits == ["chunk_01", "chunk_02"]
    assert _evidence_counts(native_env, run_uuid) == evidence_before

    # The projection still carries the stored (now missing) script path, never re-read.
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert "pending_attempt" not in state
    assert state["native_history"]["history_run_uuid"] == run_uuid

    # ``history resume`` of the same completed run is also export-only and free.
    code, resume_payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))
    assert code == 0
    assert resume_payload["mode"] == "resume"
    assert _evidence_counts(native_env, run_uuid) == evidence_before


def test_history_resume_submits_only_the_unattempted_part_without_original_script(
    tmp_path, monkeypatch, capsys, native_env
):
    """Resume rebuilds the snapshot and submits exactly the unattempted part."""
    from voiceover_pipeline.history.repository import HistoryRepository

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый.", "Второй."])

    real_reserve = HistoryRepository.reserve_paid_tts_attempt
    calls = {"n": 0}

    def flaky_reserve(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash before the second reserve")
        return real_reserve(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", flaky_reserve)
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-resume"))
    assert code == 30
    assert provider.submits == ["chunk_01"]
    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", real_reserve)

    run_root = tmp_path / "out" / "hist-resume"
    run_uuid = _run_uuid("hist-resume")
    script.unlink()

    resume_provider = FakeMediaProvider()
    _install_provider(monkeypatch, resume_provider)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "resume"))

    assert code == 0, payload
    # Part 1 was skipped; only the unattempted part was submitted.
    assert resume_provider.submits == ["chunk_02"]
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"
    assert payload["mode"] == "resume"
    status, _revision = _run_row(native_env, run_uuid)
    assert status == "completed"


def test_history_sync_known_media_id_is_get_only(tmp_path, monkeypatch, capsys, native_env):
    """Sync finishes a stored Media task id with GET recovery and never a POST."""
    provider = FakeMediaProvider()

    def accept_then_fail(inner, text, chunk_id):
        inner.on_media_task_accepted(f"task-{chunk_id}")
        raise requests.Timeout("poll timed out")

    provider.script = accept_then_fail
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Фрагмент для GET."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-get")
    )
    assert code == 30
    assert provider.submits == ["chunk_01"]

    run_uuid = _run_uuid("hist-sync-get")
    sync_provider = FakeMediaProvider()
    _install_provider(monkeypatch, sync_provider)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert sync_provider.submits == []
    assert sync_provider.recovers == ["task-chunk_01"]


def test_history_sync_reconciles_on_disk_raw_without_db_link(
    tmp_path, monkeypatch, capsys, native_env
):
    """A crash between raw bytes and the DB row is repaired locally with no network."""
    from voiceover_pipeline.history.repository import HistoryRepository

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Окно между raw и БД."])

    real_link = HistoryRepository.record_polza_media_raw_saved
    calls = {"n": 0}

    def flaky_link(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash between raw bytes and the database row")
        return real_link(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_polza_media_raw_saved", flaky_link)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-raw")
    )
    assert code == 30
    monkeypatch.setattr(HistoryRepository, "record_polza_media_raw_saved", real_link)

    run_root = tmp_path / "out" / "hist-sync-raw"
    run_uuid = _run_uuid("hist-sync-raw")
    assert (run_root / "raw" / "chunk_01.mp3.receipt.json").exists()

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert provider.submits == ["chunk_01"]


def test_history_sync_blocks_unattempted_part_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """Sync never starts a new paid submit for an unattempted part."""
    from voiceover_pipeline.history.repository import HistoryRepository

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый.", "Второй."])

    real_reserve = HistoryRepository.reserve_paid_tts_attempt
    calls = {"n": 0}

    def flaky_reserve(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash before the second reserve")
        return real_reserve(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", flaky_reserve)
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-unattempted")
    )
    assert code == 30
    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", real_reserve)

    run_uuid = _run_uuid("hist-sync-unattempted")
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNC_PAID_SUBMIT_REQUIRED"
    assert provider.submits == ["chunk_01"]


def test_history_sync_blocks_uncertain_sync_submit_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    """Sync blocks a synchronous submit that was never confirmed, with zero requests."""
    provider = FakeSyncProvider()

    def fail_before_accept(_inner, _text, _chunk_id):
        raise requests.Timeout("read timed out")

    provider.script = fail_before_accept
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Неопределённый платный исход."])
    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, script, "hist-sync-uncertain", model=POLZA_SYNC_MODEL, voice="alloy"
        ),
    )
    assert code == 30
    assert provider.calls == ["chunk_01"]

    run_uuid = _run_uuid("hist-sync-uncertain")
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert provider.calls == ["chunk_01"]


def test_history_sync_fails_closed_on_changed_db_text(tmp_path, monkeypatch, capsys, native_env):
    """A tampered committed text fails the verified view before any provider."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Текст, который потом изменят."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-tamper")
    )
    assert code == 0
    run_uuid = _run_uuid("hist-sync-tamper")

    connection = sqlite3.connect(native_env / "history.sqlite3")
    try:
        connection.execute(
            "UPDATE text_sources SET content = content || 'x' "
            "WHERE kind = 'tts_script' AND part_uuid IS NOT NULL"
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_HISTORY_UNSUPPORTED"
    assert provider.submits == ["chunk_01"]


def test_history_sync_missing_completed_final_audio_fails_closed(
    tmp_path, monkeypatch, capsys, native_env
):
    """A completed run whose final audio is gone fails closed and preserves the DB."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Готовый прогон."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-sync-final")
    )
    assert code == 0
    run_root = tmp_path / "out" / "hist-sync-final"
    run_uuid = _run_uuid("hist-sync-final")
    final_mp3 = next(run_root.glob("*-voiceover-*.mp3"))
    final_mp3.unlink()
    evidence_before = _evidence_counts(native_env, run_uuid)

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 50
    assert payload["details"]["error_code"] == "NATIVE_FINAL_AUDIO_MISSING"
    assert not final_mp3.exists()
    status, _revision = _run_row(native_env, run_uuid)
    assert status == "completed"
    assert _evidence_counts(native_env, run_uuid) == evidence_before


def test_history_sync_and_resume_reject_unknown_and_invalid_identifiers(
    tmp_path, monkeypatch, capsys, native_env
):
    """A malformed id and an unknown UUID fail before any run or write exists."""
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Известный прогон."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-identifiers")
    )
    assert code == 0
    run_uuid = _run_uuid("hist-identifiers")
    evidence_before = _evidence_counts(native_env, run_uuid)

    code, payload = _json_run(monkeypatch, capsys, _history_argv("not-a-uuid", "resume"))
    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_INVALID_RUN_ID"

    unknown = str(uuid.uuid4())
    for verb in ("resume", "sync"):
        code, payload = _json_run(monkeypatch, capsys, _history_argv(unknown, verb))
        assert code == 2
        assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"

    assert _evidence_counts(native_env, run_uuid) == evidence_before
    assert provider.submits == ["chunk_01"]


def test_history_resume_absent_database_creates_nothing(monkeypatch, capsys, native_env):
    """An absent database reports not-found and creates no home, DB, or run."""
    identifier = str(uuid.uuid4())
    code, payload = _json_run(monkeypatch, capsys, _history_argv(identifier, "resume"))

    assert code == 2
    assert payload["details"]["error_code"] == "HISTORY_RUN_NOT_FOUND"
    assert not (native_env / "history.sqlite3").exists()


def test_history_resume_rejects_an_imported_legacy_run(tmp_path, monkeypatch, capsys, native_env):
    """An imported legacy run is refused instead of being reconstructed."""
    from voiceover_pipeline.commands.history import run_history_import

    source = tmp_path / "legacy-source"
    source.mkdir()
    _write_legacy_tree(source)
    imported = run_history_import(source)
    imported_uuid = imported["runs"][0]["run_uuid"]

    from voiceover_pipeline.commands.history import list_history

    count_before = list_history()["count"]
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(cli, "read_api_key", _explode)
    for verb in ("resume", "sync"):
        code, payload = _json_run(monkeypatch, capsys, _history_argv(imported_uuid, verb))
        assert code == 30
        assert payload["details"]["error_code"] == "NATIVE_HISTORY_UNSUPPORTED"
    assert list_history()["count"] == count_before


def test_history_resume_and_sync_are_blocked_by_the_run_lock(
    tmp_path, monkeypatch, capsys, native_env
):
    """A held run lock fails both verbs closed instead of double-writing."""
    from voiceover_pipeline.history.locking import acquire_run_lock

    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Конкурентный писатель."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-lock"))
    assert code == 0
    run_uuid = _run_uuid("hist-lock")
    run_root = tmp_path / "out" / "hist-lock"

    with acquire_run_lock(run_root):
        for verb in ("resume", "sync"):
            code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, verb))
            assert code == 30
            assert payload["details"]["error_code"] == "NATIVE_RUN_LOCKED"


def test_history_sync_output_is_metadata_only(tmp_path, monkeypatch, capsys, native_env):
    """The sync JSON output never echoes committed text, secrets, or signed URLs."""
    marker = "СЕКРЕТНЫЙ_МАРКЕР_ФРАГМЕНТА"
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, [f"Первый {marker}."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "hist-privacy")
    )
    assert code == 0
    run_uuid = _run_uuid("hist-privacy")

    monkeypatch.setattr(sys, "argv", _history_argv(run_uuid, "sync"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    printed = capsys.readouterr().out
    assert marker not in printed
    assert "Authorization" not in printed
    assert "http" not in printed
    assert "signature" not in printed
