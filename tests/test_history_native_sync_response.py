"""Focused tests for the synchronous ``/audio/speech`` pre-parse response boundary.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, a real :class:`PolzaTTSProvider` with a patched ``requests.post``, and
patched FFmpeg seams. No real provider, network call, API key, ``.env``, model, or
paid request is used.

They cover the window this slice closes: a synchronous paid response is persisted
privately *before* the status check and parse, a malformed or errored response
stays private and unknown without a second paid request, and an injected crash
before the decoded raw link replays exactly that same attempt's stored body
locally -- recovering even the observed cost -- with zero POST or GET.
"""

import base64
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.history import raw_receipt
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.services import native_generation

POLZA_SYNC_MODEL = "openai/gpt-4o-mini-tts"
POST_TARGET = "voiceover_pipeline.providers.polza_tts.requests.post"
_POSIX = os.name == "posix"


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


def _generate_argv(tmp_path, script, run_id, *, extra=(), model=POLZA_SYNC_MODEL, voice="alloy"):
    return [
        "voiceover-pipeline",
        "generate",
        "--provider",
        "polza-tts",
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
    return excinfo.value.code, json.loads(capsys.readouterr().out)


def _install_provider(monkeypatch, provider):
    monkeypatch.setattr(cli, "build_provider", lambda *_a, **_k: provider)
    return provider


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("a local replay must not build a provider or read a key")


def _response(body: bytes, status_code: int = 200, headers: dict | None = None):
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response._content = body
    if headers:
        response.headers.update(headers)
    return response


def _speech_body(
    *,
    audio: bytes = b"sync-audio",
    cost: str = "0.3",
    content_type: str = "audio/mpeg",
    generation_id: str = "gen-1",
) -> bytes:
    # The usage number is written unquoted so the provider's ``parse_float=Decimal``
    # path is exercised exactly as a real numeric cost body would be.
    encoded = base64.b64encode(audio).decode()
    return (
        '{"audio": "'
        + encoded
        + '", "contentType": "'
        + content_type
        + '", "usage": {"cost_rub": '
        + cost
        + '}, "id": "'
        + generation_id
        + '"}'
    ).encode()


def _run_root(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "out" / run_id


def _response_files(run_root: Path) -> list[Path]:
    return sorted((run_root / "raw").glob("*.response"))


def _receipt_path(run_root: Path) -> Path:
    files = _response_files(run_root)
    assert len(files) == 1, files
    return files[0].with_name(files[0].name + ".receipt.json")


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    matches = [run for run in list_history()["runs"] if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _history_show(run_uuid: str) -> dict:
    from voiceover_pipeline.commands.history import show_history

    return show_history(run_uuid)


def _attempt_cost(home: Path, run_uuid: str) -> str | None:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        row = connection.execute(
            "SELECT cost FROM attempts WHERE run_uuid = ?", (run_uuid,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return row[0]


def _crash_before_decoded_link(monkeypatch):
    """Make the first decoded-raw write fail, simulating a crash after the sink.

    The synchronous response body has already been persisted by the pre-parse sink
    when this raises, so the attempt stays ``submitting`` with only the private
    response receipt on disk.
    """
    real_write = raw_receipt.write_paid_raw_receipt
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", flaky)
    return real_write


# ── provider-level pre-parse boundary ─────────────────────────────────────────


def test_audio_speech_sink_sees_exact_body_before_parse():
    """The sink is handed the exact body before a malformed payload fails the parse."""
    seen: list[tuple[bytes, int, str | None]] = []

    def sink(body, status_code, generation_id):
        seen.append((body, status_code, generation_id))

    provider = PolzaTTSProvider(
        api_key="sk-test",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
        on_raw_response=sink,
    )
    malformed = b'{"audio": "unterminated'
    from unittest.mock import patch

    with patch(
        POST_TARGET, return_value=_response(malformed, headers={"X-Generation-Id": "gen-x"})
    ):
        with pytest.raises(Exception):
            provider.synthesize_chunk("Hello", "chunk_01")

    assert seen == [(malformed, 200, "gen-x")]


def test_audio_speech_http_error_reports_status_not_body():
    """A provider HTTP error reports only the bounded status, never the body."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    secret = b'{"error": "https://signed.example/secret Bearer sk-leak"}'
    from unittest.mock import patch

    with patch(POST_TARGET, return_value=_response(secret, status_code=500)):
        with pytest.raises(RuntimeError) as excinfo:
            provider.synthesize_chunk("Hello", "chunk_01")

    assert str(excinfo.value) == "HTTP 500"
    assert "signed.example" not in str(excinfo.value)


def test_audio_speech_parser_never_echoes_untrusted_keys_or_content_type():
    """Direct provider errors are fixed, not an inventory of private response data."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    from unittest.mock import patch

    with patch(POST_TARGET, return_value=_response(b'{"Authorization-sk-not-a-key":"private"}')):
        with pytest.raises(RuntimeError, match="missing") as missing:
            provider.synthesize_chunk("Hello", "chunk_01")
    assert "Authorization-sk-not-a-key" not in str(missing.value)

    audio = base64.b64encode(b"audio-bytes").decode("ascii")
    body = json.dumps({"audio": audio, "contentType": "audio/x-sk-not-a-key"}).encode()
    with patch(POST_TARGET, return_value=_response(body)):
        with pytest.raises(RuntimeError, match="unsupported audio content type") as invalid:
            provider.synthesize_chunk("Hello", "chunk_01")
    assert "sk-not-a-key" not in str(invalid.value)


def test_audio_speech_ignores_untrusted_generation_header_and_uses_body_id():
    """A rejected header and stored None produce the same bounded replay identity."""
    from unittest.mock import patch

    body = _speech_body(generation_id="gen-from-body")
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    with patch(
        POST_TARGET,
        return_value=_response(body, headers={"X-Generation-Id": "https://signed.example/key"}),
    ):
        direct = provider.synthesize_chunk("Hello", "chunk_01")
    assert direct.generation_id == "gen-from-body"
    from voiceover_pipeline.providers.polza_tts import build_audio_speech_result

    replay = build_audio_speech_result(
        body=body,
        requested_format="mp3",
        header_generation_id=None,
        transcript="Hello",
        model=POLZA_SYNC_MODEL,
        voice="alloy",
    )
    assert replay.generation_id == direct.generation_id


# ── module-level evidence contract ────────────────────────────────────────────


def _write_args(run_root: Path, attempt: str, **overrides):
    args = {
        "run_root": run_root,
        "attempt_uuid": attempt,
        "part_uuid": str(uuid.uuid4()),
        "synthesis_fingerprint": "a" * 64,
        "chunk_id": "chunk_01",
        "number": 1,
        "provider": "polza-tts",
        "model": POLZA_SYNC_MODEL,
        "voice": "alloy",
        "response_format": "mp3",
        "http_status": 200,
        "body": b'{"audio": "x"}',
    }
    args.update(overrides)
    return args


def test_write_sync_response_receipt_is_idempotent_and_conflicts_on_change(tmp_path):
    """An identical repeat is a no-op; a different body is a cross-attempt conflict."""
    run_root = tmp_path / "out"
    run_root.mkdir()
    attempt = str(uuid.uuid4())
    args = _write_args(run_root, attempt)

    first = raw_receipt.write_sync_response_receipt(**args)
    second = raw_receipt.write_sync_response_receipt(**args)
    assert first.sha256 == second.sha256
    assert first.body_path.read_bytes() == args["body"]
    # The evidence set (this attempt's body and receipt, with no atomic-write
    # leftover) is asserted on every host; Windows stores no meaningful POSIX
    # bits on the new directory, so the private-mode claim is only made where
    # those bits describe real access.
    raw_dir = run_root / "raw"
    assert sorted(item.name for item in raw_dir.iterdir()) == sorted(
        [first.body_path.name, f"{first.body_path.name}.receipt.json"]
    )
    if _POSIX:
        assert not raw_dir.stat().st_mode & 0o077

    with pytest.raises(raw_receipt.PaidSyncResponseConflictError):
        raw_receipt.write_sync_response_receipt(**{**args, "body": b'{"audio": "y"}'})


def test_write_sync_response_receipt_rejects_empty_and_oversized(tmp_path, monkeypatch):
    """An empty or oversized body fails closed instead of being stored."""
    run_root = tmp_path / "out"
    run_root.mkdir()
    args = _write_args(run_root, str(uuid.uuid4()))

    with pytest.raises(ValueError):
        raw_receipt.write_sync_response_receipt(**{**args, "body": b""})

    monkeypatch.setattr(raw_receipt, "MAX_RESPONSE_BODY_BYTES", 4)
    with pytest.raises(raw_receipt.PaidSyncResponseError):
        raw_receipt.write_sync_response_receipt(**args)
    assert not (run_root / "raw").exists()


# ── native executor: persist before parse ─────────────────────────────────────


def test_sync_response_body_is_persisted_before_a_malformed_parse(
    tmp_path, monkeypatch, capsys, native_env
):
    """A malformed body is on disk even though the JSON parse crashed."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    malformed = b'{"audio": "unterminated'
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(malformed)

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Парсер падает."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-parse-crash")
    )

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert len(posts) == 1
    run_root = _run_root(tmp_path, "sync-parse-crash")
    files = _response_files(run_root)
    assert len(files) == 1
    # The exact malformed body survived the parse crash, so the sink ran first.
    assert files[0].read_bytes() == malformed
    assert _receipt_path(run_root).exists()


def test_sync_malformed_response_blocks_resume_without_second_post(
    tmp_path, monkeypatch, capsys, native_env
):
    """A stored-but-unparseable response never authorizes another paid request."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(b"not-json-at-all")

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Неразбираемый ответ."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-unparseable")
    )
    assert code == 30
    assert len(posts) == 1

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-unparseable", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert len(posts) == 1


# ── native executor: local replay ─────────────────────────────────────────────


def test_sync_response_replays_locally_after_crash_and_recovers_cost(
    tmp_path, monkeypatch, capsys, native_env
):
    """A crash before the decoded link replays the stored body with no POST/GET."""
    real_write = _crash_before_decoded_link(monkeypatch)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(_speech_body(cost="0.3"))

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Локальный повтор."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-replay"))
    assert code == 30
    assert len(posts) == 1
    run_root = _run_root(tmp_path, "sync-replay")
    assert len(_response_files(run_root)) == 1
    # The decoded raw receipt was never written in this crash window.
    assert not (run_root / "raw" / "chunk_01.mp3").exists()

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-replay", extra=["--resume"]),
    )

    assert code == 0, payload
    assert len(posts) == 1
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"sync-audio"
    assert (run_root / "raw" / "chunk_01.mp3").read_bytes() == b"sync-audio"
    # The exact observed cost was recovered from the replayed private body.
    assert _attempt_cost(native_env, _run_uuid("sync-replay")) == "0.3"
    assert _history_show(_run_uuid("sync-replay"))["run"]["status"] == "completed"


def test_sync_success_persists_private_response_alongside_decoded_raw(
    tmp_path, monkeypatch, capsys, native_env
):
    """A successful submit keeps both the private body receipt and the decoded raw."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    body = _speech_body()
    monkeypatch.setattr(POST_TARGET, lambda _url, **_kwargs: _response(body))
    script = _script(tmp_path, ["Успешный ответ."])

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-success")
    )

    assert code == 0
    run_root = _run_root(tmp_path, "sync-success")
    assert _response_files(run_root)[0].read_bytes() == body
    assert _receipt_path(run_root).exists()
    assert (run_root / "raw" / "chunk_01.mp3").exists()
    assert _attempt_cost(native_env, _run_uuid("sync-success")) == "0.3"


# ── native executor: fail-closed evidence ─────────────────────────────────────


def _crash_state(tmp_path, monkeypatch, capsys, run_id: str) -> Path:
    """Produce a ``submitting`` attempt that carries only the private response body."""
    _crash_before_decoded_link(monkeypatch)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    body = _speech_body()
    monkeypatch.setattr(POST_TARGET, lambda _url, **_kwargs: _response(body))
    script = _script(tmp_path, ["Окно краша."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, run_id))
    assert code == 30
    return script


def test_sync_response_identity_mismatch_blocks_resume(tmp_path, monkeypatch, capsys, native_env):
    """A receipt written for a different request identity never replays."""
    script = _crash_state(tmp_path, monkeypatch, capsys, "sync-identity")
    run_root = _run_root(tmp_path, "sync-identity")
    receipt = json.loads(_receipt_path(run_root).read_text(encoding="utf-8"))
    receipt["model"] = "openai/other-model"
    _receipt_path(run_root).write_text(json.dumps(receipt), encoding="utf-8")

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-identity", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"


def test_sync_response_tampered_body_blocks_resume(tmp_path, monkeypatch, capsys, native_env):
    """Tampered body bytes fail digest verification instead of being replayed."""
    script = _crash_state(tmp_path, monkeypatch, capsys, "sync-tamper")
    run_root = _run_root(tmp_path, "sync-tamper")
    _response_files(run_root)[0].write_bytes(b'{"audio": "dGFtcGVyZWQ="}')

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-tamper", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"


def test_sync_response_oversized_on_disk_blocks_resume(tmp_path, monkeypatch, capsys, native_env):
    """A receipt that claims an oversized body fails closed before replay."""
    script = _crash_state(tmp_path, monkeypatch, capsys, "sync-oversize")
    run_root = _run_root(tmp_path, "sync-oversize")
    receipt = json.loads(_receipt_path(run_root).read_text(encoding="utf-8"))
    receipt["size"] = raw_receipt.MAX_RESPONSE_BODY_BYTES + 1
    _receipt_path(run_root).write_text(json.dumps(receipt), encoding="utf-8")

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-oversize", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"


def test_sync_response_orphan_raw_conflict_blocks_without_adoption(
    tmp_path, monkeypatch, capsys, native_env
):
    """An orphan decoded-raw file that is not this attempt's is never adopted."""
    script = _crash_state(tmp_path, monkeypatch, capsys, "sync-orphan")
    run_root = _run_root(tmp_path, "sync-orphan")
    # A foreign decoded raw file with no receipt half: it cannot be reconciled to
    # this attempt and must block instead of being converted or overwritten.
    (run_root / "raw" / "chunk_01.mp3").write_bytes(b"foreign-paid-bytes")

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-orphan", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert (run_root / "raw" / "chunk_01.mp3").read_bytes() == b"foreign-paid-bytes"
    assert not (run_root / "chunks" / "chunk_01.mp3").exists()


def test_sync_http_error_keeps_body_private_and_blocks_retry(
    tmp_path, monkeypatch, capsys, native_env
):
    """An HTTP error body stays private, never leaks, and never triggers a retry."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    secret = b'{"error": "https://signed.example/leak sk-secret-text"}'
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(secret, status_code=503)

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Ошибка сервера."])

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-http-error")
    )

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert len(posts) == 1
    emitted = json.dumps(payload)
    assert "signed.example" not in emitted
    assert "sk-secret-text" not in emitted
    run_root = _run_root(tmp_path, "sync-http-error")
    # The error body is retained privately as evidence, never publicly.
    assert _response_files(run_root)[0].read_bytes() == secret

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-http-error", extra=["--resume"]),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert len(posts) == 1


def test_sync_response_format_mismatch_blocks_replay_without_a_new_post(
    tmp_path, monkeypatch, capsys, native_env
):
    """A receipt cannot supply a new decode format when the provider omitted contentType."""
    _crash_before_decoded_link(monkeypatch)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    audio = base64.b64encode(b"format-bound-audio").decode("ascii")
    body = json.dumps({"audio": audio, "usage": {"cost_rub": 0.3}}).encode("utf-8")
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(body)

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Формат в запросе неизменен."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-format-mismatch")
    )
    assert code == 30
    receipt_path = _receipt_path(_run_root(tmp_path, "sync-format-mismatch"))
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored["response_format"] == "mp3"
    stored["response_format"] = "pcm"
    receipt_path.write_text(json.dumps(stored), encoding="utf-8")

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sync-format-mismatch", extra=["--resume"]),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert len(posts) == 1


def test_sync_response_cost_survives_crash_after_decoded_raw_before_db_link(
    tmp_path, monkeypatch, capsys, native_env
):
    """When the raw audio receipt exists, replay still observes the same paid cost."""
    from voiceover_pipeline.history.repository import HistoryRepository

    real_record = HistoryRepository.record_polza_sync_raw_saved
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected crash before DB link")
        return real_record(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "record_polza_sync_raw_saved", fail_once)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(_speech_body(cost="0.3"))

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Стоимость после raw receipt."])
    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-late-crash")
    )
    assert code == 30
    run_root = _run_root(tmp_path, "sync-late-crash")
    assert (run_root / "raw" / "chunk_01.mp3").exists()
    assert _receipt_path(run_root).exists()
    assert len(posts) == 1

    monkeypatch.setattr(HistoryRepository, "record_polza_sync_raw_saved", real_record)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-late-crash", extra=["--resume"])
    )
    assert code == 0, payload
    assert len(posts) == 1
    assert _attempt_cost(native_env, _run_uuid("sync-late-crash")) == "0.3"


def test_sync_response_with_object_audio_blocks_resume_without_generic_error(
    tmp_path, monkeypatch, capsys, native_env
):
    """Malformed truthy JSON audio is a fixed parse failure, not a TypeError leak."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    body = b'{"audio":{"Authorization":"sk-not-a-real-key"},"contentType":"audio/mpeg"}'
    posts: list[str] = []

    def post(url, **_kwargs):
        posts.append(url)
        return _response(body)

    monkeypatch.setattr(POST_TARGET, post)
    script = _script(tmp_path, ["Неверный тип audio."])
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-bad-audio")
    )
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert "sk-not-a-real-key" not in json.dumps(payload)

    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sync-bad-audio", extra=["--resume"])
    )
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert "sk-not-a-real-key" not in json.dumps(payload)
    assert len(posts) == 1


# ── per-part voice on the synchronous route ───────────────────────────────────


def _receipts(run_root: Path) -> list[dict]:
    """Return every stored private sync-response receipt under ``raw/``."""
    return [
        json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
        for path in _response_files(run_root)
    ]


def _speech_parts_script(tmp_path: Path) -> Path:
    """Write a two-voice speech-parts script with no instruction (no vibe)."""
    path = tmp_path / "parts.yaml"
    path.write_text(
        "version: 1\nformat: speech-parts\nparts:\n"
        "  - voice: Kore\n    text: |-\n      Первая реплика.\n"
        "  - voice: Puck\n    text: |-\n      Вторая реплика.\n",
        encoding="utf-8",
    )
    return path


def _speech_parts_argv(tmp_path, script, run_id):
    return [
        "voiceover-pipeline",
        "generate",
        "--script",
        str(script),
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        POLZA_SYNC_MODEL,
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        "--json",
    ]


def _voice_named_post(posts: list[str]):
    """Patch target that records the request voice and echoes it as the audio."""

    def post(_url, **kwargs):
        voice = kwargs["json"]["voice"]
        posts.append(voice)
        return _response(_speech_body(audio=voice.encode()))

    return post


def test_audio_speech_direct_call_retains_per_part_voice_over_run_default():
    """The synchronous call honors a per-part voice and defaults to the run voice."""
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    seen: list[str] = []

    from unittest.mock import patch

    with patch(POST_TARGET, _voice_named_post(seen)):
        default = provider.synthesize_chunk("Hello", "chunk_01")
        override = provider.synthesize_chunk("Hello", "chunk_01", voice="nova")

    assert seen == ["alloy", "nova"]
    assert default.raw_metadata["voice"] == "alloy"
    assert override.raw_metadata["voice"] == "nova"


def test_audio_speech_media_route_refuses_a_per_part_voice_override():
    """ElevenLabs ``/media`` fails closed instead of silently ignoring a new voice."""
    provider = PolzaTTSProvider(
        api_key="sk-test", model="elevenlabs/text-to-speech-turbo-2-5", voice="Rachel"
    )
    with pytest.raises(ValueError, match="per-part voice"):
        provider.synthesize_chunk("Hello", "chunk_01", voice="alloy")


def test_speech_parts_two_voices_bind_each_cast_voice_to_post_and_receipt(
    tmp_path, monkeypatch, capsys, native_env
):
    """Each speech-parts part posts and privately receipts its own cast voice."""
    monkeypatch.setattr(cli, "require_confirmed_speech_parts_route", lambda **_kwargs: None)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    posts: list[str] = []
    monkeypatch.setattr(POST_TARGET, _voice_named_post(posts))
    script = _speech_parts_script(tmp_path)

    code, payload = _json_run(
        monkeypatch, capsys, _speech_parts_argv(tmp_path, script, "sp-voices")
    )

    assert code == 0, payload
    assert posts == ["Kore", "Puck"]
    run_root = _run_root(tmp_path, "sp-voices")
    assert {receipt["number"]: receipt["voice"] for receipt in _receipts(run_root)} == {
        1: "Kore",
        2: "Puck",
    }
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"Kore"
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"Puck"


def _crash_on_nth_decoded_write(monkeypatch, nth: int):
    """Make the ``nth`` decoded-raw write fail, simulating a crash after the sink."""
    real_write = raw_receipt.write_paid_raw_receipt
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == nth:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", flaky)
    return real_write


def test_speech_parts_second_part_crash_replays_locally_without_second_post(
    tmp_path, monkeypatch, capsys, native_env
):
    """A crash after the second stored response replays that part locally, one voice each."""
    monkeypatch.setattr(cli, "require_confirmed_speech_parts_route", lambda **_kwargs: None)
    real_write = _crash_on_nth_decoded_write(monkeypatch, 2)
    provider = PolzaTTSProvider(api_key="sk-test", model=POLZA_SYNC_MODEL, voice="alloy")
    _install_provider(monkeypatch, provider)
    posts: list[str] = []
    monkeypatch.setattr(POST_TARGET, _voice_named_post(posts))
    script = _speech_parts_script(tmp_path)

    code, _payload = _json_run(
        monkeypatch, capsys, _speech_parts_argv(tmp_path, script, "sp-crash")
    )
    assert code == 30
    assert posts == ["Kore", "Puck"]
    run_root = _run_root(tmp_path, "sp-crash")
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"Kore"
    assert not (run_root / "chunks" / "chunk_02.mp3").exists()
    assert len(_response_files(run_root)) == 2

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(cli, "build_provider", _explode)
    monkeypatch.setattr(POST_TARGET, _explode)
    run_uuid = _run_uuid("sp-crash")
    # The original script is gone, so the resume can only rebuild from the snapshot.
    script.unlink()
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0, capsys.readouterr().out
    # Only the stored second body replayed locally: no third POST was sent.
    assert posts == ["Kore", "Puck"]
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"Puck"
    assert (run_root / "raw" / "chunk_02.mp3").read_bytes() == b"Puck"
