"""Opt-in experimental Polza Gemini speech-parts route: one scalar POST per part.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, the real :class:`PolzaTTSProvider` with a patched ``requests.post``, a
denied socket, and patched FFmpeg seams. No real provider, network call, API key,
``.env``, model, or paid request is used.

They cover the one explicitly opt-in experimental contract: the candidate
``google/gemini-3.8-flash-tts`` model stays refused by default, and only the
ordinary-CLI opt-in flag admits it as one scalar-voice POST per speech-parts part,
with the effective instruction carried separately from the spoken text, the
private pre-parse response body and receipt retained, the observed WAV converted
to the canonical MP3 chunk, and a resume replaying the stored body with no second
POST.
"""

import base64
import json
import socket
import sqlite3
import sys
from pathlib import Path

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import POLZA_EXPERIMENTAL_GEMINI_SPEECH_PARTS_MODEL
from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.search.lexical import roles_for_scope, search_lexical
from voiceover_pipeline.services import native_generation

CANDIDATE_MODEL = POLZA_EXPERIMENTAL_GEMINI_SPEECH_PARTS_MODEL
FLASH_LITE_MODEL = "google/gemini-3.8-flash-lite-tts"
STABLE_POLZA_MODEL = "openai/gpt-4o-mini-tts"
OPT_IN = "--allow-experimental-gemini-speech-parts"
POST_TARGET = "voiceover_pipeline.providers.polza_tts.requests.post"

WAV_BYTES = b"RIFF" + (0).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 8

PARTS_SCRIPT = """\
version: 1
format: speech-parts
vibe: >-
  Образовательный подкаст о поиске по истории.
  Живая разговорная речь.
parts:
  - voice: Kore
    vibe: С любопытством.
    text: |-
      Уникальноеслово расскажет о поиске по истории.
  - voice: Puck
    text: |-
      Тогда запись найдётся по содержанию.
"""


# ── offline environment ───────────────────────────────────────────────────────


def _deny_socket(*_args, **_kwargs):  # pragma: no cover - asserted on failure only
    raise AssertionError("this offline test denies all network access")


@pytest.fixture
def socket_denied(monkeypatch):
    """Make any real outbound socket connection fail loudly."""
    monkeypatch.setattr(socket.socket, "connect", _deny_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny_socket)
    monkeypatch.setattr(socket, "create_connection", _deny_socket)


def _write_mp3(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


@pytest.fixture
def native_env(tmp_path, monkeypatch):
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
    audio: bytes = WAV_BYTES,
    cost: str | None = "0.3",
    content_type: str = "audio/wav",
    generation_id: str = "gen-1",
) -> bytes:
    """One ``/audio/speech`` body that returns WAV despite the requested MP3.

    ``cost=None`` omits the usage block exactly as a provider response without an
    observed amount would, so the stored cost stays unknown instead of zero.
    """
    encoded = base64.b64encode(audio).decode()
    usage = "" if cost is None else f', "usage": {{"cost_rub": {cost}}}'
    return (
        '{"audio": "'
        + encoded
        + '", "contentType": "'
        + content_type
        + '"'
        + usage
        + ', "id": "'
        + generation_id
        + '"}'
    ).encode()


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


def _parts_script(tmp_path: Path) -> Path:
    path = tmp_path / "parts.yaml"
    path.write_text(PARTS_SCRIPT, encoding="utf-8")
    return path


def _generate_argv(tmp_path, script, run_id, *, extra=(), model=CANDIDATE_MODEL):
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
        model,
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]


def _run_root(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "out" / run_id


def _response_files(run_root: Path) -> list[Path]:
    return sorted((run_root / "raw").glob("*.response"))


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    matches = [run for run in list_history()["runs"] if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _attempt_rows(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT status, cost, cost_source FROM attempts WHERE run_uuid = ?", (run_uuid,)
        ).fetchall()
    finally:
        connection.close()


def _stored_parts(home: Path, run_uuid: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT position, voice, prepared_text, vibe_effective FROM parts "
            "WHERE run_uuid = ? ORDER BY position",
            (run_uuid,),
        ).fetchall()
    finally:
        connection.close()


def _output_options(home: Path, run_uuid: str) -> dict:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        row = connection.execute(
            "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return json.loads(row[0])["output"]


class RecordingPost:
    """A patched ``requests.post`` that records the exact request payloads."""

    def __init__(
        self,
        *,
        audio: bytes = WAV_BYTES,
        cost: str | None = "0.3",
        status_code: int = 200,
    ) -> None:
        self.audio = audio
        self.cost = cost
        self.status_code = status_code
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, "json": kwargs.get("json"), **kwargs})
        return _response(_speech_body(audio=self.audio, cost=self.cost), self.status_code)


# ── default refusal and the explicit opt-in ───────────────────────────────────


def test_candidate_route_is_refused_without_the_opt_in_flag(
    tmp_path, monkeypatch, capsys, native_env
):
    """The default CLI still fails closed before a key, a provider, or a POST."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    built: list[str] = []

    def explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
        built.append("provider")
        raise AssertionError("the candidate route must be refused before any provider")

    monkeypatch.setattr(cli, "build_provider", explode)
    monkeypatch.setattr(cli, "read_api_key", explode)

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, _parts_script(tmp_path), "no-opt-in")
    )

    assert code == 30
    assert payload["details"]["error_code"] == "BLOCKED_PROVIDER_CONTRACT"
    assert post.calls == []
    assert built == []
    assert not _run_root(tmp_path, "no-opt-in").exists()


def test_flash_lite_stays_refused_even_with_the_opt_in_flag(
    tmp_path, monkeypatch, capsys, native_env
):
    """A Flash-Lite id is neither the candidate nor admitted by the opt-in flag."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path, _parts_script(tmp_path), "flash-lite", extra=[OPT_IN], model=FLASH_LITE_MODEL
        ),
    )
    assert code != 0
    assert post.calls == []
    assert not _run_root(tmp_path, "flash-lite").exists()


def test_opt_in_flag_with_a_stable_model_is_a_usage_error(
    tmp_path, monkeypatch, capsys, native_env
):
    """The opt-in never turns a stable model into a per-part instruction route."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            _parts_script(tmp_path),
            "stable-model",
            extra=[OPT_IN],
            model=STABLE_POLZA_MODEL,
        ),
    )
    assert code == 2
    assert post.calls == []
    assert not _run_root(tmp_path, "stable-model").exists()


def test_opt_in_flag_without_a_speech_parts_format_is_a_usage_error(
    tmp_path, monkeypatch, capsys, native_env
):
    """The flag is refused on a legacy markdown script instead of being ignored."""
    script = tmp_path / "narration.md"
    script.write_text("Обычный текст.", encoding="utf-8")
    code, payload = _json_run(
        monkeypatch,
        capsys,
        [
            "voiceover-pipeline",
            "generate",
            "--script",
            str(script),
            "--provider",
            "polza-tts",
            "--model",
            STABLE_POLZA_MODEL,
            "--voice",
            "alloy",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "legacy-flag",
            OPT_IN,
            "--json",
        ],
    )
    assert code == 2
    assert not _run_root(tmp_path, "legacy-flag").exists()


# ── the admitted experimental run ─────────────────────────────────────────────


def test_opt_in_posts_one_scalar_voice_and_separate_instructions_per_part(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """One DB-first run, one MP3, and one scalar POST per part carrying its own text."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    formats: list[str] = []

    def _record_write(_ffmpeg: str, data: bytes, fmt: str, path: Path) -> None:
        formats.append(fmt)
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", _record_write)

    script = _parts_script(tmp_path)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sp-experimental", extra=[OPT_IN])
    )

    assert code == 0, payload
    assert payload["status"] == "success"
    assert len(post.calls) == 2
    assert [call["json"]["voice"] for call in post.calls] == ["Kore", "Puck"]
    assert [call["json"]["model"] for call in post.calls] == [CANDIDATE_MODEL] * 2
    assert [call["json"]["response_format"] for call in post.calls] == ["mp3", "mp3"]
    # Each request speaks exactly its own part text; the effective instruction is a
    # separate JSON field and is never part of the spoken ``input``.
    assert post.calls[0]["json"]["input"] == ("Уникальноеслово расскажет о поиске по истории.")
    assert post.calls[1]["json"]["input"] == "Тогда запись найдётся по содержанию."
    assert "С любопытством." in post.calls[0]["json"]["instructions"]
    assert (
        post.calls[1]["json"]["instructions"]
        == post.calls[0]["json"]["instructions"].split("\n\n")[0]
    )
    for call in post.calls:
        assert call["json"]["input"] not in call["json"]["instructions"]
        assert call["url"].endswith("/audio/speech")

    run_root = _run_root(tmp_path, "sp-experimental")
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == WAV_BYTES
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == WAV_BYTES
    # The provider requested MP3 but returned a real RIFF/WAVE body; the app detects
    # the observed container and transcodes it, so the chunk conversion sees WAV.
    assert formats == ["wav", "wav"]

    # Private pre-parse response bodies and their bounded receipts are retained.
    assert len(_response_files(run_root)) == 2
    receipts = [
        json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
        for path in _response_files(run_root)
    ]
    assert {receipt["number"]: receipt["voice"] for receipt in receipts} == {1: "Kore", 2: "Puck"}
    assert all(receipt["model"] == CANDIDATE_MODEL for receipt in receipts)

    run_uuid = _run_uuid("sp-experimental")
    attempts = _attempt_rows(native_env, run_uuid)
    assert [row["status"] for row in attempts] == ["completed", "completed"]
    assert [row["cost_source"] for row in attempts] == ["exact", "exact"]
    assert [row["cost"] for row in attempts] == ["0.3", "0.3"]
    # The opt-in policy is part of the recorded run so a resume cannot switch it.
    assert _output_options(native_env, run_uuid)["experimental_speech_parts"] is True
    parts = _stored_parts(native_env, run_uuid)
    assert [row["voice"] for row in parts] == ["Kore", "Puck"]
    assert "С любопытством." in parts[0]["vibe_effective"]


def test_experimental_run_without_reported_usage_keeps_cost_unknown(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A response with no usage block never invents a zero cost."""
    post = RecordingPost(cost=None)
    monkeypatch.setattr(POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, _parts_script(tmp_path), "sp-unknown-cost", extra=[OPT_IN]),
    )

    assert code == 0, payload
    assert len(post.calls) == 2
    run_uuid = _run_uuid("sp-unknown-cost")
    attempts = _attempt_rows(native_env, run_uuid)
    assert [row["cost"] for row in attempts] == [None, None]
    assert all(row["cost_source"] != "exact" for row in attempts)
    assert payload["cost"] == {"total": None, "currency": None}


def test_experimental_run_is_searchable_in_fts5(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """The committed script and direction reach the derived FTS5 index immediately."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, _parts_script(tmp_path), "sp-search", extra=[OPT_IN]),
    )
    assert code == 0, payload
    run_uuid = _run_uuid("sp-search")

    with HistoryDatabase(native_env / "history.sqlite3") as database:
        database.connect()
        found = search_lexical(
            database.connection,
            "уникальноеслово",
            limit=20,
            roles=roles_for_scope("speech"),
            run_uuid=run_uuid,
        )
        directions = search_lexical(
            database.connection,
            "любопытством",
            limit=20,
            roles=roles_for_scope("directions"),
            run_uuid=run_uuid,
        )
    assert found["results"], found
    assert directions["results"], directions


def test_resume_replays_the_stored_body_without_a_second_post(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A crash after the second response replays it locally, never a third POST."""
    real_write = native_generation.write_paid_raw_receipt
    calls = {"n": 0}

    def _crash_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", _crash_on_second)
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    script = _parts_script(tmp_path)

    code, _payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sp-crash", extra=[OPT_IN])
    )
    assert code == 30
    assert len(post.calls) == 2
    run_root = _run_root(tmp_path, "sp-crash")
    assert (run_root / "chunks" / "chunk_01.mp3").exists()
    assert not (run_root / "chunks" / "chunk_02.mp3").exists()

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(POST_TARGET, RecordingPost())
    run_uuid = _run_uuid("sp-crash")
    # The original script is gone, so the resume rebuilds from the snapshot alone.
    script.unlink()
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0, capsys.readouterr().out
    assert len(post.calls) == 2
    attempts = _attempt_rows(native_env, run_uuid)
    assert [row["status"] for row in attempts] == ["completed", "completed"]
    assert _output_options(native_env, run_uuid)["experimental_speech_parts"] is True


def test_resume_without_the_opt_in_flag_refuses_before_any_post(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A later command cannot continue the experimental run without the opt-in."""
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    script = _parts_script(tmp_path)
    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "sp-repolicy", extra=[OPT_IN])
    )
    assert code == 0, payload
    assert len(post.calls) == 2

    script.write_text(
        PARTS_SCRIPT.replace("запись найдётся", "запись найдётся снова"), encoding="utf-8"
    )
    second = RecordingPost()
    monkeypatch.setattr(POST_TARGET, second)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sp-repolicy", extra=["--resume"]),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "BLOCKED_PROVIDER_CONTRACT"
    assert second.calls == []


# ── validate ──────────────────────────────────────────────────────────────────


def _run_validate(capsys, monkeypatch, script: Path, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover-pipeline", "validate", "--script", str(script), *extra, "--json"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


def test_validate_reports_the_experimental_route_as_admitted_with_the_flag(
    tmp_path, capsys, monkeypatch
):
    script = _parts_script(tmp_path)
    code, report = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        CANDIDATE_MODEL,
        OPT_IN,
    )
    assert code == 0
    assert report["valid"] is True
    assert report["route"] == {"admitted": True, "reason": None}


def test_validate_json_shape_is_unchanged_without_the_flag(tmp_path, capsys, monkeypatch):
    """The default report keeps its exact keys and the blocked route."""
    script = _parts_script(tmp_path)
    code, report = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        CANDIDATE_MODEL,
    )
    assert code == 0
    assert set(report) == {
        "status",
        "valid",
        "format",
        "script",
        "provider",
        "model",
        "request_char_limit",
        "parts",
        "total_chars",
        "route",
        "errors",
        "warnings",
    }
    assert report["route"] == {"admitted": False, "reason": "BLOCKED_PROVIDER_CONTRACT"}


def test_validate_rejects_experimental_opt_in_on_markdown(
    tmp_path, capsys, monkeypatch, socket_denied
):
    script = tmp_path / "narration.md"
    script.write_text("Обычный текст.", encoding="utf-8")
    code, report = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "markdown",
        "--provider",
        "polza-tts",
        "--model",
        CANDIDATE_MODEL,
        OPT_IN,
    )
    assert code == 2
    assert report["details"]["error_code"] == "EXPERIMENTAL_SPEECH_PARTS_FORMAT_UNSUPPORTED"


# ── provider contract (pure, patched transport) ───────────────────────────────


def test_sync_request_carries_the_instruction_separately_from_the_spoken_text(monkeypatch):
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=CANDIDATE_MODEL, voice="Kore")

    provider.synthesize_chunk("Привет.", "chunk_01", voice="Puck", vibe="Спокойно.")

    payload = post.calls[0]["json"]
    assert payload["input"] == "Привет."
    assert payload["voice"] == "Puck"
    assert payload["instructions"] == "Спокойно."
    assert payload["model"] == CANDIDATE_MODEL


def test_stable_sync_model_refuses_an_instruction_before_any_post(monkeypatch):
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=STABLE_POLZA_MODEL, voice="alloy")

    with pytest.raises(ValueError, match="instructions"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")

    assert post.calls == []


def test_media_route_refuses_an_instruction_before_any_post(monkeypatch):
    post = RecordingPost()
    monkeypatch.setattr(POST_TARGET, post)
    provider = PolzaTTSProvider(
        api_key="sk-test", model="elevenlabs/text-to-speech-turbo-2-5", voice="Rachel"
    )

    with pytest.raises(ValueError, match="instruction"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")

    assert post.calls == []


def test_sync_submit_does_not_follow_a_redirect(monkeypatch):
    """A 3xx stays one observed response instead of a repeated paid POST."""
    post = RecordingPost(status_code=302)
    monkeypatch.setattr(POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=CANDIDATE_MODEL, voice="Kore")

    with pytest.raises(RuntimeError, match="HTTP 302"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")

    assert len(post.calls) == 1
    assert post.calls[0]["allow_redirects"] is False


def test_experimental_run_stops_on_a_redirect_with_one_post_and_one_receipt(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A redirected first part sends exactly one POST and keeps its private body."""
    post = RecordingPost(status_code=302)
    monkeypatch.setattr(POST_TARGET, post)

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, _parts_script(tmp_path), "sp-redirect", extra=[OPT_IN]),
    )

    assert code == 30
    assert len(post.calls) == 1
    run_root = _run_root(tmp_path, "sp-redirect")
    assert len(_response_files(run_root)) == 1
    assert not (run_root / "chunks" / "chunk_01.mp3").exists()
