"""The deprecated ``--allow-experimental-gemini-speech-parts`` compatibility spelling.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, the real provider with a patched ``requests.post``, and patched FFmpeg
seams. No real provider, network call, API key, ``.env``, model, or paid request is
used.

Gemini 3.8 speech support is ordinary, so the flag no longer admits anything: it is
accepted and recorded with the run's output options (so an existing run and its
resume keep the same identity), it is never required, and it never refuses a target
that is otherwise supported -- the same Gemini 3.8 run works with or without it, and
it does not turn an unrelated script format or model into an error.
"""

import base64
import json
import socket
import sys
from pathlib import Path

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider

GEMINI_38_FLASH = "google/gemini-3.8-flash-tts"
GEMINI_38_FLASH_LITE = "google/gemini-3.8-flash-lite-tts"
STABLE_POLZA_MODEL = "openai/gpt-4o-mini-tts"
DEPRECATED_FLAG = "--allow-experimental-gemini-speech-parts"
POLZA_POST_TARGET = "voiceover_pipeline.providers.polza_tts.requests.post"

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


def _speech_body(*, audio: bytes = WAV_BYTES, cost: str = "0.3") -> bytes:
    """One ``/audio/speech`` body that returns WAV despite the requested MP3."""
    encoded = base64.b64encode(audio).decode()
    return (
        '{"audio": "'
        + encoded
        + '", "contentType": "audio/wav", "usage": {"cost_rub": '
        + cost
        + '}, "id": "gen-1"}'
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


def _generate_argv(tmp_path, script, run_id, *, extra=(), model=GEMINI_38_FLASH):
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


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    matches = [run for run in list_history()["runs"] if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def _output_options(home: Path, run_uuid: str) -> dict:
    import sqlite3

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

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _response(_speech_body(), self.status_code)


# ── the flag never gates a supported target ───────────────────────────────────


@pytest.mark.parametrize("model", [GEMINI_38_FLASH, GEMINI_38_FLASH_LITE])
def test_supported_gemini38_run_needs_no_flag(
    tmp_path, monkeypatch, capsys, native_env, socket_denied, model
):
    """The ordinary Polza run submits the same two scalar POSTs with no opt-in."""
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, _parts_script(tmp_path), "no-flag", model=model),
    )

    assert code == 0, payload
    assert len(post.calls) == 2
    assert [call["json"]["voice"] for call in post.calls] == ["Kore", "Puck"]
    run_root = _run_root(tmp_path, "no-flag")
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == WAV_BYTES
    # A run that never passed the deprecated spelling records no such option, so a
    # later resume of an ordinary run is never invalidated by it.
    assert "experimental_speech_parts" not in _output_options(native_env, _run_uuid("no-flag"))


def test_deprecated_flag_is_accepted_and_recorded_for_resume(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """The compatibility spelling still runs and is still part of the run identity."""
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, _parts_script(tmp_path), "with-flag", extra=[DEPRECATED_FLAG]),
    )

    assert code == 0, payload
    assert len(post.calls) == 2
    run_uuid = _run_uuid("with-flag")
    assert _output_options(native_env, run_uuid)["experimental_speech_parts"] is True

    # A resume of that same run replays it locally and never re-submits.
    monkeypatch.setattr(POLZA_POST_TARGET, RecordingPost())
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0, capsys.readouterr().out


def test_deprecated_flag_with_a_stable_single_voice_model_is_supported(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """The flag no longer refuses a supported model that simply cannot carry a vibe."""
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)
    script = tmp_path / "no-vibe.yaml"
    script.write_text(
        "version: 1\nformat: speech-parts\nparts:\n"
        "  - voice: alloy\n    text: |-\n      Одна реплика.\n",
        encoding="utf-8",
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "stable-flag",
            extra=[DEPRECATED_FLAG],
            model=STABLE_POLZA_MODEL,
        ),
    )

    assert code == 0, payload
    assert len(post.calls) == 1
    assert "instructions" not in post.calls[0]["json"]


def test_deprecated_flag_on_a_legacy_script_is_ignored_not_refused(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A legacy script format stays supported; the flag is not a usage error."""
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)
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
            DEPRECATED_FLAG,
            "--json",
        ],
    )

    assert code == 0, payload
    assert len(post.calls) == 1
    assert post.calls[0]["json"]["input"] == "Обычный текст."


def test_resume_without_the_recorded_option_refuses_before_any_post(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A resume that drops the recorded deprecated option is refused, not re-submitted."""
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)
    script = _parts_script(tmp_path)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sp-repolicy", extra=[DEPRECATED_FLAG]),
    )
    assert code == 0, payload
    assert len(post.calls) == 2

    second = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, second)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "sp-repolicy", extra=["--resume"]),
    )

    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_PROCESSING_UNSUPPORTED"
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


def test_validate_reports_the_ordinary_route_with_or_without_the_flag(
    tmp_path, capsys, monkeypatch
):
    """The flag changes no reported fact: the Gemini 3.8 route is admitted either way."""
    script = _parts_script(tmp_path)
    without = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        GEMINI_38_FLASH,
    )
    with_flag = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        GEMINI_38_FLASH,
        DEPRECATED_FLAG,
    )
    assert without == with_flag
    assert without[0] == 0
    assert without[1]["route"] == {"admitted": True, "reason": None}


def test_validate_accepts_the_deprecated_flag_on_a_markdown_script(
    tmp_path, capsys, monkeypatch, socket_denied
):
    """The deprecated spelling is not a usage error on an otherwise supported script."""
    script = tmp_path / "narration.md"
    script.write_text("Обычный текст.", encoding="utf-8")

    code, report = _run_validate(
        capsys, monkeypatch, script, "--format", "markdown", DEPRECATED_FLAG
    )

    assert code == 0
    assert report["valid"] is True


# ── provider contract (pure, patched transport) ───────────────────────────────


def test_sync_request_carries_the_instruction_separately_from_the_spoken_text(monkeypatch):
    post = RecordingPost()
    monkeypatch.setattr(POLZA_POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=GEMINI_38_FLASH, voice="Kore")

    provider.synthesize_chunk("Привет.", "chunk_01", voice="Puck", vibe="Спокойно.")

    payload = post.calls[0]["json"]
    assert payload["input"] == "Привет."
    assert payload["voice"] == "Puck"
    assert payload["instructions"] == "Спокойно."
    assert payload["model"] == GEMINI_38_FLASH


def test_sync_submit_does_not_follow_a_redirect(monkeypatch):
    """A 3xx stays one observed response instead of a repeated paid POST."""
    post = RecordingPost(status_code=302)
    monkeypatch.setattr(POLZA_POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=GEMINI_38_FLASH, voice="Kore")

    with pytest.raises(RuntimeError, match="HTTP 302"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")

    assert len(post.calls) == 1
    assert post.calls[0]["allow_redirects"] is False
