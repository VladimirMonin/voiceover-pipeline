"""Ordinary Gemini 3.8 speech support on both cloud speech providers.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, the real provider with a patched ``requests.post``, a denied socket, and
patched FFmpeg seams. No real provider, network call, API key, ``.env``, model, or
paid request is used.

They cover the ordinary contract: both Gemini 3.8 ids are listed and admitted for
``polza-tts`` and ``openrouter-tts`` with no opt-in flag, each part is one request
with exactly one scalar voice and its direction in a separate ``instructions``
field, the private pre-parse response body and its bounded receipt are retained for
both providers, an OpenRouter run replays that stored body locally instead of
repeating the paid POST, and ``validate`` applies the same model admission as
``generate``.
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
from voiceover_pipeline import config
from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
from voiceover_pipeline.services import native_generation

GEMINI_38_FLASH = "google/gemini-3.8-flash-tts"
GEMINI_38_FLASH_LITE = "google/gemini-3.8-flash-lite-tts"
GEMINI_38_MODELS = (GEMINI_38_FLASH, GEMINI_38_FLASH_LITE)
GEMINI_31_MODEL = "google/gemini-3.1-flash-tts-preview"
STABLE_POLZA_MODEL = "openai/gpt-4o-mini-tts"
POLZA_POST_TARGET = "voiceover_pipeline.providers.polza_tts.requests.post"
OPENROUTER_POST_TARGET = "voiceover_pipeline.providers.openrouter_tts.requests.post"
PCM_BYTES = b"\x01\x02" * 4800
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"compressed-mp3-payload" * 40

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


def _polza_body(
    *,
    audio: bytes = b"polza-gemini-audio",
    cost: str | None = "0.3",
    content_type: str = "audio/mpeg",
    generation_id: str = "gen-1",
) -> bytes:
    """One Polza ``/audio/speech`` JSON body carrying base64 audio."""
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


def _openrouter_body(
    *, audio: bytes = PCM_BYTES, content_type: str = "audio/pcm", generation_id: str = "gen-or-1"
):
    return _response(
        audio,
        headers={"Content-Type": content_type, "X-Generation-Id": generation_id},
    )


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


def _parts_script(tmp_path: Path) -> Path:
    path = tmp_path / "parts.yaml"
    path.write_text(PARTS_SCRIPT, encoding="utf-8")
    return path


def _generate_argv(tmp_path, script, run_id, *, provider, model, extra=()):
    return [
        "voiceover-pipeline",
        "generate",
        "--script",
        str(script),
        "--format",
        "speech-parts",
        "--provider",
        provider,
        "--model",
        model,
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        *extra,
        "--json",
    ]


def _text_argv(tmp_path, run_id, *, provider="openrouter-tts", model=GEMINI_38_FLASH, voice="Kore"):
    """A single-part ``--text`` run on one cloud speech route."""
    return [
        "voiceover-pipeline",
        "generate",
        "--text",
        "Одна простая реплика.",
        "--voice",
        voice,
        "--provider",
        provider,
        "--model",
        model,
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        "--json",
    ]


def _run_root(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "out" / run_id


def _response_files(run_root: Path) -> list[Path]:
    return sorted((run_root / "raw").glob("*.response"))


def _receipts(run_root: Path) -> list[dict]:
    return [
        json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
        for path in _response_files(run_root)
    ]


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
            "SELECT status, cost, cost_source FROM attempts WHERE run_uuid = ? ORDER BY rowid",
            (run_uuid,),
        ).fetchall()
    finally:
        connection.close()


def _stored_texts(home: Path, run_uuid: str) -> list[str]:
    connection = sqlite3.connect(home / "history.sqlite3")
    try:
        rows = connection.execute(
            "SELECT prepared_text FROM parts WHERE run_uuid = ? ORDER BY position", (run_uuid,)
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


class RecordingPost:
    """A patched ``requests.post`` that records the exact request payloads."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.respond(len(self.calls))


# ── listing and admission ─────────────────────────────────────────────────────


@pytest.mark.parametrize("model", GEMINI_38_MODELS)
def test_both_ids_are_listed_for_both_providers(model):
    """Both Gemini 3.8 ids appear in both providers' catalogs and are admitted."""
    assert model in config.POLZA_TTS_MODELS
    assert model in config.OPENROUTER_TTS_MODELS
    for provider in ("polza-tts", "openrouter-tts"):
        cli._validate_model_for_provider(provider, model)

    from conftest import cli_json

    code, payload = cli_json("list", "providers", "--json")
    assert code == 0
    models_by_provider = {row["id"]: row.get("models", []) for row in payload["providers"]}
    assert model in models_by_provider["polza-tts"]
    assert model in models_by_provider["openrouter-tts"]


def test_polza_gemini38_voices_are_listed_for_that_provider():
    """The Polza Gemini voice catalog is listed beside the existing families."""
    from conftest import cli_json

    code, payload = cli_json("list", "voices", "--provider", "polza-tts", "--json")
    assert code == 0
    categories = payload["voice_categories"]
    assert categories["gemini"] == config.GEMINI_TTS_VOICES
    assert categories["openai"] == config.OPENAI_TTS_VOICES
    assert categories["elevenlabs"] == config.ELEVENLABS_TTS_VOICES
    assert set(config.GEMINI_TTS_VOICES) <= set(payload["voices"])


def test_defaults_for_other_models_are_preserved():
    """The added ids never change any other provider's or model's default."""
    assert config.PROVIDER_DEFAULT_MODELS["polza-tts"] == STABLE_POLZA_MODEL
    assert config.PROVIDER_DEFAULT_MODELS["openrouter-tts"] == GEMINI_31_MODEL
    assert config.PROVIDER_DEFAULT_MODELS["polza-chat-audio"] == "openai/gpt-audio-mini"


@pytest.mark.parametrize("model", GEMINI_38_MODELS)
def test_polza_gemini38_resolves_the_gemini_voice_catalog_and_default(model):
    """Both Polza Gemini 3.8 ids use the Gemini voices and the Gemini default voice."""
    from voiceover_pipeline.voiceover_script import default_voice, voices_for_provider_model

    assert voices_for_provider_model("polza-tts", model) == config.GEMINI_TTS_VOICES
    assert default_voice("polza-tts", model) == config.DEFAULT_OPENROUTER_TTS_VOICE
    # The other Polza routes keep their own catalog and default.
    assert voices_for_provider_model("polza-tts", STABLE_POLZA_MODEL) == config.OPENAI_TTS_VOICES
    assert default_voice("polza-tts", STABLE_POLZA_MODEL) == config.DEFAULT_POLZA_TTS_VOICE
    elevenlabs = "elevenlabs/text-to-speech-turbo-2-5"
    assert voices_for_provider_model("polza-tts", elevenlabs) == config.ELEVENLABS_TTS_VOICES
    assert default_voice("polza-tts", elevenlabs) == config.DEFAULT_ELEVENLABS_VOICE


@pytest.mark.parametrize("model", GEMINI_38_MODELS)
def test_polza_gemini38_voiceover_script_accepts_a_gemini_voice(tmp_path, model):
    """An ordinary ``format: voiceover`` script with a Gemini voice validates on Polza."""
    from conftest import cli_json

    script = tmp_path / "voiceover.md"
    script.write_text(
        "---\n"
        "format: voiceover\n"
        "provider: polza-tts\n"
        f"model: {model}\n"
        "voice: Kore\n"
        "---\n\n"
        "Одна простая реплика.\n",
        encoding="utf-8",
    )
    code, report = cli_json("validate", "--script", str(script), "--json")
    assert code == 0, report
    assert report["valid"] is True
    assert report["effective_config"]["voice"] == "Kore"


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("polza-tts", "google/gemini-9.9-flash-tts"),
        ("polza-tts", GEMINI_31_MODEL),
        ("openrouter-tts", STABLE_POLZA_MODEL),
        ("openrouter-tts", "google/gemini-3.8-flash-tts-2026"),
    ],
)
def test_unknown_stale_or_mismatched_model_is_still_refused(provider, model):
    """A model the catalog does not admit for this provider is never a route."""
    with pytest.raises(cli.CliError) as excinfo:
        cli._validate_model_for_provider(provider, model)
    assert excinfo.value.code == cli._EXIT_ARGS
    assert "not valid for provider" in str(excinfo.value)
    assert cli._model_rejection_reason(provider, model) is not None


def test_gemini38_construction_refuses_a_provider_foreign_model():
    """Each adapter keeps its own catalog check before any billing."""
    from unittest.mock import patch

    with patch(OPENROUTER_POST_TARGET) as post:
        with pytest.raises(ValueError, match="not in the current OpenRouter speech catalog"):
            OpenRouterTTSProvider(api_key="sk-or", model=STABLE_POLZA_MODEL, voice="alloy")
    post.assert_not_called()


def test_pcm_maps_to_the_canonical_24k_mono_stream():
    """OpenRouter PCM is decoded as the application's single canonical raw stream."""
    assert config.SAMPLE_RATE == 24000
    assert config.CHANNELS == 1
    from voiceover_pipeline.providers.openrouter_tts import decode_audio_speech_body

    body, audio_format = decode_audio_speech_body(
        body=PCM_BYTES, content_type="audio/pcm", requested_format="pcm"
    )
    assert (body, audio_format) == (PCM_BYTES, "pcm16")


# ── validate shares the generate admission ────────────────────────────────────


def _run_validate(capsys, monkeypatch, script: Path, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover-pipeline", "validate", "--script", str(script), *extra, "--json"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("provider", "model"),
    [(p, m) for p in ("polza-tts", "openrouter-tts") for m in GEMINI_38_MODELS],
)
def test_validate_admits_every_gemini38_provider_model_pair(
    tmp_path, capsys, monkeypatch, provider, model
):
    """All four supported combinations are admitted by ``validate`` with no flag."""
    code, report = _run_validate(
        capsys,
        monkeypatch,
        _parts_script(tmp_path),
        "--format",
        "speech-parts",
        "--provider",
        provider,
        "--model",
        model,
    )
    assert code == 0
    assert report["valid"] is True
    assert report["route"] == {"admitted": True, "reason": None}


def test_validate_reports_the_generate_model_rejection(tmp_path, capsys, monkeypatch):
    """An unknown model is refused by ``validate`` before any key or request."""
    code, report = _run_validate(
        capsys,
        monkeypatch,
        _parts_script(tmp_path),
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        "google/gemini-9.9-flash-tts",
    )
    assert code == 2
    assert report["valid"] is False
    assert report["errors"][0]["code"] == "INVALID_MODEL"
    assert report["errors"][0]["message"] == cli._model_rejection_reason(
        "polza-tts", "google/gemini-9.9-flash-tts"
    )


def test_validate_and_generate_refuse_the_same_provider_mismatch(tmp_path, capsys, monkeypatch):
    """The two commands report one admission rule, not two divergent ones."""
    script = _parts_script(tmp_path)
    code, report = _run_validate(
        capsys,
        monkeypatch,
        script,
        "--format",
        "speech-parts",
        "--provider",
        "openrouter-tts",
        "--model",
        "elevenlabs/text-to-speech-turbo-2-5",
    )
    assert code == 2
    assert report["errors"][0]["code"] == "INVALID_MODEL"

    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "mismatch",
            provider="openrouter-tts",
            model="elevenlabs/text-to-speech-turbo-2-5",
        ),
    )
    assert code == 2
    assert "not valid for provider 'openrouter-tts'" in payload["error"]


def test_validate_still_blocks_an_instruction_on_a_route_without_the_field(
    tmp_path, capsys, monkeypatch
):
    """An ordinary OpenAI speech model keeps its per-part-instruction refusal."""
    code, report = _run_validate(
        capsys,
        monkeypatch,
        _parts_script(tmp_path),
        "--format",
        "speech-parts",
        "--provider",
        "polza-tts",
        "--model",
        STABLE_POLZA_MODEL,
    )
    assert code == 0
    assert report["valid"] is True
    assert report["route"] == {"admitted": False, "reason": "BLOCKED_PROVIDER_CONTRACT"}


# ── the ordinary Polza run ────────────────────────────────────────────────────


def test_polza_gemini38_posts_one_scalar_voice_and_separate_instructions(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """Two parts become two scalar POSTs with the direction outside the spoken input."""
    post = RecordingPost(lambda _n: _response(_polza_body()))
    monkeypatch.setattr(POLZA_POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            _parts_script(tmp_path),
            "polza38",
            provider="polza-tts",
            model=GEMINI_38_FLASH_LITE,
        ),
    )

    assert code == 0, payload
    assert len(post.calls) == 2
    assert [call["json"]["voice"] for call in post.calls] == ["Kore", "Puck"]
    assert [call["json"]["model"] for call in post.calls] == [GEMINI_38_FLASH_LITE] * 2
    assert post.calls[0]["json"]["input"] == ("Уникальноеслово расскажет о поиске по истории.")
    assert post.calls[1]["json"]["input"] == "Тогда запись найдётся по содержанию."
    assert "С любопытством." in post.calls[0]["json"]["instructions"]
    for call in post.calls:
        assert call["json"]["input"] not in call["json"]["instructions"]
        assert call["url"].endswith("/audio/speech")
        assert call["allow_redirects"] is False

    run_root = _run_root(tmp_path, "polza38")
    assert (run_root / "chunks" / "chunk_01.mp3").exists()
    receipts = _receipts(run_root)
    assert {receipt["number"]: receipt["voice"] for receipt in receipts} == {1: "Kore", 2: "Puck"}
    assert all(receipt["model"] == GEMINI_38_FLASH_LITE for receipt in receipts)
    assert all(receipt["response_format"] == "mp3" for receipt in receipts)
    attempts = _attempt_rows(native_env, _run_uuid("polza38"))
    assert [row["cost"] for row in attempts] == ["0.3", "0.3"]


def test_polza_gemini38_wav_response_is_converted_with_the_observed_container(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A WAV body returned for an MP3 request reaches the conversion seam as WAV."""
    formats: list[str] = []

    def _record_write(_ffmpeg: str, data: bytes, fmt: str, path: Path) -> None:
        formats.append(fmt)
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", _record_write)
    wav_bytes = b"RIFF" + (0).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 8
    post = RecordingPost(
        lambda _n: _response(_polza_body(audio=wav_bytes, content_type="audio/wav"))
    )
    monkeypatch.setattr(POLZA_POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            _parts_script(tmp_path),
            "polza38-wav",
            provider="polza-tts",
            model=GEMINI_38_FLASH_LITE,
        ),
    )

    assert code == 0, payload
    # The requested container stays MP3 in the private request identity while the
    # observed WAV bytes are what the converter receives.
    assert formats == ["wav", "wav"]
    assert [
        receipt["response_format"] for receipt in _receipts(_run_root(tmp_path, "polza38-wav"))
    ] == [
        "mp3",
        "mp3",
    ]


def test_polza_gemini38_refuses_an_instruction_on_a_stable_model(monkeypatch):
    """The separate instruction field stays limited to the Gemini 3.8 models."""
    post = RecordingPost(lambda _n: _response(_polza_body()))
    monkeypatch.setattr(POLZA_POST_TARGET, post)
    provider = PolzaTTSProvider(api_key="sk-test", model=STABLE_POLZA_MODEL, voice="alloy")

    with pytest.raises(ValueError, match="instructions"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")

    assert post.calls == []


# ── the ordinary OpenRouter run ───────────────────────────────────────────────


def test_openrouter_gemini38_posts_one_scalar_voice_and_separate_instructions(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """The OpenRouter request carries its own voice, text, and separate direction."""
    formats: list[str] = []
    real_write = _write_mp3

    def _record_write(_ffmpeg: str, data: bytes, fmt: str, path: Path) -> None:
        formats.append(fmt)
        real_write(_ffmpeg, data, fmt, path)

    monkeypatch.setattr(cli, "write_audio_as_mp3", _record_write)
    post = RecordingPost(lambda _n: _openrouter_body())
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            _parts_script(tmp_path),
            "openrouter38",
            provider="openrouter-tts",
            model=GEMINI_38_FLASH,
        ),
    )

    assert code == 0, payload
    assert len(post.calls) == 2
    assert [call["json"]["voice"] for call in post.calls] == ["Kore", "Puck"]
    assert [set(call["json"]) for call in post.calls] == [
        {"model", "input", "voice", "response_format", "instructions"}
    ] * 2
    assert [call["json"]["response_format"] for call in post.calls] == ["pcm", "pcm"]
    assert post.calls[0]["json"]["input"] == ("Уникальноеслово расскажет о поиске по истории.")
    assert post.calls[1]["json"]["input"] == "Тогда запись найдётся по содержанию."
    assert "С любопытством." in post.calls[0]["json"]["instructions"]
    for call in post.calls:
        assert call["json"]["input"] not in call["json"]["instructions"]
        assert call["allow_redirects"] is False

    # A raw PCM response is decoded as the canonical 24 kHz mono stream before the
    # conversion seam sees it.
    assert formats == ["pcm16", "pcm16"]
    run_root = _run_root(tmp_path, "openrouter38")
    receipts = _receipts(run_root)
    assert {receipt["number"]: receipt["voice"] for receipt in receipts} == {1: "Kore", 2: "Puck"}
    assert all(receipt["provider"] == "openrouter-tts" for receipt in receipts)
    assert all(receipt["response_format"] == "pcm" for receipt in receipts)
    attempts = _attempt_rows(native_env, _run_uuid("openrouter38"))
    assert [row["status"] for row in attempts] == ["completed", "completed"]
    assert [row["cost_source"] for row in attempts] == ["unknown", "unknown"]


def test_openrouter_gemini31_request_keeps_its_exact_body(monkeypatch):
    """The existing Gemini 3.1 route gains no instruction field and no new key."""
    post = RecordingPost(lambda _n: _openrouter_body())
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)
    provider = OpenRouterTTSProvider(api_key="sk-test", model=GEMINI_31_MODEL, voice="Puck")

    provider.synthesize_chunk("Привет.", "chunk_01", voice="Kore", vibe="")

    assert post.calls[0]["json"] == {
        "model": GEMINI_31_MODEL,
        "input": "Привет.",
        "voice": "Kore",
        "response_format": "pcm",
    }
    with pytest.raises(ValueError, match="Gemini 3.8"):
        provider.synthesize_chunk("Привет.", "chunk_01", vibe="Спокойно.")
    assert len(post.calls) == 1


# ── the OpenRouter pre-parse evidence boundary ────────────────────────────────


def test_openrouter_gemini38_stored_body_replays_without_a_second_post(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A crash after the second stored response replays it locally, one POST each."""
    real_write = native_generation.write_paid_raw_receipt
    calls = {"n": 0}

    def _crash_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", _crash_on_second)
    post = RecordingPost(lambda _n: _openrouter_body())
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)
    script = _parts_script(tmp_path)

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "openrouter38-crash",
            provider="openrouter-tts",
            model=GEMINI_38_FLASH,
        ),
    )
    assert code == 30
    assert len(post.calls) == 2
    run_root = _run_root(tmp_path, "openrouter38-crash")
    assert len(_response_files(run_root)) == 2
    assert not (run_root / "chunks" / "chunk_02.mp3").exists()

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider"))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, lambda *_args, **_kwargs: pytest.fail("no POST"))
    run_uuid = _run_uuid("openrouter38-crash")
    script.unlink()
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0, capsys.readouterr().out
    assert len(post.calls) == 2
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == PCM_BYTES
    assert _attempt_rows(native_env, run_uuid)[1]["status"] == "completed"


def test_openrouter_mp3_response_replays_as_mp3_after_a_post_submit_crash(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """An audio/mpeg success keeps its observed container and replays the same MP3 bytes."""
    real_write = native_generation.write_paid_raw_receipt
    calls = {"n": 0}

    def _crash_before_decoded_link(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", _crash_before_decoded_link)
    formats: list[str] = []

    def _record_write(_ffmpeg: str, data: bytes, fmt: str, path: Path) -> None:
        formats.append(fmt)
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", _record_write)
    post = RecordingPost(lambda _n: _openrouter_body(audio=MP3_BYTES, content_type="audio/mpeg"))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)

    code, _payload = _json_run(monkeypatch, capsys, _text_argv(tmp_path, "openrouter-mp3-crash"))

    assert code == 30
    assert len(post.calls) == 1
    run_root = _run_root(tmp_path, "openrouter-mp3-crash")
    receipts = _receipts(run_root)
    assert len(receipts) == 1
    # The requested container stays the request identity; the observed one is the
    # bounded evidence a local replay decodes with, written before the decode.
    assert receipts[0]["response_format"] == "pcm"
    assert receipts[0]["observed_audio_format"] == "mp3"
    assert not (run_root / "chunks" / "chunk_01.mp3").exists()

    # The local replay must not build a provider or send a second paid POST.
    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider"))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, lambda *_args, **_kwargs: pytest.fail("no POST"))
    run_uuid = _run_uuid("openrouter-mp3-crash")
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0, capsys.readouterr().out
    assert len(post.calls) == 1
    # Exactly the stored bytes reach the converter as mp3, never reinterpreted as pcm16.
    assert formats == ["mp3"]
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == MP3_BYTES
    assert (run_root / "raw" / "chunk_01.mp3").read_bytes() == MP3_BYTES


def test_receipt_without_the_observed_container_still_loads_and_replays(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """Paid evidence written before the observed container existed still replays."""
    real_write = native_generation.write_paid_raw_receipt
    calls = {"n": 0}

    def _crash_before_decoded_link(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash after the response body, before the decoded link")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", _crash_before_decoded_link)
    formats: list[str] = []

    def _record_write(_ffmpeg: str, data: bytes, fmt: str, path: Path) -> None:
        formats.append(fmt)
        path.write_bytes(data)

    monkeypatch.setattr(cli, "write_audio_as_mp3", _record_write)
    post = RecordingPost(lambda _n: _openrouter_body())
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)

    code, _payload = _json_run(
        monkeypatch, capsys, _text_argv(tmp_path, "openrouter-legacy-receipt")
    )
    assert code == 30
    assert len(post.calls) == 1

    # Drop the optional key, leaving exactly the receipt a pre-correction run wrote for
    # the same attempt, body, size, and digest.
    run_root = _run_root(tmp_path, "openrouter-legacy-receipt")
    receipt_path = _response_files(run_root)[0].with_name(
        _response_files(run_root)[0].name + ".receipt.json"
    )
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    stored.pop("observed_audio_format")
    receipt_path.write_text(json.dumps(stored), encoding="utf-8")

    monkeypatch.setattr(native_generation, "write_paid_raw_receipt", real_write)
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider"))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, lambda *_args, **_kwargs: pytest.fail("no POST"))
    run_uuid = _run_uuid("openrouter-legacy-receipt")
    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0, capsys.readouterr().out
    assert len(post.calls) == 1
    # Without the preserved container the requested one is the only remaining evidence.
    assert formats == ["pcm16"]
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == PCM_BYTES


def test_openrouter_gemini38_non_audio_body_is_kept_private_then_blocks_resume(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A wrapped JSON error is stored before it fails and never authorizes a retry."""
    body = b'{"error":"No successful provider responses"}'
    post = RecordingPost(lambda _n: _response(body, headers={"Content-Type": "application/json"}))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)
    script = _parts_script(tmp_path)

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "openrouter38-wrap",
            provider="openrouter-tts",
            model=GEMINI_38_FLASH,
        ),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "NATIVE_SYNTHESIS_FAILED"
    assert len(post.calls) == 1
    run_root = _run_root(tmp_path, "openrouter38-wrap")
    assert _response_files(run_root)[0].read_bytes() == body

    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: pytest.fail("no provider"))
    monkeypatch.setattr(OPENROUTER_POST_TARGET, lambda *_args, **_kwargs: pytest.fail("no POST"))
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "openrouter38-wrap",
            provider="openrouter-tts",
            model=GEMINI_38_FLASH,
            extra=["--resume"],
        ),
    )
    assert code == 30
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert len(post.calls) == 1


def test_openrouter_gemini38_redirect_stops_with_one_post_and_one_receipt(
    tmp_path, monkeypatch, capsys, native_env, socket_denied
):
    """A 3xx stays one observed paid response instead of being followed."""
    body = b"<html>moved</html>"
    post = RecordingPost(
        lambda _n: _response(body, status_code=302, headers={"Content-Type": "text/html"})
    )
    monkeypatch.setattr(OPENROUTER_POST_TARGET, post)

    code, _payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            _parts_script(tmp_path),
            "openrouter38-redirect",
            provider="openrouter-tts",
            model=GEMINI_38_FLASH,
        ),
    )

    assert code == 30
    assert len(post.calls) == 1
    assert post.calls[0]["allow_redirects"] is False
    # The observed refusal body is retained privately before the status is read, so
    # the one paid response is never lost.
    files = _response_files(_run_root(tmp_path, "openrouter38-redirect"))
    assert len(files) == 1
    assert files[0].read_bytes() == body


# ── --text uses the same ordinary admission ───────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "target"),
    [("polza-tts", POLZA_POST_TARGET), ("openrouter-tts", OPENROUTER_POST_TARGET)],
)
def test_text_run_carries_its_vibe_in_the_separate_field(
    tmp_path, monkeypatch, capsys, native_env, socket_denied, provider, target
):
    """``--text`` with ``--vibe`` posts one scalar voice and a separate direction."""
    is_polza = provider == "polza-tts"
    post = RecordingPost(lambda _n: _response(_polza_body()) if is_polza else _openrouter_body())
    monkeypatch.setattr(target, post)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        [
            "voiceover-pipeline",
            "generate",
            "--text",
            "Одна простая реплика.",
            "--voice",
            "Kore",
            "--vibe",
            "Спокойно.",
            "--provider",
            provider,
            "--model",
            GEMINI_38_FLASH,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "text38",
            "--json",
        ],
    )

    assert code == 0, payload
    assert len(post.calls) == 1
    assert post.calls[0]["json"]["input"] == "Одна простая реплика."
    assert post.calls[0]["json"]["voice"] == "Kore"
    assert post.calls[0]["json"]["instructions"] == "Спокойно."
    assert _stored_texts(native_env, _run_uuid("text38")) == ["Одна простая реплика."]
