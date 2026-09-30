"""End-to-end contract tests for the native DB-first ``format: voiceover`` route.

Every test is offline and synthetic: a temporary ``VOICEOVER_HOME``, a temp run
directory, a fake in-memory media provider, and patched FFmpeg/concat seams. No
real provider, network call, API key, ``.env``, model, or paid request is used.

The tests assert the committed history of a voiceover run (the one resolved
provider/model/voice identity, the private prepared source text, and the
DB-derived compatibility export) and that the constraints the voiceover validator
already enforced -- an OpenRouter style prompt, an invalid provider voice, an
OmniVoice mode that rejects a voice -- are still refused before any provider.
"""

import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import DEFAULT_OMNIVOICE_VOICE
from voiceover_pipeline.models import SynthesisResult

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
VOICEOVER_VOICE = "Rachel"


class FakeMediaProvider:
    """Offline stand-in for a voiceover run's one resolved provider."""

    def __init__(self) -> None:
        self.on_media_task_accepted = None
        self.on_media_completed = None
        self.submits: list[str] = []

    def _audio(self, text: str, chunk_id: str) -> SynthesisResult:
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.submits.append(chunk_id)
        self.on_media_task_accepted(f"task-{chunk_id}")
        self.on_media_completed(f"task-{chunk_id}", {"cost_rub": Decimal("0.3")}, f"gen-{chunk_id}")
        return self._audio(text, chunk_id)


class FakeSyncProvider:
    """Offline stand-in for a synchronous (non-media) provider submit."""

    def __init__(self, audio_format: str = "mp3") -> None:
        self.calls: list[str] = []
        self.audio_format = audio_format

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.calls.append(chunk_id)
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format=self.audio_format,
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
            raw_metadata={"voice": "ash", "provider": "polza-chat-audio"},
        )


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


def _script(
    tmp_path: Path,
    parts: list[str],
    *,
    meta_lines: tuple[str, ...] = (),
    name: str = "voiceover.md",
) -> Path:
    frontmatter = [
        "format: voiceover",
        "provider: polza-tts",
        f"model: {POLZA_MEDIA_MODEL}",
        f"voice: {VOICEOVER_VOICE}",
        *meta_lines,
    ]
    path = tmp_path / name
    path.write_text(
        "\n".join(["---", *frontmatter, "---", "\n******\n".join(parts)]),
        encoding="utf-8",
    )
    return path


def _generate_argv(
    tmp_path: Path,
    script: Path,
    run_id: str,
    *,
    extra=(),
    provider: str | None = None,
    model: str | None = None,
    voice: str | None = None,
):
    argv = [
        "voiceover-pipeline",
        "generate",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
    ]
    if provider is not None:
        argv += ["--provider", provider]
    if model is not None:
        argv += ["--model", model]
    if voice is not None:
        argv += ["--voice", voice]
    return [*argv, *extra, "--json"]


def _json_run(monkeypatch, capsys, argv) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    code = excinfo.value.code
    assert not isinstance(code, str) and code is not None
    return code, json.loads(capsys.readouterr().out)


def _install_provider(monkeypatch, provider) -> list[str]:
    builds: list[str] = []

    def fake_build(*_args, **_kwargs):
        builds.append("built")
        return provider

    monkeypatch.setattr(cli, "build_provider", fake_build)
    return builds


def _explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
    raise AssertionError("this path must not build a provider")


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


# ── fresh voiceover run ───────────────────────────────────────────────────────


def test_native_voiceover_format_generates_and_commits_its_resolved_identity(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент.", "Второй фрагмент."])

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-fresh"))

    assert code == 0, payload
    assert payload["status"] == "success"
    assert payload["provider"] == "polza-tts"
    assert payload["model"] == POLZA_MEDIA_MODEL
    # The voiceover frontmatter resolves exactly one request per part.
    assert provider.submits == ["chunk_01", "chunk_02"]

    run_root = tmp_path / "out" / "vo-fresh"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"
    assert (run_root / "chunks" / "chunk_02.mp3").read_bytes() == b"chunk_02-audio"
    assert (run_root / ".voiceover-native-history.json").exists()
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["script_format"] == "voiceover"
    assert state["native_history"]["history_run_uuid"]

    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "voiceover"
    assert chunks["provider"] == "polza-tts"
    assert chunks["model"] == POLZA_MEDIA_MODEL
    assert chunks["voice"] == VOICEOVER_VOICE
    assert [chunk["id"] for chunk in chunks["chunks"]] == ["chunk_01", "chunk_02"]

    run_uuid = _run_uuid("vo-fresh")
    parts = _rows(
        native_env,
        "SELECT position, voice, prepared_text FROM parts WHERE run_uuid = ? ORDER BY position",
        (run_uuid,),
    )
    assert [tuple(row) for row in parts] == [
        (1, VOICEOVER_VOICE, "Первый фрагмент."),
        (2, VOICEOVER_VOICE, "Второй фрагмент."),
    ]
    attempts = _rows(
        native_env,
        "SELECT status, cost FROM attempts WHERE run_uuid = ? ORDER BY rowid",
        (run_uuid,),
    )
    assert [(row["status"], row["cost"]) for row in attempts] == [
        ("completed", "0.3"),
        ("completed", "0.3"),
    ]
    # The private run-level source is the exact prepared chunk text.
    script_source = _rows(
        native_env,
        "SELECT content FROM text_sources WHERE run_uuid = ? AND kind = 'tts_script' "
        "AND part_uuid IS NULL",
        (run_uuid,),
    )
    assert [row["content"] for row in script_source] == ["Первый фрагмент.\nВторой фрагмент."]


def test_native_voiceover_format_detects_the_explicit_format_flag_too(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Один фрагмент."])

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-flag", extra=("--format", "voiceover")),
    )

    assert code == 0, payload
    assert provider.submits == ["chunk_01"]
    run_root = tmp_path / "out" / "vo-flag"
    assert (run_root / ".voiceover-native-history.json").exists()
    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "voiceover"


@pytest.mark.parametrize(
    ("provider", "model", "voice", "fallback_voice"),
    [
        ("polza-tts", "openai/gpt-4o-mini-tts", "alloy", None),
        ("polza-chat-audio", "openai/gpt-audio-mini", "ash", "onyx"),
    ],
)
def test_native_voiceover_format_admits_every_ordinary_provider(
    tmp_path, monkeypatch, capsys, native_env, provider, model, voice, fallback_voice
):
    fake = FakeSyncProvider(audio_format="mp3" if provider == "polza-tts" else "pcm16")
    _install_provider(monkeypatch, fake)
    script = tmp_path / "voiceover-ordinary.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: voiceover",
                f"provider: {provider}",
                f"model: {model}",
                f"voice: {voice}",
                "---",
                "Первый фрагмент.",
            ]
        ),
        encoding="utf-8",
    )

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, f"vo-{provider}")
    )

    assert code == 0, payload

    assert fake.calls == ["chunk_01"]
    run_uuid = _run_uuid(f"vo-{provider}")
    config = _rows(native_env, "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (run_uuid,))[
        0
    ]["config_snapshot"]
    snapshot = json.loads(config)
    assert snapshot["script_format"] == "voiceover"
    assert snapshot["provider"] == provider
    assert snapshot["model"] == model
    assert snapshot["voice"] == voice
    # Only polza-chat-audio records its resolved compatibility fallback voice.
    assert snapshot.get("fallback_voice") == fallback_voice


# ── completed-run resume and export repair ────────────────────────────────────


def test_native_voiceover_format_resume_never_resubmits_a_completed_run(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент.", "Второй фрагмент."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-resume"))
    assert code == 0
    assert provider.submits == ["chunk_01", "chunk_02"]

    monkeypatch.setattr(cli, "build_provider", _explode)
    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(tmp_path, script, "vo-resume", extra=("--resume",)),
    )

    assert code == 0, payload
    # No second paid POST: the completed run only repairs its exports.
    assert provider.submits == ["chunk_01", "chunk_02"]


def test_native_voiceover_format_history_sync_repairs_exports_without_the_script(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент."])
    code, _payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-sync"))
    assert code == 0
    run_uuid = _run_uuid("vo-sync")
    script.unlink()
    monkeypatch.setattr(cli, "build_provider", _explode)

    code, payload = _json_run(monkeypatch, capsys, _history_argv(run_uuid, "sync"))

    assert code == 0, payload
    assert payload["mode"] == "sync"
    assert provider.submits == ["chunk_01"]
    run_root = tmp_path / "out" / "vo-sync"
    assert (run_root / "chunks" / "chunk_01.mp3").read_bytes() == b"chunk_01-audio"


# ── blocked invalid format/voice combinations ─────────────────────────────────


def test_native_voiceover_format_rejects_openrouter_style_prompt_before_provider(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    builds = _install_provider(monkeypatch, provider)
    script = _script(
        tmp_path,
        ["Первый фрагмент."],
        meta_lines=("style_prompt: Speak as a calm narrator.",),
        name="voiceover-openrouter.md",
    )
    script.write_text(
        script.read_text(encoding="utf-8")
        .replace("provider: polza-tts", "provider: openrouter-tts")
        .replace(f"model: {POLZA_MEDIA_MODEL}", "model: google/gemini-3.1-flash-tts-preview")
        .replace(f"voice: {VOICEOVER_VOICE}", "voice: Puck"),
        encoding="utf-8",
    )

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-style"))

    assert code == 2, payload
    assert "style prompt" in payload["error"]
    assert builds == []
    assert not (tmp_path / "out" / "vo-style").exists()


def test_native_voiceover_format_rejects_an_invalid_provider_voice(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    builds = _install_provider(monkeypatch, provider)
    script = _script(tmp_path, ["Первый фрагмент."], name="voiceover-bad-voice.md")
    script.write_text(
        script.read_text(encoding="utf-8").replace(
            "voice: Rachel", "voice: not-a-elevenlabs-voice"
        ),
        encoding="utf-8",
    )

    code, payload = _json_run(monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-bad-voice"))

    assert code == 2, payload
    assert {item["code"] for item in payload["errors"]} == {"INVALID_VOICE"}
    assert builds == []


def test_native_voiceover_format_rejects_an_omnivoice_mode_that_refuses_a_voice(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    builds = _install_provider(monkeypatch, provider)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixture")
    script = _script(tmp_path, ["Первый фрагмент."], name="voiceover-omni.md")
    script.write_text(
        script.read_text(encoding="utf-8")
        .replace("provider: polza-tts", "provider: omnivoice-local")
        .replace(f"model: {POLZA_MEDIA_MODEL}", "model: audio-cpp/omnivoice-q8_0")
        .replace(f"voice: {VOICEOVER_VOICE}", f"voice: {DEFAULT_OMNIVOICE_VOICE}"),
        encoding="utf-8",
    )

    code, payload = _json_run(
        monkeypatch,
        capsys,
        _generate_argv(
            tmp_path,
            script,
            "vo-omni-mode",
            extra=(
                "--mode",
                "clone",
                "--reference-audio",
                str(reference),
                "--reference-text",
                "reference",
            ),
        ),
    )

    # The voiceover format always supplies a voice, so a non-preset OmniVoice mode
    # that rejects `--voice` stays a usage error before any provider.
    assert code == 2, payload
    assert "voice controls" in payload["error"]
    assert builds == []


def test_native_voiceover_format_rejects_an_invalid_qwen_preset_voice(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = FakeMediaProvider()
    builds = _install_provider(monkeypatch, provider)
    script = tmp_path / "voiceover-qwen.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: voiceover",
                "provider: qwen-local",
                "voice: not-a-qwen-preset-speaker",
                "---",
                "Первый фрагмент.",
            ]
        ),
        encoding="utf-8",
    )

    code, payload = _json_run(
        monkeypatch, capsys, _generate_argv(tmp_path, script, "vo-qwen-voice")
    )

    # The voiceover validator still checks a local Qwen voice against the preset
    # speaker catalog before any route or provider exists.
    assert code == 2, payload
    assert {item["code"] for item in payload["errors"]} == {"INVALID_VOICE"}
    assert builds == []


def test_native_voiceover_format_local_route_gate():
    import argparse

    def _args(provider: str, **overrides):
        base = dict(
            provider=provider,
            model=POLZA_MEDIA_MODEL,
            no_trim=False,
            with_timings=False,
            timing_provider="faster-whisper",
            tts_quality_provider=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    # The ordinary non-dialogue providers are admitted for both script formats.
    for provider, model in (
        ("polza-tts", POLZA_MEDIA_MODEL),
        ("polza-tts", "openai/gpt-4o-mini-tts"),
        ("openrouter-tts", "google/gemini-3.1-flash-tts-preview"),
        ("polza-chat-audio", "openai/gpt-audio-mini"),
    ):
        args = _args(provider, model=model)
        assert cli._native_route_eligible(args, "markdown") is True
        assert cli._native_route_eligible(args, "voiceover") is True
    # Every functioning local Qwen voiceover mode is admitted: the voiceover
    # validator resolves one voice and `_resolve_qwen_mode_identity` selects the
    # mode's own effective voice, exactly as the legacy executor does.
    assert (
        cli._native_route_eligible(
            _args(
                "qwen-local",
                model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                mode="preset",
            ),
            "voiceover",
        )
        is True
    )
    assert (
        cli._native_route_eligible(
            _args("qwen-local", model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", mode="clone", sample="x"),
            "voiceover",
        )
        is True
    )
    assert (
        cli._native_route_eligible(
            _args("qwen-local", model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", mode="design"),
            "voiceover",
        )
        is True
    )
    # Only the OmniVoice preset bank route is admitted for voiceover; a non-preset
    # mode always rejects the voice such a script supplies.
    assert (
        cli._native_route_eligible(
            _args(
                "omnivoice-local",
                model="audio-cpp/omnivoice-q8_0",
                mode="preset",
                voice_bank_catalog=object(),
                voice_bank_profile=object(),
            ),
            "voiceover",
        )
        is True
    )
    assert (
        cli._native_route_eligible(
            _args("omnivoice-local", model="audio-cpp/omnivoice-q8_0", mode="auto"), "voiceover"
        )
        is False
    )
    # A non-voiceover, non-markdown format is never admitted.
    assert cli._native_route_eligible(_args("polza-tts"), "something-else") is False
