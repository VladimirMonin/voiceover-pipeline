"""Offline contract tests for the ``speech-parts`` format, ``--text``, and wav.

Every test is deterministic and offline: a temporary ``VOICEOVER_HOME``, a temp
run directory, fake in-memory media providers, and patched FFmpeg seams. No real
provider, network call, API key, ``.env``, model, or paid request is used. The one
real-FFmpeg test is skipped when ``ffmpeg`` is not installed.
"""

import json
import shutil
import sqlite3
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline import media
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.speech_parts import (
    SPEECH_PARTS_REQUEST_CHAR_LIMIT,
    SpeechPartsError,
    build_speech_parts_document,
    compose_effective_vibe,
    speech_part_request_chars,
)

GEMINI_MODEL = "google/gemini-3.1-flash-tts-preview"
CANDIDATE_MODEL = "google/gemini-3.8-flash-tts"

SPEECH_PARTS_SCRIPT = """\
version: 1
format: speech-parts
vibe: >-
  Образовательный подкаст о программировании.
  Естественная разговорная речь.
parts:
  - voice: Kore
    vibe: С любопытством.
    text: |-
      Что изменится, если сохранить историю запусков?
  - voice: Puck
    vibe: Спокойно.
    text: |-
      Тогда результат можно найти по содержанию.
  - voice: Kore
    text: |-
      Значит, старые записи перестанут теряться среди папок.
"""


# ── parser and budget (pure) ──────────────────────────────────────────────────


def test_build_document_keeps_text_and_both_vibes_separate():
    document = build_speech_parts_document(SPEECH_PARTS_SCRIPT)
    assert document.global_vibe is not None
    assert [part.voice for part in document.parts] == ["Kore", "Puck", "Kore"]
    first = document.parts[0]
    assert first.text == "Что изменится, если сохранить историю запусков?"
    # The effective instruction contains the document and part vibe, and the spoken
    # text is never part of it.
    assert first.effective_vibe == f"{document.global_vibe}\n\nС любопытством."
    assert first.text not in first.effective_vibe
    # The last part carries only the document vibe; no part-level vibe was authored.
    assert document.parts[2].vibe is None
    assert document.parts[2].effective_vibe == document.global_vibe


def test_build_document_strips_a_leading_bom_before_parsing():
    document = build_speech_parts_document("\ufeff" + SPEECH_PARTS_SCRIPT)
    assert [part.voice for part in document.parts] == ["Kore", "Puck", "Kore"]


@pytest.mark.parametrize(
    "document, expected",
    [
        (
            "version: 1\nformat: speech-parts\nparts:\n  - voice: A\n    voice: B\n    text: x\n",
            "duplicate",
        ),
        ("version: 2\nformat: speech-parts\nparts:\n  - voice: A\n    text: x\n", "version"),
        ("version: 1\nformat: other\nparts:\n  - voice: A\n    text: x\n", "format"),
        ("version: 1\nformat: speech-parts\nparts: []\n", "parts"),
        (
            "version: 1\nformat: speech-parts\nparts:\n  - voice: A\n    speech: x\n    text: y\n",
            "unsupported",
        ),
        (
            "version: 1\nformat: speech-parts\nparts:\n  - voice: A\n    text: '   '\n",
            "non-empty text",
        ),
        ("version: 1\nformat: speech-parts\nparts:\n  - text: x\n", "non-empty voice"),
        (
            "version: 1\nformat: speech-parts\nunknown: 1\nparts:\n  - voice: A\n    text: x\n",
            "top-level",
        ),
    ],
)
def test_build_document_rejects_structural_violations(document, expected):
    with pytest.raises(SpeechPartsError) as excinfo:
        build_speech_parts_document(document)
    assert expected in str(excinfo.value)


def test_build_document_rejects_tab_after_spaces_in_indentation():
    document = "version: 1\nformat: speech-parts\nparts:\n  \t- voice: A\n    text: x\n"
    with pytest.raises(SpeechPartsError, match="tab indentation"):
        build_speech_parts_document(document)


def test_compose_effective_vibe_skips_missing_values():
    assert compose_effective_vibe(None, None) == ""
    assert compose_effective_vibe("A", None) == "A"
    assert compose_effective_vibe(None, "B") == "B"
    assert compose_effective_vibe("A", "B") == "A\n\nB"


def test_budget_boundary_counts_emoji_vibe_and_wrapper_exactly():
    wrapper = "read this exactly:  \n"
    emoji = "🙂"
    # Every character -- a multi-byte emoji, a newline, and the wrapper -- counts once.
    assert len(emoji) == 1
    budget = SPEECH_PARTS_REQUEST_CHAR_LIMIT
    vibe = "x"
    text_at_limit = "a" * (budget - len(vibe) - len(wrapper))
    assert speech_part_request_chars(text_at_limit, vibe, wrapper) == budget
    assert speech_part_request_chars(text_at_limit[:-1], vibe, wrapper) == budget - 1
    assert speech_part_request_chars(text_at_limit + emoji, vibe, wrapper) == budget + 1
    assert SPEECH_PARTS_REQUEST_CHAR_LIMIT > 0


# ── CLI validate ──────────────────────────────────────────────────────────────


def _run_validate(capsys, monkeypatch, script: Path, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover-pipeline", "validate", "--script", str(script), *extra, "--json"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


def test_validate_reports_the_unverified_candidate_route_as_blocked(tmp_path, capsys, monkeypatch):
    script = tmp_path / "podcast.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
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
    assert report["valid"] is True
    assert report["route"] == {"admitted": False, "reason": "BLOCKED_PROVIDER_CONTRACT"}
    assert report["parts"][0]["request_chars"] > report["parts"][0]["chars"]


def test_validate_reports_a_syntax_error_with_a_usage_exit(tmp_path, capsys, monkeypatch):
    script = tmp_path / "broken.yaml"
    script.write_text("version: 1\nformat: speech-parts\nparts:\n  - voice: A\n", encoding="utf-8")
    code, report = _run_validate(capsys, monkeypatch, script, "--format", "speech-parts")
    assert code == 2
    assert report["valid"] is False
    assert report["errors"][0]["code"] == "SPEECH_PARTS_INVALID"


# ── CLI argument admission ────────────────────────────────────────────────────


def _run_generate(capsys, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["voiceover-pipeline", "generate", *argv, "--json"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code, json.loads(capsys.readouterr().out)


@pytest.fixture
def media_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    return tmp_path


def test_text_and_explicit_script_are_mutually_exclusive(media_env, capsys, monkeypatch):
    script = media_env / "s.md"
    script.write_text("hi", encoding="utf-8")
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        ["--text", "hi", "--script", str(script), "--voice", "ash"],
    )
    assert code == 2
    assert "mutually exclusive" in payload["error"]


def test_speech_parts_script_rejects_a_cli_voice_override(media_env, capsys, monkeypatch):
    script = media_env / "p.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        ["--script", str(script), "--format", "speech-parts", "--voice", "Kore"],
    )
    assert code == 2
    assert "hidden override" in payload["error"]


def test_vibe_is_blocked_before_any_key_or_provider(media_env, capsys, monkeypatch):
    def explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("no provider may be built for a blocked vibe route")

    monkeypatch.setattr(cli, "build_provider", explode)
    monkeypatch.setattr(cli, "read_api_key", explode)
    code, payload = _run_generate(
        capsys, monkeypatch, ["--text", "Привет", "--voice", "ash", "--vibe", "Спокойно"]
    )
    assert code == 30
    assert payload["details"]["error_code"] == "BLOCKED_PROVIDER_CONTRACT"


def test_markdown_script_vibe_refuses_before_key_or_provider(media_env, capsys, monkeypatch):
    script = media_env / "narration.md"
    script.write_text("A real legacy narration.", encoding="utf-8")

    def explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("a legacy script must not silently drop --vibe")

    monkeypatch.setattr(cli, "build_provider", explode)
    monkeypatch.setattr(cli, "read_api_key", explode)
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--vibe",
            "Спокойно",
            "--output-dir",
            str(media_env / "out"),
        ],
    )
    assert code == 2
    assert payload["details"]["error_code"] == "VIBE_UNSUPPORTED_FORMAT"
    assert not (media_env / "out").exists()


def test_candidate_model_is_blocked_before_any_key_or_provider(media_env, capsys, monkeypatch):
    script = media_env / "p.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")

    def explode(*_args, **_kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("no provider may be built for a candidate route")

    monkeypatch.setattr(cli, "build_provider", explode)
    monkeypatch.setattr(cli, "read_api_key", explode)
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "polza-tts",
            "--model",
            CANDIDATE_MODEL,
        ],
    )
    assert code == 30
    assert payload["details"]["error_code"] == "BLOCKED_PROVIDER_CONTRACT"


def test_a_late_over_limit_part_posts_nothing(media_env, capsys, monkeypatch):
    built: list[str] = []

    def fake_build(*_args, **_kwargs):
        built.append("built")
        raise AssertionError("no provider may be built when a late part is over limit")

    monkeypatch.setattr(cli, "build_provider", fake_build)
    script = media_env / "long.yaml"
    # The last part alone exceeds the budget; no earlier part may be submitted.
    long_text = "a" * (SPEECH_PARTS_REQUEST_CHAR_LIMIT + 1)
    script.write_text(
        "version: 1\nformat: speech-parts\nparts:\n"
        "  - voice: ash\n    text: short\n"
        f"  - voice: ash\n    text: {long_text}\n",
        encoding="utf-8",
    )
    code, payload = _run_generate(
        capsys, monkeypatch, ["--script", str(script), "--format", "speech-parts"]
    )
    assert code == 2
    assert payload["details"]["error_code"] == "SPEECH_PART_TOO_LONG"
    assert payload["details"]["part"] == 2
    assert built == []


# ── native DB-first execution (fake provider) ─────────────────────────────────


class RecordingSpeechProvider:
    """Offline provider that accepts a per-part voice and vibe and records both."""

    def __init__(self, audio_format: str = "mp3") -> None:
        self.audio_format = audio_format
        self.calls: list[tuple[str, str, str | None, str | None]] = []

    def synthesize_chunk(
        self, text: str, chunk_id: str, voice: str | None = None, vibe: str | None = None
    ) -> SynthesisResult:
        self.calls.append((chunk_id, text, voice, vibe))
        return SynthesisResult(
            audio_bytes=f"{chunk_id}-audio".encode(),
            audio_format=self.audio_format,
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="requests",
            raw_metadata={"voice": voice, "provider": "openrouter-tts"},
        )


def _write_audio(_ffmpeg: str, data: bytes, _fmt: str, path: Path) -> None:
    path.write_bytes(data)


def _concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path in paths))


def _concat_dialogue(_ffmpeg: str, turns: list[tuple[Path, int]], output: Path) -> None:
    output.write_bytes(b"".join(path.read_bytes() for path, _pause in turns))


@pytest.fixture
def native_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "write_audio_as_mp3", _write_audio)
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _path: 1000)
    monkeypatch.setattr(cli, "concat_audio_files", _concat)
    monkeypatch.setattr(cli, "concat_dialogue_turns", _concat_dialogue)
    # The confirmed fake provider below is the only route that may carry a vibe, so
    # this test stubs only the route admission the production build refuses.
    monkeypatch.setattr(cli, "require_confirmed_speech_parts_route", lambda **_kwargs: None)
    return home


def _install_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: provider)


def _rows(home: Path, sql: str, params=()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(home / "history.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _run_uuid(prefix: str) -> str:
    from voiceover_pipeline.commands.history import list_history

    matches = [run for run in list_history()["runs"] if run["user_label"] == prefix]
    assert len(matches) == 1, matches
    return matches[0]["run_uuid"]


def test_speech_parts_three_parts_two_voices_distinct_vibes(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = RecordingSpeechProvider()
    _install_provider(monkeypatch, provider)
    script = tmp_path / "podcast.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")

    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sp-run",
        ],
    )
    assert code == 0, payload
    assert payload["status"] == "success"

    # Exactly three ordered single-voice requests, each with its own cast voice and
    # its own effective vibe; the spoken text is what is synthesized.
    assert [call[0] for call in provider.calls] == ["chunk_01", "chunk_02", "chunk_03"]
    assert [call[2] for call in provider.calls] == ["Kore", "Puck", "Kore"]
    assert (
        provider.calls[0][3]
        == f"{build_speech_parts_document(SPEECH_PARTS_SCRIPT).global_vibe}\n\nС любопытством."
    )
    assert (
        provider.calls[1][3]
        == f"{build_speech_parts_document(SPEECH_PARTS_SCRIPT).global_vibe}\n\nСпокойно."
    )
    assert provider.calls[2][3] == build_speech_parts_document(SPEECH_PARTS_SCRIPT).global_vibe
    for _chunk_id, text, _voice, vibe in provider.calls:
        assert vibe is None or text not in vibe
        assert text in {
            part.text for part in build_speech_parts_document(SPEECH_PARTS_SCRIPT).parts
        }

    run_uuid = _run_uuid("sp-run")
    parts = _rows(
        native_env,
        "SELECT position, voice, prepared_text, vibe_shared, vibe_specific, vibe_effective "
        "FROM parts WHERE run_uuid = ? ORDER BY position",
        (run_uuid,),
    )
    expected = build_speech_parts_document(SPEECH_PARTS_SCRIPT)
    assert [row["position"] for row in parts] == [1, 2, 3]
    assert [row["voice"] for row in parts] == ["Kore", "Puck", "Kore"]
    assert [row["vibe_specific"] for row in parts] == ["С любопытством.", "Спокойно.", None]
    assert [row["vibe_effective"] for row in parts] == [p.effective_vibe for p in expected.parts]
    assert [row["vibe_shared"] for row in parts] == [expected.global_vibe] * 3

    chunks = json.loads(Path(payload["files"]["chunks_json"]).read_text(encoding="utf-8"))
    assert chunks["script_format"] == "speech-parts"
    assert [chunk["id"] for chunk in chunks["chunks"]] == ["chunk_01", "chunk_02", "chunk_03"]


def test_speech_parts_resume_is_script_free_and_reports_order(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = RecordingSpeechProvider()
    _install_provider(monkeypatch, provider)
    script = tmp_path / "podcast.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sp-resume",
        ],
    )
    assert code == 0, payload
    run_uuid = _run_uuid("sp-resume")

    # The original script is deleted, so a resume can only rebuild from the snapshot.
    script.unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover-pipeline", "history", "resume", run_uuid, "--json"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_payload["status"] == "success"
    # No second provider submit happened: the run was already complete.
    assert [call[0] for call in provider.calls] == ["chunk_01", "chunk_02", "chunk_03"]


def test_text_run_has_no_script_path_and_is_resumable(tmp_path, monkeypatch, capsys, native_env):
    provider = RecordingSpeechProvider()
    _install_provider(monkeypatch, provider)
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--text",
            "Одна простая реплика.",
            "--voice",
            "Kore",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "text-run",
        ],
    )
    assert code == 0, payload
    assert [call[0] for call in provider.calls] == ["chunk_01"]
    assert provider.calls[0][2] == "Kore"
    assert provider.calls[0][3] in (None, "")
    run_uuid = _run_uuid("text-run")

    monkeypatch.setattr(
        sys, "argv", ["voiceover-pipeline", "history", "resume", run_uuid, "--json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_text_run_uses_the_default_provider_and_legacy_signature(
    tmp_path, monkeypatch, capsys, native_env
):
    calls: list[str] = []

    class LegacyProvider:
        def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
            calls.append(chunk_id)
            return SynthesisResult(
                audio_bytes=b"audio",
                audio_format="mp3",
                transcript=text,
                generation_id=f"gen-{chunk_id}",
                client_path="requests",
                raw_metadata={"voice": "ash", "provider": "polza-chat-audio"},
            )

    _install_provider(monkeypatch, LegacyProvider())
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--text",
            "Одна реплика без vibe.",
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "text-default",
        ],
    )
    assert code == 0, payload
    assert payload["provider"] == "polza-chat-audio"
    assert calls == ["chunk_01"]


def test_resume_refuses_a_changed_audio_format(tmp_path, monkeypatch, capsys, native_env):
    provider = RecordingSpeechProvider()
    _install_provider(monkeypatch, provider)
    script = tmp_path / "podcast.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sp-fmt",
        ],
    )
    assert code == 0, payload
    # The run recorded the default mp3; a resume asking for wav on the same run root
    # must fail closed rather than silently change the container.
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sp-fmt",
            "--audio-format",
            "wav",
            "--resume",
            "--json",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
    output = capsys.readouterr().out
    assert "recorded output-processing settings differ" in output


# ── audio format ──────────────────────────────────────────────────────────────


def test_wav_concat_writes_a_real_riff_container(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:  # pragma: no cover - environment dependent
        pytest.skip("ffmpeg is not installed")

    def _write_wav(path: Path, sample: int) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(struct.pack("<" + "h" * 2400, *([sample] * 2400)))

    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, 100)
    _write_wav(second, 200)
    output = tmp_path / "merged.wav"
    media.concat_audio_files(ffmpeg, [first, second], output)
    header = output.read_bytes()[:12]
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-of",
            "csv=p=0",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert b"wav" in probe.stdout


def test_audio_format_wav_uses_wav_extension_and_is_recorded(
    tmp_path, monkeypatch, capsys, native_env
):
    provider = RecordingSpeechProvider(audio_format="wav")
    _install_provider(monkeypatch, provider)
    seen: list[Path] = []

    def _record_concat(_ffmpeg: str, paths: list[Path], output: Path) -> None:
        seen.append(output)
        output.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE")

    monkeypatch.setattr(cli, "concat_audio_files", _record_concat)
    script = tmp_path / "podcast.yaml"
    script.write_text(SPEECH_PARTS_SCRIPT, encoding="utf-8")
    code, payload = _run_generate(
        capsys,
        monkeypatch,
        [
            "--script",
            str(script),
            "--format",
            "speech-parts",
            "--provider",
            "openrouter-tts",
            "--model",
            GEMINI_MODEL,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sp-wav",
            "--audio-format",
            "wav",
        ],
    )
    assert code == 0, payload
    # The merged output path the concat seam receives ends in .wav and the file is
    # published under that name.
    assert seen and seen[0].suffix == ".wav"
    final = Path(payload["files"]["full_mp3"])
    assert final.suffix == ".wav"
    assert final.read_bytes()[:4] == b"RIFF"
    run_uuid = _run_uuid("sp-wav")
    config = _rows(
        native_env,
        "SELECT config_snapshot FROM runs WHERE run_uuid = ?",
        (run_uuid,),
    )[0]["config_snapshot"]
    assert json.loads(config)["output"]["audio_format"] == "wav"
