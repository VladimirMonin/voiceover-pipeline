from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path

import pytest


def _write_dialogue_script(path: Path, *, script_format: str) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f"format: {script_format}",
                "language: ru",
                "model: google/gemini-3.1-flash-tts-preview",
                "speakers:",
                "  Host:",
                "    voice: Kore",
                "  Guest:",
                "    voice: Puck",
                "---",
                "Host: Привет.",
                "Guest: Здравствуйте.",
                "******",
                "Host: Продолжим.",
            ]
        ),
        encoding="utf-8",
    )


def _write_mono_wav(path: Path, sample: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(sample * 8_000)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _omnivoice_dialogue_args(tmp_path: Path, catalog, *, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        provider="omnivoice-local",
        model="audio-cpp/omnivoice-q8_0",
        voice="host-profile",
        script=tmp_path / "dialogue.md",
        output_dir=tmp_path / "out",
        run_id="dialogue",
        format="dialogue",
        limit_chunks=None,
        retries=1,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=True,
        no_trim=True,
        json_output=True,
        json_events=False,
        resume=resume,
        with_timings=False,
        speaker_voice_map={"Host": "host-profile", "Guest": "guest-profile"},
        voice_bank_catalog=catalog,
        mode="preset",
    )


def _dialogue_catalog(tmp_path: Path):
    from voiceover_pipeline.omnivoice_voice_bank import load_voice_bank

    root = tmp_path / "bank"
    host_digest = _write_mono_wav(root / "voices" / "host.wav", b"\x00\x00")
    guest_digest = _write_mono_wav(root / "voices" / "guest.wav", b"\xff\x7f")
    catalog = {
        "schema_version": 1,
        "default_voice": "host-profile",
        "voices": [
            {
                "id": "host-profile",
                "display_name": "Host",
                "description": "",
                "language": "ru",
                "reference_audio": "voices/host.wav",
                "reference_text": "private host reference",
                "reference_sha256": host_digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 1},
            },
            {
                "id": "guest-profile",
                "display_name": "Guest",
                "description": "",
                "language": "ru",
                "reference_audio": "voices/guest.wav",
                "reference_text": "private guest reference",
                "reference_sha256": guest_digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 2},
            },
        ],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return load_voice_bank(catalog_path)


def test_build_provider_accepts_unbound_omnivoice_dialogue_catalog(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli

    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_environment(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cli.OmniVoiceLocalTTSProvider, "from_environment", fake_from_environment)
    catalog = _dialogue_catalog(tmp_path)
    args = argparse.Namespace(
        provider="omnivoice-local",
        mode="preset",
        format="dialogue",
        voice=None,
        voice_bank=catalog.root / "catalog.json",
        voice_bank_catalog=catalog,
    )

    provider = cli.build_provider(args, api_key="", style_prompt=None, prompt_mode="none")

    assert provider is sentinel
    assert captured == {"mode": "preset"}


def test_dialogue_and_gemini_dialogue_alias_produce_one_canonical_plan(tmp_path):
    from voiceover_pipeline.gemini_dialogue import (
        dialogue_turns_from_validation,
        validate_gemini_dialogue_file,
    )

    canonical = tmp_path / "canonical.md"
    legacy = tmp_path / "legacy.md"
    _write_dialogue_script(canonical, script_format="dialogue")
    _write_dialogue_script(legacy, script_format="gemini-dialogue")

    canonical_report = validate_gemini_dialogue_file(canonical)
    legacy_report = validate_gemini_dialogue_file(legacy)

    assert canonical_report["valid"] is True
    assert legacy_report["valid"] is True
    assert canonical_report["format"] == legacy_report["format"] == "dialogue"
    assert dialogue_turns_from_validation(canonical_report) == dialogue_turns_from_validation(
        legacy_report
    )


def test_voice_bank_rejects_distinct_profile_ids_with_same_reference_digest(tmp_path):
    from voiceover_pipeline.omnivoice_voice_bank import VoiceBankError, load_voice_bank

    root = tmp_path / "bank"
    digest = _write_mono_wav(root / "voices" / "shared.wav", b"\x00\x00")
    for name in ("first.wav", "second.wav"):
        (root / "voices" / name).write_bytes((root / "voices" / "shared.wav").read_bytes())
    catalog = {
        "schema_version": 1,
        "default_voice": "first",
        "voices": [
            {
                "id": "first",
                "display_name": "First",
                "description": "",
                "language": "ru",
                "reference_audio": "voices/first.wav",
                "reference_text": "private first reference",
                "reference_sha256": digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 1},
            },
            {
                "id": "second",
                "display_name": "Second",
                "description": "",
                "language": "ru",
                "reference_audio": "voices/second.wav",
                "reference_text": "private second reference",
                "reference_sha256": digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 2},
            },
        ],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(VoiceBankError, match="reference_sha256"):
        load_voice_bank(catalog_path)


def _write_tone_mp3(path: Path, frequency: int) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate=24000:duration=0.5",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


def _silence_durations(path: Path, minimum_seconds: float) -> list[float]:
    detect = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise=-45dB:d={minimum_seconds}",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert detect.returncode == 0, detect.stderr
    return [float(match) for match in re.findall(r"silence_duration:\s*([0-9.]+)", detect.stderr)]


def test_dialogue_concat_preserves_pcm_pause_with_real_ffmpeg(tmp_path):
    from voiceover_pipeline import media

    first = tmp_path / "turn_0001.mp3"
    second = tmp_path / "turn_0002.mp3"
    _write_tone_mp3(first, 440)
    _write_tone_mp3(second, 660)
    output = tmp_path / "full.mp3"

    media.concat_dialogue_turns(
        "ffmpeg",
        [(first, 600), (second, 0)],
        output,
    )

    assert 1_500 <= media.mp3_duration_ms("ffprobe", output) <= 1_700
    assert any(duration >= 0.4 for duration in _silence_durations(output, 0.4))
    assert not list(tmp_path.glob("dialogue-silence-*.wav"))


def test_dialogue_concat_preserves_each_requested_pause_duration(tmp_path):
    from voiceover_pipeline import media

    first = tmp_path / "turn_0001.mp3"
    second = tmp_path / "turn_0002.mp3"
    third = tmp_path / "turn_0003.mp3"
    for path, frequency in ((first, 440), (second, 660), (third, 880)):
        _write_tone_mp3(path, frequency)
    output = tmp_path / "full.mp3"

    media.concat_dialogue_turns(
        "ffmpeg",
        [(first, 250), (second, 600), (third, 0)],
        output,
    )

    assert 2_250 <= media.mp3_duration_ms("ffprobe", output) <= 2_450
    silence_durations = _silence_durations(output, 0.15)
    assert any(abs(duration - 0.25) <= 0.025 for duration in silence_durations)
    assert any(abs(duration - 0.6) <= 0.025 for duration in silence_durations)


def test_omnivoice_dialogue_resume_rejects_cast_change_before_synthesis(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ScriptChunk
    from voiceover_pipeline.run_state import atomic_write_json, initial_state

    class Provider:
        calls = 0

        def synthesize_chunk(self, _text: str, _chunk_id: str):
            self.calls += 1
            raise AssertionError("provider must not run after a resume identity mismatch")

    catalog = _dialogue_catalog(tmp_path)
    args = _omnivoice_dialogue_args(tmp_path, catalog, resume=True)
    chunks = [
        ScriptChunk(
            1,
            "turn_0001",
            "Host line",
            speaker="Host",
            voice="host-profile",
            pause_after_ms=250,
        ),
        ScriptChunk(
            2,
            "turn_0002",
            "Guest line",
            speaker="Guest",
            voice="guest-profile",
            pause_after_ms=0,
        ),
    ]
    identity = cli._dialogue_synthesis_identity(args, None, "auto", chunks)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="dialogue",
        run_id=args.run_id,
        synthesis_identity=identity,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    args.speaker_voice_map = {"Host": "host-profile", "Guest": "host-profile"}
    changed_chunks = [
        chunks[0],
        ScriptChunk(
            2,
            "turn_0002",
            "Guest line",
            speaker="Guest",
            voice="host-profile",
            pause_after_ms=0,
        ),
    ]
    provider = Provider()

    with pytest.raises(cli.CliError, match="synthesis identity changed") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", changed_chunks, "key", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert provider.calls == 0


def test_dialogue_resume_refuses_orphan_turn_audio_without_trusted_state(tmp_path):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ScriptChunk

    class Provider:
        calls = 0

        def synthesize_chunk(self, _text: str, _chunk_id: str):
            self.calls += 1
            raise AssertionError("provider must not run with orphan dialogue audio")

    catalog = _dialogue_catalog(tmp_path)
    args = _omnivoice_dialogue_args(tmp_path, catalog, resume=True)
    chunks = [
        ScriptChunk(
            1,
            "turn_0001",
            "Host line",
            speaker="Host",
            voice="host-profile",
            pause_after_ms=0,
        )
    ]
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    (paths.chunks_dir / "turn_0001.mp3").write_bytes(b"orphan")
    provider = Provider()

    with pytest.raises(cli.CliError, match="orphan dialogue audio") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert provider.calls == 0
