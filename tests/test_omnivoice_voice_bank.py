from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from voiceover_pipeline.omnivoice_voice_bank import (
    VoiceBankError,
    load_voice_bank,
    resolve_bank_profile,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reference_wav(path: Path, *, rate: int = 24_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes((b"\x00\x00" * 8000) + (b"\xff\x7f" * 8000))


def build_bank(tmp_path: Path, *, voices=None, default_voice="main", write_wav=True, **overrides):
    """Build a valid voice bank in tmp_path and return the catalog path."""
    voices = voices or [
        {
            "id": "main",
            "display_name": "Main Narrator",
            "description": "Primary narration voice",
            "language": "ru",
            "reference_audio": "voices/main.wav",
            "reference_text": "Эталонная фраза для клонирования.",
            "reference_sha256": "x",
            "origin": {"mode": "owner-reference", "instruction": None, "seed": 7},
        }
    ]
    bank_root = tmp_path / "voice-bank"
    bank_root.mkdir(exist_ok=True)
    catalog: dict = {
        "schema_version": 1,
        "default_voice": default_voice,
        "voices": voices,
    }
    catalog.update(overrides)
    if write_wav:
        for item in voices:
            audio_path = bank_root / item["reference_audio"]
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            _write_reference_wav(audio_path)
            item["reference_sha256"] = _sha256(audio_path)
    else:
        for item in voices:
            if item.get("reference_sha256") == "x":
                item["reference_sha256"] = "0" * 64
    catalog_path = bank_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path


def test_valid_catalog_loads_and_default_profile_resolves(tmp_path):
    catalog_path = build_bank(tmp_path)
    catalog = load_voice_bank(catalog_path)

    assert catalog.default_voice == "main"
    assert [profile.id for profile in catalog.profiles] == ["main"]
    profile = catalog.profiles[0]
    assert profile.display_name == "Main Narrator"
    assert profile.language == "ru"
    assert profile.reference_sha256 == _sha256(catalog.root / "voices/main.wav")
    assert profile.origin == {"mode": "owner-reference", "instruction": None, "seed": 7}

    resolved_profile, reference_path = resolve_bank_profile(catalog, "main")
    assert resolved_profile is profile
    assert reference_path == catalog.root / "voices/main.wav"
    assert reference_path.is_file()


def test_duplicate_profile_ids_fail(tmp_path):
    voices = [
        {
            "id": "dup",
            "display_name": "One",
            "description": "",
            "language": "ru",
            "reference_audio": "voices/one.wav",
            "reference_text": "text",
            "reference_sha256": "x",
            "origin": {"mode": "auto", "instruction": None, "seed": None},
        },
        {
            "id": "dup",
            "display_name": "Two",
            "description": "",
            "language": "ru",
            "reference_audio": "voices/two.wav",
            "reference_text": "text",
            "reference_sha256": "x",
            "origin": {"mode": "auto", "instruction": None, "seed": None},
        },
    ]
    catalog_path = build_bank(tmp_path, voices=voices)

    with pytest.raises(VoiceBankError, match="duplicate"):
        load_voice_bank(catalog_path)


def test_missing_default_voice_fails(tmp_path):
    catalog_path = build_bank(tmp_path, default_voice="ghost")

    with pytest.raises(VoiceBankError, match="default_voice"):
        load_voice_bank(catalog_path)


@pytest.mark.parametrize(
    "reference_audio",
    [
        "C:/voices/main.wav",
        "C:\\voices\\main.wav",
        "D:/abs.wav",
        "/abs/path.wav",
        "../voices/main.wav",
        "..\\voices\\main.wav",
        "voices/../main.wav",
        "voices\\..\\main.wav",
        "voices/../sub/main.wav",
    ],
)
def test_escape_and_absolute_reference_paths_fail(tmp_path, reference_audio):
    voices = [
        {
            "id": "main",
            "display_name": "Main",
            "description": "",
            "language": "ru",
            "reference_audio": reference_audio,
            "reference_text": "text",
            "reference_sha256": "x",
            "origin": {"mode": "owner-reference", "instruction": None, "seed": None},
        }
    ]
    catalog_path = build_bank(tmp_path, voices=voices, write_wav=False)

    with pytest.raises(VoiceBankError, match="escape|outside the bank"):
        load_voice_bank(catalog_path)


def test_reference_sha256_mismatch_fails(tmp_path):
    catalog_path = build_bank(tmp_path)
    catalog = load_voice_bank(catalog_path)
    catalog = _replace_sha256(catalog, "0" * 64)

    with pytest.raises(VoiceBankError, match="SHA-256 mismatch"):
        resolve_bank_profile(catalog, "main")


def test_non_wav_reference_fails(tmp_path):
    catalog_path = build_bank(tmp_path, write_wav=False)
    bank_root = catalog_path.parent
    audio_path = bank_root / "voices/main.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"not a wav file")
    catalog = _load_catalog_with_digest(catalog_path, audio_path)

    with pytest.raises(VoiceBankError, match="mono WAV"):
        resolve_bank_profile(catalog, "main")


def test_stereo_wav_reference_fails(tmp_path):
    catalog_path = build_bank(tmp_path, write_wav=False)
    bank_root = catalog_path.parent
    audio_path = bank_root / "voices/main.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * 8000)
    catalog = _load_catalog_with_digest(catalog_path, audio_path)

    with pytest.raises(VoiceBankError, match="mono WAV"):
        resolve_bank_profile(catalog, "main")


def test_valid_wav_passes_resolution_even_at_non_24k_rate(tmp_path):
    catalog_path = build_bank(tmp_path, write_wav=False)
    bank_root = catalog_path.parent
    audio_path = bank_root / "voices/main.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _write_reference_wav(audio_path, rate=16_000)
    catalog = _load_catalog_with_digest(catalog_path, audio_path)

    profile, path = resolve_bank_profile(catalog, "main")
    assert profile.id == "main"
    assert path.is_file()


def test_schema_version_must_be_one(tmp_path):
    catalog_path = build_bank(tmp_path, schema_version=2)

    with pytest.raises(VoiceBankError, match="schema_version"):
        load_voice_bank(catalog_path)


def test_catalog_missing_file_fails():
    with pytest.raises(VoiceBankError, match="not a readable file"):
        load_voice_bank(Path("does-not-exist-catalog.json"))


def test_invalid_reference_sha256_format_fails(tmp_path):
    voices = [
        {
            "id": "main",
            "display_name": "Main",
            "description": "",
            "language": "ru",
            "reference_audio": "voices/main.wav",
            "reference_text": "text",
            "reference_sha256": "not-hex",
            "origin": {"mode": "auto", "instruction": None, "seed": None},
        }
    ]
    catalog_path = build_bank(tmp_path, voices=voices, write_wav=False)

    with pytest.raises(VoiceBankError, match="sha256"):
        load_voice_bank(catalog_path)


def test_missing_reference_file_fails(tmp_path):
    catalog_path = build_bank(tmp_path, write_wav=False)
    catalog = load_voice_bank(catalog_path)

    with pytest.raises(VoiceBankError, match="not a readable file"):
        resolve_bank_profile(catalog, "main")


def _replace_sha256(catalog, digest):
    return type(catalog)(
        root=catalog.root,
        default_voice=catalog.default_voice,
        profiles=tuple(
            type(profile)(
                id=profile.id,
                display_name=profile.display_name,
                description=profile.description,
                language=profile.language,
                reference_audio=profile.reference_audio,
                reference_text=profile.reference_text,
                reference_sha256=digest,
                origin=profile.origin,
            )
            for profile in catalog.profiles
        ),
    )


def _load_catalog_with_digest(catalog_path: Path, audio_path: Path):
    catalog = load_voice_bank(catalog_path)
    return _replace_sha256(catalog, _sha256(audio_path))
