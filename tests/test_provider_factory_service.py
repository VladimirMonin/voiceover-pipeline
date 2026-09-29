"""Direct service-boundary tests for TTS provider construction and dialogue binding.

These exercise ``services.provider_factory`` without going through
``cli.build_provider``: the selected provider type and request identity, the
per-cast dialogue provider map, and the typed errors the CLI wrapper translates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import pytest

from voiceover_pipeline.config import QWEN_INSTRUCT
from voiceover_pipeline.omnivoice_voice_bank import load_voice_bank
from voiceover_pipeline.providers import (
    OmniVoiceLocalTTSProvider,
    PolzaChatAudioProvider,
    PolzaTTSProvider,
    QwenLocalTTSProvider,
)
from voiceover_pipeline.services.provider_factory import (
    DialogueProviderBindingError,
    ProviderConfigurationError,
    bind_dialogue_voice_bank_providers,
    build_tts_provider,
)


def _mono_wav(path: Path, sample: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(sample * 8_000)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _voice_bank(tmp_path: Path):
    root = tmp_path / "bank"
    host_digest = _mono_wav(root / "voices" / "host.wav", b"\x00\x00")
    guest_digest = _mono_wav(root / "voices" / "guest.wav", b"\xff\x7f")
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
                "reference_text": "host reference",
                "reference_sha256": host_digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 1},
            },
            {
                "id": "guest-profile",
                "display_name": "Guest",
                "description": "",
                "language": "ru",
                "reference_audio": "voices/guest.wav",
                "reference_text": "guest reference",
                "reference_sha256": guest_digest,
                "origin": {"mode": "owner-reference", "instruction": None, "seed": 2},
            },
        ],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return load_voice_bank(catalog_path)


def test_build_tts_provider_constructs_selected_provider_with_request_identity():
    chat = build_tts_provider(
        argparse.Namespace(
            provider="polza-chat-audio",
            model="gpt-audio",
            voice="Alloy",
            fallback_voice="Echo",
        ),
        api_key="sk-test",
        style_prompt=None,
        prompt_mode="none",
    )
    assert isinstance(chat, PolzaChatAudioProvider)
    assert (chat.model, chat.voice, chat.fallback_voice) == ("gpt-audio", "Alloy", "Echo")

    plain = build_tts_provider(
        argparse.Namespace(provider="polza-tts", model="elevenlabs/foo", voice="Alice"),
        api_key="sk-test",
        style_prompt=None,
        prompt_mode="none",
    )
    assert isinstance(plain, PolzaTTSProvider)
    assert (plain.model, plain.voice) == ("elevenlabs/foo", "Alice")

    local = build_tts_provider(
        argparse.Namespace(
            provider="qwen-local",
            mode="preset",
            voice="Serena",
            sample=None,
            sample_text="",
        ),
        api_key="",
        style_prompt=None,
        prompt_mode="none",
    )
    assert isinstance(local, QwenLocalTTSProvider)
    assert local._instruct == QWEN_INSTRUCT


def test_bind_dialogue_voice_bank_providers_clones_each_cast_voice_in_order(tmp_path):
    catalog = _voice_bank(tmp_path)
    provider = OmniVoiceLocalTTSProvider()

    providers = bind_dialogue_voice_bank_providers(
        provider, catalog, {"Host": "host-profile", "Guest": "guest-profile"}
    )

    assert list(providers) == ["host-profile", "guest-profile"]
    for voice_id in ("host-profile", "guest-profile"):
        cloned = providers[voice_id]
        assert isinstance(cloned, OmniVoiceLocalTTSProvider)
        assert cloned is not provider
        assert cloned._voice_bank is not None
        assert cloned._voice_bank[0].id == voice_id

    with pytest.raises(DialogueProviderBindingError, match="did not build an OmniVoice provider"):
        bind_dialogue_voice_bank_providers(object(), catalog, {"Host": "host-profile"})


def test_build_tts_provider_translates_qwen_runtime_and_unsupported_provider(monkeypatch):
    monkeypatch.setenv("VOICEOVER_QWEN_TTS_RUNTIME", "automatic")
    qwen_args = argparse.Namespace(
        provider="qwen-local",
        mode="preset",
        voice="Serena",
        sample=None,
        sample_text="",
    )

    with pytest.raises(ProviderConfigurationError, match="VOICEOVER_QWEN_TTS_RUNTIME"):
        build_tts_provider(qwen_args, api_key="", style_prompt=None, prompt_mode="none")

    with pytest.raises(RuntimeError, match="Unsupported provider: mystery") as error:
        build_tts_provider(
            argparse.Namespace(provider="mystery"),
            api_key="",
            style_prompt=None,
            prompt_mode="none",
        )
    assert type(error.value) is RuntimeError

    import voiceover_pipeline.cli as cli

    with pytest.raises(cli.CliError, match="VOICEOVER_QWEN_TTS_RUNTIME") as cli_error:
        cli.build_provider(qwen_args, api_key="", style_prompt=None, prompt_mode="none")
    assert cli_error.value.code == 2
