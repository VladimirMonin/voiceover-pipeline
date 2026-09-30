"""Finite dispatch matrix for the DB-first (native) execution routes of S05.

Plan section 5 requires every *actually functioning* fresh TTS/ASR/timing/verify
journey to record its canonical history, while a route that is genuinely
unimplemented or invalid stays on the untouched legacy writer and reports a
usage error. This module is the single consolidated table for that decision: each
row records one finite route combination and whether ``cli._native_route_eligible``
selects the canonical-history executor (``native``) or keeps the legacy writer.

The table is deliberately semantic, not a flag Cartesian product: it covers
provider submission shape and local mode, the effective script format, the
recorded trimming semantics and the applicable integrated timing/quality steps.
A parser choice or an ignored flag is *not* a functioning route; the two routes
that are explicitly unimplemented -- ``qwen-local --mode auto`` and the
``openrouter-whisper`` timing provider -- are asserted separately as fail-closed
usage/rejection errors rather than silent legacy work.

Everything here is offline and in-memory: no provider is built, no key is read,
no file is written, and no network call is possible.
"""

from __future__ import annotations

import argparse

import pytest

import voiceover_pipeline.cli as cli
from voiceover_pipeline.config import (
    DEFAULT_OMNIVOICE_VOICE,
    OMNIVOICE_LOCAL_MODEL_ID,
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
)

POLZA_MEDIA_MODEL = "elevenlabs/text-to-speech-turbo-2-5"
POLZA_SYNC_MODEL = "openai/gpt-4o-mini-tts"
POLZA_CHAT_AUDIO_MODEL = "openai/gpt-audio-mini"
GEMINI_TTS_MODEL = "google/gemini-3.1-flash-tts-preview"

# A stand-in for the resolved voice-bank catalog/profile a preset route carries.
_BANK = object()


def _args(**overrides: object) -> argparse.Namespace:
    """Build a namespace with the shared non-dialogue defaults overridden in place."""
    base: dict[str, object] = {
        "provider": "polza-tts",
        "model": POLZA_MEDIA_MODEL,
        "mode": None,
        "no_trim": False,
        "with_timings": False,
        "timing_provider": "faster-whisper",
        "tts_quality_provider": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# (id, overrides, script_format, expected_native, reason)
_ROUTE_MATRIX = [
    # ── ordinary non-dialogue cloud TTS ──────────────────────────────────────
    (
        "polza-media-markdown",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL},
        "markdown",
        True,
        "async /media ordinary route",
    ),
    (
        "polza-sync-markdown",
        {"provider": "polza-tts", "model": POLZA_SYNC_MODEL},
        "markdown",
        True,
        "synchronous /audio/speech ordinary route",
    ),
    (
        "polza-chat-audio-markdown",
        {"provider": "polza-chat-audio", "model": POLZA_CHAT_AUDIO_MODEL},
        "markdown",
        True,
        "ordinary chat-audio route",
    ),
    (
        "openrouter-tts-markdown",
        {"provider": "openrouter-tts", "model": GEMINI_TTS_MODEL},
        "markdown",
        True,
        "ordinary OpenRouter route",
    ),
    # ── voiceover format for the ordinary cloud routes ───────────────────────
    (
        "polza-media-voiceover",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL},
        "voiceover",
        True,
        "voiceover validator resolves one provider/model/voice",
    ),
    (
        "polza-chat-audio-voiceover",
        {"provider": "polza-chat-audio", "model": POLZA_CHAT_AUDIO_MODEL},
        "voiceover",
        True,
        "voiceover validator resolves one provider/model/voice",
    ),
    (
        "openrouter-tts-voiceover",
        {"provider": "openrouter-tts", "model": GEMINI_TTS_MODEL},
        "voiceover",
        True,
        "voiceover validator resolves one provider/model/voice",
    ),
    # ── local Qwen non-dialogue modes ────────────────────────────────────────
    (
        "qwen-preset-markdown",
        {"provider": "qwen-local", "model": QWEN_MODEL_CUSTOMVOICE, "mode": "preset"},
        "markdown",
        True,
        "instructed preset mode",
    ),
    (
        "qwen-clone-markdown",
        {"provider": "qwen-local", "model": QWEN_MODEL_BASE, "mode": "clone", "sample": "ref.wav"},
        "markdown",
        True,
        "clone with a reference sample",
    ),
    (
        "qwen-design-markdown",
        {"provider": "qwen-local", "model": QWEN_MODEL_VOICE_DESIGN, "mode": "design"},
        "markdown",
        True,
        "instructed design mode",
    ),
    (
        "qwen-preset-voiceover",
        {"provider": "qwen-local", "model": QWEN_MODEL_CUSTOMVOICE, "mode": "preset"},
        "voiceover",
        True,
        "voiceover keeps the validated preset voice",
    ),
    (
        "qwen-clone-voiceover",
        {"provider": "qwen-local", "model": QWEN_MODEL_BASE, "mode": "clone", "sample": "ref.wav"},
        "voiceover",
        True,
        "clone replaces the voice with its mode marker as in legacy",
    ),
    (
        "qwen-design-voiceover",
        {"provider": "qwen-local", "model": QWEN_MODEL_VOICE_DESIGN, "mode": "design"},
        "voiceover",
        True,
        "design replaces the voice with its mode marker as in legacy",
    ),
    # ── local OmniVoice non-dialogue modes ───────────────────────────────────
    (
        "omnivoice-preset-markdown",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "preset",
            "voice_bank_catalog": _BANK,
            "voice_bank_profile": _BANK,
        },
        "markdown",
        True,
        "preset bank monologue route",
    ),
    (
        "omnivoice-auto-markdown",
        {"provider": "omnivoice-local", "model": OMNIVOICE_LOCAL_MODEL_ID, "mode": "auto"},
        "markdown",
        True,
        "auto mode leaves voice selection to the runtime",
    ),
    (
        "omnivoice-clone-markdown",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "clone",
            "reference_audio": "ref.wav",
            "reference_text": "reference",
        },
        "markdown",
        True,
        "clone with reference audio and text",
    ),
    (
        "omnivoice-design-markdown",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "design",
            "design_instruction": "warm narrator",
        },
        "markdown",
        True,
        "design with an instruction",
    ),
    (
        "omnivoice-preset-voiceover",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "preset",
            "voice_bank_catalog": _BANK,
            "voice_bank_profile": _BANK,
        },
        "voiceover",
        True,
        "the only voiceover-admitted OmniVoice route",
    ),
    # ── recorded trimming and integrated steps ───────────────────────────────
    (
        "cloud-no-trim",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL, "no_trim": True},
        "markdown",
        True,
        "recorded trimming semantics",
    ),
    (
        "cloud-timing-local",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL, "with_timings": True},
        "markdown",
        True,
        "integrated local faster-whisper timing",
    ),
    (
        "cloud-timing-groq",
        {
            "provider": "polza-tts",
            "model": POLZA_MEDIA_MODEL,
            "with_timings": True,
            "timing_provider": "groq-whisper",
        },
        "markdown",
        True,
        "integrated paid cloud Groq timing",
    ),
    (
        "cloud-timing-xai",
        {
            "provider": "polza-tts",
            "model": POLZA_MEDIA_MODEL,
            "with_timings": True,
            "timing_provider": "xai-stt",
        },
        "markdown",
        True,
        "integrated paid cloud xAI timing",
    ),
    (
        "cloud-quality-qwen",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL, "tts_quality_provider": "qwen-local"},
        "markdown",
        True,
        "installed local quality provider",
    ),
    (
        "cloud-quality-nemotron",
        {
            "provider": "polza-tts",
            "model": POLZA_MEDIA_MODEL,
            "tts_quality_provider": "nemotron-local",
        },
        "markdown",
        True,
        "installed local quality provider",
    ),
    (
        "cloud-timing-and-quality-combined",
        {
            "provider": "polza-tts",
            "model": POLZA_MEDIA_MODEL,
            "with_timings": True,
            "timing_provider": "groq-whisper",
            "tts_quality_provider": "nemotron-local",
        },
        "markdown",
        True,
        "combined integrated timing and local quality",
    ),
    # ── native dialogue routes ───────────────────────────────────────────────
    (
        "openrouter-dialogue-local-quality",
        {
            "provider": "openrouter-tts",
            "model": GEMINI_TTS_MODEL,
            "tts_quality_provider": "qwen-local",
        },
        "dialogue",
        True,
        "required installed local per-turn gate",
    ),
    (
        "openrouter-dialogue-xai-quality",
        {
            "provider": "openrouter-tts",
            "model": GEMINI_TTS_MODEL,
            "tts_quality_provider": "xai-stt",
        },
        "dialogue",
        True,
        "paid cloud xai per-turn gate",
    ),
    (
        "openrouter-dialogue-cloud-timing",
        {
            "provider": "openrouter-tts",
            "model": GEMINI_TTS_MODEL,
            "tts_quality_provider": "qwen-local",
            "with_timings": True,
            "timing_provider": "groq-whisper",
        },
        "dialogue",
        True,
        "integrated paid cloud timing on the dialogue route",
    ),
    (
        "omnivoice-dialogue-no-quality",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "preset",
            "voice_bank_catalog": _BANK,
        },
        "dialogue",
        True,
        "preset bank dialogue with optional quality",
    ),
    (
        "omnivoice-dialogue-xai-quality",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "preset",
            "voice_bank_catalog": _BANK,
            "tts_quality_provider": "xai-stt",
        },
        "dialogue",
        True,
        "paid cloud xai per-turn gate on the OmniVoice route",
    ),
    (
        "omnivoice-dialogue-cloud-timing",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "preset",
            "voice_bank_catalog": _BANK,
            "with_timings": True,
            "timing_provider": "xai-stt",
        },
        "dialogue",
        True,
        "integrated paid cloud timing on the OmniVoice dialogue route",
    ),
    # ── legacy-preserved / not-a-functioning-route dispositions ──────────────
    (
        "polza-tts-dialogue",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL},
        "dialogue",
        False,
        "Polza dialogue is not a validated Gemini cast route (usage error)",
    ),
    (
        "polza-chat-audio-dialogue",
        {"provider": "polza-chat-audio", "model": POLZA_CHAT_AUDIO_MODEL},
        "dialogue",
        False,
        "chat-audio is not the Gemini dialogue model (usage error)",
    ),
    (
        "omnivoice-nonpreset-dialogue",
        {"provider": "omnivoice-local", "model": OMNIVOICE_LOCAL_MODEL_ID, "mode": "auto"},
        "dialogue",
        False,
        "only the OmniVoice preset bank carries cast profiles",
    ),
    (
        "omnivoice-nonpreset-voiceover",
        {
            "provider": "omnivoice-local",
            "model": OMNIVOICE_LOCAL_MODEL_ID,
            "mode": "clone",
            "reference_audio": "ref.wav",
            "reference_text": "reference",
            "voice": DEFAULT_OMNIVOICE_VOICE,
        },
        "voiceover",
        False,
        "a non-preset mode rejects the voice a voiceover script supplies",
    ),
    (
        "cloud-quality-xai-nondialogue",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL, "tts_quality_provider": "xai-stt"},
        "markdown",
        False,
        "non-dialogue xai quality is an ignored flag, not a functioning route",
    ),
    (
        "unknown-quality-provider",
        {"provider": "polza-tts", "model": POLZA_MEDIA_MODEL, "tts_quality_provider": "nope-local"},
        "markdown",
        False,
        "an unregistered quality provider is not admitted",
    ),
    (
        "openrouter-dialogue-without-quality",
        {"provider": "openrouter-tts", "model": GEMINI_TTS_MODEL},
        "dialogue",
        False,
        "the per-turn gate is required on the OpenRouter dialogue route",
    ),
]


@pytest.mark.parametrize(
    ("overrides", "script_format", "expected_native", "reason"),
    [pytest.param(row[1], row[2], row[3], row[4], id=row[0]) for row in _ROUTE_MATRIX],
)
def test_native_route_dispatch_matrix(overrides, script_format, expected_native, reason):
    """Each finite route combination resolves to its documented writer."""
    verdict = cli._native_route_eligible(_args(**overrides), script_format)
    assert verdict is expected_native, reason


@pytest.mark.parametrize("timing_provider", ["openrouter-whisper"])
def test_openrouter_whisper_timing_is_never_admitted(timing_provider):
    """The timestamp-less provider never reaches the native writer."""
    args = _args(
        provider="polza-tts",
        model=POLZA_MEDIA_MODEL,
        with_timings=True,
        timing_provider=timing_provider,
    )
    assert cli._native_timing_step_admitted(args) is False
    assert cli._native_route_eligible(args, "markdown") is False


def test_unimplemented_qwen_auto_mode_fails_as_a_usage_error():
    """``qwen-local --mode auto`` is rejected before any provider or model is chosen."""
    args = _args(provider="qwen-local", model="unset", mode="auto")
    with pytest.raises(cli.CliError) as excinfo:
        cli._resolve_qwen_mode_identity(args)
    assert excinfo.value.code == cli._EXIT_ARGS
    assert "not implemented" in str(excinfo.value)
    # No mode was substituted and no model/voice was silently invented.
    assert args.model == "unset"
