"""Admission policy for the ``speech-parts`` / ``--text`` request route.

The Gemini 3.8 speech routes are ordinary on both cloud speech providers. Every
``/audio/speech`` request carries exactly one scalar ``voice`` plus the part's
effective direction in a separate ``instructions`` field, so the spoken ``input`` is
the exact part text and no request ever implies more than one speaker. One request
per part therefore honors different per-part voices. The verified external two-part
run sent exactly this body shape (``model``, ``input``, ``voice``,
``instructions``, ``response_format``) for every request on Polza and OpenRouter.

Two rules decide admission of a real request:

* a per-part instruction is admitted only on a route whose request carries it in a
  separate field; on every other route the provider would read the direction aloud
  or drop it, so the request is refused instead; and
* parts whose voices differ are admitted only on a route that already sends one
  voice per request; every other route speaks with its single run-level voice.

This module states only the request shape the application sends. It promises no
audible behavior, availability, or price.
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import GEMINI_38_TTS_MODELS

BLOCKED_PROVIDER_CONTRACT = "BLOCKED_PROVIDER_CONTRACT"

GEMINI_31_FLASH_TTS_MODEL = "google/gemini-3.1-flash-tts-preview"

# Routes whose request carries the part's effective direction in a separate
# ``instructions`` field, never inside the spoken ``input``.
INSTRUCTION_FIELD_SPEECH_ROUTES = frozenset(
    (provider, model)
    for provider in ("polza-tts", "openrouter-tts")
    for model in GEMINI_38_TTS_MODELS
)

# Routes that already send exactly one voice per request, so a part whose voice
# differs from the run voice is honored by its own request instead of being
# refused.
_CONFIRMED_PER_PART_VOICE_ROUTES = INSTRUCTION_FIELD_SPEECH_ROUTES | {
    ("openrouter-tts", GEMINI_31_FLASH_TTS_MODEL)
}


class SpeechPartsRouteError(RuntimeError):
    """A speech-parts request the application refuses before any key read or POST.

    The ``error_code`` is the stable machine code the CLI reports in its JSON
    envelope, so a route that cannot carry this exact request is always an explicit
    ``BLOCKED_PROVIDER_CONTRACT`` rather than a silent fallback or a dropped
    instruction.
    """

    error_code = BLOCKED_PROVIDER_CONTRACT


def require_confirmed_speech_parts_route(
    *,
    provider: str,
    model: str,
    voices: Iterable[str],
    has_vibe: bool,
) -> None:
    """Fail closed unless this exact speech-parts request is an admitted route.

    A non-empty effective instruction is admitted only on a route that transmits it
    in a separate field (the Gemini 3.8 speech routes); on any other route it would
    be read aloud or silently dropped, so the request is refused. Parts whose voices
    differ are admitted only on a route that already sends one voice per request. The
    message names the actionable state and never claims a route works.
    """
    if has_vibe and (provider, model) not in INSTRUCTION_FIELD_SPEECH_ROUTES:
        raise SpeechPartsRouteError(
            "BLOCKED_PROVIDER_CONTRACT: no admitted route for "
            f"{provider}/{model} carries a per-part instruction in a separate field, so the "
            "direction would be read aloud or silently dropped. Select a Gemini 3.8 speech "
            "model or remove the vibe. No request was sent and no API key was read."
        )
    distinct_voices = {voice for voice in voices if voice}
    if len(distinct_voices) > 1 and (provider, model) not in _CONFIRMED_PER_PART_VOICE_ROUTES:
        raise SpeechPartsRouteError(
            "BLOCKED_PROVIDER_CONTRACT: this provider/model speaks with one voice per run and "
            "cannot honor different per-part voices; only a route that is already confirmed to "
            "send one voice per request may. No request was sent and no API key was read."
        )
