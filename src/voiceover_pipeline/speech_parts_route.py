"""Offline admission policy for the ``speech-parts`` / ``--text`` request route.

The stable Gemini speech-parts contract through Polza remains **blocked**.
Authenticated model details and live experiments established its ``/audio/speech``
endpoint, scalar per-part voices, a returned WAV container despite requesting
MP3, and published RUB price components. A text instruction requesting a second
voice in the same POST produced one audible voice. Polza's published schemas show
one scalar voice per request, not a model-specific multi-speaker payload; the
``instructions`` field is documented only for a different model, and actual
external billing remains unverified. The separate experimental parts that yielded
one 6:17 MP3 with three voices required twelve distinct POSTs; they do not make
a multi-speaker request or the stable provider route confirmed. This module keeps
the candidate inert and refuses a stable submit before key access or network.

Two independent rules decide admission of a real request:

* only a confirmed provider/model route may send a per-part voice on its own
  request (currently the OpenRouter Gemini dialogue route); and
* no admitted route may silently discard or read aloud an effective vibe.

The candidate model is deliberately absent from the stable ``POLZA_TTS_MODELS``
catalog and from the ``list`` command, so nothing advertises it as available. It
exists here only as an inert, unverified mapping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

BLOCKED_PROVIDER_CONTRACT = "BLOCKED_PROVIDER_CONTRACT"

# The one candidate model the plan names. Its endpoint, scalar voice transport,
# actual WAV response and published price components were observed, but a
# documented multi-speaker payload and external billing were not. Experimental
# per-part instructions do not establish stable instruction semantics. Flash-Lite
# remains absent, and neither model is advertised as a stable route.
_CANDIDATE_GEMINI_SPEECH_PARTS_MODEL = "google/gemini-3.8-flash-tts"

# Routes on which a per-part cast voice is already confirmed: the OpenRouter Gemini
# dialogue route sends one cast voice per request today. Any other provider speaks
# with its single run-level voice, so parts whose voices differ cannot be honored.
_CONFIRMED_PER_PART_VOICE_ROUTES = frozenset(
    {("openrouter-tts", "google/gemini-3.1-flash-tts-preview")}
)


@dataclass(frozen=True)
class SpeechPartsRoute:
    """One candidate speech-parts request contract and its verification state.

    ``required_text_wrapper`` is the exact service text the adapter must add around
    the spoken transcript to carry the instruction; it is charged to the pre-submit
    character budget. For the unverified candidate it is empty because no real
    wrapper has been observed, which is recorded as a residual rather than guessed.
    """

    provider: str
    model: str
    verified: bool
    carries_vibe: bool
    per_part_voice: bool
    required_text_wrapper: str


_CANDIDATE_ROUTES: tuple[SpeechPartsRoute, ...] = (
    SpeechPartsRoute(
        provider="polza-tts",
        model=_CANDIDATE_GEMINI_SPEECH_PARTS_MODEL,
        verified=False,
        carries_vibe=False,
        per_part_voice=False,
        required_text_wrapper="",
    ),
)


class SpeechPartsRouteError(RuntimeError):
    """A speech-parts request the application refuses before any key read or POST.

    The ``error_code`` is the stable machine code the CLI reports in its JSON
    envelope, so an unconfirmed provider contract is always an explicit
    ``BLOCKED_PROVIDER_CONTRACT`` rather than a silent fallback or a dropped vibe.
    """

    error_code = BLOCKED_PROVIDER_CONTRACT


def candidate_speech_parts_route(provider: str, model: str) -> SpeechPartsRoute | None:
    """Return the named candidate route for this provider/model, if any."""
    for route in _CANDIDATE_ROUTES:
        if route.provider == provider and route.model == model:
            return route
    return None


def is_candidate_speech_parts_model(provider: str, model: str) -> bool:
    """Whether this provider/model names an inert, unverified candidate route."""
    return candidate_speech_parts_route(provider, model) is not None


def speech_parts_required_text_wrapper(provider: str, model: str) -> str:
    """Return the adapter service text charged to one speech-parts request budget.

    Only a candidate route can name one; any other route records none, so its
    budget is exactly the spoken text plus the effective instruction.
    """
    route = candidate_speech_parts_route(provider, model)
    return "" if route is None else route.required_text_wrapper


def require_confirmed_speech_parts_route(
    *, provider: str, model: str, voices: Iterable[str], has_vibe: bool
) -> None:
    """Fail closed unless this exact speech-parts request is a confirmed route.

    A named candidate route is unverified and always refused; a non-empty effective
    vibe is refused because no confirmed route can carry an instruction without the
    provider reading it; and parts whose voices differ are refused unless this exact
    provider/model is a route that already sends one cast voice per request. The
    message names the actionable state and never claims a route works.
    """
    route = candidate_speech_parts_route(provider, model)
    if route is not None and not route.verified:
        raise SpeechPartsRouteError(
            f"BLOCKED_PROVIDER_CONTRACT: the provider/model route {provider}/{model} is a "
            "candidate only; its catalog-confirmed ID is not a verified speech-parts route. "
            "Polza lists /audio/speech, but its published schema has one scalar voice per POST; "
            "there is no documented payload for two distinct voices in one POST. An approved "
            "instruction-only two-voice experiment was heard as one voice. Separate single-voice "
            "POSTs have produced a multi-voice final MP3, but do not verify that request contract "
            "or stable Gemini instruction behavior. Published prices and API usage are not an "
            "external invoice or a guaranteed bill ceiling. This candidate remains blocked "
            "before key access or any paid request; no request was sent and no API key was read."
        )
    if has_vibe:
        raise SpeechPartsRouteError(
            "BLOCKED_PROVIDER_CONTRACT: no confirmed provider route can carry a per-part vibe "
            "instruction without reading it aloud, so the instruction would be silently "
            "dropped. A route that transmits an instruction must be confirmed with one "
            "approved live probe before this run may submit. No request was sent and no API "
            "key was read."
        )
    distinct_voices = {voice for voice in voices if voice}
    if len(distinct_voices) > 1 and (provider, model) not in _CONFIRMED_PER_PART_VOICE_ROUTES:
        raise SpeechPartsRouteError(
            "BLOCKED_PROVIDER_CONTRACT: this provider/model speaks with one voice per run and "
            "cannot honor different per-part voices; only a route that is already confirmed to "
            "send one cast voice per request may. No request was sent and no API key was read."
        )
