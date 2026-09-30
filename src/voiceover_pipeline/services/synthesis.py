"""Single provider synthesis invocation for one prepared speech part."""

import inspect
from typing import Any

from ..models import SynthesisResult
from ..speech_parts_route import SpeechPartsRouteError
from .prepare import PreparedPart


def _accepts_keyword(synthesize_chunk: Any, keyword: str) -> bool:
    """Return whether a provider callable accepts ``keyword``.

    Signature inspection happens before any provider invocation, so a legacy
    two-argument signature is never sent a first submit that a rejected keyword
    would then duplicate. A signature that cannot be inspected propagates rather
    than guessing, which would risk an unconfirmed paid call.
    """
    parameters = inspect.signature(synthesize_chunk).parameters
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def synthesize_part(provider: Any, part: PreparedPart) -> SynthesisResult:
    """Call the resolved provider once for one prepared part.

    Only provider selection, the existing OpenRouter ``voice`` argument, and the
    ``speech-parts`` ``vibe`` argument are handled here. A part that carries a
    non-empty effective instruction is sent only to a provider callable that accepts
    a ``vibe`` keyword; otherwise the call fails closed before any POST instead of
    dropping the instruction silently. Paid attempt markers, recovery, retry policy,
    cost accounting, and media conversion stay in the ``services.execution`` part
    loop that calls this function, so no second executor exists for a paid
    synthesis. Each part is invoked exactly once, and any provider error propagates
    unretried.
    """
    selected_provider: Any = provider
    if isinstance(provider, dict):
        selected_provider = provider[part.voice]
    synthesize_chunk = selected_provider.synthesize_chunk
    effective_vibe = part.direction.effective_vibe if part.direction is not None else ""
    accepts_voice = _accepts_keyword(synthesize_chunk, "voice")
    accepts_vibe = _accepts_keyword(synthesize_chunk, "vibe")
    if effective_vibe and not accepts_vibe:
        raise SpeechPartsRouteError(
            "BLOCKED_PROVIDER_CONTRACT: the selected provider cannot carry a per-part vibe "
            "without reading it aloud, so the instruction would be silently dropped. The "
            "request was not sent."
        )
    kwargs: dict[str, str | None] = {}
    if accepts_voice:
        kwargs["voice"] = part.voice
    if accepts_vibe:
        kwargs["vibe"] = effective_vibe
    return synthesize_chunk(part.chunk.text, part.chunk.id, **kwargs)
