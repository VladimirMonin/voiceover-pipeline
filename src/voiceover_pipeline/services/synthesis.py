"""Single provider synthesis invocation for one prepared speech part."""

import inspect
from typing import Any

from ..models import SynthesisResult
from ..providers import OpenRouterTTSProvider
from .prepare import PreparedPart


def _accepts_voice(synthesize_chunk: Any) -> bool:
    """Return whether a provider callable accepts the OpenRouter ``voice`` keyword.

    Signature inspection happens before any provider invocation, so a legacy
    two-argument signature is never sent a first submit that a rejected keyword
    would then duplicate. A signature that cannot be inspected propagates rather
    than guessing, which would risk an unconfirmed paid call.
    """
    parameters = inspect.signature(synthesize_chunk).parameters
    return "voice" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def synthesize_part(provider: Any, part: PreparedPart) -> SynthesisResult:
    """Call the resolved provider once for one prepared part.

    Only provider selection and the existing OpenRouter ``voice`` argument are
    handled here. Paid attempt markers, recovery, retry policy, cost accounting,
    and media conversion stay in the ``services.execution`` part loop that calls
    this function, so no second executor exists for a paid synthesis. Each part
    is invoked exactly once, and any provider error propagates unretried.
    """
    selected_provider: Any = provider
    if isinstance(provider, dict):
        selected_provider = provider[part.voice]
    synthesize_chunk = selected_provider.synthesize_chunk
    if not isinstance(selected_provider, OpenRouterTTSProvider) or not _accepts_voice(
        synthesize_chunk
    ):
        return synthesize_chunk(part.chunk.text, part.chunk.id)
    return synthesize_chunk(part.chunk.text, part.chunk.id, voice=part.voice)
