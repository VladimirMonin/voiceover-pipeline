"""Single provider synthesis invocation for one prepared speech part."""

from typing import Any

from ..models import SynthesisResult
from ..providers import OpenRouterTTSProvider
from .prepare import PreparedPart


def synthesize_part(provider: Any, part: PreparedPart) -> SynthesisResult:
    """Call the resolved provider once for one prepared part.

    Only provider selection and the existing OpenRouter ``voice`` argument are
    handled here. Paid attempt markers, recovery, retry policy, cost accounting,
    and media conversion stay in the ``services.execution`` part loop that calls
    this function, so no second executor exists for a paid synthesis.
    """
    selected_provider: Any = provider
    if isinstance(provider, dict):
        selected_provider = provider[part.voice]
    if not isinstance(selected_provider, OpenRouterTTSProvider):
        return selected_provider.synthesize_chunk(part.chunk.text, part.chunk.id)
    try:
        return selected_provider.synthesize_chunk(part.chunk.text, part.chunk.id, voice=part.voice)
    except TypeError as error:
        if "unexpected keyword argument 'voice'" not in str(error):
            raise
        return selected_provider.synthesize_chunk(part.chunk.text, part.chunk.id)
