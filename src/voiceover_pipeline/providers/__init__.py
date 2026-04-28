from .base import TTSProvider
from .openrouter_tts import OpenRouterTTSProvider
from .polza_chat_audio import PolzaChatAudioProvider
from .polza_tts import PolzaTTSProvider

try:
    from .qwen_local import QwenLocalTTSProvider
except ModuleNotFoundError:
    QwenLocalTTSProvider = None  # type: ignore[assignment]

__all__ = ["TTSProvider", "PolzaChatAudioProvider", "OpenRouterTTSProvider", "PolzaTTSProvider"]
if QwenLocalTTSProvider is not None:
    __all__.append("QwenLocalTTSProvider")
