from .base import TTSProvider, TranscriptionProvider
from .faster_whisper import FasterWhisperProvider
from .openrouter_tts import OpenRouterTTSProvider
from .openrouter_whisper import OpenRouterWhisperProvider
from .polza_chat_audio import PolzaChatAudioProvider
from .polza_tts import PolzaTTSProvider

try:
    from .qwen_local import QwenLocalTTSProvider
except ModuleNotFoundError:
    QwenLocalTTSProvider = None  # type: ignore[assignment]

__all__ = [
    "TTSProvider",
    "TranscriptionProvider",
    "FasterWhisperProvider",
    "OpenRouterWhisperProvider",
    "PolzaChatAudioProvider",
    "OpenRouterTTSProvider",
    "PolzaTTSProvider",
]
if QwenLocalTTSProvider is not None:
    __all__.append("QwenLocalTTSProvider")
