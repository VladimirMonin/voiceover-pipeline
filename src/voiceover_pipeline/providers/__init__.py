from typing import Any

from .asr_registry import (
    ASRDependencyHealth,
    ASRProviderNotFoundError,
    ASRProviderRegistry,
    ASRProviderSpec,
    get_asr_provider_spec,
    list_asr_provider_specs,
)
from .audio_cpp_omnivoice_tts import OmniVoiceLocalTTSProvider
from .base import ASRProvider, TranscriptionProvider, TTSProvider
from .faster_whisper import FasterWhisperProvider
from .groq_whisper import GroqWhisperProvider
from .openrouter_tts import OpenRouterTTSProvider
from .openrouter_whisper import OpenRouterWhisperProvider
from .polza_chat_audio import PolzaChatAudioProvider
from .polza_tts import PolzaTTSProvider
from .xai_stt import XAISttProvider

QwenLocalTTSProvider: Any
try:
    from .qwen_local import QwenLocalTTSProvider as _QwenLocalTTSProvider

    QwenLocalTTSProvider = _QwenLocalTTSProvider
except ModuleNotFoundError:
    QwenLocalTTSProvider = None

__all__ = [
    "ASRDependencyHealth",
    "ASRProvider",
    "ASRProviderNotFoundError",
    "ASRProviderRegistry",
    "ASRProviderSpec",
    "TTSProvider",
    "TranscriptionProvider",
    "FasterWhisperProvider",
    "GroqWhisperProvider",
    "OpenRouterWhisperProvider",
    "PolzaChatAudioProvider",
    "OpenRouterTTSProvider",
    "PolzaTTSProvider",
    "XAISttProvider",
    "OmniVoiceLocalTTSProvider",
    "get_asr_provider_spec",
    "list_asr_provider_specs",
]
if QwenLocalTTSProvider is not None:
    __all__.append("QwenLocalTTSProvider")
