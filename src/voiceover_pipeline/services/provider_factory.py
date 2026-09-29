"""TTS provider construction and dialogue voice-bank binding for the CLI entry point.

This module owns the provider-type dispatch the CLI performs after argument
validation, plus the dialogue voice-bank map that clones one provider per cast
voice. It imports the provider classes directly and never imports ``cli``, so
the CLI keeps only a thin wrapper that forwards already-validated arguments and
translates the typed errors here into the existing exit codes and messages.

This constructor does not read API keys or submit TTS requests. The optional
audio.cpp Qwen provider is imported only when its runtime is selected.
Unsupported provider identifiers still raise the exact
``RuntimeError`` the CLI wrapper lets propagate unchanged.
"""

import argparse
import os
from collections.abc import Mapping
from typing import Any

from ..config import QWEN_INSTRUCT
from ..gemini_dialogue import is_dialogue_format
from ..omnivoice_voice_bank import VoiceBankCatalog, resolve_bank_profile
from ..providers import (
    OmniVoiceLocalTTSProvider,
    OpenRouterTTSProvider,
    PolzaChatAudioProvider,
    PolzaTTSProvider,
    QwenLocalTTSProvider,
    TTSProvider,
)


class ProviderConfigurationError(ValueError):
    """A provider/model/mode argument combination the CLI reports as a usage error."""


class DialogueProviderBindingError(RuntimeError):
    """A dialogue run whose built provider cannot be cloned per cast voice."""


def build_tts_provider(
    args: argparse.Namespace, api_key: str, style_prompt: str | None, prompt_mode: str
) -> TTSProvider:
    """Construct the TTS provider for the validated provider/model/voice identity.

    Argument validation (including ``--mode`` and voice-bank admission) already
    happened in the CLI wrapper. The only errors raised here are
    ``ProviderConfigurationError`` for an unsupported Qwen runtime or a missing
    resolved voice-bank profile, and the exact ``RuntimeError`` for an unknown
    provider identifier.
    """
    if args.provider == "polza-chat-audio":
        return PolzaChatAudioProvider(
            api_key=api_key, model=args.model, voice=args.voice, fallback_voice=args.fallback_voice
        )
    if args.provider == "polza-tts":
        return PolzaTTSProvider(api_key=api_key, model=args.model, voice=args.voice)
    if args.provider == "openrouter-tts":
        return OpenRouterTTSProvider(
            api_key=api_key,
            model=args.model,
            voice=args.voice,
            style_prompt=style_prompt,
            prompt_mode=prompt_mode,
        )
    if args.provider == "qwen-local":
        instruct = getattr(args, "qwen_instruct", None)
        provider_kwargs = {
            "mode": args.mode,
            "voice": None if args.mode == "design" else args.voice,
            "instruct": QWEN_INSTRUCT if instruct is None else instruct,
            "sample_path": args.sample,
            "sample_text": getattr(args, "sample_text", None) or "",
        }
        runtime = os.environ.get("VOICEOVER_QWEN_TTS_RUNTIME", "python").strip()
        if runtime == "audio-cpp":
            from ..providers.audio_cpp_qwen_tts import AudioCppQwenTTSProvider

            return AudioCppQwenTTSProvider.from_environment(**provider_kwargs)
        if runtime != "python":
            raise ProviderConfigurationError(
                "VOICEOVER_QWEN_TTS_RUNTIME must be either 'python' or 'audio-cpp'."
            )
        return QwenLocalTTSProvider(**provider_kwargs)
    if args.provider == "omnivoice-local":
        omni_kwargs: dict[str, Any] = {}
        mode = getattr(args, "mode", "preset")
        if mode == "auto":
            omni_kwargs.update({"mode": "auto"})
        elif mode == "preset":
            catalog = getattr(args, "voice_bank_catalog", None)
            profile = getattr(args, "voice_bank_profile", None)
            if catalog is None:
                raise ProviderConfigurationError(
                    "omnivoice-local preset mode requires --voice-bank catalog.json"
                )
            if profile is None:
                if not is_dialogue_format(getattr(args, "format", "markdown")):
                    raise ProviderConfigurationError(
                        "omnivoice-local preset mode requires a resolved voice-bank profile"
                    )
                omni_kwargs.update({"mode": "preset"})
            else:
                reference_path = resolve_bank_profile(catalog, profile.id)[1]
                omni_kwargs.update(
                    {
                        "mode": "preset",
                        "voice_bank": (profile, reference_path),
                    }
                )
        elif mode == "clone":
            omni_kwargs.update(
                {
                    "mode": "clone",
                    "reference_audio_path": args.reference_audio,
                    "reference_text": args.reference_text,
                }
            )
        elif mode == "design":
            omni_kwargs.update(
                {
                    "mode": "design",
                    "design_instruction": args.design_instruction,
                }
            )
        return OmniVoiceLocalTTSProvider.from_environment(**omni_kwargs)
    raise RuntimeError(f"Unsupported provider: {args.provider}")


def bind_dialogue_voice_bank_providers(
    provider: Any,
    catalog: VoiceBankCatalog,
    speaker_voice_map: Mapping[str, str],
) -> dict[str, OmniVoiceLocalTTSProvider]:
    """Clone one OmniVoice provider per dialogue cast voice, in cast order.

    The built provider supplies its own admitted runtime/identity, and each cast
    voice is resolved to its admitted bank profile by ``resolve_bank_profile``
    before cloning, so every part keeps the voice identity its cast assigned.
    """
    if not isinstance(provider, OmniVoiceLocalTTSProvider):
        raise DialogueProviderBindingError(
            "omnivoice-local dialogue did not build an OmniVoice provider"
        )
    providers: dict[str, OmniVoiceLocalTTSProvider] = {}
    for voice_id in speaker_voice_map.values():
        profile, reference_path = resolve_bank_profile(catalog, voice_id)
        providers[voice_id] = provider.for_voice_bank_profile(profile, reference_path)
    return providers
