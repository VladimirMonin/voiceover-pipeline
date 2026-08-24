import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_TTS_MODELS,
    PODCAST_NARRATION_PROMPT,
    TTS_PROMPT_MODE_NATIVE,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider
from voiceover_pipeline.tts_prompting import (
    build_request_body,
    resolve_prompt_mode,
)

# ── OpenRouter app attribution headers ──────────────────────────────────────
_APP_TITLE = "Voiceover Pipeline"
_APP_REFERER = "https://github.com/visper-io/voiceover-pipeline"
_GEMINI_31_FLASH_TTS = "google/gemini-3.1-flash-tts-preview"
_AUDIO_CONTENT_TYPES = {
    "audio/mpeg": "mp3",
    "audio/pcm": "pcm16",
}


def _response_audio_format(response: requests.Response, fallback: str) -> str:
    content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].lower()
    if content_type:
        audio_format = _AUDIO_CONTENT_TYPES.get(content_type)
        if audio_format is None:
            raise RuntimeError("OpenRouter TTS returned a non-audio response.")
        return audio_format

    prefix = response.content.lstrip()[:16].lower()
    if prefix.startswith((b"{", b"[", b"data:")):
        raise RuntimeError("OpenRouter TTS returned a non-audio response without a content type.")
    return fallback


class OpenRouterTTSProvider(TTSProvider):
    provider_id = "openrouter-tts"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        style_prompt: str | None = PODCAST_NARRATION_PROMPT,
        prompt_mode: str = "auto",
        speaker_voice_map: dict[str, str] | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        response_format: str = "pcm",
        timeout_seconds: int = 240,
    ) -> None:
        if model not in OPENROUTER_TTS_MODELS:
            raise ValueError(
                f"OpenRouter TTS model '{model}' is not in the current OpenRouter speech catalog. "
                f"Supported models: {OPENROUTER_TTS_MODELS}"
            )
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.style_prompt = style_prompt

        self._raw_prompt_mode = prompt_mode
        self.prompt_mode = resolve_prompt_mode(self.provider_id, model, prompt_mode)
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds

    @property
    def _is_openai_model(self) -> bool:
        return self.model.startswith("openai/")

    @property
    def _uses_documented_gemini_speech_contract(self) -> bool:
        return self.model == _GEMINI_31_FLASH_TTS

    def synthesize_chunk(
        self, text: str, chunk_id: str, voice: str | None = None
    ) -> SynthesisResult:
        active_voice = voice or self.voice
        if self._is_openai_model:
            return self._request_audio(text=text, style_prompt=None, voice=active_voice)
        if (
            self._uses_documented_gemini_speech_contract
            and self.prompt_mode == TTS_PROMPT_MODE_NATIVE
        ):
            raise ValueError(
                "OpenRouter Gemini 3.1 Flash TTS does not document a separate prompt field; "
                "use prompt_mode=auto, prefix, or none."
            )

        return self._request_audio(text=text, style_prompt=self.style_prompt, voice=active_voice)

    def _request_audio(
        self, text: str, style_prompt: str | None, voice: str | None = None
    ) -> SynthesisResult:
        body = build_request_body(
            model=self.model,
            text=text,
            voice=voice or self.voice,
            response_format=self.response_format,
            style_prompt=style_prompt,
            prompt_mode=self.prompt_mode,
        )
        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": _APP_TITLE,
                "HTTP-Referer": _APP_REFERER,
            },
            json=body,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter TTS request failed with HTTP {response.status_code}.")

        if not response.content:
            raise RuntimeError("OpenRouter TTS returned an empty audio body.")

        fallback_format = "pcm16" if self.response_format == "pcm" else self.response_format
        audio_format = _response_audio_format(response, fallback_format)

        return SynthesisResult(
            audio_bytes=response.content,
            audio_format=audio_format,
            transcript=text,
            generation_id=response.headers.get("X-Generation-Id"),
            client_path="requests",
            raw_metadata={
                "voice": voice or self.voice,
                "provider": self.provider_id,
                "style_prompt": style_prompt,
                "prompt_mode": self.prompt_mode,
            },
        )
