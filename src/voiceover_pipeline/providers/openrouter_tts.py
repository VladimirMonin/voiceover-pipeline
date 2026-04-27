import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    PODCAST_NARRATION_FALLBACK_PROMPT,
    PODCAST_NARRATION_PROMPT,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider


class OpenRouterTTSProvider(TTSProvider):
    provider_id = "openrouter-tts"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        style_prompt: str = PODCAST_NARRATION_PROMPT,
        base_url: str = OPENROUTER_BASE_URL,
        response_format: str = "pcm",
        timeout_seconds: int = 240,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.style_prompt = style_prompt
        self.fallback_style_prompt = PODCAST_NARRATION_FALLBACK_PROMPT
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        try:
            return self._request_audio(text=text, style_prompt=self.style_prompt)
        except RuntimeError as error:
            if "No successful provider responses" not in str(error):
                raise
            print(f"Style prompt failed for {chunk_id}; retrying with shorter podcast style prompt.")
            return self._request_audio(text=text, style_prompt=self.fallback_style_prompt)

    def _request_audio(self, text: str, style_prompt: str) -> SynthesisResult:
        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": self._build_input(text, style_prompt=style_prompt),
                "voice": self.voice,
                "response_format": self.response_format,
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        if not response.content:
            raise RuntimeError("OpenRouter TTS returned an empty audio body.")

        return SynthesisResult(
            audio_bytes=response.content,
            audio_format="pcm16" if self.response_format == "pcm" else self.response_format,
            transcript=text,
            generation_id=response.headers.get("X-Generation-Id"),
            client_path="requests",
            raw_metadata={
                "voice": self.voice,
                "provider": self.provider_id,
                "style_prompt": style_prompt,
            },
        )

    def _build_input(self, text: str, style_prompt: str) -> str:
        return f"{style_prompt}\n\n{text}"
