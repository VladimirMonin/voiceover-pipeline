import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    PODCAST_NARRATION_FALLBACK_PROMPT,
    PODCAST_NARRATION_PROMPT,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider
from voiceover_pipeline.tts_prompting import (
    build_request_body,
    resolve_prompt_mode,
)


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
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.style_prompt = style_prompt
        self.speaker_voice_map = speaker_voice_map or {}
        self.fallback_style_prompt = PODCAST_NARRATION_FALLBACK_PROMPT
        self._raw_prompt_mode = prompt_mode
        self.prompt_mode = resolve_prompt_mode(self.provider_id, model, prompt_mode)
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds

    @property
    def _is_openai_model(self) -> bool:
        return self.model.startswith("openai/")

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        if self._is_openai_model:
            return self._request_audio(text=text, style_prompt=None)

        try:
            return self._request_audio(text=text, style_prompt=self.style_prompt)
        except RuntimeError as error:
            if "No successful provider responses" not in str(error):
                raise
            print(f"Style prompt failed for {chunk_id}; retrying with shorter podcast style prompt.")
            return self._request_audio(text=text, style_prompt=self.fallback_style_prompt)

    def _request_audio(self, text: str, style_prompt: str | None) -> SynthesisResult:
        body = build_request_body(
            model=self.model,
            text=text,
            voice=self.voice,
            response_format=self.response_format,
            style_prompt=style_prompt,
            prompt_mode=self.prompt_mode,
        )
        if self.speaker_voice_map:
            body["multi_speaker_voice_config"] = {
                "speaker_voice_configs": [
                    {
                        "speaker": speaker,
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": voice}
                        },
                    }
                    for speaker, voice in self.speaker_voice_map.items()
                ]
            }
        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
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
                "prompt_mode": self.prompt_mode,
                "speaker_voice_map": self.speaker_voice_map,
            },
        )
