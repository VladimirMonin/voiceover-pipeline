import base64
import json

import requests

from voiceover_pipeline.config import POLZA_BASE_URL, POLZA_CHAT_NARRATION_SYSTEM_PROMPT
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider


class PolzaChatAudioProvider(TTSProvider):
    provider_id = "polza-chat-audio"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        fallback_voice: str | None = None,
        base_url: str = POLZA_BASE_URL,
        timeout_seconds: int = 240,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.fallback_voice = fallback_voice
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        last_error: Exception | None = None
        voices = [self.voice]
        if self.fallback_voice and self.fallback_voice not in voices:
            voices.append(self.fallback_voice)

        for voice in voices:
            try:
                return self._synthesize_with_voice(text=text, chunk_id=chunk_id, voice=voice)
            except Exception as error:
                last_error = error
                print(f"Voice '{voice}' failed for {chunk_id}: {error}")

        raise RuntimeError(f"All voices failed for {chunk_id}: {last_error}")

    def _synthesize_with_voice(self, text: str, chunk_id: str, voice: str) -> SynthesisResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": POLZA_CHAT_NARRATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Произнеси вслух только текст ниже, дословно, без вступления, "
                        "комментариев и добавлений:\n\n" + text
                    ),
                },
            ],
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": "pcm16"},
            "temperature": 0,
            "max_tokens": 4096,
            "stream": True,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            stream=True,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        audio_parts: list[str] = []
        transcript_parts: list[str] = []

        for raw_line in response.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            audio = chunk.get("choices", [{}])[0].get("delta", {}).get("audio", {})
            if audio.get("data"):
                audio_parts.append(audio["data"])
            if audio.get("transcript"):
                transcript_parts.append(audio["transcript"])

        if not audio_parts:
            raise RuntimeError("Polza stream finished without audio chunks.")

        return SynthesisResult(
            audio_bytes=base64.b64decode("".join(audio_parts)),
            audio_format="pcm16",
            transcript="".join(transcript_parts),
            generation_id=response.headers.get("X-Generation-Id"),
            client_path="requests",
            raw_metadata={"voice": voice, "provider": self.provider_id},
        )
