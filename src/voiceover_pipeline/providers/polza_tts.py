import base64
import time

import requests

from voiceover_pipeline.config import (
    DEFAULT_POLZA_TTS_RESPONSE_FORMAT,
    POLZA_BASE_URL,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider

_MEDIA_POLL_INTERVAL = 5
_MEDIA_POLL_MAX = 60


class PolzaTTSProvider(TTSProvider):
    provider_id = "polza-tts"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        base_url: str = POLZA_BASE_URL,
        response_format: str = DEFAULT_POLZA_TTS_RESPONSE_FORMAT,
        timeout_seconds: int = 240,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds

    @property
    def _is_elevenlabs(self) -> bool:
        return self.model.startswith("elevenlabs/")

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        if self._is_elevenlabs:
            return self._synthesize_media(text, chunk_id)
        return self._synthesize_audio_speech(text, chunk_id)

    def _synthesize_audio_speech(self, text: str, chunk_id: str) -> SynthesisResult:
        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": self.response_format,
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        if not response.content:
            raise RuntimeError("Polza TTS returned an empty body.")

        resp_json = response.json()
        audio_b64 = resp_json.get("audio")
        if not audio_b64:
            raise RuntimeError(
                f"Polza TTS response missing 'audio' field. Keys: {list(resp_json.keys())}"
            )

        audio_bytes = base64.b64decode(audio_b64)
        content_type = resp_json.get("contentType", "audio/mpeg")
        usage = resp_json.get("usage")

        generation_id = (
            response.headers.get("X-Generation-Id")
            or resp_json.get("id")
            or resp_json.get("generation_id")
        )

        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format="mp3" if "mpeg" in content_type else "pcm16",
            transcript=text,
            generation_id=generation_id,
            client_path="requests",
            raw_metadata={
                "voice": self.voice,
                "provider": self.provider_id,
                "model": self.model,
                "content_type": content_type,
                "usage_direct": usage,
            },
        )

    def _synthesize_media(self, text: str, chunk_id: str) -> SynthesisResult:
        payload = {
            "model": self.model,
            "input": {
                "prompt": text,
                "voice": self.voice,
                "language_code": "ru",
            },
            "async": True,
        }

        submit = requests.post(
            f"{self.base_url}/media",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        if submit.status_code >= 400:
            raise RuntimeError(f"HTTP {submit.status_code}: {submit.text}")

        submit_json = submit.json()
        task_id = submit_json.get("id")
        if not task_id:
            raise RuntimeError(f"Polza Media response missing 'id'. Keys: {list(submit_json.keys())}")

        result = self._poll_media(task_id, chunk_id)
        audio_bytes, usage, generation_id = result
        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format="mp3",
            transcript=text,
            generation_id=generation_id or task_id,
            client_path="requests",
            raw_metadata={
                "voice": self.voice,
                "provider": self.provider_id,
                "model": self.model,
                "usage_direct": usage,
            },
        )

    def _poll_media(self, task_id: str, chunk_id: str) -> tuple[bytes, dict | None, str | None]:
        for attempt in range(_MEDIA_POLL_MAX):
            time.sleep(_MEDIA_POLL_INTERVAL)
            response = requests.get(
                f"{self.base_url}/media/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

            js = response.json()
            status = js.get("status") or js.get("state")

            if status == "completed":
                data = js.get("data")
                usage = js.get("usage")
                gen_id = js.get("id")
                return self._extract_media_audio(data, chunk_id), usage, gen_id

            if status == "failed":
                error = js.get("error")
                raise RuntimeError(f"Polza Media task {task_id} failed: {error}")

        raise RuntimeError(
            f"Polza Media task {task_id} still pending after {_MEDIA_POLL_MAX * _MEDIA_POLL_INTERVAL}s"
        )

    def _extract_media_audio(self, data, chunk_id: str) -> bytes:
        url = None
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            url = data[0].get("url")
        elif isinstance(data, dict):
            url = data.get("url") or data.get("audio")

        if not url:
            raise RuntimeError(
                f"Polza Media response missing audio URL. data={type(data).__name__}"
            )

        dl = requests.get(url, timeout=120)
        if dl.status_code >= 400:
            raise RuntimeError(f"Failed to download audio from {url}: HTTP {dl.status_code}")
        if not dl.content:
            raise RuntimeError(f"Downloaded empty audio from {url}")

        return dl.content
