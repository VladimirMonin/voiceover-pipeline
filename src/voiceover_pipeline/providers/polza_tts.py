import base64
import binascii
import json
import re
import time
from collections.abc import Callable
from decimal import Decimal

import requests

from voiceover_pipeline.config import (
    DEFAULT_POLZA_TTS_RESPONSE_FORMAT,
    POLZA_BASE_URL,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider

POLZA_TTS_PROVIDER_ID = "polza-tts"

_MEDIA_POLL_INTERVAL = 5
_MEDIA_POLL_MAX = 60

# A validated task id is spliced into the GET path ``/media/<id>`` and a
# completed generation id is reported to callers, so only opaque token
# characters are accepted. Anything that could change the target endpoint
# (slash, query, scheme, dot traversal, whitespace) fails closed before any
# request, and a non-token id is never echoed into an error or report.
_MEDIA_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")

# The OpenAI-style ``/audio/speech`` route returns base64 audio in JSON and names
# its container through ``contentType``. Only documented audio MIME types map to
# a raw format; anything else fails closed instead of being guessed as raw PCM.
_POLZA_AUDIO_CONTENT_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/pcm": "pcm16",
}
# ``audio/l16`` is intentionally absent: L16 is network byte order (big-endian)
# while ``media.write_audio_as_mp3`` decodes ``pcm16`` as little-endian s16le, so
# mapping it would byte-swap every sample. It stays unsupported until an explicit
# correct endian/rate implementation exists.
# Format used when the provider omits a content type: the one the caller asked
# for in ``response_format`` is the only remaining evidence.
_REQUESTED_AUDIO_FORMATS = {"mp3": "mp3", "wav": "wav", "pcm": "pcm16", "pcm16": "pcm16"}


def _has_wav_header(audio_bytes: bytes) -> bool:
    return len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _audio_format_from_response(
    content_type: object, audio_bytes: bytes, requested_format: str
) -> str:
    """Map the response container to a raw format, failing closed when unknown.

    An unsupported reported MIME type fails closed, and a reported WAV without a
    valid RIFF/WAVE header is rejected rather than stored. A real RIFF/WAVE
    container then wins over a missing or incorrect declaration: the receipt and
    the conversion step must see the actual format, so real WAV bytes are never
    recorded as raw MP3/PCM.
    """
    base = str(content_type).split(";", 1)[0].strip().lower() if content_type else ""
    if not base:
        audio_format = _REQUESTED_AUDIO_FORMATS.get(requested_format)
        if audio_format is None:
            raise RuntimeError("Polza TTS returned audio without a content type.")
    else:
        audio_format = _POLZA_AUDIO_CONTENT_TYPES.get(base)
        if audio_format is None:
            raise RuntimeError("Polza TTS returned an unsupported audio content type.")
        if audio_format == "wav" and not _has_wav_header(audio_bytes):
            raise RuntimeError("Polza TTS reported WAV audio without a valid WAV header.")
    if _has_wav_header(audio_bytes):
        return "wav"
    return audio_format


def _safe_media_id(value: object) -> str | None:
    """Return a bounded opaque id, or None when the value is not one."""
    if isinstance(value, str) and _MEDIA_SAFE_ID_PATTERN.fullmatch(value) is not None:
        return value
    return None


def build_audio_speech_result(
    *,
    body: bytes,
    requested_format: str,
    header_generation_id: str | None,
    transcript: str,
    model: str,
    voice: str,
) -> SynthesisResult:
    """Parse one ``/audio/speech`` JSON body into a :class:`SynthesisResult`.

    A pure function of the already-received body (plus the untrusted
    ``X-Generation-Id`` header value), so the live submit and a local replay of a
    stored private body share exactly one parser. ``parse_float=Decimal`` keeps an
    unquoted usage cost exact. An empty body, a non-object payload, a missing
    ``audio`` field, an unsupported content type, or a WAV declaration without a
    RIFF header fails closed rather than guessing an audio container.
    """
    if not body:
        raise RuntimeError("Polza TTS returned an empty body.")
    try:
        resp_json = json.loads(body, parse_float=Decimal)
    except (UnicodeError, ValueError, RecursionError):
        raise RuntimeError("Polza TTS response is not valid JSON.") from None
    if not isinstance(resp_json, dict):
        raise RuntimeError("Polza TTS response is not a JSON object.")
    audio_b64 = resp_json.get("audio")
    if not isinstance(audio_b64, str) or not audio_b64:
        raise RuntimeError("Polza TTS response is missing a valid audio field.")
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError("Polza TTS response has invalid base64 audio.") from None
    if not audio_bytes:
        raise RuntimeError("Polza TTS response has empty decoded audio.")
    content_type = resp_json.get("contentType")
    audio_format = _audio_format_from_response(content_type, audio_bytes, requested_format)
    usage = resp_json.get("usage")

    generation_id = (
        _safe_media_id(header_generation_id)
        or _safe_media_id(resp_json.get("id"))
        or _safe_media_id(resp_json.get("generation_id"))
    )

    return SynthesisResult(
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        transcript=transcript,
        generation_id=generation_id,
        client_path="requests",
        raw_metadata={
            "voice": voice,
            "provider": POLZA_TTS_PROVIDER_ID,
            "model": model,
            "content_type": content_type,
            "usage_direct": usage,
        },
    )


MediaTaskAcceptedCallback = Callable[[str], None]
MediaCompletedCallback = Callable[[str, dict | None, str | None], None]
# A synchronous response sink persists the exact HTTP body (with its observed
# status and the untrusted ``X-Generation-Id`` header) *before* the status check
# and the fallible parse, so an accepted paid response is never lost to a
# malformed payload. A sink failure propagates and stops the parse.
RawResponseCallback = Callable[[bytes, int, str | None], None]


class PolzaTTSProvider(TTSProvider):
    provider_id = POLZA_TTS_PROVIDER_ID

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        base_url: str = POLZA_BASE_URL,
        response_format: str = DEFAULT_POLZA_TTS_RESPONSE_FORMAT,
        timeout_seconds: int = 240,
        on_media_task_accepted: MediaTaskAcceptedCallback | None = None,
        on_media_completed: MediaCompletedCallback | None = None,
        on_raw_response: RawResponseCallback | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds
        self.on_media_task_accepted = on_media_task_accepted
        self.on_media_completed = on_media_completed
        self.on_raw_response = on_raw_response

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
        if self.on_raw_response is not None:
            # Hand the exact response body to the caller before the status check
            # and the fallible JSON parse, so a paid response is persisted first.
            # A sink failure propagates instead of parsing a response that could
            # not be stored.
            self.on_raw_response(
                response.content,
                response.status_code,
                response.headers.get("X-Generation-Id"),
            )
        if response.status_code >= 400:
            # The provider error body is untrusted and may echo the request text
            # or a signed URL, so only the bounded status is reported.
            raise RuntimeError(f"HTTP {response.status_code}")

        return build_audio_speech_result(
            body=response.content,
            requested_format=self.response_format,
            header_generation_id=response.headers.get("X-Generation-Id"),
            transcript=text,
            model=self.model,
            voice=self.voice,
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
            # The raw error body can echo the request or a signed URL, so only
            # the bounded HTTP status is reported.
            raise RuntimeError(f"HTTP {submit.status_code}")

        submit_json = submit.json()
        task_id = submit_json.get("id")
        if not task_id:
            raise RuntimeError(
                f"Polza Media response missing 'id'. Keys: {list(submit_json.keys())}"
            )

        task_id = self._validate_media_task_id(task_id)
        if self.on_media_task_accepted is not None:
            # The accepted id is paid for, so it is persisted before any poll or
            # download. A persistence failure must propagate instead of
            # continuing into requests that could send a second paid POST.
            self.on_media_task_accepted(task_id)

        return self._complete_media_task(task_id, text, chunk_id)

    def recover_media_task(self, task_id: str, text: str, chunk_id: str) -> SynthesisResult:
        """Finish an already accepted media task with GET calls only.

        The id is validated as a safe opaque token and then polled and
        downloaded through GET requests; this never submits a new task, so a
        known paid id cannot be billed twice.
        """
        validated_id = self._validate_media_task_id(task_id)
        return self._complete_media_task(validated_id, text, chunk_id)

    def _complete_media_task(self, task_id: str, text: str, chunk_id: str) -> SynthesisResult:
        data, usage, generation_id = self._poll_media(task_id)
        # The completed payload's id is untrusted; drop anything that is not a
        # bounded opaque token so a URL from the provider cannot be reported as
        # the generation id. The known safe task id stays the fallback below.
        generation_id = _safe_media_id(generation_id)
        if self.on_media_completed is not None:
            # A completed task already carries its exact usage cost. Report it
            # before the signed-URL download, which can fail on its own.
            self.on_media_completed(task_id, usage, generation_id)
        audio_bytes = self._extract_media_audio(data, chunk_id)
        return self._build_media_result(
            text=text,
            task_id=task_id,
            audio_bytes=audio_bytes,
            usage=usage,
            generation_id=generation_id,
        )

    @staticmethod
    def _validate_media_task_id(task_id: object) -> str:
        validated = _safe_media_id(task_id)
        if validated is None:
            # A fixed message keeps an untrusted value (for example a signed URL
            # from provider output or tampered state) out of the error report.
            raise ValueError("Polza Media task id is not a safe opaque token.")
        return validated

    def _build_media_result(
        self,
        *,
        text: str,
        task_id: str,
        audio_bytes: bytes,
        usage: dict | None,
        generation_id: str | None,
    ) -> SynthesisResult:
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

    def _poll_media(self, task_id: str) -> tuple[object, dict | None, str | None]:
        for attempt in range(_MEDIA_POLL_MAX):
            time.sleep(_MEDIA_POLL_INTERVAL)
            response = requests.get(
                f"{self.base_url}/media/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}")

            # The completed payload carries usage cost; keep unquoted numbers
            # exact. The submit response above stays plain JSON.
            js = response.json(parse_float=Decimal)
            status = js.get("status") or js.get("state")

            if status == "completed":
                return js.get("data"), js.get("usage"), js.get("id")

            if status == "failed":
                # The provider error payload is untrusted and may carry a signed
                # URL or request text, so it is not echoed; the validated task
                # id is a bounded opaque token and stays for correlation.
                raise RuntimeError(f"Polza Media task {task_id} failed.")

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

        try:
            dl = requests.get(url, timeout=120)
        except requests.Timeout:
            # The signed download URL can appear inside the requests exception
            # text. Re-raise the same retryable class from a constant message so
            # the URL never reaches a log or report.
            raise requests.Timeout("Audio download timed out.") from None
        except requests.ConnectionError:
            raise requests.ConnectionError("Audio download connection failed.") from None
        except requests.RequestException:
            raise requests.RequestException("Audio download failed.") from None

        if dl.status_code >= 400:
            raise RuntimeError(f"Failed to download audio: HTTP {dl.status_code}")
        if not dl.content:
            raise RuntimeError("Downloaded empty audio.")

        return dl.content
