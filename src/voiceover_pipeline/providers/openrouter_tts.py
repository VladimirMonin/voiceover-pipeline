from collections.abc import Callable

import requests

from voiceover_pipeline.config import (
    DEFAULT_OPENROUTER_TTS_RESPONSE_FORMAT,
    GEMINI_38_TTS_MODELS,
    OPENROUTER_BASE_URL,
    OPENROUTER_TTS_MODELS,
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
# Only these two documented raw audio streams are admitted; anything else (a wrapped
# JSON error, prose, or a container the adapter cannot decode) fails closed instead
# of being guessed as raw PCM.
_AUDIO_CONTENT_TYPES = {
    "audio/mpeg": "mp3",
    "audio/pcm": "pcm16",
}
# The requested container is the only remaining evidence when the provider omits a
# content type. ``pcm`` is the application's canonical raw stream (24 kHz mono
# s16le), which ``media.write_audio_as_mp3`` demuxes with the same parameters.
_REQUESTED_AUDIO_FORMATS = {"mp3": "mp3", "wav": "wav", "pcm": "pcm16", "pcm16": "pcm16"}
# The containers this adapter can decode, so a container preserved in the paid evidence
# is accepted only when it is one this decoder really supports.
_DOCUMENTED_AUDIO_FORMATS = frozenset(_AUDIO_CONTENT_TYPES.values())

# A synchronous response sink persists the exact HTTP body (with its observed status,
# the untrusted ``X-Generation-Id`` header value, and the bounded container the response
# reported) *before* the status check and the fallible decode, so an accepted paid
# response is never lost to an unreadable payload and a local replay decodes it with the
# container it actually arrived in. A sink failure propagates and stops the decode. The
# container is already mapped to a bounded value, never a raw header.
RawResponseCallback = Callable[[bytes, int, str | None, str | None], None]


def audio_format_from_content_type(content_type: object) -> str | None:
    """The bounded raw container a reported response content type means, or ``None``.

    One mapping serves both the live decode and the value kept in the paid evidence, so
    the container a replay uses is exactly the one the live submit would have used. A
    missing header and an undocumented type both report no container; the undocumented
    type still fails closed at decode time.
    """
    base = str(content_type).split(";", 1)[0].strip().lower() if content_type else ""
    return _AUDIO_CONTENT_TYPES.get(base) if base else None


def decode_audio_speech_body(
    *,
    body: bytes,
    content_type: object = None,
    requested_format: str,
    observed_audio_format: str | None = None,
) -> tuple[bytes, str]:
    """Return the raw ``(audio_bytes, audio_format)`` of one ``/audio/speech`` body.

    The container precedence is: the bounded container the provider observed, else the
    reported response content type, else -- when the provider reported neither -- the
    requested ``response_format``, which is then the only remaining evidence, exactly as
    the Polza ``/audio/speech`` parser treats a missing ``contentType``. A local replay
    passes the container from the paid evidence; a live submit passes the observed
    header. A reported or requested type that is not a documented raw stream fails
    closed, and a body that looks like a JSON error or an event-stream payload is never
    treated as audio.
    """
    if not body:
        raise RuntimeError("OpenRouter TTS returned an empty audio body.")
    if observed_audio_format is not None:
        if observed_audio_format not in _DOCUMENTED_AUDIO_FORMATS:
            raise RuntimeError("OpenRouter TTS evidence carries an unsupported audio container.")
        return body, observed_audio_format
    base = str(content_type).split(";", 1)[0].strip().lower() if content_type else ""
    if base:
        audio_format = _AUDIO_CONTENT_TYPES.get(base)
        if audio_format is None:
            raise RuntimeError("OpenRouter TTS returned a non-audio response.")
        return body, audio_format
    if body.lstrip()[:16].lower().startswith((b"{", b"[", b"data:")):
        raise RuntimeError("OpenRouter TTS returned a non-audio response without a content type.")
    return body, _REQUESTED_AUDIO_FORMATS.get(requested_format, requested_format)


def build_audio_speech_result(
    *,
    body: bytes,
    requested_format: str,
    header_generation_id: str | None,
    transcript: str,
    model: str,
    voice: str,
    observed_audio_format: str | None = None,
) -> SynthesisResult:
    """Decode one stored raw ``/audio/speech`` body into a :class:`SynthesisResult`.

    A pure function of the already-received body plus the untrusted
    ``X-Generation-Id`` value, so the live submit and a local replay of a stored
    private body share exactly one decoder. ``observed_audio_format`` is the bounded
    container the paid evidence preserved for this body; a receipt written before that
    evidence existed passes ``None`` and falls back to the requested ``response_format``.
    """
    audio_bytes, audio_format = decode_audio_speech_body(
        body=body,
        requested_format=requested_format,
        observed_audio_format=observed_audio_format,
    )
    return SynthesisResult(
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        transcript=transcript,
        generation_id=header_generation_id,
        client_path="requests",
        raw_metadata={
            "voice": voice,
            "provider": OpenRouterTTSProvider.provider_id,
            "model": model,
        },
    )


class OpenRouterTTSProvider(TTSProvider):
    provider_id = "openrouter-tts"

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        style_prompt: str | None = None,
        prompt_mode: str = "auto",
        speaker_voice_map: dict[str, str] | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        response_format: str = DEFAULT_OPENROUTER_TTS_RESPONSE_FORMAT,
        timeout_seconds: int = 240,
        on_raw_response: RawResponseCallback | None = None,
    ) -> None:
        if model not in OPENROUTER_TTS_MODELS:
            raise ValueError(
                f"OpenRouter TTS model '{model}' is not in the current OpenRouter speech catalog. "
                f"Supported models: {OPENROUTER_TTS_MODELS}"
            )
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.style_prompt = None

        self._raw_prompt_mode = prompt_mode
        self.prompt_mode = resolve_prompt_mode(self.provider_id, model, "none")
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds
        self.on_raw_response = on_raw_response

    @property
    def _is_openai_model(self) -> bool:
        return self.model.startswith("openai/")

    @property
    def _uses_documented_gemini_speech_contract(self) -> bool:
        return self.model == _GEMINI_31_FLASH_TTS

    @property
    def _uses_instruction_field(self) -> bool:
        """Whether this model's request carries its direction in ``instructions``."""
        return self.model in GEMINI_38_TTS_MODELS

    def synthesize_chunk(
        self, text: str, chunk_id: str, voice: str | None = None, vibe: str | None = None
    ) -> SynthesisResult:
        """Synthesize one chunk, honoring a per-part cast voice and instruction.

        ``voice`` is the part's own cast voice (a ``speech-parts`` part or a dialogue
        turn) while an omitted one keeps the configured run voice, so each request
        carries exactly one scalar voice and never implies more than one speaker. A
        non-empty ``vibe`` is that part's effective direction on a Gemini 3.8 speech
        model, which carries it in the separate ``instructions`` field; on any other
        model it fails closed before the POST instead of composing the direction into
        the spoken ``input``.
        """
        if vibe and not self._uses_instruction_field:
            raise ValueError(
                "OpenRouter instructions are only carried by the Gemini 3.8 speech models."
            )
        return self._request_audio(
            text=text, style_prompt=None, voice=voice or self.voice, instructions=vibe
        )

    def _request_audio(
        self,
        text: str,
        style_prompt: str | None,
        voice: str | None = None,
        instructions: str | None = None,
    ) -> SynthesisResult:
        body = build_request_body(
            model=self.model,
            text=text,
            voice=voice or self.voice,
            response_format=self.response_format,
            style_prompt=style_prompt,
            prompt_mode=self.prompt_mode,
            instructions=instructions,
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
            # A redirect would let the transport re-send this paid POST body to
            # another target, which the one-submit invariant forbids. The observed
            # response (including any 3xx) is therefore reported back unchanged and
            # handled as a bounded status below.
            allow_redirects=False,
        )
        # The observed container is mapped before the response is handed on, so the
        # private paid evidence can keep it and a local replay decodes the body with
        # the container it actually arrived in instead of the requested one.
        observed = audio_format_from_content_type(response.headers.get("Content-Type"))
        if self.on_raw_response is not None:
            # Hand the exact response body (with its bounded observed container) to the
            # caller before the status check and the fallible decode, so a paid response
            # is persisted first. A sink failure propagates instead of decoding a
            # response that could not be stored.
            self.on_raw_response(
                response.content,
                response.status_code,
                response.headers.get("X-Generation-Id"),
                observed,
            )
        if response.status_code >= 300:
            # The provider error body is untrusted and may echo the request text or a
            # signed URL, so only the bounded status is reported.
            raise RuntimeError(f"OpenRouter TTS request failed with HTTP {response.status_code}.")

        audio_bytes, audio_format = decode_audio_speech_body(
            body=response.content,
            content_type=response.headers.get("Content-Type"),
            requested_format=self.response_format,
        )

        return SynthesisResult(
            audio_bytes=audio_bytes,
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
