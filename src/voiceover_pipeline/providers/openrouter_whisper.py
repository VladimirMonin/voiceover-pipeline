import base64
import shutil
import sys
from pathlib import Path
from typing import Any

import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    read_openrouter_key,
)
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider

# ── OpenRouter app attribution headers ──────────────────────────────────────
# See: https://openrouter.ai/docs/app-attribution
_APP_TITLE = "Voiceover Pipeline"
_APP_REFERER = "https://github.com/visper-io/voiceover-pipeline"


def _or_headers(api_key: str) -> dict[str, str]:
    """Build OpenRouter request headers with app attribution."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": _APP_TITLE,
        "HTTP-Referer": _APP_REFERER,
    }


_NO_TIMESTAMPS_WARNING = (
    "WARNING: openrouter-whisper does NOT return per-segment or word-level timestamps. "
    "The API returns a single 'text' field — one segment covering the entire audio. "
    "word_timestamps=True is ignored. For real timestamps use faster-whisper (local) "
    "or groq-whisper (cloud with GROQ_API_KEY)."
)


def _detect_audio_duration_ms(audio_path: Path) -> int:
    """Get audio duration in ms using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        import subprocess

        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return 0


_AUDIO_FORMAT_MAP = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".flac": "flac",
    ".m4a": "m4a",
    ".webm": "webm",
    ".aac": "aac",
}


class OpenRouterWhisperProvider(TranscriptionProvider):
    provider_id = "openrouter-whisper"

    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or read_openrouter_key()
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "openai/whisper-large-v3-turbo",
                "description": "Optimized Whisper Large V3 — fast, 99+ languages",
                "speed": "fast",
            },
            {
                "id": "openai/whisper-large-v3",
                "description": "Whisper Large V3 — highest accuracy, expensive",
                "speed": "balanced",
            },
            {
                "id": "openai/whisper-1",
                "description": "Whisper v1 — legacy, cheapest",
                "speed": "fastest",
            },
        ]

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
    ) -> TimingResult:
        if word_timestamps and not quiet:
            print(_NO_TIMESTAMPS_WARNING, file=sys.stderr)

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ext = audio_path.suffix.lower()
        audio_format = _AUDIO_FORMAT_MAP.get(ext, "mp3")

        # Read and base64-encode
        with open(audio_path, "rb") as fh:
            audio_bytes = fh.read()

        if not quiet:
            print(
                f"Encoding {len(audio_bytes) / 1024 / 1024:.1f} MB audio to base64...",
                file=sys.stderr,
            )

        b64_data = base64.b64encode(audio_bytes).decode("utf-8")

        body: dict[str, Any] = {
            "model": self.model,
            "input_audio": {
                "data": b64_data,
                "format": audio_format,
            },
        }
        if language:
            body["language"] = language

        headers = _or_headers(self.api_key)

        if not quiet:
            print(
                f"Sending {len(b64_data) / 1024 / 1024:.1f} MB base64 to OpenRouter...",
                file=sys.stderr,
            )

        resp = requests.post(
            f"{self.base_url}/audio/transcriptions",
            headers=headers,
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        result = resp.json()

        text = result.get("text", "")
        duration_ms = _detect_audio_duration_ms(audio_path)

        # The basic API returns full text without segments.
        # We create a single segment covering the entire audio.
        segments = [
            TimingSegment(
                id=0,
                start_sec=0.0,
                end_sec=round(duration_ms / 1000, 3),
                start_ms=0,
                end_ms=duration_ms,
                duration_ms=duration_ms,
                text=text.strip(),
                words=None,
            )
        ]

        return TimingResult(
            segments=segments,
            model=self.model,
            backend="whisper",
            provider=self.provider_id,
            language=language,
            source_audio=str(audio_path.resolve()),
        )
