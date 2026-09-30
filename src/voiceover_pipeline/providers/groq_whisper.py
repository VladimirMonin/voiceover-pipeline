"""Groq Whisper speech-to-text provider.

Direct Groq API access — supports timestamp_granularities for word- and
segment-level timestamps (unlike OpenRouter which drops everything but text).

API reference: https://console.groq.com/docs/speech-to-text
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from voiceover_pipeline.config import (
    GROQ_BASE_URL,
    read_groq_key,
)
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider

_AUDIO_FORMAT_MAP = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".flac": "flac",
    ".m4a": "m4a",
    ".webm": "webm",
    ".aac": "aac",
    ".mp4": "mp4",
    ".mpeg": "mpeg",
    ".mpga": "mpga",
}


def parse_groq_timing_result(
    result: dict[str, Any],
    *,
    model: str,
    language: str,
    word_timestamps: bool,
    audio_path: Path,
) -> TimingResult:
    """Convert one Groq ``verbose_json`` body into a :class:`TimingResult`.

    This is a pure function of the already-received provider payload, so a stored
    raw response can be replayed locally with no second POST. It preserves the
    provider's own segment spans and records an honest ``timestamp_basis``:
    ``provider_segment_timestamps`` when the provider returned segments, and
    ``fallback_full_text`` for the single-span fallback.
    """
    full_text = result.get("text", "").strip()
    language_detected = result.get("language", language or "")

    segments: list[TimingSegment] = []
    raw_segments = result.get("segments", [])

    if raw_segments:
        for seg in raw_segments:
            seg_id = seg.get("id", 0)
            start_sec = seg.get("start", 0.0)
            end_sec = seg.get("end", 0.0)
            seg_text = seg.get("text", "").strip()
            start_ms = round(start_sec * 1000)
            end_ms = round(end_sec * 1000)
            duration_ms = end_ms - start_ms

            words_list = None
            raw_words = seg.get("words", [])
            if word_timestamps and raw_words:
                words_list = [
                    {
                        "word": w.get("word", "").strip(),
                        "start_ms": round(w.get("start", 0) * 1000),
                        "end_ms": round(w.get("end", 0) * 1000),
                    }
                    for w in raw_words
                ]

            segments.append(
                TimingSegment(
                    id=seg_id,
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    text=seg_text,
                    words=words_list,
                )
            )
    else:
        # Fallback: single segment from full text (shouldn't happen often)
        duration_ms = round((result.get("duration", 0) or 0) * 1000)
        segments = [
            TimingSegment(
                id=0,
                start_sec=0.0,
                end_sec=round(duration_ms / 1000, 3),
                start_ms=0,
                end_ms=duration_ms,
                duration_ms=duration_ms,
                text=full_text,
                words=None,
            )
        ]

    return TimingResult(
        segments=segments,
        model=model,
        backend="groq-whisper",
        provider="groq-whisper",
        language=language_detected,
        source_audio=str(audio_path.resolve()),
        timestamp_basis=("provider_segment_timestamps" if raw_segments else "fallback_full_text"),
    )


class GroqWhisperProvider(TranscriptionProvider):
    """Transcription via Groq API — full segments + optional word timestamps."""

    provider_id = "groq-whisper"

    def __init__(
        self,
        model: str = "whisper-large-v3-turbo",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or read_groq_key()
        self.base_url = GROQ_BASE_URL.rstrip("/")

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "whisper-large-v3-turbo",
                "description": "Optimized Whisper Large V3 Turbo — fast, 99+ languages (default)",
                "speed": "fast",
            },
            {
                "id": "whisper-large-v3",
                "description": "Whisper Large V3 — highest accuracy",
                "speed": "balanced",
            },
        ]

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
        on_raw_response: Callable[[bytes, str], None] | None = None,
    ) -> TimingResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ext = audio_path.suffix.lower()
        audio_format = _AUDIO_FORMAT_MAP.get(ext)
        if audio_format is None:
            raise ValueError(
                f"Unsupported audio format: {ext}. "
                f"Supported: {', '.join(sorted(set(_AUDIO_FORMAT_MAP.values())))}"
            )

        # Build multipart form data
        file_size_mb = audio_path.stat().st_size / 1024 / 1024

        timestamp_granularities: list[str] = ["segment"]
        if word_timestamps:
            timestamp_granularities.append("word")

        data: dict[str, Any] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": timestamp_granularities,
        }
        if language:
            data["language"] = language

        if not quiet:
            print(
                f"Uploading {file_size_mb:.1f} MB to Groq ({self.model}, "
                f"timestamps: {timestamp_granularities})...",
                file=sys.stderr,
            )

        start = time.monotonic()
        with open(audio_path, "rb") as fh:
            # Build multipart form: file + scalar fields + repeated timestamp_granularities[]
            multipart_data: list[tuple[str, tuple]] = [
                ("model", (None, self.model)),
                ("response_format", (None, "verbose_json")),
            ]
            if language:
                multipart_data.append(("language", (None, language)))
            for tg in timestamp_granularities:
                multipart_data.append(("timestamp_granularities[]", (None, tg)))
            multipart_files = {"file": (audio_path.name, fh, f"audio/{audio_format}")}

            resp = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=multipart_data,
                files=multipart_files,
                timeout=300,
            )
        elapsed = time.monotonic() - start

        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise RuntimeError(f"Groq API error {resp.status_code}: {detail}")

        if on_raw_response is not None:
            # Hand the exact successful response body to the caller before the
            # fallible JSON parse, so a paid request can persist its private raw
            # evidence first. A callback failure propagates and stops the parse.
            on_raw_response(resp.content, resp.headers.get("Content-Type", "") or "")

        result = resp.json()

        if not quiet:
            print(
                f"  Done in {elapsed:.1f}s",
                file=sys.stderr,
            )

        return parse_groq_timing_result(
            result,
            model=self.model,
            language=language,
            word_timestamps=word_timestamps,
            audio_path=audio_path,
        )
