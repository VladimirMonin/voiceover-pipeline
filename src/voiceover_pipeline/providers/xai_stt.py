"""xAI Speech-to-Text provider.

xAI STT API — batch file upload with word-level timestamps.
Endpoint: POST https://api.x.ai/v1/stt

API reference: https://docs.x.ai/developers/model-capabilities/audio/speech-to-text
"""

import sys
import time
from pathlib import Path
from typing import Any

import requests

from voiceover_pipeline.config import (
    XAI_BASE_URL,
    read_xai_key,
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


class XAISttProvider(TranscriptionProvider):
    """Transcription via xAI API — word-level timestamps with confidence."""

    provider_id = "xai-stt"

    # xAI has one STT model, not selectable — but we expose it for consistency
    DEFAULT_MODEL = "grok-stt"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or read_xai_key()
        self.base_url = XAI_BASE_URL.rstrip("/")

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": self.DEFAULT_MODEL,
                "description": "Grok STT — word-level timestamps, multichannel, diarization, 12 audio formats",
                "speed": "fast",
            },
        ]

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
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

        file_size_mb = audio_path.stat().st_size / 1024 / 1024

        # Build multipart form data
        multipart_data: list[tuple[str, tuple]] = []
        if language:
            multipart_data.append(("language", (None, language)))
        # format=true — requests structured JSON with word timestamps
        multipart_data.append(("format", (None, "true")))

        if not quiet:
            print(
                f"Uploading {file_size_mb:.1f} MB to xAI STT (language={language or 'auto'})...",
                file=sys.stderr,
            )

        start = time.monotonic()
        with open(audio_path, "rb") as fh:
            multipart_files = {"file": (audio_path.name, fh, f"audio/{audio_format}")}
            resp = requests.post(
                f"{self.base_url}/stt",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=multipart_data,
                files=multipart_files,
                timeout=300,
            )
        elapsed = time.monotonic() - start

        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise RuntimeError(f"xAI STT API error {resp.status_code}: {detail}")

        result = resp.json()

        if not quiet:
            duration = result.get("duration", 0)
            print(
                f"  Done in {elapsed:.1f}s (audio: {duration}s)",
                file=sys.stderr,
            )

        full_text = result.get("text", "").strip()
        language_detected = result.get("language", language or "")

        # Parse word-level data into segments
        raw_words = result.get("words", [])
        segments: list[TimingSegment] = []

        if raw_words:
            # Group words into pseudo-segments based on pauses
            # Start a new segment when gap > 0.5s
            current_words: list[dict[str, Any]] = []
            segment_start = raw_words[0].get("start", 0.0)
            last_end = segment_start
            seg_id = 0

            for w in raw_words:
                w_start = w.get("start", 0.0)
                # New segment if gap > 0.5s
                if current_words and (w_start - last_end) > 0.5:
                    seg_text = " ".join(cw.get("text", "").strip() for cw in current_words)
                    seg_end = current_words[-1].get("end", last_end)
                    start_ms = round(segment_start * 1000)
                    end_ms = round(seg_end * 1000)
                    segments.append(
                        TimingSegment(
                            id=seg_id,
                            start_sec=round(segment_start, 3),
                            end_sec=round(seg_end, 3),
                            start_ms=start_ms,
                            end_ms=end_ms,
                            duration_ms=end_ms - start_ms,
                            text=seg_text,
                            words=[
                                {
                                    "word": cw.get("text", "").strip(),
                                    "start_ms": round(cw.get("start", 0) * 1000),
                                    "end_ms": round(cw.get("end", 0) * 1000),
                                    "confidence": cw.get("confidence"),
                                }
                                for cw in current_words
                            ]
                            if word_timestamps
                            else None,
                        )
                    )
                    seg_id += 1
                    current_words = []
                    segment_start = w_start

                current_words.append(w)
                last_end = w.get("end", w_start)

            # Flush last segment
            if current_words:
                seg_text = " ".join(cw.get("text", "").strip() for cw in current_words)
                seg_end = current_words[-1].get("end", last_end)
                start_ms = round(segment_start * 1000)
                end_ms = round(seg_end * 1000)
                segments.append(
                    TimingSegment(
                        id=seg_id,
                        start_sec=round(segment_start, 3),
                        end_sec=round(seg_end, 3),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        duration_ms=end_ms - start_ms,
                        text=seg_text,
                        words=[
                            {
                                "word": cw.get("text", "").strip(),
                                "start_ms": round(cw.get("start", 0) * 1000),
                                "end_ms": round(cw.get("end", 0) * 1000),
                                "confidence": cw.get("confidence"),
                            }
                            for cw in current_words
                        ]
                        if word_timestamps
                        else None,
                    )
                )

        # Fallback: single segment from full text
        if not segments:
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
            model=self.model,
            backend="grok-stt",
            provider=self.provider_id,
            language=language_detected or language or "",
            source_audio=str(audio_path.resolve()),
        )
