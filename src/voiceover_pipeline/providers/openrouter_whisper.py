from pathlib import Path
from typing import Any

import requests

from voiceover_pipeline.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_WHISPER_MODELS,
    read_openrouter_key,
)
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider


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
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as fh:
            files = {
                "file": (audio_path.name, fh, self._mime_type(audio_path)),
            }
            data: dict[str, str] = {
                "model": self.model,
                "language": language,
                "response_format": "verbose_json",
            }
            if word_timestamps:
                data["timestamp_granularities[]"] = "word"

            headers = {"Authorization": f"Bearer {self.api_key}"}

            resp = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()

        segments: list[TimingSegment] = []
        for idx, seg in enumerate(result.get("segments", [])):
            start_ms = round(seg["start"] * 1000)
            end_ms = round(seg["end"] * 1000)
            words_list = None
            if word_timestamps and seg.get("words"):
                words_list = [
                    {
                        "word": w["word"].strip(),
                        "start_ms": round(w["start"] * 1000),
                        "end_ms": round(w["end"] * 1000),
                    }
                    for w in seg["words"]
                ]
            segments.append(
                TimingSegment(
                    id=idx,
                    start_sec=round(seg["start"], 3),
                    end_sec=round(seg["end"], 3),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    text=seg["text"].strip(),
                    words=words_list,
                )
            )

        return TimingResult(
            segments=segments,
            model=self.model,
            backend="whisper",
            provider=self.provider_id,
            language=language,
            source_audio=str(audio_path.resolve()),
        )

    @staticmethod
    def _mime_type(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".webm": "audio/webm",
        }.get(ext, "audio/mpeg")
