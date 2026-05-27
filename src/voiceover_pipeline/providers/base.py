from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from voiceover_pipeline.models import SynthesisResult, TimingResult


class TTSProvider(ABC):
    provider_id: str

    @abstractmethod
    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        raise NotImplementedError


class TranscriptionProvider(ABC):
    provider_id: str

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
    ) -> TimingResult:
        raise NotImplementedError

    def list_models(self) -> list[dict[str, Any]]:
        raise NotImplementedError
