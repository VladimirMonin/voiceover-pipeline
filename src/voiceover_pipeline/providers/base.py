from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from voiceover_pipeline.models import ASRRequest, ASRResult, SynthesisResult, TimingResult


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


class ASRProvider(ABC):
    """Finite-audio ASR protocol, kept separate from timing providers."""

    provider_id: str

    @abstractmethod
    def transcribe(self, request: ASRRequest) -> ASRResult:
        raise NotImplementedError

    def list_models(self) -> list[dict[str, Any]]:
        raise NotImplementedError


def validate_asr_response(request: ASRRequest, result: ASRResult) -> ASRResult:
    """Apply caller-intent validation without adding runtime details to results."""
    result.validate_for_request(request)
    return result
