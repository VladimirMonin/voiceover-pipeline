from abc import ABC, abstractmethod

from voiceover_pipeline.models import SynthesisResult


class TTSProvider(ABC):
    provider_id: str

    @abstractmethod
    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        raise NotImplementedError
