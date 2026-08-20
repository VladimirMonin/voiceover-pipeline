import importlib
import os
from typing import Any

from voiceover_pipeline.models import ASRCapabilities, ASRExecutionReceipt, ASRRequest, ASRResult
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider

NEMOTRON_ASR_PROVIDER_ID = "nemotron-local"
NEMOTRON_ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_ASR_INSTALL_REMEDIATION = "Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying."
_NEMOTRON_LANGUAGE_LOCALES = {
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "ru": "ru-RU",
}


def nemotron_asr_dependency_probe() -> ASRDependencyHealth:
    if os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip():
        from voiceover_pipeline.providers.audio_cpp_nemotron_asr import (
            audio_cpp_nemotron_asr_dependency_probe,
        )

        return audio_cpp_nemotron_asr_dependency_probe()
    try:
        transformers = importlib.import_module("transformers")
        importlib.import_module("accelerate")
        importlib.import_module("librosa")
    except ImportError:
        return ASRDependencyHealth(available=False, remediation=NEMOTRON_ASR_INSTALL_REMEDIATION)
    if not all(hasattr(transformers, name) for name in ("AutoModelForRNNT", "AutoProcessor")):
        return ASRDependencyHealth(available=False, remediation=NEMOTRON_ASR_INSTALL_REMEDIATION)
    return ASRDependencyHealth(available=True, remediation="")


def _nemotron_language_locale(language: str | None) -> str:
    if not language:
        return "auto"
    return _NEMOTRON_LANGUAGE_LOCALES.get(language.lower(), language)


class NemotronLocalASRProvider(ASRProvider):
    """Deferred-import Transformers batch-ASR adapter; no bias or alignment contract is assumed."""

    provider_id = NEMOTRON_ASR_PROVIDER_ID

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._loaded_model_id: str | None = None
        self._loaded_device: str | None = None
        self._loaded_compute: str | None = None
        self._runtime_version: str | None = None

    def _load_model(self, request: ASRRequest) -> None:
        import transformers
        from transformers import AutoProcessor

        auto_model_for_rnnt = getattr(transformers, "AutoModelForRNNT")
        model_id = request.model_id or NEMOTRON_ASR_MODEL_ID
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = auto_model_for_rnnt.from_pretrained(
            model_id, device_map=request.device
        ).eval()
        self._loaded_model_id = model_id
        self._loaded_device = request.device
        self._loaded_compute = request.compute
        self._runtime_version = getattr(transformers, "__version__", None)

    def transcribe(self, request: ASRRequest) -> ASRResult:
        if request.timestamp_mode == "word":
            raise ValueError(
                "Nemotron Python fallback does not support word timestamps; "
                "set VOICEOVER_AUDIO_CPP_BINARY to use the native audio.cpp route"
            )
        model_id = request.model_id or NEMOTRON_ASR_MODEL_ID
        if (
            self._model is None
            or self._loaded_model_id != model_id
            or self._loaded_device != request.device
            or self._loaded_compute != request.compute
        ):
            self._load_model(request)

        assert self._model is not None
        assert self._processor is not None
        from transformers.audio_utils import load_audio

        sampling_rate = self._processor.feature_extractor.sampling_rate
        audio = load_audio(str(request.audio_path), sampling_rate=sampling_rate)
        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            language=_nemotron_language_locale(request.language),
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        results = self._model.generate(**inputs, return_dict_in_generate=True)
        decoded = self._processor.decode(results.sequences, skip_special_tokens=True)
        transcript = decoded[0] if isinstance(decoded, list) and len(decoded) == 1 else decoded
        if not isinstance(transcript, str):
            raise ValueError("nemotron ASR response has no text result")

        return ASRResult(
            transcript=transcript,
            provider_id=self.provider_id,
            model_id=model_id,
            language=request.language or "",
            execution=ASRExecutionReceipt(
                runtime="transformers-nemotron-3.5-asr",
                runtime_version=self._runtime_version,
                resolved_device=request.device,
                resolved_compute=request.compute,
            ),
        )


def nemotron_asr_provider_factory() -> ASRProvider:
    """Choose the explicit audio.cpp route without renaming the family provider."""
    if os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip():
        from voiceover_pipeline.providers.audio_cpp_nemotron_asr import AudioCppNemotronASRProvider

        return AudioCppNemotronASRProvider.from_environment()
    return NemotronLocalASRProvider()


NEMOTRON_ASR_PROVIDER_SPEC = ASRProviderSpec(
    provider_id=NEMOTRON_ASR_PROVIDER_ID,
    description="Local Nemotron 3.5 ASR with runtime-selected native word timestamps.",
    factory=nemotron_asr_provider_factory,
    models=({"id": NEMOTRON_ASR_MODEL_ID, "default": True},),
    capabilities=ASRCapabilities(
        batch_audio=True,
        forced_language=True,
        segment_timestamps=True,
        word_timestamps=True,
        device_modes=("cpu", "cuda"),
        compute_modes=("auto",),
    ),
    dependency_probe=nemotron_asr_dependency_probe,
)
