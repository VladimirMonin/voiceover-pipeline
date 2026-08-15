import importlib
from typing import Any

from voiceover_pipeline.models import ASRCapabilities, ASRExecutionReceipt, ASRRequest, ASRResult
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider


QWEN_ASR_PROVIDER_ID = "qwen-local"
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
QWEN_ASR_INSTALL_REMEDIATION = "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
_QWEN_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "ru": "Russian",
}


def qwen_asr_dependency_probe() -> ASRDependencyHealth:
    try:
        importlib.import_module("qwen_asr")
        importlib.import_module("torch")
    except ModuleNotFoundError:
        return ASRDependencyHealth(available=False, remediation=QWEN_ASR_INSTALL_REMEDIATION)
    return ASRDependencyHealth(available=True, remediation="")


def _qwen_language_name(language: str | None) -> str | None:
    if language is None:
        return None
    return _QWEN_LANGUAGE_NAMES.get(language.casefold(), language)


class QwenLocalASRProvider(ASRProvider):
    """Deferred-import Qwen3 text ASR adapter; alignment stays a separate stage."""

    provider_id = QWEN_ASR_PROVIDER_ID

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_model_id: str | None = None
        self._loaded_device: str | None = None
        self._loaded_compute: str | None = None
        self._resolved_compute: str | None = None
        self._runtime_version: str | None = None

    def _load_model(self, request: ASRRequest) -> None:
        import torch
        import qwen_asr
        from qwen_asr import Qwen3ASRModel

        model_id = request.model_id or QWEN_ASR_MODEL_ID
        resolved_compute = request.compute
        if resolved_compute == "auto":
            resolved_compute = "bfloat16" if request.device == "cuda" else "float32"
        dtype = getattr(torch, resolved_compute)

        self._model = Qwen3ASRModel.from_pretrained(
            model_id,
            device_map=request.device,
            dtype=dtype,
        )
        self._loaded_model_id = model_id
        self._loaded_device = request.device
        self._loaded_compute = request.compute
        self._resolved_compute = resolved_compute
        self._runtime_version = getattr(qwen_asr, "__version__", None)

    def transcribe(self, request: ASRRequest) -> ASRResult:
        model_id = request.model_id or QWEN_ASR_MODEL_ID
        if (
            self._model is None
            or self._loaded_model_id != model_id
            or self._loaded_device != request.device
            or self._loaded_compute != request.compute
        ):
            self._load_model(request)

        assert self._model is not None
        results = self._model.transcribe(
            audio=str(request.audio_path),
            context=request.hints.context_text,
            language=_qwen_language_name(request.language),
        )
        try:
            response = results[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("qwen-asr returned no transcription result") from exc
        transcript = getattr(response, "text", None)
        if not isinstance(transcript, str):
            raise ValueError("qwen-asr response has no text result")
        language = getattr(response, "language", None) or request.language or ""

        return ASRResult(
            transcript=transcript,
            provider_id=self.provider_id,
            model_id=model_id,
            language=language,
            execution=ASRExecutionReceipt(
                runtime="qwen-asr",
                runtime_version=self._runtime_version,
                resolved_device=request.device,
                resolved_compute=self._resolved_compute or request.compute,
            ),
        )


QWEN_ASR_PROVIDER_SPEC = ASRProviderSpec(
    provider_id=QWEN_ASR_PROVIDER_ID,
    description="Local Qwen3 text ASR without timestamp or alignment output.",
    factory=QwenLocalASRProvider,
    models=({"id": QWEN_ASR_MODEL_ID, "default": True},),
    capabilities=ASRCapabilities(
        batch_audio=True,
        forced_language=True,
        contextual_bias=True,
        device_modes=("cpu", "cuda"),
        compute_modes=("auto", "bfloat16", "float32"),
    ),
    dependency_probe=qwen_asr_dependency_probe,
)
