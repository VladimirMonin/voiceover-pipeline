import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any, Final

from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider, validate_asr_response

QWEN_ASR_PROVIDER_ID = "qwen-local"
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
QWEN_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_ASR_STORAGE_ROOT: Final = Path("/media/v/storage/voiceover-pipeline/qwen-asr")
QWEN_ASR_MODEL_PATH: Final = QWEN_ASR_STORAGE_ROOT / "models" / "Qwen3-ASR-0.6B"
QWEN_FORCED_ALIGNER_MODEL_PATH: Final = (
    QWEN_ASR_STORAGE_ROOT / "models" / "Qwen3-ForcedAligner-0.6B"
)
QWEN_ASR_CACHE_DIR: Final = QWEN_ASR_STORAGE_ROOT / "huggingface-cache"
QWEN_ASR_INSTALL_REMEDIATION = (
    "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
)
QWEN_ASR_STORAGE_REMEDIATION = (
    "Qwen local assets are unavailable under /media/v/storage. "
    "Install the approved Qwen ASR model and cache there before retrying."
)
QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION = (
    "Qwen word timestamps require Qwen3-ForcedAligner-0.6B. "
    "Install the approved official aligner under /media/v/storage before retrying."
)
_QWEN_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "ru": "Russian",
}


class _LazyNagisaModule(types.ModuleType):
    """Delay DyNet-backed Japanese tokenization until it is actually used."""

    def __init__(self, spec: importlib.machinery.ModuleSpec) -> None:
        super().__init__("nagisa")
        self.__spec__ = spec
        self.__file__ = spec.origin
        self.__loader__ = spec.loader
        self.__package__ = "nagisa"
        self.__path__ = list(spec.submodule_search_locations or ())
        self._real_module: types.ModuleType | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> types.ModuleType:
        with self._load_lock:
            if self._real_module is not None:
                return self._real_module
            if sys.modules.get("nagisa") is self:
                del sys.modules["nagisa"]
            try:
                real_module = importlib.import_module("nagisa")
            except BaseException:
                sys.modules["nagisa"] = self
                raise
            self._real_module = real_module
            sys.modules["nagisa"] = real_module
            return real_module

    def __getattr__(self, name: str) -> object:
        return getattr(self._load(), name)


def _prepare_qwen_asr_import() -> None:
    """Avoid loading unstable DyNet when non-Japanese Qwen alignment is used."""

    if "nagisa" in sys.modules:
        return
    spec = importlib.util.find_spec("nagisa")
    if spec is not None:
        sys.modules["nagisa"] = _LazyNagisaModule(spec)


def qwen_asr_dependency_probe() -> ASRDependencyHealth:
    if (
        os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        or os.environ.get("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", "").strip()
    ):
        from voiceover_pipeline.providers.audio_cpp_qwen_asr import (
            audio_cpp_qwen_asr_dependency_probe,
        )

        return audio_cpp_qwen_asr_dependency_probe()
    try:
        _prepare_qwen_asr_import()
        importlib.import_module("qwen_asr")
        importlib.import_module("torch")
    except ModuleNotFoundError:
        return ASRDependencyHealth(available=False, remediation=QWEN_ASR_INSTALL_REMEDIATION)
    if not QWEN_ASR_MODEL_PATH.is_dir() or not QWEN_ASR_CACHE_DIR.is_dir():
        return ASRDependencyHealth(available=False, remediation=QWEN_ASR_STORAGE_REMEDIATION)
    return ASRDependencyHealth(available=True, remediation="")


def _qwen_language_name(language: str | None) -> str | None:
    if language is None:
        return None
    return _QWEN_LANGUAGE_NAMES.get(language.casefold(), language)


def _admit_local_qwen_storage(*, timestamp_mode: str) -> tuple[Path, Path | None]:
    if not QWEN_ASR_MODEL_PATH.is_dir() or not QWEN_ASR_CACHE_DIR.is_dir():
        raise ModuleNotFoundError(QWEN_ASR_STORAGE_REMEDIATION)
    if timestamp_mode != "word":
        return QWEN_ASR_MODEL_PATH, None
    if not QWEN_FORCED_ALIGNER_MODEL_PATH.is_dir():
        raise ModuleNotFoundError(QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION)
    return QWEN_ASR_MODEL_PATH, QWEN_FORCED_ALIGNER_MODEL_PATH


class QwenLocalASRProvider(ASRProvider):
    """Deferred-import Qwen3 ASR adapter with optional official forced alignment."""

    provider_id = QWEN_ASR_PROVIDER_ID

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_model_id: str | None = None
        self._loaded_device: str | None = None
        self._loaded_compute: str | None = None
        self._loaded_with_forced_aligner = False
        self._resolved_compute: str | None = None
        self._runtime_version: str | None = None

    def _load_model(self, request: ASRRequest) -> None:
        model_path, forced_aligner_path = _admit_local_qwen_storage(
            timestamp_mode=request.timestamp_mode
        )
        _prepare_qwen_asr_import()
        import qwen_asr
        import torch
        from qwen_asr import Qwen3ASRModel

        model_id = request.model_id or QWEN_ASR_MODEL_ID
        resolved_compute = request.compute
        if resolved_compute == "auto":
            resolved_compute = "bfloat16" if request.device == "cuda" else "float32"
        dtype = getattr(torch, resolved_compute)

        load_options: dict[str, object] = {
            "cache_dir": str(QWEN_ASR_CACHE_DIR),
            "device_map": request.device,
            "dtype": dtype,
            "local_files_only": True,
        }
        if request.timestamp_mode == "word":
            assert forced_aligner_path is not None
            load_options["forced_aligner"] = str(forced_aligner_path)
            load_options["forced_aligner_kwargs"] = {
                "cache_dir": str(QWEN_ASR_CACHE_DIR),
                "device_map": request.device,
                "dtype": dtype,
                "local_files_only": True,
            }
        try:
            self._model = Qwen3ASRModel.from_pretrained(str(model_path), **load_options)
        except (OSError, RuntimeError, ValueError) as exc:
            if request.timestamp_mode == "word":
                raise ModuleNotFoundError(QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION) from exc
            raise
        self._loaded_model_id = model_id
        self._loaded_device = request.device
        self._loaded_compute = request.compute
        self._loaded_with_forced_aligner = request.timestamp_mode == "word"
        self._resolved_compute = resolved_compute
        self._runtime_version = getattr(qwen_asr, "__version__", None)

    def transcribe(self, request: ASRRequest) -> ASRResult:
        model_id = request.model_id or QWEN_ASR_MODEL_ID
        if (
            self._model is None
            or self._loaded_model_id != model_id
            or self._loaded_device != request.device
            or self._loaded_compute != request.compute
            or (request.timestamp_mode == "word" and not self._loaded_with_forced_aligner)
        ):
            self._load_model(request)

        assert self._model is not None
        transcribe_options: dict[str, object] = {
            "audio": str(request.audio_path),
            "context": request.hints.context_text,
            "language": _qwen_language_name(request.language),
        }
        if request.timestamp_mode == "word":
            transcribe_options["return_time_stamps"] = True
        results = self._model.transcribe(**transcribe_options)
        try:
            response = results[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("qwen-asr returned no transcription result") from exc
        transcript = getattr(response, "text", None)
        if not isinstance(transcript, str):
            raise ValueError("qwen-asr response has no text result")
        language = getattr(response, "language", None) or request.language or ""

        words = (
            _forced_words(response, transcript=transcript)
            if request.timestamp_mode == "word"
            else ()
        )
        result = ASRResult(
            transcript=transcript,
            provider_id=self.provider_id,
            model_id=model_id,
            language=language,
            words=words,
            alignment_origin="forced" if request.timestamp_mode == "word" else None,
            execution=ASRExecutionReceipt(
                runtime="qwen-asr",
                runtime_version=self._runtime_version,
                resolved_device=request.device,
                resolved_compute=self._resolved_compute or request.compute,
            ),
        )
        return validate_asr_response(request, result)


def _forced_words(response: object, *, transcript: str) -> tuple[ASRWordSpan, ...]:
    raw_words = getattr(response, "time_stamps", None)
    if raw_words is None:
        raise ValueError(
            "Qwen3-ForcedAligner-0.6B did not return time_stamps for a word timestamp request"
        )
    aligned_items: tuple[object, ...]
    if isinstance(raw_words, (list, tuple)):
        aligned_items = tuple(raw_words)
    else:
        candidate_items = getattr(raw_words, "items", None)
        if not isinstance(candidate_items, (list, tuple)):
            raise ValueError("Qwen3-ForcedAligner-0.6B time_stamps must contain an items sequence")
        aligned_items = tuple(candidate_items)
    if not aligned_items:
        if transcript.strip():
            raise ValueError("Qwen3-ForcedAligner-0.6B returned no words for speech output")
        return ()
    validated_items: list[tuple[str, float, float]] = []
    for index, raw_word in enumerate(aligned_items):
        text = getattr(raw_word, "text", None)
        start_s = getattr(raw_word, "start_time", None)
        end_s = getattr(raw_word, "end_time", None)
        if not isinstance(text, str) or not text:
            raise ValueError(f"Qwen3-ForcedAligner-0.6B word {index} has no text")
        if (
            isinstance(start_s, bool)
            or isinstance(end_s, bool)
            or not isinstance(start_s, (int, float))
            or not isinstance(end_s, (int, float))
        ):
            raise ValueError(f"Qwen3-ForcedAligner-0.6B word {index} has invalid bounds")
        try:
            validated_items.append((text, float(start_s), float(end_s)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Qwen3-ForcedAligner-0.6B word {index} has invalid bounds") from exc
    restored_texts = _restore_forced_word_texts(
        transcript, tuple(text for text, _start_s, _end_s in validated_items)
    )
    return tuple(
        ASRWordSpan(text=text, start_s=start_s, end_s=end_s)
        for text, (_aligned_text, start_s, end_s) in zip(restored_texts, validated_items)
    )


def _restore_forced_word_texts(transcript: str, aligned_texts: tuple[str, ...]) -> tuple[str, ...]:
    """Map cleaned aligner items back to exact, ordered transcript slices."""

    match_starts: list[int] = []
    cursor = 0
    for index, aligned_text in enumerate(aligned_texts):
        if not any(character.isalnum() for character in aligned_text):
            raise ValueError(
                "Qwen3-ForcedAligner-0.6B word texts contain an ambiguous "
                f"non-speech-only item at word {index}"
            )
        match_start = transcript.find(aligned_text, cursor)
        if match_start < 0 or any(
            character.isalnum() for character in transcript[cursor:match_start]
        ):
            raise ValueError(
                "Qwen3-ForcedAligner-0.6B word texts cannot be mapped exactly and sequentially "
                f"onto the transcript at word {index}"
            )
        match_starts.append(match_start)
        cursor = match_start + len(aligned_text)

    if any(character.isalnum() for character in transcript[cursor:]):
        raise ValueError(
            "Qwen3-ForcedAligner-0.6B word texts do not cover the transcript's remaining speech text"
        )

    return tuple(
        transcript[0 if index == 0 else match_start : next_start]
        for index, (match_start, next_start) in enumerate(
            zip(match_starts, (*match_starts[1:], len(transcript)))
        )
    )


def qwen_asr_provider_factory() -> ASRProvider:
    """Choose a local Qwen runtime while retaining the canonical family provider ID."""
    if (
        os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        or os.environ.get("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", "").strip()
    ):
        from voiceover_pipeline.providers.audio_cpp_qwen_asr import AudioCppQwenASRProvider

        return AudioCppQwenASRProvider.from_environment()
    return QwenLocalASRProvider()


QWEN_ASR_PROVIDER_SPEC = ASRProviderSpec(
    provider_id=QWEN_ASR_PROVIDER_ID,
    description="Local Qwen3 ASR with runtime-selected optional forced alignment.",
    factory=qwen_asr_provider_factory,
    models=({"id": QWEN_ASR_MODEL_ID, "default": True},),
    capabilities=ASRCapabilities(
        batch_audio=True,
        forced_language=True,
        contextual_bias=True,
        segment_timestamps=True,
        word_timestamps=True,
        forced_alignment=True,
        device_modes=("cpu", "cuda"),
        compute_modes=("auto", "bfloat16", "float32"),
    ),
    dependency_probe=qwen_asr_dependency_probe,
)
