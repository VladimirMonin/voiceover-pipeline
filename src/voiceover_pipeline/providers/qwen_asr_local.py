import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
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
from voiceover_pipeline.settings import SettingsError, load_qwen_asr_local_settings

QWEN_ASR_PROVIDER_ID = "qwen-local"
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
QWEN_ASR_LARGE_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

# Each selectable model owns exactly one on-disk directory name, so a selected
# size resolves only its own weights and can never be served by another size's
# directory.
QWEN_ASR_MODEL_DIRECTORY_NAMES: Final = {
    QWEN_ASR_MODEL_ID: "Qwen3-ASR-0.6B",
    QWEN_ASR_LARGE_MODEL_ID: "Qwen3-ASR-1.7B",
}
QWEN_ASR_MODELS_DIRECTORY_NAME: Final = "models"
QWEN_ASR_CACHE_DIRECTORY_NAME: Final = "huggingface-cache"
QWEN_ASR_FORCED_ALIGNER_DIRECTORY_NAME: Final = "Qwen3-ForcedAligner-0.6B"
# The one implicit asset root the project used before paths were configurable. It
# stays a documented, warned fallback rather than the only accepted location.
QWEN_ASR_LEGACY_STORAGE_ROOT: Final = Path("/media/v/storage/voiceover-pipeline/qwen-asr")

QWEN_ASR_MODELS_ROOT_ENV: Final = "VOICEOVER_QWEN_ASR_MODELS_ROOT"
QWEN_ASR_CACHE_DIR_ENV: Final = "VOICEOVER_QWEN_ASR_CACHE_DIR"
QWEN_ASR_REVISION_ENV: Final = "VOICEOVER_QWEN_ASR_REVISION"

QWEN_ASR_INSTALL_REMEDIATION = (
    "qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying."
)
QWEN_ASR_STORAGE_REMEDIATION = (
    "Qwen local ASR weights or cache for the selected model are unavailable. Install the "
    "approved Qwen3-ASR weights and cache under the configured models root "
    "(VOICEOVER_QWEN_ASR_MODELS_ROOT or settings.toml [asr.qwen_local] models_root) or under "
    "the legacy /media/v/storage/voiceover-pipeline/qwen-asr layout before retrying."
)
QWEN_ASR_REVISION_REMEDIATION = (
    "The configured Qwen3-ASR revision is unavailable: the resolved weights directory must be the "
    "matching Hugging Face snapshot. The models root (VOICEOVER_QWEN_ASR_MODELS_ROOT or "
    "settings.toml [asr.qwen_local] models_root) is the parent of models/<selected-name>, so make "
    "that selected-name directory the matching snapshot or clear VOICEOVER_QWEN_ASR_REVISION / "
    "settings.toml [asr.qwen_local] revision before retrying."
)
QWEN_ASR_IDENTITY_REMEDIATION = (
    "The resolved Qwen3-ASR weights directory does not match the selected model. Point "
    "models/<selected-name> under the models root at that model's own weights directory or at its "
    "matching Hugging Face snapshot before retrying."
)
QWEN_ASR_ASSET_CHANGE_REMEDIATION = (
    "The local Qwen3-ASR weights or cache configuration changed after the model was loaded. "
    "Keep one asset configuration for the whole run and retry with a fresh process."
)
QWEN_ASR_LEGACY_STORAGE_WARNING = (
    "Qwen local ASR is using the legacy /media/v/storage/voiceover-pipeline/qwen-asr asset "
    "root. Set VOICEOVER_QWEN_ASR_MODELS_ROOT or settings.toml [asr.qwen_local] models_root "
    "to an explicit location."
)
QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION = (
    "Qwen word timestamps require Qwen3-ForcedAligner-0.6B. "
    "Install the approved official aligner under the configured models root before retrying."
)
_QWEN_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "ru": "Russian",
}


class QwenASRLegacyStorageWarning(UserWarning):
    """The implicit legacy /media/v/storage Qwen ASR asset root is still in use."""


@dataclass(frozen=True)
class QwenASRLocalAssets:
    """The resolved local directories, revision, and selection identity for one model.

    ``revision`` is the configured pin, ``observed_revision`` is the revision the
    resolved weights path itself declares (an HF snapshot directory name, or
    ``None``), and ``selection_verified`` records whether that resolved path
    provably belongs to the selected model.
    """

    model_id: str
    model_path: Path
    forced_aligner_path: Path
    cache_dir: Path
    resolved_model_path: Path
    resolved_forced_aligner_path: Path
    resolved_cache_dir: Path
    revision: str | None
    observed_revision: str | None
    selection_verified: bool
    legacy_storage_root: bool


_legacy_storage_warning_emitted = False


def _first_configured_string(*candidates: str | None) -> str | None:
    """Return the first non-blank candidate, so an environment value wins over settings."""
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _first_configured_path(*candidates: str | None) -> Path | None:
    value = _first_configured_string(*candidates)
    return None if value is None else Path(value).expanduser()


def _warn_legacy_storage_root() -> None:
    """Warn once per process that the implicit legacy asset root is being used."""
    global _legacy_storage_warning_emitted
    if _legacy_storage_warning_emitted:
        return
    _legacy_storage_warning_emitted = True
    warnings.warn(QWEN_ASR_LEGACY_STORAGE_WARNING, QwenASRLegacyStorageWarning, stacklevel=3)


def _hf_repository_directory_name(model_id: str) -> str:
    """Return the ``models--<org>--<name>`` directory name Hugging Face uses."""
    return "models--" + model_id.replace("/", "--")


def _resolve_local_weights_identity(
    model_id: str, model_path: Path
) -> tuple[Path, str | None, bool]:
    """Return the exact resolved target, its revision, and selection proof.

    A weights directory that is the direct child of a ``snapshots`` directory *is*
    that snapshot: it declares its own directory name as its revision and proves
    the selection only when its ``models--<org>--<name>`` parent matches the
    selected model. Any other layout declares no revision and proves the selection
    only when the resolved directory name equals the selected model's own
    directory name, so an alias into another size's directory or snapshot is
    refused instead of being assumed.
    """
    resolved = model_path.expanduser().resolve()
    if resolved.parent.name == "snapshots":
        return (
            resolved,
            resolved.name,
            resolved.parent.parent.name == _hf_repository_directory_name(model_id),
        )
    return resolved, None, resolved.name == QWEN_ASR_MODEL_DIRECTORY_NAMES[model_id]


def _admitted_asset_targets(assets: QwenASRLocalAssets) -> tuple[Path, Path, Path]:
    """Return the canonical weights, aligner, and cache targets of one admission.

    These targets were captured by the resolver itself, alongside the identity
    proof, not re-resolved after admission. Retargeting an alias before or during
    the runtime load therefore cannot change the admitted weights or receipt.
    """
    return (
        assets.resolved_model_path,
        assets.resolved_forced_aligner_path,
        assets.resolved_cache_dir,
    )


def _assets_identity(assets: QwenASRLocalAssets) -> tuple[Path, Path, Path, str | None]:
    """Return the canonical asset identity a cached model must keep matching."""
    return (*_admitted_asset_targets(assets), assets.revision)


def resolve_qwen_asr_local_assets(
    model_id: str, *, environ: Mapping[str, str] | None = None
) -> QwenASRLocalAssets:
    """Resolve one selected model's weights directory, cache, revision, and identity.

    Only paths are computed: nothing is loaded, downloaded, or created. The
    explicit ``VOICEOVER_QWEN_ASR_*`` environment variables win over the
    ``[asr.qwen_local]`` settings, and the documented legacy storage root is the
    last fallback. An unknown model id raises ``ValueError`` instead of resolving
    to another model's directory.
    """
    directory_name = QWEN_ASR_MODEL_DIRECTORY_NAMES.get(model_id)
    if directory_name is None:
        raise ValueError(f"Unknown local Qwen ASR model: {model_id}")
    active_environ = os.environ if environ is None else environ
    settings = load_qwen_asr_local_settings()
    configured_root = _first_configured_path(
        active_environ.get(QWEN_ASR_MODELS_ROOT_ENV), settings.models_root
    )
    models_root = QWEN_ASR_LEGACY_STORAGE_ROOT if configured_root is None else configured_root
    cache_dir = _first_configured_path(
        active_environ.get(QWEN_ASR_CACHE_DIR_ENV), settings.cache_dir
    )
    models_dir = models_root / QWEN_ASR_MODELS_DIRECTORY_NAME
    model_path = models_dir / directory_name
    forced_aligner_path = models_dir / QWEN_ASR_FORCED_ALIGNER_DIRECTORY_NAME
    effective_cache_dir = cache_dir or models_root / QWEN_ASR_CACHE_DIRECTORY_NAME
    resolved_model_path, observed_revision, selection_verified = _resolve_local_weights_identity(
        model_id, model_path
    )
    return QwenASRLocalAssets(
        model_id=model_id,
        model_path=model_path,
        forced_aligner_path=forced_aligner_path,
        cache_dir=effective_cache_dir,
        resolved_model_path=resolved_model_path,
        resolved_forced_aligner_path=forced_aligner_path.expanduser().resolve(),
        resolved_cache_dir=effective_cache_dir.expanduser().resolve(),
        revision=_first_configured_string(
            active_environ.get(QWEN_ASR_REVISION_ENV), settings.revision
        ),
        observed_revision=observed_revision,
        selection_verified=selection_verified,
        legacy_storage_root=configured_root is None,
    )


def admit_qwen_asr_local_assets(
    model_id: str, *, timestamp_mode: str = "none", environ: Mapping[str, str] | None = None
) -> QwenASRLocalAssets:
    """Return the selected model's assets, or fail closed before any model load.

    The check runs before the ``qwen_asr``/``torch`` import and before any
    ``from_pretrained`` call, and it never downloads. Missing weights or cache, a
    resolved path that does not prove the selected model, a configured revision
    the resolved weights cannot prove, and a missing forced aligner for a
    word-timestamp request each raise ``ModuleNotFoundError`` with a fixed
    remediation. Using the legacy asset root also emits one deprecation warning.
    An unknown model id raises ``ValueError``, and an unreadable ``settings.toml``
    raises :class:`SettingsError`.
    """
    assets = resolve_qwen_asr_local_assets(model_id, environ=environ)
    if assets.legacy_storage_root:
        _warn_legacy_storage_root()
    if not assets.resolved_model_path.is_dir() or not assets.resolved_cache_dir.is_dir():
        raise ModuleNotFoundError(QWEN_ASR_STORAGE_REMEDIATION)
    if not assets.selection_verified:
        raise ModuleNotFoundError(QWEN_ASR_IDENTITY_REMEDIATION)
    if assets.revision is not None and assets.observed_revision != assets.revision:
        raise ModuleNotFoundError(QWEN_ASR_REVISION_REMEDIATION)
    if timestamp_mode == "word" and not assets.resolved_forced_aligner_path.is_dir():
        raise ModuleNotFoundError(QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION)
    return assets


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


def _qwen_python_runtime_health() -> ASRDependencyHealth | None:
    """Return the install failure when the Python runtime cannot import, else ``None``."""
    try:
        _prepare_qwen_asr_import()
        importlib.import_module("qwen_asr")
        importlib.import_module("torch")
    except ModuleNotFoundError:
        return ASRDependencyHealth(available=False, remediation=QWEN_ASR_INSTALL_REMEDIATION)
    return None


def qwen_asr_python_dependency_probe(model_id: str | None = None) -> ASRDependencyHealth:
    """Probe only the Python route; a selected model is admitted alone when known.

    Nothing is constructed or downloaded: the runtime package must import and,
    when ``model_id`` is given, that model must pass the same admission a real
    request uses. Without a selected model the probe reports available when any
    selectable model is usable, and otherwise the last admission failure.
    """
    runtime_health = _qwen_python_runtime_health()
    if runtime_health is not None:
        return runtime_health
    candidates = (model_id,) if model_id is not None else tuple(QWEN_ASR_MODEL_DIRECTORY_NAMES)
    failure: Exception | None = None
    for candidate in candidates:
        try:
            admit_qwen_asr_local_assets(candidate)
        except (ModuleNotFoundError, SettingsError, ValueError) as exc:
            failure = exc
            continue
        return ASRDependencyHealth(available=True, remediation="")
    return ASRDependencyHealth(
        available=False,
        remediation=str(failure) if failure is not None else QWEN_ASR_STORAGE_REMEDIATION,
    )


def qwen_asr_dependency_probe() -> ASRDependencyHealth:
    """Report whether the selected local Qwen3 ASR runtime can run without a download.

    When an audio.cpp install is configured the probe reports that runtime's
    health; otherwise it reports the Python route's health, with no model
    constructed and no download.
    """
    if (
        os.environ.get("VOICEOVER_AUDIO_CPP_BINARY", "").strip()
        or os.environ.get("VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE", "").strip()
    ):
        from voiceover_pipeline.providers.audio_cpp_qwen_asr import (
            audio_cpp_qwen_asr_dependency_probe,
        )

        return audio_cpp_qwen_asr_dependency_probe()
    return qwen_asr_python_dependency_probe()


def _qwen_language_name(language: str | None) -> str | None:
    if language is None:
        return None
    return _QWEN_LANGUAGE_NAMES.get(language.casefold(), language)


class QwenLocalASRProvider(ASRProvider):
    """Deferred-import Qwen3 ASR adapter with optional official forced alignment."""

    provider_id = QWEN_ASR_PROVIDER_ID

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded_model_id: str | None = None
        self._loaded_model_revision: str | None = None
        self._loaded_model_path: str | None = None
        self._loaded_assets_identity: tuple[Path, Path, Path, str | None] | None = None
        self._loaded_device: str | None = None
        self._loaded_compute: str | None = None
        self._loaded_with_forced_aligner = False
        self._resolved_compute: str | None = None
        self._runtime_version: str | None = None

    def _load_model(self, request: ASRRequest) -> None:
        model_id = request.model_id or QWEN_ASR_MODEL_ID
        assets = admit_qwen_asr_local_assets(model_id, timestamp_mode=request.timestamp_mode)
        # Capture the admitted canonical targets once, before importing or calling
        # the runtime, so the load and the receipt use the exact weights, aligner,
        # and cache that admission verified rather than a later alias resolution.
        model_target, aligner_target, cache_target = _admitted_asset_targets(assets)
        _prepare_qwen_asr_import()
        import qwen_asr
        import torch
        from qwen_asr import Qwen3ASRModel

        resolved_compute = request.compute
        if resolved_compute == "auto":
            resolved_compute = "bfloat16" if request.device == "cuda" else "float32"
        dtype = getattr(torch, resolved_compute)

        load_options: dict[str, object] = {
            "cache_dir": str(cache_target),
            "device_map": request.device,
            "dtype": dtype,
            "local_files_only": True,
        }
        if request.timestamp_mode == "word":
            load_options["forced_aligner"] = str(aligner_target)
            load_options["forced_aligner_kwargs"] = {
                "cache_dir": str(cache_target),
                "device_map": request.device,
                "dtype": dtype,
                "local_files_only": True,
            }
        try:
            self._model = Qwen3ASRModel.from_pretrained(str(model_target), **load_options)
        except (OSError, RuntimeError, ValueError) as exc:
            if request.timestamp_mode == "word":
                raise ModuleNotFoundError(QWEN_FORCED_ALIGNER_INSTALL_REMEDIATION) from exc
            raise
        self._loaded_model_id = model_id
        # The observed snapshot revision is the effective one; the configured pin
        # stays admission-only. A plain directory with no pin reports ``None``.
        self._loaded_model_revision = assets.observed_revision
        # Record the same admitted canonical target the model loaded from, never a
        # fresh resolution of an alias that may have been retargeted during the call.
        self._loaded_model_path = str(model_target)
        self._loaded_assets_identity = (model_target, aligner_target, cache_target, assets.revision)
        self._loaded_device = request.device
        self._loaded_compute = request.compute
        self._loaded_with_forced_aligner = request.timestamp_mode == "word"
        self._resolved_compute = resolved_compute
        self._runtime_version = getattr(qwen_asr, "__version__", None)

    def _require_unchanged_assets(self, model_id: str) -> None:
        """Fail closed when the asset configuration changed after a load.

        A cached model must keep serving the weights it loaded; silently reusing
        it after the configured root, cache, or revision changed would run the
        wrong model. Re-resolving only compares configured paths, so the error is
        raised before any runtime call instead of swapping weights mid-run.
        """
        current = resolve_qwen_asr_local_assets(model_id)
        if self._loaded_assets_identity != _assets_identity(current):
            raise ModuleNotFoundError(QWEN_ASR_ASSET_CHANGE_REMEDIATION)

    def transcribe(self, request: ASRRequest) -> ASRResult:
        model_id = request.model_id or QWEN_ASR_MODEL_ID
        if self._model is not None and self._loaded_model_id == model_id:
            self._require_unchanged_assets(model_id)
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
                model_revision=self._loaded_model_revision,
                model_path=self._loaded_model_path,
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


def qwen_asr_python_provider_factory() -> ASRProvider:
    """Explicit Python route; never selects the native audio.cpp package."""
    return QwenLocalASRProvider()


def qwen_asr_audio_cpp_provider_factory() -> ASRProvider:
    """Explicit native route; never falls back to the Python runtime."""
    from voiceover_pipeline.providers.audio_cpp_qwen_asr import AudioCppQwenASRProvider

    return AudioCppQwenASRProvider.from_environment()


QWEN_ASR_PROVIDER_SPEC = ASRProviderSpec(
    provider_id=QWEN_ASR_PROVIDER_ID,
    description=(
        "Local Qwen3 ASR with selectable 0.6B/1.7B weights and runtime-selected "
        "optional forced alignment."
    ),
    factory=qwen_asr_provider_factory,
    models=(
        {"id": QWEN_ASR_MODEL_ID, "default": True},
        {"id": QWEN_ASR_LARGE_MODEL_ID},
    ),
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
