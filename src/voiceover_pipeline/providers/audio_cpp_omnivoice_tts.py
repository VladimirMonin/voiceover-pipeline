from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from shutil import which
from typing import Any
from uuid import uuid4

from voiceover_pipeline.audio_cpp.inventory import PINNED_AUDIO_CPP_REVISION
from voiceover_pipeline.config import (
    OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
    OMNIVOICE_DEFAULT_LANGUAGE,
    OMNIVOICE_DEFAULT_SEED,
    OMNIVOICE_DEFAULT_STEPS,
    OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE,
    OMNIVOICE_LOCAL_MODEL_ID,
    OMNIVOICE_STYLE_CONDITION,
)
from voiceover_pipeline.local_runtime.contracts import (
    LocalTTSRequest,
    OmniVoiceMode,
    RuntimeDriverHealth,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.drivers.audio_cpp import AudioCppRuntimeDriver
from voiceover_pipeline.local_runtime.gpu_lease import GPULeaseManager
from voiceover_pipeline.local_runtime.lifecycle import GPULifecycleOwner, probe_local_gpu_state
from voiceover_pipeline.local_runtime.manager import LocalAudioRuntime
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry
from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    NATIVE_AUDIO_CPP_EXECUTABLE_ENV,
    AudioCppNativeCLITransport,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_omnivoice import (
    PINNED_AUDIO_CPP_OMNIVOICE_BINARY_SHA256,
    AudioCppOmniVoiceCLITransport,
    VerifiedOmniVoiceModel,
    admit_omnivoice_model,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    AudioCppPackageError,
    admit_audio_cpp_native_package,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.omnivoice_voice_bank import VoiceProfile
from voiceover_pipeline.providers.base import TTSProvider

OMNIVOICE_FAMILY = "omnivoice"
OMNIVOICE_LOCAL_PROVIDER_ID = "omnivoice-local"
_OMNIVOICE_MODEL_ENV = "VOICEOVER_OMNIVOICE_MODEL"
_OMNIVOICE_CONTAINER_COMMAND_ENV = "VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON"
_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ENV = "VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE"
_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ACKNOWLEDGMENT = "accept-cc-by-nc-4.0-local-use"
_OMNIVOICE_MIN_FREE_VRAM_MB = 4096
_OMNIVOICE_MAX_GPU_UTILIZATION_PERCENT = 90
OMNIVOICE_INSTALL_REMEDIATION = (
    "OmniVoice local runtime is unavailable. Set VOICEOVER_OMNIVOICE_MODEL to the approved "
    "local Q8_0 artifact, set VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE to "
    "accept-cc-by-nc-4.0-local-use, and configure a local container command before retrying."
)


def _container_command_from_environment() -> tuple[str, ...]:
    raw = os.environ.get(_OMNIVOICE_CONTAINER_COMMAND_ENV, "").strip()
    if not raw:
        return ("docker",)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_OMNIVOICE_CONTAINER_COMMAND_ENV} must be a JSON array of argv strings"
        ) from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(part, str) and part and "\x00" not in part for part in parsed)
    ):
        raise ValueError(f"{_OMNIVOICE_CONTAINER_COMMAND_ENV} must be a JSON array of argv strings")
    return tuple(parsed)


def _admitted_omnivoice_model_from_environment() -> VerifiedOmniVoiceModel | None:
    model_path = os.environ.get(_OMNIVOICE_MODEL_ENV, "").strip()
    acknowledgment = os.environ.get(_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ENV, "").strip()
    if not model_path or acknowledgment != _OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ACKNOWLEDGMENT:
        return None
    try:
        return admit_omnivoice_model(model_path=Path(model_path), model_id=OMNIVOICE_LOCAL_MODEL_ID)
    except (OSError, ValueError):
        return None


def _re_admit_omnivoice_model(
    admitted_model: VerifiedOmniVoiceModel | None,
) -> VerifiedOmniVoiceModel | None:
    """Re-check caller-provided admission at each exported provider boundary."""
    if admitted_model is None:
        return _admitted_omnivoice_model_from_environment()
    acknowledgment = os.environ.get(_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ENV, "").strip()
    if acknowledgment != _OMNIVOICE_NONCOMMERCIAL_LOCAL_USE_ACKNOWLEDGMENT:
        return None
    try:
        return admit_omnivoice_model(
            model_path=admitted_model.model_path,
            model_id=admitted_model.model_id,
        )
    except (OSError, ValueError):
        return None


def omnivoice_local_dependency_probe(
    admitted_model: VerifiedOmniVoiceModel | None = None,
) -> RuntimeDriverHealth:
    model = _re_admit_omnivoice_model(admitted_model)
    if model is None:
        return RuntimeDriverHealth(available=False, remediation=OMNIVOICE_INSTALL_REMEDIATION)
    if sys.platform.startswith("win"):
        native_executable = os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip()
        if not native_executable:
            return RuntimeDriverHealth(
                available=False,
                remediation=OMNIVOICE_INSTALL_REMEDIATION,
                reason_code="missing_native_executable",
            )
        try:
            admit_audio_cpp_native_package(
                Path(native_executable), required_model_paths=(model.model_path,)
            )
        except AudioCppPackageError as exc:
            return RuntimeDriverHealth(
                available=False,
                remediation=OMNIVOICE_INSTALL_REMEDIATION,
                reason_code=exc.code,
            )
        except ValueError:
            return RuntimeDriverHealth(
                available=False,
                remediation=OMNIVOICE_INSTALL_REMEDIATION,
                reason_code="invalid_native_package",
            )
        return RuntimeDriverHealth(available=True)
    try:
        command = _container_command_from_environment()
    except ValueError:
        return RuntimeDriverHealth(available=False, remediation=OMNIVOICE_INSTALL_REMEDIATION)
    if which(command[0]) is None:
        return RuntimeDriverHealth(available=False, remediation=OMNIVOICE_INSTALL_REMEDIATION)
    return RuntimeDriverHealth(available=True)


class OmniVoiceLocalTTSProvider(TTSProvider):
    """Offline OmniVoice route with fixed-style, clone, and design modes."""

    provider_id = OMNIVOICE_LOCAL_PROVIDER_ID

    def __init__(
        self,
        runtime: LocalAudioRuntime | None = None,
        *,
        admitted_model: VerifiedOmniVoiceModel | None = None,
        model_id: str = OMNIVOICE_LOCAL_MODEL_ID,
        language: str = OMNIVOICE_DEFAULT_LANGUAGE,
        seed: int = OMNIVOICE_DEFAULT_SEED,
        num_inference_steps: int = OMNIVOICE_DEFAULT_STEPS,
        guidance_scale: float = OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
        mode: str = "preset",
        reference_audio_path: Path | str | None = None,
        reference_text: str | None = None,
        design_instruction: str | None = None,
        voice_bank: tuple[VoiceProfile, Path] | None = None,
    ) -> None:
        self._runtime = runtime
        self._admitted_model = _re_admit_omnivoice_model(admitted_model)
        if self._admitted_model is not None and model_id != self._admitted_model.model_id:
            raise ValueError("OmniVoice provider model_id does not match the admitted model")
        self._model_id = (
            self._admitted_model.model_id if self._admitted_model is not None else model_id
        )
        self._language = language
        self._seed = seed
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._mode = mode
        self._reference_audio_path = reference_audio_path
        self._reference_text = reference_text
        self._design_instruction = design_instruction
        self._voice_bank = voice_bank

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "OmniVoiceLocalTTSProvider":
        admitted_model = _admitted_omnivoice_model_from_environment()
        health = omnivoice_local_dependency_probe(admitted_model)
        if not health.available:
            return cls(**kwargs)
        assert admitted_model is not None
        if sys.platform.startswith("win"):
            install = admit_audio_cpp_native_package(
                Path(os.environ[NATIVE_AUDIO_CPP_EXECUTABLE_ENV]),
                required_model_paths=(admitted_model.model_path,),
            )
            transport = AudioCppNativeCLITransport(
                executable_path=install.executable_path,
                model_paths={OMNIVOICE_FAMILY: admitted_model.model_path},
            )
            binary_path: Path | None = install.executable_path
            build_hash = install.files[install.executable_path.name]
            transport_name = "native-cli"
        else:
            transport = AudioCppOmniVoiceCLITransport(
                model=admitted_model,
                container_command=_container_command_from_environment(),
            )
            binary_path = None
            build_hash = PINNED_AUDIO_CPP_OMNIVOICE_BINARY_SHA256
            transport_name = "container-cli"
        driver = AudioCppRuntimeDriver(
            binary_path=binary_path,
            source_revision=PINNED_AUDIO_CPP_REVISION,
            transport=transport,
            build_hash=build_hash,
            transport_name=transport_name,
        )
        return cls(
            LocalAudioRuntime(
                LocalRuntimeRegistry((driver,)),
                lifecycle=_omnivoice_gpu_lifecycle(),
            ),
            admitted_model=admitted_model,
            **kwargs,
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        if self._runtime is None:
            raise ModuleNotFoundError(OMNIVOICE_INSTALL_REMEDIATION)
        if self._admitted_model is None:
            raise RuntimeUnavailableError("OmniVoice runtime requires an admitted model")
        (
            omnivoice_mode,
            style_condition,
            design_instruction,
            reference_audio_path,
            reference_text,
        ) = self._mode_request_fields()
        response = self._runtime.execute_tts(
            LocalTTSRequest(
                request_id=uuid4().hex,
                family=OMNIVOICE_FAMILY,
                provider_id=self.provider_id,
                text=text,
                model_id=self._model_id,
                voice=None,
                language=self._language,
                text_chunk_size=OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE,
                seed=self._seed,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                omnivoice_mode=omnivoice_mode,
                style_condition=style_condition,
                design_instruction=design_instruction,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
            ),
            runtime_choice="audio-cpp",
        )
        if response.audio_bytes is None or response.audio_format != "wav":
            raise RuntimeUnavailableError("OmniVoice runtime did not return validated WAV audio")
        metadata: dict[str, Any] = {
            "provider": self.provider_id,
            "family": OMNIVOICE_FAMILY,
            "seed": self._seed,
            "voice_selection": self._voice_selection_metadata(),
            "voice_session": self._voice_session_metadata(),
            "sample_rate_hz": response.payload.get("sample_rate_hz"),
            "channels": response.payload.get("channels"),
            "duration_s": response.payload.get("duration_s"),
        }
        metadata["runtime_receipt"] = self._admitted_model.public_receipt()
        if response.receipt is not None:
            metadata["runtime"] = {
                "driver_id": response.receipt.driver_id,
                "transport": response.receipt.transport,
                "source_revision": response.receipt.source_revision,
                "build_hash": response.receipt.build_hash,
            }
        return SynthesisResult(
            audio_bytes=response.audio_bytes,
            audio_format="wav",
            transcript=text,
            client_path=self.provider_id,
            raw_metadata=metadata,
        )

    def for_voice_bank_profile(
        self, profile: VoiceProfile, reference_path: Path
    ) -> "OmniVoiceLocalTTSProvider":
        """Bind another admitted bank profile while reusing this run's runtime."""
        return OmniVoiceLocalTTSProvider(
            self._runtime,
            admitted_model=self._admitted_model,
            model_id=self._model_id,
            language=self._language,
            seed=self._seed,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            mode="preset",
            voice_bank=(profile, reference_path),
        )

    def _mode_request_fields(
        self,
    ) -> tuple[OmniVoiceMode, str | None, str | None, Path | str | None, str | None]:
        if self._mode == "auto":
            return ("auto", None, None, None, None)
        if self._mode == "preset" and self._voice_bank is not None:
            _profile, reference_path = self._voice_bank
            return ("clone", None, None, reference_path, _profile.reference_text)
        if self._mode == "clone":
            return (
                "clone",
                None,
                None,
                self._reference_audio_path,
                self._reference_text,
            )
        if self._mode == "design":
            return ("design", None, self._design_instruction, None, None)
        return ("fixed-style", OMNIVOICE_STYLE_CONDITION, None, None, None)

    def _voice_selection_metadata(self) -> dict[str, Any]:
        if self._mode == "auto":
            return {
                "kind": "auto-voice",
                "named_preset": False,
                "voice_cloning": False,
                "voice_design": False,
            }
        if self._mode == "preset" and self._voice_bank is not None:
            profile, _reference_path = self._voice_bank
            return {
                "kind": "bank-preset",
                "voice_id": profile.id,
                "voice_fingerprint": profile.reference_sha256,
            }
        if self._mode == "clone":
            return {
                "kind": "reference-clone",
                "named_preset": False,
                "voice_cloning": True,
                "voice_design": False,
            }
        if self._mode == "design":
            return {
                "kind": "design-instruction",
                "named_preset": False,
                "voice_cloning": False,
                "voice_design": True,
            }
        return {
            "kind": "built-in-style-condition",
            "condition": OMNIVOICE_STYLE_CONDITION,
            "named_preset": False,
            "voice_cloning": False,
            "voice_design": False,
        }

    def _voice_session_metadata(self) -> dict[str, Any]:
        if self._mode == "preset" and self._voice_bank is not None:
            strategy = "bank-preset-native-session"
        elif self._mode == "auto":
            strategy = "auto-voice-native-session"
        elif self._mode == "clone":
            strategy = "reference-isolated-native-session"
        elif self._mode == "design":
            strategy = "design-instruction-native-session"
        else:
            strategy = "single-native-invocation-internal-text-chunking"
        return {
            "strategy": strategy,
            "seed": self._seed,
            "internal_text_chunk_size": OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE,
        }


def _omnivoice_gpu_lifecycle() -> GPULifecycleOwner:
    return GPULifecycleOwner(
        GPULeaseManager(
            metadata_path=Path(tempfile.gettempdir()) / "voiceover-pipeline-gpu-lease.json"
        ),
        probe=probe_local_gpu_state,
        min_free_vram_mb=_OMNIVOICE_MIN_FREE_VRAM_MB,
        max_utilization_percent=_OMNIVOICE_MAX_GPU_UTILIZATION_PERCENT,
    )
