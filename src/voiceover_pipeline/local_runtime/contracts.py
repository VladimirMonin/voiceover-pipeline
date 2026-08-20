from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Protocol

from voiceover_pipeline.models import ASRRuntimeChoice

RuntimeOperation = Literal["asr", "tts"]
RuntimeChoice = ASRRuntimeChoice
OmniVoiceMode = Literal["fixed-style", "clone", "design"]


class RuntimeErrorBase(RuntimeError):
    """Base error for the local runtime seam; messages exclude private payloads."""


class RuntimeDriverNotFoundError(RuntimeErrorBase):
    pass


class RuntimeUnavailableError(RuntimeErrorBase):
    pass


class RuntimeProtocolError(RuntimeErrorBase):
    pass


class RuntimeTransportError(RuntimeErrorBase):
    pass


@dataclass(frozen=True)
class RuntimeDriverHealth:
    available: bool
    remediation: str = ""
    reason_code: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionReceipt:
    driver_id: str
    transport: str
    source_revision: str | None = None
    build_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.driver_id.strip():
            raise ValueError("Local runtime driver_id must not be blank")
        if not self.transport.strip():
            raise ValueError("Local runtime transport must not be blank")


@dataclass(frozen=True)
class LocalRuntimeRequest:
    request_id: str
    operation: RuntimeOperation
    family: str
    provider_id: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("Local runtime request_id must not be blank")
        if self.operation not in ("asr", "tts"):
            raise ValueError("Local runtime operation must be asr or tts")
        if not self.family.strip():
            raise ValueError("Local runtime family must not be blank")
        if not self.provider_id.strip():
            raise ValueError("Local runtime provider_id must not be blank")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "operation": self.operation,
            "family": self.family,
            "provider_id": self.provider_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class LocalRuntimeResponse:
    request_id: str
    payload: Mapping[str, object] = field(default_factory=dict)
    receipt: RuntimeExecutionReceipt | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("Local runtime response request_id must not be blank")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class LocalASRRequest:
    request_id: str
    family: str
    provider_id: str
    audio_path: Path | str
    model_id: str
    language: str | None = None
    timestamp_mode: Literal["none", "word"] = "none"
    context_text: str | None = None
    provider_options: Mapping[str, object] = field(default_factory=dict)
    runtime_choice: RuntimeChoice = "auto"

    def __post_init__(self) -> None:
        if self.runtime_choice not in ("auto", "python", "audio-cpp"):
            raise ValueError("Local ASR runtime choice must be auto, python, or audio-cpp")
        object.__setattr__(self, "provider_options", MappingProxyType(dict(self.provider_options)))

    def to_runtime_request(self) -> LocalRuntimeRequest:
        payload: dict[str, object] = {
            "audio_path": str(self.audio_path),
            "model_id": self.model_id,
            "language": self.language,
            "timestamp_mode": self.timestamp_mode,
            "context_text": self.context_text,
        }
        payload.update(self.provider_options)
        return LocalRuntimeRequest(
            request_id=self.request_id,
            operation="asr",
            family=self.family,
            provider_id=self.provider_id,
            payload=payload,
        )


@dataclass(frozen=True)
class LocalASRResponse:
    transcript: str
    language: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_runtime_response(cls, response: LocalRuntimeResponse) -> "LocalASRResponse":
        transcript = response.payload.get("transcript")
        if not isinstance(transcript, str):
            raise RuntimeProtocolError("Local ASR response has no transcript")
        language = response.payload.get("language", "")
        if not isinstance(language, str):
            raise RuntimeProtocolError("Local ASR response language must be a string")
        return cls(transcript=transcript, language=language, payload=response.payload)


@dataclass(frozen=True)
class OmniVoiceRequest:
    """Mode-only contract kept separate from Qwen's runtime mode vocabulary."""

    mode: OmniVoiceMode = "fixed-style"
    style_condition: str | None = None
    instruction: str | None = None
    reference_audio_path: Path | str | None = None
    reference_text: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("fixed-style", "clone", "design"):
            raise ValueError("OmniVoice mode must be fixed-style, clone, or design")

        has_style_condition = self.style_condition is not None and self.style_condition.strip()
        has_instruction = self.instruction is not None and self.instruction.strip()
        has_reference_audio = (
            self.reference_audio_path is not None and str(self.reference_audio_path).strip()
        )
        has_reference_text = self.reference_text is not None and self.reference_text.strip()

        if self.mode == "clone":
            if not has_reference_audio:
                raise ValueError("OmniVoice clone mode requires reference audio")
            if not has_reference_text:
                raise ValueError("OmniVoice clone mode requires non-empty reference text")
            if has_style_condition or has_instruction:
                raise ValueError("OmniVoice clone mode does not accept style or design fields")
        elif self.mode == "design":
            if not has_instruction:
                raise ValueError("OmniVoice design mode requires non-empty instruction")
            if has_style_condition or has_reference_audio or has_reference_text:
                raise ValueError(
                    "OmniVoice design mode does not accept fixed-style or clone fields"
                )
        elif has_reference_audio or has_reference_text or has_instruction:
            raise ValueError("OmniVoice fixed-style mode does not accept clone/design-only fields")


@dataclass(frozen=True)
class LocalTTSRequest:
    request_id: str
    family: str
    provider_id: str
    text: str
    model_id: str
    model_artifact_path: Path | str | None = None
    voice: str | None = None
    language: str | None = None
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    mode: Literal["custom-voice", "voice-clone", "voice-design"] | None = None
    instruction: str | None = None
    text_chunk_size: int | None = None
    reference_audio_path: Path | str | None = None
    reference_text: str | None = field(default=None, repr=False)
    omnivoice_mode: OmniVoiceMode | None = None
    style_condition: str | None = None
    design_instruction: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.omnivoice_mode is None:
            return
        if self.omnivoice_mode not in ("fixed-style", "clone", "design"):
            raise ValueError("OmniVoice mode must be fixed-style, clone, or design")
        if self.instruction is not None:
            raise ValueError("OmniVoice requests do not accept the Qwen instruction field")
        has_style_condition = self.style_condition is not None and self.style_condition.strip()
        has_design_instruction = (
            self.design_instruction is not None and self.design_instruction.strip()
        )
        has_reference_audio = (
            self.reference_audio_path is not None and str(self.reference_audio_path).strip()
        )
        has_reference_text = self.reference_text is not None and self.reference_text.strip()
        if self.omnivoice_mode == "clone":
            if not has_reference_audio:
                raise ValueError("OmniVoice clone mode requires reference audio")
            if not has_reference_text:
                raise ValueError("OmniVoice clone mode requires non-empty reference text")
            if has_style_condition or has_design_instruction:
                raise ValueError("OmniVoice clone mode does not accept style or design fields")
        elif self.omnivoice_mode == "design":
            if not has_design_instruction:
                raise ValueError("OmniVoice design mode requires non-empty instruction")
            if has_style_condition or has_reference_audio or has_reference_text:
                raise ValueError(
                    "OmniVoice design mode does not accept fixed-style or clone fields"
                )
        elif has_reference_audio or has_reference_text or has_design_instruction:
            raise ValueError("OmniVoice fixed-style mode does not accept clone/design-only fields")

    def to_runtime_request(self) -> LocalRuntimeRequest:
        payload: dict[str, object] = {
            "text": self.text,
            "model_id": self.model_id,
            "voice": self.voice,
        }
        if self.model_artifact_path is not None:
            payload["model_artifact_path"] = str(self.model_artifact_path)
        if self.language is not None:
            payload["language"] = self.language
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.num_inference_steps is not None:
            payload["num_inference_steps"] = self.num_inference_steps
        if self.guidance_scale is not None:
            payload["guidance_scale"] = self.guidance_scale
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.instruction is not None:
            payload["instruction"] = self.instruction
        if self.text_chunk_size is not None:
            payload["text_chunk_size"] = self.text_chunk_size
        if self.reference_audio_path is not None:
            payload["reference_audio_path"] = str(self.reference_audio_path)
        if self.reference_text is not None:
            payload["reference_text"] = self.reference_text
        if self.omnivoice_mode is not None:
            payload["omnivoice_mode"] = self.omnivoice_mode
        if self.style_condition is not None:
            payload["style_condition"] = self.style_condition
        if self.design_instruction is not None:
            payload["design_instruction"] = self.design_instruction
        return LocalRuntimeRequest(
            request_id=self.request_id,
            operation="tts",
            family=self.family,
            provider_id=self.provider_id,
            payload=payload,
        )


@dataclass(frozen=True)
class LocalTTSResponse:
    audio_path: str | None = None
    audio_bytes: bytes | None = None
    audio_format: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    receipt: RuntimeExecutionReceipt | None = None

    @classmethod
    def from_runtime_response(cls, response: LocalRuntimeResponse) -> "LocalTTSResponse":
        audio_path = response.payload.get("audio_path")
        if audio_path is not None and (not isinstance(audio_path, str) or not audio_path):
            raise RuntimeProtocolError("Local TTS response has no audio_path")
        audio_bytes = response.payload.get("audio_bytes")
        if audio_bytes is not None and (not isinstance(audio_bytes, bytes) or not audio_bytes):
            raise RuntimeProtocolError("Local TTS response audio_bytes must be non-empty bytes")
        if audio_path is None and audio_bytes is None:
            raise RuntimeProtocolError("Local TTS response has no audio_path or audio_bytes")
        audio_format = response.payload.get("audio_format")
        if audio_bytes is not None and (not isinstance(audio_format, str) or not audio_format):
            raise RuntimeProtocolError("Local TTS response audio_bytes require an audio_format")
        return cls(
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            audio_format=audio_format if isinstance(audio_format, str) else None,
            payload=response.payload,
            receipt=response.receipt,
        )


class LocalAudioRuntimeDriver(Protocol):
    driver_id: str

    def health(self) -> RuntimeDriverHealth: ...

    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse: ...

    def cancel(self, request_id: str) -> None: ...

    def close(self) -> None: ...
