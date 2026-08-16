from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Protocol


RuntimeOperation = Literal["asr", "tts"]
RuntimeChoice = Literal["python", "audio-cpp", "auto"]


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

    def to_runtime_request(self) -> LocalRuntimeRequest:
        return LocalRuntimeRequest(
            request_id=self.request_id,
            operation="asr",
            family=self.family,
            provider_id=self.provider_id,
            payload={
                "audio_path": str(self.audio_path),
                "model_id": self.model_id,
                "language": self.language,
                "timestamp_mode": self.timestamp_mode,
                "context_text": self.context_text,
            },
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
class LocalTTSRequest:
    request_id: str
    family: str
    provider_id: str
    text: str
    model_id: str
    voice: str | None = None

    def to_runtime_request(self) -> LocalRuntimeRequest:
        return LocalRuntimeRequest(
            request_id=self.request_id,
            operation="tts",
            family=self.family,
            provider_id=self.provider_id,
            payload={"text": self.text, "model_id": self.model_id, "voice": self.voice},
        )


@dataclass(frozen=True)
class LocalTTSResponse:
    audio_path: str
    payload: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_runtime_response(cls, response: LocalRuntimeResponse) -> "LocalTTSResponse":
        audio_path = response.payload.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise RuntimeProtocolError("Local TTS response has no audio_path")
        return cls(audio_path=audio_path, payload=response.payload)


class LocalAudioRuntimeDriver(Protocol):
    driver_id: str

    def health(self) -> RuntimeDriverHealth: ...

    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse: ...

    def cancel(self, request_id: str) -> None: ...

    def close(self) -> None: ...
