from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from voiceover_pipeline.audio_cpp.inventory import PINNED_AUDIO_CPP_REVISION
from voiceover_pipeline.local_runtime.contracts import (
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    RuntimeDriverHealth,
    RuntimeExecutionReceipt,
    RuntimeProtocolError,
)
from voiceover_pipeline.local_runtime.transports.subprocess import SubprocessJSONTransport


class AudioCppTransport(Protocol):
    def invoke(self, request_id: str, payload: Mapping[str, object]) -> Mapping[str, object]: ...

    def cancel(self, request_id: str) -> None: ...

    def close(self) -> None: ...


class AudioCppRuntimeDriver:
    """Strict JSON-only driver for the pinned audio.cpp source candidate."""

    driver_id = "audio-cpp"

    def __init__(
        self,
        *,
        binary_path: Path | None,
        source_revision: str,
        transport: AudioCppTransport | None = None,
        build_hash: str | None = None,
        transport_name: str = "subprocess-json",
    ) -> None:
        if source_revision != PINNED_AUDIO_CPP_REVISION:
            raise ValueError("audio.cpp source revision must match the pinned candidate")
        self._binary_path = binary_path
        self._source_revision = source_revision
        if binary_path is None and transport is None:
            raise ValueError("audio.cpp driver requires a binary or a transport")
        self._build_hash = build_hash
        self._transport = transport or SubprocessJSONTransport((str(binary_path),))
        self._transport_name = transport_name

    def health(self) -> RuntimeDriverHealth:
        if self._binary_path is None:
            return RuntimeDriverHealth(available=True)
        if not self._binary_path.is_file():
            return RuntimeDriverHealth(
                available=False,
                remediation="The pinned audio.cpp binary is not installed.",
            )
        return RuntimeDriverHealth(available=True)

    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse:
        raw_response = self._transport.invoke(request.request_id, self._to_audio_cpp_wire(request))
        schema_version = raw_response.get("schema_version")
        if schema_version != 1:
            raise RuntimeProtocolError("audio.cpp response has an unsupported schema version")
        if raw_response.get("request_id") != request.request_id:
            raise RuntimeProtocolError("audio.cpp response request_id did not match the request")
        if raw_response.get("ok") is not True:
            raise RuntimeProtocolError("audio.cpp response did not declare success")
        payload = raw_response.get("response")
        if not isinstance(payload, Mapping):
            raise RuntimeProtocolError("audio.cpp response payload must be an object")
        return LocalRuntimeResponse(
            request_id=request.request_id,
            payload=payload,
            receipt=RuntimeExecutionReceipt(
                driver_id=self.driver_id,
                transport=self._transport_name,
                source_revision=self._source_revision,
                build_hash=self._build_hash,
            ),
        )

    @staticmethod
    def _to_audio_cpp_wire(request: LocalRuntimeRequest) -> dict[str, object]:
        """Map a runtime-neutral request to the pinned VOP JSON adapter contract."""

        return request.to_wire()

    def cancel(self, request_id: str) -> None:
        self._transport.cancel(request_id)

    def close(self) -> None:
        self._transport.close()
