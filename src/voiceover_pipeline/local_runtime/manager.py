from __future__ import annotations

from collections.abc import Iterable

from voiceover_pipeline.local_runtime.contracts import (
    LocalASRRequest,
    LocalASRResponse,
    LocalAudioRuntimeDriver,
    LocalRuntimeRequest,
    LocalRuntimeResponse,
    LocalTTSRequest,
    LocalTTSResponse,
    RuntimeProtocolError,
    RuntimeUnavailableError,
)
from voiceover_pipeline.local_runtime.registry import LocalRuntimeRegistry


class LocalAudioRuntime:
    """Runtime choice and rollback seam; GPU leasing arrives in a later slice."""

    def __init__(
        self,
        registry: LocalRuntimeRegistry,
        *,
        promoted_families: Iterable[str] = (),
    ) -> None:
        self._registry = registry
        self._promoted_families = frozenset(promoted_families)
        self._active_by_family: dict[str, LocalAudioRuntimeDriver] = {}

    def _healthy_driver(self, driver_id: str) -> LocalAudioRuntimeDriver:
        driver = self._registry.get(driver_id)
        health = driver.health()
        if not health.available:
            detail = f": {health.remediation}" if health.remediation else ""
            raise RuntimeUnavailableError(f"Local runtime {driver_id} is unavailable{detail}")
        return driver

    def _select(self, family: str, runtime_choice: str) -> LocalAudioRuntimeDriver:
        if runtime_choice == "auto":
            if family not in self._promoted_families:
                return self._healthy_driver("python")
            try:
                return self._healthy_driver("audio-cpp")
            except RuntimeUnavailableError:
                return self._healthy_driver("python")
        return self._healthy_driver(runtime_choice)

    def execute(self, request: LocalRuntimeRequest, *, runtime_choice: str = "auto") -> LocalRuntimeResponse:
        driver = self._select(request.family, runtime_choice)
        response = driver.invoke(request)
        if response.request_id != request.request_id:
            raise RuntimeProtocolError("Local runtime response request_id did not match the request")
        self._active_by_family[request.family] = driver
        return response

    def execute_asr(self, request: LocalASRRequest, *, runtime_choice: str = "auto") -> LocalASRResponse:
        return LocalASRResponse.from_runtime_response(self.execute(request.to_runtime_request(), runtime_choice=runtime_choice))

    def execute_tts(self, request: LocalTTSRequest, *, runtime_choice: str = "auto") -> LocalTTSResponse:
        return LocalTTSResponse.from_runtime_response(self.execute(request.to_runtime_request(), runtime_choice=runtime_choice))

    def cancel(self, request_id: str, *, family: str) -> None:
        try:
            driver = self._active_by_family[family]
        except KeyError as exc:
            raise RuntimeUnavailableError(f"No active local runtime for family: {family}") from exc
        driver.cancel(request_id)

    def unload(self, family: str | None = None) -> None:
        if family is not None:
            driver = self._active_by_family.pop(family, None)
            if driver is not None and driver not in self._active_by_family.values():
                driver.close()
            return
        active_drivers = {id(driver): driver for driver in self._active_by_family.values()}
        self._active_by_family.clear()
        for driver in active_drivers.values():
            driver.close()
