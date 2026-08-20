from __future__ import annotations

from collections.abc import Iterable

from voiceover_pipeline.local_runtime.contracts import (
    LocalAudioRuntimeDriver,
    RuntimeDriverNotFoundError,
)


class LocalRuntimeRegistry:
    """Finite driver registry; selecting a driver never starts inference."""

    def __init__(self, drivers: Iterable[LocalAudioRuntimeDriver] = ()) -> None:
        driver_list = tuple(drivers)
        self._drivers = {driver.driver_id: driver for driver in driver_list}
        if len(self._drivers) != len(driver_list):
            raise ValueError("Local runtime driver IDs must be unique")

    def get(self, driver_id: str) -> LocalAudioRuntimeDriver:
        try:
            return self._drivers[driver_id]
        except KeyError as exc:
            raise RuntimeDriverNotFoundError(f"Unknown local runtime driver: {driver_id}") from exc

    def driver_ids(self) -> tuple[str, ...]:
        return tuple(self._drivers)
