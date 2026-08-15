from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from voiceover_pipeline.models import ASRCapabilities
from voiceover_pipeline.providers.base import ASRProvider


@dataclass(frozen=True)
class ASRDependencyHealth:
    available: bool
    remediation: str


class ASRProviderNotFoundError(ValueError):
    def __init__(self, provider_id: str):
        super().__init__(f"Unknown ASR provider: {provider_id}")
        self.provider_id = provider_id


@dataclass(frozen=True)
class ASRProviderSpec:
    provider_id: str
    description: str
    factory: Callable[[], ASRProvider]
    models: tuple[dict[str, Any], ...]
    capabilities: ASRCapabilities
    dependency_probe: Callable[[], ASRDependencyHealth]

    def listing(self) -> dict[str, Any]:
        capabilities = self.capabilities
        return {
            "id": self.provider_id,
            "description": self.description,
            "models": [dict(model) for model in self.models],
            "capabilities": {
                "batch_audio": capabilities.batch_audio,
                "streaming": capabilities.streaming,
                "forced_language": capabilities.forced_language,
                "contextual_bias": capabilities.contextual_bias,
                "phrase_boosting": capabilities.phrase_boosting,
                "segment_timestamps": capabilities.segment_timestamps,
                "word_timestamps": capabilities.word_timestamps,
                "forced_alignment": capabilities.forced_alignment,
                "confidence": capabilities.confidence,
                "device_modes": list(capabilities.device_modes),
                "compute_modes": list(capabilities.compute_modes),
            },
        }


class ASRProviderRegistry:
    def __init__(self, specs: Iterable[ASRProviderSpec] = ()) -> None:
        spec_list = tuple(specs)
        self._specs = {spec.provider_id: spec for spec in spec_list}
        if len(self._specs) != len(spec_list):
            raise ValueError("ASR provider IDs must be unique")

    def get(self, provider_id: str) -> ASRProviderSpec:
        try:
            return self._specs[provider_id]
        except KeyError as exc:
            raise ASRProviderNotFoundError(provider_id) from exc

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ASRProviderSpec, ...]:
        return tuple(self._specs.values())

    def listing(self) -> list[dict[str, Any]]:
        return [spec.listing() for spec in self.specs()]


def _default_asr_provider_specs() -> tuple[ASRProviderSpec, ...]:
    # Adapter modules are safe to import here: optional runtimes stay inside
    # their selected adapter/probe paths.
    from voiceover_pipeline.providers.nemotron_asr_local import NEMOTRON_ASR_PROVIDER_SPEC
    from voiceover_pipeline.providers.qwen_asr_local import QWEN_ASR_PROVIDER_SPEC

    return (QWEN_ASR_PROVIDER_SPEC, NEMOTRON_ASR_PROVIDER_SPEC)


ASR_PROVIDER_REGISTRY = ASRProviderRegistry(_default_asr_provider_specs())


def get_asr_provider_spec(provider_id: str) -> ASRProviderSpec:
    return ASR_PROVIDER_REGISTRY.get(provider_id)


def list_asr_provider_specs() -> list[ASRProviderSpec]:
    return list(ASR_PROVIDER_REGISTRY.specs())
