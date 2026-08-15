import pytest

from voiceover_pipeline.models import ASRCapabilities, ASRRequest, ASRResult
from voiceover_pipeline.providers.asr_registry import (
    ASRDependencyHealth,
    ASRProviderNotFoundError,
    ASRProviderRegistry,
    ASRProviderSpec,
)


class DummyASRProvider:
    provider_id = "fixture-local"

    def transcribe(self, request: ASRRequest) -> ASRResult:
        raise AssertionError("Registry tests must not execute provider inference")


def _spec(*, probe=lambda: ASRDependencyHealth(available=True, remediation="")) -> ASRProviderSpec:
    return ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=DummyASRProvider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            contextual_bias=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=probe,
    )


def test_registry_lists_declared_capabilities_without_constructing_provider():
    constructed = False

    def factory():
        nonlocal constructed
        constructed = True
        return DummyASRProvider()

    spec = ASRProviderSpec(
        provider_id="fixture-local",
        description="Offline fixture provider",
        factory=factory,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            contextual_bias=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    registry = ASRProviderRegistry((spec,))

    assert registry.provider_ids() == ("fixture-local",)
    assert registry.listing() == [
        {
            "id": "fixture-local",
            "description": "Offline fixture provider",
            "models": [{"id": "fixture-model", "default": True}],
            "capabilities": {
                "batch_audio": True,
                "streaming": False,
                "forced_language": False,
                "contextual_bias": True,
                "phrase_boosting": False,
                "segment_timestamps": False,
                "word_timestamps": False,
                "forced_alignment": False,
                "confidence": False,
                "device_modes": ["cpu"],
                "compute_modes": ["float32"],
            },
        }
    ]
    assert constructed is False


def test_registry_unknown_provider_fails_closed_without_fallback():
    registry = ASRProviderRegistry((_spec(),))

    with pytest.raises(ASRProviderNotFoundError, match="Unknown ASR provider: missing"):
        registry.get("missing")


def test_dependency_probe_returns_redacted_remedy_without_factory_execution():
    registry = ASRProviderRegistry(
        (_spec(probe=lambda: ASRDependencyHealth(
            available=False,
            remediation="Install the approved optional ASR runtime.",
        )),)
    )

    health = registry.get("fixture-local").dependency_probe()

    assert health.available is False
    assert health.remediation == "Install the approved optional ASR runtime."
