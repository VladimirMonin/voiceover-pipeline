from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Literal


PINNED_AUDIO_CPP_REVISION = "502b5b74bd26e9b4aed267d1776ecf131cae7215"
AudioCppBackend = Literal["cpu", "cuda", "hip", "vulkan", "metal"]


@dataclass(frozen=True)
class AudioCppFamilyInventory:
    family: str
    provider_id: str
    model_id: str
    timestamp_origin: Literal["none", "native", "forced"]
    prompt_contract: str
    promotion_state: Literal["inventory-only", "promoted"] = "inventory-only"
    model_sha256: str | None = None
    quantization: str | None = None
    license: str | None = None
    provenance: str | None = None


AUDIO_CPP_FAMILY_INVENTORY: tuple[AudioCppFamilyInventory, ...] = (
    AudioCppFamilyInventory(
        family="qwen3-asr",
        provider_id="qwen-local",
        model_id="Qwen/Qwen3-ASR-0.6B",
        timestamp_origin="none",
        prompt_contract="free contextual text",
        license="not-recorded-until-artifact-install",
        provenance="Qwen official model identifier; no model artifact is installed",
    ),
    AudioCppFamilyInventory(
        family="qwen3-forced-aligner",
        provider_id="qwen-local",
        model_id="Qwen/Qwen3-ForcedAligner-0.6B",
        timestamp_origin="forced",
        prompt_contract="alignment-only companion to qwen3-asr",
        license="not-recorded-until-artifact-install",
        provenance="Qwen official model identifier; no model artifact is installed",
    ),
    AudioCppFamilyInventory(
        family="nemotron-3.5-asr",
        provider_id="nemotron-local",
        model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        timestamp_origin="native",
        prompt_contract="typed language/task dictionary; phrase hints unavailable",
        license="not-recorded-until-artifact-install",
        provenance="NVIDIA model identifier; no model artifact is installed",
    ),
    AudioCppFamilyInventory(
        family="qwen3-tts",
        provider_id="qwen-local",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        timestamp_origin="none",
        prompt_contract="typed voice and instruction fields",
        license="not-recorded-until-artifact-install",
        provenance="Qwen model identifier; no model artifact is installed",
    ),
    AudioCppFamilyInventory(
        family="omnivoice",
        provider_id="omnivoice-local",
        model_id="not-selected",
        timestamp_origin="none",
        prompt_contract="typed voice and style fields",
        license="not-recorded-until-artifact-install",
        provenance="model selection and license gate remain pending",
    ),
)


@dataclass(frozen=True)
class AudioCppBuildPlan:
    source_dir: Path
    source_revision: str
    build_dir: Path
    backend: AudioCppBackend
    compiler: str = "c++"
    cmake_definitions: tuple[tuple[str, str], ...] = ()
    supported_backends: tuple[AudioCppBackend, ...] = field(
        default=("cpu", "cuda", "hip", "vulkan", "metal"),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.source_revision != PINNED_AUDIO_CPP_REVISION:
            raise ValueError("audio.cpp build plan source revision must match the pinned candidate")
        if self.backend not in self.supported_backends:
            raise ValueError(f"Unsupported audio.cpp backend: {self.backend}")
        if not self.compiler.strip():
            raise ValueError("audio.cpp compiler must not be blank")
        source_dir = self.source_dir.resolve()
        build_dir = self.build_dir.resolve()
        if build_dir.is_relative_to(source_dir):
            raise ValueError("audio.cpp build output must live outside the source tree")
        definitions = tuple(self.cmake_definitions)
        if len({name for name, _value in definitions}) != len(definitions):
            raise ValueError("audio.cpp CMake definition names must be unique")
        if any(not name.strip() or not value.strip() for name, value in definitions):
            raise ValueError("audio.cpp CMake definition names and values must not be blank")
        if any(name == "ENGINE_ENABLE_CUDA" for name, _value in definitions):
            raise ValueError("audio.cpp ENGINE_ENABLE_CUDA is derived from the selected backend")
        object.__setattr__(self, "source_dir", source_dir)
        object.__setattr__(self, "build_dir", build_dir)
        object.__setattr__(self, "cmake_definitions", definitions)

    def cmake_command(self) -> tuple[str, ...]:
        command = (
            "cmake",
            "-S",
            str(self.source_dir),
            "-B",
            str(self.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={self.compiler}",
        )
        backend_definitions = (
            ("ENGINE_ENABLE_CUDA", "ON" if self.backend == "cuda" else "OFF"),
        )
        return command + tuple(
            f"-D{name}={value}" for name, value in (*backend_definitions, *self.cmake_definitions)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_dir": str(self.source_dir),
            "source_revision": self.source_revision,
            "build_dir": str(self.build_dir),
            "backend": self.backend,
            "compiler": self.compiler,
            "cmake_definitions": dict(self.cmake_definitions),
            "cmake_command": list(self.cmake_command()),
            "supported_backends": list(self.supported_backends),
            "mlx": "not-installed-not-implemented",
        }


@dataclass(frozen=True)
class AudioCppBuildReceipt:
    source_revision: str
    backend: AudioCppBackend
    compiler: str
    binary_path: str
    binary_sha256: str
    model_families: tuple[str, ...]
    cmake_definitions: tuple[tuple[str, str], ...]


def find_family_inventory(family: str) -> AudioCppFamilyInventory:
    for item in AUDIO_CPP_FAMILY_INVENTORY:
        if item.family == family:
            return item
    raise ValueError(f"Unknown audio.cpp model family: {family}")


def probe_audio_cpp_binary(binary_path: Path) -> tuple[bool, str]:
    """A doctor-safe probe: inspect only the binary path, never model weights."""
    if binary_path.is_file():
        return True, ""
    return False, "The pinned audio.cpp binary is not installed."


def inspect_pinned_source(source_dir: Path) -> tuple[str, bool]:
    """Read Git metadata only; never configure, build, or load model weights."""
    try:
        revision = subprocess.run(
            ("git", "-C", str(source_dir), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(source_dir), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("audio.cpp source checkout is not a readable Git worktree") from exc
    if revision != PINNED_AUDIO_CPP_REVISION:
        raise ValueError("audio.cpp source checkout does not match the pinned candidate")
    return revision, bool(status.strip())


def build_receipt(
    plan: AudioCppBuildPlan,
    *,
    binary_path: Path,
    expected_binary_sha256: str | None = None,
) -> AudioCppBuildReceipt:
    source_revision, source_dirty = inspect_pinned_source(plan.source_dir)
    if source_revision != plan.source_revision:
        raise ValueError("audio.cpp source checkout does not match the build plan")
    if source_dirty:
        raise ValueError("audio.cpp build receipt refuses a dirty source tree")
    if not binary_path.is_file():
        raise ValueError("audio.cpp build receipt requires an existing binary")
    digest = sha256(binary_path.read_bytes()).hexdigest()
    if expected_binary_sha256 is not None and digest != expected_binary_sha256:
        raise ValueError("audio.cpp build receipt binary hash did not match the expected hash")
    return AudioCppBuildReceipt(
        source_revision=plan.source_revision,
        backend=plan.backend,
        compiler=plan.compiler,
        binary_path=str(binary_path.resolve()),
        binary_sha256=digest,
        model_families=tuple(item.family for item in AUDIO_CPP_FAMILY_INVENTORY),
        cmake_definitions=plan.cmake_definitions,
    )
