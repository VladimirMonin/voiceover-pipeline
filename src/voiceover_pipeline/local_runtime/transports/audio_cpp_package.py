from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, Mapping

from voiceover_pipeline.audio_cpp.inventory import PINNED_AUDIO_CPP_REVISION

NATIVE_AUDIO_CPP_CLOSURE_MANIFEST: Final = "audio_cpp_dependency_closure.json"
NATIVE_AUDIO_CPP_BUILD_RECEIPT: Final = "build_receipt.json"
_STREAMING_HASH_CHUNK_BYTES: Final = 1 << 20
_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_revision",
        "backend",
        "compiler",
        "cmake_version",
        "cuda_toolkit_version",
        "architecture",
        "build_flags",
        "binary_sha256",
        "model_families",
    }
)
_HEX_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTED_ARCHITECTURES: Final = frozenset({"x86_64", "amd64"})
_ABSOLUTE_TOKEN_RE: Final = re.compile(r"(^|[/\\\s=])([A-Za-z]:[\\/]|[\\/]{2})|\.\.[\\/]|:\\")

NativeRejectionCode = Literal[
    "malformed_manifest",
    "missing_executable",
    "missing_dll_closure",
    "missing_model_artifact",
    "modified_bytes",
    "path_escape",
    "missing_receipt",
    "invalid_receipt",
    "unexpected_extra_files",
    "unsupported_schema_version",
]


@dataclass(frozen=True)
class NativeAudioCppInstall:
    """A checked host-native audio.cpp executable and its dependency closure."""

    executable_path: Path
    closure_manifest_path: Path
    files: Mapping[str, str]
    source_revision: str
    backend: str
    architecture: str
    model_families: tuple[str, ...]


@dataclass(frozen=True)
class AudioCppPackageReceipt:
    """Structural facts from the build receipt that the package checker needs."""

    source_revision: str
    backend: str
    architecture: str
    binary_sha256: str
    model_families: tuple[str, ...]


@dataclass(frozen=True)
class AudioCppPackageError(Exception):
    """Typed package admission rejection with a non-private reason."""

    code: NativeRejectionCode
    message: str

    def __str__(self) -> str:
        return self.message


class AudioCppPackageAdmissionError(AudioCppPackageError, ValueError):
    """Admission failure is a ValueError so existing callers fail closed."""


def stream_sha256(path: Path) -> str:
    """Hash a file in bounded chunks instead of loading it into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_STREAMING_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _reject(code: NativeRejectionCode, message: str) -> None:
    raise AudioCppPackageAdmissionError(code=code, message=message)


def _require_relative_sibling(package_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        _reject("malformed_manifest", "audio.cpp native dependency closure manifest is invalid")
    candidate = package_root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(package_root.resolve())
    except ValueError:
        _reject("path_escape", "audio.cpp native dependency closure escapes the package root")
    return candidate


def _require_inside(package_root: Path, candidate: Path) -> Path:
    if ".." in candidate.parts:
        _reject("path_escape", "audio.cpp native model is outside the checked package")
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError:
        _reject("path_escape", "audio.cpp native model is outside the checked package")
    return resolved


def _load_manifest(executable: Path) -> tuple[dict[str, str], Path]:
    package_root = executable.parent
    manifest_path = package_root / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _reject(
            "malformed_manifest",
            "audio.cpp native dependency closure manifest is unavailable or malformed",
        )
    if not isinstance(document, Mapping):
        _reject("malformed_manifest", "audio.cpp native dependency closure manifest is invalid")
    if document.get("schema_version") != 1:
        _reject(
            "unsupported_schema_version",
            "audio.cpp native dependency closure manifest has an unsupported schema version",
        )
    raw_files = document.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        _reject("malformed_manifest", "audio.cpp native dependency closure manifest has no files")
    files: dict[str, str] = {}
    for relative_path, expected_digest in raw_files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            _reject("malformed_manifest", "audio.cpp native dependency closure manifest is invalid")
        if _HEX_DIGEST_RE.fullmatch(expected_digest.casefold()) is None:
            _reject("malformed_manifest", "audio.cpp native dependency closure manifest is invalid")
        _require_relative_sibling(package_root, relative_path)
        files[Path(relative_path).as_posix()] = expected_digest.casefold()
    return files, manifest_path


def _validate_manifest_files(executable: Path, files: dict[str, str]) -> None:
    package_root = executable.parent
    for relative_path, expected_digest in files.items():
        candidate = _require_relative_sibling(package_root, relative_path)
        if candidate.is_symlink():
            _reject("path_escape", "audio.cpp native dependency closure contains a symlink")
        if not candidate.is_file():
            _reject(
                "missing_dll_closure",
                "audio.cpp native dependency closure is incomplete",
            )
        actual_digest = stream_sha256(candidate)
        if actual_digest != expected_digest:
            _reject("modified_bytes", "audio.cpp native dependency closure checksum did not match")
    if executable.name not in files:
        _reject(
            "malformed_manifest",
            "audio.cpp native dependency closure does not cover the executable",
        )
    if not any(relative_path.casefold().endswith(".dll") for relative_path in files):
        _reject(
            "missing_dll_closure",
            "audio.cpp native dependency closure has no DLL dependencies",
        )


def _validate_receipt(executable: Path) -> AudioCppPackageReceipt:
    receipt_path = executable.parent / NATIVE_AUDIO_CPP_BUILD_RECEIPT
    if not receipt_path.is_file():
        _reject("missing_receipt", "audio.cpp native build receipt is unavailable")
    try:
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _reject("invalid_receipt", "audio.cpp native build receipt is malformed")
    if not isinstance(document, Mapping):
        _reject("invalid_receipt", "audio.cpp native build receipt is invalid")
    unknown = set(document) - _RECEIPT_FIELDS
    if unknown or document.get("schema_version") != 1:
        _reject("invalid_receipt", "audio.cpp native build receipt is invalid")
    source_revision = document.get("source_revision")
    backend = document.get("backend")
    architecture = document.get("architecture")
    compiler = document.get("compiler")
    cmake_version = document.get("cmake_version")
    cuda_toolkit_version = document.get("cuda_toolkit_version")
    build_flags = document.get("build_flags")
    binary_sha256 = document.get("binary_sha256")
    model_families = document.get("model_families")
    if (
        source_revision != PINNED_AUDIO_CPP_REVISION
        or not isinstance(backend, str)
        or not backend.strip()
        or not isinstance(architecture, str)
        or architecture.casefold() not in _ACCEPTED_ARCHITECTURES
        or not isinstance(compiler, str)
        or not compiler.strip()
        or not isinstance(cmake_version, str)
        or not cmake_version.strip()
        or not isinstance(cuda_toolkit_version, str)
        or not cuda_toolkit_version.strip()
        or not isinstance(build_flags, str)
        or not build_flags.strip()
        or not isinstance(binary_sha256, str)
        or _HEX_DIGEST_RE.fullmatch(binary_sha256.casefold()) is None
        or not isinstance(model_families, list)
        or not model_families
        or not all(isinstance(family, str) and family.strip() for family in model_families)
        or len({family.casefold() for family in model_families}) != len(model_families)
    ):
        _reject("invalid_receipt", "audio.cpp native build receipt is invalid")
    for field in (
        "source_revision",
        "backend",
        "compiler",
        "cmake_version",
        "cuda_toolkit_version",
        "architecture",
        "build_flags",
    ):
        value = document.get(field)
        if isinstance(value, str) and _ABSOLUTE_TOKEN_RE.search(value):
            _reject("invalid_receipt", "audio.cpp native build receipt leaks an absolute path")
    actual = stream_sha256(executable)
    if actual != binary_sha256.casefold():
        _reject("invalid_receipt", "audio.cpp native build receipt checksum did not match")
    return AudioCppPackageReceipt(
        source_revision=source_revision,
        backend=backend,
        architecture=architecture,
        binary_sha256=binary_sha256.casefold(),
        model_families=tuple(model_families),
    )


def _validate_model_path(package_root: Path, model_path: Path, files: Mapping[str, str]) -> None:
    resolved = _require_inside(package_root, model_path)
    if resolved.is_symlink():
        _reject("path_escape", "audio.cpp native model artifact escapes the package root")
    if resolved.is_file():
        key = resolved.relative_to(package_root).as_posix()
        if key not in files:
            _reject(
                "missing_model_artifact",
                "audio.cpp native dependency closure does not cover a required model",
            )
        return
    if not resolved.is_dir():
        _reject("missing_model_artifact", "audio.cpp native model artifact is unavailable")
    members = [member for member in resolved.rglob("*") if member.is_file() or member.is_symlink()]
    if not members:
        _reject("missing_model_artifact", "audio.cpp native model directory is empty")
    for member in members:
        if member.is_symlink():
            _reject("path_escape", "audio.cpp native model directory contains a symlink")
        key = member.relative_to(package_root).as_posix()
        if key not in files:
            _reject(
                "missing_model_artifact",
                "audio.cpp native dependency closure does not cover a required model",
            )


def _scan_unexpected_files(
    package_root: Path,
    files: Mapping[str, str],
    model_dirs: tuple[Path, ...],
) -> None:
    declared = set(files)
    model_prefixes = [tuple(model_dir.relative_to(package_root).parts) for model_dir in model_dirs]
    for member in package_root.rglob("*"):
        if not member.is_file() and not member.is_symlink():
            continue
        relative = member.relative_to(package_root)
        key = relative.as_posix()
        if key == NATIVE_AUDIO_CPP_CLOSURE_MANIFEST or key == NATIVE_AUDIO_CPP_BUILD_RECEIPT:
            continue
        if key in declared:
            continue
        parts = tuple(relative.parts)
        if any(parts[: len(prefix)] == prefix for prefix in model_prefixes):
            _reject(
                "missing_model_artifact",
                "audio.cpp native model directory contains undeclared files",
            )
        _reject(
            "unexpected_extra_files",
            "audio.cpp native package contains undeclared files",
        )


def admit_audio_cpp_native_package(
    executable_path: Path, *, required_model_paths: tuple[Path, ...] = ()
) -> NativeAudioCppInstall:
    """Admit a complete, checksummed native package or fail closed."""
    executable = executable_path.expanduser().resolve()
    if executable.suffix.casefold() != ".exe" or not executable.is_file():
        _reject("missing_executable", "audio.cpp native executable is unavailable")
    package_root = executable.parent
    files, manifest_path = _load_manifest(executable)
    _validate_manifest_files(executable, files)
    receipt = _validate_receipt(executable)
    model_dirs: list[Path] = []
    for model_path in required_model_paths:
        resolved = _require_inside(package_root, model_path)
        _validate_model_path(package_root, resolved, files)
        if resolved.is_dir():
            model_dirs.append(resolved)
    _scan_unexpected_files(package_root, files, tuple(model_dirs))
    return NativeAudioCppInstall(
        executable_path=executable,
        closure_manifest_path=manifest_path,
        files=files,
        source_revision=receipt.source_revision,
        backend=receipt.backend,
        architecture=receipt.architecture,
        model_families=receipt.model_families,
    )
