from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voiceover_pipeline.local_runtime.transports.audio_cpp_cli import (
    discover_native_audio_cpp_install,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    NATIVE_AUDIO_CPP_BUILD_RECEIPT,
    AudioCppPackageError,
    admit_audio_cpp_native_package,
    stream_sha256,
)

PINNED_AUDIO_CPP_REVISION = "502b5b74bd26e9b4aed267d1776ecf131cae7215"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_receipt(
    package: Path, *, binary_bytes: bytes, override: dict[str, object] | None = None
) -> None:
    document = {
        "schema_version": 1,
        "source_revision": PINNED_AUDIO_CPP_REVISION,
        "compiler": "cl.exe",
        "cmake_version": "3.30.1",
        "cuda_toolkit_version": "12.6.0",
        "architecture": "x86_64",
        "build_flags": "Release",
        "binary_sha256": _sha256_bytes(binary_bytes),
        "model_families": ["qwen3-asr", "omnivoice"],
    }
    if override is not None:
        document.update(override)
    (package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )


def _write_closure(package: Path, *, files: list[Path], receipt: bool = True) -> None:
    if receipt:
        _write_receipt(package, binary_bytes=(package / "audiocpp_cli.exe").read_bytes())
    manifest_files: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(package).as_posix()
        manifest_files[relative] = _sha256_bytes(path.read_bytes())
    (package / "audio_cpp_dependency_closure.json").write_text(
        json.dumps({"schema_version": 1, "files": manifest_files}, sort_keys=True),
        encoding="utf-8",
    )


def _make_package(tmp_path: Path, *, receipt: bool = True) -> tuple[Path, Path]:
    package = tmp_path / "Аудио CPP package"
    package.mkdir()
    executable = package / "audiocpp_cli.exe"
    executable.write_bytes(b"native executable bytes")
    runtime_dll = package / "audio_cpp_runtime.dll"
    runtime_dll.write_bytes(b"native runtime dependency bytes")
    _write_closure(package, files=[executable, runtime_dll], receipt=receipt)
    return package, executable


def _write_model(package: Path, relative: str, data: bytes = b"model weights") -> Path:
    model = package / relative
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(data)
    return model


def _package_with_model(package: Path, executable: Path, model: Path) -> None:
    _write_closure(
        package,
        files=[executable, package / "audio_cpp_runtime.dll", model],
        receipt=True,
    )


def test_streaming_hash_matches_whole_file_without_reading_it_into_memory(
    tmp_path: Path, monkeypatch
):
    payload = b"\xa5" * ((1 << 20) * 3 + 17)
    large = tmp_path / "large model.bin"
    large.write_bytes(payload)

    def explode():
        raise AssertionError("read_bytes must not load large files into memory")

    monkeypatch.setattr(Path, "read_bytes", explode)

    assert stream_sha256(large) == hashlib.sha256(payload).hexdigest()


def test_package_admission_accepts_checksummed_package_and_unicode_space_paths(
    tmp_path: Path,
):
    package, executable = _make_package(tmp_path)

    install = admit_audio_cpp_native_package(executable)

    assert install.executable_path == executable.resolve()
    assert install.closure_manifest_path == package / "audio_cpp_dependency_closure.json"
    assert set(install.files) == {
        "audiocpp_cli.exe",
        "audio_cpp_runtime.dll",
    }
    assert install.files["audiocpp_cli.exe"] == _sha256_bytes(b"native executable bytes")


def test_discover_native_install_delegates_to_structured_admission(tmp_path: Path):
    _package, executable = _make_package(tmp_path)

    install = discover_native_audio_cpp_install(executable)

    assert install.executable_path == executable.resolve()


def test_missing_executable_is_rejected_with_structured_code(tmp_path: Path):
    package, _executable = _make_package(tmp_path)

    with pytest.raises(AudioCppPackageError, match="unavailable") as excinfo:
        admit_audio_cpp_native_package(package / "missing_cli.exe")

    assert excinfo.value.code == "missing_executable"


def test_malformed_manifest_json_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_dependency_closure.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(AudioCppPackageError, match="manifest") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "malformed_manifest"


def test_manifest_without_files_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_dependency_closure.json").write_text(
        json.dumps({"schema_version": 1, "files": {}}), encoding="utf-8"
    )

    with pytest.raises(AudioCppPackageError, match="manifest") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "malformed_manifest"


def test_unsupported_manifest_schema_version_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_dependency_closure.json").write_text(
        json.dumps({"schema_version": 2, "files": {"audiocpp_cli.exe": "0" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(AudioCppPackageError, match="schema version") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "unsupported_schema_version"


def test_manifest_path_escape_entries_are_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    for evil in ("..\\..\\evil.dll", "C:/evil/evil.dll", "other\\..\\evil.dll"):
        (package / "audio_cpp_dependency_closure.json").write_text(
            json.dumps({"schema_version": 1, "files": {evil: "0" * 64}}),
            encoding="utf-8",
        )
        with pytest.raises(AudioCppPackageError) as excinfo:
            admit_audio_cpp_native_package(executable)
        assert excinfo.value.code == "malformed_manifest"


def test_declared_file_missing_from_disk_is_rejected_as_dll_closure(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_dependency_closure.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "audiocpp_cli.exe": _sha256_bytes(b"native executable bytes"),
                    "missing_runtime.dll": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AudioCppPackageError) as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "missing_dll_closure"


def test_manifest_without_dll_dependencies_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_dependency_closure.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {"audiocpp_cli.exe": _sha256_bytes(b"native executable bytes")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AudioCppPackageError) as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "missing_dll_closure"


def test_modified_manifest_file_bytes_are_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "audio_cpp_runtime.dll").write_bytes(b"tampered dll")

    with pytest.raises(AudioCppPackageError, match="checksum") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "modified_bytes"


def test_model_path_outside_package_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    outside = tmp_path / "outside" / "model.gguf"
    outside.parent.mkdir()
    outside.write_bytes(b"model")

    with pytest.raises(AudioCppPackageError, match="outside") as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(outside,))

    assert excinfo.value.code == "path_escape"


def test_model_path_with_dotdot_escape_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    package_models = package / "models"
    package_models.mkdir()
    (package_models / "qwen.gguf").write_bytes(b"model")
    escape = package_models / ".." / "sibling.gguf"

    with pytest.raises(AudioCppPackageError, match="outside") as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(escape,))

    assert excinfo.value.code == "path_escape"


def test_undeclared_required_model_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    model = _write_model(package, "models/qwen.gguf")

    with pytest.raises(AudioCppPackageError, match="cover a required model") as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(model,))

    assert excinfo.value.code == "missing_model_artifact"


def test_modified_model_bytes_are_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    model = _write_model(package, "models/qwen.gguf")
    _package_with_model(package, executable, model)
    model.write_bytes(b"tampered model weights")

    with pytest.raises(AudioCppPackageError, match="checksum") as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(model,))

    assert excinfo.value.code == "modified_bytes"


def test_model_directory_admission_hashes_every_declared_file(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    first = _write_model(package, "models/qwen3/qwen.gguf", b"qwen weights")
    second = _write_model(package, "models/qwen3/aligner.bin", b"aligner weights")
    _write_closure(
        package,
        files=[executable, package / "audio_cpp_runtime.dll", first, second],
        receipt=True,
    )
    models = package / "models" / "qwen3"

    install = admit_audio_cpp_native_package(executable, required_model_paths=(models,))

    assert install.files["models/qwen3/qwen.gguf"] == _sha256_bytes(b"qwen weights")
    assert install.files["models/qwen3/aligner.bin"] == _sha256_bytes(b"aligner weights")


def test_model_directory_undeclared_extra_file_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    first = _write_model(package, "models/qwen3/qwen.gguf")
    _write_closure(
        package,
        files=[executable, package / "audio_cpp_runtime.dll", first],
        receipt=True,
    )
    _write_model(package, "models/qwen3/extra.bin", b"undeclared weights")
    models = package / "models" / "qwen3"

    with pytest.raises(AudioCppPackageError) as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(models,))

    assert excinfo.value.code == "missing_model_artifact"


def test_empty_model_directory_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    models = package / "models"
    models.mkdir()

    with pytest.raises(AudioCppPackageError, match="directory is empty") as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(models,))

    assert excinfo.value.code == "missing_model_artifact"


def test_symlink_inside_model_directory_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    model = _write_model(package, "models/qwen.gguf")
    _package_with_model(package, executable, model)
    models = package / "models"
    link = models / "linked.gguf"
    try:
        link.symlink_to(model)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    with pytest.raises(AudioCppPackageError) as excinfo:
        admit_audio_cpp_native_package(executable, required_model_paths=(models,))

    assert excinfo.value.code == "path_escape"


def test_missing_build_receipt_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path, receipt=False)
    _write_closure(package, files=[executable, package / "audio_cpp_runtime.dll"], receipt=False)

    with pytest.raises(AudioCppPackageError, match="receipt") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "missing_receipt"


def test_malformed_build_receipt_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).write_text("not json", encoding="utf-8")

    with pytest.raises(AudioCppPackageError, match="receipt") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "invalid_receipt"


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": 2},
        {"source_revision": ""},
        {"compiler": None},
        {"binary_sha256": "0" * 64},
        {"binary_sha256": "not-a-sha"},
        {"model_families": []},
        {"model_families": ["qwen3-asr", "qwen3-asr"]},
        {"unexpected_field": "boom"},
        {"compiler": "C:\\Users\\builder\\cl.exe"},
        {"cmake_version": "3.30.1", "build_flags": "-DCMAKE_PREFIX_PATH=/opt/toolchain"},
    ],
)
def test_structural_receipt_defects_are_rejected(tmp_path: Path, override: dict[str, object]):
    package, executable = _make_package(tmp_path)
    (package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": PINNED_AUDIO_CPP_REVISION,
                "compiler": "cl.exe",
                "cmake_version": "3.30.1",
                "cuda_toolkit_version": "12.6.0",
                "architecture": "x86_64",
                "build_flags": "Release",
                "binary_sha256": _sha256_bytes(b"native executable bytes"),
                "model_families": ["qwen3-asr", "omnivoice"],
                **override,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AudioCppPackageError, match="receipt") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "invalid_receipt"


def test_receipt_binary_hash_mismatch_is_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    _write_receipt(package, binary_bytes=b"different binary", override={"binary_sha256": "0" * 64})

    with pytest.raises(AudioCppPackageError, match="receipt") as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "invalid_receipt"


def test_extra_undeclared_files_in_package_root_are_rejected(tmp_path: Path):
    package, executable = _make_package(tmp_path)
    (package / "stray_debug.symbols").write_bytes(b"stray")

    with pytest.raises(AudioCppPackageError) as excinfo:
        admit_audio_cpp_native_package(executable)

    assert excinfo.value.code == "unexpected_extra_files"


def test_undeclared_build_receipt_is_an_explicit_allowed_exception(tmp_path: Path):
    package, executable = _make_package(tmp_path)

    install = admit_audio_cpp_native_package(executable)

    assert NATIVE_AUDIO_CPP_BUILD_RECEIPT not in install.files
