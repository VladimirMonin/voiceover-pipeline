from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    NATIVE_AUDIO_CPP_BUILD_RECEIPT,
    NATIVE_AUDIO_CPP_CLOSURE_MANIFEST,
    admit_audio_cpp_native_package,
    stream_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_audio_cpp.py"
PINNED_AUDIO_CPP_REVISION = "502b5b74bd26e9b4aed267d1776ecf131cae7215"

RECEIPT_FIELDS = {
    "schema_version",
    "source_revision",
    "compiler",
    "cmake_version",
    "cuda_toolkit_version",
    "architecture",
    "build_flags",
    "binary_sha256",
    "model_families",
}


def _package_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    package = tmp_path / "Аудио CPP пакет"
    package.mkdir()
    executable = package / "audiocpp_cli.exe"
    executable.write_bytes(b"native executable bytes")
    runtime = package / "audio_cpp_runtime.dll"
    runtime.write_bytes(b"native runtime dependency bytes")
    model = package / "models" / "qwen3 model.gguf"
    model.parent.mkdir()
    model.write_bytes(b"model weights")
    return package, executable, runtime, model


def _producer_command(
    *,
    executable: Path,
    runtime: Path,
    model: Path,
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--emit-receipt",
        "--binary",
        str(executable),
        "--closure-dll",
        str(runtime),
        "--model-file",
        str(model),
        "--compiler",
        "cl.exe",
        "--compiler-version",
        "19.40.33811",
        "--cmake-version",
        "3.30.1",
        "--cuda-toolkit-version",
        "12.6.0",
        "--architecture",
        "x86_64",
        "--build-flag",
        "Release",
        "--model-family",
        "qwen3-asr",
        "--model-family",
        "omnivoice",
    ]
    command.extend(extra or [])
    return command


def _run(executable: Path, runtime: Path, model: Path, extra: list[str] | None = None):
    return subprocess.run(
        _producer_command(executable=executable, runtime=runtime, model=model, extra=extra),
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _string_values(document: object) -> list[str]:
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return values


def test_producer_writes_receipt_and_closure_with_matching_stream_hashes(tmp_path: Path):
    package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(executable, runtime, model)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).read_text(encoding="utf-8"))
    closure = json.loads((package / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST).read_text(encoding="utf-8"))

    assert receipt["schema_version"] == 1
    assert receipt["source_revision"] == PINNED_AUDIO_CPP_REVISION
    assert receipt["compiler"] == "cl.exe 19.40.33811"
    assert receipt["cmake_version"] == "3.30.1"
    assert receipt["cuda_toolkit_version"] == "12.6.0"
    assert receipt["architecture"] == "x86_64"
    assert receipt["build_flags"] == "Release"
    assert receipt["model_families"] == ["qwen3-asr", "omnivoice"]
    assert receipt["binary_sha256"] == stream_sha256(executable)

    assert closure["schema_version"] == 1
    assert closure["files"] == {
        "audiocpp_cli.exe": stream_sha256(executable),
        "audio_cpp_runtime.dll": stream_sha256(runtime),
        "models/qwen3 model.gguf": stream_sha256(model),
    }


def test_produced_package_passes_admission(tmp_path: Path):
    package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(executable, runtime, model)
    assert completed.returncode == 0, completed.stderr

    install = admit_audio_cpp_native_package(executable, required_model_paths=(model,))

    assert install.executable_path == executable.resolve()
    assert install.closure_manifest_path == package / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST
    assert set(install.files) == {
        "audiocpp_cli.exe",
        "audio_cpp_runtime.dll",
        "models/qwen3 model.gguf",
    }


def test_receipt_matches_the_native_package_contract_exactly(tmp_path: Path):
    _package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(executable, runtime, model)
    assert completed.returncode == 0, completed.stderr

    receipt = json.loads(
        (executable.parent / NATIVE_AUDIO_CPP_BUILD_RECEIPT).read_text(encoding="utf-8")
    )
    assert set(receipt) == RECEIPT_FIELDS


def test_receipt_and_closure_never_contain_absolute_paths(tmp_path: Path):
    package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(executable, runtime, model)
    assert completed.returncode == 0, completed.stderr

    receipt = json.loads((package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).read_text(encoding="utf-8"))
    closure = json.loads((package / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST).read_text(encoding="utf-8"))
    absolute_token = re.compile(r"(^|[/\\\s=])([A-Za-z]:[\\/]|/)")
    for value in _string_values(receipt):
        assert absolute_token.search(value) is None, value
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in closure["files"])
    assert all(".." not in key for key in closure["files"])


def test_producer_still_refuses_implicit_build(tmp_path: Path):
    package, executable, runtime, model = _package_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--binary",
            str(executable),
            "--closure-dll",
            str(runtime),
            "--model-file",
            str(model),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert completed.returncode != 0
    assert "Refusing" in completed.stderr
    assert not (package / NATIVE_AUDIO_CPP_BUILD_RECEIPT).exists()
    assert not (package / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST).exists()


def test_producer_rejects_missing_binary_file(tmp_path: Path):
    _package, _executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(
        executable=tmp_path / "missing_cli.exe",
        runtime=runtime,
        model=model,
        extra=["--source-revision", PINNED_AUDIO_CPP_REVISION],
    )

    assert completed.returncode != 0
    assert "not an existing file" in completed.stderr.casefold()


def test_producer_rejects_missing_closure_dll(tmp_path: Path):
    package, executable, model, _runtime = _package_fixture(tmp_path)

    completed = _run(
        executable=executable,
        runtime=package / "missing_runtime.dll",
        model=model,
        extra=["--source-revision", PINNED_AUDIO_CPP_REVISION],
    )

    assert completed.returncode != 0
    assert "not an existing file" in completed.stderr.casefold()


def test_producer_rejects_non_pinned_source_revision(tmp_path: Path):
    _package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(
        executable,
        runtime,
        model,
        extra=["--source-revision", "not-the-pinned-revision"],
    )

    assert completed.returncode != 0
    assert "pinned" in completed.stderr.casefold()


def test_producer_rejects_unknown_model_family(tmp_path: Path):
    _package, executable, runtime, model = _package_fixture(tmp_path)

    completed = _run(
        executable,
        runtime,
        model,
        extra=["--model-family", "alien-model"],
    )

    assert completed.returncode != 0
    assert "model family" in completed.stderr.casefold()


def test_producer_rejects_model_outside_package_root(tmp_path: Path):
    _package, executable, runtime, _model = _package_fixture(tmp_path)
    outside = tmp_path / "outside" / "model.gguf"
    outside.parent.mkdir()
    outside.write_bytes(b"model")

    completed = _run(
        executable,
        runtime,
        outside,
        extra=["--source-revision", PINNED_AUDIO_CPP_REVISION],
    )

    assert completed.returncode != 0
    assert "package" in completed.stderr.casefold()
