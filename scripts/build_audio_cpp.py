#!/usr/bin/env python3
"""Emit pinned audio.cpp build metadata; this helper never downloads models.

Two explicit metadata modes are supported:
- ``--emit-plan`` prints the declarative build plan (never configures/compiles).
- ``--emit-receipt`` records evidence of an already-built package as
  ``build_receipt.json`` plus ``audio_cpp_dependency_closure.json`` next to the
  executable. It runs no toolchain (no cmake/ninja/cl/nvcc), downloads
  nothing, and stores only relative paths so the artifacts are relocatable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiceover_pipeline.audio_cpp.inventory import (
    PINNED_AUDIO_CPP_REVISION,
    AudioCppBuildPlan,
    find_family_inventory,
)
from voiceover_pipeline.local_runtime.transports.audio_cpp_package import (
    NATIVE_AUDIO_CPP_BUILD_RECEIPT,
    NATIVE_AUDIO_CPP_CLOSURE_MANIFEST,
    stream_sha256,
)

BUILD_RECEIPT_SCHEMA_VERSION = 1
CLOSURE_MANIFEST_SCHEMA_VERSION = 1


def _cmake_definition(value: str) -> tuple[str, str]:
    name, separator, definition = value.partition("=")
    if not separator or not name or not definition:
        raise argparse.ArgumentTypeError("CMake definitions must use NAME=VALUE")
    return name, definition


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("values must not be blank")
    return value


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not an existing file: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--emit-plan",
        action="store_true",
        help="Print the declarative build plan only; do not configure or compile.",
    )
    modes.add_argument(
        "--emit-receipt",
        action="store_true",
        help=(
            "Record build evidence as build_receipt.json and "
            "audio_cpp_dependency_closure.json next to the executable; "
            "never runs a toolchain."
        ),
    )
    parser.add_argument("--source", type=Path, help="Pinned audio.cpp checkout; never modified.")
    parser.add_argument(
        "--build-dir", type=Path, help="Build directory outside the source checkout."
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip", "vulkan", "metal"))
    plan_arguments = parser.add_argument_group("build plan evidence")
    plan_arguments.add_argument("--compiler", default="c++", type=_non_blank)
    plan_arguments.add_argument(
        "--cmake-definition", action="append", type=_cmake_definition, default=[]
    )
    receipt_arguments = parser.add_argument_group("build receipt evidence")
    receipt_arguments.add_argument("--binary", type=_existing_file)
    receipt_arguments.add_argument("--closure-dll", action="append", type=Path, default=[])
    receipt_arguments.add_argument("--model-file", action="append", type=Path, default=[])
    receipt_arguments.add_argument(
        "--source-revision",
        default=PINNED_AUDIO_CPP_REVISION,
        type=_non_blank,
    )
    receipt_arguments.add_argument("--compiler-version", type=_non_blank)
    receipt_arguments.add_argument("--cmake-version", type=_non_blank)
    receipt_arguments.add_argument("--cuda-toolkit-version", type=_non_blank)
    receipt_arguments.add_argument("--architecture", type=_non_blank)
    receipt_arguments.add_argument("--build-flag", action="append", type=_non_blank, default=[])
    receipt_arguments.add_argument("--model-family", action="append", type=_non_blank, default=[])
    return parser


def _refuse_implicit_build() -> None:
    raise SystemExit(
        "Refusing to configure or compile implicitly; pass --emit-plan or --emit-receipt for metadata only."
    )


def _require_inside(package_root: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise SystemExit(f"audio.cpp build receipt {label} must live inside the package") from exc
    if not resolved.is_file():
        raise SystemExit(f"audio.cpp build receipt {label} is not an existing file: {resolved}")
    return resolved


def _write_receipt(
    *,
    executable: Path,
    source_revision: str,
    compiler: str,
    compiler_version: str | None,
    cmake_version: str,
    cuda_toolkit_version: str,
    architecture: str,
    build_flags: tuple[str, ...],
    binary_sha256: str,
    model_families: tuple[str, ...],
) -> None:
    document = {
        "schema_version": BUILD_RECEIPT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "compiler": " ".join(part for part in (compiler, compiler_version) if part),
        "cmake_version": cmake_version,
        "cuda_toolkit_version": cuda_toolkit_version,
        "architecture": architecture,
        "build_flags": " ".join(build_flags),
        "binary_sha256": binary_sha256,
        "model_families": list(model_families),
    }
    (executable.parent / NATIVE_AUDIO_CPP_BUILD_RECEIPT).write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _write_closure_manifest(
    *,
    executable: Path,
    package_root: Path,
    closure_dlls: tuple[Path, ...],
    model_files: tuple[Path, ...],
) -> None:
    entries: dict[str, str] = {}
    for path in (executable, *closure_dlls, *model_files):
        relative = path.resolve().relative_to(package_root).as_posix()
        entries[relative] = stream_sha256(path)
    (package_root / NATIVE_AUDIO_CPP_CLOSURE_MANIFEST).write_text(
        json.dumps(
            {"schema_version": CLOSURE_MANIFEST_SCHEMA_VERSION, "files": entries},
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def emit_build_evidence(args: argparse.Namespace) -> None:
    if args.source_revision != PINNED_AUDIO_CPP_REVISION:
        raise SystemExit("audio.cpp build evidence source revision must match the pinned candidate")
    if args.binary is None:
        raise SystemExit("audio.cpp build evidence requires the built executable path")
    if not args.closure_dll:
        raise SystemExit("audio.cpp build evidence requires at least one closure DLL")
    if args.cmake_version is None or args.cuda_toolkit_version is None or args.architecture is None:
        raise SystemExit(
            "audio.cpp build evidence requires cmake version, CUDA toolkit version and architecture"
        )
    for family in args.model_family:
        try:
            find_family_inventory(family)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if not args.model_family:
        raise SystemExit("audio.cpp build evidence requires at least one model family")
    package_root = args.binary.resolve().parent
    if args.binary.suffix.casefold() != ".exe":
        raise SystemExit("audio.cpp build receipt requires a Windows .exe executable")
    binary_sha256 = stream_sha256(args.binary)
    closure_dlls = tuple(
        _require_inside(package_root, dll, "closure DLL") for dll in args.closure_dll
    )
    model_files = tuple(
        _require_inside(package_root, model, "model file") for model in args.model_file
    )
    _write_receipt(
        executable=args.binary,
        source_revision=args.source_revision,
        compiler=args.compiler,
        compiler_version=args.compiler_version,
        cmake_version=args.cmake_version,
        cuda_toolkit_version=args.cuda_toolkit_version,
        architecture=args.architecture,
        build_flags=tuple(args.build_flag),
        binary_sha256=binary_sha256,
        model_families=tuple(args.model_family),
    )
    _write_closure_manifest(
        executable=args.binary,
        package_root=package_root,
        closure_dlls=closure_dlls,
        model_files=model_files,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.emit_receipt:
        emit_build_evidence(args)
        return 0
    if not args.emit_plan:
        _refuse_implicit_build()
    if args.source is None or args.build_dir is None or args.backend is None:
        raise SystemExit("audio.cpp build plan requires --source, --build-dir and --backend")
    plan = AudioCppBuildPlan(
        source_dir=args.source,
        source_revision=PINNED_AUDIO_CPP_REVISION,
        build_dir=args.build_dir,
        backend=args.backend,
        compiler=args.compiler,
        cmake_definitions=tuple(args.cmake_definition),
    )
    print(json.dumps(plan.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
