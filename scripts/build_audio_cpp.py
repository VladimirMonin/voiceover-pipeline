#!/usr/bin/env python3
"""Emit pinned audio.cpp build metadata; this helper never downloads models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiceover_pipeline.audio_cpp.inventory import AudioCppBuildPlan, PINNED_AUDIO_CPP_REVISION


def _cmake_definition(value: str) -> tuple[str, str]:
    name, separator, definition = value.partition("=")
    if not separator or not name or not definition:
        raise argparse.ArgumentTypeError("CMake definitions must use NAME=VALUE")
    return name, definition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned audio.cpp checkout; it is never modified.")
    parser.add_argument("--build-dir", type=Path, required=True, help="Build directory outside the source checkout.")
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip", "vulkan", "metal"), required=True)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--cmake-definition", action="append", type=_cmake_definition, default=[])
    parser.add_argument(
        "--emit-plan",
        action="store_true",
        help="Print the declarative build plan only; do not configure or compile.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = AudioCppBuildPlan(
        source_dir=args.source,
        source_revision=PINNED_AUDIO_CPP_REVISION,
        build_dir=args.build_dir,
        backend=args.backend,
        compiler=args.compiler,
        cmake_definitions=tuple(args.cmake_definition),
    )
    if not args.emit_plan:
        raise SystemExit("Refusing to configure or compile implicitly; pass --emit-plan for metadata only.")
    print(json.dumps(plan.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
