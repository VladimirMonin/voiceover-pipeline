from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_audio_cpp.py"
PINNED_AUDIO_CPP_REVISION = "502b5b74bd26e9b4aed267d1776ecf131cae7215"


def test_build_helper_emits_cpu_cuda_selection_metadata_without_configuring_or_downloading(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(tmp_path / "audio.cpp"),
            "--build-dir",
            str(tmp_path / "build-cuda"),
            "--backend",
            "cuda",
            "--compiler",
            "clang++",
            "--cmake-definition",
            "CMAKE_CUDA_ARCHITECTURES=86",
            "--emit-plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    plan = json.loads(completed.stdout)

    assert plan["source_revision"] == PINNED_AUDIO_CPP_REVISION
    assert plan["backend"] == "cuda"
    assert plan["cmake_definitions"] == {"CMAKE_CUDA_ARCHITECTURES": "86"}
    assert "-DCMAKE_CXX_COMPILER=clang++" in plan["cmake_command"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=86" in plan["cmake_command"]
    assert plan["mlx"] == "not-installed-not-implemented"
