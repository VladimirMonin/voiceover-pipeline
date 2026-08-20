from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import voiceover_pipeline.audio_cpp.inventory as inventory
from voiceover_pipeline.audio_cpp.inventory import (
    AUDIO_CPP_FAMILY_INVENTORY,
    PINNED_AUDIO_CPP_REVISION,
    AudioCppBuildPlan,
    build_receipt,
    find_family_inventory,
)


def test_inventory_covers_first_release_local_families_and_preserves_existing_ids():
    by_family = {item.family: item for item in AUDIO_CPP_FAMILY_INVENTORY}

    assert by_family["qwen3-asr"].provider_id == "qwen-local"
    assert by_family["qwen3-forced-aligner"].timestamp_origin == "forced"
    assert by_family["nemotron-3.5-asr"].provider_id == "nemotron-local"
    assert by_family["nemotron-3.5-asr"].timestamp_origin == "native"
    assert (
        by_family["nemotron-3.5-asr"].prompt_contract
        == "typed language/task dictionary; phrase hints unavailable"
    )
    assert by_family["qwen3-tts"].provider_id == "qwen-local"
    omnivoice = by_family["omnivoice"]
    assert omnivoice.provider_id == "omnivoice-local"
    assert find_family_inventory("omnivoice").promotion_state == "inventory-only"
    assert (
        omnivoice.model_sha256 == "2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b"
    )
    assert omnivoice.quantization == "Q8_0 GGUF"
    assert all(
        item.model_sha256 is None and item.quantization is None
        for item in AUDIO_CPP_FAMILY_INVENTORY
        if item.family != "omnivoice"
    )
    assert all(item.license and item.provenance for item in AUDIO_CPP_FAMILY_INVENTORY)


def test_pinned_build_plan_selects_declared_backend_without_claiming_mlx_support(tmp_path: Path):
    plan = AudioCppBuildPlan(
        source_dir=tmp_path / "audio.cpp",
        source_revision=PINNED_AUDIO_CPP_REVISION,
        build_dir=tmp_path / "build",
        backend="cuda",
        compiler="clang++",
        cmake_definitions=(("CMAKE_CUDA_ARCHITECTURES", "86"),),
    )

    assert plan.as_dict()["source_revision"] == PINNED_AUDIO_CPP_REVISION
    assert plan.as_dict()["backend"] == "cuda"
    assert "mlx" not in plan.supported_backends
    assert plan.cmake_command() == (
        "cmake",
        "-S",
        str((tmp_path / "audio.cpp").resolve()),
        "-B",
        str((tmp_path / "build").resolve()),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DENGINE_ENABLE_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=86",
    )
    cpu_plan = AudioCppBuildPlan(
        source_dir=tmp_path / "audio.cpp",
        source_revision=PINNED_AUDIO_CPP_REVISION,
        build_dir=tmp_path / "cpu-build",
        backend="cpu",
    )
    assert cpu_plan.cmake_command() == (
        "cmake",
        "-S",
        str((tmp_path / "audio.cpp").resolve()),
        "-B",
        str((tmp_path / "cpu-build").resolve()),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=c++",
        "-DENGINE_ENABLE_CUDA=OFF",
    )
    with pytest.raises(ValueError, match="derived"):
        AudioCppBuildPlan(
            source_dir=tmp_path / "override-source",
            source_revision=PINNED_AUDIO_CPP_REVISION,
            build_dir=tmp_path / "build-with-override",
            backend="cpu",
            cmake_definitions=(("ENGINE_ENABLE_CUDA", "ON"),),
        )
    with pytest.raises(ValueError, match="pinned"):
        AudioCppBuildPlan(
            source_dir=tmp_path,
            source_revision="not-pinned",
            build_dir=tmp_path / "build",
            backend="cpu",
        )


def test_receipt_requires_a_verified_clean_pinned_source(tmp_path: Path, monkeypatch):
    binary = tmp_path / "audio-cpp-fixture"
    binary.write_bytes(b"synthetic fixture binary")
    plan = AudioCppBuildPlan(
        source_dir=tmp_path / "audio.cpp",
        source_revision=PINNED_AUDIO_CPP_REVISION,
        build_dir=tmp_path / "build",
        backend="cpu",
    )

    with pytest.raises(ValueError, match="readable Git worktree"):
        build_receipt(plan, binary_path=binary)

    source_dir = plan.source_dir
    source_dir.mkdir()
    subprocess.run(("git", "init", str(source_dir)), check=True, capture_output=True, text=True)
    subprocess.run(
        ("git", "-C", str(source_dir), "config", "user.email", "fixture@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(source_dir), "config", "user.name", "Fixture"), check=True)
    (source_dir / "fixture.txt").write_text("fixture", encoding="utf-8")
    subprocess.run(("git", "-C", str(source_dir), "add", "fixture.txt"), check=True)
    subprocess.run(
        ("git", "-C", str(source_dir), "commit", "-m", "fixture"),
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(ValueError, match="pinned candidate"):
        build_receipt(plan, binary_path=binary)

    monkeypatch.setattr(
        inventory, "inspect_pinned_source", lambda _source: (PINNED_AUDIO_CPP_REVISION, False)
    )
    receipt = build_receipt(plan, binary_path=binary)

    assert (
        receipt.binary_sha256 == "1415625b88688bab59304bae5c6eba0a682cc322bc348013a20e0611f9977835"
    )
    assert receipt.model_families == tuple(item.family for item in AUDIO_CPP_FAMILY_INVENTORY)
    with pytest.raises(ValueError, match="hash"):
        build_receipt(
            plan,
            binary_path=binary,
            expected_binary_sha256="0" * 64,
        )
    monkeypatch.setattr(
        inventory, "inspect_pinned_source", lambda _source: (PINNED_AUDIO_CPP_REVISION, True)
    )
    with pytest.raises(ValueError, match="dirty"):
        build_receipt(plan, binary_path=binary)
