"""Offline contracts for script-fragment preparation and runtime chunk merging."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from conftest import cli_json

from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.local_tts_text import merge_omnivoice_session_fragments
from voiceover_pipeline.models import ScriptChunk
from voiceover_pipeline.services.prepare import (
    PreparationError,
    prepare_runtime_chunks,
    prepare_script_fragments,
)

MARKDOWN_BODY = "Первое предложение.\n******\nВторое предложение.\n"


def _chunks(*texts: str) -> list[ScriptChunk]:
    return [
        ScriptChunk(number=index, id=f"chunk_{index:02d}", text=text)
        for index, text in enumerate(texts, start=1)
    ]


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, Any] = {
        "provider": "polza-tts",
        "model": "openai/gpt-4o-mini-tts",
        "limit_chunks": None,
        "mode": "preset",
        "reference_audio": None,
        "reference_text": None,
        "design_instruction": None,
        "voice_bank_profile": None,
        "voice_bank_catalog": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _identity_prepare(
    chunks: Iterable[ScriptChunk], provider: str, model: str | None
) -> list[ScriptChunk]:
    return list(chunks)


def _build_voice_bank(tmp_path: Path) -> Path:
    import wave

    bank_root = tmp_path / "bank"
    (bank_root / "voices").mkdir(parents=True)
    reference = bank_root / "voices" / "main.wav"
    with wave.open(str(reference), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * 4000)
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    catalog_path = bank_root / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_voice": "main",
                "voices": [
                    {
                        "id": "main",
                        "display_name": "Main Narrator",
                        "description": "",
                        "language": "ru",
                        "reference_audio": "voices/main.wav",
                        "reference_text": "Эталонная фраза.",
                        "reference_sha256": digest,
                        "origin": {"mode": "owner-reference", "instruction": None, "seed": 7},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return catalog_path


def _run_generate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    script_body: str,
    *extra_args: str,
) -> tuple[int, dict]:
    import voiceover_pipeline.cli as cli

    script = tmp_path / "prepare-script.md"
    script.write_text(script_body, encoding="utf-8")
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "prepare-run",
            "--dry-run-cost",
            "--json",
            *extra_args,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert int(exit_info.value.code or 0) == exit_info.value.code
    return exit_info.value.code, json.loads(capsys.readouterr().out)


def test_prepare_script_fragments_keeps_chunk_identity_and_reports_counts() -> None:
    chunks = _chunks("Первое предложение.", "Второе предложение.")
    forwarded: dict[str, Any] = {}

    def local_prepare(
        source: Iterable[ScriptChunk], provider: str, model: str | None
    ) -> list[ScriptChunk]:
        forwarded["source"] = list(source)
        forwarded["provider"] = provider
        forwarded["model"] = model
        return list(source)

    prepared = prepare_script_fragments(
        _args(provider="openrouter-tts", model="openai/gpt-4o-mini-tts"),
        chunks,
        "markdown",
        local_prepare=local_prepare,
    )

    assert forwarded["source"] == chunks
    assert forwarded["provider"] == "openrouter-tts"
    assert forwarded["model"] == "openai/gpt-4o-mini-tts"
    assert prepared.chunks[0] is chunks[0]
    assert prepared.chunks[1] is chunks[1]
    assert prepared.original_count == 2
    assert prepared.requested_count == 2


def test_prepare_script_fragments_dialogue_bypasses_local_preparation() -> None:
    chunks = _chunks("Реплика первая.", "Реплика вторая.")
    calls: list[tuple[str, str | None]] = []

    def local_prepare(
        source: Iterable[ScriptChunk], provider: str, model: str | None
    ) -> list[ScriptChunk]:
        calls.append((provider, model))
        return []

    prepared = prepare_script_fragments(
        _args(), chunks, "gemini-dialogue", local_prepare=local_prepare
    )

    assert calls == []
    assert prepared.chunks is chunks
    assert prepared.original_count == 2
    assert prepared.requested_count == 2


def test_prepare_script_fragments_limits_before_runtime_merging() -> None:
    chunks = _chunks("Первое предложение.", "Второе предложение.", "Третье предложение.")

    prepared = prepare_script_fragments(
        _args(limit_chunks=2), chunks, "markdown", local_prepare=_identity_prepare
    )
    merged = prepare_runtime_chunks(
        _args(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="clone",
            reference_audio="reference.wav",
            reference_text="эталонная фраза",
        ),
        prepared.chunks,
        "markdown",
        merge_session_fragments=merge_omnivoice_session_fragments,
    )

    assert prepared.original_count == 3
    assert prepared.requested_count == 2
    assert [chunk.text for chunk in prepared.chunks] == [
        "Первое предложение.",
        "Второе предложение.",
    ]
    assert merged == [
        ScriptChunk(
            number=1,
            id="chunk_01_omnivoice_session",
            text="Первое предложение. Второе предложение.",
        )
    ]


def test_prepare_script_fragments_rejects_empty_scripts() -> None:
    with pytest.raises(PreparationError, match="produced no chunks"):
        prepare_script_fragments(_args(), [], "markdown", local_prepare=_identity_prepare)


def test_prepare_script_fragments_checks_empty_before_limit_and_limit_is_positive() -> None:
    with pytest.raises(PreparationError, match="produced no chunks"):
        prepare_script_fragments(
            _args(limit_chunks=0), [], "markdown", local_prepare=_identity_prepare
        )
    with pytest.raises(PreparationError, match="greater than zero"):
        prepare_script_fragments(
            _args(limit_chunks=0),
            _chunks("Первое предложение."),
            "markdown",
            local_prepare=_identity_prepare,
        )


def test_prepare_runtime_chunks_returns_originals_outside_omnivoice_non_dialogue() -> None:
    chunks = _chunks("Первое предложение.")
    merges: list[dict[str, Any]] = []

    def merge_session(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        merges.append(kwargs)
        return []

    assert (
        prepare_runtime_chunks(
            _args(provider="polza-tts"),
            chunks,
            "markdown",
            merge_session_fragments=merge_session,
        )
        is chunks
    )
    assert (
        prepare_runtime_chunks(
            _args(provider="omnivoice-local", model=OMNIVOICE_LOCAL_MODEL_ID),
            chunks,
            "gemini-dialogue",
            merge_session_fragments=merge_session,
        )
        is chunks
    )
    assert (
        prepare_runtime_chunks(
            _args(provider="omnivoice-local", model="audio-cpp/other-model"),
            chunks,
            "markdown",
            merge_session_fragments=merge_session,
        )
        is chunks
    )
    assert merges == []


def test_prepare_runtime_chunks_clones_admitted_bank_profile_reference(
    tmp_path: Path,
) -> None:
    bank_root = tmp_path / "bank"
    profile = argparse.Namespace(
        reference_audio="voices/main.wav", reference_text="Эталонная фраза."
    )
    catalog = argparse.Namespace(root=bank_root)
    recorded: dict[str, Any] = {}

    def merge_session(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        recorded["fragments"] = list(fragments)
        recorded.update(kwargs)
        return []

    chunks = _chunks("Первое предложение.")
    prepare_runtime_chunks(
        _args(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="preset",
            voice_bank_profile=profile,
            voice_bank_catalog=catalog,
        ),
        chunks,
        "markdown",
        merge_session_fragments=merge_session,
    )

    assert recorded["fragments"] == chunks
    assert recorded["mode"] == "clone"
    assert recorded["reference_audio_path"] == str(bank_root / "voices" / "main.wav")
    assert recorded["reference_text"] == "Эталонная фраза."

    recorded.clear()
    prepare_runtime_chunks(
        _args(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="preset",
            voice_bank_profile=profile,
            voice_bank_catalog=None,
        ),
        chunks,
        "markdown",
        merge_session_fragments=merge_session,
    )

    assert recorded["mode"] == "clone"
    assert recorded["reference_audio_path"] == str(Path("voices/main.wav"))
    assert recorded["reference_text"] == "Эталонная фраза."


def test_prepare_runtime_chunks_forwards_explicit_mode_reference_and_design() -> None:
    recorded: dict[str, Any] = {}

    def merge_session(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        recorded.clear()
        recorded.update(kwargs)
        return []

    prepare_runtime_chunks(
        _args(
            provider="omnivoice-local",
            model=OMNIVOICE_LOCAL_MODEL_ID,
            mode="design",
            design_instruction="тёплый спокойный голос",
            reference_audio="explicit.wav",
            reference_text="эталон",
        ),
        _chunks("Первое предложение."),
        "markdown",
        merge_session_fragments=merge_session,
    )

    assert recorded == {
        "mode": "design",
        "reference_audio_path": "explicit.wav",
        "reference_text": "эталон",
        "design_instruction": "тёплый спокойный голос",
    }

    prepare_runtime_chunks(
        _args(provider="omnivoice-local", model=OMNIVOICE_LOCAL_MODEL_ID),
        _chunks("Первое предложение."),
        "markdown",
        merge_session_fragments=merge_session,
    )

    assert recorded == {
        "mode": "preset",
        "reference_audio_path": None,
        "reference_text": None,
        "design_instruction": None,
    }


def test_generate_dry_run_forwards_cli_local_prep_binding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import voiceover_pipeline.cli as cli

    forwarded: dict[str, Any] = {}

    def fake_prepare(
        source: Iterable[ScriptChunk], provider: str, model: str | None
    ) -> list[ScriptChunk]:
        forwarded["source"] = list(source)
        forwarded["provider"] = provider
        forwarded["model"] = model
        return [ScriptChunk(number=1, id="prepared_01", text="Подготовленный текст.")]

    monkeypatch.setattr(cli, "prepare_local_tts_chunks", fake_prepare)

    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        MARKDOWN_BODY,
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["chunks"] == 1
    assert payload["original_chunks"] == 1
    assert payload["runtime_sessions"] == 1
    assert [chunk.text for chunk in forwarded["source"]] == [
        "Первое предложение.",
        "Второе предложение.",
    ]
    assert forwarded["provider"] == "polza-tts"
    assert forwarded["model"] == "openai/gpt-4o-mini-tts"


def test_generate_dry_run_limits_before_forwarded_omnivoice_bank_merge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import voiceover_pipeline.cli as cli

    catalog_path = _build_voice_bank(tmp_path)
    recorded: dict[str, Any] = {}

    def fake_merge(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        recorded["fragments"] = list(fragments)
        recorded["kwargs"] = kwargs
        return [ScriptChunk(number=1, id="chunk_01_omnivoice_session", text="merged")]

    monkeypatch.setattr(cli, "merge_omnivoice_session_fragments", fake_merge)

    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        MARKDOWN_BODY,
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--voice-bank",
        str(catalog_path),
        "--limit-chunks",
        "1",
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["chunks"] == 1
    assert payload["original_chunks"] == 2
    assert payload["runtime_sessions"] == 1
    assert [chunk.text for chunk in recorded["fragments"]] == ["Первое предложение."]
    assert recorded["kwargs"] == {
        "mode": "clone",
        "reference_audio_path": str(tmp_path / "bank" / "voices" / "main.wav"),
        "reference_text": "Эталонная фраза.",
    }


def test_generate_dry_run_forwards_explicit_omnivoice_clone_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import voiceover_pipeline.cli as cli

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"RIFF0000WAVE")
    recorded: dict[str, Any] = {}

    def fake_merge(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        recorded["kwargs"] = kwargs
        return [ScriptChunk(number=1, id="chunk_01_omnivoice_session", text="merged")]

    monkeypatch.setattr(cli, "merge_omnivoice_session_fragments", fake_merge)

    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        MARKDOWN_BODY,
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--mode",
        "clone",
        "--reference-audio",
        str(reference),
        "--reference-text",
        "эталонная фраза",
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert recorded["kwargs"] == {
        "mode": "clone",
        "reference_audio_path": reference,
        "reference_text": "эталонная фраза",
        "design_instruction": None,
    }


def test_generate_runs_design_route_guard_between_preparation_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import voiceover_pipeline.cli as cli

    catalog_path = _build_voice_bank(tmp_path)
    calls: list[str] = []

    def fake_prepare(
        source: Iterable[ScriptChunk], provider: str, model: str | None
    ) -> list[ScriptChunk]:
        calls.append("prepare_local")
        return list(source)

    def fake_guard(args: argparse.Namespace, chunks: list[ScriptChunk]) -> None:
        calls.append(f"guard:{len(chunks)}")

    def fake_merge(fragments: Iterable[ScriptChunk], **kwargs: object) -> list[ScriptChunk]:
        calls.append("merge")
        return list(fragments)

    monkeypatch.setattr(cli, "prepare_local_tts_chunks", fake_prepare)
    monkeypatch.setattr(cli, "_enforce_omnivoice_design_route", fake_guard)
    monkeypatch.setattr(cli, "merge_omnivoice_session_fragments", fake_merge)

    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        MARKDOWN_BODY,
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--voice-bank",
        str(catalog_path),
        "--limit-chunks",
        "1",
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert calls == ["prepare_local", "guard:1", "merge"]


def test_generate_rejects_non_positive_limit_with_args_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        MARKDOWN_BODY,
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
        "--limit-chunks",
        "0",
    )

    assert code == 2
    assert payload == {
        "status": "error",
        "error": "--limit-chunks must be greater than zero",
        "code": 2,
    }
    assert not (tmp_path / "out").exists()


def test_generate_rejects_script_without_chunks_with_args_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, payload = _run_generate(
        monkeypatch,
        capsys,
        tmp_path,
        "   \n",
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
    )

    assert code == 2
    assert payload["code"] == 2
    assert payload["status"] == "error"
    assert payload["error"] == "Script produced no chunks. Check delimiter and content."
    assert not (tmp_path / "out").exists()


def test_generate_limit_chunks_zero_reports_args_error_over_process_boundary(
    tmp_path: Path,
) -> None:
    script = tmp_path / "limit-zero.md"
    script.write_text(MARKDOWN_BODY, encoding="utf-8")

    code, payload = cli_json(
        "generate",
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "prepare-run",
        "--limit-chunks",
        "0",
        "--dry-run-cost",
        "--json",
    )

    assert code == 2
    assert payload["status"] == "error"
    assert payload["code"] == 2
    assert payload["error"] == "--limit-chunks must be greater than zero"
    assert not (tmp_path / "out").exists()
