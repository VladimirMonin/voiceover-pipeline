"""Offline CLI journey: paid-shaped TTS, local ASR, SQLite, FTS5 and recovery.

Only synthetic bytes and a fake provider enter the network boundary. FFmpeg,
FFprobe, SQLite and the public CLI are real; no model or credential is loaded.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import wave
from decimal import Decimal

import pytest
import requests

import voiceover_pipeline.cli as cli
from voiceover_pipeline.commands.history import list_history
from voiceover_pipeline.history.paths import history_database_path
from voiceover_pipeline.history.repository import HistoryRepository
from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRSegment,
    SynthesisResult,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider

_MODEL = "Qwen/Qwen3-ASR-0.6B"
_PRICE = Decimal("0.100000000000000001")
_TRANSCRIPT = "Расшифровка отмечает кварцовый эксперимент."


def _synthetic_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(
            b"".join(
                struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(8000)
            )
        )
    return buffer.getvalue()


class _OfflineSpeech:
    def __init__(self, wav: bytes) -> None:
        self.wav = wav
        self.submits: list[str] = []

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        self.submits.append(chunk_id)
        return SynthesisResult(
            audio_bytes=self.wav,
            audio_format="wav",
            transcript=text,
            generation_id=f"synthetic-{chunk_id}",
            client_path="requests",
            raw_metadata={"usage_direct": {"cost_rub": _PRICE}},
        )


class _OfflineASR(ASRProvider):
    provider_id = "qwen-local"

    def transcribe(self, request: ASRRequest) -> ASRResult:
        return ASRResult(
            transcript=_TRANSCRIPT,
            provider_id=self.provider_id,
            model_id=_MODEL,
            language="ru",
            duration_s=1.5,
            segments=(ASRSegment(text=_TRANSCRIPT, start_s=0.0, end_s=1.2),),
            alignment_origin="native",
            execution=ASRExecutionReceipt(
                runtime="synthetic-runtime",
                runtime_version="1.0",
                resolved_device="cpu",
                resolved_compute="float32",
                measurements={},
            ),
        )


def _cli_json(monkeypatch, capsys, *argv: str) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", ["voiceover", *argv, "--json"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    stdout = capsys.readouterr().out
    return stopped.value.code, json.loads(stdout)


def test_real_media_sqlite_fts5_asr_and_exact_cost_recovery(tmp_path, monkeypatch, capsys):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("real FFmpeg/FFprobe unavailable on this host")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-synthetic-not-a-secret")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    def no_network(*_args, **_kwargs):
        pytest.fail("offline journey attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    provider = _OfflineSpeech(_synthetic_wav())
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: provider)
    script = tmp_path / "script.md"
    script.write_text(
        "Кварцовый вступительный фрагмент.\n******\n"
        "Грушевый средний фрагмент.\n******\nСливовый заключительный фрагмент.",
        encoding="utf-8",
    )
    run_id = "s11-offline"
    real_reserve = HistoryRepository.reserve_paid_tts_attempt
    reserves = 0

    def fail_before_third(self, *args, **kwargs):
        nonlocal reserves
        reserves += 1
        if reserves == 3:
            raise RuntimeError("synthetic local crash before third paid reservation")
        return real_reserve(self, *args, **kwargs)

    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", fail_before_third)
    code, failed = _cli_json(
        monkeypatch,
        capsys,
        "generate",
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "alloy",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        run_id,
        "--no-trim",
    )
    assert code == 30, failed
    assert provider.submits == ["chunk_01", "chunk_02"]
    run_uuid = next(
        item["run_uuid"] for item in list_history()["runs"] if item["user_label"] == run_id
    )
    code, costs_before = _cli_json(monkeypatch, capsys, "history", "costs")
    assert code == 0
    assert costs_before["totals"] == [
        {
            "currency": "RUB",
            "known_amount": "0.200000000000000002",
            "known_attempts": 2,
            "exact_attempts": 2,
            "non_exact_attempts": 0,
            "unknown_attempts": 0,
        }
    ]

    script.unlink()
    monkeypatch.setattr(HistoryRepository, "reserve_paid_tts_attempt", real_reserve)
    code, resumed = _cli_json(monkeypatch, capsys, "history", "resume", run_uuid)
    assert code == 0, resumed
    assert provider.submits == ["chunk_01", "chunk_02", "chunk_03"]
    run_root = tmp_path / "out" / run_id
    final = next(run_root.glob("*-voiceover-*.mp3"))
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(final)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(json.loads(probe.stdout)["format"]["duration"]) > 0
    original_manifest = (run_root / "manifest.json").read_bytes()
    (run_root / "manifest.json").unlink()

    monkeypatch.setattr(
        cli,
        "build_provider",
        lambda *_args, **_kwargs: pytest.fail("recovery must not build a paid provider"),
    )
    code, synced = _cli_json(monkeypatch, capsys, "history", "sync", run_uuid)
    assert code == 0, synced
    assert (run_root / "manifest.json").read_bytes() == original_manifest
    assert provider.submits == ["chunk_01", "chunk_02", "chunk_03"]

    code, speech = _cli_json(monkeypatch, capsys, "search", "кварцовый", "--scope", "speech")
    assert code == 0, speech
    speech_hit = next(row for row in speech["results"] if row["run_uuid"] == run_uuid)
    assert speech_hit["role"] == "speech"
    assert speech_hit["audio"]["availability"] == "present"
    assert speech_hit["audio"]["path"] == str(final)

    spec = ASRProviderSpec(
        provider_id="qwen-local",
        description="Synthetic offline ASR",
        factory=_OfflineASR,
        models=({"id": _MODEL, "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            segment_timestamps=True,
            word_timestamps=True,
            device_modes=("cpu",),
            compute_modes=("auto", "float32"),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(
        "voiceover_pipeline.providers.qwen_asr_local.qwen_asr_python_dependency_probe",
        lambda _model_id: ASRDependencyHealth(available=True, remediation=""),
    )
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider: spec)
    monkeypatch.setattr(
        cli, "transcribe_prerecorded_long_form", lambda asr, request: asr.transcribe(request)
    )
    code, asr = _cli_json(
        monkeypatch,
        capsys,
        "transcribe",
        "--audio",
        str(final),
        "--provider",
        "qwen-local",
        "--model",
        _MODEL,
        "--language",
        "ru",
        "--device",
        "cpu",
        "--compute",
        "auto",
    )
    assert code == 0, asr
    assert asr["transcript"] == _TRANSCRIPT
    asr_uuid = asr["history"]["run_uuid"]
    code, asr_search = _cli_json(monkeypatch, capsys, "search", "расшифровка", "--role", "asr")
    assert code == 0, asr_search
    asr_hit = next(row for row in asr_search["results"] if row["run_uuid"] == asr_uuid)
    assert asr_hit["role"] == "asr"
    assert (asr_hit["start_ms"], asr_hit["end_ms"]) == (0, 1200)
    assert asr_hit["audio"]["path"] == str(final)

    code, costs = _cli_json(monkeypatch, capsys, "history", "costs")
    assert code == 0, costs
    assert costs["attempts"] == 4
    assert costs["local_attempts_without_api_charge"] == 1
    assert costs["totals"][0]["known_amount"] == "0.300000000000000003"
    assert costs["totals"][0]["exact_attempts"] == 3
    assert costs["completeness"] == "complete"

    # Read-only backup/restore of a test database; never touch the user's home.
    copied = tmp_path / "backup.sqlite3"
    with sqlite3.connect(history_database_path(home)) as source:
        with sqlite3.connect(copied) as target:
            source.backup(target)
    with sqlite3.connect(copied) as restored:
        assert restored.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
        assert restored.execute("SELECT count(*) FROM attempts").fetchone()[0] == 4
        assert restored.execute("SELECT count(*) FROM text_sources").fetchone()[0] >= 2
        amounts = [
            Decimal(row[0])
            for row in restored.execute("SELECT cost FROM attempts WHERE cost IS NOT NULL")
        ]
    assert sum(amounts) == Decimal("0.300000000000000003")

    final.unlink()
    code, missing = _cli_json(monkeypatch, capsys, "search", "кварцовый", "--scope", "speech")
    assert code == 0, missing
    missing_hit = next(row for row in missing["results"] if row["run_uuid"] == run_uuid)
    assert missing_hit["audio"]["availability"] == "missing"
    assert provider.submits == ["chunk_01", "chunk_02", "chunk_03"]
