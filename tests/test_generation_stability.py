import argparse
import builtins
import hashlib
import json

import pytest
import requests
from conftest import cli_json, fixture_path, run_cli


class FakeProvider:
    def __init__(self, failures=0, raw_metadata=None):
        self.failures = failures
        self.calls = []
        self.raw_metadata = raw_metadata or {}

    def synthesize_chunk(self, text, chunk_id):
        from voiceover_pipeline.models import SynthesisResult

        self.calls.append(chunk_id)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("HTTP 503: temporary server error")
        return SynthesisResult(
            audio_bytes=b"audio",
            audio_format="mp3",
            transcript=text,
            generation_id=f"gen-{chunk_id}",
            client_path="fake",
            raw_metadata=self.raw_metadata,
        )


def make_args(tmp_path, run_id="stable-run", resume=False):
    return argparse.Namespace(
        provider="polza-tts",
        model="openai/gpt-4o-mini-tts",
        voice="ash",
        script=fixture_path("smoke_test.md"),
        output_dir=tmp_path / "out",
        run_id=run_id,
        format="markdown",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        no_trim=True,
        json_output=True,
        json_events=False,
        resume=resume,
        with_timings=False,
    )


def patch_generation_io(monkeypatch):
    import voiceover_pipeline.cli as cli

    monkeypatch.setattr(
        cli, "write_audio_as_mp3", lambda _ffmpeg, _audio, _fmt, path: path.write_bytes(b"mp3")
    )
    monkeypatch.setattr(cli, "trim_final_silence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda *_args, **_kwargs: 1000)
    monkeypatch.setattr(
        cli,
        "concat_mp3_chunks",
        lambda _ffmpeg, _chunks_dir, output_path: output_path.write_bytes(b"full"),
    )
    monkeypatch.setattr(
        cli,
        "concat_dialogue_turns",
        lambda _ffmpeg, _turns, output_path: output_path.write_bytes(b"full"),
    )
    monkeypatch.setattr(
        cli, "attach_costs", lambda _provider, _api_key, _model, _started, chunks: chunks
    )


def build_voice_bank(tmp_path):
    import hashlib
    import json
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


def test_generate_step_writes_state_and_log_after_each_chunk(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["completed_count"] == 2
    assert [item["id"] for item in state["chunks"]] == ["chunk_01", "chunk_02"]
    assert "gen-chunk_01" == state["chunks"][0]["generation_id"]
    log_text = (paths.output_root / "generation.log").read_text(encoding="utf-8")
    assert "chunk_started" in log_text
    assert "chunk_state_saved" in log_text


def test_generate_step_marks_exact_cost_unavailable_without_observed_exact(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="legacy-cost-only")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    def attach_legacy_float_cost(_provider, _api_key, _model, _started, artifacts):
        return [
            ChunkArtifact(**{**artifact.__dict__, "cost": 0.5, "cost_currency": "RUB"})
            for artifact in artifacts
        ]

    monkeypatch.setattr(cli, "attach_costs", attach_legacy_float_cost)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert manifest["cost_exact_available"] is False
    assert "cost_total_exact" not in manifest
    assert manifest["cost_total"] == 0.5


def test_resume_normalizes_malformed_exact_cost_and_keeps_float_total(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        initial_state,
        upsert_completed_chunk,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="malformed-exact", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    (paths.chunks_dir / "chunk_01.mp3").write_bytes(b"existing")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
            cost=0.5,
            cost_exact=0.1,
            cost_currency="RUB",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert manifest["cost_exact_available"] is False
    assert "cost_total_exact" not in manifest
    assert manifest["cost_total"] == 0.5


def test_resume_preserves_well_formed_exact_cost_string(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        initial_state,
        upsert_completed_chunk,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="exact-roundtrip", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    (paths.chunks_dir / "chunk_01.mp3").write_bytes(b"existing")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
            cost=0.1234567890123456789,
            cost_exact="0.1234567890123456789",
            cost_currency="RUB",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert manifest["cost_exact_available"] is True
    assert manifest["cost_total_exact"] == "0.1234567890123456789"
    assert manifest["cost_total"] == round(0.1234567890123456789, 8)


def _raw_json_response(body: str, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response._content = body.encode("utf-8")
    return response


def _wav_bytes() -> bytes:
    """A minimal valid 44-byte PCM RIFF/WAVE header with no samples."""
    return (
        b"RIFF"
        + (36).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (24000).to_bytes(4, "little")
        + (48000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (0).to_bytes(4, "little")
    )


def _polza_chat_audio_args(tmp_path, run_id, resume):
    args = make_args(tmp_path, run_id=run_id, resume=resume)
    args.provider = "polza-chat-audio"
    args.model = "polza/model"
    return args


def test_resume_keeps_history_detail_costs_persisted_in_run_state(tmp_path, monkeypatch):
    """A history-detail cost observed after the chunk save survives a resume.

    The detail lookup is a real raw JSON body parsed by
    ``pricing.fetch_polza_generation_detail`` with ``parse_float=Decimal``, so the
    exact cost crosses the provider boundary instead of arriving as a
    monkeypatched dict. The resumed run cannot obtain any detail, so its exact
    total must come from the run state written after the first run's lookup.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.pricing as pricing
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    real_attach_costs = cli.attach_costs
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "attach_costs", real_attach_costs)

    exact = "0.1234567890123456789"
    detail_bodies = {
        "gen-chunk_01": (
            '{"id": "gen-chunk_01", "model": "polza/model", '
            f'"clientCost": {exact}, "generationTimeMs": 1500, '
            '"usage": {"cost_rub": 0.00000000000000000001, "tokens": 3}, '
            '"createdAt": "2026-01-01T00:00:00Z"}'
        ),
        "gen-chunk_02": '{"id": "gen-chunk_02", "model": "polza/model", "clientCost": 0}',
    }
    requested: list[str | None] = []

    def history_get(url, headers=None, params=None, timeout=None):
        generation_id = url.rsplit("/", 1)[-1]
        requested.append(generation_id)
        body = detail_bodies.get(generation_id)
        return _raw_json_response(body) if body else _raw_json_response("{}", 404)

    monkeypatch.setattr(pricing.requests, "get", history_get)

    args = _polza_chat_audio_args(tmp_path, "money-state", resume=False)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    provider = FakeProvider()

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_01", "chunk_02"]
    assert requested == ["gen-chunk_01", "gen-chunk_02"]
    state_path = paths.output_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    state_by_id = {item["id"]: item for item in state["chunks"]}
    assert state_by_id["chunk_01"]["cost_exact"] == exact
    assert state_by_id["chunk_01"]["cost_rub_exact"] == exact
    assert state_by_id["chunk_01"]["cost_currency"] == "RUB"
    assert state_by_id["chunk_01"]["cost"] == float(exact)
    assert state_by_id["chunk_01"]["usage"] == {"cost_rub": 1e-20, "tokens": 3}
    assert state_by_id["chunk_01"]["generation_time_ms"] == 1500
    assert state_by_id["chunk_01"]["generated_at"] == "2026-01-01T00:00:00Z"
    assert state_by_id["chunk_01"]["generation_detail_source"] == (
        "Polza GET /api/v1/history/generations/{id}"
    )
    assert state_by_id["chunk_02"]["cost_exact"] == "0"
    assert state_by_id["chunk_02"]["cost"] == 0.0
    json.dumps(state, allow_nan=False)

    manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert manifest["cost_exact_available"] is True
    assert manifest["cost_total_exact"] == exact
    assert manifest["cost_total"] == round(float(exact), 8)

    monkeypatch.setattr(
        pricing.requests, "get", lambda *args, **kwargs: _raw_json_response("{}", 404)
    )
    resume_args = _polza_chat_audio_args(tmp_path, "money-state", resume=True)
    resume_provider = FakeProvider()

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            resume_provider,
            "ffmpeg",
            "ffprobe",
            chunks,
            "key",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    assert resume_provider.calls == []
    resumed_state = json.loads(state_path.read_text(encoding="utf-8"))
    resumed_by_id = {item["id"]: item for item in resumed_state["chunks"]}
    assert resumed_by_id["chunk_01"]["cost_exact"] == exact
    assert resumed_by_id["chunk_01"]["cost_currency"] == "RUB"
    assert resumed_by_id["chunk_01"]["usage"] == {"cost_rub": 1e-20, "tokens": 3}
    assert resumed_by_id["chunk_02"]["cost_exact"] == "0"
    json.dumps(resumed_state, allow_nan=False)
    resumed_manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert resumed_manifest["cost_exact_available"] is True
    assert resumed_manifest["cost_total_exact"] == exact
    assert resumed_manifest["cost_total"] == round(float(exact), 8)


def test_generate_step_persists_openrouter_detail_cost_in_run_state(tmp_path, monkeypatch):
    """The same late-cost persistence covers the OpenRouter detail boundary.

    Phase 1 observes the cost from a real unquoted raw JSON body, the shape
    ``pricing.fetch_openrouter_generation_detail`` parses with
    ``parse_float=Decimal``. A resume whose detail lookup is unavailable must not
    synthesize again and must keep the previously observed exact total,
    currency, and source from the trusted run state.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.pricing as pricing
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    real_attach_costs = cli.attach_costs
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "attach_costs", real_attach_costs)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    body = (
        '{"data": {"id": "gen-chunk_01", "total_cost": 0.0001234567890123456789, '
        '"generationTimeMs": 1200, "usage": {"prompt_tokens": 5}, '
        '"createdAt": "2026-01-02T00:00:00Z"}}'
    )
    monkeypatch.setattr(pricing.requests, "get", lambda *args, **kwargs: _raw_json_response(body))

    args = make_args(tmp_path, run_id="openrouter-state")
    args.provider = "openrouter-tts"
    args.model = "openrouter/model"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    entry = state["chunks"][0]
    assert entry["generation_id"] == "gen-chunk_01"
    assert entry["cost_exact"] == "0.0001234567890123456789"
    assert entry["cost"] == 0.0001234567890123456789
    assert entry["cost_currency"] == "USD"
    assert entry["usage"] == {"prompt_tokens": 5}
    assert entry["generation_time_ms"] == 1200
    assert entry["generated_at"] == "2026-01-02T00:00:00Z"
    json.dumps(state, allow_nan=False)
    manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert manifest["cost_total_exact"] == "0.0001234567890123456789"
    assert manifest["cost_currency"] == "USD"

    monkeypatch.setattr(
        pricing.requests, "get", lambda *args, **kwargs: _raw_json_response("{}", 404)
    )
    resume_args = make_args(tmp_path, run_id="openrouter-state", resume=True)
    resume_args.provider = "openrouter-tts"
    resume_args.model = "openrouter/model"
    resume_provider = FakeProvider()

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            resume_provider,
            "ffmpeg",
            "ffprobe",
            chunks,
            "key",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    assert resume_provider.calls == []
    resumed_state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    resumed_entry = resumed_state["chunks"][0]
    assert resumed_entry["generation_id"] == "gen-chunk_01"
    assert resumed_entry["cost_exact"] == "0.0001234567890123456789"
    assert resumed_entry["cost_currency"] == "USD"
    assert resumed_entry["usage"] == {"prompt_tokens": 5}
    json.dumps(resumed_state, allow_nan=False)
    resumed_manifest = json.loads(paths.chunks_json.read_text(encoding="utf-8"))
    assert resumed_manifest["cost_exact_available"] is True
    assert resumed_manifest["cost_total_exact"] == "0.0001234567890123456789"
    assert resumed_manifest["cost_total"] == round(0.0001234567890123456789, 8)
    assert resumed_manifest["cost_currency"] == "USD"


def test_openrouter_foreign_declared_id_never_reaches_state_or_manifest(tmp_path, monkeypatch):
    """A detail declaring another generation id must not charge this run.

    The lookup is addressed by the chunk's own ``generation_id``, so a raw body
    declaring ``gen-other`` must leave the state's generation id and absent cost
    untouched and keep the manifest total unknown instead of adopting the
    foreign price.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.pricing as pricing
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    real_attach_costs = cli.attach_costs
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "attach_costs", real_attach_costs)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    lookups: list[tuple[str, dict | None]] = []

    def fake_get(url, headers=None, params=None, timeout=None):
        lookups.append((url, params))
        return _raw_json_response('{"data": {"id": "gen-other", "total_cost": 0.5}}')

    monkeypatch.setattr(pricing.requests, "get", fake_get)

    args = make_args(tmp_path, run_id="openrouter-foreign")
    args.provider = "openrouter-tts"
    args.model = "openrouter/model"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    # One targeted lookup by the chunk's own id; the returned foreign id is not
    # retried and never requested.
    assert len(lookups) == 1
    assert lookups[0][0].endswith("/generation")
    assert lookups[0][1] == {"id": "gen-chunk_01"}
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    entry = state["chunks"][0]
    assert entry["generation_id"] == "gen-chunk_01"
    assert entry.get("cost") is None
    assert entry.get("cost_exact") is None
    assert entry.get("cost_currency") is None
    json.dumps(state, allow_nan=False)
    manifest_text = paths.chunks_json.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["cost_exact_available"] is False
    assert manifest.get("cost_total") is None
    assert manifest.get("cost_currency") is None
    assert "gen-other" not in manifest_text


def test_generate_step_persists_the_public_omnivoice_runtime_receipt(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    receipt = {
        "model_id": "audio-cpp/omnivoice-q8_0",
        "sha256": "2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b",
        "quantization": "Q8_0 GGUF",
        "license": "CC-BY-NC-4.0 upstream weights; local noncommercial research only",
        "provenance": "audio-cpp/audio.cpp-gguf@fixture; converted from OmniVoice",
    }
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    args = make_args(tmp_path, run_id="omnivoice-receipt")
    args.provider = "omnivoice-local"
    args.model = "audio-cpp/omnivoice-q8_0"
    args.voice = "built-in-female-style-condition"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args,
            FakeProvider(
                raw_metadata={
                    "runtime_receipt": receipt,
                    "voice_selection": {
                        "kind": "built-in-style-condition",
                        "condition": "female",
                        "named_preset": False,
                        "voice_cloning": False,
                        "voice_design": False,
                    },
                    "voice_session": {
                        "strategy": "single-native-invocation-internal-text-chunking",
                        "seed": 1234,
                        "internal_text_chunk_size": 420,
                    },
                }
            ),
            "ffmpeg",
            "ffprobe",
            chunks,
            "",
            None,
            paths,
            None,
            "none",
        )

    assert exit_info.value.code == 0
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(paths.run_json.read_text(encoding="utf-8"))
    assert state["chunks"][0]["runtime_receipt"] == receipt
    assert run_manifest["chunks"][0]["runtime_receipt"] == receipt
    assert state["chunks"][0]["voice_selection"]["condition"] == "female"
    assert state["chunks"][0]["voice_session"] == {
        "strategy": "single-native-invocation-internal-text-chunking",
        "seed": 1234,
        "internal_text_chunk_size": 420,
    }
    assert run_manifest["chunks"][0]["voice_selection"] == state["chunks"][0]["voice_selection"]
    assert run_manifest["chunks"][0]["voice_session"] == state["chunks"][0]["voice_session"]
    assert run_manifest["execution_source"] == state["execution_source"]
    assert run_manifest["execution_source"]["package_version"] == "0.6.1"
    assert str(tmp_path) not in json.dumps(run_manifest["execution_source"])
    assert str(tmp_path) not in json.dumps(run_manifest["chunks"][0]["runtime_receipt"])


def test_generate_step_retries_retryable_provider_errors(tmp_path, monkeypatch):
    """A local engine keeps the documented retry behavior for transient failures."""
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.retry as retry
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(retry.time, "sleep", lambda _delay: None)
    args = make_args(tmp_path, run_id="local-retry")
    args.provider = "qwen-local"
    args.model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = FakeProvider(failures=1)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_01", "chunk_01"]
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert "pending_attempt" not in state


def _paid_submit_args(tmp_path, run_id, *, resume=False, model=None, voice=None):
    args = make_args(tmp_path, run_id=run_id, resume=resume)
    if model is not None:
        args.model = model
    if voice is not None:
        args.voice = voice
    return args


def _paid_timeout_post(posts):
    def post(url, **_kwargs):
        posts.append(url)
        raise requests.Timeout("read timed out")

    return post


def test_paid_submit_timeout_never_retries_and_blocks_resume(tmp_path, monkeypatch):
    """A paid submit whose response never arrived is never sent twice.

    ``--retries 3`` no longer re-sends a paid POST, the failed attempt stays in
    run state, and the same run cannot be resumed into a second POST for that
    chunk.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    monkeypatch.setattr(polza_tts.requests, "post", _paid_timeout_post(posts))

    args = _paid_submit_args(tmp_path, "paid-submit-timeout")
    assert args.retries == 3
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice)

    with pytest.raises(cli.CliError, match="Failed to synthesize chunk_01") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert len(posts) == 1
    state_path = paths.output_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    attempt = state["pending_attempt"]
    assert set(attempt) == {"id", "number", "status", "at"}
    assert attempt["id"] == "chunk_01"
    assert attempt["number"] == 1
    assert attempt["status"] == "outcome_unknown"
    assert state["chunks"] == []
    assert "timed out" not in json.dumps(attempt)

    resume_args = _paid_submit_args(tmp_path, "paid-submit-timeout", resume=True)
    resumed_provider = PolzaTTSProvider(
        api_key="sk-test", model=resume_args.model, voice=resume_args.voice
    )

    with pytest.raises(cli.CliError, match="unconfirmed paid submit") as resume_error:
        cli._generate_step(
            resume_args,
            resumed_provider,
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_error.value.code == 30
    assert resume_error.value.details["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert len(posts) == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))["chunks"] == []


def test_paid_chat_audio_submit_timeout_never_retries(tmp_path, monkeypatch):
    """The other paid Polza route obeys the same single-submit policy."""
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_chat_audio as polza_chat_audio
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_chat_audio import PolzaChatAudioProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    monkeypatch.setattr(polza_chat_audio.requests, "post", _paid_timeout_post(posts))

    args = _paid_submit_args(tmp_path, "paid-chat-audio-timeout")
    args.provider = "polza-chat-audio"
    args.model = "openai/gpt-audio-mini"
    args.fallback_voice = args.voice
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = PolzaChatAudioProvider(
        api_key="sk-test",
        model=args.model,
        voice=args.voice,
        fallback_voice=args.fallback_voice,
    )

    with pytest.raises(cli.CliError, match="Failed to synthesize chunk_01") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert len(posts) == 1
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["pending_attempt"]["status"] == "outcome_unknown"
    assert state["pending_attempt"]["id"] == "chunk_01"


def test_paid_media_poll_failure_makes_no_second_submit(tmp_path, monkeypatch, capsys):
    """A known media task id is stored before the first poll and never re-submitted.

    The accepted id is on disk before the first GET, so the failed poll leaves a
    marker that a later ``--resume`` finishes with GET calls only: the paid
    submit is counted once and is not repeated for that chunk.
    """
    import sys

    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    state_at_first_poll: list[dict] = []

    def submit(url, **_kwargs):
        posts.append(url)
        return _raw_json_response('{"id": "task-1", "status": "pending"}')

    def poll_timeout(*_args, **_kwargs):
        state_at_first_poll.append(
            json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
        )
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(polza_tts.requests, "post", submit)
    monkeypatch.setattr(polza_tts.requests, "get", poll_timeout)

    args = _paid_submit_args(
        tmp_path, "paid-media-poll", model="elevenlabs/text-to-speech-turbo-2-5", voice="Rachel"
    )
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice)

    with pytest.raises(cli.CliError, match="Failed to synthesize chunk_01") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/media"]
    state_path = paths.output_root / "run_state.json"
    # Persisted before the first poll: the accepted id is already in the marker.
    assert state_at_first_poll[0]["pending_attempt"]["remote_task_id"] == "task-1"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pending_attempt"]["status"] == "outcome_unknown"
    assert state["pending_attempt"]["remote_task_id"] == "task-1"

    signed_url = "https://cdn.example.com/paid.mp3?token=sk-live-secret-12345"

    def resume_get(url, **_kwargs):
        if url.endswith("/media/task-1"):
            return _raw_json_response(
                json.dumps(
                    {
                        "status": "completed",
                        "data": {"url": signed_url},
                        "usage": {"cost_rub": 0.3},
                    }
                )
            )
        return _raw_json_response("recovered-audio")

    monkeypatch.setattr(polza_tts.requests, "get", resume_get)
    built: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "build_provider",
        lambda *_args, **_kwargs: (
            built.append("built")
            or PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--limit-chunks",
            "1",
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as resume_exit:
        cli.main()

    assert resume_exit.value.code == 0
    assert built == ["built"]
    # The whole resume used GET calls: the paid submit was not sent again.
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/media"]
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert "pending_attempt" not in resumed
    assert resumed["chunks"][0]["id"] == "chunk_01"
    assert resumed["chunks"][0]["cost_exact"] == "0.3"
    assert signed_url not in state_path.read_text(encoding="utf-8")
    assert "sk-live-secret-12345" not in capsys.readouterr().out


def test_paid_media_download_timeout_keeps_exact_cost_for_get_only_recovery(tmp_path, monkeypatch):
    """A completed paid usage cost survives a failed signed-URL download.

    The completed poll reports the billed cost before the download, so the
    marker keeps that exact amount when the download times out. A resume then
    finishes the same task with GET calls only, and a completion payload without
    usage still leaves the observed cost on the saved chunk.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    state_before_download: list[dict] = []
    signed_url = "https://cdn.example.com/paid.mp3?token=sk-live-secret-12345"

    def submit(url, **_kwargs):
        posts.append(url)
        return _raw_json_response('{"id": "task-1", "status": "pending"}')

    def get_before_resume(url, **_kwargs):
        if url.endswith("/media/task-1"):
            return _raw_json_response(
                json.dumps(
                    {
                        "status": "completed",
                        "data": {"url": signed_url},
                        "usage": {"cost_rub": 0.3},
                    }
                )
            )
        state_before_download.append(
            json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
        )
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(polza_tts.requests, "post", submit)
    monkeypatch.setattr(polza_tts.requests, "get", get_before_resume)

    args = _paid_submit_args(
        tmp_path, "paid-media-download", model="elevenlabs/text-to-speech-turbo-2-5", voice="Rachel"
    )
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(cli.CliError, match="Failed to synthesize chunk_01") as error:
        cli._generate_step(
            args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert error.value.code == 30
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/media"]
    # The exact cost was already in the marker when the download was attempted.
    assert state_before_download[0]["pending_attempt"]["cost_exact"] == "0.3"
    state_path = paths.output_root / "run_state.json"
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["status"] == "outcome_unknown"
    assert marker["remote_task_id"] == "task-1"
    assert marker["cost"] == 0.3
    assert marker["cost_exact"] == "0.3"

    def resume_get(url, **_kwargs):
        # The recovered completion payload omits usage on purpose.
        if url.endswith("/media/task-1"):
            return _raw_json_response(
                json.dumps({"status": "completed", "data": {"url": signed_url}})
            )
        return _raw_json_response("recovered-audio")

    monkeypatch.setattr(polza_tts.requests, "get", resume_get)
    resume_args = _paid_submit_args(
        tmp_path, "paid-media-download", resume=True, model=args.model, voice=args.voice
    )

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    assert len(posts) == 1
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert "pending_attempt" not in resumed
    assert resumed["chunks"][0]["cost"] == 0.3
    assert resumed["chunks"][0]["cost_exact"] == "0.3"
    assert resumed["chunks"][0]["cost_rub_exact"] == "0.3"


@pytest.mark.parametrize(
    ("case_name", "overrides", "remote_task_id"),
    [
        ("untrusted_id", {}, "https://cdn.example.com/paid.mp3?token=sk-live-secret-12345"),
        ("mismatched_identity", {"model": "elevenlabs/text-to-speech-turbo-2-5"}, "task-1"),
        ("other_provider", {"provider": "polza-chat-audio"}, "task-1"),
    ],
)
def test_media_recovery_marker_mismatch_blocks_resume_before_provider_work(
    tmp_path, monkeypatch, capsys, case_name, overrides, remote_task_id
):
    """Only a matching, bounded media id may skip the paid-submit block.

    A tampered id, another provider/model/voice/script, or any other marker keeps
    the documented PAID_SUBMIT_UNCONFIRMED failure before any key read, provider
    build, pricing lookup, or network request.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id=f"paid-recovery-{case_name}", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    if remote_task_id == "task-1":
        record_media_task_accepted(
            state, task_id=remote_task_id, chunk_id=chunks[0].id, number=chunks[0].number
        )
    else:
        # The writer refuses this id, so it is written directly the way an
        # edited state file can hold it.
        state["pending_attempt"]["remote_task_id"] = remote_task_id
    for key, value in overrides.items():
        state[key] = value
    atomic_write_json(paths.output_root / "run_state.json", state)

    key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert payload["details"]["chunk_id"] == "chunk_01"
    assert key_reads == []
    assert built == []
    assert priced == []
    assert "sk-live-secret-12345" not in json.dumps(payload)


def test_media_recovery_marker_for_later_chunk_keeps_the_paid_block(tmp_path, monkeypatch, capsys):
    """A stored attempt that is not the in-flight chunk is not resumed.

    The marker names ``chunk_02`` while ``chunk_01`` is still unfinished, so it
    cannot be the paid attempt that was in flight. Overwriting it could pay for
    ``chunk_02`` twice, so the documented block wins before any provider work.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="paid-recovery-later-chunk", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    second = chunks[1]
    begin_chunk_attempt(state, chunk_id=second.id, number=second.number)
    record_media_task_accepted(state, task_id="task-2", chunk_id=second.id, number=second.number)
    atomic_write_json(paths.output_root / "run_state.json", state)

    key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []
    monkeypatch.setattr(cli, "read_api_key", lambda _args: key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_02",
        "chunk_number": 2,
        "attempt_status": "submitting",
    }
    assert key_reads == []
    assert built == []
    assert priced == []


def test_recovery_marker_for_other_chunk_survives_missing_completed_audio(tmp_path, monkeypatch):
    """An in-flight marker is not overwritten when an earlier chunk lost its audio.

    State says ``chunk_01`` completed but its file is gone, so the fresh submit
    for it would erase the stored ``chunk_02`` paid id. The documented block wins
    before that write, and the marker keeps its accepted id.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="paid-recovery-missing-audio", resume=True)
    args.model = "elevenlabs/text-to-speech-turbo-2-5"
    args.voice = "Rachel"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    state["chunks"] = [
        {"status": "completed", "number": 1, "id": "chunk_01", "file": "chunk_01.mp3"}
    ]
    state["completed_count"] = 1
    second = chunks[1]
    begin_chunk_attempt(state, chunk_id=second.id, number=second.number)
    record_media_task_accepted(state, task_id="task-2", chunk_id=second.id, number=second.number)
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)

    posts: list[str] = []
    monkeypatch.setattr(polza_tts.requests, "post", lambda url, **_kw: posts.append(url))

    with pytest.raises(cli.CliError, match="unconfirmed paid submit") as error:
        cli._generate_step(
            args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert error.value.code == 30
    assert error.value.details["chunk_id"] == "chunk_02"
    assert posts == []
    kept = json.loads(state_path.read_text(encoding="utf-8"))
    assert kept["pending_attempt"]["remote_task_id"] == "task-2"


def test_non_media_model_fake_id_blocks_resume_and_status(tmp_path, monkeypatch, capsys):
    """A paid marker on the ``/audio/speech`` route is never a media recovery.

    ``openai/gpt-4o-mini-tts`` submits synchronously, so a stored
    ``remote_task_id`` cannot name an async media task. ``--resume`` and
    ``status`` must both keep the documented paid-submit block instead of
    issuing GET ``/media/task-1`` or promising a resume.
    """
    import sys

    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="non-media-fake-id", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    record_media_task_accepted(
        state, task_id="task-1", chunk_id=chunks[0].id, number=chunks[0].number
    )
    atomic_write_json(paths.output_root / "run_state.json", state)

    key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []
    media_gets: list[str] = []
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        cli,
        "build_provider",
        lambda *_args, **_kwargs: (
            built.append("built")
            or PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice)
        ),
    )
    monkeypatch.setattr(
        polza_tts.requests,
        "get",
        lambda url, **_kwargs: media_gets.append(url) or _raw_json_response("{}"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert key_reads == []
    assert built == []
    assert priced == []
    assert media_gets == []

    code, data = cli_json(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        args.run_id,
        "--json",
    )
    assert code == 0
    assert data["can_resume"] is False
    assert data["resume_block_reason"] == "paid_submit_unconfirmed"


def test_paid_chunk_write_failure_recovers_saved_raw_without_resubmit(tmp_path, monkeypatch):
    """An accepted paid response that cannot be converted is rebuilt from saved raw.

    The raw bytes and their bounded receipt are persisted before FFmpeg, so a
    failed conversion leaves a recoverable attempt instead of one that blocks a
    resume. The resume rebuilds the chunk from those exact bytes with no provider
    call, and the raw file stays as the rebuild source.
    """
    import hashlib

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    write_calls = {"count": 0}

    def flaky_write(_ffmpeg, _audio, _fmt, path):
        write_calls["count"] += 1
        if write_calls["count"] == 1:
            raise OSError("disk full")
        path.write_bytes(b"mp3")

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)

    args = _paid_submit_args(tmp_path, "paid-write-failure")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = FakeProvider()

    with pytest.raises(cli.CliError, match="Failed to write chunk audio") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert error.value.code == 50
    assert provider.calls == ["chunk_01"]
    state_path = paths.output_root / "run_state.json"
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["status"] == "raw_saved"
    # The raw bytes and receipt were persisted before the conversion failed.
    raw_path = paths.output_root / "raw" / "chunk_01.mp3"
    assert raw_path.read_bytes() == b"audio"
    assert marker["raw"] == {
        "format": "mp3",
        "path": "raw/chunk_01.mp3",
        "sha256": hashlib.sha256(b"audio").hexdigest(),
        "generation_id": "gen-chunk_01",
    }
    assert "disk full" not in json.dumps(marker)

    # The resume rebuilds the chunk from saved raw; no provider call is made.
    resume_args = _paid_submit_args(tmp_path, "paid-write-failure", resume=True)
    resumed_provider = FakeProvider()

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            resumed_provider,
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    assert resumed_provider.calls == []
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert "pending_attempt" not in resumed
    assert resumed["chunks"][0]["generation_id"] == "gen-chunk_01"
    # The paid raw file is a rebuild source, not a disposable cache.
    assert raw_path.read_bytes() == b"audio"


def test_paid_media_ffmpeg_failure_resumes_from_raw_without_any_provider_call(
    tmp_path, monkeypatch
):
    """A completed paid media download survives an FFmpeg failure without a new submit.

    The signed-URL download succeeds and the exact cost is already in the marker;
    the raw MP3 is saved before conversion, so the failed conversion leaves a
    recoverable attempt. The resume rebuilds from raw with zero POST and zero
    provider GET, and the exact cost stays on the saved chunk.
    """
    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    get_urls: list[str] = []
    signed_url = "https://cdn.example.com/paid.mp3?token=sk-live-secret-12345"

    def submit(url, **_kwargs):
        posts.append(url)
        return _raw_json_response('{"id": "task-1", "status": "pending"}')

    def get(url, **_kwargs):
        get_urls.append(url)
        if url.endswith("/media/task-1"):
            return _raw_json_response(
                json.dumps(
                    {
                        "status": "completed",
                        "data": {"url": signed_url},
                        "usage": {"cost_rub": 0.3},
                    }
                )
            )
        return _raw_json_response("recovered-audio")

    monkeypatch.setattr(polza_tts.requests, "post", submit)
    monkeypatch.setattr(polza_tts.requests, "get", get)

    write_calls = {"count": 0}

    def flaky_write(_ffmpeg, _audio, _fmt, path):
        write_calls["count"] += 1
        if write_calls["count"] == 1:
            raise OSError("ffmpeg exploded")
        path.write_bytes(b"mp3")

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)

    args = _paid_submit_args(
        tmp_path, "paid-media-raw", model="elevenlabs/text-to-speech-turbo-2-5", voice="Rachel"
    )
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(cli.CliError, match="Failed to write chunk audio") as error:
        cli._generate_step(
            args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert error.value.code == 50
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/media"]
    state_path = paths.output_root / "run_state.json"
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["status"] == "raw_saved"
    assert marker["remote_task_id"] == "task-1"
    assert marker["cost_exact"] == "0.3"
    raw_path = paths.output_root / "raw" / "chunk_01.mp3"
    assert raw_path.read_bytes() == b"recovered-audio"
    assert signed_url not in state_path.read_text(encoding="utf-8")

    posts_before = list(posts)
    gets_before = list(get_urls)
    resume_args = _paid_submit_args(
        tmp_path, "paid-media-raw", resume=True, model=args.model, voice=args.voice
    )

    def unexpected_synthesis(*_args, **_kwargs):
        raise AssertionError("a recovered paid attempt must not reach the submit seam")

    monkeypatch.setattr(cli, "synthesize_part", unexpected_synthesis)

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    # The resume used the saved raw bytes: no new POST and no provider GET.
    assert posts == posts_before
    assert get_urls == gets_before
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert "pending_attempt" not in resumed
    assert resumed["chunks"][0]["cost_exact"] == "0.3"
    assert resumed["chunks"][0]["cost_rub_exact"] == "0.3"
    assert raw_path.read_bytes() == b"recovered-audio"


def test_audio_speech_wav_response_is_receipted_and_converted_as_wav(tmp_path, monkeypatch):
    """A real RIFF/WAVE response without a content type is receipted and converted as WAV.

    The synchronous ``/audio/speech`` route may omit ``contentType`` while still
    returning real WAV bytes. The raw receipt must record WAV, and the conversion
    must decode the container through FFmpeg instead of fast-copying it into the
    ``.mp3`` output or piping it through the raw s16le demuxer.
    """
    import base64
    import subprocess as subprocess_module
    from pathlib import Path

    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.media as media
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    # Use the real conversion so the RIFF/WAVE routing is exercised end to end.
    monkeypatch.setattr(cli, "write_audio_as_mp3", media.write_audio_as_mp3)
    ffmpeg_calls: list[list[str]] = []

    class _FakeMediaSubprocess:
        PIPE = subprocess_module.PIPE

        @staticmethod
        def run(command, **_kwargs):
            ffmpeg_calls.append(list(command))
            # Mimic ffmpeg writing the requested output file.
            Path(command[-1]).write_bytes(b"mp3")
            return subprocess_module.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    # Scope the fake to media so unrelated subprocess callers keep working.
    monkeypatch.setattr(media, "subprocess", _FakeMediaSubprocess())

    wav_bytes = _wav_bytes()
    body = json.dumps({"audio": base64.b64encode(wav_bytes).decode(), "usage": {"cost_rub": 0.3}})
    monkeypatch.setattr(polza_tts.requests, "post", lambda url, **_kwargs: _raw_json_response(body))

    args = _paid_submit_args(tmp_path, "paid-audio-speech-wav")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = PolzaTTSProvider(
        api_key="sk-test", model=args.model, voice=args.voice, response_format="mp3"
    )

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    # The receipt path follows the actual container, not the requested format.
    assert (paths.output_root / "raw" / "chunk_01.wav").read_bytes() == wav_bytes
    assert not (paths.output_root / "raw" / "chunk_01.mp3").exists()
    assert ffmpeg_calls
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-i") + 1].endswith(".wav")
    assert all("s16le" not in call for call in ffmpeg_calls)
    # The ``.mp3`` output is the converted result, never the raw WAV bytes.
    assert (paths.chunks_dir / "chunk_01.mp3").read_bytes() == b"mp3"


def test_audio_speech_direct_cost_survives_ffmpeg_failure_and_raw_resume(tmp_path, monkeypatch):
    """The direct usage cost is stored with the raw receipt, not after conversion.

    ``/audio/speech`` reports its billed amount only in the synchronous response.
    It must join the same atomic marker write as the raw receipt so a later FFmpeg
    failure cannot lose it: the raw-only resume keeps the exact cost on the
    completed chunk with no new POST.
    """
    import base64

    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.providers.polza_tts import PolzaTTSProvider
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    body = json.dumps(
        {
            "audio": base64.b64encode(b"paid-mp3").decode(),
            "contentType": "audio/mpeg",
            "usage": {"cost_rub": 0.3},
        }
    )

    def post(url, **_kwargs):
        posts.append(url)
        return _raw_json_response(body)

    monkeypatch.setattr(polza_tts.requests, "post", post)

    write_calls = {"count": 0}

    def flaky_write(_ffmpeg, _audio, _fmt, path):
        write_calls["count"] += 1
        if write_calls["count"] == 1:
            raise OSError("ffmpeg exploded")
        path.write_bytes(b"mp3")

    monkeypatch.setattr(cli, "write_audio_as_mp3", flaky_write)

    args = _paid_submit_args(tmp_path, "paid-audio-speech-cost")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]

    with pytest.raises(cli.CliError, match="Failed to write chunk audio") as error:
        cli._generate_step(
            args,
            PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert error.value.code == 50
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/audio/speech"]
    state_path = paths.output_root / "run_state.json"
    marker = json.loads(state_path.read_text(encoding="utf-8"))["pending_attempt"]
    assert marker["status"] == "raw_saved"
    # The direct usage cost landed in the same atomic write as the raw receipt.
    assert marker["cost"] == 0.3
    assert marker["cost_exact"] == "0.3"
    assert (paths.output_root / "raw" / "chunk_01.mp3").read_bytes() == b"paid-mp3"

    resume_args = _paid_submit_args(tmp_path, "paid-audio-speech-cost", resume=True)

    with pytest.raises(SystemExit) as resume_exit:
        cli._generate_step(
            resume_args,
            PolzaTTSProvider(api_key="sk-test", model=resume_args.model, voice=resume_args.voice),
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_exit.value.code == 0
    # The raw-only recovery sent no second POST.
    assert posts == [f"{polza_tts.POLZA_BASE_URL}/audio/speech"]
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert "pending_attempt" not in resumed
    assert resumed["chunks"][0]["cost_exact"] == "0.3"
    assert resumed["chunks"][0]["cost_rub"] == 0.3


def test_corrupt_saved_raw_receipt_blocks_resume_instead_of_resubmitting(tmp_path, monkeypatch):
    """A saved raw file whose digest no longer matches fails closed.

    The receipt is evidence that a paid response exists, so a corrupt or replaced
    file must block the run rather than be treated as a fresh submit.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    def failing_write(_ffmpeg, _audio, _fmt, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_audio_as_mp3", failing_write)

    args = _paid_submit_args(tmp_path, "paid-corrupt-raw")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = FakeProvider()

    with pytest.raises(cli.CliError, match="Failed to write chunk audio"):
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )
    assert provider.calls == ["chunk_01"]

    # The paid raw file is replaced after the receipt was recorded.
    (paths.output_root / "raw" / "chunk_01.mp3").write_bytes(b"replaced")

    resume_args = _paid_submit_args(tmp_path, "paid-corrupt-raw", resume=True)
    resumed_provider = FakeProvider()

    with pytest.raises(cli.CliError, match="unconfirmed paid submit") as resume_error:
        cli._generate_step(
            resume_args,
            resumed_provider,
            "ffmpeg",
            "ffprobe",
            chunks,
            "sk-test",
            None,
            paths,
            None,
            "auto",
        )

    assert resume_error.value.code == 30
    assert resumed_provider.calls == []


def test_unconfirmed_paid_submit_blocks_resume_with_documented_json_error(
    tmp_path, monkeypatch, capsys
):
    """The blocked resume keeps the documented error envelope and exit code 30."""
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="paid-blocked-json", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    atomic_write_json(paths.output_root / "run_state.json", state)

    provider = FakeProvider()
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    assert provider.calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["code"] == 30
    assert "unconfirmed paid submit" in payload["error"]
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_01",
        "chunk_number": 1,
        "attempt_status": "submitting",
    }


def test_paid_submit_overwrite_after_timeout_is_refused_and_keeps_evidence(
    tmp_path, monkeypatch, capsys
):
    """A run that only holds an unconfirmed paid attempt is never overwritten.

    The timeout leaves a marker and zero MP3, so ``--overwrite
    --confirm-delete-paid-audio`` must not delete that evidence and must not
    reach provider construction or a second POST. A different ``--run-id``
    remains the explicit way to start a new attempt.
    """
    import sys

    import voiceover_pipeline.cli as cli
    import voiceover_pipeline.providers.polza_tts as polza_tts
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    posts: list[str] = []
    monkeypatch.setattr(polza_tts.requests, "post", _paid_timeout_post(posts))

    args = _paid_submit_args(tmp_path, "paid-overwrite")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    provider = polza_tts.PolzaTTSProvider(api_key="sk-test", model=args.model, voice=args.voice)

    with pytest.raises(cli.CliError, match="Failed to synthesize chunk_01"):
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert len(posts) == 1
    state_path = paths.output_root / "run_state.json"
    assert "pending_attempt" in json.loads(state_path.read_text(encoding="utf-8"))
    assert not any(paths.chunks_dir.glob("chunk_*.mp3"))

    built: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--overwrite",
            "--confirm-delete-paid-audio",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as overwrite_exit:
        cli.main()

    assert overwrite_exit.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert "unconfirmed paid submit" in payload["error"]
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_01",
        "chunk_number": 1,
        "attempt_status": "outcome_unknown",
    }
    assert built == []
    assert len(posts) == 1
    assert paths.output_root.exists()
    assert "pending_attempt" in json.loads(state_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            "paid-overwrite-fresh",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as fresh_exit:
        cli.main()

    assert fresh_exit.value.code == 0
    fresh_paths = build_run_paths(args.output_dir, args.model, "paid-overwrite-fresh")
    assert (fresh_paths.output_root / "run_state.json").exists()
    # The refused run keeps its own evidence while the new run starts separately.
    assert "pending_attempt" in json.loads(state_path.read_text(encoding="utf-8"))


def test_overwrite_reports_unconfirmed_marker_before_paid_audio_confirmation(
    tmp_path, monkeypatch, capsys
):
    """A saved MP3 must not mask the documented PAID_SUBMIT_UNCONFIRMED refusal.

    The run already saved ``chunk_01.mp3`` and later left an unconfirmed attempt
    for ``chunk_02``. ``--overwrite`` without ``--confirm-delete-paid-audio`` used
    to report the generic paid-audio delete error before the marker check, hiding
    the documented ``details.error_code``. The marker check now runs first: exit
    code 30, bounded details, no provider build, no POST, and the completed MP3
    and state stay intact.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        ATTEMPT_OUTCOME_UNKNOWN,
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_chunk_attempt_outcome,
        upsert_completed_chunk,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = _paid_submit_args(tmp_path, "paid-marker-over-mp3")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    chunk_audio = paths.chunks_dir / "chunk_01.mp3"
    chunk_audio.write_bytes(b"paid")
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    begin_chunk_attempt(state, chunk_id="chunk_02", number=2)
    record_chunk_attempt_outcome(state, status=ATTEMPT_OUTCOME_UNKNOWN)
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)
    original_state = state_path.read_text(encoding="utf-8")

    built: list[str] = []
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--overwrite",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_02",
        "chunk_number": 2,
        "attempt_status": "outcome_unknown",
    }
    assert built == []
    assert chunk_audio.read_bytes() == b"paid"
    assert state_path.read_text(encoding="utf-8") == original_state
    assert paths.output_root.exists()


def test_overwrite_refuses_unreadable_run_state(tmp_path, monkeypatch, capsys):
    """An unparseable run state cannot prove no paid submit is pending."""
    import sys

    import voiceover_pipeline.cli as cli

    run_dir = tmp_path / "out" / "corrupt-run"
    (run_dir / "chunks").mkdir(parents=True)
    (run_dir / "run_state.json").write_text("{not json", encoding="utf-8")

    built: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "polza-tts",
            "--model",
            "openai/gpt-4o-mini-tts",
            "--voice",
            "ash",
            "--script",
            str(fixture_path("smoke_test.md")),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "corrupt-run",
            "--overwrite",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert payload["details"]["attempt_status"] == "unknown"
    assert built == []
    assert run_dir.exists()


def test_resume_and_overwrite_together_are_rejected_before_deletion(tmp_path, monkeypatch, capsys):
    """``--resume`` continues a run and ``--overwrite`` deletes it, so both fail early.

    The combination used to bypass the paid-audio delete confirmation and delete
    a completed run that had no pending marker. The conflict must be refused by
    argument validation, before any deletion, provider construction, or POST.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json, initial_state
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="resume-overwrite", resume=True)
    args.overwrite = True
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    paid_audio = paths.chunks_dir / "chunk_01.mp3"
    paid_audio.write_bytes(b"paid")

    built: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--overwrite",
            "--confirm-delete-paid-audio",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["code"] == 2
    assert "--resume" in payload["error"]
    assert "--overwrite" in payload["error"]
    assert built == []
    assert paid_audio.exists()
    assert json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8")) == state


def test_resume_only_run_is_not_rejected_by_conflict_guard(tmp_path, monkeypatch, capsys):
    """The mutual-exclusion guard must not reject an ordinary ``--resume`` run."""
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        initial_state,
        upsert_completed_chunk,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="resume-only", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    (paths.chunks_dir / "chunk_01.mp3").write_bytes(b"existing")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    provider = FakeProvider()

    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(cli, "build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_02"]


@pytest.mark.parametrize("payload", ["null", "[]", '"text"', "42"])
def test_overwrite_refuses_non_object_run_state(tmp_path, monkeypatch, capsys, payload):
    """Valid JSON that is not an object cannot prove no paid submit is pending.

    ``null``, a list, or a scalar loads without error but carries no trusted run
    evidence, so it fails closed like an unreadable state file and never reaches
    provider construction.
    """
    import sys

    import voiceover_pipeline.cli as cli

    run_dir = tmp_path / "out" / "non-object-run"
    (run_dir / "chunks").mkdir(parents=True)
    state_path = run_dir / "run_state.json"
    state_path.write_text(payload, encoding="utf-8")

    built: list[str] = []
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "polza-tts",
            "--model",
            "openai/gpt-4o-mini-tts",
            "--voice",
            "ash",
            "--script",
            str(fixture_path("smoke_test.md")),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "non-object-run",
            "--overwrite",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    result = json.loads(capsys.readouterr().out)
    assert result["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": None,
        "chunk_number": None,
        "attempt_status": "unknown",
    }
    assert built == []
    assert state_path.read_text(encoding="utf-8") == payload


def test_overwrite_allows_legitimate_state_without_pending_attempt(tmp_path, monkeypatch):
    """An old object state without a marker is not evidence of a paid submit."""
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "legacy-run"
    (run_dir / "chunks").mkdir(parents=True)
    atomic_write_json(run_dir / "run_state.json", {"chunks": [], "completed_count": 0})

    built: list[str] = []
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "polza-tts",
            "--model",
            "openai/gpt-4o-mini-tts",
            "--voice",
            "ash",
            "--script",
            str(fixture_path("smoke_test.md")),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "legacy-run",
            "--overwrite",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert built == ["built"]
    rewritten = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert rewritten["chunks"]


def test_overwrite_redacts_unsafe_attempt_id(tmp_path, monkeypatch, capsys):
    """An id-shaped secret in editable state never reaches the error envelope."""
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import atomic_write_json

    secret = "https://cdn.example.com/paid.mp3?token=sk-live-secret-12345"
    run_dir = tmp_path / "out" / "unsafe-id-run"
    (run_dir / "chunks").mkdir(parents=True)
    atomic_write_json(
        run_dir / "run_state.json",
        {"pending_attempt": {"id": secret, "number": 1, "status": "submitting"}},
    )

    built: list[str] = []
    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "polza-tts",
            "--model",
            "openai/gpt-4o-mini-tts",
            "--voice",
            "ash",
            "--script",
            str(fixture_path("smoke_test.md")),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "unsafe-id-run",
            "--overwrite",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    envelope = capsys.readouterr().out
    result = json.loads(envelope)
    assert result["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": None,
        "chunk_number": 1,
        "attempt_status": "submitting",
    }
    assert "sk-live-secret-12345" not in envelope
    assert "cdn.example.com" not in envelope
    assert built == []
    assert run_dir.exists()


def test_malformed_pending_attempt_blocks_resume_without_provider_call(tmp_path, monkeypatch):
    """A marker that is not a dict cannot be proved resolved and fails closed.

    The bounded attempt details stay JSON-friendly and never echo the marker's
    raw text back into diagnostics.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json, initial_state
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="paid-malformed-marker", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    state["pending_attempt"] = "submitting chunk_01 body=secret-request-text"
    atomic_write_json(paths.output_root / "run_state.json", state)

    provider = FakeProvider()

    with pytest.raises(cli.CliError, match="unconfirmed paid submit") as error:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "sk-test", None, paths, None, "auto"
        )

    assert error.value.code == 30
    assert error.value.details == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": None,
        "chunk_number": None,
        "attempt_status": "unknown",
    }
    assert provider.calls == []
    assert "secret-request-text" not in json.dumps(error.value.details)


def test_resume_preflight_blocks_unconfirmed_paid_submit_before_any_provider_work(
    tmp_path, monkeypatch, capsys
):
    """A blocked resume fails before key read, provider build, or pricing I/O.

    The marker check runs in ``generate`` before the provider-related preflights,
    so the documented PAID_SUBMIT_UNCONFIRMED envelope can never be replaced by a
    pricing or provider error, and no key read, provider build, pricing lookup, or
    POST is attempted.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="paid-early-resume", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    atomic_write_json(paths.output_root / "run_state.json", state)

    key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_01",
        "chunk_number": 1,
        "attempt_status": "submitting",
    }
    assert key_reads == []
    assert built == []
    assert priced == []


def test_dialogue_resume_marker_preflight_wins_over_identity_mismatch(
    tmp_path, monkeypatch, capsys
):
    """The paid marker blocks before the dialogue identity mismatch check.

    The same dialogue state without a marker reports the stale synthesis
    identity; once a paid marker is present, the bounded
    PAID_SUBMIT_UNCONFIRMED envelope wins and no key read, provider build, or
    pricing lookup runs.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )

    script, report, chunks = _write_dialogue_fixture(tmp_path)
    model = "google/gemini-3.1-flash-tts-preview"
    paths = build_run_paths(tmp_path / "out", model, "dialogue-marker")
    paths.chunks_dir.mkdir(parents=True)
    state = initial_state(
        provider="openrouter-tts",
        model=model,
        voice=next(iter(report["speaker_voice_map"].values())),
        script_path=script,
        chunks=chunks,
        script_format="dialogue",
        run_id="dialogue-marker",
        synthesis_identity="stale-identity",
    )
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)

    key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "_preflight_tts_quality_provider", lambda _args: None)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--script",
            str(script),
            "--model",
            model,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "dialogue-marker",
            "--tts-quality-provider",
            "fixture-asr",
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as control_exit:
        cli.main()

    assert control_exit.value.code == 30
    control_payload = json.loads(capsys.readouterr().out)
    assert "dialogue synthesis identity changed" in control_payload["error"]

    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    atomic_write_json(state_path, state)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "turn_0001",
        "chunk_number": 1,
        "attempt_status": "submitting",
    }
    assert key_reads == []
    assert built == []
    assert priced == []


def test_resume_marker_preflight_wins_over_timing_dependency_preflight(
    tmp_path, monkeypatch, capsys
):
    """A marked resume never reaches the optional timing key preflight.

    ``--with-timings --timing-provider openrouter-whisper`` reads the OpenRouter
    timing key before any marker check used to run. The guarded marker check now
    runs first, so the bounded PAID_SUBMIT_UNCONFIRMED envelope is reported
    instead of the key error, no timing key is read, and the run stays untouched.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="paid-timing-resume", resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)

    timing_key_reads: list[str] = []
    api_key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []

    def fake_read_openrouter_key() -> str:
        timing_key_reads.append("openrouter")
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_openrouter_key", fake_read_openrouter_key)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: api_key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--resume",
            "--with-timings",
            "--timing-provider",
            "openrouter-whisper",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "chunk_01",
        "chunk_number": 1,
        "attempt_status": "submitting",
    }
    assert timing_key_reads == []
    assert api_key_reads == []
    assert built == []
    assert priced == []
    assert state_path.exists()
    preserved = json.loads(state_path.read_text(encoding="utf-8"))
    assert preserved["pending_attempt"]["id"] == "chunk_01"


def test_resume_marker_preflight_wins_over_tts_quality_preflight(tmp_path, monkeypatch, capsys):
    """A marked dialogue resume never reaches the optional ASR quality preflight.

    ``--tts-quality-provider xai-stt`` reads the xAI key before any marker check
    used to run. The guarded marker check now runs first, so the bounded
    PAID_SUBMIT_UNCONFIRMED envelope is reported instead of the key error, no
    quality key is read, and the run stays untouched.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
    )

    script, report, chunks = _write_dialogue_fixture(tmp_path)
    model = "google/gemini-3.1-flash-tts-preview"
    paths = build_run_paths(tmp_path / "out", model, "dialogue-quality-resume")
    paths.chunks_dir.mkdir(parents=True)
    state = initial_state(
        provider="openrouter-tts",
        model=model,
        voice=next(iter(report["speaker_voice_map"].values())),
        script_path=script,
        chunks=chunks,
        script_format="dialogue",
        run_id="dialogue-quality-resume",
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)

    quality_key_reads: list[str] = []
    api_key_reads: list[str] = []
    built: list[str] = []
    priced: list[str] = []

    def fake_read_xai_key() -> str:
        quality_key_reads.append("xai")
        raise RuntimeError("XAI_API_KEY is not set")

    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_xai_key", fake_read_xai_key)
    monkeypatch.setattr(cli, "read_api_key", lambda _args: api_key_reads.append("key") or "sk-test")
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: priced.append("priced") or None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--script",
            str(script),
            "--model",
            model,
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "dialogue-quality-resume",
            "--tts-quality-provider",
            "xai-stt",
            "--resume",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert payload["details"] == {
        "error_code": "PAID_SUBMIT_UNCONFIRMED",
        "chunk_id": "turn_0001",
        "chunk_number": 1,
        "attempt_status": "submitting",
    }
    assert quality_key_reads == []
    assert api_key_reads == []
    assert built == []
    assert priced == []
    assert state_path.exists()
    preserved = json.loads(state_path.read_text(encoding="utf-8"))
    assert preserved["pending_attempt"]["id"] == "turn_0001"


def test_unconfirmed_attempt_rejects_every_present_marker():
    """Any persisted marker blocks; a completed sibling is not proof of safety."""
    from voiceover_pipeline.run_state import unconfirmed_attempt

    completed = [{"status": "completed", "number": 1, "id": "chunk_01"}]

    assert unconfirmed_attempt(None) is None
    assert unconfirmed_attempt({}) is None
    assert unconfirmed_attempt({"chunks": completed}) is None

    malformed = unconfirmed_attempt({"chunks": completed, "pending_attempt": ["opaque"]})
    assert malformed == {"id": None, "number": None, "status": "unknown"}

    conflicting = unconfirmed_attempt(
        {
            "chunks": completed,
            "pending_attempt": {"id": "chunk_99", "number": 1, "status": "failed"},
        }
    )
    assert conflicting == {"id": None, "number": 1, "status": "failed"}

    stale = unconfirmed_attempt(
        {
            "chunks": completed,
            "pending_attempt": {"id": "chunk_01", "number": 1, "status": "failed"},
        }
    )
    assert stale == {"id": "chunk_01", "number": 1, "status": "failed"}

    bounded = unconfirmed_attempt(
        {
            "pending_attempt": {
                "id": "chunk_01",
                "number": 1,
                "status": "signed-url-body-text",
                "body": "secret-request-text",
            }
        }
    )
    assert bounded == {"id": "chunk_01", "number": 1, "status": "unknown"}
    assert "secret-request-text" not in json.dumps(bounded)


def test_attempt_marker_keeps_only_generated_id_for_number():
    """An id survives only when it equals the generated form for its number.

    The projection previously accepted any ``chunk_``-prefixed token, so a
    malicious ``chunk_sk_live_secret`` from editable state reached the error
    envelope. Now the id must match the generated ``chunk_{n:02d}`` or
    ``turn_{n:04d}`` form for the same number.
    """
    from voiceover_pipeline.run_state import unconfirmed_attempt

    def marker(chunk_id, number):
        return unconfirmed_attempt(
            {"pending_attempt": {"id": chunk_id, "number": number, "status": "submitting"}}
        )

    assert marker("chunk_01", 1)["id"] == "chunk_01"
    assert marker("chunk_12", 12)["id"] == "chunk_12"
    assert marker("turn_0003", 3)["id"] == "turn_0003"
    assert marker("chunk_01_omnivoice_session", 1)["id"] == "chunk_01_omnivoice_session"

    for unsafe, number in [
        ("chunk_sk_live_secret", 1),
        ("chunk_sk_live_secret", 12),
        ("chunk_01", 2),
        ("turn_0003", 1),
        ("chunk_12_omnivoice_session", 1),
        ("chunk_01_omnivoice_session", 2),
        ("https://cdn.example.com/paid.mp3?token=sk-live-secret-12345", 1),
        ("chunk_01 body=secret-request-text", 1),
        ("chunk_01\nsecret", 1),
        ("chunk_" + "a" * 100, 1),
        ("x" * 100, 1),
        ("", 1),
        (123, 1),
        (None, 1),
    ]:
        bounded = marker(unsafe, number)
        assert bounded["id"] is None
        assert bounded["number"] == number
        assert bounded["status"] == "submitting"
        assert "secret" not in json.dumps(bounded)


def test_attempt_marker_bounds_number_and_drops_id_without_it():
    """A boolean, non-positive, or oversized number is reported as unknown.

    Generated chunk and turn numbers start at 1 and stay well below the bounded
    maximum, so anything else is not a number this repository wrote. Without a
    bounded number there is no generated form to compare an id against, so the
    id is dropped too.
    """
    from voiceover_pipeline.run_state import unconfirmed_attempt

    def marker(number, chunk_id="chunk_01"):
        return unconfirmed_attempt(
            {"pending_attempt": {"id": chunk_id, "number": number, "status": "submitting"}}
        )

    for bounded_number in [0, -1, -1_000_000, 1_000_001, 10**12]:
        bounded = marker(bounded_number)
        assert bounded == {"id": None, "number": None, "status": "submitting"}

    for boolean in [True, False]:
        bounded = marker(boolean, chunk_id="chunk_01")
        assert bounded["number"] is None
        assert bounded["id"] is None

    assert marker(1)["id"] == "chunk_01"
    assert marker(1_000_000, chunk_id="chunk_1000000")["id"] == "chunk_1000000"


def test_pending_media_recovery_requires_bound_chunk_and_opaque_id():
    """Only a marker bound to its generated chunk with an opaque id is reusable."""
    from voiceover_pipeline.run_state import (
        begin_chunk_attempt,
        pending_media_recovery,
        record_media_observed_cost,
        record_media_task_accepted,
    )

    state: dict = {}
    begin_chunk_attempt(state, chunk_id="chunk_01", number=1)
    assert pending_media_recovery(state) is None

    with pytest.raises(ValueError, match="different chunk"):
        record_media_task_accepted(state, task_id="task-1", chunk_id="chunk_02", number=2)
    with pytest.raises(ValueError, match="opaque"):
        record_media_task_accepted(
            state,
            task_id="https://cdn.example.com/paid.mp3?token=sk-live-secret",
            chunk_id="chunk_01",
            number=1,
        )

    record_media_task_accepted(state, task_id="task-1", chunk_id="chunk_01", number=1)
    record_media_observed_cost(state, cost=0.3, cost_exact="0.3")
    recovery = pending_media_recovery(state)
    assert recovery == {
        "id": "chunk_01",
        "number": 1,
        "status": "submitting",
        "remote_task_id": "task-1",
        "cost": 0.3,
        "cost_exact": "0.3",
    }
    # A later completion payload without usage does not erase the observed cost.
    record_media_observed_cost(state, cost=None, cost_exact=None)
    assert pending_media_recovery(state)["cost_exact"] == "0.3"

    marker = state["pending_attempt"]
    tampered = {"pending_attempt": {**marker, "remote_task_id": "signed/url?token=secret"}}
    assert pending_media_recovery(tampered) is None
    # The id must be the generated form for its own number to be recoverable.
    mismatched_form = {"pending_attempt": {**marker, "id": "chunk_07", "number": 1}}
    assert pending_media_recovery(mismatched_form) is None


def test_pending_raw_recovery_bounds_path_format_and_digest():
    """Only a bounded raw receipt can be reported, and it never echoes a URL."""
    from voiceover_pipeline.run_state import (
        begin_chunk_attempt,
        pending_raw_recovery,
        record_raw_audio_saved,
    )

    state: dict = {}
    begin_chunk_attempt(state, chunk_id="chunk_01", number=1)
    assert pending_raw_recovery(state) is None
    digest = "a" * 64
    record_raw_audio_saved(
        state,
        chunk_id="chunk_01",
        number=1,
        audio_format="mp3",
        relative_path="raw/chunk_01.mp3",
        sha256=digest,
        generation_id="gen-1",
    )
    recovery = pending_raw_recovery(state)
    assert recovery == {
        "id": "chunk_01",
        "number": 1,
        "status": "raw_saved",
        "raw_format": "mp3",
        "raw_path": "raw/chunk_01.mp3",
        "raw_sha256": digest,
        "generation_id": "gen-1",
        "cost": None,
        "cost_exact": None,
    }

    with pytest.raises(ValueError, match="different chunk"):
        record_raw_audio_saved(
            state,
            chunk_id="chunk_02",
            number=2,
            audio_format="mp3",
            relative_path="raw/chunk_02.mp3",
            sha256=digest,
            generation_id=None,
        )
    with pytest.raises(ValueError, match="deterministic generated path"):
        record_raw_audio_saved(
            state,
            chunk_id="chunk_01",
            number=1,
            audio_format="mp3",
            relative_path="https://cdn.example.com/paid.mp3?token=sk-live-secret",
            sha256=digest,
            generation_id=None,
        )
    with pytest.raises(ValueError, match="sha256"):
        record_raw_audio_saved(
            state,
            chunk_id="chunk_01",
            number=1,
            audio_format="mp3",
            relative_path="raw/chunk_01.mp3",
            sha256="not-a-digest",
            generation_id=None,
        )

    # A tampered marker cannot point at an arbitrary path or echo a URL id.
    url_path = {
        "pending_attempt": {
            "id": "chunk_01",
            "number": 1,
            "status": "raw_saved",
            "raw": {
                "format": "mp3",
                "path": "https://cdn.example.com/paid.mp3?token=sk-live-secret",
                "sha256": digest,
                "generation_id": "gen-1",
            },
        }
    }
    assert pending_raw_recovery(url_path) is None
    url_id = {
        "pending_attempt": {
            "id": "chunk_01",
            "number": 1,
            "status": "raw_saved",
            "raw": {
                "format": "mp3",
                "path": "raw/chunk_01.mp3",
                "sha256": digest,
                "generation_id": "https://cdn.example.com/paid.mp3?token=sk-live-secret",
            },
        }
    }
    assert pending_raw_recovery(url_id)["generation_id"] is None


def test_recovery_service_prefers_saved_raw_over_known_media(tmp_path):
    """A marker holding both a raw receipt and a media id rebuilds locally first.

    The recovery service is read-only and keeps the CLI's paid precedence: saved
    raw bytes need no provider request, so they win over a known media task id.
    """
    from voiceover_pipeline.run_state import (
        begin_chunk_attempt,
        initial_state,
        raw_audio_relative_path,
        record_media_task_accepted,
        record_raw_audio_saved,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import recovery

    run_root = tmp_path / "out" / "raw-precedence"
    chunks_dir = run_root / "chunks"
    chunks_dir.mkdir(parents=True)
    script = fixture_path("smoke_test.md")
    chunks = split_markdown_by_delimiter(script, "******")
    model = "elevenlabs/text-to-speech-turbo-2-5"
    voice = "Rachel"
    state = initial_state(
        provider="polza-tts",
        model=model,
        voice=voice,
        script_path=script,
        chunks=chunks,
        script_format="markdown",
        run_id="raw-precedence",
    )
    first = chunks[0]
    begin_chunk_attempt(state, chunk_id=first.id, number=first.number)
    record_media_task_accepted(state, task_id="task-1", chunk_id=first.id, number=first.number)
    relative = raw_audio_relative_path(first.id, "mp3")
    raw_path = run_root / relative
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"paid-mp3")
    record_raw_audio_saved(
        state,
        chunk_id=first.id,
        number=first.number,
        audio_format="mp3",
        relative_path=relative,
        sha256=hashlib.sha256(b"paid-mp3").hexdigest(),
        generation_id="gen-1",
    )

    attempts = recovery.recoverable_paid_attempts(
        state,
        provider="polza-tts",
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_root,
    )
    assert attempts[first.number]["kind"] == "raw"
    assert attempts[first.number]["raw_path"] == relative
    # The media fallback exists on its own, so the raw branch above is a choice.
    media = recovery.recoverable_media_attempts(
        state,
        provider="polza-tts",
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
    )
    assert media[first.number]["kind"] == "media"


def test_recovery_service_fails_closed_on_corrupt_raw_or_changed_identity(tmp_path):
    """A replaced raw file or a changed identity yields no recoverable attempt.

    The service only reads bounded state and files, so an empty result is the
    same documented block the CLI keeps; no provider request can follow.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import (
        begin_chunk_attempt,
        initial_state,
        pending_raw_recovery,
        raw_audio_relative_path,
        record_raw_audio_saved,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services import recovery

    run_root = tmp_path / "out" / "raw-fail-closed"
    chunks_dir = run_root / "chunks"
    chunks_dir.mkdir(parents=True)
    script = fixture_path("smoke_test.md")
    chunks = split_markdown_by_delimiter(script, "******")
    model = "openai/gpt-4o-mini-tts"
    voice = "ash"
    state = initial_state(
        provider="polza-tts",
        model=model,
        voice=voice,
        script_path=script,
        chunks=chunks,
        script_format="markdown",
        run_id="raw-fail-closed",
    )
    first = chunks[0]
    begin_chunk_attempt(state, chunk_id=first.id, number=first.number)
    relative = raw_audio_relative_path(first.id, "mp3")
    raw_path = run_root / relative
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"paid-mp3")
    record_raw_audio_saved(
        state,
        chunk_id=first.id,
        number=first.number,
        audio_format="mp3",
        relative_path=relative,
        sha256=hashlib.sha256(b"paid-mp3").hexdigest(),
        generation_id="gen-1",
    )

    def attempts(**overrides):
        kwargs = {
            "provider": "polza-tts",
            "model": model,
            "voice": voice,
            "chunks": chunks,
            "chunks_dir": chunks_dir,
            "run_root": run_root,
        }
        kwargs.update(overrides)
        return recovery.recoverable_paid_attempts(state, **kwargs)

    assert attempts()[first.number]["kind"] == "raw"
    # A replaced raw file no longer matches its receipt digest.
    raw_path.write_bytes(b"replaced")
    assert attempts() == {}
    assert (
        cli._recoverable_paid_attempts(
            state,
            provider="polza-tts",
            model=model,
            voice=voice,
            chunks=chunks,
            chunks_dir=chunks_dir,
            run_root=run_root,
        )
        == {}
    )
    assert pending_raw_recovery(state) is not None
    # A changed provider, model, voice, or script is not the stored attempt.
    raw_path.write_bytes(b"paid-mp3")
    assert attempts(model="openai/gpt-4o-mini-tts-v2") == {}
    assert attempts(voice="nova") == {}
    assert attempts(provider="polza-chat-audio") == {}
    assert attempts(chunks=list(reversed(chunks))) == {}
    # A synchronous ``/audio/speech`` model can never hold a media task id.
    assert recovery.known_media_recovery(state) is None


def test_status_marks_unconfirmed_paid_submit_as_not_resumable(tmp_path):
    """A blocked run reports can_resume false with a bounded machine reason."""
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "paid-status"
    (run_dir / "chunks").mkdir(parents=True)
    atomic_write_json(
        run_dir / "run_state.json",
        {
            "chunk_count": 2,
            "completed_count": 0,
            "chunks": [],
            "errors": [],
            "pending_attempt": {
                "id": "chunk_01",
                "number": 1,
                "status": "outcome_unknown",
                "at": "2026-09-29T00:00:00Z",
            },
        },
    )

    code, data = cli_json(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "paid-status",
        "--json",
    )

    assert code == 0
    assert data["can_resume"] is False
    assert data["resume_block_reason"] == "paid_submit_unconfirmed"


def test_status_reports_known_media_attempt_as_resumable(tmp_path):
    """A stored bounded media id keeps the paid marker resumable.

    That chunk is finished with GET calls only, so ``status`` does not report the
    unconfirmed block and never echoes the stored task id.
    """
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    run_dir = tmp_path / "out" / "paid-media-status"
    (run_dir / "chunks").mkdir(parents=True)
    script = fixture_path("smoke_test.md")
    chunks = split_markdown_by_delimiter(script, "******")
    state = initial_state(
        provider="polza-tts",
        model="elevenlabs/text-to-speech-turbo-2-5",
        voice="Rachel",
        script_path=script,
        chunks=chunks,
        script_format="markdown",
        run_id="paid-media-status",
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    record_media_task_accepted(
        state, task_id="task-1", chunk_id=chunks[0].id, number=chunks[0].number
    )
    atomic_write_json(run_dir / "run_state.json", state)

    code, data = cli_json(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "paid-media-status",
        "--json",
    )

    assert code == 0
    assert data["can_resume"] is True
    assert data["resume_block_reason"] is None
    assert "task-1" not in json.dumps(data)


def test_status_chunk2_media_recovery_requires_preceding_chunk_on_disk(
    tmp_path, monkeypatch, capsys
):
    """A stored chunk_02 task stays resumable only when chunk_01 is truly done.

    ``--resume`` regenerates an earlier chunk whose state entry says completed
    while its MP3 is missing, and that paid submit would collide with the stored
    chunk_02 id. ``status`` must report the same paid-submit block instead of
    promising a resume, and a completed chunk_01 with its MP3 present resumes.
    """
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    run_id = "media-status-predecessor"
    run_dir = tmp_path / "out" / run_id
    (run_dir / "chunks").mkdir(parents=True)
    script = fixture_path("smoke_test.md")
    chunks = split_markdown_by_delimiter(script, "******")
    state = initial_state(
        provider="polza-tts",
        model="elevenlabs/text-to-speech-turbo-2-5",
        voice="Rachel",
        script_path=script,
        chunks=chunks,
        script_format="markdown",
        run_id=run_id,
    )
    second = chunks[1]
    begin_chunk_attempt(state, chunk_id=second.id, number=second.number)
    record_media_task_accepted(state, task_id="task-2", chunk_id=second.id, number=second.number)
    state_path = run_dir / "run_state.json"
    atomic_write_json(state_path, state)

    def status():
        return cli_json(
            "status", "--output-dir", str(tmp_path / "out"), "--run-id", run_id, "--json"
        )

    # chunk_01 was never completed, so the stored chunk_02 task is not next.
    code, data = status()
    assert code == 0
    assert data["can_resume"] is False
    assert data["resume_block_reason"] == "paid_submit_unconfirmed"

    # chunk_01 looks completed in state but its MP3 is gone.
    state["chunks"] = [
        {"status": "completed", "number": 1, "id": "chunk_01", "file": "chunk_01.mp3"}
    ]
    state["completed_count"] = 1
    atomic_write_json(state_path, state)
    code, data = status()
    assert code == 0
    assert data["can_resume"] is False
    assert data["resume_block_reason"] == "paid_submit_unconfirmed"

    # The real generate preflight must agree before key/provider/pricing work.
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: pytest.fail("key read before block"))
    monkeypatch.setattr(cli, "build_provider", lambda *_args: pytest.fail("provider built"))
    monkeypatch.setattr(
        cli, "fetch_pricing_snapshot", lambda *_args: pytest.fail("pricing fetched")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            "polza-tts",
            "--model",
            "elevenlabs/text-to-speech-turbo-2-5",
            "--voice",
            "Rachel",
            "--script",
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            run_id,
            "--resume",
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 30
    assert json.loads(capsys.readouterr().out)["details"]["error_code"] == (
        "PAID_SUBMIT_UNCONFIRMED"
    )

    # chunk_01 completed with its MP3 present lets chunk_02 recover.
    (run_dir / "chunks" / "chunk_01.mp3").write_bytes(b"mp3")
    code, data = status()
    assert code == 0
    assert data["can_resume"] is True
    assert data["resume_block_reason"] is None


def test_overwrite_stays_blocked_for_known_media_id_marker(tmp_path, monkeypatch, capsys):
    """--overwrite never deletes a run that holds any paid attempt marker."""
    import sys

    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        begin_chunk_attempt,
        initial_state,
        record_media_task_accepted,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    args = make_args(tmp_path, run_id="paid-media-overwrite")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    begin_chunk_attempt(state, chunk_id=chunks[0].id, number=chunks[0].number)
    record_media_task_accepted(
        state, task_id="task-1", chunk_id=chunks[0].id, number=chunks[0].number
    )
    state_path = paths.output_root / "run_state.json"
    atomic_write_json(state_path, state)

    built: list[str] = []
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(cli, "read_api_key", lambda _args: "sk-test")
    monkeypatch.setattr(cli, "fetch_pricing_snapshot", lambda *_args: None)
    monkeypatch.setattr(
        cli, "build_provider", lambda *_args, **_kwargs: built.append("built") or FakeProvider()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover-pipeline",
            "generate",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--voice",
            args.voice,
            "--script",
            str(args.script),
            "--output-dir",
            str(args.output_dir),
            "--run-id",
            args.run_id,
            "--overwrite",
            "--confirm-delete-paid-audio",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 30
    payload = json.loads(capsys.readouterr().out)
    assert "unconfirmed paid submit" in payload["error"]
    assert payload["details"]["error_code"] == "PAID_SUBMIT_UNCONFIRMED"
    assert built == []
    assert paths.output_root.exists()
    assert "pending_attempt" in json.loads(state_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "payload",
    [
        '{"token": "sk-live-secret-12345", ',
        "null",
        "[]",
        '"text"',
        "42",
    ],
)
def test_status_reports_unreadable_state_as_not_resumable(tmp_path, payload):
    """An existing state that cannot prove resumability is reported, not crashed.

    Invalid JSON, ``null``, a list, or a scalar cannot prove the run is
    resumable, and ``status`` must not raise or falsely imply resume. It emits
    one parseable success object with ``can_resume: false`` and the bounded
    ``resume_block_reason: "run_state_unreadable"``, leaves the state and chunk
    files untouched, and never echoes the raw content back.
    """
    run_dir = tmp_path / "out" / "unreadable-status"
    (run_dir / "chunks").mkdir(parents=True)
    chunk_audio = run_dir / "chunks" / "chunk_01.mp3"
    chunk_audio.write_bytes(b"paid")
    state_path = run_dir / "run_state.json"
    state_path.write_text(payload, encoding="utf-8")

    proc = run_cli(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "unreadable-status",
        "--json",
    )

    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["status"] == "success"
    assert data["can_resume"] is False
    assert data["resume_block_reason"] == "run_state_unreadable"
    assert state_path.read_text(encoding="utf-8") == payload
    assert chunk_audio.read_bytes() == b"paid"
    assert run_dir.exists()
    assert "sk-live-secret-12345" not in proc.stdout


def test_resume_does_not_regenerate_completed_chunks(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import (
        atomic_write_json,
        initial_state,
        upsert_completed_chunk,
    )
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]
    (paths.chunks_dir / "chunk_01.mp3").write_bytes(b"existing")
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
    )
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=1,
            id="chunk_01",
            file="chunk_01.mp3",
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    provider = FakeProvider()

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert provider.calls == ["chunk_02"]


def test_clone_voice_identity_is_deterministic_across_processes(tmp_path, monkeypatch):
    import subprocess
    import sys

    import voiceover_pipeline.cli as cli

    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"not-a-real-wav-but-fine")
    args = argparse.Namespace(
        provider="omnivoice-local",
        mode="clone",
        reference_audio=str(reference),
        reference_text="Всем привет. Это образец голоса.",
    )
    expected = cli._omnivoice_voice_identity(args)
    assert expected is not None
    assert "hash(" not in expected and "0x" not in expected

    script = (
        "import sys, types\n"
        "import voiceover_pipeline.cli as cli\n"
        "ref = r'%s'\n"
        "args = types.SimpleNamespace(provider='omnivoice-local', mode='clone',"
        " reference_audio=ref, reference_text='Всем привет. Это образец голоса.')\n"
        "print(cli._omnivoice_voice_identity(args))\n" % (str(reference),)
    )
    for seed in ("1", "987654"):
        env = dict(monkeypatch.__dict__.get("env", {}))
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected


def test_resume_rejects_changed_voice_identity(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json, initial_state
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="voice-identity", resume=True)
    args.provider = "omnivoice-local"
    args.model = "audio-cpp/omnivoice-q8_0"
    args.voice = "main"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:1]
    state = initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="markdown",
        run_id=args.run_id,
        voice_identity="preset:main:" + "1" * 64,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)

    args.voice_bank_catalog = object()
    args.voice_bank_profile = type(
        "Profile",
        (),
        {"id": "main", "reference_sha256": "2" * 64},
    )()

    with pytest.raises(cli.CliError, match="voice identity changed") as error:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "none"
        )

    assert error.value.code == 30


def _write_dialogue_fixture(tmp_path):
    from voiceover_pipeline.gemini_dialogue import (
        dialogue_turns_from_validation,
        validate_gemini_dialogue_file,
    )

    script = tmp_path / "podcast.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: gemini-dialogue",
                "language: ru",
                "model: google/gemini-3.1-flash-tts-preview",
                "speakers:",
                "  Host:",
                "    display_name: Ведущая",
                "    voice: Kore",
                "    profile: warm host",
                "  Guest:",
                "    display_name: Гость",
                "    voice: Puck",
                "    profile: calm expert",
                "vibe: >",
                "  Russian technical podcast. Natural question-and-answer conversation.",
                "allowed_tags:",
                "  - warmly",
                "  - curious",
                "max_chunk_bytes: 3500",
                "---",
                "Host: [warmly] Что умеет утилита?",
                "Guest: Она создаёт озвучку и субтитры.",
                "******",
                "Host: [curious] Можно работать локально?",
                "Guest: Да, для одноголосой озвучки есть OmniVoice.",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_gemini_dialogue_file(script)
    assert report["valid"] is True
    return script, report, dialogue_turns_from_validation(report)


def make_dialogue_args(tmp_path, script, speaker_voice_map, run_id="dialogue-run", resume=False):
    return argparse.Namespace(
        provider="openrouter-tts",
        model="google/gemini-3.1-flash-tts-preview",
        voice=next(iter(speaker_voice_map.values())),
        script=script,
        output_dir=tmp_path / "out",
        run_id=run_id,
        format="dialogue",
        limit_chunks=None,
        retries=3,
        retry_delay=0,
        retry_max_delay=0,
        no_retry=False,
        no_trim=True,
        json_output=True,
        json_events=False,
        resume=resume,
        with_timings=False,
        speaker_voice_map=speaker_voice_map,
    )


def _dialogue_state(paths, args, chunks, style_prompt, identity):
    from voiceover_pipeline.run_state import initial_state

    return initial_state(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        script_path=args.script,
        chunks=chunks,
        script_format="dialogue",
        run_id=args.run_id,
        synthesis_identity=identity,
    )


def test_dialogue_resume_same_cast_skips_completed_chunks(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.models import ChunkArtifact
    from voiceover_pipeline.run_state import atomic_write_json, upsert_completed_chunk

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    style_prompt = report["style_prompt"]
    prompt_mode = "native"
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    identity = cli._gemini_dialogue_identity(args, style_prompt, prompt_mode, chunks)
    assert identity is not None
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    first_path = paths.chunks_dir / f"{chunks[0].id}.mp3"
    first_path.write_bytes(b"existing")
    state = _dialogue_state(paths, args, chunks, style_prompt, identity)
    upsert_completed_chunk(
        state,
        artifact=ChunkArtifact(
            number=chunks[0].number,
            id=chunks[0].id,
            file=first_path.name,
            duration_ms=1000,
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            text_characters=len(chunks[0].text),
            transcript=None,
            client_path="fake",
            generation_id="old-gen",
            turn_index=chunks[0].number,
            speech_duration_ms=1000,
            audio_sha256=hashlib.sha256(b"existing").hexdigest(),
            pause_after_ms=chunks[0].pause_after_ms,
        ),
        model=args.model,
        voice=args.voice,
        text=chunks[0].text,
        include_text=False,
        include_transcript=False,
    )
    atomic_write_json(paths.output_root / "run_state.json", state)
    provider = FakeProvider()

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, style_prompt, "native"
        )

    assert exit_info.value.code == 0
    assert provider.calls == [chunk.id for chunk in chunks[1:]]
    final_state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert final_state["synthesis_identity"] == identity


def test_dialogue_quality_failure_prevents_final_concat(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths

    patch_generation_io(monkeypatch)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"])
    args.tts_quality_provider = "fixture-asr"
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)

    def fail_quality(*_args, **_kwargs):
        raise cli.CliError("quality gate failed", 60)

    monkeypatch.setattr(cli, "_verify_dialogue_turns_before_concat", fail_quality)

    def unexpected_concat(*_args, **_kwargs):
        raise AssertionError("final concat must not run after quality failure")

    monkeypatch.setattr(cli, "concat_dialogue_turns", unexpected_concat)

    with pytest.raises(cli.CliError, match="quality gate failed") as error:
        cli._generate_step(
            args,
            FakeProvider(),
            "ffmpeg",
            "ffprobe",
            chunks,
            "key",
            None,
            paths,
            None,
            "none",
        )

    assert error.value.code == 60
    assert not paths.full_mp3.exists()


def test_dialogue_resume_rejects_changed_speaker_voice(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json

    patch_generation_io(monkeypatch)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    style_prompt = report["style_prompt"]
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    identity = cli._gemini_dialogue_identity(args, style_prompt, "native", chunks)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    state = _dialogue_state(paths, args, chunks, style_prompt, identity)
    atomic_write_json(paths.output_root / "run_state.json", state)

    changed_map = dict(report["speaker_voice_map"])
    changed_map["Guest"] = "Charon"
    changed_args = make_dialogue_args(
        tmp_path, script, changed_map, run_id="dialogue-run", resume=True
    )

    with pytest.raises(cli.CliError, match="synthesis identity changed") as error:
        cli._generate_step(
            changed_args,
            FakeProvider(),
            "ffmpeg",
            "ffprobe",
            chunks,
            "key",
            None,
            paths,
            style_prompt,
            "native",
        )

    assert error.value.code == 30


def test_dialogue_resume_rejects_changed_model(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json

    patch_generation_io(monkeypatch)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    style_prompt = report["style_prompt"]
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    identity = cli._gemini_dialogue_identity(args, style_prompt, "native", chunks)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    state = _dialogue_state(paths, args, chunks, style_prompt, identity)
    atomic_write_json(paths.output_root / "run_state.json", state)

    changed_args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    changed_args.model = "google/gemini-3.1-flash-tts-preview-2"

    with pytest.raises(cli.CliError, match="synthesis identity changed") as error:
        cli._generate_step(
            changed_args,
            FakeProvider(),
            "ffmpeg",
            "ffprobe",
            chunks,
            "none",
            None,
            paths,
            style_prompt,
            "native",
        )

    assert error.value.code == 30


def test_dialogue_resume_rejects_changed_style_prompt(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json

    patch_generation_io(monkeypatch)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    style_prompt = report["style_prompt"]
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    identity = cli._gemini_dialogue_identity(args, style_prompt, "native", chunks)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    state = _dialogue_state(paths, args, chunks, style_prompt, identity)
    atomic_write_json(paths.output_root / "run_state.json", state)

    changed_style = style_prompt + " Тон должен быть другим."
    changed_args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)

    with pytest.raises(cli.CliError, match="synthesis identity changed") as error:
        cli._generate_step(
            changed_args,
            FakeProvider(),
            "ffmpeg",
            "ffprobe",
            chunks,
            "none",
            None,
            paths,
            changed_style,
            "native",
        )

    assert error.value.code == 30


def test_dialogue_resume_old_state_without_identity_fails_closed(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.run_state import atomic_write_json

    patch_generation_io(monkeypatch)
    script, report, chunks = _write_dialogue_fixture(tmp_path)
    style_prompt = report["style_prompt"]
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"], resume=True)
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    state = _dialogue_state(paths, args, chunks, style_prompt, None)
    assert "synthesis_identity" not in state
    atomic_write_json(paths.output_root / "run_state.json", state)

    with pytest.raises(cli.CliError, match="predates dialogue synthesis identity") as error:
        cli._generate_step(
            args,
            FakeProvider(),
            "ffmpeg",
            "ffprobe",
            chunks,
            "none",
            None,
            paths,
            style_prompt,
            "native",
        )

    assert error.value.code == 30


def test_overwrite_refuses_to_delete_existing_paid_chunks_without_confirmation(tmp_path):
    run_dir = tmp_path / "out" / "paid-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunk_01.mp3").write_bytes(b"paid")

    code, data = cli_json(
        "generate",
        "--provider",
        "polza-chat-audio",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "paid-run",
        "--overwrite",
        "--json",
    )

    assert code == 30
    assert "confirm-delete-paid-audio" in data["error"]
    assert (chunks_dir / "chunk_01.mp3").exists()


def test_dry_run_cost_with_limit_chunks_makes_no_tts_request(tmp_path):
    code, data = cli_json(
        "generate",
        "--provider",
        "polza-tts",
        "--model",
        "openai/gpt-4o-mini-tts",
        "--voice",
        "ash",
        "--script",
        str(fixture_path("smoke_test.md")),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "dry-run",
        "--dry-run-cost",
        "--limit-chunks",
        "1",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["chunks"] == 1
    assert data["original_chunks"] == 2
    assert not (tmp_path / "out" / "dry-run").exists()


def test_local_tts_dry_run_reports_sentence_packed_inference_chunks(tmp_path):
    script = tmp_path / "local-report.md"
    script.write_text(
        " ".join(
            f"Предложение номер словами содержит достаточно полезного текста {word}."
            for word in (
                "первое",
                "второе",
                "третье",
                "четвёртое",
                "пятое",
                "шестое",
                "седьмое",
                "восьмое",
            )
        ),
        encoding="utf-8",
    )

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "local-dry-run",
        "--voice-bank",
        str(build_voice_bank(tmp_path)),
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 1
    assert data["original_chunks"] == data["chunks"]
    assert not (tmp_path / "out" / "local-dry-run").exists()


def test_local_tts_cli_rejects_raw_digits_before_runtime(tmp_path):
    script = tmp_path / "digits.md"
    script.write_text("Версия 3.5 готова на 25 процентов.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "digits",
        "--voice-bank",
        str(build_voice_bank(tmp_path)),
        "--dry-run-cost",
        "--json",
    )

    assert code == 2
    assert "raw digits" in data["error"]
    assert not (tmp_path / "out" / "digits").exists()


def test_omnivoice_cli_rejects_style_controls_before_dry_run(tmp_path):
    script = tmp_path / "style.md"
    script.write_text("Проверка готового автоматического голоса.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "omnivoice-local",
        "--model",
        "audio-cpp/omnivoice-q8_0",
        "--style-prompt",
        "тёплый голос",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "style",
        "--dry-run-cost",
        "--json",
    )

    assert code == 2
    assert "style controls" in data["error"]
    assert not (tmp_path / "out" / "style").exists()


def test_unprofiled_qwen_cli_dry_run_does_not_apply_omnivoice_digit_policy(tmp_path):
    script = tmp_path / "qwen-digits.md"
    script.write_text("Версия 3.5 готова на 25 процентов.", encoding="utf-8")

    code, data = cli_json(
        "generate",
        "--provider",
        "qwen-local",
        "--script",
        str(script),
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "qwen-digits",
        "--dry-run-cost",
        "--json",
    )

    assert code == 0
    assert data["dry_run"] is True
    assert data["chunks"] == 1
    assert not (tmp_path / "out" / "qwen-digits").exists()


def test_status_reports_partial_run_12_of_105(tmp_path):
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "partial-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    for number in range(1, 13):
        chunk_id = f"chunk_{number:02d}"
        (chunks_dir / f"{chunk_id}.mp3").write_bytes(b"mp3")
        chunks.append(
            {"status": "completed", "number": number, "id": chunk_id, "file": f"{chunk_id}.mp3"}
        )
    atomic_write_json(
        run_dir / "run_state.json",
        {"chunk_count": 105, "completed_count": 12, "chunks": chunks, "errors": []},
    )

    code, data = cli_json(
        "status",
        "--output-dir",
        str(tmp_path / "out"),
        "--run-id",
        "partial-run",
        "--json",
    )

    assert code == 0
    assert data["total_chunks"] == 105
    assert data["completed_chunks"] == 12
    assert data["next_chunk"] == 13
    assert data["can_resume"] is True
    assert data["resume_block_reason"] is None


def test_concat_writes_partial_ogg_name(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.run_state import atomic_write_json

    run_dir = tmp_path / "out" / "partial-run"
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    for number in range(1, 13):
        (chunks_dir / f"chunk_{number:02d}.mp3").write_bytes(b"mp3")
    atomic_write_json(
        run_dir / "run_state.json", {"chunk_count": 105, "completed_count": 12, "chunks": []}
    )
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        cli,
        "concat_audio_files",
        lambda _ffmpeg, _files, output_path: output_path.write_bytes(b"ogg"),
    )

    args = argparse.Namespace(
        output_dir=tmp_path / "out", run_id="partial-run", format="ogg", json_output=False
    )
    cli.concat_cmd(args)

    assert (run_dir / "partial-12-of-105.ogg").read_bytes() == b"ogg"


def test_whisper_install_message_uses_extra(monkeypatch):
    import voiceover_pipeline.cli as cli

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(cli.CliError) as error:
        cli._preflight_timing_dependency()

    assert "uv sync --extra timing-whisper" in str(error.value)


def test_timing_artifact_writer_keeps_legacy_file_names_and_manifest_shape(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.models import TimingResult, TimingSegment

    audio = tmp_path / "fixture.mp3"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(cli.shutil, "which", lambda _command: "ffprobe")
    monkeypatch.setattr(cli, "mp3_duration_ms", lambda _ffprobe, _audio: 1000)
    timing = TimingResult(
        segments=[
            TimingSegment(
                id=1,
                start_sec=0.0,
                end_sec=1.0,
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                text="fixture",
            )
        ],
        model="small",
        backend="faster-whisper",
    )

    result = cli._write_timing_artifacts(audio, tmp_path, "legacy", timing)

    manifest = json.loads((tmp_path / "legacy.timings.json").read_text(encoding="utf-8"))
    assert result == {"segment_count": 1, "total_duration_ms": 1000}
    assert "provider" not in manifest
    assert (tmp_path / "legacy.srt").read_text(
        encoding="utf-8"
    ) == "1\n00:00:00,000 --> 00:00:01,000\nfixture\n"


def test_prepare_run_wraps_the_original_chunks_without_changing_identity(tmp_path):
    """Preparation reuses the exact hashed chunks and keeps the run identity.

    A prepared part must hold the original ``ScriptChunk`` object rather than a
    regenerated one, so ids, text, order, and ``script_hash`` stay identical
    between preparation and synthesis.
    """
    from voiceover_pipeline.run_state import script_hash
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services.prepare import prepare_run

    args = make_args(tmp_path, run_id="prepared-identity")
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]

    prepared = prepare_run(args, chunks, "run-scoped style", "auto")

    assert prepared.provider == args.provider
    assert prepared.model == args.model
    assert prepared.voice == args.voice
    assert prepared.style_prompt == "run-scoped style"
    assert prepared.prompt_mode == "auto"
    assert [part.chunk for part in prepared.parts] == chunks
    assert all(part.chunk is chunk for part, chunk in zip(prepared.parts, chunks))
    assert [part.voice for part in prepared.parts] == [chunk.voice for chunk in chunks]
    assert script_hash([part.chunk for part in prepared.parts]) == script_hash(chunks)


def test_generate_step_drives_every_synthesis_call_with_a_prepared_part(tmp_path, monkeypatch):
    """The run loop calls the single synthesis seam once per prepared part.

    Each call receives the prepared part wrapping the original chunk, so the
    order and identity that reach the provider are the ones the run hashed and
    stored.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter
    from voiceover_pipeline.services.synthesis import synthesize_part

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="prepared-parts")
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]
    seen: list[tuple[object, str, str | None]] = []

    def recording_synthesize(provider, part):
        seen.append((part.chunk, part.chunk.text, part.voice))
        return synthesize_part(provider, part)

    monkeypatch.setattr(cli, "synthesize_part", recording_synthesize)
    provider = FakeProvider()

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, provider, "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert [item[1] for item in seen] == [chunk.text for chunk in chunks]
    assert [item[2] for item in seen] == [chunk.voice for chunk in chunks]
    assert all(item[0] is chunk for item, chunk in zip(seen, chunks))
    assert provider.calls == [chunk.id for chunk in chunks]
    state = json.loads((paths.output_root / "run_state.json").read_text(encoding="utf-8"))
    assert [item["voice"] for item in state["chunks"]] == [args.voice] * len(chunks)


def test_synthesize_part_keeps_the_openrouter_cast_voice_per_turn(tmp_path, monkeypatch):
    """Prepared dialogue parts still send their own turn text and cast voice.

    Every OpenRouter call must carry ``voice=<turn voice>`` from the prepared
    part, and no turn may leak into another request body.
    """
    import voiceover_pipeline.providers.openrouter_tts as openrouter_tts
    from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
    from voiceover_pipeline.services.prepare import prepare_run
    from voiceover_pipeline.services.synthesis import synthesize_part

    class FakeResponse:
        status_code = 200
        content = b"pcm"
        headers = {"Content-Type": "audio/pcm"}

    script, report, chunks = _write_dialogue_fixture(tmp_path)
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"])
    prepared = prepare_run(args, chunks, report["style_prompt"], "native")
    bodies: list[dict] = []

    def post(_url, **kwargs):
        bodies.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(openrouter_tts.requests, "post", post)
    provider = OpenRouterTTSProvider(
        api_key="test",
        model=args.model,
        voice=args.voice,
        speaker_voice_map=report["speaker_voice_map"],
        prompt_mode="native",
    )

    results = [synthesize_part(provider, part) for part in prepared.parts]

    assert [body["voice"] for body in bodies] == [chunk.voice for chunk in chunks]
    assert [body["input"] for body in bodies] == [chunk.text for chunk in chunks]
    assert [result.transcript for result in results] == [chunk.text for chunk in chunks]


def test_synthesize_part_selects_the_cast_provider_and_tolerates_legacy_signatures(
    tmp_path, monkeypatch
):
    """Provider selection follows the part voice via signature inspection.

    A voice-keyed cast map must route each part to its own provider, and a legacy
    OpenRouter signature without a ``voice`` parameter is detected up front and
    receives the two-argument call while a ``TypeError`` raised by a
    voice-accepting signature still propagates.
    """
    from voiceover_pipeline.models import SynthesisResult
    from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
    from voiceover_pipeline.services.prepare import prepare_run
    from voiceover_pipeline.services.synthesis import synthesize_part

    script, report, chunks = _write_dialogue_fixture(tmp_path)
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"])
    prepared = prepare_run(args, chunks, report["style_prompt"], "native")
    host_voice, guest_voice = chunks[0].voice, chunks[1].voice
    providers = {host_voice: FakeProvider(), guest_voice: FakeProvider()}

    for part in prepared.parts:
        synthesize_part(providers, part)

    assert providers[host_voice].calls == [
        chunk.id for chunk in chunks if chunk.voice == host_voice
    ]
    assert providers[guest_voice].calls == [
        chunk.id for chunk in chunks if chunk.voice == guest_voice
    ]

    openrouter_args = make_args(tmp_path, run_id="legacy-openrouter")
    openrouter_args.provider = "openrouter-tts"
    openrouter_args.model = "google/gemini-3.1-flash-tts-preview"
    legacy_part = prepare_run(openrouter_args, chunks[:1], None, "none").parts[0]
    provider = OpenRouterTTSProvider(
        api_key="test", model=openrouter_args.model, voice="Kore", prompt_mode="none"
    )
    legacy_calls: list[tuple[str, str]] = []

    def legacy_synthesize_chunk(self, text, chunk_id):
        legacy_calls.append((text, chunk_id))
        return SynthesisResult(audio_bytes=b"audio", audio_format="mp3")

    monkeypatch.setattr(OpenRouterTTSProvider, "synthesize_chunk", legacy_synthesize_chunk)

    result = synthesize_part(provider, legacy_part)

    assert legacy_calls == [(chunks[0].text, chunks[0].id)]
    assert result.audio_bytes == b"audio"

    def broken_synthesize_chunk(self, text, chunk_id, voice=None):
        raise TypeError("internal provider failure")

    monkeypatch.setattr(OpenRouterTTSProvider, "synthesize_chunk", broken_synthesize_chunk)

    with pytest.raises(TypeError, match="internal provider failure"):
        synthesize_part(provider, legacy_part)


def test_synthesize_part_does_not_resubmit_after_a_voice_typeerror(tmp_path, monkeypatch):
    """A rejected first submit must not trigger a second paid OpenRouter call.

    If an OpenRouter submit has already left the process and then raises
    ``TypeError`` mentioning ``voice``, the single synthesis seam must let that
    error escape after exactly one provider call. Re-invoking the provider would
    be a second paid submit for the same part.
    """
    from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
    from voiceover_pipeline.services.prepare import prepare_run
    from voiceover_pipeline.services.synthesis import synthesize_part

    script, report, chunks = _write_dialogue_fixture(tmp_path)
    args = make_dialogue_args(tmp_path, script, report["speaker_voice_map"])
    part = prepare_run(args, chunks[:1], report["style_prompt"], "native").parts[0]
    provider = OpenRouterTTSProvider(
        api_key="test",
        model=args.model,
        voice=args.voice,
        speaker_voice_map=report["speaker_voice_map"],
        prompt_mode="native",
    )
    calls: list[tuple[str, str]] = []

    def submitted_then_rejected(self, text, chunk_id, voice=None):
        calls.append((text, chunk_id))
        raise TypeError("unexpected keyword argument 'voice'")

    monkeypatch.setattr(OpenRouterTTSProvider, "synthesize_chunk", submitted_then_rejected)

    with pytest.raises(TypeError, match="unexpected keyword argument 'voice'"):
        synthesize_part(provider, part)

    assert calls == [(part.chunk.text, part.chunk.id)]


def test_generate_step_routes_media_hooks_through_patched_cli_functions(tmp_path, monkeypatch):
    """The extracted part loop still calls the CLI-bound media and duration functions.

    ``write_audio_as_mp3``, ``trim_final_silence``, and ``mp3_duration_ms`` are
    patched on ``cli`` throughout the suite; the moved loop must invoke those
    exact callables in the same per-chunk order so existing patches keep
    controlling conversion, trimming, and timing.
    """
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths
    from voiceover_pipeline.script_splitter import split_markdown_by_delimiter

    patch_generation_io(monkeypatch)
    args = make_args(tmp_path, run_id="media-hook-parity")
    args.no_trim = False
    paths = build_run_paths(args.output_dir, args.model, args.run_id)
    paths.chunks_dir.mkdir(parents=True)
    chunks = split_markdown_by_delimiter(args.script, "******")[:2]
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    def write_hook(_ffmpeg, _audio, _fmt, path):
        calls.append(("write", path.name))
        path.write_bytes(b"mp3")

    def trim_hook(_ffmpeg, _ffprobe, path):
        calls.append(("trim", path.name))

    def duration_hook(_ffprobe, path):
        calls.append(("duration", path.name))
        return 1000

    monkeypatch.setattr(cli, "write_audio_as_mp3", write_hook)
    monkeypatch.setattr(cli, "trim_final_silence", trim_hook)
    monkeypatch.setattr(cli, "mp3_duration_ms", duration_hook)

    with pytest.raises(SystemExit) as exit_info:
        cli._generate_step(
            args, FakeProvider(), "ffmpeg", "ffprobe", chunks, "key", None, paths, None, "auto"
        )

    assert exit_info.value.code == 0
    assert calls == [
        ("write", "chunk_01.mp3"),
        ("trim", "chunk_01.mp3"),
        ("duration", "chunk_01.mp3"),
        ("write", "chunk_02.mp3"),
        ("trim", "chunk_02.mp3"),
        ("duration", "chunk_02.mp3"),
        ("duration", paths.full_mp3.name),
    ]
