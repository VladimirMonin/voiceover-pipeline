from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import voiceover_pipeline.asr_longform as longform
from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import ASRDependencyHealth, ASRProviderSpec
from voiceover_pipeline.providers.base import ASRProvider


def _result(
    words: tuple[ASRWordSpan, ...],
    *,
    measurements: dict[str, float] | None = None,
    alignment_origin: str = "forced",
    provider_id: str = "qwen-local",
) -> ASRResult:
    return ASRResult(
        transcript="".join(word.text for word in words),
        provider_id=provider_id,
        model_id="fixture-model",
        language="ru",
        words=words,
        alignment_origin=alignment_origin,
        execution=ASRExecutionReceipt(
            runtime="fixture-asr",
            runtime_version="1.0",
            resolved_device="cpu",
            resolved_compute="float32",
            measurements=measurements or {},
        ),
    )


def _text_result(text: str) -> ASRResult:
    return ASRResult(
        transcript=text,
        provider_id="nemotron-local",
        model_id="fixture-model",
        language="ru",
        execution=ASRExecutionReceipt(
            runtime="fixture-asr",
            runtime_version="1.0",
            resolved_device="cpu",
            resolved_compute="float32",
        ),
    )


class _ChunkProvider(ASRProvider):
    provider_id = "qwen-local"

    def __init__(self, results: list[ASRResult]) -> None:
        self._results = results
        self.requests: list[ASRRequest] = []

    def transcribe(self, request: ASRRequest) -> ASRResult:
        self.requests.append(request)
        return self._results[len(self.requests) - 1]


def _patch_media(monkeypatch: pytest.MonkeyPatch, duration_s: float) -> None:
    monkeypatch.setattr(longform, "_source_duration_s", lambda _source: duration_s)
    monkeypatch.setattr(longform, "_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(longform, "_silence_boundaries", lambda *_args: ())

    def extract(
        _ffmpeg_path: str,
        _source: Path,
        _chunk: longform.ASRChunkPlan,
        output_path: Path,
    ) -> float:
        output_path.write_bytes(b"fixture")
        return _chunk.input_duration_s

    monkeypatch.setattr(longform, "_extract_chunk", extract)


def _request(source: Path) -> ASRRequest:
    return ASRRequest(
        audio_path=source,
        model_id="fixture-model",
        language="ru",
        device="cpu",
        compute="float32",
        timestamp_mode="word",
    )


@pytest.mark.parametrize(
    ("duration_s", "expected_chunks"),
    [
        (longform.LONG_FORM_HARD_MAX_S - 0.001, 1),
        (longform.LONG_FORM_HARD_MAX_S, 1),
        (longform.LONG_FORM_HARD_MAX_S + 0.001, 2),
    ],
)
def test_chunk_plan_covers_n_minus_one_n_and_n_plus_one_without_model_overrun(
    duration_s: float,
    expected_chunks: int,
) -> None:
    chunks = longform.plan_prerecorded_chunks(duration_s)

    assert len(chunks) == expected_chunks
    assert chunks[0].coverage_start_s == pytest.approx(0.0)
    assert chunks[-1].coverage_end_s == pytest.approx(duration_s)
    assert all(
        chunk.input_end_s - chunk.input_start_s <= longform.LONG_FORM_HARD_MAX_S for chunk in chunks
    )
    assert all(
        later.coverage_start_s == pytest.approx(previous.coverage_end_s)
        for previous, later in zip(chunks, chunks[1:])
    )


def test_chunk_plan_prefers_nearby_silence_without_creating_coverage_gaps() -> None:
    chunks = longform.plan_prerecorded_chunks(360.0, silence_boundaries=(107.0, 218.0))

    assert chunks[0].coverage_end_s == pytest.approx(107.0)
    assert chunks[1].coverage_end_s == pytest.approx(218.0)
    assert chunks[-1].coverage_end_s == pytest.approx(360.0)
    assert all(
        later.coverage_start_s == pytest.approx(previous.coverage_end_s)
        for previous, later in zip(chunks, chunks[1:])
    )


def test_chunk_plan_rejects_silence_before_the_minimum_long_form_target() -> None:
    chunks = longform.plan_prerecorded_chunks(250.0, silence_boundaries=(20.0,))

    assert chunks[0].coverage_end_s == pytest.approx(longform.LONG_FORM_TARGET_S)
    assert [(chunk.coverage_start_s, chunk.coverage_end_s) for chunk in chunks] == [
        (0.0, 110.0),
        (110.0, 220.0),
        (220.0, 250.0),
    ]


def test_long_form_merges_absolute_word_timestamps_and_retains_unique_boundary_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "six-minutes.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=360.0)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="opening ", start_s=0.1, end_s=0.4),
                    ASRWordSpan(text="boundary ", start_s=108.9, end_s=109.1),
                    ASRWordSpan(text="words", start_s=109.2, end_s=109.4),
                )
            ),
            _result(
                (
                    ASRWordSpan(text="boundary ", start_s=0.0, end_s=0.2),
                    ASRWordSpan(text="words ", start_s=0.3, end_s=0.5),
                    ASRWordSpan(text="unique", start_s=0.6, end_s=0.9),
                )
            ),
            _result((ASRWordSpan(text="middle", start_s=0.4, end_s=0.8),)),
            _result((ASRWordSpan(text="tail", start_s=30.7, end_s=30.95),)),
        ]
    )

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert len(provider.requests) == 4
    assert [Path(request.audio_path).name for request in provider.requests] == [
        "chunk-0000.wav",
        "chunk-0001.wav",
        "chunk-0002.wav",
        "chunk-0003.wav",
    ]
    assert {
        (
            request.model_id,
            request.language,
            request.device,
            request.compute,
            request.hints,
            request.timestamp_mode,
        )
        for request in provider.requests
    } == {("fixture-model", "ru", "cpu", "float32", provider.requests[0].hints, "word")}
    assert result.duration_s == pytest.approx(360.0)
    assert result.transcript == "opening boundary words unique middle tail"
    assert [word.text.strip() for word in result.words] == [
        "opening",
        "boundary",
        "words",
        "unique",
        "middle",
        "tail",
    ]
    assert [word.start_s for word in result.words] == pytest.approx(
        [0.1, 108.9, 109.2, 109.6, 219.4, 359.7]
    )
    assert [(segment.start_s, segment.end_s) for segment in result.segments] == [
        (0.0, 110.0),
        (110.0, 220.0),
        (220.0, 330.0),
        (330.0, 360.0),
    ]
    assert result.alignment_origin == "forced"
    assert result.execution.long_form is not None
    assert result.execution.long_form["coverage_verified"] is True
    assert result.execution.long_form["processed_duration_s"] == pytest.approx(363.0)
    assert result.execution.long_form["planned_processed_duration_s"] == pytest.approx(363.0)
    chunks = result.execution.long_form["chunks"]
    assert isinstance(chunks, list)
    first_chunk = chunks[0]
    assert isinstance(first_chunk, dict)
    assert first_chunk["output_duration_s"] == pytest.approx(110.0)
    assert first_chunk["output_status"] == "duration_verified"
    last_chunk = chunks[-1]
    assert isinstance(last_chunk, dict)
    assert last_chunk["coverage_end_s"] == pytest.approx(360.0)


def test_long_form_forced_word_merge_preserves_readable_spacing_and_punctuation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "two-minutes.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="Первый, ", start_s=0.1, end_s=0.4),
                    ASRWordSpan(text="фрагмент.", start_s=109.0, end_s=109.4),
                )
            ),
            _result((ASRWordSpan(text="Второй!", start_s=0.4, end_s=0.8),)),
        ]
    )

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert result.transcript == "Первый, фрагмент. Второй!"
    assert "".join(word.text for word in result.words) == result.transcript
    assert [word.start_s for word in result.words] == pytest.approx([0.1, 109.0, 109.4])
    assert result.alignment_origin == "forced"


def test_long_form_nemotron_native_words_apply_chunk_offset_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "nemotron-long.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="opening ", start_s=0.1, end_s=0.4),
                    ASRWordSpan(text="boundary", start_s=108.9, end_s=109.4),
                ),
                measurements={"wall_s": 0.5},
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (
                    ASRWordSpan(text="boundary ", start_s=0.0, end_s=0.2),
                    ASRWordSpan(text="unique", start_s=0.5, end_s=0.8),
                ),
                measurements={"wall_s": 0.5},
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert result.provider_id == "nemotron-local"
    assert result.alignment_origin == "native"
    assert result.transcript == "opening boundary unique"
    assert [word.start_s for word in result.words] == pytest.approx([0.1, 108.9, 109.5])
    assert [word.end_s for word in result.words] == pytest.approx([0.4, 109.4, 109.8])
    assert result.execution.long_form is not None
    assert result.execution.long_form["deduplicated_word_count"] == 1


def test_long_form_nemotron_reconciles_mismatched_boundary_words_by_absolute_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "mismatched-boundary.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="alpha ", start_s=108.16, end_s=109.6),
                    ASRWordSpan(text="beta", start_s=109.6, end_s=110.0),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (
                    ASRWordSpan(text="gamma ", start_s=0.32, end_s=0.56),
                    ASRWordSpan(text="delta", start_s=0.64, end_s=2.64),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert result.provider_id == "nemotron-local"
    assert result.alignment_origin == "native"
    assert [word.start_s for word in result.words] == pytest.approx([108.16, 109.64])
    assert [word.end_s for word in result.words] == pytest.approx([109.6, 111.64])
    assert [word.text for word in result.words] == ["alpha ", "delta"]
    assert result.transcript == "alpha delta"
    assert result.execution.long_form is not None
    assert result.execution.long_form["deduplicated_word_count"] == 2
    assert [(segment.text, segment.start_s, segment.end_s) for segment in result.segments] == [
        ("alpha delta", 0.0, 110.0),
        ("", 110.0, 120.001),
    ]


def test_long_form_overlap_keeps_identical_text_at_distinct_nonoverlapping_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "repeated-words.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="lead ", start_s=108.0, end_s=108.4),
                    ASRWordSpan(text="repeat ", start_s=109.0, end_s=109.1),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (
                    ASRWordSpan(text="repeat ", start_s=0.8, end_s=0.9),
                    ASRWordSpan(text="tail", start_s=2.4, end_s=2.8),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert [word.text for word in result.words] == ["lead ", "repeat ", "repeat ", "tail"]
    assert [word.start_s for word in result.words] == pytest.approx([108.0, 109.0, 109.8, 111.4])
    assert result.execution.long_form is not None
    assert result.execution.long_form["deduplicated_word_count"] == 0


def test_long_form_overlap_reconciliation_never_emits_sub_epsilon_word_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sub-epsilon-overlap.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (ASRWordSpan(text="tail ", start_s=109.5, end_s=110.0),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (ASRWordSpan(text="head", start_s=0.99, end_s=1.2),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert [(word.start_s, word.end_s) for word in result.words] == pytest.approx([(109.99, 110.2)])
    assert result.transcript == "head"
    assert result.execution.long_form is not None
    assert result.execution.long_form["deduplicated_word_count"] == 1


def test_long_form_overlap_reconciliation_drops_crossing_hypothesis_instead_of_pre_overlap_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "unspliceable-seam.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (ASRWordSpan(text="longword ", start_s=108.0, end_s=110.0),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (ASRWordSpan(text="crossing", start_s=0.9, end_s=2.0),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert [word.text for word in result.words] == ["longword "]
    assert result.transcript == "longword"
    assert result.execution.long_form is not None
    assert result.execution.long_form["deduplicated_word_count"] == 1


def test_long_form_fails_closed_when_a_provider_reports_token_limit_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "over-limit.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (ASRWordSpan(text="truncated", start_s=0.0, end_s=0.1),),
                measurements={"token_limit_reached": 1.0},
            ),
            _result((ASRWordSpan(text="tail", start_s=0.0, end_s=0.1),)),
        ]
    )

    with pytest.raises(longform.LongFormASRError, match="token_limit_reached"):
        longform.transcribe_prerecorded_long_form(provider, _request(source))


def test_long_form_supplies_chunk_timing_when_a_selected_route_has_no_word_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "text-only.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider([_text_result("alpha beta"), _text_result("gamma tail")])
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(
        provider,
        ASRRequest(
            audio_path=source,
            model_id="fixture-model",
            language="ru",
            device="cpu",
            compute="float32",
        ),
    )

    assert result.transcript == "alpha beta gamma tail"
    assert result.words == ()
    assert result.alignment_origin == "chunked"
    assert result.duration_s == pytest.approx(120.001)
    assert [(segment.start_s, segment.end_s) for segment in result.segments] == [
        (0.0, 110.0),
        (110.0, 120.001),
    ]


def test_long_form_text_only_uses_contiguous_inputs_and_preserves_repeated_boundary_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "repeated-boundary-text.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider([_text_result("alpha beta"), _text_result("alpha beta unique-tail")])
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(
        provider,
        ASRRequest(audio_path=source, model_id="fixture-model", language="ru"),
    )

    assert result.transcript == "alpha beta alpha beta unique-tail"
    assert result.execution.long_form is not None
    assert result.execution.long_form["overlap_s"] == pytest.approx(0.0)
    chunks = result.execution.long_form["chunks"]
    assert isinstance(chunks, list)
    assert chunks[1]["input_start_s"] == pytest.approx(chunks[0]["input_end_s"])


def test_long_form_validates_chunk_timestamps_against_actual_extracted_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "extracted-duration.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)

    def late_extract(
        _ffmpeg_path: str,
        _source: Path,
        chunk: longform.ASRChunkPlan,
        output_path: Path,
    ) -> float:
        output_path.write_bytes(b"fixture")
        return chunk.input_duration_s + 0.05

    monkeypatch.setattr(longform, "_extract_chunk", late_extract)
    provider = _ChunkProvider(
        [
            _result(
                (ASRWordSpan(text="first", start_s=0.0, end_s=0.2),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (ASRWordSpan(text="tail", start_s=0.0, end_s=0.2),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert [word.start_s for word in result.words] == pytest.approx([0.0, 109.0])
    assert result.transcript == "first tail"


def test_long_form_rejects_chunk_timestamp_beyond_actual_extracted_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "beyond-extracted.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="alpha ", start_s=0.0, end_s=0.4),
                    ASRWordSpan(text="beta", start_s=109.6, end_s=110.1),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (ASRWordSpan(text="tail", start_s=0.0, end_s=0.2),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    with pytest.raises(longform.LongFormASRError, match="beyond its bounded input"):
        longform.transcribe_prerecorded_long_form(provider, _request(source))


def test_long_form_accepts_chunk_timestamp_inside_extraction_delta_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "delta-window.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)

    def late_extract(
        _ffmpeg_path: str,
        _source: Path,
        chunk: longform.ASRChunkPlan,
        output_path: Path,
    ) -> float:
        output_path.write_bytes(b"fixture")
        return chunk.input_duration_s + 0.05

    monkeypatch.setattr(longform, "_extract_chunk", late_extract)
    provider = _ChunkProvider(
        [
            _result(
                (
                    ASRWordSpan(text="alpha ", start_s=0.0, end_s=0.4),
                    ASRWordSpan(text="beta", start_s=109.6, end_s=110.06),
                ),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
            _result(
                (ASRWordSpan(text="tail", start_s=0.0, end_s=0.2),),
                alignment_origin="native",
                provider_id="nemotron-local",
            ),
        ]
    )
    provider.provider_id = "nemotron-local"

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert [word.start_s for word in result.words] == pytest.approx([0.0, 109.6])
    assert [word.end_s for word in result.words] == pytest.approx([0.4, 110.06])
    assert result.transcript == "alpha beta"


def test_long_form_fails_closed_when_an_extracted_chunk_duration_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "short-extract.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)

    def short_extract(
        _ffmpeg_path: str,
        _source: Path,
        chunk: longform.ASRChunkPlan,
        output_path: Path,
    ) -> float:
        output_path.write_bytes(b"fixture")
        return chunk.input_duration_s - 1.0

    monkeypatch.setattr(longform, "_extract_chunk", short_extract)
    provider = _ChunkProvider(
        [_result((ASRWordSpan(text="must-not-run", start_s=0.0, end_s=0.1),))]
    )

    with pytest.raises(longform.LongFormASRError, match="ASR chunk 0 output duration"):
        longform.transcribe_prerecorded_long_form(provider, _request(source))

    assert provider.requests == []


def test_extract_chunk_measures_and_rejects_a_short_ffmpeg_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "chunk.wav"
    plan = longform.ASRChunkPlan(
        index=4,
        input_start_s=109.0,
        input_end_s=120.001,
        coverage_start_s=110.0,
        coverage_end_s=120.001,
    )

    def fake_run(_args: list[str], **_kwargs: object) -> SimpleNamespace:
        output.write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(longform.subprocess, "run", fake_run)
    monkeypatch.setattr(
        longform,
        "_measured_audio_duration_s",
        lambda _path, *, description: plan.input_duration_s - 1.0,
    )

    with pytest.raises(longform.LongFormASRMediaError, match="ASR chunk 4 output duration"):
        longform._extract_chunk("ffmpeg", source, plan, output)


def test_extracted_chunk_duration_accepts_a_bounded_codec_seek_delta() -> None:
    plan = longform.ASRChunkPlan(
        index=4,
        input_start_s=329.0,
        input_end_s=388.252,
        coverage_start_s=330.0,
        coverage_end_s=388.252,
    )

    longform._validate_extracted_duration(plan, 59.182)
    with pytest.raises(longform.LongFormASRMediaError, match="ASR chunk 4 output duration"):
        longform._validate_extracted_duration(plan, 59.151)


def test_long_form_json_payload_exposes_chunk_manifest_without_replacing_existing_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voiceover_pipeline.cli as cli

    source = tmp_path / "long.wav"
    source.write_bytes(b"source")
    _patch_media(monkeypatch, duration_s=120.001)
    provider = _ChunkProvider(
        [
            _result((ASRWordSpan(text="first", start_s=0.0, end_s=0.2),)),
            _result((ASRWordSpan(text="tail", start_s=0.0, end_s=0.2),)),
        ]
    )

    result = longform.transcribe_prerecorded_long_form(provider, _request(source))
    payload = cli._asr_result_payload(result, source)

    assert payload["status"] == "success"
    assert payload["duration_s"] == pytest.approx(120.001)
    assert payload["segments"]
    long_form = payload["execution"]["long_form"]
    assert long_form["coverage_verified"] is True
    assert long_form["extract_duration_tolerance_s"] == pytest.approx(0.1)
    assert len(long_form["chunks"]) == 2
    assert long_form["chunks"][0]["output_duration_s"] == pytest.approx(110.0)
    assert long_form["chunks"][0]["output_duration_delta_s"] == pytest.approx(0.0)
    assert long_form["chunks"][0]["output_duration_tolerance_s"] == pytest.approx(0.1)


@pytest.mark.parametrize("provider_id", ["qwen-local", "nemotron-local"])
def test_public_transcribe_routes_selected_local_families_through_long_form_layer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider_id: str,
) -> None:
    import voiceover_pipeline.cli as cli

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    provider = _ChunkProvider([_result((ASRWordSpan(text="fixture", start_s=0.0, end_s=0.2),))])
    provider.provider_id = provider_id
    result = ASRResult(
        transcript="fixture",
        provider_id=provider_id,
        model_id="fixture-model",
        duration_s=10.0,
        segments=(ASRSegment(text="fixture", start_s=0.0, end_s=10.0),),
        alignment_origin="chunked",
        execution=ASRExecutionReceipt(
            runtime="fixture-asr",
            resolved_device="cpu",
            resolved_compute="float32",
            long_form={"coverage_verified": True, "chunks": []},
        ),
    )
    spec = ASRProviderSpec(
        provider_id=provider_id,
        description="fixture",
        factory=lambda: provider,
        models=({"id": "fixture-model", "default": True},),
        capabilities=ASRCapabilities(
            batch_audio=True,
            forced_language=True,
            segment_timestamps=True,
            word_timestamps=True,
            forced_alignment=True,
            device_modes=("cpu",),
            compute_modes=("float32",),
        ),
        dependency_probe=lambda: ASRDependencyHealth(available=True, remediation=""),
    )
    captured: list[tuple[object, ASRRequest]] = []
    monkeypatch.setattr(cli, "get_asr_provider_spec", lambda _provider_id: spec)
    monkeypatch.setattr(
        cli,
        "transcribe_prerecorded_long_form",
        lambda supplied_provider, request: captured.append((supplied_provider, request)) or result,
    )
    args = argparse.Namespace(
        audio=str(source),
        provider=provider_id,
        model="fixture-model",
        language="ru",
        device="cpu",
        compute="float32",
        word_timestamps=False,
        json_output=True,
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.transcribe_cmd(args)

    assert exit_info.value.code == 0
    assert captured == [(provider, ASRRequest(source, "fixture-model", "ru", "cpu", "float32"))]
    assert (
        json.loads(capsys.readouterr().out)["execution"]["long_form"]["coverage_verified"] is True
    )
