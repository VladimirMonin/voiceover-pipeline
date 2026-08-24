import argparse
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from voiceover_pipeline.gemini_dialogue import (
    dialogue_turns_from_validation,
    validate_gemini_dialogue_file,
)
from voiceover_pipeline.models import ASRExecutionReceipt, ASRResult, ChunkArtifact
from voiceover_pipeline.providers.openrouter_tts import OpenRouterTTSProvider
from voiceover_pipeline.tts_quality import evaluate_tts_transcript

_MODEL = "google/gemini-3.1-flash-tts-preview"
_TURNS = [
    "Первый синтетический вопрос проверяет точную изоляцию реплики.",
    "Второй синтетический ответ не должен попадать в соседний запрос.",
    "Третий синтетический вопрос проверяет порядок после возобновления.",
    "Четвёртый синтетический ответ нельзя повторять после завершения.",
]


def _dialogue_plan(tmp_path):
    script = tmp_path / "dialogue.md"
    script.write_text(
        "\n".join(
            [
                "---",
                "format: dialogue",
                "language: ru",
                f"model: {_MODEL}",
                "speakers:",
                "  Host:",
                "    display_name: Ведущая",
                "    voice: Kore",
                "    profile: warm profile forbidden in synthesis input",
                "  Guest:",
                "    display_name: Гость",
                "    voice: Puck",
                "    profile: calm profile forbidden in synthesis input",
                "vibe: forbidden shared dialogue vibe",
                "---",
                f"Host: {_TURNS[0]}",
                f"Guest: {_TURNS[1]}",
                "******",
                f"Host: {_TURNS[2]}",
                f"Guest: {_TURNS[3]}",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_gemini_dialogue_file(script)
    assert report["valid"] is True
    return report, dialogue_turns_from_validation(report)


def _capture_requests(report, chunks):
    response = MagicMock()
    response.status_code = 200
    response.content = b"pcm"
    response.headers = {"Content-Type": "audio/pcm"}
    provider = OpenRouterTTSProvider(
        api_key="test",
        model=_MODEL,
        voice="Kore",
        style_prompt=report["style_prompt"],
        speaker_voice_map=report["speaker_voice_map"],
        prompt_mode="prefix",
    )
    with patch(
        "voiceover_pipeline.providers.openrouter_tts.requests.post", return_value=response
    ) as post:
        for chunk in chunks:
            provider.synthesize_chunk(chunk.text, chunk.id, voice=chunk.voice)
    return [call.kwargs["json"] for call in post.call_args_list]


def _assert_verbatim_bodies(bodies, expected_turns):
    assert len(bodies) == len(expected_turns)
    for index, (body, expected) in enumerate(zip(bodies, expected_turns, strict=True)):
        assert set(body) == {"model", "input", "voice", "response_format"}
        assert body["model"] == _MODEL
        assert body["input"] == expected
        assert body["response_format"] == "pcm"
        for other_index, other_turn in enumerate(_TURNS):
            if other_index != index:
                assert other_turn not in body["input"]
        assert not any(
            forbidden in body["input"]
            for forbidden in ("Host", "Guest", "Kore", "Puck", "profile", "vibe")
        )


def test_openrouter_dialogue_sends_all_four_turns_verbatim(tmp_path):
    report, chunks = _dialogue_plan(tmp_path)

    bodies = _capture_requests(report, chunks)

    _assert_verbatim_bodies(bodies, _TURNS)
    assert [body["voice"] for body in bodies] == ["Kore", "Puck", "Kore", "Puck"]


def test_openrouter_limit_chunks_cannot_leak_later_turns(tmp_path):
    report, chunks = _dialogue_plan(tmp_path)

    bodies = _capture_requests(report, chunks[:2])

    _assert_verbatim_bodies(bodies, _TURNS[:2])
    assert _TURNS[2] not in json.dumps(bodies, ensure_ascii=False)
    assert _TURNS[3] not in json.dumps(bodies, ensure_ascii=False)


def test_openrouter_resumed_suffix_cannot_leak_completed_turns(tmp_path):
    report, chunks = _dialogue_plan(tmp_path)

    bodies = _capture_requests(report, chunks[2:])

    assert [body["input"] for body in bodies] == _TURNS[2:]
    assert _TURNS[0] not in json.dumps(bodies, ensure_ascii=False)
    assert _TURNS[1] not in json.dumps(bodies, ensure_ascii=False)


def _artifact(number, chunk_id):
    return ChunkArtifact(
        number=number,
        id=chunk_id,
        file=f"{chunk_id}.mp3",
        duration_ms=1000,
        duration_sec=1.0,
        start_ms=(number - 1) * 1000,
        end_ms=number * 1000,
        text_characters=len(_TURNS[number - 1]),
        transcript=None,
        client_path="fixture",
        generation_id=f"generation-{number}",
        turn_index=number,
        audio_sha256=hashlib.sha256(f"audio-{number}".encode()).hexdigest(),
    )


def _quality_args():
    return argparse.Namespace(
        tts_quality_provider="fixture-asr",
        tts_quality_model="fixture-model",
        tts_quality_language="ru",
        tts_quality_device="cpu",
        tts_quality_compute="auto",
        tts_quality_runtime="auto",
    )


def _assert_receipt_has_no_plain_transcript(receipt):
    assert "transcript" not in receipt
    assert all("transcript" not in turn for turn in receipt["turns"])
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert not any(turn in serialized for turn in _TURNS)


def test_dialogue_quality_gate_fails_before_concat_on_inserted_next_turn(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths

    report, chunks = _dialogue_plan(tmp_path)
    paths = build_run_paths(tmp_path / "out", _MODEL, "quality-fail")
    paths.chunks_dir.mkdir(parents=True)
    for chunk in chunks:
        (paths.chunks_dir / f"{chunk.id}.mp3").write_bytes(f"audio-{chunk.number}".encode())
    transcripts = iter([_TURNS[0] + " " + _TURNS[1], *_TURNS[1:]])

    def transcribe(args):
        return (
            ASRResult(
                transcript=next(transcripts),
                provider_id="fixture-asr",
                model_id="fixture-model",
                execution=ASRExecutionReceipt(runtime="fixture-runtime"),
            ),
            args.audio,
        )

    monkeypatch.setattr(cli, "_transcribe_result", transcribe)

    with pytest.raises(cli.CliError, match="quality gate failed") as error:
        cli._verify_dialogue_turns_before_concat(
            _quality_args(), chunks, [_artifact(i, c.id) for i, c in enumerate(chunks, 1)], paths
        )

    assert error.value.code == 60
    receipt = json.loads((paths.output_root / "tts_quality.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "quality_failed"
    assert receipt["turns"][0]["passed"] is False
    _assert_receipt_has_no_plain_transcript(receipt)


def test_dialogue_quality_gate_accepts_exact_four_turns_and_writes_hashes(tmp_path, monkeypatch):
    import voiceover_pipeline.cli as cli
    from voiceover_pipeline.artifacts import build_run_paths

    _report, chunks = _dialogue_plan(tmp_path)
    paths = build_run_paths(tmp_path / "out", _MODEL, "quality-pass")
    paths.chunks_dir.mkdir(parents=True)
    for chunk in chunks:
        (paths.chunks_dir / f"{chunk.id}.mp3").write_bytes(f"audio-{chunk.number}".encode())
    transcripts = iter(_TURNS)

    def transcribe(args):
        return (
            ASRResult(
                transcript=next(transcripts),
                provider_id="fixture-asr",
                model_id="fixture-model",
                execution=ASRExecutionReceipt(runtime="fixture-runtime"),
            ),
            args.audio,
        )

    monkeypatch.setattr(cli, "_transcribe_result", transcribe)

    receipt = cli._verify_dialogue_turns_before_concat(
        _quality_args(), chunks, [_artifact(i, c.id) for i, c in enumerate(chunks, 1)], paths
    )

    assert receipt["status"] == "success"
    assert receipt["passed"] is True
    assert [turn["turn_index"] for turn in receipt["turns"]] == [1, 2, 3, 4]
    assert all(turn["passed"] for turn in receipt["turns"])
    _assert_receipt_has_no_plain_transcript(receipt)


def test_observed_repeated_fourth_and_six_turn_aggregate_fail_quality():
    repeated_fourth = evaluate_tts_transcript(
        expected_text=_TURNS[3], actual_transcript=f"{_TURNS[3]} {_TURNS[3]}"
    )
    six_turn_aggregate = evaluate_tts_transcript(
        expected_text=" ".join(_TURNS),
        actual_transcript=" ".join(
            [_TURNS[0], _TURNS[1], _TURNS[1], _TURNS[2], _TURNS[3], _TURNS[3]]
        ),
    )

    assert repeated_fourth.passed is False
    assert "unexpected_words" in repeated_fourth.failure_reasons
    assert six_turn_aggregate.passed is False
    assert "unexpected_words" in six_turn_aggregate.failure_reasons
