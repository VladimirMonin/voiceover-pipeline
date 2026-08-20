from pathlib import Path
from typing import MutableMapping, cast

import pytest

from voiceover_pipeline.models import (
    ASRAlignmentOrigin,
    ASRContextHints,
    ASRExecutionReceipt,
    ASRGlossaryHint,
    ASRPhraseHint,
    ASRPhraseStrength,
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRTimestampMode,
    ASRWordSpan,
)
from voiceover_pipeline.providers.base import validate_asr_response


def _receipt() -> ASRExecutionReceipt:
    return ASRExecutionReceipt(
        runtime="fixture-runtime",
        runtime_version="1.0",
        resolved_device="cpu",
        resolved_compute="float32",
    )


def test_text_only_asr_result_is_valid_without_timestamp_claims():
    result = ASRResult(
        transcript="Привет, мир",
        provider_id="fixture-local",
        model_id="fixture-model",
        execution=_receipt(),
    )

    assert result.transcript == "Привет, мир"
    assert result.segments == ()
    assert result.words == ()
    assert result.alignment_origin is None
    with pytest.raises(TypeError):
        cast(MutableMapping[str, float], result.execution.measurements)["wall_s"] = 1.0


def test_native_and_forced_alignment_origins_are_distinguishable():
    segment = ASRSegment(text="Привет", start_s=0.0, end_s=0.5)

    native = ASRResult(
        transcript="Привет",
        provider_id="fixture-local",
        model_id="fixture-model",
        execution=_receipt(),
        segments=(segment,),
        alignment_origin="native",
    )
    forced = ASRResult(
        transcript="Привет",
        provider_id="fixture-local",
        model_id="fixture-model",
        execution=_receipt(),
        segments=(segment,),
        alignment_origin="forced",
    )

    assert native.alignment_origin == "native"
    assert forced.alignment_origin == "forced"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ASRSegment(text="bad", start_s=-0.1, end_s=0.1), "non-negative"),
        (lambda: ASRSegment(text="bad", start_s=1.0, end_s=0.5), "before start"),
        (lambda: ASRWordSpan(text="bad", start_s=-0.1, end_s=0.1), "non-negative"),
        (lambda: ASRWordSpan(text="bad", start_s=1.0, end_s=0.5), "before start"),
    ],
)
def test_timestamp_spans_reject_negative_and_reversed_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_result_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="monotonic"):
        ASRResult(
            transcript="two segments",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            segments=(
                ASRSegment(text="later", start_s=1.0, end_s=1.5),
                ASRSegment(text="earlier", start_s=0.5, end_s=0.9),
            ),
            alignment_origin="native",
        )


def test_result_rejects_overlapping_timestamp_spans_even_when_starts_increase():
    with pytest.raises(ValueError, match="non-overlapping"):
        ASRResult(
            transcript="first second",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            words=(
                ASRWordSpan(text="first ", start_s=0.0, end_s=99.0),
                ASRWordSpan(text="second", start_s=1.0, end_s=2.0),
            ),
            alignment_origin="native",
        )


def test_context_glossary_and_phrase_hints_are_typed_without_generic_prompt():
    hints = ASRContextHints(
        context_text="Профиль: серверная документация",
        glossary=ASRGlossaryHint(profile_id="ops-v1", digest="sha256:fixture"),
        phrase_hints=(ASRPhraseHint(text="Whisper Voice Machine", strength="strong"),),
    )
    request = ASRRequest(audio_path=Path("fixture.wav"), hints=hints)

    assert request.hints.glossary.profile_id == "ops-v1"
    assert request.hints.phrase_hints[0].strength == "strong"
    assert not hasattr(request, "prompt")


def test_timestamp_mode_is_an_explicit_closed_request_intent_with_text_only_default():
    assert ASRRequest(audio_path=Path("fixture.wav")).timestamp_mode == "none"
    assert (
        ASRRequest(audio_path=Path("fixture.wav"), timestamp_mode="word").timestamp_mode == "word"
    )

    with pytest.raises(ValueError, match="timestamp mode"):
        ASRRequest(audio_path=Path("fixture.wav"), timestamp_mode=cast(ASRTimestampMode, "segment"))


def test_requested_word_timestamps_reject_speech_without_words_and_out_of_bounds_spans():
    request = ASRRequest(audio_path=Path("fixture.wav"), timestamp_mode="word")
    text_only = ASRResult(
        transcript="spoken fixture",
        provider_id="fixture-local",
        model_id="fixture-model",
        execution=_receipt(),
    )

    with pytest.raises(ValueError, match="requested word timestamps"):
        validate_asr_response(request, text_only)
    with pytest.raises(ValueError, match="source duration"):
        ASRResult(
            transcript="spoken fixture",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            duration_s=1.0,
            words=(ASRWordSpan(text="spoken fixture", start_s=0.0, end_s=1.1),),
            alignment_origin="native",
        )


def test_word_spans_must_describe_non_empty_transcript_text():
    valid = ASRResult(
        transcript="Привет, мир!",
        provider_id="fixture-local",
        model_id="fixture-model",
        execution=_receipt(),
        words=(
            ASRWordSpan(text="Привет", start_s=0.0, end_s=0.2),
            ASRWordSpan(text="мир", start_s=0.3, end_s=0.5),
        ),
        alignment_origin="native",
    )

    assert valid.words[0].text == "Привет"
    with pytest.raises(ValueError, match="non-empty speech transcript"):
        ASRResult(
            transcript="",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            words=(ASRWordSpan(text="fabricated", start_s=0.0, end_s=0.2),),
            alignment_origin="native",
        )
    with pytest.raises(ValueError, match="correspond to the transcript"):
        ASRResult(
            transcript="expected words",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            words=(ASRWordSpan(text="unrelated", start_s=0.0, end_s=0.2),),
            alignment_origin="native",
        )


def test_phrase_hint_strength_and_alignment_origin_are_closed_sets():
    with pytest.raises(ValueError, match="strength"):
        ASRPhraseHint(text="fixture", strength=cast(ASRPhraseStrength, "maximum"))
    with pytest.raises(ValueError, match="alignment origin"):
        ASRResult(
            transcript="fixture",
            provider_id="fixture-local",
            model_id="fixture-model",
            execution=_receipt(),
            alignment_origin=cast(ASRAlignmentOrigin, "synthetic"),
        )
