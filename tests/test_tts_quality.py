import hashlib

from voiceover_pipeline.tts_quality import evaluate_tts_transcript


def test_quality_passes_small_punctuation_and_inflection_drift():
    result = evaluate_tts_transcript(
        expected_text="Это точный тест качества речи.",
        actual_transcript="Это точный тест качества речи",
    )

    assert result.passed is True
    assert result.similarity == 1.0
    assert result.unexpected_word_count == 0
    assert result.missing_word_count == 0


def test_quality_fails_hallucinated_insertion_without_publishing_text():
    expected = "Один два три четыре пять шесть семь восемь девять десять"
    actual = expected + " посторонняя речь которой не было в исходном сценарии"

    result = evaluate_tts_transcript(expected_text=expected, actual_transcript=actual)
    payload = result.public_receipt(
        audio_sha256="a" * 64,
        asr_provider="fixture-asr",
        asr_model="fixture-model",
        asr_runtime="fixture-runtime",
        asr_model_revision="fixture-revision",
    )

    assert result.passed is False
    assert result.unexpected_word_count == 8
    assert result.repeated_ngram_excess == 0
    assert "unexpected_words" in result.failure_reasons
    assert (
        payload["expected_text_sha256"]
        == hashlib.sha256(
            "один два три четыре пять шесть семь восемь девять десять".encode()
        ).hexdigest()
    )
    assert (
        payload["actual_transcript_sha256"]
        == hashlib.sha256(
            "один два три четыре пять шесть семь восемь девять десять "
            "посторонняя речь которой не было в исходном сценарии".encode()
        ).hexdigest()
    )
    assert expected not in str(payload)
    assert actual not in str(payload)


def test_quality_fails_major_omission_and_repetition():
    result = evaluate_tts_transcript(
        expected_text="раз два три четыре пять шесть семь восемь девять десять",
        actual_transcript="раз два два два два",
    )

    assert result.passed is False
    assert result.missing_word_count >= 8
    assert result.repeated_ngram_excess > 0
    assert "missing_words" in result.failure_reasons
    assert "repeated_ngram" in result.failure_reasons
