import pytest

from voiceover_pipeline.audio_cpp.nemotron_words import normalize_nemotron_word_timestamps


def test_normalizes_sentencepiece_cyrillic_latin_terms_punctuation_and_shared_frames():
    words = normalize_nemotron_word_timestamps(
        (
            {"text": "При", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
            {"text": "вет", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
            {"text": ",", "start_s": 0.0, "end_s": 0.1, "frame_index": 1},
            {"text": " мир", "start_s": 0.4, "end_s": 0.6, "frame_index": 4},
            {"text": " Post", "start_s": 0.8, "end_s": 0.9, "frame_index": 8},
            {"text": "gre", "start_s": 0.8, "end_s": 1.0, "frame_index": 8},
            {"text": "SQL", "start_s": 1.0, "end_s": 1.1, "frame_index": 10},
        ),
        response_offset_s=10.0,
    )

    assert [(word.text, word.start_s, word.end_s, word.confidence) for word in words] == [
        ("Привет, ", 10.0, 10.1, None),
        ("мир ", 10.4, 10.6, None),
        ("PostgreSQL", 10.8, 11.1, None),
    ]


def test_normalizer_keeps_zero_duration_spans_and_drops_blank_or_out_of_keep_entries():
    words = normalize_nemotron_word_timestamps(
        (
            {"text": "▁ноль", "start_s": 0.0, "end_s": 0.0},
            {"text": "", "start_s": 0.1, "end_s": 0.2},
            {"text": "   ", "start_s": 0.2, "end_s": 0.25},
            {"text": "▁skip", "start_s": 0.1, "end_s": 0.2, "keep": False},
            {"text": "▁one", "start_s": 0.3, "end_s": 0.3},
        )
    )

    assert [(word.text, word.start_s, word.end_s) for word in words] == [
        ("ноль ", 0.0, 0.0),
        ("one", 0.3, 0.3),
    ]


@pytest.mark.parametrize(
    ("entries", "message"),
    (
        (({"text": 3, "start_s": 0.0, "end_s": 0.1},), "text"),
        (({"text": "▁word", "start_s": "bad", "end_s": 0.1},), "start_s"),
        (({"text": "▁word", "start_s": 0.2, "end_s": 0.1},), "end_s"),
        (({"text": "▁word", "start_s": 0.0, "end_s": 0.1, "keep": "yes"},), "keep"),
    ),
)
def test_normalizer_rejects_malformed_raw_entries(entries, message):
    with pytest.raises(ValueError, match=message):
        normalize_nemotron_word_timestamps(entries)


def test_normalizer_accepts_entries_without_a_keep_field():
    words = normalize_nemotron_word_timestamps(
        (
            {"text": "▁first", "start_s": 0.0, "end_s": 0.2},
            {"text": "▁second", "start_s": 0.3, "end_s": 0.5},
        )
    )

    assert [(word.text, word.start_s, word.end_s) for word in words] == [
        ("first ", 0.0, 0.2),
        ("second", 0.3, 0.5),
    ]


def test_normalizer_rejects_overlapping_canonical_words_at_chunk_boundary():
    with pytest.raises(ValueError, match="monotonic"):
        normalize_nemotron_word_timestamps(
            (
                {"text": "▁first", "start_s": 0.0, "end_s": 0.8},
                {"text": "▁second", "start_s": 0.7, "end_s": 1.0},
            )
        )


def test_normalizer_keeps_raw_end_beyond_duration_for_later_clamping():
    words = normalize_nemotron_word_timestamps(
        (
            {"text": "▁first", "start_s": 0.0, "end_s": 0.4},
            {"text": "▁trailing", "start_s": 0.9, "end_s": 1.031},
        )
    )

    assert [(word.text, word.start_s, word.end_s) for word in words] == [
        ("first ", 0.0, 0.4),
        ("trailing", 0.9, 1.031),
    ]
    assert words[-1].end_s > 1.0
