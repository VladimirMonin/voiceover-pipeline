import pytest

from voiceover_pipeline.omnivoice_design import (
    OMNIVOICE_DESIGN_ACCENTS,
    OMNIVOICE_DESIGN_AGES,
    OMNIVOICE_DESIGN_ALL_VOCAB,
    OMNIVOICE_DESIGN_DIALECTS,
    OMNIVOICE_DESIGN_GENDERS,
    OMNIVOICE_DESIGN_PITCHES,
    OMNIVOICE_DESIGN_WHISPER,
    normalize_omnivoice_design_instruction,
)


def test_vocabulary_matches_pinned_native_attribute_set():
    assert OMNIVOICE_DESIGN_GENDERS == {"male", "female"}
    assert OMNIVOICE_DESIGN_AGES == {
        "child",
        "teenager",
        "young adult",
        "middle-aged",
        "elderly",
    }
    assert OMNIVOICE_DESIGN_PITCHES == {
        "very low pitch",
        "low pitch",
        "moderate pitch",
        "high pitch",
        "very high pitch",
    }
    assert OMNIVOICE_DESIGN_WHISPER == {"whisper"}
    assert OMNIVOICE_DESIGN_ACCENTS == {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    }
    assert OMNIVOICE_DESIGN_DIALECTS == {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    }
    assert OMNIVOICE_DESIGN_ALL_VOCAB == (
        OMNIVOICE_DESIGN_GENDERS
        | OMNIVOICE_DESIGN_AGES
        | OMNIVOICE_DESIGN_PITCHES
        | OMNIVOICE_DESIGN_WHISPER
        | OMNIVOICE_DESIGN_ACCENTS
        | OMNIVOICE_DESIGN_DIALECTS
    )


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("female", "female"),
        ("female, young adult, moderate pitch", "female, young adult, moderate pitch"),
        (" male, middle-aged, low pitch, whisper ", "male, middle-aged, low pitch, whisper"),
        ("Female, Young Adult", "female, young adult"),
        ("whisper", "whisper"),
        ("american accent", "american accent"),
        ("indian accent, high pitch", "indian accent, high pitch"),
        ("东北话", "东北话"),
        ("河南话", "河南话"),
        ("female，young adult", "female, young adult"),
        ("  male  ,  young adult  ,  high pitch  ", "male, young adult, high pitch"),
    ],
)
def test_normalize_accepts_supported_vocabulary(instruction, expected):
    assert normalize_omnivoice_design_instruction(instruction) == expected


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        "   ",
        " , ， ",
    ],
)
def test_normalize_rejects_empty_instruction(instruction):
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_omnivoice_design_instruction(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        "warm and clear",
        "dramatic",
        "warm",
        "female, warm and clear",
        "deep voice",
        "soft spoken",
    ],
)
def test_normalize_rejects_unsupported_items(instruction):
    with pytest.raises(ValueError, match="unsupported OmniVoice design items"):
        normalize_omnivoice_design_instruction(instruction)


def test_normalize_quotes_each_unknown_item():
    with pytest.raises(ValueError) as exc:
        normalize_omnivoice_design_instruction("warm, clear")
    assert "unsupported OmniVoice design items: 'warm', 'clear'" in str(exc.value)


@pytest.mark.parametrize(
    ("instruction", "message"),
    [
        ("female, female", "duplicate"),
        ("male, male", "duplicate"),
        ("female, female, young adult", "duplicate"),
    ],
)
def test_normalize_rejects_duplicate_items(instruction, message):
    with pytest.raises(ValueError, match=message):
        normalize_omnivoice_design_instruction(instruction)


@pytest.mark.parametrize(
    ("instruction", "pair"),
    [
        ("male, female", "'male' vs 'female'"),
        ("child, elderly", "'child' vs 'elderly'"),
        ("low pitch, high pitch", "'low pitch' vs 'high pitch'"),
        ("male, female, young adult", "'male' vs 'female'"),
        ("american accent, british accent", "'american accent' vs 'british accent'"),
        ("河南话, 陕西话", "'河南话' vs '陕西话'"),
    ],
)
def test_normalize_rejects_conflicting_items_in_one_category(instruction, pair):
    with pytest.raises(ValueError, match="conflicting OmniVoice design items in the same category"):
        normalize_omnivoice_design_instruction(instruction)
    with pytest.raises(ValueError) as exc:
        normalize_omnivoice_design_instruction(instruction)
    assert pair in str(exc.value)


def test_normalize_rejects_mixing_dialect_and_accent():
    with pytest.raises(ValueError, match="cannot mix Chinese dialect and English accent"):
        normalize_omnivoice_design_instruction("河南话, american accent")
    with pytest.raises(ValueError, match="cannot mix Chinese dialect and English accent"):
        normalize_omnivoice_design_instruction("american accent, 东北话")


def test_normalize_accepts_whisper_with_other_single_group_items():
    assert (
        normalize_omnivoice_design_instruction("whisper, female, young adult")
        == "whisper, female, young adult"
    )


def test_normalize_rejects_english_accent_attributes_for_russian_synthesis():
    with pytest.raises(ValueError, match="English speech only"):
        normalize_omnivoice_design_instruction(
            "female, middle-aged, low pitch, russian accent", language="ru"
        )


def test_normalize_accepts_accent_attributes_for_english_synthesis():
    assert (
        normalize_omnivoice_design_instruction(
            "female, middle-aged, low pitch, russian accent", language="en"
        )
        == "female, middle-aged, low pitch, russian accent"
    )


def test_normalize_rejects_chinese_dialect_attributes_outside_chinese_synthesis():
    with pytest.raises(ValueError, match="Chinese speech only"):
        normalize_omnivoice_design_instruction("female, 东北话", language="ru")
