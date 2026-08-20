from __future__ import annotations

import re

import pytest

from voiceover_pipeline.config import OMNIVOICE_LOCAL_MODEL_ID
from voiceover_pipeline.local_tts_text import (
    merge_omnivoice_session_fragments,
    prepare_local_tts_chunks,
)
from voiceover_pipeline.models import ScriptChunk


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def test_local_tts_chunks_are_sentence_bounded_and_preserve_all_tokens() -> None:
    source = ScriptChunk(
        number=7,
        id="chapter_07",
        text=(
            "Первое короткое предложение. Второе предложение немного длиннее и тоже закончено. "
            "Третье предложение сохраняет границу. Четвёртое завершает исходный раздел."
        ),
    )

    chunks = prepare_local_tts_chunks(
        [source], "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=90
    )

    assert len(chunks) == 2
    assert [chunk.number for chunk in chunks] == [1, 2]
    assert [chunk.id for chunk in chunks] == [
        "chapter_07_part_01",
        "chapter_07_part_02",
    ]
    assert all(len(chunk.text) <= 90 for chunk in chunks)
    assert _tokens(" ".join(chunk.text for chunk in chunks)) == _tokens(source.text)
    assert all(chunk.text[-1] in ".!?" for chunk in chunks)


def test_local_tts_rejects_raw_digits_in_spoken_text() -> None:
    with pytest.raises(ValueError, match="digits"):
        prepare_local_tts_chunks(
            [ScriptChunk(number=1, id="chunk_01", text="Версия 3.5 готова на 25 процентов.")],
            "omnivoice-local",
            OMNIVOICE_LOCAL_MODEL_ID,
        )


def test_cloud_tts_chunks_are_not_rewritten_or_digit_checked() -> None:
    source = ScriptChunk(number=4, id="cloud_04", text="Version 3.5 is ready.")

    chunks = prepare_local_tts_chunks([source], "openrouter-tts", max_chars=10)

    assert chunks == [source]


def test_only_matching_omnivoice_profile_rewrites_local_chunks() -> None:
    source = ScriptChunk(
        number=1,
        id="local_01",
        text="Первое предложение. Второе предложение. Третье предложение.",
    )

    chunks = prepare_local_tts_chunks(
        [source], "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=25
    )
    unknown_model_chunks = prepare_local_tts_chunks(
        [source], "omnivoice-local", "audio-cpp/other-model", max_chars=25
    )
    qwen_chunks = prepare_local_tts_chunks([source], "qwen-local", "Qwen/Qwen3-TTS-12Hz-1.7B")

    assert len(chunks) == 3
    assert unknown_model_chunks == [source]
    assert qwen_chunks == [source]


def test_omnivoice_profile_defaults_to_420_character_atoms() -> None:
    source = ScriptChunk(number=1, id="long_01", text=" ".join(["слово"] * 150))

    chunks = prepare_local_tts_chunks([source], "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID)

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 420 for chunk in chunks)
    assert _tokens(" ".join(chunk.text for chunk in chunks)) == _tokens(source.text)


def test_local_tts_long_sentence_splits_without_losing_or_reordering_words() -> None:
    source = ScriptChunk(
        number=1,
        id="chunk_01",
        text=(
            "Очень длинная фраза, которая сначала делится по смысловой запятой, "
            "а если этого недостаточно то использует безопасную границу между словами без потери текста."
        ),
    )

    chunks = prepare_local_tts_chunks(
        [source], "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=70
    )

    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 70 for chunk in chunks)
    assert _tokens(" ".join(chunk.text for chunk in chunks)) == _tokens(source.text)


def test_local_tts_chunk_ids_are_stable_across_repeated_preparation() -> None:
    sources = [
        ScriptChunk(number=1, id="chunk_01", text="Первое предложение. Второе предложение."),
        ScriptChunk(number=2, id="chunk_02", text="Третье предложение. Четвёртое предложение."),
    ]

    first = prepare_local_tts_chunks(
        sources, "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=25
    )
    second = prepare_local_tts_chunks(
        sources, "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=25
    )

    assert first == second
    assert len({chunk.id for chunk in first}) == len(first)


def test_omnivoice_session_merge_preserves_prepared_fragment_order_in_one_request() -> None:
    fragments = [
        ScriptChunk(number=1, id="chunk_01_part_01", text="Первое предложение."),
        ScriptChunk(number=2, id="chunk_01_part_02", text="Второе предложение."),
        ScriptChunk(number=3, id="chunk_02_part_01", text="Третье предложение."),
    ]

    merged = merge_omnivoice_session_fragments(fragments)

    assert merged == [
        ScriptChunk(
            number=1,
            id="chunk_01_omnivoice_session",
            text="Первое предложение. Второе предложение. Третье предложение.",
        )
    ]


def test_local_tts_does_not_orphan_a_short_intro_from_its_following_sentence() -> None:
    source = ScriptChunk(
        number=1,
        id="chunk_01",
        text=(
            "Первое длинное предложение занимает почти весь доступный атом речи целиком. "
            "Десятый пример. Версия три точка пять."
        ),
    )

    chunks = prepare_local_tts_chunks(
        [source], "omnivoice-local", OMNIVOICE_LOCAL_MODEL_ID, max_chars=90
    )

    assert len(chunks) == 2
    assert chunks[0].text.endswith("целиком.")
    assert chunks[1].text == "Десятый пример. Версия три точка пять."
