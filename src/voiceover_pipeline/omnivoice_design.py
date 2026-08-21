"""Gate OmniVoice design instructions to the pinned native attribute vocabulary.

The vocabulary and exclusivity rules mirror the pinned audio.cpp OmniVoice
prompt builder (``prompt_builder.cpp`` ``resolve_instruct``): items are split
on ASCII or full-width commas, trimmed, ASCII-lowercased, and must be one of
the supported attribute tokens; at most one item per exclusivity group is
allowed; Chinese dialect tokens cannot be mixed with English accent tokens.
"""

import re
from typing import Final

OMNIVOICE_DESIGN_GENDERS: Final[frozenset[str]] = frozenset({"male", "female"})

OMNIVOICE_DESIGN_AGES: Final[frozenset[str]] = frozenset(
    {"child", "teenager", "young adult", "middle-aged", "elderly"}
)

OMNIVOICE_DESIGN_PITCHES: Final[frozenset[str]] = frozenset(
    {"very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"}
)

OMNIVOICE_DESIGN_WHISPER: Final[frozenset[str]] = frozenset({"whisper"})

OMNIVOICE_DESIGN_ACCENTS: Final[frozenset[str]] = frozenset(
    {
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
)

OMNIVOICE_DESIGN_DIALECTS: Final[frozenset[str]] = frozenset(
    {
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
)

OMNIVOICE_DESIGN_ALL_VOCAB: Final[frozenset[str]] = (
    OMNIVOICE_DESIGN_GENDERS
    | OMNIVOICE_DESIGN_AGES
    | OMNIVOICE_DESIGN_PITCHES
    | OMNIVOICE_DESIGN_WHISPER
    | OMNIVOICE_DESIGN_ACCENTS
    | OMNIVOICE_DESIGN_DIALECTS
)

_EXCLUSIVITY_GROUPS: Final[tuple[frozenset[str], ...]] = (
    OMNIVOICE_DESIGN_GENDERS,
    OMNIVOICE_DESIGN_AGES,
    OMNIVOICE_DESIGN_PITCHES,
    OMNIVOICE_DESIGN_WHISPER,
    OMNIVOICE_DESIGN_ACCENTS,
    OMNIVOICE_DESIGN_DIALECTS,
)

_DESIGN_ITEM_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"\s*[,，]\s*")


def normalize_omnivoice_design_instruction(instruction: str) -> str:
    """Split, trim, lowercase, and validate an OmniVoice design instruction.

    Returns the canonical lowercase comma-joined form. Raises ``ValueError``
    for empty input, items outside the pinned vocabulary, duplicate items,
    multiple items from one exclusivity group, or a Chinese dialect mixed with
    an English accent.
    """
    raw_items = [item.strip().lower() for item in _DESIGN_ITEM_SEPARATOR.split(instruction)]
    items = [item for item in raw_items if item]
    if not items:
        raise ValueError("OmniVoice design instruction must not be empty")

    unknown = [item for item in items if item not in OMNIVOICE_DESIGN_ALL_VOCAB]
    if unknown:
        raise ValueError(
            "unsupported OmniVoice design items: " + ", ".join(f"'{item}'" for item in unknown)
        )

    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise ValueError(
            "OmniVoice design instruction contains duplicate items: "
            + ", ".join(f"'{item}'" for item in duplicates)
        )

    for group in _EXCLUSIVITY_GROUPS:
        hits = [item for item in items if item in group]
        if len(hits) > 1:
            raise ValueError(
                "conflicting OmniVoice design items in the same category: "
                + " vs ".join(f"'{item}'" for item in hits)
            )

    has_dialect = any(item in OMNIVOICE_DESIGN_DIALECTS for item in items)
    has_accent = any(item in OMNIVOICE_DESIGN_ACCENTS for item in items)
    if has_dialect and has_accent:
        raise ValueError(
            "OmniVoice design instruction cannot mix Chinese dialect and English accent"
        )

    return ", ".join(items)
