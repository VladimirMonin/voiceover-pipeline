"""Gate OmniVoice design instructions to the pinned native attribute vocabulary.

The vocabulary and exclusivity rules mirror the pinned audio.cpp OmniVoice
prompt builder (``prompt_builder.cpp`` ``resolve_instruct``): items are split
on ASCII or full-width commas, trimmed, ASCII-lowercased, and must be one of
the supported attribute tokens; at most one item per exclusivity group is
allowed; Chinese dialect tokens cannot be mixed with English accent tokens.
"""

import re
from dataclasses import dataclass
from typing import Final, Literal

OMNIVOICE_LONG_FORM_THRESHOLD_SECONDS: Final = 30.0
_OMNIVOICE_ESTIMATED_WORDS_PER_SECOND: Final = 2.5

_OMNIVOICE_DESIGN_ALTERNATIVES: Final[tuple[dict[str, object], ...]] = (
    {
        "id": "omnivoice-clone",
        "provider": "omnivoice-local",
        "mode": "clone",
        "requires": "Russian reference audio and its transcript",
        "experimental": False,
    },
    {
        "id": "omnivoice-preset",
        "provider": "omnivoice-local",
        "mode": "preset",
        "requires": "an available accepted voice-bank profile",
        "experimental": False,
    },
    {
        "id": "short-design-clips",
        "provider": "omnivoice-local",
        "mode": "design",
        "requires": "separate acceptance of every clip at or below thirty estimated seconds",
        "experimental": True,
    },
    {
        "id": "other-tts-provider",
        "provider": None,
        "mode": None,
        "requires": "explicit selection of another TTS provider",
        "experimental": False,
    },
)


@dataclass(frozen=True)
class OmniVoiceDesignRoutePolicy:
    """Pre-runtime truth for one OmniVoice mode/language/text combination."""

    status: Literal["allowed", "experimental", "rejected"]
    language: str
    mode: str
    estimated_duration_seconds: float
    warning: str | None = None
    alternatives: tuple[dict[str, object], ...] = ()

    def error_details(self) -> dict[str, object]:
        """Return the stable machine contract for a rejected design request."""
        if self.status != "rejected":
            raise ValueError("OmniVoice design error details require a rejected policy")
        return {
            "error_code": "OMNIVOICE_DESIGN_UNSUPPORTED_LONG_LANGUAGE",
            "provider": "omnivoice-local",
            "mode": self.mode,
            "language": self.language,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "threshold_seconds": OMNIVOICE_LONG_FORM_THRESHOLD_SECONDS,
            "alternatives": [dict(item) for item in self.alternatives],
            "automatic_fallback": False,
        }


def estimate_omnivoice_duration_seconds(text: str) -> float:
    """Estimate speech duration offline before model or GPU admission.

    Upstream's learned duration estimator is part of the admitted model. VOP
    therefore uses a deterministic conservative speech-rate estimate at the
    pre-admission boundary and applies upstream's thirty-second long-form
    threshold to that estimate.
    """
    word_count = len(re.findall(r"\S+", text))
    return word_count / _OMNIVOICE_ESTIMATED_WORDS_PER_SECOND


def evaluate_omnivoice_design_route(
    text: str, *, language: str, mode: str
) -> OmniVoiceDesignRoutePolicy:
    """Classify design support without constructing a provider or runtime."""
    normalized_language = language.strip().lower().replace("_", "-")
    estimated_duration = estimate_omnivoice_duration_seconds(text)
    if mode != "design" or normalized_language.startswith(("en", "zh")):
        return OmniVoiceDesignRoutePolicy(
            status="allowed",
            language=normalized_language,
            mode=mode,
            estimated_duration_seconds=estimated_duration,
        )

    warning = (
        "OmniVoice Voice Design is trained only for Chinese and English; "
        f"{normalized_language or 'this language'} design is experimental and requires "
        "separate acceptance."
    )
    status: Literal["experimental", "rejected"] = (
        "rejected" if estimated_duration > OMNIVOICE_LONG_FORM_THRESHOLD_SECONDS else "experimental"
    )
    return OmniVoiceDesignRoutePolicy(
        status=status,
        language=normalized_language,
        mode=mode,
        estimated_duration_seconds=estimated_duration,
        warning=warning,
        alternatives=_OMNIVOICE_DESIGN_ALTERNATIVES,
    )


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


def normalize_omnivoice_design_instruction(instruction: str, *, language: str | None = None) -> str:
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

    normalized_language = (language or "").strip().lower().replace("_", "-")
    if has_accent and normalized_language and not normalized_language.startswith("en"):
        raise ValueError(
            "OmniVoice accent attributes are supported for English speech only; "
            f"language '{normalized_language}' must omit accent attributes"
        )
    if has_dialect and normalized_language and not normalized_language.startswith("zh"):
        raise ValueError(
            "OmniVoice dialect attributes are supported for Chinese speech only; "
            f"language '{normalized_language}' must omit dialect attributes"
        )

    return ", ".join(items)
