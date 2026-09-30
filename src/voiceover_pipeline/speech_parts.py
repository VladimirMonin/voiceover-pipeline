"""Strict parser for the ``speech-parts`` script format and its char budget.

A ``speech-parts`` document is the one explicit multi-part input the CLI accepts:
an ordered list of parts, each with exactly one voice and exactly one spoken
``text``, plus an optional document-level ``vibe`` and an optional per-part
``vibe``. The two vibes are kept separate from the spoken text: the effective
instruction is the document vibe, then a blank line, then the part vibe, and the
spoken text is only ever ``text``.

The module is deliberately self-contained. It reads no file, provider, network,
or history, and it is the single owner of the format's version, its safe parser,
its validation rules, and its request-character budget. The CLI, the preparation
service, and the native snapshot all consume :class:`SpeechPartsDocument` rather
than re-parsing YAML.

Because the application must stay dependency-free for this format, the loader is
a bounded subset of YAML, not a general one: top-level scalars and a ``parts``
block sequence of mappings whose values are scalars or literal/folded block
scalars. It is deliberately strict -- an unknown key, a duplicate key, a nested
mapping, or a tab-indented line is refused rather than silently ignored -- so a
typo cannot drop a voice, a vibe, or a part. ``utf-8-sig`` input is accepted, so a
leading byte-order mark never becomes part of the first key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPEECH_PARTS_FORMAT = "speech-parts"
SPEECH_PARTS_VERSION = 1
# Conservative application compatibility budget for one request, taken from the
# published Polza ``input`` limit. It is deliberately not presented as a proven
# Gemini limit; the exact provider contract stays unverified, and this number is
# the app-side policy the pre-submit gate enforces for every part.
SPEECH_PARTS_REQUEST_CHAR_LIMIT = 5000
# The two characters that join the document-level and the per-part vibe.
_VIBE_SEPARATOR = "\n\n"
_ALLOWED_TOP_LEVEL_KEYS = frozenset({"version", "format", "vibe", "parts"})
_ALLOWED_PART_KEYS = frozenset({"voice", "text", "vibe"})
_BLOCK_SCALAR_MARKERS = frozenset({"|", "|-", "|+", ">", ">-", ">+"})
# Plain scalars are kept as text unless they are a plain integer or boolean, so
# ``version: 1`` is an integer while a numeric-looking voice or text stays text.
_INTEGER_PATTERN = re.compile(r"-?[0-9]+")


class SpeechPartsError(ValueError):
    """A ``speech-parts`` document or budget violation the CLI reports as a usage error."""


@dataclass(frozen=True)
class SpeechPart:
    """One validated part: its 1-based position, one voice, text, and two vibes.

    ``vibe`` is the part-level instruction exactly as authored; ``effective_vibe``
    is the composed instruction the adapter would send. ``text`` is the exact
    spoken text from the document, never trimmed or composed with an instruction.
    """

    position: int
    voice: str
    text: str
    vibe: str | None
    effective_vibe: str


@dataclass(frozen=True)
class SpeechPartsDocument:
    """One validated ``speech-parts`` document: the document vibe and its parts."""

    global_vibe: str | None
    parts: tuple[SpeechPart, ...]


def compose_effective_vibe(global_vibe: str | None, part_vibe: str | None) -> str:
    """Compose the effective instruction from the two authored vibes.

    Missing values are skipped, and both present are joined by a blank line, so
    the document-level direction comes first and the part-level direction second.
    Neither the spoken text nor any placeholder is ever mixed in here.
    """
    global_text = "" if global_vibe is None else global_vibe
    part_text = "" if part_vibe is None else part_vibe
    if global_text and part_text:
        return f"{global_text}{_VIBE_SEPARATOR}{part_text}"
    return global_text or part_text


def speech_part_request_chars(text: str, effective_vibe: str, required_text_wrapper: str) -> int:
    """Return the exact pre-submit character count for one part's request.

    The count is the spoken text plus the effective instruction plus the exact
    service text the adapter must add to carry that instruction, so a wrapper that
    surrounds the transcript is charged to the same budget. Nothing is stripped or
    normalized here: the caller passes the exact strings the adapter would send.
    """
    return len(text) + len(effective_vibe) + len(required_text_wrapper)


def build_speech_parts_document(raw: str) -> SpeechPartsDocument:
    """Parse and validate one ``speech-parts`` document, failing closed.

    The input is treated as UTF-8 text with an optional BOM. Every structural
    violation raises :class:`SpeechPartsError`; no partial document is returned.
    """
    if not isinstance(raw, str):
        raise SpeechPartsError("speech-parts document must be text")
    root, parts_raw = _parse_document(raw.lstrip("\ufeff"))
    unknown = sorted(set(root) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise SpeechPartsError(
            f"speech-parts document has unsupported top-level keys: {', '.join(unknown)}"
        )
    version = root.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SPEECH_PARTS_VERSION:
        raise SpeechPartsError(f"speech-parts version must be {SPEECH_PARTS_VERSION}")
    if root.get("format") != SPEECH_PARTS_FORMAT:
        raise SpeechPartsError(f"speech-parts format must be {SPEECH_PARTS_FORMAT}")
    global_vibe = root.get("vibe")
    if global_vibe is None or isinstance(global_vibe, str):
        normalized_global = global_vibe
    else:
        raise SpeechPartsError("speech-parts vibe must be a string")
    if parts_raw is None or not parts_raw:
        raise SpeechPartsError("speech-parts requires a non-empty parts list")
    parts: list[SpeechPart] = []
    for index, entry in enumerate(parts_raw, start=1):
        parts.append(_build_part(index, entry, normalized_global))
    return SpeechPartsDocument(global_vibe=normalized_global, parts=tuple(parts))


def _build_part(index: int, entry: object, global_vibe: str | None) -> SpeechPart:
    if not isinstance(entry, dict):
        raise SpeechPartsError(f"speech-parts part {index} must be a mapping")
    unknown = sorted(set(entry) - _ALLOWED_PART_KEYS)
    if unknown:
        raise SpeechPartsError(
            f"speech-parts part {index} has unsupported keys: {', '.join(unknown)}"
        )
    voice = entry.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        raise SpeechPartsError(f"speech-parts part {index} requires a non-empty voice")
    text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SpeechPartsError(f"speech-parts part {index} requires a non-empty text")
    part_vibe = entry.get("vibe")
    if part_vibe is not None and not isinstance(part_vibe, str):
        raise SpeechPartsError(f"speech-parts part {index} vibe must be a string")
    return SpeechPart(
        position=index,
        voice=voice,
        text=text,
        vibe=part_vibe,
        effective_vibe=compose_effective_vibe(global_vibe, part_vibe),
    )


# ── bounded YAML-subset loader ────────────────────────────────────────────────


def _parse_document(text: str) -> tuple[dict[str, object], list[object] | None]:
    lines = text.splitlines()
    # YAML indentation cannot contain tabs, including after leading spaces. A
    # per-block first-character check misses `  \t- voice: ...` because strip()
    # later removes the tab and accidentally admits a different structure.
    for raw in lines:
        indentation = raw[: len(raw) - len(raw.lstrip(" \t"))]
        if "\t" in indentation:
            raise SpeechPartsError("speech-parts must not use tab indentation")
    root: dict[str, object] = {}
    parts: list[object] | None = None
    index = 0
    while index < len(lines):
        raw = lines[index]
        if _is_skippable(raw):
            index += 1
            continue
        if raw[:1] in (" ", "\t"):
            raise SpeechPartsError("speech-parts top-level keys must not be indented")
        key, rest = _split_key(raw, "top-level")
        if key == "parts":
            if rest:
                raise SpeechPartsError("speech-parts parts must be a block sequence")
            entries, index = _parse_parts(lines, index + 1)
            parts = entries
            _assign(root, key, entries)
            continue
        value, index = _parse_value(lines, index, rest)
        _assign(root, key, value)
    return root, parts


def _parse_parts(lines: list[str], start: int) -> tuple[list[object], int]:
    entries: list[object] = []
    index = start
    while index < len(lines):
        raw = lines[index]
        if _is_skippable(raw):
            index += 1
            continue
        if raw[:1] == "\t":
            raise SpeechPartsError("speech-parts must not use tab indentation")
        indent = _indent_of(raw)
        if indent == 0:
            break
        body = raw.strip()
        if body == "-":
            entry, index = _parse_mapping(lines, index + 1, indent + 2, "part")
        elif body.startswith("- "):
            entry = {}
            key, rest = _split_key(body[2:], "part")
            value, index = _parse_value_from(lines, index, rest, indent)
            _assign(entry, key, value)
            continuation, index = _parse_mapping(lines, index, indent + 2, "part")
            for extra_key, extra_value in continuation.items():
                _assign(entry, extra_key, extra_value)
        else:
            raise SpeechPartsError("speech-parts parts entries must be '- ' sequence items")
        entries.append(entry)
    return entries, index


def _parse_mapping(
    lines: list[str], start: int, base_indent: int, context: str
) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    index = start
    while index < len(lines):
        raw = lines[index]
        if _is_skippable(raw):
            index += 1
            continue
        if raw[:1] == "\t":
            raise SpeechPartsError(f"speech-parts {context} must not use tab indentation")
        indent = _indent_of(raw)
        if indent < base_indent:
            break
        if indent > base_indent:
            raise SpeechPartsError(f"speech-parts {context} has unexpected indentation")
        key, rest = _split_key(raw, context)
        value, index = _parse_value_from(lines, index, rest, indent)
        _assign(result, key, value)
    return result, index


def _parse_value(lines: list[str], index: int, rest: str) -> tuple[object, int]:
    return _parse_value_from(lines, index, rest, 0)


def _parse_value_from(
    lines: list[str], index: int, rest: str, key_indent: int
) -> tuple[object, int]:
    if rest in _BLOCK_SCALAR_MARKERS:
        return _parse_block_scalar(lines, index + 1, rest, key_indent)
    if rest == "":
        raise SpeechPartsError("speech-parts nested blocks are not supported")
    return _parse_inline_scalar(rest), index + 1


def _parse_block_scalar(
    lines: list[str], start: int, marker: str, key_indent: int
) -> tuple[str, int]:
    content: list[str] = []
    index = start
    while index < len(lines):
        raw = lines[index]
        if raw.strip() == "":
            content.append("")
            index += 1
            continue
        if raw[:1] == "\t":
            raise SpeechPartsError("speech-parts block scalar must not use tab indentation")
        if _indent_of(raw) <= key_indent:
            break
        content.append(raw)
        index += 1
    while content and content[-1] == "":
        content.pop()
    text = _render_block_scalar(content, marker)
    return text, index


def _render_block_scalar(content: list[str], marker: str) -> str:
    style = marker[0]
    chomp = marker[1:] or ""
    nonempty = [line for line in content if line.strip()]
    if not nonempty:
        return ""
    dedent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
    stripped = [line[dedent:] if line.strip() else "" for line in content]
    if style == "|":
        rendered = "\n".join(stripped)
    else:
        rendered = _fold_lines(stripped)
    if chomp == "-":
        return rendered
    if chomp == "+":
        return rendered + "\n"
    return rendered + "\n" if content else rendered


def _fold_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line == "":
            paragraphs.append(" ".join(current))
            current = []
        else:
            current.append(line)
    paragraphs.append(" ".join(current))
    return "\n".join(paragraphs)


def _parse_inline_scalar(rest: str) -> object:
    value = rest.strip()
    if not value:
        raise SpeechPartsError("speech-parts empty scalar value is not supported")
    if value[0] == '"':
        return _parse_double_quoted(value)
    if value[0] == "'":
        return _parse_single_quoted(value)
    plain = _strip_plain_comment(value).rstrip()
    if _INTEGER_PATTERN.fullmatch(plain):
        return int(plain)
    if plain in ("true", "false"):
        return plain == "true"
    return plain


def _parse_double_quoted(value: str) -> str:
    if len(value) < 2 or value[-1] != '"':
        raise SpeechPartsError("speech-parts unterminated double-quoted scalar")
    body = value[1:-1]
    out: list[str] = []
    index = 0
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "0": "\0"}
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            following = body[index + 1]
            if following in escapes:
                out.append(escapes[following])
                index += 2
                continue
            raise SpeechPartsError("speech-parts unsupported escape in double-quoted scalar")
        out.append(char)
        index += 1
    return "".join(out)


def _parse_single_quoted(value: str) -> str:
    if len(value) < 2 or value[-1] != "'":
        raise SpeechPartsError("speech-parts unterminated single-quoted scalar")
    return value[1:-1].replace("''", "'")


def _strip_plain_comment(value: str) -> str:
    for position, char in enumerate(value):
        if char == "#" and (position == 0 or value[position - 1] == " "):
            return value[:position]
    return value


def _split_key(line: str, context: str) -> tuple[str, str]:
    body = _strip_plain_comment(line).rstrip()
    if ":" not in body:
        raise SpeechPartsError(f"speech-parts {context} line is not 'key: value'")
    key, _, rest = body.partition(":")
    key = key.strip()
    if not key:
        raise SpeechPartsError(f"speech-parts {context} key must not be empty")
    return key, rest.strip()


def _assign(mapping: dict[str, object], key: str, value: object) -> None:
    if key in mapping:
        raise SpeechPartsError(f"speech-parts duplicate key: {key}")
    mapping[key] = value


def _is_skippable(raw: str) -> bool:
    stripped = raw.strip()
    return not stripped or stripped.startswith("#")


def _indent_of(raw: str) -> int:
    return len(raw) - len(raw.lstrip(" "))
