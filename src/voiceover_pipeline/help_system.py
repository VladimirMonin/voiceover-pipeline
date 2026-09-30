"""Packaged atomic help topics read from ``voiceover_pipeline/resources/help``.

Plan section 9 keeps one canonical copy of the user documentation inside the
installed package, so an agent can ask one narrow question and read one short
Markdown topic instead of a large repository README. Every topic is an ordinary
``<topic>.md`` file next to the code, and this module is the only reader.

Boundaries this module owns:

* A topic name is a strict lowercase dotted identifier. Resolution goes through
  :func:`importlib.resources.files` and never through a caller-supplied path, so
  an arbitrary path, ``..``, a drive letter, or ``.env`` cannot be reached.
* Frontmatter is a bounded ``key: value`` block with exactly ``topic``, ``title``,
  ``summary``, ``related``. There is no YAML parser, no templating, no expression
  evaluation, and no new dependency.
* One registry is built from the packaged directory. A duplicate topic, a topic
  name that disagrees with its file stem, an unsupported frontmatter key, a
  malformed ``related`` link, a stray non-Markdown file, or a missing default
  topic fails closed instead of silently serving a partial topic list.
* Nothing here reads a secret, a history database, a provider, or the network, and
  importing this module pulls in no Torch, FFmpeg, key, or HTTP dependency.

The CLI owns printing and exit codes; :mod:`voiceover_pipeline.cli` maps
:class:`HelpError` to the machine envelope. Tests replace the
:func:`_help_directory` seam with a temporary directory to prove the validation
rules without shipping a broken package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

HELP_PACKAGE = "voiceover_pipeline"
HELP_RESOURCE_PARTS = ("resources", "help")
HELP_FILE_SUFFIX = ".md"
DEFAULT_HELP_TOPIC = "index"

# Usage errors keep the documented exit code 2; an unreadable or inconsistent
# packaged help directory is a runtime failure and keeps the generic code 30.
_EXIT_ARGS = 2
_EXIT_RUNTIME = 30

_FRONTMATTER_KEYS = ("topic", "title", "summary", "related")
_MAX_TOPIC_LENGTH = 64
_MAX_TITLE_LENGTH = 120
_MAX_SUMMARY_LENGTH = 300
_MAX_RELATED_LINKS = 32
_TOPIC_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)*")

_LIST_TOPICS_HINT = "Run 'voiceover help' to list the available topics."


class HelpError(RuntimeError):
    """A help request the CLI reports with a numeric code and a machine error code."""

    def __init__(self, message: str, *, code: int, error_code: str) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code
        self.details: dict[str, Any] = {"error_code": error_code}


class InvalidHelpTopicError(HelpError):
    """The requested topic name is not a strict lowercase dotted identifier."""

    def __init__(self) -> None:
        super().__init__(
            f"Invalid help topic name; expected a lowercase dotted name such as "
            f"'{DEFAULT_HELP_TOPIC}'. {_LIST_TOPICS_HINT}",
            code=_EXIT_ARGS,
            error_code="HELP_INVALID_TOPIC",
        )


class UnknownHelpTopicError(HelpError):
    """The topic name is well formed but no packaged topic carries it."""

    def __init__(self, topic: str) -> None:
        super().__init__(
            f"Unknown help topic '{topic}'. {_LIST_TOPICS_HINT}",
            code=_EXIT_ARGS,
            error_code="HELP_UNKNOWN_TOPIC",
        )


class HelpResourceError(HelpError):
    """The packaged help directory is missing, unreadable, or internally inconsistent."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"Packaged help is unusable: {message}",
            code=_EXIT_RUNTIME,
            error_code="HELP_RESOURCE_ERROR",
        )


@dataclass(frozen=True)
class HelpTopic:
    """One validated help topic: its bounded metadata and its Markdown body."""

    topic: str
    title: str
    summary: str
    related: tuple[str, ...]
    markdown: str


def is_valid_topic_name(value: object) -> bool:
    """Whether ``value`` is a strict lowercase dotted topic name."""
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_TOPIC_LENGTH:
        return False
    return _TOPIC_NAME_PATTERN.fullmatch(value) is not None


def _help_directory() -> Traversable:
    """Return the packaged help directory without reading any topic file."""
    return resources.files(HELP_PACKAGE).joinpath(*HELP_RESOURCE_PARTS)


def load_topic(name: str | None = None) -> HelpTopic:
    """Return one topic by name, or the default index topic when ``name`` is None."""
    candidate = DEFAULT_HELP_TOPIC if name is None else name
    if not is_valid_topic_name(candidate):
        raise InvalidHelpTopicError()
    registry = _load_registry()
    topic = registry.get(candidate)
    if topic is None:
        raise UnknownHelpTopicError(candidate)
    return topic


def topic_names() -> tuple[str, ...]:
    """Return every packaged topic name in a stable order."""
    return tuple(_load_registry())


def build_payload(topic: HelpTopic) -> dict[str, Any]:
    """Return the machine payload of one topic: metadata plus the same Markdown."""
    return {
        "topic": topic.topic,
        "title": topic.title,
        "summary": topic.summary,
        "related": list(topic.related),
        "markdown": topic.markdown,
    }


def render_raw(topic: HelpTopic) -> str:
    """Return the topic's Markdown exactly as packaged, without frontmatter or ANSI."""
    return topic.markdown


def render_human(topic: HelpTopic) -> str:
    """Return the topic with its title as the top-level heading for a terminal."""
    return f"# {topic.title}\n\n{topic.markdown}"


def _load_registry() -> dict[str, HelpTopic]:
    try:
        directory = _help_directory()
        if not directory.is_dir():
            raise HelpResourceError("the packaged resource directory is absent")
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise HelpResourceError("the packaged resource directory is unreadable") from exc
    registry: dict[str, HelpTopic] = {}
    for entry in entries:
        if not entry.name.endswith(HELP_FILE_SUFFIX):
            raise HelpResourceError(
                f"the packaged help directory holds an unexpected resource '{entry.name}'"
            )
        try:
            regular_file = entry.is_file()
        except OSError as exc:
            raise HelpResourceError("a packaged help resource is unreadable") from exc
        if not regular_file:
            raise HelpResourceError(
                f"the packaged help resource '{entry.name}' is not a regular file"
            )
        stem = entry.name[: -len(HELP_FILE_SUFFIX)]
        if not is_valid_topic_name(stem):
            raise HelpResourceError(f"the resource '{entry.name}' has an invalid topic name")
        if stem in registry:
            raise HelpResourceError(f"duplicate packaged help topic '{stem}'")
        registry[stem] = _parse_topic_document(stem, _read_resource_text(entry))
    if not registry:
        raise HelpResourceError("no help topic is packaged")
    _validate_registry(registry)
    return registry


def _read_resource_text(entry: Traversable) -> str:
    try:
        return entry.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HelpResourceError(f"the resource '{entry.name}' is unreadable") from exc


def _parse_topic_document(stem: str, text: str) -> HelpTopic:
    """Parse one bounded frontmatter document and its Markdown body."""
    lines = text.split("\n")
    if lines[0].strip() != "---":
        raise HelpResourceError(f"the topic '{stem}' does not start with frontmatter")
    fields: dict[str, str] = {}
    index = 1
    while index < len(lines) and lines[index].strip() != "---":
        line = lines[index]
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not key:
            raise HelpResourceError(f"the topic '{stem}' has a malformed frontmatter line")
        if key not in _FRONTMATTER_KEYS:
            raise HelpResourceError(f"the topic '{stem}' uses the unsupported key '{key}'")
        if key in fields:
            raise HelpResourceError(f"the topic '{stem}' repeats the frontmatter key '{key}'")
        fields[key] = value.strip()
        index += 1
    if index >= len(lines):
        raise HelpResourceError(f"the topic '{stem}' has unterminated frontmatter")

    missing = [key for key in _FRONTMATTER_KEYS if key not in fields]
    if missing:
        raise HelpResourceError(f"the topic '{stem}' is missing the frontmatter key '{missing[0]}'")
    topic = fields["topic"]
    if topic != stem:
        raise HelpResourceError(
            f"the topic '{stem}' declares the topic '{topic}' instead of its own file name"
        )
    title = _bounded_single_line(stem, "title", fields["title"], _MAX_TITLE_LENGTH)
    summary = _bounded_single_line(stem, "summary", fields["summary"], _MAX_SUMMARY_LENGTH)
    related = _parse_related(stem, fields["related"])
    markdown = "\n".join(lines[index + 1 :]).strip("\n")
    if not markdown.strip():
        raise HelpResourceError(f"the topic '{stem}' has an empty Markdown body")
    return HelpTopic(
        topic=topic,
        title=title,
        summary=summary,
        related=related,
        markdown=markdown + "\n",
    )


def _bounded_single_line(stem: str, key: str, value: str, limit: int) -> str:
    if not value:
        raise HelpResourceError(f"the topic '{stem}' has an empty '{key}'")
    if len(value) > limit:
        raise HelpResourceError(f"the topic '{stem}' has a '{key}' longer than {limit} characters")
    if any(character < " " or character == "\x7f" for character in value):
        raise HelpResourceError(f"the topic '{stem}' has a control character in '{key}'")
    return value


def _parse_related(stem: str, value: str) -> tuple[str, ...]:
    """Parse and validate the bounded comma-separated ``related`` topic list."""
    links: list[str] = []
    for raw in value.split(","):
        link = raw.strip()
        if not link:
            continue
        if not is_valid_topic_name(link):
            raise HelpResourceError(f"the topic '{stem}' has an invalid related name '{link}'")
        if link == stem:
            raise HelpResourceError(f"the topic '{stem}' links to itself")
        if link in links:
            raise HelpResourceError(f"the topic '{stem}' repeats the related name '{link}'")
        links.append(link)
    if len(links) > _MAX_RELATED_LINKS:
        raise HelpResourceError(f"the topic '{stem}' has more than {_MAX_RELATED_LINKS} links")
    return tuple(links)


def _validate_registry(registry: dict[str, HelpTopic]) -> None:
    """Prove every related link resolves and the default index covers every topic.

    The index topic is the default answer to ``voiceover help``, so it must link
    every other packaged topic. Without that check a newly added topic would stay
    invisible to an agent that starts from the index.
    """
    if DEFAULT_HELP_TOPIC not in registry:
        raise HelpResourceError(f"the default topic '{DEFAULT_HELP_TOPIC}' is not packaged")
    for name, topic in registry.items():
        for link in topic.related:
            if link not in registry:
                raise HelpResourceError(f"the topic '{name}' links to the missing topic '{link}'")
    index_topic = registry[DEFAULT_HELP_TOPIC]
    expected = set(registry) - {DEFAULT_HELP_TOPIC}
    if set(index_topic.related) != expected:
        raise HelpResourceError(
            f"the topic '{DEFAULT_HELP_TOPIC}' must link every other packaged topic"
        )
