"""Unit contract tests for the packaged atomic help registry.

These prove the packaged topics themselves and the validation the reader applies
before serving anything: every topic name is unique, every frontmatter block is
bounded and complete, every ``related`` link resolves, the default index covers
every topic, and a stray non-Markdown resource such as ``.env`` can never be read.
The failure cases drive the reader against a temporary directory through the
``_help_directory`` seam, so no broken package has to be shipped to test them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from voiceover_pipeline import help_system

EXPECTED_TOPICS = {
    "index",
    "start.quick",
    "speech.simple",
    "speech.parts",
    "speech.legacy",
    "runs.resume",
    "asr.transcribe",
    "history.find",
    "history.costs",
    "search.lexical",
    "search.semantic",
    "cli.json",
    "providers.polza",
}


def _document(topic: str, *, related: str = "", body: str = "Тело темы.") -> str:
    return (
        "---\n"
        f"topic: {topic}\n"
        f"title: Тема {topic}\n"
        "summary: Краткое описание темы.\n"
        f"related: {related}\n"
        "---\n"
        f"\n{body}\n"
    )


def _point_at(monkeypatch, directory: Path) -> None:
    """Serve a temporary directory instead of the installed help resources."""
    monkeypatch.setattr(help_system, "_help_directory", lambda: directory)


# ── the packaged topics ───────────────────────────────────────────────────────


def test_packaged_topic_names_are_unique_and_complete():
    names = help_system.topic_names()

    assert set(names) == EXPECTED_TOPICS
    assert len(names) == len(set(names))
    assert names == tuple(sorted(names))


def test_index_links_every_other_topic():
    index = help_system.load_topic()

    assert index.topic == "index"
    assert set(index.related) == EXPECTED_TOPICS - {"index"}


@pytest.mark.parametrize("name", sorted(EXPECTED_TOPICS))
def test_every_topic_is_bounded_and_carries_no_ansi(name):
    topic = help_system.load_topic(name)

    assert topic.title and topic.summary
    assert topic.topic == name
    assert topic.markdown.endswith("\n")
    assert topic.markdown.strip()
    assert "\x1b" not in topic.markdown
    assert not topic.markdown.startswith("---")
    for link in topic.related:
        assert help_system.load_topic(link).topic == link


def test_payload_carries_metadata_and_the_same_markdown():
    topic = help_system.load_topic("search.semantic")

    payload = help_system.build_payload(topic)

    assert payload == {
        "topic": "search.semantic",
        "title": topic.title,
        "summary": topic.summary,
        "related": list(topic.related),
        "markdown": topic.markdown,
    }
    assert "DEFERRED" in payload["markdown"]


def test_render_human_adds_only_the_title_heading():
    topic = help_system.load_topic("index")

    assert help_system.render_raw(topic) == topic.markdown
    assert help_system.render_human(topic) == f"# {topic.title}\n\n{topic.markdown}"


def test_help_system_import_pulls_no_key_network_or_ml_module(tmp_path):
    script = (
        "import json, sys\n"
        "import voiceover_pipeline.help_system as help_system\n"
        "topic = help_system.load_topic('start.quick')\n"
        "heavy = sorted(\n"
        "    name for name in (\n"
        "        'torch', 'numpy', 'requests', 'openai', 'faster_whisper',\n"
        "        'soundfile', 'transformers', 'sqlite3', 'ctranslate2',\n"
        "    ) if name in sys.modules\n"
        ")\n"
        "print(json.dumps({'heavy': heavy, 'topic': topic.topic}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), capture_output=True, text=True
    )

    payload = json.loads(proc.stdout)
    assert payload == {"heavy": [], "topic": "start.quick"}


# ── topic names ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    ["index", "start.quick", "speech.parts", "providers.polza", "a1.b2"],
)
def test_valid_topic_names(value):
    assert help_system.is_valid_topic_name(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Index",
        ".env",
        ".env.local",
        "speech.",
        ".quick",
        "speech..quick",
        "speech/parts",
        "../index",
        "speech_ .parts",
        "speech parts",
        "speech\tparts",
        "speech\nparts",
        "speech-parts",
        "1index",
        "index ",
        "index/../../etc/passwd",
        "/etc/passwd",
        "a" * 65,
        None,
        7,
    ],
)
def test_invalid_topic_names(value):
    assert help_system.is_valid_topic_name(value) is False


def test_load_topic_rejects_invalid_and_unknown_names():
    with pytest.raises(help_system.InvalidHelpTopicError) as invalid:
        help_system.load_topic("../.env")

    assert invalid.value.code == 2
    assert invalid.value.error_code == "HELP_INVALID_TOPIC"
    assert ".env" not in str(invalid.value)

    with pytest.raises(help_system.UnknownHelpTopicError) as unknown:
        help_system.load_topic("speech.zzz")

    assert unknown.value.code == 2
    assert unknown.value.error_code == "HELP_UNKNOWN_TOPIC"
    assert "speech.zzz" in str(unknown.value)


# ── registry validation through the temporary-directory seam ──────────────────


def _write(directory: Path, files: dict[str, str], *, dirs: tuple[str, ...] = ()) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in dirs:
        (directory / name).mkdir()
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")


def test_temporary_registry_serves_one_linked_topic(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    _write(
        directory,
        {
            "index.md": _document("index", related="speech.simple"),
            "speech.simple.md": _document("speech.simple", related="index"),
        },
    )
    _point_at(monkeypatch, directory)

    assert help_system.topic_names() == ("index", "speech.simple")
    assert help_system.load_topic("index").related == ("speech.simple",)
    assert help_system.load_topic().topic == "index"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {
                "index.md": _document("index"),
                "start.quick.md": _document("start.quick"),
            },
            "must link every other packaged topic",
        ),
        (
            {
                "index.md": _document("index", related="speech.simple"),
                "start.quick.md": _document("start.quick"),
            },
            "links to the missing topic 'speech.simple'",
        ),
        (
            {"index.md": _document("index", related="index")},
            "links to itself",
        ),
        (
            {
                "index.md": (
                    "---\ntopic: index\ntitle: Тема index\nsummary: Краткое описание темы.\n"
                    "title: Второй заголовок\nrelated:\n---\n\nТело.\n"
                )
            },
            "repeats the frontmatter key 'title'",
        ),
        (
            {
                "index.md": (
                    "---\ntopic: index\ntitle: Тема\nsummary: Краткое описание темы.\n"
                    "provider: polza-tts\nrelated:\n---\n\nТело.\n"
                )
            },
            "unsupported key 'provider'",
        ),
        (
            {"index.md": ("---\ntopic: index\ntitle: Тема\nrelated:\n---\n\nТело.\n")},
            "missing the frontmatter key 'summary'",
        ),
        (
            {"index.md": "topic: index\ntitle: Тема\nsummary: Краткое.\nrelated:\n"},
            "does not start with frontmatter",
        ),
        (
            {
                "index.md": (
                    "---\ntopic: index\ntitle: Тема\nsummary: Краткое описание темы.\n"
                    "related:\n\nТело.\n"
                )
            },
            "malformed frontmatter line",
        ),
        (
            {
                "index.md": (
                    "---\ntopic: index\ntitle: Тема\nsummary: Краткое описание темы.\nrelated:"
                )
            },
            "unterminated frontmatter",
        ),
        (
            {"index.md": _document("other")},
            "declares the topic 'other'",
        ),
        (
            {"index.md": _document("index", body="")},
            "has an empty Markdown body",
        ),
        (
            {"index.md": _document("index", related="speech.Simple")},
            "invalid related name 'speech.Simple'",
        ),
        (
            {"Index.md": _document("index")},
            "invalid topic name",
        ),
        (
            {"index.md": _document("index", related="index, index")},
            "links to itself",
        ),
        (
            {"index.markdown": _document("index")},
            "unexpected resource 'index.markdown'",
        ),
    ],
)
def test_registry_rejects_inconsistent_packaged_help(monkeypatch, tmp_path, files, expected):
    directory = tmp_path / "help"
    _write(directory, files)
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert expected in str(excinfo.value)
    assert excinfo.value.code == 30
    assert excinfo.value.error_code == "HELP_RESOURCE_ERROR"


def test_registry_requires_the_default_topic(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    _write(directory, {"start.quick.md": _document("start.quick")})
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert "default topic 'index' is not packaged" in str(excinfo.value)


def test_registry_refuses_duplicate_packaged_entries(monkeypatch):
    class DuplicateFile:
        name = "index.md"

        def is_file(self):
            return True

        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return _document("index")

    class DuplicateDirectory:
        def is_dir(self):
            return True

        def iterdir(self):
            return iter((DuplicateFile(), DuplicateFile()))

    monkeypatch.setattr(help_system, "_help_directory", DuplicateDirectory)

    with pytest.raises(help_system.HelpResourceError, match="duplicate packaged help topic"):
        help_system.load_topic()


def test_registry_refuses_an_empty_directory(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    _write(directory, {})
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert "no help topic is packaged" in str(excinfo.value)


@pytest.mark.parametrize("failure", ["directory_probe", "iteration", "resource_probe"])
def test_resource_io_errors_do_not_reflect_private_paths(monkeypatch, failure):
    class BrokenFile:
        name = "index.md"

        def is_file(self):
            raise OSError("synthetic-private-path/.env")

    class BrokenDirectory:
        def is_dir(self):
            if failure == "directory_probe":
                raise OSError("synthetic-private-path/.env")
            return True

        def iterdir(self):
            if failure == "iteration":
                raise OSError("synthetic-private-path/.env")
            return iter([BrokenFile()])

    monkeypatch.setattr(help_system, "_help_directory", BrokenDirectory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert excinfo.value.code == 30
    assert excinfo.value.error_code == "HELP_RESOURCE_ERROR"
    assert "synthetic-private-path" not in str(excinfo.value)
    assert ".env" not in str(excinfo.value)


def test_registry_refuses_a_subdirectory_and_an_absent_directory(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    _write(directory, {"index.md": _document("index")}, dirs=("extra.md",))
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()
    assert "is not a regular file" in str(excinfo.value)

    _point_at(monkeypatch, tmp_path / "absent")
    with pytest.raises(help_system.HelpResourceError) as absent:
        help_system.load_topic()
    assert "resource directory is absent" in str(absent.value)


def test_registry_refuses_a_bogus_env_instead_of_reading_it(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    _write(
        directory,
        {
            "index.md": _document("index"),
            ".env": "POLZA_API_KEY=synthetic-do-not-read\n",
            ".env.example": "POLZA_API_KEY=\n",
        },
    )
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert "synthetic-do-not-read" not in str(excinfo.value)
    assert ".env" in str(excinfo.value)


def test_registry_refuses_an_over_long_title(monkeypatch, tmp_path):
    directory = tmp_path / "help"
    long_title = "Т" * (help_system._MAX_TITLE_LENGTH + 1)
    document = (
        f"---\ntopic: index\ntitle: {long_title}\nsummary: Краткое описание темы.\n"
        "related:\n---\n\nТело.\n"
    )
    _write(directory, {"index.md": document})
    _point_at(monkeypatch, directory)

    with pytest.raises(help_system.HelpResourceError) as excinfo:
        help_system.load_topic()

    assert "longer than" in str(excinfo.value)
