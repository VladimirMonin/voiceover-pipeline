"""CLI contract tests for ``voiceover help`` and its packaged topics.

These exercise the machine-facing envelope of the S10 packaged help through the
real CLI entry point: the exact JSON payload, verbatim ``--raw`` Markdown without
ANSI, the default topic, the sanitized usage errors for an unknown or unsafe topic
name, the fact that a bogus ``.env`` and an unrelated working directory are never
needed, and that the ordinary argparse ``--help`` and the existing commands still
work. Everything here stays offline and creates no history.
"""

from __future__ import annotations

import json
import sys

import pytest
import requests
from conftest import cli_json, run_cli

from voiceover_pipeline import cli, config, help_system

EXPECTED_JSON_KEYS = {"status", "topic", "title", "summary", "related", "markdown"}


def _raw(topic: str) -> str:
    return help_system.render_raw(help_system.load_topic(topic))


def _human(topic: str) -> str:
    return help_system.render_human(help_system.load_topic(topic))


def test_help_json_is_one_object_with_exact_payload():
    returncode, payload = cli_json("help", "speech.parts", "--json")
    proc = run_cli("help", "speech.parts", "--json")

    assert returncode == 0
    assert proc.stdout.strip().count("\n") == 0
    assert set(payload) == EXPECTED_JSON_KEYS
    assert payload["status"] == "success"
    assert payload["topic"] == "speech.parts"
    assert payload["title"] and payload["summary"]
    assert payload["related"] == [
        "speech.simple",
        "speech.legacy",
        "runs.resume",
        "providers.polza",
    ]
    assert payload["markdown"] == _raw("speech.parts")


def test_help_json_defaults_to_the_index_topic():
    returncode, payload = cli_json("help", "--json")

    assert returncode == 0
    assert payload["topic"] == "index"
    assert payload["markdown"] == _raw("index")


def test_help_raw_prints_packaged_markdown_without_ansi():
    proc = run_cli("help", "speech.parts", "--raw")

    assert proc.returncode == 0
    assert proc.stdout == _raw("speech.parts")
    assert proc.stderr == ""
    assert "\x1b" not in proc.stdout
    assert "\x1b" not in _raw("speech.parts")
    assert not proc.stdout.startswith("---")
    assert "topic: speech.parts" not in proc.stdout


def test_help_human_output_adds_only_the_title_heading():
    proc = run_cli("help", "runs.resume")

    assert proc.returncode == 0
    assert proc.stdout == _human("runs.resume")
    assert proc.stdout.startswith("# ")
    assert "\x1b" not in proc.stdout


def test_help_output_does_not_depend_on_the_working_directory(tmp_path):
    outside = run_cli("help", "start.quick", "--raw", cwd=tmp_path)

    assert outside.returncode == 0
    assert outside.stdout == run_cli("help", "start.quick", "--raw").stdout
    assert outside.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_help_never_reads_a_bogus_env_in_the_working_directory(tmp_path):
    (tmp_path / ".env").write_text(
        "POLZA_API_KEY=synthetic-help-must-not-read\n"
        "OPENROUTER_API_KEY=synthetic-help-must-not-read\n",
        encoding="utf-8",
    )

    code, payload = cli_json("help", "providers.polza", "--json", cwd=tmp_path)

    assert code == 0
    assert payload["topic"] == "providers.polza"
    assert "synthetic-help-must-not-read" not in json.dumps(payload)
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]


def test_help_cli_makes_no_key_lookup_or_http_request(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("help must not access credentials or make a provider request")

    for name in ("read_polza_key", "read_openrouter_key", "read_groq_key", "read_xai_key"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(config, "read_env_file", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(sys, "argv", ["voiceover", "help", "speech.parts", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 0
    assert payload["status"] == "success"
    assert payload["topic"] == "speech.parts"


@pytest.mark.parametrize(
    ("topic", "error_code"),
    [
        ("speech.zzz", "HELP_UNKNOWN_TOPIC"),
        ("runs", "HELP_UNKNOWN_TOPIC"),
        ("../../etc/passwd", "HELP_INVALID_TOPIC"),
        (".env", "HELP_INVALID_TOPIC"),
        ("../.env", "HELP_INVALID_TOPIC"),
        ("/etc/passwd", "HELP_INVALID_TOPIC"),
        ("speech/parts", "HELP_INVALID_TOPIC"),
        ("Speech.Parts", "HELP_INVALID_TOPIC"),
        ("speech..parts", "HELP_INVALID_TOPIC"),
    ],
)
def test_unknown_or_unsafe_topic_is_a_usage_error(topic, error_code):
    returncode, payload = cli_json("help", topic, "--json")

    assert returncode == 2
    assert set(payload) == {"status", "error", "code", "details"}
    assert payload["status"] == "error"
    assert payload["code"] == 2
    assert payload["details"]["error_code"] == error_code
    assert "\x1b" not in payload["error"]
    assert ".env" not in payload["error"]
    assert "/etc" not in payload["error"]
    assert "passwd" not in payload["error"]


def test_unknown_topic_without_json_reports_on_stderr():
    proc = run_cli("help", "speech.zzz")

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "Unknown help topic 'speech.zzz'" in proc.stderr


def test_help_resource_probe_failure_is_sanitized(monkeypatch, capsys):
    class UnreadableDirectory:
        def is_dir(self):
            raise OSError("synthetic-private-path/.env: permission denied")

    monkeypatch.setattr(help_system, "_help_directory", UnreadableDirectory)
    monkeypatch.setattr(sys, "argv", ["voiceover", "help", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 30
    assert payload["status"] == "error"
    assert payload["code"] == 30
    assert payload["details"]["error_code"] == "HELP_RESOURCE_ERROR"
    assert "synthetic-private-path" not in json.dumps(payload)
    assert ".env" not in json.dumps(payload)


def test_help_rejects_combined_raw_and_json():
    proc = run_cli("help", "index", "--raw", "--json")

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload == {
        "status": "error",
        "error": "Invalid command-line arguments",
        "code": 2,
    }


def test_argparse_help_and_existing_commands_are_preserved():
    top = run_cli("--help")
    assert top.returncode == 0
    assert "help" in top.stdout

    assert run_cli("help", "--help").returncode == 0
    assert run_cli("search", "--help").returncode == 0
    assert run_cli("list", "providers").returncode == 0
    assert run_cli("history", "costs", "--json").returncode == 0
