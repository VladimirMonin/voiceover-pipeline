"""Runtime ``.env`` precedence contract for ``voiceover``.

These tests pin the secret-resolution order the S10 plan describes: the process
environment wins first; an explicit ``voiceover --env-file PATH <command>`` is
resolved next and never falls through to an unrelated working-directory ``.env``;
and without the flag ``<call-time CWD>/.env`` is read for compatibility without any
hidden parent search. They also pin the safety edges: a read-only ``help`` call
never reads an env file even when the flag is present, an unusable explicit file
fails closed with a redacted message, and the scoped override is reset on every
exit. Everything is offline and uses synthetic keys in temporary directories; the
real user ``.env`` is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from conftest import cli_json, run_cli

from voiceover_pipeline import cli, config

_SYNTHETIC_KEY = "VOICEOVER_TEST_ENV_KEY"
_REDACTED_MESSAGE = "Explicit --env-file is missing or not a regular file."


def _write_env(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── resolution order ─────────────────────────────────────────────────────────


def test_process_environment_value_wins_over_explicit_env_file(tmp_path, monkeypatch):
    explicit = _write_env(tmp_path / "explicit.env", f"{_SYNTHETIC_KEY}=from-file")
    monkeypatch.setenv(_SYNTHETIC_KEY, "from-process-env")

    with config.use_env_file(explicit):
        assert config.get_secret(_SYNTHETIC_KEY) == "from-process-env"


def test_process_key_does_not_bypass_invalid_explicit_file(tmp_path, monkeypatch):
    monkeypatch.setenv("POLZA_API_KEY", "synthetic-process-env-key")
    missing = tmp_path / "missing-explicit.env"

    with config.use_env_file(missing):
        with pytest.raises(config.EnvFileError) as excinfo:
            config.get_secret("POLZA_API_KEY")
        with pytest.raises(cli.CliError) as paid_preflight:
            cli.read_api_key(argparse.Namespace(provider="polza-chat-audio"))

    assert str(excinfo.value) == _REDACTED_MESSAGE
    assert paid_preflight.value.code == 20
    assert str(tmp_path) not in str(paid_preflight.value)


def test_explicit_env_file_is_read_when_environment_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    explicit = _write_env(tmp_path / "explicit.env", f"{_SYNTHETIC_KEY}=from-file")

    with config.use_env_file(explicit):
        assert config.get_secret(_SYNTHETIC_KEY) == "from-file"


def test_explicit_env_file_suppresses_an_unrelated_cwd_env(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_env(cwd / ".env", f"{_SYNTHETIC_KEY}=from-cwd")
    explicit = _write_env(tmp_path / "explicit.env", "SOME_OTHER_KEY=1")
    monkeypatch.chdir(cwd)

    with config.use_env_file(explicit):
        assert config.get_secret(_SYNTHETIC_KEY) is None


def test_runtime_lookup_uses_call_time_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_env(cwd / ".env", f"{_SYNTHETIC_KEY}=from-call-time-cwd")

    monkeypatch.chdir(cwd)
    assert config.get_secret(_SYNTHETIC_KEY) == "from-call-time-cwd"


def test_no_implicit_parent_directory_search(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    _write_env(tmp_path / ".env", f"{_SYNTHETIC_KEY}=from-parent")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)

    assert config.get_secret(_SYNTHETIC_KEY) is None


def test_compatibility_cwd_env_file_may_be_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    monkeypatch.chdir(tmp_path)

    assert config.get_secret(_SYNTHETIC_KEY) is None
    assert config.read_env_file() == {}


# ── direct-call compatibility ────────────────────────────────────────────────


def test_direct_call_explicit_path_reads_exactly_that_file(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    direct = _write_env(tmp_path / "direct.env", f"{_SYNTHETIC_KEY}=direct")
    other = _write_env(tmp_path / "other.env", f"{_SYNTHETIC_KEY}=other")

    assert config.read_env_file(direct) == {_SYNTHETIC_KEY: "direct"}
    assert config.get_secret(_SYNTHETIC_KEY, env_path=direct) == "direct"

    with config.use_env_file(other):
        # An explicit direct argument keeps winning over the scoped override.
        assert config.get_secret(_SYNTHETIC_KEY, env_path=direct) == "direct"


# ── fail-closed diagnostics ──────────────────────────────────────────────────


def test_missing_explicit_env_file_fails_closed_without_leaking_path(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    missing = tmp_path / "missing.env"

    with config.use_env_file(missing):
        with pytest.raises(config.EnvFileError) as excinfo:
            config.get_secret(_SYNTHETIC_KEY)
        with pytest.raises(config.EnvFileError):
            config.read_env_file()

    message = str(excinfo.value)
    assert message == _REDACTED_MESSAGE
    assert str(tmp_path) not in message
    assert _SYNTHETIC_KEY not in message


def test_directory_explicit_env_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    directory = tmp_path / "not-a-file.env"
    directory.mkdir()

    with config.use_env_file(directory):
        with pytest.raises(config.EnvFileError):
            config.get_secret(_SYNTHETIC_KEY)


def test_undecodable_explicit_env_file_fails_closed_without_content(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    undecodable = tmp_path / "undecodable.env"
    undecodable.write_bytes(b"\xff\xfe\x00synthetic-not-utf8")

    with config.use_env_file(undecodable):
        with pytest.raises(config.EnvFileError) as excinfo:
            config.get_secret(_SYNTHETIC_KEY)

    message = str(excinfo.value)
    assert message == "Explicit --env-file could not be read."
    assert str(tmp_path) not in message
    assert "synthetic-not-utf8" not in message


def test_explicit_file_probe_oserror_is_path_free(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    explicit = tmp_path / "synthetic-private-path.env"
    original_is_file = Path.is_file

    def failing_probe(path):
        if path == explicit:
            raise OSError("synthetic-private-path.env: permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", failing_probe)
    with config.use_env_file(explicit):
        with pytest.raises(config.EnvFileError) as excinfo:
            config.get_secret(_SYNTHETIC_KEY)

    assert "synthetic-private-path" not in str(excinfo.value)
    assert "permission denied" not in str(excinfo.value)


def test_default_file_probe_oserror_is_path_free(tmp_path, monkeypatch):
    monkeypatch.delenv(_SYNTHETIC_KEY, raising=False)
    monkeypatch.chdir(tmp_path)
    default = tmp_path / ".env"
    original_exists = Path.exists

    def failing_probe(path):
        if path == default:
            raise OSError("synthetic-private-path/.env: permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", failing_probe)
    with pytest.raises(config.EnvFileError) as excinfo:
        config.get_secret(_SYNTHETIC_KEY)

    assert "synthetic-private-path" not in str(excinfo.value)
    assert "permission denied" not in str(excinfo.value)


# ── scoped override lifecycle ────────────────────────────────────────────────


def test_context_manager_resets_scope_on_exit_and_error(tmp_path):
    explicit = _write_env(tmp_path / "explicit.env", f"{_SYNTHETIC_KEY}=from-file")

    with config.use_env_file(explicit):
        assert config.resolved_env_file_path() == explicit
    assert config.resolved_env_file_path() == Path.cwd() / ".env"

    with pytest.raises(ValueError):
        with config.use_env_file(explicit):
            raise ValueError("boom")
    assert config.resolved_env_file_path() == Path.cwd() / ".env"


# ── CLI surface ──────────────────────────────────────────────────────────────


def test_global_env_file_flag_works_before_subcommand_and_doctor_reports_path(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    explicit = _write_env(tmp_path / "explicit.env", "POLZA_API_KEY=synthetic-doctor-key")

    code, payload = cli_json(
        "--env-file",
        str(explicit),
        "doctor",
        "--provider",
        "polza-chat-audio",
        "--json",
        cwd=tmp_path,
    )

    assert code == 0
    assert payload["status"] == "success"
    checks = payload["checks"]
    assert checks["env_file"]["path"] == str(explicit)
    assert checks["env_file"]["ok"] is True
    assert checks["polza_key"]["ok"] is True
    assert "synthetic-doctor-key" not in json.dumps(payload)


def test_doctor_suppresses_an_unrelated_cwd_env(tmp_path, monkeypatch):
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_env(cwd / ".env", "POLZA_API_KEY=synthetic-cwd-secret")
    explicit = _write_env(tmp_path / "explicit.env", "OTHER_KEY=1")

    code, payload = cli_json(
        "--env-file",
        str(explicit),
        "doctor",
        "--provider",
        "polza-chat-audio",
        "--json",
        cwd=cwd,
    )

    assert code == 0
    assert payload["checks"]["polza_key"]["ok"] is False
    assert payload["checks"]["env_file"]["path"] == str(explicit)
    assert "synthetic-cwd-secret" not in json.dumps(payload)


def test_doctor_reports_a_missing_explicit_file_without_failing(tmp_path, monkeypatch):
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    missing = tmp_path / "missing.env"

    code, payload = cli_json(
        "--env-file",
        str(missing),
        "doctor",
        "--provider",
        "polza-chat-audio",
        "--json",
        cwd=tmp_path,
    )

    assert code == 0
    assert payload["checks"]["env_file"]["ok"] is False
    assert payload["checks"]["env_file"]["path"] == str(missing)
    assert payload["checks"]["polza_key"]["ok"] is False


def test_doctor_stat_error_does_not_reflect_os_error(monkeypatch, capsys, tmp_path):
    explicit = tmp_path / "synthetic-private-path.env"
    original_is_file = Path.is_file

    def failing_probe(path):
        if path == explicit:
            raise OSError("synthetic-private-path.env: permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", failing_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover", "--env-file", str(explicit), "doctor", "--json"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 0
    assert payload["status"] == "success"
    assert payload["checks"]["env_file"]["ok"] is False
    assert "permission denied" not in json.dumps(payload)


def test_process_environment_wins_in_doctor(tmp_path, monkeypatch):
    monkeypatch.setenv("POLZA_API_KEY", "synthetic-process-secret")
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_env(cwd / ".env", "POLZA_API_KEY=synthetic-cwd-secret")
    explicit = _write_env(tmp_path / "explicit.env", "POLZA_API_KEY=synthetic-file-secret")

    code, payload = cli_json(
        "--env-file",
        str(explicit),
        "doctor",
        "--provider",
        "polza-chat-audio",
        "--json",
        cwd=cwd,
    )

    assert code == 0
    assert payload["checks"]["polza_key"]["ok"] is True
    rendered = json.dumps(payload)
    assert "synthetic-process-secret" not in rendered
    assert "synthetic-cwd-secret" not in rendered
    assert "synthetic-file-secret" not in rendered


def test_help_does_not_read_env_file_even_when_flag_is_present(tmp_path, monkeypatch):
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    cwd = tmp_path / "work"
    cwd.mkdir()
    _write_env(cwd / ".env", "POLZA_API_KEY=synthetic-help-secret")
    missing = tmp_path / "does-not-exist.env"

    code, payload = cli_json(
        "--env-file",
        str(missing),
        "help",
        "start.quick",
        "--json",
        cwd=cwd,
    )

    assert code == 0
    assert payload["status"] == "success"
    assert "synthetic-help-secret" not in json.dumps(payload)


def test_argparse_help_still_works_with_env_file_flag(tmp_path):
    top_level = run_cli("--env-file", str(tmp_path / "x.env"), "--help")
    subcommand = run_cli("generate", "--help")

    assert top_level.returncode == 0
    assert "usage" in top_level.stdout.lower()
    assert subcommand.returncode == 0
    assert "usage" in subcommand.stdout.lower()


def test_read_api_key_fails_closed_with_redacted_message(tmp_path, monkeypatch):
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    args = argparse.Namespace(provider="polza-chat-audio")

    with config.use_env_file(tmp_path / "missing.env"):
        with pytest.raises(cli.CliError) as excinfo:
            cli.read_api_key(args)

    assert excinfo.value.code == 20
    assert str(excinfo.value) == _REDACTED_MESSAGE
    assert str(tmp_path) not in str(excinfo.value)


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("groq-whisper", "GROQ_API_KEY"), ("xai-stt", "X_AI_API_KEY")],
)
@pytest.mark.parametrize("process_key", [None, "synthetic-process-key"])
def test_cloud_timing_invalid_env_is_rejected_before_paid_reservation(
    tmp_path, monkeypatch, capsys, provider, key_name, process_key
):
    if process_key is None:
        monkeypatch.delenv(key_name, raising=False)
    else:
        monkeypatch.setenv(key_name, process_key)
    home = tmp_path / "home"
    output_root = tmp_path / "out" / "timing-synthetic"
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic-audio-bytes")
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))

    calls = []

    def fake_provider(**_kwargs):
        calls.append("provider")
        if provider == "groq-whisper":
            return config.read_groq_key()
        return config.read_xai_key()

    monkeypatch.setattr(cli, "_extract_timings", fake_provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover",
            "--env-file",
            str(tmp_path / "missing-explicit.env"),
            "timings",
            "--audio",
            str(audio),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-synthetic",
            "--timing-provider",
            provider,
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 20
    assert payload["code"] == 20
    assert payload["error"] == _REDACTED_MESSAGE
    assert "missing-explicit.env" not in payload["error"]
    assert calls == []
    assert not output_root.exists()
    assert not (home / "history.sqlite3").exists()


@pytest.mark.parametrize("provider", ["groq-whisper", "xai-stt"])
def test_cloud_timing_skip_existing_needs_no_key_or_env_file(
    tmp_path, monkeypatch, capsys, provider
):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("X_AI_API_KEY", raising=False)
    home = tmp_path / "home"
    output_root = tmp_path / "out" / "timing-synthetic"
    output_root.mkdir(parents=True)
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic-audio-bytes")
    monkeypatch.setenv("VOICEOVER_HOME", str(home))
    monkeypatch.setattr(cli, "check_media_tools", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        cli,
        "_extract_timings",
        lambda **_kwargs: pytest.fail("skip-existing must not call the provider"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceover",
            "--env-file",
            str(tmp_path / "missing-explicit.env"),
            "timings",
            "--audio",
            str(audio),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "timing-synthetic",
            "--timing-provider",
            provider,
            "--skip-existing",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 0
    assert payload["status"] == "skipped"
    assert not (home / "history.sqlite3").exists()


def test_main_maps_env_file_error_to_redacted_json(tmp_path, monkeypatch, capsys):
    def _exploding_generate(_args):
        raise config.EnvFileError(_REDACTED_MESSAGE)

    monkeypatch.setattr(cli, "generate", _exploding_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["voiceover", "--env-file", str(tmp_path / "missing.env"), "generate", "--json"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 20
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert payload["error"] == _REDACTED_MESSAGE
    assert str(tmp_path) not in out


def test_main_resets_env_file_scope_after_each_call(tmp_path, monkeypatch):
    explicit = _write_env(tmp_path / "explicit.env", f"{_SYNTHETIC_KEY}=1")

    monkeypatch.setattr(sys, "argv", ["voiceover", "--env-file", str(explicit), "help", "--json"])
    with pytest.raises(SystemExit):
        cli.main()
    assert config.resolved_env_file_path() == Path.cwd() / ".env"

    # A later call without the flag must not inherit the previous override.
    monkeypatch.setattr(sys, "argv", ["voiceover", "help", "--json"])
    with pytest.raises(SystemExit):
        cli.main()
    assert config.resolved_env_file_path() == Path.cwd() / ".env"
