"""``[search] default_mode`` settings contract for ``voiceover search``.

The S10 slice lets the non-secret ``settings.toml`` choose the default search
mode while an explicit ``--mode`` always wins. These tests stay offline and cover
the settings reader, the CLI wiring, the deferred refusal of a configured
``semantic``/``hybrid`` mode without any database or model, and the fixed,
path-free ``SEARCH_SETTINGS_INVALID`` failure for a malformed or unsupported
setting. They never read ``.env``.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import cli_json, run_cli

from voiceover_pipeline import settings
from voiceover_pipeline.history.database import HistoryDatabase
from voiceover_pipeline.history.paths import ensure_history_home, history_database_path
from voiceover_pipeline.history.repository import TEXT_KIND_TTS_SCRIPT, HistoryRepository

SEARCH_SETTINGS_INVALID_MESSAGE = (
    "settings.toml is invalid; [search] default_mode must be lexical, semantic, or hybrid."
)


def _write_settings(cwd, body: str):
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "settings.toml").write_text(body, encoding="utf-8")
    return cwd


def _build_home(home):
    ensure_history_home(home)
    database_path = history_database_path(home)
    with HistoryDatabase(database_path) as database:
        database.migrate()
        repository = HistoryRepository(database)
        run = repository.create_run(
            operation="tts", run_root=str(home / "runs" / "a"), user_label="метка поиска"
        )
        repository.add_text_source(
            run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin="native_snapshot",
            content="Проверяемый сценарный текст",
        )
    return database_path


@pytest.fixture
def home(tmp_path, monkeypatch):
    resolved = tmp_path / "home"
    monkeypatch.setenv("VOICEOVER_HOME", str(resolved))
    _build_home(resolved)
    return resolved


# ─── settings reader ────────────────────────────────────────────────────────


def test_search_settings_default_to_lexical_without_a_file(tmp_path):
    assert settings.load_search_settings(tmp_path / "settings.toml").default_mode == "lexical"


@pytest.mark.parametrize("body", ["[history]\nenabled = true\n", "[search]\n"])
def test_search_settings_default_to_lexical_without_the_key(tmp_path, body):
    workdir = _write_settings(tmp_path / "cwd", body)

    assert settings.load_search_settings(workdir / "settings.toml").default_mode == "lexical"


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_search_settings_accept_every_documented_mode(tmp_path, mode):
    workdir = _write_settings(tmp_path / "cwd", f'[search]\ndefault_mode = "{mode}"\n')

    assert settings.load_search_settings(workdir / "settings.toml").default_mode == mode


@pytest.mark.parametrize(
    "body",
    [
        'search = "lexical"\n',
        "[search]\ndefault_mode = 3\n",
        '[search]\ndefault_mode = ""\n',
        '[search]\ndefault_mode = "   "\n',
        '[search]\ndefault_mode = "fuzzy"\n',
    ],
)
def test_search_settings_reject_a_malformed_section(tmp_path, body):
    workdir = _write_settings(tmp_path / "cwd", body)

    with pytest.raises(settings.SettingsError):
        settings.load_search_settings(workdir / "settings.toml")


def test_search_settings_reject_a_malformed_file(tmp_path):
    workdir = _write_settings(tmp_path / "cwd", "[search\ndefault_mode = lexical\n")

    with pytest.raises(settings.SettingsError):
        settings.load_search_settings(workdir / "settings.toml")


def test_search_settings_reader_loads_without_torch(tmp_path):
    workdir = _write_settings(tmp_path / "cwd", '[search]\ndefault_mode = "hybrid"\n')
    program = (
        "import sys\n"
        "from voiceover_pipeline import settings\n"
        "assert 'torch' not in sys.modules, 'settings reader imported torch'\n"
        "print(settings.load_search_settings(sys.argv[1]).default_mode)\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", program, str(workdir / "settings.toml")],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "hybrid"


# ─── CLI wiring ─────────────────────────────────────────────────────────────


def test_search_uses_the_configured_lexical_default(home, tmp_path):
    workdir = _write_settings(tmp_path / "cwd", '[search]\ndefault_mode = "lexical"\n')

    returncode, payload = cli_json("search", "сценарный", "--json", cwd=workdir)

    assert returncode == 0
    assert payload["mode"] == "lexical"
    assert payload["count"] == 1


def test_search_without_a_settings_file_keeps_lexical(home):
    returncode, payload = cli_json("search", "сценарный", "--json")

    assert returncode == 0
    assert payload["mode"] == "lexical"


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_search_configured_deferred_mode_refuses_without_touching_the_database(
    tmp_path, monkeypatch, mode
):
    resolved = tmp_path / "empty"
    monkeypatch.setenv("VOICEOVER_HOME", str(resolved))
    workdir = _write_settings(tmp_path / "cwd", f'[search]\ndefault_mode = "{mode}"\n')

    returncode, payload = cli_json("search", "сценарный", "--json", cwd=workdir)

    assert returncode == 2
    assert payload["details"]["error_code"] == "SEARCH_MODE_DEFERRED"
    assert not (resolved / "history.sqlite3").exists()


@pytest.mark.parametrize(
    "body",
    ['[search]\ndefault_mode = "fuzzy"\n', "[search\ndefault_mode = lexical\n", "search = 3\n"],
)
def test_search_invalid_settings_fail_with_a_fixed_path_free_code(tmp_path, monkeypatch, body):
    resolved = tmp_path / "empty"
    monkeypatch.setenv("VOICEOVER_HOME", str(resolved))
    workdir = _write_settings(tmp_path / "cwd", body)
    proc = run_cli("search", "сценарный", "--json", cwd=workdir)
    payload = json.loads(proc.stdout)

    assert proc.returncode == 2
    assert payload["details"]["error_code"] == "SEARCH_SETTINGS_INVALID"
    assert payload["error"] == SEARCH_SETTINGS_INVALID_MESSAGE
    assert str(workdir) not in proc.stdout
    assert "fuzzy" not in proc.stdout
    assert not (resolved / "history.sqlite3").exists()


def test_search_settings_stat_error_is_fixed_and_path_free(tmp_path, monkeypatch, capsys):
    from voiceover_pipeline import cli as cli_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOICEOVER_HOME", str(tmp_path / "empty"))
    target = tmp_path / "settings.toml"
    original_exists = Path.exists

    def denied(path):
        if path == target:
            raise OSError("synthetic-private-location/settings.toml: permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", denied)
    monkeypatch.setattr(sys, "argv", ["voiceover", "search", "сценарный", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert excinfo.value.code == 2
    assert payload["details"]["error_code"] == "SEARCH_SETTINGS_INVALID"
    assert payload["error"] == SEARCH_SETTINGS_INVALID_MESSAGE
    assert "synthetic-private-location" not in json.dumps(payload)
    assert not (tmp_path / "empty" / "history.sqlite3").exists()


@pytest.mark.parametrize(
    "body",
    ['[search]\ndefault_mode = "fuzzy"\n', "[search\ndefault_mode = lexical\n", "search = 3\n"],
)
def test_search_explicit_mode_overrides_invalid_settings(home, tmp_path, body):
    workdir = _write_settings(tmp_path / "cwd", body)

    returncode, payload = cli_json(
        "search", "сценарный", "--mode", "lexical", "--json", cwd=workdir
    )

    assert returncode == 0
    assert payload["mode"] == "lexical"
    assert payload["count"] == 1


def test_search_never_reads_the_env_file(home, tmp_path, monkeypatch):
    from voiceover_pipeline import cli as cli_module
    from voiceover_pipeline import config

    workdir = _write_settings(tmp_path / "cwd", '[search]\ndefault_mode = "lexical"\n')
    (workdir / ".env").write_text("POLZA_API_KEY=synthetic-placeholder-key\n", encoding="utf-8")
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(
        config, "read_env_file", lambda *args, **kwargs: pytest.fail("search read .env")
    )
    args = cli_module.build_parser().parse_args(["search", "сценарный", "--json"])

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(SystemExit) as excinfo:
            cli_module.search_cmd(args)

    assert excinfo.value.code == 0


def test_search_settings_come_from_toml_not_env(home, tmp_path):
    workdir = _write_settings(tmp_path / "cwd", '[search]\ndefault_mode = "lexical"\n')
    (workdir / ".env").write_text(
        "VOICEOVER_SEARCH_DEFAULT_MODE=semantic\nPOLZA_API_KEY=synthetic-placeholder-key\n",
        encoding="utf-8",
    )
    proc = run_cli("search", "сценарный", "--json", cwd=workdir)
    payload = json.loads(proc.stdout)

    assert proc.returncode == 0
    assert payload["mode"] == "lexical"
    assert "synthetic-placeholder-key" not in proc.stdout
    assert "synthetic-placeholder-key" not in proc.stderr
