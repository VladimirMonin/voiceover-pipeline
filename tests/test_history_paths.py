"""Contract tests for the private history-home resolver.

These tests exercise ``voiceover_pipeline.history.paths`` without any filesystem
side effect where the contract forbids one: the ``VOICEOVER_HOME`` override, the
platform data-directory default for Linux, macOS, and Windows, and the derived
layout. Only :func:`ensure_history_home` is allowed to create the private home,
and that is asserted explicitly.
"""

import stat
from pathlib import Path

import pytest

from voiceover_pipeline.history.paths import (
    APP_DIR_NAME,
    HISTORY_DATABASE_FILENAME,
    PRIVATE_DIR_MODE,
    HistoryHomePermissionError,
    HistoryPathsError,
    default_history_home,
    ensure_history_home,
    history_backups_dir,
    history_database_path,
    history_home,
    history_logs_dir,
    history_runs_dir,
    resolve_history_home,
)


def test_voiceover_home_override_wins():
    assert resolve_history_home({"VOICEOVER_HOME": "/srv/vo"}) == Path("/srv/vo")


def test_voiceover_home_override_expands_user():
    assert resolve_history_home({"VOICEOVER_HOME": "~/vo"}) == Path("~/vo").expanduser()


def test_relative_voiceover_home_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(HistoryPathsError):
        resolve_history_home({"VOICEOVER_HOME": "relative-home"})


def test_relative_voiceover_home_is_not_resolved_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOICEOVER_HOME", "relative-home")

    with pytest.raises(HistoryPathsError):
        ensure_history_home()

    assert list(tmp_path.iterdir()) == []


def test_windows_drive_home_is_relative_on_posix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(HistoryPathsError):
        resolve_history_home({"VOICEOVER_HOME": "C:/relative-on-posix"}, platform="linux")

    assert list(tmp_path.iterdir()) == []


def test_relative_xdg_data_home_is_ignored_for_absolute_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = default_history_home(
        platform="linux", environ={"XDG_DATA_HOME": "relative-data"}, home=Path("/home/u")
    )

    assert resolved == Path("/home/u") / ".local" / "share" / APP_DIR_NAME
    assert resolved.is_absolute()
    assert not str(resolved).startswith(str(tmp_path))


def test_windows_drive_xdg_value_is_ignored_on_posix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = default_history_home(
        platform="linux", environ={"XDG_DATA_HOME": "C:/relative-on-posix"}, home=Path("/home/u")
    )

    assert resolved == Path("/home/u") / ".local" / "share" / APP_DIR_NAME
    assert resolved.is_absolute()


def test_relative_local_app_data_falls_back_on_windows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = default_history_home(
        platform="win32", environ={"LOCALAPPDATA": "relative\\Local"}, home=Path("/w/u")
    )

    assert resolved == Path("/w/u") / "AppData" / "Local" / APP_DIR_NAME
    assert resolved.is_absolute()


def test_blank_override_falls_back_to_platform_default():
    resolved = resolve_history_home(
        {"VOICEOVER_HOME": "   ", "XDG_DATA_HOME": "/data"}, platform="linux", home=Path("/home/u")
    )

    assert resolved == Path("/data") / APP_DIR_NAME


def test_default_home_uses_xdg_data_home_on_linux():
    resolved = default_history_home(
        platform="linux", environ={"XDG_DATA_HOME": "/data"}, home=Path("/home/u")
    )

    assert resolved == Path("/data") / APP_DIR_NAME


def test_default_home_linux_falls_back_to_local_share():
    resolved = default_history_home(platform="linux", environ={}, home=Path("/home/u"))

    assert resolved == Path("/home/u") / ".local" / "share" / APP_DIR_NAME


def test_default_home_uses_application_support_on_macos():
    resolved = default_history_home(platform="darwin", environ={}, home=Path("/Users/u"))

    assert resolved == Path("/Users/u") / "Library" / "Application Support" / APP_DIR_NAME


def test_default_home_uses_local_app_data_on_windows():
    resolved = default_history_home(
        platform="win32", environ={"LOCALAPPDATA": "C:/Users/u/AppData/Local"}, home=Path("/w/u")
    )

    assert resolved == Path("C:/Users/u/AppData/Local") / APP_DIR_NAME


def test_default_home_windows_falls_back_to_appdata_local():
    resolved = default_history_home(platform="win32", environ={}, home=Path("/w/u"))

    assert resolved == Path("/w/u") / "AppData" / "Local" / APP_DIR_NAME


def test_default_home_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = default_history_home(platform="linux", environ={}, home=Path("/home/u"))

    assert resolved.is_absolute()
    assert not str(resolved).startswith(str(tmp_path))


def test_layout_helpers_derive_from_home(tmp_path):
    home = tmp_path / "home"

    assert history_home(home) == home
    assert history_database_path(home) == home / HISTORY_DATABASE_FILENAME
    assert history_runs_dir(home) == home / "runs"
    assert history_backups_dir(home) == home / "backups"
    assert history_logs_dir(home) == home / "logs"


def test_resolution_creates_nothing(tmp_path):
    home = tmp_path / "missing"

    assert resolve_history_home({"VOICEOVER_HOME": str(home)}) == home
    assert history_database_path(home) == home / HISTORY_DATABASE_FILENAME
    assert not home.exists()


def test_ensure_history_home_creates_private_layout_and_is_idempotent(tmp_path):
    home = tmp_path / "home"

    created = ensure_history_home(home)

    assert created == home
    for directory in (
        home,
        history_runs_dir(home),
        history_backups_dir(home),
        history_logs_dir(home),
    ):
        assert directory.is_dir()
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0
    assert PRIVATE_DIR_MODE == 0o700

    assert ensure_history_home(home) == home


def test_ensure_history_home_rejects_insecure_existing_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)

    with pytest.raises(HistoryHomePermissionError):
        ensure_history_home(home)

    assert stat.S_IMODE(home.stat().st_mode) == 0o755
    assert not history_database_path(home).exists()


def test_ensure_history_home_rejects_insecure_existing_subdir(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=PRIVATE_DIR_MODE)
    runs = history_runs_dir(home)
    runs.mkdir(mode=0o755)

    with pytest.raises(HistoryHomePermissionError):
        ensure_history_home(home)

    assert stat.S_IMODE(runs.stat().st_mode) == 0o755


def test_ensure_history_home_accepts_private_existing_home(tmp_path):
    home = tmp_path / "home"
    ensure_history_home(home)

    assert ensure_history_home(home) == home
