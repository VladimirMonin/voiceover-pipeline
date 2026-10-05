"""Contract tests for the private history-home resolver.

These tests exercise ``voiceover_pipeline.history.paths`` without any filesystem
side effect where the contract forbids one: the ``VOICEOVER_HOME`` override, the
platform data-directory default for Linux, macOS, and Windows, and the derived
layout. Only :func:`ensure_history_home` is allowed to create the private home,
and that is asserted explicitly.
"""

import os
import stat
from pathlib import Path

import pytest

from voiceover_pipeline.history import paths as history_paths
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

_POSIX = os.name == "posix"


class _SimulatedPosixOs:
    """An ``os`` shim that reports POSIX while proxying the host module.

    Windows stores no meaningful POSIX access bits and applies no group/world
    check, so the private-history policy is exercised by presenting the platform
    those bits belong to; every other attribute stays the host answer. On a POSIX
    host this only restates reality.
    """

    name = "posix"

    def __getattr__(self, attribute: str) -> object:
        return getattr(os, attribute)


def _simulate_posix_permission_modes(monkeypatch, modes: dict[Path, int]) -> None:
    """Present the POSIX platform and access bits the host kernel cannot report.

    Windows reports 0o777 for every directory, so the group- or world-access
    policy can only be exercised there by injecting both the platform the bits
    belong to and the mode answer for each path. On a POSIX host the injected
    modes match the real ``chmod`` above, so the same test covers the native
    answer as well.
    """
    monkeypatch.setattr(history_paths, "os", _SimulatedPosixOs())
    original_stat = Path.stat

    def simulated_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        status = original_stat(self, follow_symlinks=follow_symlinks)
        mode = modes.get(Path(self))
        if mode is None:
            return status
        return os.stat_result(
            (
                mode,
                status.st_ino,
                status.st_dev,
                status.st_nlink,
                status.st_uid,
                status.st_gid,
                status.st_size,
                status.st_atime,
                status.st_mtime,
                status.st_ctime,
            )
        )

    monkeypatch.setattr(Path, "stat", simulated_stat)


def test_voiceover_home_override_wins(tmp_path):
    absolute = tmp_path / "vo"

    assert resolve_history_home({"VOICEOVER_HOME": str(absolute)}) == absolute


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
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    home = tmp_path / "home"

    resolved = default_history_home(
        platform="linux", environ={"XDG_DATA_HOME": "relative-data"}, home=home
    )

    assert resolved == home / ".local" / "share" / APP_DIR_NAME
    assert resolved.is_absolute()
    assert not resolved.is_relative_to(cwd)


def test_windows_drive_xdg_value_is_ignored_on_posix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"

    resolved = default_history_home(
        platform="linux", environ={"XDG_DATA_HOME": "C:/relative-on-posix"}, home=home
    )

    assert resolved == home / ".local" / "share" / APP_DIR_NAME
    assert resolved.is_absolute()


def test_relative_local_app_data_falls_back_on_windows(tmp_path):
    home = tmp_path / "home"

    resolved = default_history_home(
        platform="win32", environ={"LOCALAPPDATA": "relative\\Local"}, home=home
    )

    assert resolved == home / "AppData" / "Local" / APP_DIR_NAME
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
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    home = tmp_path / "home"

    resolved = default_history_home(platform="linux", environ={}, home=home)

    assert resolved.is_absolute()
    assert not resolved.is_relative_to(cwd)


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
    # Windows stores no meaningful POSIX access bits for the new directories, so
    # the private mode is only asserted where those bits describe real access.
    if _POSIX:
        assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0
    assert PRIVATE_DIR_MODE == 0o700

    assert ensure_history_home(home) == home


def test_ensure_history_home_rejects_insecure_existing_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)
    before_mode = os.stat(home).st_mode
    _simulate_posix_permission_modes(monkeypatch, {home: stat.S_IFDIR | 0o755})

    with pytest.raises(HistoryHomePermissionError):
        ensure_history_home(home)

    assert os.stat(home).st_mode == before_mode
    assert not history_database_path(home).exists()


def test_ensure_history_home_rejects_insecure_existing_subdir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=PRIVATE_DIR_MODE)
    runs = history_runs_dir(home)
    runs.mkdir(mode=0o755)
    before_mode = os.stat(runs).st_mode
    _simulate_posix_permission_modes(
        monkeypatch, {home: stat.S_IFDIR | PRIVATE_DIR_MODE, runs: stat.S_IFDIR | 0o755}
    )

    with pytest.raises(HistoryHomePermissionError):
        ensure_history_home(home)

    assert os.stat(runs).st_mode == before_mode


def test_ensure_history_home_accepts_private_existing_home(tmp_path):
    home = tmp_path / "home"
    ensure_history_home(home)

    assert ensure_history_home(home) == home
