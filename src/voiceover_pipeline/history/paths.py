"""Private history-home resolution and the managed layout of plan section 6.

The history database, managed run storage, backups, and logs live under one
explicit home directory instead of the current working directory, so a
tool-install location can never become the user's data store. This module
resolves that home and the OS data-directory default purely: no function here
creates a file or directory. :func:`ensure_history_home` and
:func:`ensure_private_directory` are the only explicit writers, and they reject
an existing home or subdirectory that is group- or world-accessible instead of
chmodding a user-owned tree.

The default is the platform data directory, so resolution never depends on the
current working directory:

* Linux and other POSIX: ``$XDG_DATA_HOME/voiceover-pipeline`` (falling back to
  ``~/.local/share/voiceover-pipeline``).
* macOS: ``~/Library/Application Support/voiceover-pipeline``.
* Windows: ``%LOCALAPPDATA%\\voiceover-pipeline`` (falling back to
  ``~/AppData/Local/voiceover-pipeline``).

``VOICEOVER_HOME`` overrides the whole path when set to a nonempty value, but it
must be absolute: a relative value would follow the current working directory,
which plan section 6 forbids, so it is rejected with
:class:`HistoryPathsError` instead of being resolved. A relative
``XDG_DATA_HOME`` is ignored in favor of the absolute user-home default, as the
XDG basedir specification requires. The resolver never reads ``.env`` or any
other configuration channel.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

# Directory under a platform data root that holds this application's history.
APP_DIR_NAME = "voiceover-pipeline"
HISTORY_DATABASE_FILENAME = "history.sqlite3"
RUNS_DIR_NAME = "runs"
BACKUPS_DIR_NAME = "backups"
LOGS_DIR_NAME = "logs"
_HOME_ENV_VAR = "VOICEOVER_HOME"
# Directories this module creates hold private user data and never become
# world- or group-readable.
PRIVATE_DIR_MODE = 0o700


class HistoryPathsError(RuntimeError):
    """Base class for history path contract violations."""


class HistoryHomePermissionError(HistoryPathsError):
    """An existing history directory is not private enough for plaintext history."""


def _is_absolute_config_path(value: str, *, platform: str | None = None) -> bool:
    """Check absoluteness for the active platform, not either path syntax.

    An injected Windows platform must accept ``C:\\Users\\u`` on a POSIX test
    host, and an injected POSIX platform must treat ``C:/...`` as relative even
    on a Windows host. Each platform is checked with the pure path class that
    spells it, so the answer never depends on the host running the check. On
    POSIX that same string is relative and must not make history follow the
    current working directory.
    """
    active_platform = sys.platform if platform is None else platform
    if active_platform.startswith("win"):
        return PureWindowsPath(value).is_absolute()
    return PurePosixPath(value).is_absolute()


def default_history_home(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the platform data-directory default for the history home.

    The platform, environment mapping, and user home are injectable so the
    contract can be exercised without depending on the host, but the returned
    path is always the fixed OS data location rather than anything relative to
    the current working directory.
    """
    active_platform = sys.platform if platform is None else platform
    active_environ = os.environ if environ is None else environ
    active_home = Path.home() if home is None else home

    if active_platform.startswith("win"):
        local_app_data = active_environ.get("LOCALAPPDATA")
        if isinstance(local_app_data, str) and _is_absolute_config_path(
            local_app_data, platform=active_platform
        ):
            base = Path(local_app_data)
        else:
            base = active_home / "AppData" / "Local"
    elif active_platform == "darwin":
        base = active_home / "Library" / "Application Support"
    else:
        xdg_data_home = active_environ.get("XDG_DATA_HOME")
        # A relative XDG_DATA_HOME is ignored, per the XDG basedir spec, so the
        # absolute user-home default keeps the history location CWD-independent.
        if isinstance(xdg_data_home, str) and _is_absolute_config_path(
            xdg_data_home, platform=active_platform
        ):
            base = Path(xdg_data_home)
        else:
            base = active_home / ".local" / "share"
    return base / APP_DIR_NAME


def resolve_history_home(
    environ: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the active history home without creating it.

    ``VOICEOVER_HOME`` wins when it is set to a nonempty absolute value; ``~`` is
    expanded first. A relative override raises :class:`HistoryPathsError` rather
    than resolving against the current working directory. Otherwise the platform
    data-directory default is used. Nothing is created or modified, so a dry run
    cannot have a side effect.
    """
    active_environ = os.environ if environ is None else environ
    override = active_environ.get(_HOME_ENV_VAR)
    if isinstance(override, str) and override.strip():
        candidate = Path(override.strip()).expanduser()
        if not _is_absolute_config_path(str(candidate), platform=platform):
            raise HistoryPathsError(
                f"{_HOME_ENV_VAR} must be an absolute path, got {override.strip()!r}"
            )
        return candidate
    return default_history_home(platform=platform, environ=active_environ, home=home)


def history_database_path(home: Path | str | None = None) -> Path:
    """Return ``<home>/history.sqlite3`` without creating anything."""
    return history_home(home) / HISTORY_DATABASE_FILENAME


def history_runs_dir(home: Path | str | None = None) -> Path:
    """Return ``<home>/runs`` without creating anything."""
    return history_home(home) / RUNS_DIR_NAME


def history_backups_dir(home: Path | str | None = None) -> Path:
    """Return ``<home>/backups`` without creating anything."""
    return history_home(home) / BACKUPS_DIR_NAME


def history_logs_dir(home: Path | str | None = None) -> Path:
    """Return ``<home>/logs`` without creating anything."""
    return history_home(home) / LOGS_DIR_NAME


def history_home(home: Path | str | None = None) -> Path:
    """Return an explicit home or the resolved one, still without writing."""
    return resolve_history_home() if home is None else Path(home).expanduser()


def ensure_private_directory(path: Path | str, *, parents: bool = False) -> Path:
    """Create or verify a private directory that will hold plaintext history.

    A missing directory is created with private mode. An existing directory is
    accepted only when it is already private: this function never chmods a
    user-owned directory, so a group- or world-accessible path is rejected with
    :class:`HistoryHomePermissionError` before any plaintext history is written
    into it. A symlinked directory is rejected rather than followed.
    """
    target = Path(path).expanduser()
    if target.is_symlink():
        raise HistoryHomePermissionError(f"history directory must not be a symlink: {target}")
    if target.exists():
        _require_private_directory(target)
        return target
    target.mkdir(mode=PRIVATE_DIR_MODE, parents=parents, exist_ok=True)
    _require_private_directory(target)
    return target


def _require_private_directory(path: Path) -> None:
    try:
        status = path.stat()
    except OSError as exc:
        raise HistoryHomePermissionError(f"cannot stat history directory {path}: {exc}") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise HistoryHomePermissionError(f"history path is not a directory: {path}")
    # Windows does not expose meaningful POSIX mode bits, so only enforce the
    # group/world check where those bits describe real access.
    if os.name == "posix" and stat.S_IMODE(status.st_mode) & 0o077:
        raise HistoryHomePermissionError(
            f"history directory {path} is group- or world-accessible; refusing to store "
            "plaintext history there instead of changing permissions of a user-owned "
            "directory"
        )


def ensure_history_home(home: Path | str | None = None) -> Path:
    """Create or verify the private home and its managed subdirectories.

    Directories are created with private mode and applied idempotently, but an
    already-existing home or subdirectory that is group- or world-accessible is
    rejected instead of being silently relaxed or chmodded.
    """
    root = history_home(home)
    ensure_private_directory(root, parents=True)
    for name in (RUNS_DIR_NAME, BACKUPS_DIR_NAME, LOGS_DIR_NAME):
        ensure_private_directory(root / name)
    return root
