"""Minimal non-secret file settings read from an optional ``settings.toml``.

Plan section 9 keeps three independent configuration roles: ``.env`` for
development secrets, ``settings.toml`` for durable non-secret settings, and
``speech-parts.yaml`` for content. This module is the small ``settings.toml``
reader for the one setting the canonical history path needs today:
``[history] enabled``.

The reader stays deliberately small: it reads one optional TOML file, imports no
optional runtime (no Torch, no provider package), opens no network connection,
and never reads ``.env`` or a secret. A missing file means the documented
defaults; a malformed file or a wrong value type fails closed with
:class:`SettingsError` instead of silently ignoring an explicit
``enabled = false``.

The file is resolved against the current working directory at call time (the
same place the existing ``.env`` default lives), so a run can be pointed at a
temporary file without adding a new flag or environment variable.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SETTINGS_FILENAME = "settings.toml"
HISTORY_SECTION = "history"
HISTORY_ENABLED_KEY = "enabled"


class SettingsError(RuntimeError):
    """The settings file is unreadable or holds an unsupported value."""


@dataclass(frozen=True)
class HistorySettings:
    """The ``[history]`` section of ``settings.toml``; ``enabled`` defaults true."""

    enabled: bool = True


def default_settings_path() -> Path:
    """Return ``<cwd>/settings.toml`` resolved now, not at import time."""
    return Path.cwd() / SETTINGS_FILENAME


def load_settings(path: Path | str | None = None) -> dict[str, Any]:
    """Read the optional settings file; a missing file yields an empty mapping."""
    target = default_settings_path() if path is None else Path(path).expanduser()
    if not target.exists():
        return {}
    try:
        with target.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"unable to read settings file {target}") from exc
    return data


def load_history_settings(path: Path | str | None = None) -> HistorySettings:
    """Return the validated ``[history]`` settings, failing closed on bad input.

    A missing file, a missing section, and a missing key all mean the documented
    default of enabled history. A non-table ``history`` section or a non-boolean
    ``enabled`` value raises :class:`SettingsError` so a typo cannot silently
    mean "enabled" when the user wrote something else.
    """
    data = load_settings(path)
    section = data.get(HISTORY_SECTION, {})
    if not isinstance(section, Mapping):
        raise SettingsError("the [history] settings section must be a TOML table")
    enabled = section.get(HISTORY_ENABLED_KEY, True)
    if not isinstance(enabled, bool):
        raise SettingsError("settings.toml history.enabled must be a boolean")
    return HistorySettings(enabled=enabled)
