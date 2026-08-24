"""Privacy-safe identity for the exact package bytes executing a run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _distribution_direct_url() -> dict[str, Any] | None:
    try:
        distribution = metadata.distribution("voiceover-pipeline")
    except metadata.PackageNotFoundError:
        return None
    for entry in distribution.files or ():
        if str(entry).endswith(".dist-info/direct_url.json"):
            try:
                value = json.loads(
                    Path(str(distribution.locate_file(entry))).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValueError, TypeError):
                return None
            return value if isinstance(value, dict) else None
    return None


def _package_tree_sha256(root: Path) -> str:
    records: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {relative}\n".encode("utf-8"))
    return hashlib.sha256(b"".join(records)).hexdigest()


def _git_identity(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return revision or None, bool(status)


def build_execution_identity() -> dict[str, Any]:
    """Describe installed/editable bytes without exposing local filesystem paths."""
    root = _package_root()
    direct_url = _distribution_direct_url()
    editable = bool((direct_url or {}).get("dir_info", {}).get("editable"))
    if editable:
        source_kind = "editable-checkout"
        revision, dirty = _git_identity(root)
    elif direct_url is not None:
        source_kind = "installed-wheel"
        revision, dirty = None, None
    else:
        revision, dirty = _git_identity(root)
        source_kind = "source-tree" if revision is not None else "installed-wheel"

    return {
        "package_version": __version__,
        "source_kind": source_kind,
        "source_revision": revision,
        "source_dirty": dirty,
        "package_tree_sha256": _package_tree_sha256(root),
    }
