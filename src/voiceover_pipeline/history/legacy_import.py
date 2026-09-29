"""Safe offline import of legacy run directories into the canonical history.

Plan section 6 stage S04 makes old ``out/<run-id>`` trees searchable through the
SQLite history without touching the current generation writers. This module owns
reading a legacy run tree, projecting it onto the typed history records, and
importing it atomically. It performs no provider, ASR, FFmpeg, model, embedding,
paid, or network call, never rewrites or deletes an original file, and never
follows a symlink or a path that escapes the source tree.

The same projection backs a future ``history import --dry-run``:
:func:`preview_legacy_import` performs no write at all -- no database, directory,
original, or WAL sidecar -- and reports discovered roots, counts, missing texts
and files, prices, and conflicts without exposing full text or secret values.
:func:`import_legacy_runs` writes only through the caller-supplied database path.

Safety contracts enforced here:

* Identity fails closed. A malformed, symlinked, oversized, or ambiguous
  identity file (``manifest.json``, ``run_state.json``, the run manifest) makes a
  root non-importable instead of being guessed around; orphan audio is never
  "legalized" into a run.
* An external script path from the JSON is read only when it is a small,
  text-like regular file inside the source tree and no path component is hidden
  (``.env``, ``.private/credentials.txt``) or a symlink; anything else leaves the
  run with a hash-only ``text_completeness=incomplete`` text source.
* A symlinked chunk or main-audio file, or a file under a symlinked ancestor
  directory such as ``chunks/``, is not referenced at all, not even as a missing
  artifact, because its path can resolve outside the source tree.
* The preview validates the existing database's migration ledger and schema on
  an ``immutable=1`` read-only connection, so a foreign, corrupt, too-new, or
  WAL-backed database reports an unknown status (``database_status_unknown``)
  instead of an unearned conflict-free one, and never creates a sidecar. A
  symlinked database file, or one under a symlinked home directory, is rejected
  before any connect, probe, or write. The importer refuses to write plaintext
  history into a group- or world-accessible existing directory instead of
  chmodding a user-owned tree.
* Costs prefer a valid exact string and otherwise keep the legacy JSON numeric
  lexeme with ``source=legacy_import`` and ``exact_available=False``. A stored
  ``0`` is a real zero, a missing price is unknown, currencies never merge, and
  a run total is imported only when no per-chunk price exists, so a per-chunk
  charge is never counted twice.
* Imported config snapshots are whitelisted field-by-field and still pass
  through the repository's fail-closed secret redaction. Log events carry only
  bounded labels and counts; a conflict label that looks like a signed URL or a
  secret token is replaced by a short digest instead of the raw run ID or root
  name.
* Every part, attempt, artifact, and text source lands in one caller-managed
  transaction with its run, so a failed import rolls back completely and a retry
  imports the whole run exactly once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .database import (
    MIGRATIONS,
    HistoryDatabase,
    HistoryDatabaseError,
    _validate_schema,
)
from .paths import ensure_history_home, ensure_private_directory, history_database_path
from .repository import (
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    COST_SOURCE_LEGACY_FLOAT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    TEXT_COMPLETENESS_COMPLETE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    Cost,
    HistoryRepository,
    _is_sensitive_string,
)

_logger = logging.getLogger("voiceover_pipeline.history")

LEGACY_ORIGIN = "legacy_import"
LEGACY_OPERATION = "tts"

MANIFEST_FILE = "manifest.json"
STATE_FILE = "run_state.json"
CHUNKS_MANIFEST_RELATIVE = "chunks/chunks.json"
RUN_JSON_GLOB = "*-voiceover-*.json"
RUN_AUDIO_GLOB = "*-voiceover-*.mp3"
TIMINGS_GLOB = "*.timings.json"
SRT_GLOB = "*.srt"

# Bounds keep a hostile or accidental tree from making the scan unbounded.
MAX_SCAN_DEPTH = 3
MAX_RUN_ROOTS = 500
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_CHUNKS_PER_RUN = 20_000
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_LABEL_LENGTH = 64
# A bounded run ID or root name can still be a short signed URL or a
# secret-bearing token; a conflict log line replaces such a label with a digest.
_UNSAFE_LOG_LABEL_MARKERS = ("://", "?", "#", "@")

_RUN_MANIFEST_ROLE = "run_manifest"
_MIME_JSON = "application/json"
_MIME_MP3 = "audio/mpeg"
_MIME_SRT = "application/x-subrip"

# A legacy manifest's ``script`` field names the original input path, which is
# untrusted JSON. Only a text-like regular file inside the source tree is read;
# a hidden file (including ``.env``), a key, a certificate, or any other
# non-text file is left unavailable with a hash-only incomplete marker.
_SCRIPT_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".text", ".json", ".srt", ".vtt", ".yml", ".yaml"}
)

_VALID_STATUSES = frozenset({"running", "completed", "failed"})
_MALFORMED_JSON_CODES = frozenset(
    {"malformed_json", "json_unreadable", "json_too_large", "symlinked_json"}
)
# A declared identity file that exists only as a symlink must block the import
# rather than let the run fall back to a different identity file.
_BLOCKING_SELECTION_CODES = frozenset({"symlinked_run_manifest"})
# Directories that never contain a nested run and would only waste scan budget.
_SKIPPED_SCAN_DIRS = frozenset(
    {
        "__pycache__",
        "backups",
        "chunks",
        "logs",
        "node_modules",
        "raw",
        "source",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COST_SNAPSHOT_KEYS = (
    "artifact_type",
    "provider",
    "model",
    "voice",
    "style_prompt",
    "prompt_mode",
    "script_format",
    "format",
    "speaker_voice_map",
    "pricing_snapshot",
    "execution_source",
    "voice_identity",
    "synthesis_identity",
    "tts_quality",
    "chunk_count",
    "limited_to_chunks",
    "total_duration_ms",
    "total_duration_sec",
    "main_duration_ms",
    "main_duration_sec",
    "cost_exact_available",
    "cost_currency",
    "cost_source",
    "cost_per_minute",
    "duration_source",
    "concat_method",
)


@dataclass(frozen=True)
class LegacyChunkPreview:
    """One discovered chunk, without its prepared text or transcript."""

    number: int
    chunk_id: str | None
    has_text: bool
    audio_relative: str | None
    audio_available: bool
    cost_amount: str | None
    cost_currency: str | None
    cost_source: str
    cost_exact_available: bool


@dataclass(frozen=True)
class LegacyRunPreview:
    """What a dry run reports about one legacy run root."""

    source_root: str
    run_id: str | None
    operation: str
    status: str
    provider: str | None
    model: str | None
    voice: str | None
    chunk_count: int
    chunks_with_text: int
    chunks_missing_text: int
    chunks_with_audio: int
    chunks_missing_audio: int
    chunks_with_cost: int
    cost_total: str | None
    cost_total_exact: str | None
    cost_currency: str | None
    script_text_available: bool
    already_imported: bool | None
    importable: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class LegacyImportPreview:
    """The aggregate dry-run result for a source directory."""

    source: str
    database_path: str
    database_exists: bool
    database_readable: bool
    runs: tuple[LegacyRunPreview, ...]
    scan_conflicts: tuple[str, ...] = ()

    @property
    def discovered_count(self) -> int:
        return len(self.runs)

    @property
    def importable_count(self) -> int:
        return sum(1 for run in self.runs if run.importable)

    @property
    def already_imported_count(self) -> int:
        return sum(1 for run in self.runs if run.already_imported is True)

    @property
    def missing_text_count(self) -> int:
        return sum(run.chunks_missing_text for run in self.runs)

    @property
    def missing_audio_count(self) -> int:
        return sum(run.chunks_missing_audio for run in self.runs)


@dataclass(frozen=True)
class LegacyRunImportResult:
    """What one run root contributed to an import call."""

    source_root: str
    run_uuid: str | None
    created: bool
    parts: int
    attempts: int
    artifacts: int
    text_sources: int
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class LegacyImportResult:
    """The aggregate import result for a source directory."""

    source: str
    database_path: str
    runs: tuple[LegacyRunImportResult, ...]

    @property
    def imported_count(self) -> int:
        return sum(1 for run in self.runs if run.created)

    @property
    def skipped_count(self) -> int:
        return sum(1 for run in self.runs if not run.created and run.run_uuid is not None)

    @property
    def rejected_count(self) -> int:
        return sum(1 for run in self.runs if run.run_uuid is None and not run.created)


@dataclass(frozen=True)
class _LegacyChunk:
    number: int
    chunk_id: str | None
    text: str | None
    text_hash: str | None
    voice: str | None
    audio_relative: str | None
    audio_path: Path | None
    audio_available: bool
    audio_sha256: str | None
    audio_size_bytes: int | None
    cost: Cost


@dataclass(frozen=True)
class _LegacyArtifact:
    role: str
    path: Path
    available: bool
    mime: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    part_number: int | None = None


@dataclass(frozen=True)
class _LegacyRun:
    root: Path
    run_id: str | None
    operation: str
    status: str
    provider: str | None
    model: str | None
    voice: str | None
    chunks: tuple[_LegacyChunk, ...]
    has_chunk_cost: bool
    run_total_cost: Cost
    config_snapshot: dict[str, Any] | None
    script_text: str | None
    script_hash: str | None
    style_prompt: str | None
    artifacts: tuple[_LegacyArtifact, ...]
    conflicts: tuple[str, ...]
    importable: bool


# -- small bounded helpers ----------------------------------------------------


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _has_symlink_component(path: Path, root: Path) -> bool:
    """Return whether any component from ``root`` down to ``path`` is a symlink.

    The final filename alone is not enough: a symlinked ``chunks`` directory
    redirects ``<root>/chunks/chunks.json`` outside the run tree while the leaf
    file itself is a regular file, so every component below ``root`` is checked.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _bounded_text(value: Any, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_length or any(ch.isspace() for ch in text):
        return None
    return text


def _bounded_sha256(value: Any) -> str | None:
    if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None:
        return value
    return None


def _safe_artifact_name(value: Any) -> str | None:
    """Keep only a plain file name that cannot traverse out of its directory."""
    if not isinstance(value, str) or not value:
        return None
    if value in {".", ".."} or "/" in value or "\\" in value or value.startswith("."):
        return None
    return value


def _is_decimal_text(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Decimal(value).is_finite()
    except InvalidOperation:
        return False


def _is_readable_file(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not _is_within(resolved, root):
        return False
    return resolved.is_file()


def _file_artifact(
    *,
    role: str,
    path: Path,
    root: Path,
    mime: str | None,
    part_number: int | None = None,
    sha256: str | None = None,
) -> _LegacyArtifact | None:
    """Return an artifact for a regular, in-tree file, else ``None``."""
    if _has_symlink_component(path, root) or not path.is_file():
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not _is_within(resolved, root) or not resolved.is_file():
        return None
    try:
        size_bytes = resolved.stat().st_size
    except OSError:
        size_bytes = None
    return _LegacyArtifact(
        role=role,
        path=resolved,
        available=True,
        mime=mime,
        size_bytes=size_bytes,
        sha256=sha256,
        part_number=part_number,
    )


def _load_json_object(path: Path, root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON object with a bounded size, reporting why it was rejected.

    A symlink is never followed, and neither is a symlinked ancestor directory:
    a symlinked ``chunks`` directory would otherwise let ``chunks/chunks.json``
    resolve outside the run tree while the leaf file itself is not a symlink. A
    missing file is not an error; every other failure returns a code the caller
    turns into a conflict rather than guessing around it.
    """
    if _has_symlink_component(path, root):
        return None, "symlinked_json"
    if not path.is_file():
        return None, None
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None, "json_too_large"
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, "json_unreadable"
    try:
        data = json.loads(raw, parse_float=Decimal)
    except (ValueError, RecursionError):
        return None, "malformed_json"
    if not isinstance(data, dict):
        return None, "malformed_json"
    return data, None


def _snapshot_safe(value: Any) -> Any:
    """Recursively project a legacy JSON value onto JSON-serializable types.

    Only whitelisted snapshot fields reach this function, and the repository
    still redacts secrets afterwards. ``Decimal`` becomes its captured string so
    a price lexeme survives without a binary float.
    """
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _snapshot_safe(item) for key, item in value.items() if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [_snapshot_safe(item) for item in value]
    return None


def _first_text(sources: tuple[Any, ...], key: str) -> str | None:
    for source in sources:
        if isinstance(source, dict):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


# -- scan ---------------------------------------------------------------------


def _looks_like_run_root(directory: Path) -> bool:
    return os.path.lexists(directory / MANIFEST_FILE) or os.path.lexists(directory / STATE_FILE)


def _discover_run_roots(source_root: Path) -> tuple[list[Path], list[str]]:
    """Find bounded, symlink-safe run roots beneath ``source_root``."""
    conflicts: list[str] = []
    if not source_root.is_dir():
        return [], ["source_not_found"]

    roots: list[Path] = []
    seen: set[Path] = set()
    stack: list[tuple[Path, int]] = [(source_root, 0)]
    while stack:
        directory, depth = stack.pop()
        if _looks_like_run_root(directory):
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)
            continue
        if depth >= MAX_SCAN_DEPTH:
            continue
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            conflicts.append("unreadable_directory")
            continue
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                continue
            if entry.name.startswith(".") or entry.name in _SKIPPED_SCAN_DIRS:
                continue
            stack.append((entry, depth + 1))

    roots.sort()
    if len(roots) > MAX_RUN_ROOTS:
        conflicts.append("too_many_roots")
        roots = roots[:MAX_RUN_ROOTS]
    return roots, conflicts


def _select_run_json(root: Path, manifest: dict[str, Any] | None) -> tuple[Path | None, str | None]:
    """Choose the run manifest among top-level candidates, or report ambiguity.

    A run manifest named by ``manifest.json`` that exists only as a symlink is a
    blocking conflict: the run's declared identity is not trustworthy, so the
    importer must not silently fall back to another file.
    """
    candidates = [
        path
        for path in sorted(root.glob(RUN_JSON_GLOB))
        if path.is_file() and not path.is_symlink()
    ]
    declared_name: str | None = None
    if isinstance(manifest, dict):
        declared = manifest.get("run_json")
        if isinstance(declared, str) and declared.strip():
            declared_name = Path(declared).name
    if declared_name is not None:
        if (root / declared_name).is_symlink():
            return None, "symlinked_run_manifest"
        for candidate in candidates:
            if candidate.name == declared_name:
                return candidate, None
    if not candidates:
        return None, None
    if len(candidates) == 1:
        return candidates[0], None
    return None, "ambiguous_run_json"


# -- projection ---------------------------------------------------------------


def _chunk_cost(entry: dict[str, Any], default_currency: str | None) -> Cost:
    currency = entry.get("cost_currency")
    if not isinstance(currency, str) or not currency:
        currency = default_currency
    exact = entry.get("cost_exact")
    if isinstance(exact, str) and _is_decimal_text(exact):
        return Cost.exact(exact, currency=currency, source=COST_SOURCE_LEGACY_FLOAT)
    rub_exact = entry.get("cost_rub_exact")
    if isinstance(rub_exact, str) and _is_decimal_text(rub_exact):
        return Cost.exact(rub_exact, currency="RUB", source=COST_SOURCE_LEGACY_FLOAT)
    for field, field_currency in (("cost", currency), ("cost_rub", "RUB")):
        value = entry.get(field)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, Decimal, str)):
            cost = Cost.legacy_float(value, currency=field_currency)
            if cost.amount is not None:
                return cost
    return Cost.unknown()


def _run_total_cost(
    manifest: dict[str, Any] | None,
    state: dict[str, Any] | None,
    run_json: dict[str, Any] | None,
    default_currency: str | None,
) -> Cost:
    for source in (manifest, state, run_json):
        if not isinstance(source, dict):
            continue
        currency = source.get("cost_currency")
        if not isinstance(currency, str) or not currency:
            currency = default_currency
        exact = source.get("cost_total_exact")
        if isinstance(exact, str) and _is_decimal_text(exact):
            return Cost.exact(exact, currency=currency, source=COST_SOURCE_LEGACY_FLOAT)
        total = source.get("cost_total")
        if (
            total is not None
            and not isinstance(total, bool)
            and isinstance(total, (int, Decimal, str))
        ):
            cost = Cost.legacy_float(total, currency=currency)
            if cost.amount is not None:
                return cost
    return Cost.unknown()


def _chunk_from_entry(
    entry: dict[str, Any], root: Path, default_currency: str | None, conflicts: list[str]
) -> _LegacyChunk | None:
    number = entry.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        return None
    status = entry.get("status")
    if status not in (None, "completed"):
        return None

    audio_relative: str | None = None
    audio_path: Path | None = None
    audio_available = False
    file_value = entry.get("file")
    if isinstance(file_value, str) and file_value:
        name = _safe_artifact_name(file_value)
        if name is None:
            conflicts.append("unsafe_chunk_path")
        else:
            candidate = root / "chunks" / name
            if _has_symlink_component(candidate, root):
                # A symlink on the file or on any ancestor such as ``chunks``
                # can point outside the run root, so its path is not even
                # recorded as a missing artifact.
                conflicts.append("symlinked_artifact")
            else:
                audio_relative = f"chunks/{name}"
                audio_path = candidate
                audio_available = _is_readable_file(candidate, root)

    audio_size_bytes = None
    if audio_available and audio_path is not None:
        try:
            audio_size_bytes = audio_path.resolve().stat().st_size
        except OSError:
            audio_size_bytes = None

    text = entry.get("text")
    return _LegacyChunk(
        number=number,
        chunk_id=_bounded_text(entry.get("id"), 128),
        text=text if isinstance(text, str) else None,
        text_hash=_bounded_sha256(entry.get("text_hash")),
        voice=_bounded_text(entry.get("voice"), 128),
        audio_relative=audio_relative,
        audio_path=audio_path,
        audio_available=audio_available,
        audio_sha256=_bounded_sha256(entry.get("audio_sha256")),
        audio_size_bytes=audio_size_bytes,
        cost=_chunk_cost(entry, default_currency),
    )


def _chunk_entries(sources: tuple[Any, ...]) -> list[dict[str, Any]]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        entries = source.get("chunks")
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _collect_chunks(
    state: dict[str, Any] | None,
    chunks_json: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    run_json: dict[str, Any] | None,
    root: Path,
    default_currency: str | None,
    conflicts: list[str],
) -> list[_LegacyChunk]:
    entries = _chunk_entries((state, chunks_json, manifest, run_json))
    if len(entries) > MAX_CHUNKS_PER_RUN:
        conflicts.append("too_many_chunks")
        entries = entries[:MAX_CHUNKS_PER_RUN]
    chunks: list[_LegacyChunk] = []
    seen_numbers: set[int] = set()
    for entry in entries:
        chunk = _chunk_from_entry(entry, root, default_currency, conflicts)
        if chunk is None:
            conflicts.append("invalid_chunk_entry")
            continue
        if chunk.number in seen_numbers:
            conflicts.append("duplicate_chunk_number")
            continue
        seen_numbers.add(chunk.number)
        chunks.append(chunk)
    chunks.sort(key=lambda chunk: chunk.number)
    return chunks


def _main_audio_artifact(
    root: Path, manifest: dict[str, Any] | None, conflicts: list[str]
) -> _LegacyArtifact | None:
    declared_candidates: list[Path] = []
    if isinstance(manifest, dict):
        declared = manifest.get("full_mp3")
        if isinstance(declared, str) and declared.strip():
            name = _safe_artifact_name(Path(declared).name)
            if name is None:
                conflicts.append("unsafe_main_audio_path")
            else:
                declared_candidates.append(root / name)
    glob_candidates = sorted(root.glob(RUN_AUDIO_GLOB))
    for candidate in [*declared_candidates, *glob_candidates]:
        artifact = _file_artifact(role="main_audio", path=candidate, root=root, mime=_MIME_MP3)
        if artifact is not None:
            return artifact
    # The real file is gone. Record a missing artifact only for a candidate that
    # is not a symlink, because a symlinked name may resolve outside the root.
    for candidate in [*declared_candidates, *glob_candidates]:
        if not _has_symlink_component(candidate, root):
            return _LegacyArtifact(
                role="main_audio", path=candidate, available=False, mime=_MIME_MP3
            )
    return None


def _collect_artifacts(
    root: Path,
    manifest: dict[str, Any] | None,
    run_json_path: Path | None,
    chunks: list[_LegacyChunk],
    conflicts: list[str],
) -> list[_LegacyArtifact]:
    artifacts: list[_LegacyArtifact] = []
    for chunk in chunks:
        if chunk.audio_path is None:
            continue
        path = chunk.audio_path.resolve() if chunk.audio_available else chunk.audio_path
        artifacts.append(
            _LegacyArtifact(
                role="chunk_audio",
                path=path,
                available=chunk.audio_available,
                mime=_MIME_MP3,
                size_bytes=chunk.audio_size_bytes,
                sha256=chunk.audio_sha256,
                part_number=chunk.number,
            )
        )
    main_audio = _main_audio_artifact(root, manifest, conflicts)
    if main_audio is not None:
        artifacts.append(main_audio)
    for role, relative in (
        ("manifest", MANIFEST_FILE),
        ("run_state", STATE_FILE),
        ("chunks_manifest", CHUNKS_MANIFEST_RELATIVE),
    ):
        artifact = _file_artifact(role=role, path=root / relative, root=root, mime=_MIME_JSON)
        if artifact is not None:
            artifacts.append(artifact)
    if run_json_path is not None:
        artifact = _file_artifact(
            role=_RUN_MANIFEST_ROLE, path=run_json_path, root=root, mime=_MIME_JSON
        )
        if artifact is not None:
            artifacts.append(artifact)
    for role, pattern, mime in (
        ("timings", TIMINGS_GLOB, _MIME_JSON),
        ("srt", SRT_GLOB, _MIME_SRT),
    ):
        for path in sorted(root.glob(pattern)):
            artifact = _file_artifact(role=role, path=path, root=root, mime=mime)
            if artifact is not None:
                artifacts.append(artifact)
                break
    return artifacts


def _build_config_snapshot(
    state: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    run_json: dict[str, Any] | None,
    run_id: str | None,
    run_total_cost: Cost,
) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    for source in (state, manifest, run_json):
        if not isinstance(source, dict):
            continue
        for key in _COST_SNAPSHOT_KEYS:
            if key in snapshot:
                continue
            value = _snapshot_safe(source.get(key))
            if value is not None:
                snapshot[key] = value
    if run_id is not None:
        snapshot["run_id"] = run_id
    if run_total_cost.amount is not None:
        snapshot["cost_total"] = run_total_cost.amount
        if run_total_cost.exact_available:
            snapshot["cost_total_exact"] = run_total_cost.amount
    if run_total_cost.currency is not None:
        snapshot["cost_currency"] = run_total_cost.currency
    snapshot.setdefault("artifact_type", "voiceover-run")
    return snapshot or None


def _has_hidden_or_symlink_component(path: Path, root: Path) -> bool:
    """Return whether any component between ``root`` and ``path`` is hidden or a symlink.

    The final filename alone is not enough: a path such as
    ``<source>/.private/credentials.txt`` hides a secret in an ancestor directory,
    and a symlinked ancestor can redirect the read outside the intended tree.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    if any(part.startswith(".") for part in relative.parts):
        return True
    return _has_symlink_component(path, root)


def _safe_script_text(source_root: Path, raw: Any) -> str | None:
    """Read a script only when it is a small text file inside the source tree.

    The JSON-supplied path is untrusted, so it must be an absolute, text-like
    regular file inside the source tree whose every path component is neither
    hidden (which excludes ``.env`` and ``.private/credentials.txt``) nor a
    symlink. Anything else keeps the run's hash-only
    ``text_completeness=incomplete`` marker.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip())
    if path.suffix.lower() not in _SCRIPT_SUFFIXES:
        return None
    if not path.is_absolute():
        return None
    if _has_hidden_or_symlink_component(path, source_root):
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not _is_within(resolved, source_root):
        return None
    if _has_hidden_or_symlink_component(resolved, source_root):
        return None
    try:
        if not resolved.is_file() or resolved.stat().st_size > MAX_TEXT_BYTES:
            return None
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _bounded_status(state: dict[str, Any] | None, manifest: dict[str, Any] | None) -> str:
    if isinstance(state, dict):
        value = state.get("status")
        if isinstance(value, str) and value in _VALID_STATUSES:
            return value
    return "completed" if manifest is not None else "running"


def _parse_run_root(root: Path, source_root: Path) -> _LegacyRun:
    """Project one run root onto typed records, failing closed on bad identity."""
    conflicts: list[str] = []
    blocking = False

    state, state_code = _load_json_object(root / STATE_FILE, root)
    if state_code in _MALFORMED_JSON_CODES:
        conflicts.append(f"run_state_{state_code}")
        blocking = True
    manifest, manifest_code = _load_json_object(root / MANIFEST_FILE, root)
    if manifest_code in _MALFORMED_JSON_CODES:
        conflicts.append(f"manifest_{manifest_code}")
        blocking = True
    chunks_json, chunks_code = _load_json_object(root / CHUNKS_MANIFEST_RELATIVE, root)
    if chunks_code is not None:
        conflicts.append(f"chunks_manifest_{chunks_code}")

    run_json_path, run_json_conflict = _select_run_json(root, manifest)
    if run_json_conflict is not None:
        conflicts.append(run_json_conflict)
        if run_json_conflict in _BLOCKING_SELECTION_CODES:
            blocking = True
    run_json: dict[str, Any] | None = None
    if run_json_path is not None:
        run_json, run_json_code = _load_json_object(run_json_path, root)
        if run_json_code in _MALFORMED_JSON_CODES:
            conflicts.append(f"run_manifest_{run_json_code}")
            blocking = True

    if state is None and manifest is None and run_json is None:
        conflicts.append("no_readable_identity")
        blocking = True

    explicit_ids: list[str] = []
    for source in (state, manifest, run_json):
        if isinstance(source, dict):
            value = source.get("run_id")
            if isinstance(value, str) and value.strip():
                explicit_ids.append(value.strip())
    if len(set(explicit_ids)) > 1:
        conflicts.append("run_id_mismatch")
        blocking = True
    run_id = (
        _bounded_text(explicit_ids[0], MAX_LABEL_LENGTH)
        if explicit_ids
        else _bounded_text(root.name, MAX_LABEL_LENGTH)
    )

    default_currency: str | None = None
    for source in (manifest, state, run_json):
        if isinstance(source, dict):
            value = source.get("cost_currency")
            if isinstance(value, str) and value:
                default_currency = value
                break

    chunks = _collect_chunks(
        state, chunks_json, manifest, run_json, root, default_currency, conflicts
    )
    run_total_cost = _run_total_cost(manifest, state, run_json, default_currency)

    sources: tuple[Any, ...] = (state, manifest, run_json)
    script_raw = _first_text(sources, "script")
    script_text = _safe_script_text(source_root, script_raw)
    script_hash = _bounded_sha256(_first_text(sources, "script_hash"))

    artifacts = _collect_artifacts(root, manifest, run_json_path, chunks, conflicts)

    return _LegacyRun(
        root=root,
        run_id=run_id,
        operation=LEGACY_OPERATION,
        status=_bounded_status(state, manifest),
        provider=_first_text(sources, "provider"),
        model=_first_text(sources, "model"),
        voice=_first_text(sources, "voice"),
        chunks=tuple(chunks),
        has_chunk_cost=any(chunk.cost.amount is not None for chunk in chunks),
        run_total_cost=run_total_cost,
        config_snapshot=_build_config_snapshot(state, manifest, run_json, run_id, run_total_cost),
        script_text=script_text,
        script_hash=script_hash,
        style_prompt=_first_text(sources, "style_prompt"),
        artifacts=tuple(artifacts),
        conflicts=tuple(conflicts),
        importable=not blocking,
    )


def _to_preview(run: _LegacyRun, already_imported: bool | None) -> LegacyRunPreview:
    chunks = run.chunks
    return LegacyRunPreview(
        source_root=str(run.root),
        run_id=run.run_id,
        operation=run.operation,
        status=run.status,
        provider=run.provider,
        model=run.model,
        voice=run.voice,
        chunk_count=len(chunks),
        chunks_with_text=sum(1 for chunk in chunks if chunk.text is not None),
        chunks_missing_text=sum(1 for chunk in chunks if chunk.text is None),
        chunks_with_audio=sum(1 for chunk in chunks if chunk.audio_available),
        chunks_missing_audio=sum(
            1 for chunk in chunks if chunk.audio_path is not None and not chunk.audio_available
        ),
        chunks_with_cost=sum(1 for chunk in chunks if chunk.cost.amount is not None),
        cost_total=run.run_total_cost.amount,
        cost_total_exact=run.run_total_cost.amount if run.run_total_cost.exact_available else None,
        cost_currency=run.run_total_cost.currency,
        script_text_available=run.script_text is not None,
        already_imported=already_imported,
        importable=run.importable,
        conflicts=run.conflicts,
    )


# -- public API ---------------------------------------------------------------


def _resolve_database_path(database_path: Path | str | None) -> Path:
    path = history_database_path() if database_path is None else Path(database_path).expanduser()
    return path.absolute()


def _probe_database(database_path: Path) -> tuple[str, set[str]]:
    """Classify an existing database read-only and return its imported legacy roots.

    Returns ``(status, imported_roots)`` where status is one of ``absent``,
    ``empty``, ``ok``, or ``unknown``. An ordinary ``mode=ro`` connection is not a
    no-write operation: on a WAL database it can create or touch ``-shm``/``-wal``
    sidecars. This check instead opens the main file directly with
    ``immutable=1`` and never creates or modifies the database or a sidecar.
    ``immutable=1`` is only trustworthy when no WAL or journal can hold committed
    state outside the main file, so the presence of a ``-wal`` or ``-journal``
    file (even an empty leftover) reports ``unknown``.

    The status is ``ok`` or ``empty`` only when the database passes the *same*
    migration-ledger and schema validation that :meth:`HistoryDatabase.migrate`
    applies, so a foreign, corrupt, or too-new database is reported ``unknown``
    instead of being mistaken for an importable database without conflicts. The
    canonical validator is reused rather than reimplemented so the preview and
    the import can never disagree on what is acceptable. A symlinked database
    file, or one under a symlinked home directory, is never followed: the target
    is reported ``unknown`` without being read or written.
    """
    if database_path.is_symlink() or database_path.parent.is_symlink():
        return "unknown", set()
    if not database_path.exists():
        return "absent", set()
    for suffix in ("-wal", "-journal"):
        sidecar = database_path.with_name(database_path.name + suffix)
        try:
            if sidecar.exists():
                return "unknown", set()
        except OSError:
            return "unknown", set()
    try:
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro&immutable=1", uri=True, timeout=2.0
        )
    except (sqlite3.Error, ValueError):
        return "unknown", set()
    try:
        try:
            recorded = _validate_schema(connection, MIGRATIONS)
        except (HistoryDatabaseError, sqlite3.Error):
            return "unknown", set()
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not recorded:
            # No ledger: only a genuinely empty database is acceptable, exactly
            # as migrate() treats it. Anything else is unrecognized.
            return ("empty", set()) if not tables else ("unknown", set())
        if "runs" not in tables:
            return "unknown", set()
        rows = connection.execute(
            "SELECT legacy_source_root FROM runs WHERE legacy_source_root IS NOT NULL"
        ).fetchall()
        return "ok", {str(value[0]) for value in rows}
    except sqlite3.Error:
        return "unknown", set()
    finally:
        connection.close()


def preview_legacy_import(
    source: Path | str, *, database_path: Path | str | None = None
) -> LegacyImportPreview:
    """Report what an import of ``source`` would do without writing anything.

    No database, directory, original file, or WAL sidecar is created or modified.
    ``already_imported`` is ``True`` or ``False`` only when an existing database
    could be read read-only; otherwise it is ``None`` (unknown), never a guess.
    """
    source_root = Path(source).expanduser().resolve()
    discovered, scan_conflicts = _discover_run_roots(source_root)
    database = _resolve_database_path(database_path)
    status, imported_roots = _probe_database(database)
    database_exists = status != "absent"
    database_readable = status != "unknown"
    if database_exists and not database_readable:
        # An existing database whose import status cannot be validated read-only
        # is reported explicitly rather than silently treated as empty.
        scan_conflicts.append("database_status_unknown")
    runs: list[LegacyRunPreview] = []
    for root in discovered:
        run = _parse_run_root(root, source_root)
        already_imported = str(run.root) in imported_roots if database_readable else None
        runs.append(_to_preview(run, already_imported))
    return LegacyImportPreview(
        source=str(source_root),
        database_path=str(database),
        database_exists=database_exists,
        database_readable=database_readable,
        runs=tuple(runs),
        scan_conflicts=tuple(scan_conflicts),
    )


def _prepare_database_location(database_path: Path) -> None:
    """Create or verify the private directory an import writes into.

    The default home gets its full managed layout; an explicit path gets only its
    own private parent directory. An existing directory that is group- or
    world-accessible is rejected before any plaintext database is created.
    """
    default_path = history_database_path().absolute()
    if database_path == default_path:
        ensure_history_home(database_path.parent)
    else:
        ensure_private_directory(database_path.parent, parents=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _log_conflict(run: _LegacyRun) -> None:
    _logger.warning(
        "history_conflict root=%s codes=%s",
        _conflict_log_label(run),
        ",".join(run.conflicts) or "none",
    )


def _conflict_log_label(run: _LegacyRun) -> str:
    """Return a content-free label for a ``history_conflict`` log event.

    A bounded run ID can still be a short signed URL, and the run root name can
    carry the same secret, so a label with a URL scheme, query, fragment, or
    user-info (or a secret-shaped token) is replaced by a short digest. A plain
    label such as ``prod`` is kept so the event stays actionable.
    """
    candidate = _bounded_text(run.run_id or run.root.name, MAX_LABEL_LENGTH)
    if candidate is None:
        return "unknown"
    if any(marker in candidate for marker in _UNSAFE_LOG_LABEL_MARKERS):
        return _label_digest(candidate)
    if _is_sensitive_string(candidate):
        return _label_digest(candidate)
    return candidate


def _label_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _import_run(repository: HistoryRepository, run: _LegacyRun) -> LegacyRunImportResult:
    """Insert one run and all its entities in a single caller-managed transaction."""
    parts = 0
    attempts = 0
    artifacts = 0
    text_sources = 0
    with repository.transaction():
        record, created = repository.create_legacy_run(
            operation=run.operation,
            run_root=str(run.root),
            legacy_source_root=str(run.root),
            status=run.status,
            user_label=run.run_id,
            config_snapshot=run.config_snapshot,
        )
        if not created:
            return LegacyRunImportResult(
                source_root=str(run.root),
                run_uuid=record.run_uuid,
                created=False,
                parts=0,
                attempts=0,
                artifacts=0,
                text_sources=0,
                conflicts=run.conflicts,
            )

        if run.script_text is not None:
            repository.add_text_source(
                record.run_uuid,
                kind=TEXT_KIND_TTS_SCRIPT,
                origin=LEGACY_ORIGIN,
                content=run.script_text,
                content_hash=_hash_text(run.script_text),
                text_completeness=TEXT_COMPLETENESS_COMPLETE,
            )
            text_sources += 1
        elif run.script_hash is not None:
            repository.add_text_source(
                record.run_uuid,
                kind=TEXT_KIND_TTS_SCRIPT,
                origin=LEGACY_ORIGIN,
                content=None,
                content_hash=run.script_hash,
                text_completeness=TEXT_COMPLETENESS_INCOMPLETE,
            )
            text_sources += 1
        if run.style_prompt:
            repository.add_text_source(
                record.run_uuid,
                kind=TEXT_KIND_TTS_DIRECTION,
                origin=LEGACY_ORIGIN,
                content=run.style_prompt,
                content_hash=_hash_text(run.style_prompt),
                text_completeness=TEXT_COMPLETENESS_COMPLETE,
            )
            text_sources += 1

        part_uuids: dict[int, str] = {}
        for chunk in run.chunks:
            part = repository.add_part(
                record.run_uuid,
                position=chunk.number,
                prepared_text=chunk.text,
                voice=chunk.voice or run.voice,
                fingerprint=chunk.text_hash,
                stage="completed",
            )
            parts += 1
            part_uuids[chunk.number] = part.part_uuid
            if chunk.text is not None:
                repository.add_text_source(
                    record.run_uuid,
                    kind=TEXT_KIND_TTS_SCRIPT,
                    origin=LEGACY_ORIGIN,
                    content=chunk.text,
                    part_uuid=part.part_uuid,
                    content_hash=chunk.text_hash or _hash_text(chunk.text),
                    text_completeness=TEXT_COMPLETENESS_COMPLETE,
                )
            else:
                repository.add_text_source(
                    record.run_uuid,
                    kind=TEXT_KIND_TTS_SCRIPT,
                    origin=LEGACY_ORIGIN,
                    content=None,
                    part_uuid=part.part_uuid,
                    content_hash=chunk.text_hash,
                    text_completeness=TEXT_COMPLETENESS_INCOMPLETE,
                )
            text_sources += 1

        if run.has_chunk_cost:
            for chunk in run.chunks:
                repository.add_attempt(
                    record.run_uuid,
                    call_type="tts_chunk",
                    part_uuid=part_uuids[chunk.number],
                    provider=run.provider,
                    model=run.model,
                    status=run.status,
                    cost=chunk.cost,
                )
                attempts += 1
        elif run.run_total_cost.amount is not None:
            # Only when no per-chunk price exists: a stored total is imported
            # once, so a per-chunk charge is never counted twice.
            repository.add_attempt(
                record.run_uuid,
                call_type="tts_run_total",
                provider=run.provider,
                model=run.model,
                status=run.status,
                cost=run.run_total_cost,
            )
            attempts += 1

        for artifact in run.artifacts:
            part_uuid = (
                part_uuids[artifact.part_number]
                if artifact.part_number is not None and artifact.part_number in part_uuids
                else None
            )
            repository.add_artifact(
                record.run_uuid,
                role=artifact.role,
                path_kind=PATH_KIND_EXTERNAL_ABSOLUTE,
                path=str(artifact.path),
                part_uuid=part_uuid,
                mime=artifact.mime,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                availability=AVAILABILITY_PRESENT if artifact.available else AVAILABILITY_MISSING,
            )
            artifacts += 1

    _logger.info(
        "history_imported run_uuid=%s parts=%d attempts=%d artifacts=%d text_sources=%d",
        record.run_uuid,
        parts,
        attempts,
        artifacts,
        text_sources,
    )
    return LegacyRunImportResult(
        source_root=str(run.root),
        run_uuid=record.run_uuid,
        created=True,
        parts=parts,
        attempts=attempts,
        artifacts=artifacts,
        text_sources=text_sources,
        conflicts=run.conflicts,
    )


def import_legacy_runs(
    source: Path | str, *, database_path: Path | str | None = None
) -> LegacyImportResult:
    """Import every importable legacy run under ``source`` into the database.

    A repeated import of the same source root converges on the single existing
    run without duplicating parts, attempts, costs, or text sources. A run whose
    identity fails closed is reported with no run UUID and no writes. A newer
    schema than this binary knows raises before any write, so the existing
    database is left untouched.
    """
    source_root = Path(source).expanduser().resolve()
    database_path_resolved = _resolve_database_path(database_path)
    discovered, _scan_conflicts = _discover_run_roots(source_root)
    results: list[LegacyRunImportResult] = []

    # Validate the existing database read-only before creating any directory or
    # file. A database this binary must reject (foreign, corrupt, or a newer
    # schema) therefore leaves the filesystem exactly as it was.
    status, _imported_roots = _probe_database(database_path_resolved)
    if status == "unknown":
        # migrate() validates and may reject this database. Refuse plaintext
        # history in an insecure existing container, but create no managed
        # directory on this path.
        ensure_private_directory(database_path_resolved.parent)
    else:
        _prepare_database_location(database_path_resolved)
    database = HistoryDatabase(database_path_resolved)
    try:
        database.migrate()
        repository = HistoryRepository(database)
        for root in discovered:
            run = _parse_run_root(root, source_root)
            if run.conflicts:
                _log_conflict(run)
            if not run.importable:
                results.append(
                    LegacyRunImportResult(
                        source_root=str(run.root),
                        run_uuid=None,
                        created=False,
                        parts=0,
                        attempts=0,
                        artifacts=0,
                        text_sources=0,
                        conflicts=run.conflicts,
                    )
                )
                continue
            results.append(_import_run(repository, run))
    finally:
        database.close()

    return LegacyImportResult(
        source=str(source_root),
        database_path=str(database_path_resolved),
        runs=tuple(results),
    )


__all__ = [
    "LEGACY_OPERATION",
    "LEGACY_ORIGIN",
    "LegacyChunkPreview",
    "LegacyImportPreview",
    "LegacyImportResult",
    "LegacyRunImportResult",
    "LegacyRunPreview",
    "import_legacy_runs",
    "preview_legacy_import",
]
