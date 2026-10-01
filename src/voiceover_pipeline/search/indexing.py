"""Offline FTS5 index maintenance rebuilt from SQLite alone.

Plan section 8 stage S08 keeps the lexical index derived: canonical text lives
in ``text_sources`` and explicit part vibes in ``parts.vibe_effective``. Every
table this module writes -- ``search_chunks``,
``search_fts``, ``search_index_state``, ``search_index_pending`` -- can be dropped
and rebuilt from SQLite with no model, key, network, or user-file scan. The
transcription/synthesis parts, audio, and cost are never touched.

Guarantees:

* A normal text-source save indexes immediately through
  :func:`index_committed_text_sources`, called by the repository after the
  canonical transaction commits. If that derived write fails, the canonical text
  stays committed and the source is recorded in ``search_index_pending`` so
  ``index build`` retries only what is missing.
* Every write runs in one ``BEGIN IMMEDIATE`` transaction, so a build, rebuild, or
  incremental retry is atomic and recoverable: a failure leaves the previous index
  unchanged rather than half-rewritten, and a concurrent writer cannot interleave.
* Indexing a source is idempotent: its existing chunks and FTS rows are removed
  before the new ones are written, and ``index build`` only touches sources that
  are not yet indexed, so a repeat build adds nothing.
* Truthful counts: :func:`index_status` derives indexed, pending, private
  (excluded), and incomplete counts from the database, and never claims a corpus
  complete while a source is pending or an imported text only survives as a hash.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

from ..asr_timing_map import ObservedTimingSpan, chunk_time_range, observed_spans_from_snapshot
from ..history.database import utc_now
from ..history.native_asr import NATIVE_ASR_ORIGIN
from ..history.repository import TEXT_KIND_ASR_TRANSCRIPT, TEXT_KIND_TTS_DIRECTION
from .lexical import (
    CHUNKER_VERSION,
    KIND_RUN_LABEL,
    PRIVATE_TEXT_KINDS,
    ROLE_BY_KIND,
    ROLE_DIRECTIONS,
    ROLE_LABEL,
    chunk_text,
    is_indexable_kind,
    normalize_search_text,
    role_for_kind,
    text_hash,
)

_SOURCE_COLUMNS = "text_source_uuid, run_uuid, part_uuid, artifact_uuid, kind, origin, content"
_INDEXABLE_KIND_VALUES = tuple(ROLE_BY_KIND)
_INDEXABLE_KIND_PLACEHOLDERS = ", ".join("?" for _ in _INDEXABLE_KIND_VALUES)


@contextmanager
def _immediate(connection: sqlite3.Connection) -> Iterator[None]:
    """Run a derived-index write as one short immediate transaction.

    Fails closed if the caller already opened a transaction, so a derived write
    never silently joins and rolls back with an unrelated canonical write.
    """
    if connection.in_transaction:
        raise sqlite3.OperationalError("the lexical index writer must own its transaction")
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def index_committed_text_sources(
    connection: sqlite3.Connection,
    text_source_uuids: Sequence[str],
    *,
    run_uuids: Sequence[str] = (),
) -> int:
    """Index the named, already-committed text sources and run labels.

    Called by the repository after a canonical save commits. Each source is
    re-indexed by deleting its existing chunks first, so a retry is idempotent, and
    its ``search_index_pending`` row is cleared on success. ``run_uuids`` names
    runs whose label should also be indexed even when they carry no text source, so
    a labeled run is searchable by its label as soon as it is created. Returns the
    number of chunks written. Raises on a derived write failure so the caller can
    record the source as pending; the canonical text is already durable.
    """
    now = utc_now()
    written = 0
    with _immediate(connection):
        runs_to_label: set[str] = set(run_uuids)
        for text_source_uuid in text_source_uuids:
            row = connection.execute(
                f"SELECT {_SOURCE_COLUMNS} FROM text_sources WHERE text_source_uuid = ?",
                (text_source_uuid,),
            ).fetchone()
            if row is None:
                continue
            runs_to_label.add(row["run_uuid"])
            if _is_unused_style_source(connection, row):
                # Old speech-parts snapshots saved the generic Polza default as a
                # direction although the adapter never sent it. Keep the canonical
                # source for snapshot identity, but remove misleading FTS rows.
                _delete_source_chunks(connection, text_source_uuid)
                connection.execute(
                    "DELETE FROM search_index_pending WHERE text_source_uuid = ?",
                    (text_source_uuid,),
                )
                continue
            if not _is_indexable_source(row["kind"], row["content"]):
                connection.execute(
                    "DELETE FROM search_index_pending WHERE text_source_uuid = ?",
                    (text_source_uuid,),
                )
                continue
            written += _replace_source_chunks(connection, row, now)
            connection.execute(
                "DELETE FROM search_index_pending WHERE text_source_uuid = ?",
                (text_source_uuid,),
            )
        for run_uuid in runs_to_label:
            _ensure_run_label_chunks(connection, run_uuid, now)
            # Speech-parts directions are canonical columns on parts, not separate
            # text_sources. Derive their opt-in search chunks after that run's
            # normal text-source save commits, without changing paid history.
            for part in _unindexed_part_directions(connection, run_uuid=run_uuid):
                written += _replace_part_direction_chunks(connection, part, now)
        _ensure_state_chunker_version(connection)
    return written


def mark_text_sources_pending(
    connection: sqlite3.Connection, text_source_uuids: Sequence[str], *, reason: str
) -> None:
    """Record that a source's derived index could not be written.

    Idempotent per source: an existing pending row is refreshed. Raises when the
    pending table itself is unavailable; the caller treats that as a warning only,
    because the canonical text is already committed.
    """
    now = utc_now()
    with _immediate(connection):
        for text_source_uuid in text_source_uuids:
            connection.execute(
                "INSERT INTO search_index_pending (text_source_uuid, reason, created_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(text_source_uuid) DO UPDATE SET "
                "reason = excluded.reason, created_at = excluded.created_at",
                (text_source_uuid, reason, now),
            )


def build_index(connection: sqlite3.Connection) -> dict[str, Any]:
    """Index missing labels, text sources and part vibes; drop unused old styles.

    One atomic transaction. Only absent sources/part directions are added, so a
    repeat build is a no-op; historical speech-parts defaults never sent to a
    provider are removed from the derived index, not from canonical history.
    """
    now = utc_now()
    sources_written = 0
    part_directions_written = 0
    chunks_written = 0
    with _immediate(connection):
        inactive_style_chunks_removed = _purge_unused_style_chunks(connection)
        inactive_style_pending_removed = _purge_unused_style_pending(connection)
        for run_uuid in _runs_missing_label_chunk(connection):
            chunks_written += _ensure_run_label_chunks(connection, run_uuid, now)
        for row in _unindexed_sources(connection):
            chunks_written += _replace_source_chunks(connection, row, now)
            connection.execute(
                "DELETE FROM search_index_pending WHERE text_source_uuid = ?",
                (row["text_source_uuid"],),
            )
            sources_written += 1
        for part in _unindexed_part_directions(connection):
            chunks_written += _replace_part_direction_chunks(connection, part, now)
            part_directions_written += 1
        _record_build(
            connection,
            mode="build",
            sources=sources_written + part_directions_written,
            chunks=chunks_written,
        )
        pending = _pending_count(connection)
    return {
        "mode": "build",
        "sources_written": sources_written,
        "part_directions_written": part_directions_written,
        "inactive_style_chunks_removed": inactive_style_chunks_removed,
        "inactive_style_pending_removed": inactive_style_pending_removed,
        "chunks_written": chunks_written,
        "pending_remaining": pending,
    }


def rebuild_index(connection: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild the derived index from ``text_sources``, ``parts`` and ``runs`` only.

    One atomic transaction, so a failure leaves the previous index intact. Every
    pending marker is cleared because the rebuild re-attempts every source, and the
    chunker version is stamped so a later status can tell the index matches the
    current algorithm. Returns the rebuilt source and chunk counts.
    """
    now = utc_now()
    with _immediate(connection):
        connection.execute("DELETE FROM search_chunks")  # mirror trigger clears linked FTS rows
        connection.execute("DELETE FROM search_fts")  # clear any historical orphan rows as well
        connection.execute("DELETE FROM search_index_pending")
        sources_written = 0
        part_directions_written = 0
        chunks_written = 0
        for run_uuid in _runs_with_label(connection):
            chunks_written += _ensure_run_label_chunks(connection, run_uuid, now)
        for row in _indexable_sources(connection):
            chunks_written += _replace_source_chunks(connection, row, now)
            sources_written += 1
        for part in _unindexed_part_directions(connection):
            chunks_written += _replace_part_direction_chunks(connection, part, now)
            part_directions_written += 1
        _record_build(
            connection,
            mode="rebuild",
            sources=sources_written + part_directions_written,
            chunks=chunks_written,
        )
    return {
        "mode": "rebuild",
        "sources_written": sources_written,
        "part_directions_written": part_directions_written,
        "inactive_style_chunks_removed": 0,  # full rebuild clears every old derived row
        "chunks_written": chunks_written,
    }


def index_status(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return truthful counts of the derived index from the database alone.

    ``complete`` is true only when the index exists, nothing is pending, every
    indexable source and part vibe is indexed, no unused style remains, and no
    source is stored as an incomplete hash --
    so an old imported corpus that only kept a hash is reported ``complete: false``
    with its ``sources_incomplete`` count instead of being claimed searchable.
    ``needs_rebuild`` is true when the index is missing or was built by another
    chunking version.
    """
    if not _table_exists(connection, "search_chunks"):
        return {
            "available": False,
            "schema_version": _schema_version(connection),
            "chunker_version": CHUNKER_VERSION,
        }
    state = connection.execute(
        "SELECT chunker_version, last_build_at, last_build_mode, last_build_sources, "
        "last_build_chunks FROM search_index_state WHERE id = 1"
    ).fetchone()
    sources_total = _scalar(connection, "SELECT COUNT(*) FROM text_sources")
    sources_indexable = _count_kinds(
        connection, ROLE_BY_KIND, require_content=True, exclude_unused_style=True
    )
    sources_private = _count_kinds(connection, PRIVATE_TEXT_KINDS)
    sources_incomplete = _scalar(
        connection,
        "SELECT COUNT(*) FROM text_sources WHERE content IS NULL OR TRIM(content) = ''",
    )
    sources_indexed = _scalar(
        connection,
        "SELECT COUNT(DISTINCT c.text_source_uuid) FROM search_chunks AS c "
        "JOIN text_sources AS s ON s.text_source_uuid = c.text_source_uuid "
        f"WHERE {_active_source_clause('s')}",
        (TEXT_KIND_TTS_DIRECTION,),
    )
    inactive_style_chunks = _count_unused_style_chunks(connection)
    indexed_chunks = _scalar(connection, "SELECT COUNT(*) FROM search_chunks")
    part_directions_total = _scalar(
        connection,
        "SELECT COUNT(*) FROM parts WHERE vibe_effective IS NOT NULL "
        "AND TRIM(vibe_effective) <> ''",
    )
    part_directions_indexed = _scalar(
        connection,
        "SELECT COUNT(*) FROM parts AS p WHERE p.vibe_effective IS NOT NULL "
        "AND TRIM(p.vibe_effective) <> '' AND EXISTS ("
        "SELECT 1 FROM search_chunks AS c WHERE c.part_uuid = p.part_uuid "
        "AND c.text_source_uuid IS NULL AND c.kind = ? AND c.role = ?)",
        (TEXT_KIND_TTS_DIRECTION, ROLE_DIRECTIONS),
    )
    part_directions_missing = part_directions_total - part_directions_indexed
    pending = _pending_count(connection)
    label_runs = _scalar(
        connection,
        "SELECT COUNT(*) FROM runs WHERE user_label IS NOT NULL AND TRIM(user_label) <> ''",
    )
    # A failed derived write for a label-only run has no text source to record in
    # search_index_pending; count unindexed labels directly so `complete` never
    # hides one. The next `index build` already retries these missing labels.
    labels_missing = len(_runs_missing_label_chunk(connection))
    built_chunker_version = None if state is None else int(state["chunker_version"])
    complete = (
        pending == 0
        and labels_missing == 0
        and sources_incomplete == 0
        and sources_indexed == sources_indexable
        and part_directions_missing == 0
        and inactive_style_chunks == 0
        and built_chunker_version == CHUNKER_VERSION
    )
    return {
        "available": True,
        "schema_version": _schema_version(connection),
        "chunker_version": CHUNKER_VERSION,
        "built_chunker_version": built_chunker_version,
        "last_build_at": None if state is None else state["last_build_at"],
        "last_build_mode": None if state is None else state["last_build_mode"],
        "last_build_sources": 0 if state is None else int(state["last_build_sources"]),
        "last_build_chunks": 0 if state is None else int(state["last_build_chunks"]),
        "sources_total": sources_total,
        "sources_indexable": sources_indexable,
        "sources_indexed": sources_indexed,
        "part_directions_total": part_directions_total,
        "part_directions_indexed": part_directions_indexed,
        "part_directions_missing": part_directions_missing,
        "inactive_style_chunks": inactive_style_chunks,
        "sources_pending": pending,
        "sources_private_excluded": sources_private,
        "sources_incomplete": sources_incomplete,
        "label_runs": label_runs,
        "labels_missing": labels_missing,
        "indexed_chunks": indexed_chunks,
        "complete": complete,
        "needs_rebuild": built_chunker_version != CHUNKER_VERSION,
    }


# -- internals ----------------------------------------------------------------


def _is_indexable_source(kind: str, content: str | None) -> bool:
    """Whether a source contributes chunks: an indexable kind with real text."""
    return is_indexable_kind(kind) and content is not None and content.strip() != ""


def _unused_style_clause(alias: str) -> str:
    """An old run-level direction unused by an explicit speech-parts adapter.

    Only speech-parts persists a non-NULL ``vibe_effective`` on its parts, even
    when the authored vibe is empty. Its generic run-level style prompt was never
    transmitted; preserve that canonical source for native-view integrity, but
    exclude it from the derived lexical index.
    """
    return (
        f"{alias}.kind = ? AND {alias}.part_uuid IS NULL "
        "AND EXISTS (SELECT 1 FROM parts AS p "
        f"WHERE p.run_uuid = {alias}.run_uuid AND p.vibe_effective IS NOT NULL)"
    )


def _active_source_clause(alias: str) -> str:
    return f"NOT ({_unused_style_clause(alias)})"


def _is_unused_style_source(connection: sqlite3.Connection, row: sqlite3.Row) -> bool:
    if row["kind"] != TEXT_KIND_TTS_DIRECTION or row["part_uuid"] is not None:
        return False
    return (
        connection.execute(
            "SELECT 1 FROM parts WHERE run_uuid = ? AND vibe_effective IS NOT NULL LIMIT 1",
            (row["run_uuid"],),
        ).fetchone()
        is not None
    )


def _count_unused_style_chunks(connection: sqlite3.Connection) -> int:
    return _scalar(
        connection,
        "SELECT COUNT(*) FROM search_chunks AS c "
        "JOIN text_sources AS s ON s.text_source_uuid = c.text_source_uuid "
        f"WHERE {_unused_style_clause('s')}",
        (TEXT_KIND_TTS_DIRECTION,),
    )


def _purge_unused_style_chunks(connection: sqlite3.Connection) -> int:
    cursor = connection.execute(
        "DELETE FROM search_chunks WHERE text_source_uuid IN ("
        "SELECT s.text_source_uuid FROM text_sources AS s "
        f"WHERE {_unused_style_clause('s')})",
        (TEXT_KIND_TTS_DIRECTION,),
    )
    return int(cursor.rowcount)


def _purge_unused_style_pending(connection: sqlite3.Connection) -> int:
    cursor = connection.execute(
        "DELETE FROM search_index_pending WHERE text_source_uuid IN ("
        "SELECT s.text_source_uuid FROM text_sources AS s "
        f"WHERE {_unused_style_clause('s')})",
        (TEXT_KIND_TTS_DIRECTION,),
    )
    return int(cursor.rowcount)


def _replace_source_chunks(connection: sqlite3.Connection, row: sqlite3.Row, now: str) -> int:
    """Replace one source's chunks and mirror them into the FTS index.

    The original chunk text is stored in ``search_chunks``; only the derived
    ``normalized_text`` column of ``search_fts`` is folded. Existing rows are
    removed first, so the operation is idempotent. The chunk's ``start_ms`` /
    ``end_ms`` are derived from the observed timing spans the run stored in its
    canonical snapshot, intersected with the chunk's character window, so a
    rebuild from SQLite alone reproduces the exact same ranges.
    """
    text_source_uuid = row["text_source_uuid"]
    _delete_source_chunks(connection, text_source_uuid)
    role = role_for_kind(row["kind"])
    if role is None:
        return 0
    spans = _observed_spans_for_run(connection, row)
    written = 0
    for chunk_index, (char_start, char_end, chunk) in enumerate(chunk_text(row["content"] or "")):
        start_ms, end_ms = chunk_time_range(
            spans, char_start=char_start, char_end=char_end, chunk_text=chunk
        )
        written += _insert_chunk(
            connection,
            text_source_uuid=text_source_uuid,
            run_uuid=row["run_uuid"],
            part_uuid=row["part_uuid"],
            artifact_uuid=row["artifact_uuid"],
            kind=row["kind"],
            role=role,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            chunk=chunk,
            start_ms=start_ms,
            end_ms=end_ms,
            now=now,
        )
    return written


def _replace_part_direction_chunks(
    connection: sqlite3.Connection, part: sqlite3.Row, now: str
) -> int:
    """Derive direction FTS rows from committed parts without rewriting paid history.

    A NULL text_source_uuid is intentional: the canonical instruction is the
    immutable ``parts.vibe_effective`` column. This is the same derived-only
    representation used by run-label chunks. The part UUID preserves attribution.
    """
    connection.execute(
        "DELETE FROM search_chunks WHERE text_source_uuid IS NULL "
        "AND kind = ? AND role = ? AND part_uuid = ?",
        (TEXT_KIND_TTS_DIRECTION, ROLE_DIRECTIONS, part["part_uuid"]),
    )
    written = 0
    for index, (start, end, chunk) in enumerate(chunk_text(part["vibe_effective"])):
        written += _insert_chunk(
            connection,
            text_source_uuid=None,
            run_uuid=part["run_uuid"],
            part_uuid=part["part_uuid"],
            artifact_uuid=None,
            kind=TEXT_KIND_TTS_DIRECTION,
            role=ROLE_DIRECTIONS,
            chunk_index=index,
            char_start=start,
            char_end=end,
            chunk=chunk,
            start_ms=None,
            end_ms=None,
            now=now,
        )
    return written


def _unindexed_part_directions(
    connection: sqlite3.Connection, *, run_uuid: str | None = None
) -> list[sqlite3.Row]:
    """Find canonical part vibes whose derived direction chunks are absent."""
    sql = (
        "SELECT p.part_uuid, p.run_uuid, p.vibe_effective FROM parts AS p "
        "WHERE p.vibe_effective IS NOT NULL AND TRIM(p.vibe_effective) <> '' "
        "AND NOT EXISTS (SELECT 1 FROM search_chunks AS c "
        "WHERE c.part_uuid = p.part_uuid AND c.text_source_uuid IS NULL "
        "AND c.kind = ? AND c.role = ?)"
    )
    parameters: list[str] = [TEXT_KIND_TTS_DIRECTION, ROLE_DIRECTIONS]
    if run_uuid is not None:
        sql += " AND p.run_uuid = ?"
        parameters.append(run_uuid)
    sql += " ORDER BY p.run_uuid, p.position"
    return list(connection.execute(sql, parameters))


def _observed_spans_for_run(
    connection: sqlite3.Connection, source: sqlite3.Row
) -> tuple[ObservedTimingSpan, ...]:
    """Read only spans bound to this exact canonical ASR transcript source.

    A run may also carry scripts, instructions, or an independently imported
    transcript. Neither a matching character offset nor a matching run UUID is
    evidence that such text was recognized at the ASR transcript's times.
    """
    if source["kind"] != TEXT_KIND_ASR_TRANSCRIPT or source["origin"] != NATIVE_ASR_ORIGIN:
        return ()
    content = source["content"]
    if not isinstance(content, str):
        return ()
    run = connection.execute(
        "SELECT config_snapshot FROM runs WHERE run_uuid = ?", (source["run_uuid"],)
    ).fetchone()
    if run is None or run["config_snapshot"] is None:
        return ()
    try:
        snapshot = json.loads(run["config_snapshot"])
    except (TypeError, ValueError):
        return ()
    if not isinstance(snapshot, dict) or snapshot.get("operation_origin") != NATIVE_ASR_ORIGIN:
        return ()
    observed = snapshot.get("observed_spans")
    if (
        not isinstance(observed, dict)
        or observed.get("text_source_uuid") != source["text_source_uuid"]
    ):
        return ()
    return observed_spans_from_snapshot(snapshot, source_text=content)


def _ensure_run_label_chunks(connection: sqlite3.Connection, run_uuid: str, now: str) -> int:
    """Add a run's label chunks when it has a label and no label chunk yet.

    Returns the number of chunks written, so a build or rebuild can report the
    truthful derived document count including run labels.
    """
    row = connection.execute(
        "SELECT user_label FROM runs WHERE run_uuid = ?", (run_uuid,)
    ).fetchone()
    if row is None:
        return 0
    label = row["user_label"]
    if label is None or label.strip() == "":
        return 0
    existing = connection.execute(
        "SELECT 1 FROM search_chunks WHERE run_uuid = ? AND role = ? LIMIT 1",
        (run_uuid, ROLE_LABEL),
    ).fetchone()
    if existing is not None:
        return 0
    _delete_label_chunks(connection, run_uuid)
    written = 0
    for chunk_index, (char_start, char_end, chunk) in enumerate(chunk_text(label)):
        written += _insert_chunk(
            connection,
            text_source_uuid=None,
            run_uuid=run_uuid,
            part_uuid=None,
            artifact_uuid=None,
            kind=KIND_RUN_LABEL,
            role=ROLE_LABEL,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            chunk=chunk,
            start_ms=None,
            end_ms=None,
            now=now,
        )
    return written


def _insert_chunk(
    connection: sqlite3.Connection,
    *,
    text_source_uuid: str | None,
    run_uuid: str,
    part_uuid: str | None,
    artifact_uuid: str | None,
    kind: str,
    role: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    chunk: str,
    start_ms: int | None,
    end_ms: int | None,
    now: str,
) -> int:
    """Insert one chunk row and its FTS row, sharing the chunk row id."""
    cursor = connection.execute(
        "INSERT INTO search_chunks ("
        "text_source_uuid, run_uuid, part_uuid, artifact_uuid, kind, role, chunk_index, "
        "char_start, char_end, text, text_hash, start_ms, end_ms, chunker_version, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            text_source_uuid,
            run_uuid,
            part_uuid,
            artifact_uuid,
            kind,
            role,
            chunk_index,
            char_start,
            char_end,
            chunk,
            text_hash(chunk),
            start_ms,
            end_ms,
            CHUNKER_VERSION,
            now,
        ),
    )
    chunk_id = int(cursor.lastrowid or 0)
    connection.execute(
        "INSERT INTO search_fts (rowid, normalized_text) VALUES (?, ?)",
        (chunk_id, normalize_search_text(chunk)),
    )
    return 1


def _delete_source_chunks(connection: sqlite3.Connection, text_source_uuid: str) -> None:
    # The migration trigger removes the matching FTS row for every chunk,
    # including canonical ON DELETE CASCADE and ordinary source replacement.
    connection.execute("DELETE FROM search_chunks WHERE text_source_uuid = ?", (text_source_uuid,))


def _delete_label_chunks(connection: sqlite3.Connection, run_uuid: str) -> None:
    connection.execute(
        "DELETE FROM search_chunks WHERE run_uuid = ? AND role = ?", (run_uuid, ROLE_LABEL)
    )


def _indexable_sources(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            f"SELECT {_SOURCE_COLUMNS} FROM text_sources AS s "
            f"WHERE s.kind IN ({_INDEXABLE_KIND_PLACEHOLDERS}) "
            "AND s.content IS NOT NULL AND TRIM(s.content) <> '' "
            f"AND {_active_source_clause('s')}",
            (*_INDEXABLE_KIND_VALUES, TEXT_KIND_TTS_DIRECTION),
        )
    )


def _unindexed_sources(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            f"SELECT {_SOURCE_COLUMNS} FROM text_sources AS s "
            f"WHERE s.kind IN ({_INDEXABLE_KIND_PLACEHOLDERS}) "
            "AND s.content IS NOT NULL AND TRIM(s.content) <> '' "
            f"AND {_active_source_clause('s')} "
            "AND NOT EXISTS ("
            "SELECT 1 FROM search_chunks AS c WHERE c.text_source_uuid = s.text_source_uuid"
            ")",
            (*_INDEXABLE_KIND_VALUES, TEXT_KIND_TTS_DIRECTION),
        )
    )


def _runs_with_label(connection: sqlite3.Connection) -> list[str]:
    return [
        row["run_uuid"]
        for row in connection.execute(
            "SELECT run_uuid FROM runs WHERE user_label IS NOT NULL AND TRIM(user_label) <> ''"
        )
    ]


def _runs_missing_label_chunk(connection: sqlite3.Connection) -> list[str]:
    return [
        row["run_uuid"]
        for row in connection.execute(
            "SELECT run_uuid FROM runs AS r "
            "WHERE r.user_label IS NOT NULL AND TRIM(r.user_label) <> '' "
            "AND NOT EXISTS ("
            "SELECT 1 FROM search_chunks AS c WHERE c.run_uuid = r.run_uuid AND c.role = ?"
            ")",
            (ROLE_LABEL,),
        )
    ]


def _record_build(connection: sqlite3.Connection, *, mode: str, sources: int, chunks: int) -> None:
    connection.execute(
        "INSERT INTO search_index_state ("
        "id, chunker_version, last_build_at, last_build_mode, last_build_sources, "
        "last_build_chunks) VALUES (1, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "chunker_version = excluded.chunker_version, "
        "last_build_at = excluded.last_build_at, "
        "last_build_mode = excluded.last_build_mode, "
        "last_build_sources = excluded.last_build_sources, "
        "last_build_chunks = excluded.last_build_chunks",
        (CHUNKER_VERSION, utc_now(), mode, sources, chunks),
    )


def _ensure_state_chunker_version(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO search_index_state (id, chunker_version) VALUES (1, ?) "
        "ON CONFLICT(id) DO NOTHING",
        (CHUNKER_VERSION,),
    )


def _pending_count(connection: sqlite3.Connection) -> int:
    return _scalar(connection, "SELECT COUNT(*) FROM search_index_pending")


def _count_kinds(
    connection: sqlite3.Connection,
    kinds: Iterable[str],
    *,
    require_content: bool = False,
    exclude_unused_style: bool = False,
) -> int:
    if not kinds:
        return 0
    placeholders = ", ".join("?" for _ in kinds)
    condition = f"s.kind IN ({placeholders})"
    parameters = list(kinds)
    if require_content:
        condition += " AND s.content IS NOT NULL AND TRIM(s.content) <> ''"
    if exclude_unused_style:
        condition += f" AND {_active_source_clause('s')}"
        parameters.append(TEXT_KIND_TTS_DIRECTION)
    return _scalar(
        connection,
        f"SELECT COUNT(*) FROM text_sources AS s WHERE {condition}",
        parameters,
    )


def _scalar(connection: sqlite3.Connection, sql: str, parameters: Sequence[Any] = ()) -> int:
    return int(connection.execute(sql, parameters).fetchone()[0])


def _schema_version(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, "schema_migrations"):
        return 0
    return _scalar(connection, "SELECT COALESCE(MAX(version), 0) FROM schema_migrations")


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )
