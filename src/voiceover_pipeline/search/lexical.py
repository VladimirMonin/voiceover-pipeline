"""Derived lexical-search form, chunking, and FTS5 query for saved history text.

Plan section 8 stage S08 searches the *saved text linked to audio* rather than the
audio signal: prepared scripts, ASR transcripts, and verification transcripts are
searchable by default, a run's short label is searchable as its own role, and the
director instructions are reachable only through an explicit ``--scope
directions`` so a query like "calm podcast" does not match every file sharing a
vibe.

Derived, never canonical:

* :func:`normalize_search_text` folds case and ``ё``/``Ё`` to ``е``/``Е``. It is a
  lexical fold, not a morphological one, and it only ever writes to the derived
  index and the query: the stored ``text_sources.content`` keeps its original
  bytes, and a returned snippet is sliced from that original text.
* :func:`split_text_chunks` windows long text into overlapping search chunks near
  paragraph/sentence boundaries. It never touches TTS parts, audio, or cost.
* :func:`build_match_expression` turns a user string into a literal FTS5 MATCH
  expression of quoted, normalized word tokens joined by ``AND``. Quotes, hyphens,
  punctuation, and SQL-looking fragments in the input are tokenized away instead
  of becoming MATCH operators or SQL, so a query can neither inject SQL nor raise
  an unchecked syntax error.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from ..history.native_asr import (
    ARTIFACT_ROLE_SOURCE_AUDIO,
    ATTEMPT_CALL_TYPE_ASR,
    ATTEMPT_CALL_TYPE_TIMING,
    ATTEMPT_CALL_TYPE_VERIFY,
    OPERATION_ASR,
    OPERATION_TIMINGS,
    OPERATION_VERIFY,
)
from ..history.paid_transcription import (
    ATTEMPT_CALL_TYPE_PAID_QUALITY,
    ATTEMPT_CALL_TYPE_PAID_TIMING,
    PAID_TRANSCRIPTION_ORIGIN,
)
from ..history.repository import (
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_FINAL_AUDIO,
    ARTIFACT_ROLE_LOCAL_RAW_AUDIO,
    ARTIFACT_ROLE_PAID_RAW_AUDIO,
    ATTEMPT_CALL_TYPE_LOCAL_TTS,
    ATTEMPT_CALL_TYPE_TTS_CHUNK,
    AVAILABILITY_MISSING,
    AVAILABILITY_PRESENT,
    PATH_KIND_EXTERNAL_ABSOLUTE,
    PATH_KIND_MANAGED_RELATIVE,
    RUN_STATUS_COMPLETED,
    TEXT_KIND_ASR_CONTEXT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
)

# The chunking algorithm version stored with the index. Changing the window size,
# overlap, or boundary policy requires a derived rebuild, not a new transcription
# or synthesis, so it is versioned rather than silently mixed.
CHUNKER_VERSION = 1
# Initial policy from plan section 8: about 1200 Unicode characters with a small
# overlap of up to 150, preferably at paragraph/sentence boundaries. These are
# search-layer settings, never fields of a user speech scenario.
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# Roles a search chunk can carry. A result reports the role so a caller can tell a
# spoken script from recognized speech, a director instruction, or a run label.
ROLE_SPEECH = "speech"
ROLE_ASR = "asr"
ROLE_DIRECTIONS = "directions"
ROLE_LABEL = "label"
ROLES = (ROLE_SPEECH, ROLE_ASR, ROLE_DIRECTIONS, ROLE_LABEL)

# Kind a run-label chunk carries, so a result can name it without a text source.
KIND_RUN_LABEL = "run_label"

# The kind -> role map of every indexable text source. A kind absent from this map
# is not indexed at all: the optional ASR prompt (``asr_context``) is a private
# input hint rather than recognized speech, so it never enters the lexical index.
ROLE_BY_KIND: dict[str, str] = {
    TEXT_KIND_TTS_SCRIPT: ROLE_SPEECH,
    TEXT_KIND_ASR_TRANSCRIPT: ROLE_ASR,
    TEXT_KIND_VERIFICATION_TRANSCRIPT: ROLE_ASR,
    TEXT_KIND_TTS_DIRECTION: ROLE_DIRECTIONS,
}
# Private kinds preserved in canonical history but deliberately excluded from the
# derived index, so a prompt the user supplied for ASR is never surfaced by a
# content search or counted as indexed.
PRIVATE_TEXT_KINDS = frozenset({TEXT_KIND_ASR_CONTEXT})

# Scope presets. The default searches speech and recognized text (plus the short
# run label); ``directions`` adds the director instructions; ``all`` is every role.
DEFAULT_SCOPE = "speech"
SCOPE_DIRECTIONS = "directions"
SCOPE_ALL = "all"
SCOPES = (DEFAULT_SCOPE, SCOPE_DIRECTIONS, SCOPE_ALL)
_SCOPE_ROLES: dict[str, frozenset[str] | None] = {
    DEFAULT_SCOPE: frozenset({ROLE_SPEECH, ROLE_ASR, ROLE_LABEL}),
    SCOPE_DIRECTIONS: frozenset({ROLE_SPEECH, ROLE_ASR, ROLE_LABEL, ROLE_DIRECTIONS}),
    SCOPE_ALL: None,
}

# Bounded snippet window and query-token budget so one result row and one query
# stay small regardless of the stored document size.
MAX_SNIPPET_CHARS = 300
MAX_QUERY_TOKENS = 32
_SNIPPET_LEAD = 80

# Word tokens only. ``\w`` under ``re.UNICODE`` keeps Cyrillic and Latin letters
# and digits and drops every FTS operator or SQL metacharacter, so a token is never
# more than a quoted literal inside the MATCH expression.
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
# A boundary character ends a preferred chunk: a sentence terminator or a newline.
_BOUNDARY_PATTERN = re.compile(r"[.!?…\n]")


def normalize_search_text(text: str) -> str:
    """Return the derived search form: lowercase with ``ё``/``Ё`` folded to ``е``/``Е``.

    The original text is never modified; only the FTS index column and the query
    pass through this fold. It is a lexical fold, not a morphological one: word
    forms such as "модель" and "моделями" are not guaranteed to match without
    additional processing, and no English Porter stemmer is applied to a Russian
    corpus.
    """
    return text.replace("Ё", "Е").replace("ё", "е").lower()


def query_tokens(query: str) -> list[str]:
    """Return bounded, normalized word tokens of a user query, in order."""
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(normalize_search_text(query)):
        if token:
            tokens.append(token)
        if len(tokens) >= MAX_QUERY_TOKENS:
            break
    return tokens


def build_match_expression(tokens: Sequence[str]) -> str | None:
    """Return a literal FTS5 MATCH expression from normalized tokens, or ``None``.

    Each token is already only word characters, and it is wrapped in double quotes
    and joined with ``AND``, so the expression contains no FTS operator a caller
    could inject and cannot raise a MATCH syntax error. ``None`` means the query
    held no searchable term.
    """
    if not tokens:
        return None
    return " AND ".join(f'"{token}"' for token in tokens)


def split_text_chunks(text: str) -> list[tuple[int, int]]:
    """Return overlapping ``(char_start, char_end)`` windows over ``text``.

    Long text is split near a paragraph or sentence boundary inside the second
    half of each window; when no boundary exists the window is cut at the hard
    limit. Consecutive chunks overlap by at most :data:`CHUNK_OVERLAP_CHARS` and
    every step advances, so the loop always terminates. Blank text yields no
    chunks.
    """
    length = len(text)
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        hard_end = min(start + CHUNK_MAX_CHARS, length)
        end = hard_end
        if hard_end < length:
            window_start = start + CHUNK_MAX_CHARS // 2
            boundary = _last_boundary(text, window_start, hard_end)
            if boundary is not None:
                end = boundary
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= length:
            break
        next_start = end - CHUNK_OVERLAP_CHARS
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return spans


def _last_boundary(text: str, window_start: int, hard_end: int) -> int | None:
    """Return the offset just after the last boundary character in a window."""
    last: int | None = None
    for match in _BOUNDARY_PATTERN.finditer(text, window_start, hard_end):
        last = match.end()
    return last


def chunk_text(text: str) -> list[tuple[int, int, str]]:
    """Return ``(char_start, char_end, chunk_text)`` for every nonblank chunk."""
    return [(start, end, text[start:end]) for start, end in split_text_chunks(text)]


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of one chunk's exact text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_snippet(text: str, tokens: Sequence[str], *, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """Return a bounded snippet of ``text`` around the earliest matching token.

    The snippet is sliced from the original text so it keeps its original case and
    ``ё``; matching runs over the derived fold with an explicit offset map, so a
    fold that changes length cannot shift the slice. A window with no matched token
    (for example a match on a label the caller does not see) starts at the text
    beginning.
    """
    if len(text) <= max_chars:
        return text
    normalized, offsets = _normalized_with_offsets(text)
    position: int | None = None
    for token in tokens:
        found = normalized.find(token)
        if found != -1 and (position is None or found < position):
            position = found
    if position is None:
        return text[:max_chars] + "…"
    original_position = offsets[position]
    snippet_start = max(0, original_position - _SNIPPET_LEAD)
    snippet_end = min(len(text), snippet_start + max_chars)
    snippet = text[snippet_start:snippet_end]
    if snippet_start > 0:
        snippet = "…" + snippet
    if snippet_end < len(text):
        snippet = snippet + "…"
    return snippet


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    """Return the folded text plus a folded-index -> original-index offset map."""
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        folded = normalize_search_text(character)
        for folded_character in folded:
            characters.append(folded_character)
            offsets.append(index)
    return "".join(characters), offsets


def roles_for_scope(scope: str) -> frozenset[str] | None:
    """Return the roles a scope searches, or ``None`` when it searches every role."""
    return _SCOPE_ROLES[scope]


def is_indexable_kind(kind: str) -> bool:
    """Whether a text-source kind contributes chunks to the lexical index."""
    return kind in ROLE_BY_KIND


def role_for_kind(kind: str) -> str | None:
    """Return the search role of a text-source kind, or ``None`` when excluded."""
    return ROLE_BY_KIND.get(kind)


# Audio artifact roles, most-descriptive first, used to point a search result at
# the run audio when the text source did not carry its own artifact link.
_AUDIO_ARTIFACT_ROLES = (
    ARTIFACT_ROLE_FINAL_AUDIO,
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_PAID_RAW_AUDIO,
    ARTIFACT_ROLE_LOCAL_RAW_AUDIO,
    ARTIFACT_ROLE_SOURCE_AUDIO,
)
_PART_AUDIO_ARTIFACT_ROLES = frozenset(
    {ARTIFACT_ROLE_CHUNK_AUDIO, ARTIFACT_ROLE_PAID_RAW_AUDIO, ARTIFACT_ROLE_LOCAL_RAW_AUDIO}
)
_TTS_ATTEMPT_CALL_TYPES = frozenset({ATTEMPT_CALL_TYPE_TTS_CHUNK, ATTEMPT_CALL_TYPE_LOCAL_TTS})
_QUALITY_ATTEMPT_CALL_TYPES = frozenset({ATTEMPT_CALL_TYPE_VERIFY, ATTEMPT_CALL_TYPE_PAID_QUALITY})
_ASR_ATTEMPT_CALL_TYPES = (
    frozenset({ATTEMPT_CALL_TYPE_ASR, ATTEMPT_CALL_TYPE_TIMING, ATTEMPT_CALL_TYPE_PAID_TIMING})
    | _QUALITY_ATTEMPT_CALL_TYPES
)


def search_lexical(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    roles: Sequence[str] | frozenset[str] | None = None,
    kind: str | None = None,
    run_uuid: str | None = None,
    operation: str | None = None,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Run one offline BM25 lexical query and build bounded result rows.

    Every filter is applied in the same ``SELECT`` before ``LIMIT``, so the limit
    bounds only results that already passed the filters. The ranking orders by
    ``bm25(search_fts)`` ascending -- the standard FTS5 convention where a smaller
    value is more relevant -- and breaks ties by newest run. The user string is
    tokenized into quoted literals (never a raw MATCH expression), so quotes,
    hyphens, punctuation, and SQL-looking input are safe. ``roles`` ``None`` means
    every role; an empty set means no role matches.
    """
    tokens = query_tokens(query)
    match_expression = build_match_expression(tokens)
    if match_expression is None:
        return {"results": [], "tokens": tokens, "warnings": []}

    conditions: list[str] = ["search_fts MATCH ?"]
    parameters: list[Any] = [match_expression]
    if roles is not None:
        role_list = list(roles)
        if not role_list:
            return {"results": [], "tokens": tokens, "warnings": []}
        placeholders = ", ".join("?" for _ in role_list)
        conditions.append(f"c.role IN ({placeholders})")
        parameters.extend(role_list)
    if kind is not None:
        conditions.append("c.kind = ?")
        parameters.append(kind)
    if run_uuid is not None:
        conditions.append("c.run_uuid = ?")
        parameters.append(run_uuid)
    if operation is not None:
        conditions.append("r.operation = ?")
        parameters.append(operation)
    if provider is not None:
        conditions.append(
            "EXISTS (SELECT 1 FROM attempts a WHERE a.run_uuid = c.run_uuid AND a.provider = ?)"
        )
        parameters.append(provider)
    if since is not None:
        conditions.append("r.created_at >= ?")
        parameters.append(since)
    if until is not None:
        # ISO timestamps contain a time after the YYYY-MM-DD prefix. Compare
        # calendar dates to keep --until inclusive, including 9999-12-31.
        conditions.append("substr(r.created_at, 1, 10) <= ?")
        parameters.append(until)

    sql = (
        "SELECT c.chunk_id AS chunk_id, c.text_source_uuid AS source_uuid, "
        "c.run_uuid AS run_uuid, c.part_uuid AS part_uuid, "
        "c.artifact_uuid AS artifact_uuid, c.kind AS kind, c.role AS role, "
        "c.chunk_index AS chunk_index, c.char_start AS char_start, c.char_end AS char_end, "
        "c.text AS text, c.text_hash AS text_hash, c.start_ms AS start_ms, c.end_ms AS end_ms, "
        "r.user_label AS user_label, r.operation AS operation, r.run_root AS run_root, "
        "r.status AS run_status, r.config_snapshot AS run_snapshot, r.created_at AS created_at, "
        "sa.artifact_uuid AS source_artifact_uuid, sa.role AS source_artifact_role, "
        "sa.part_uuid AS source_artifact_part_uuid, "
        "sa.path AS source_artifact_path, sa.path_kind AS source_artifact_path_kind, "
        "sa.availability AS source_artifact_availability "
        "FROM search_fts "
        "JOIN search_chunks AS c ON c.chunk_id = search_fts.rowid "
        "JOIN runs AS r ON r.run_uuid = c.run_uuid "
        "LEFT JOIN artifacts AS sa ON sa.artifact_uuid = c.artifact_uuid "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY bm25(search_fts) ASC, r.created_at DESC, c.chunk_id ASC "
        "LIMIT ?"
    )
    parameters.append(limit)
    rows = connection.execute(sql, parameters).fetchall()

    run_uuids = {row["run_uuid"] for row in rows}
    run_audio = _resolve_run_audio(connection, run_uuids)
    run_attempts = _resolve_run_attempts(connection, run_uuids)
    results = [_result_row(row, tokens, run_audio, run_attempts) for row in rows]
    return {"results": results, "tokens": tokens, "warnings": []}


def _resolve_run_audio(
    connection: sqlite3.Connection, run_uuids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Collect candidate audio artifacts without assigning one part's bytes to another."""
    if not run_uuids:
        return {}
    run_placeholders = ", ".join("?" for _ in run_uuids)
    role_placeholders = ", ".join("?" for _ in _AUDIO_ARTIFACT_ROLES)
    role_case = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(_AUDIO_ARTIFACT_ROLES))
    sql = (
        "SELECT artifact_uuid, run_uuid, part_uuid, role, path, path_kind, availability "
        "FROM artifacts "
        f"WHERE run_uuid IN ({run_placeholders}) AND role IN ({role_placeholders}) "
        f"ORDER BY CASE role {role_case} ELSE 99 END ASC, created_at ASC"
    )
    parameters: list[Any] = [*run_uuids, *_AUDIO_ARTIFACT_ROLES, *_AUDIO_ARTIFACT_ROLES]
    resolved: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute(sql, parameters):
        resolved.setdefault(row["run_uuid"], []).append(dict(row))
    return resolved


def _resolve_run_attempts(
    connection: sqlite3.Connection, run_uuids: set[str]
) -> dict[str, list[sqlite3.Row]]:
    """Load actual run/part attempt provenance, never a GROUP BY arbitrary row."""
    if not run_uuids:
        return {}
    placeholders = ", ".join("?" for _ in run_uuids)
    sql = (
        "SELECT run_uuid, part_uuid, call_type, provider, model FROM attempts "
        f"WHERE run_uuid IN ({placeholders})"
    )
    resolved: dict[str, list[sqlite3.Row]] = {}
    for attempt in connection.execute(sql, list(run_uuids)):
        resolved.setdefault(attempt["run_uuid"], []).append(attempt)
    return resolved


def _attributed_provider_model(
    row: sqlite3.Row, attempts: Sequence[sqlite3.Row]
) -> tuple[str | None, str | None]:
    """Return a source-relevant *unique* provider/model or unknown on ambiguity.

    A run may contain TTS attempts and separate local quality-ASR attempts. The
    script and a per-turn verification transcript must not inherit an unrelated
    provider merely because the database grouped by run UUID. A paid child
    quality attempt belongs to another run and cannot be guessed here; it stays
    unknown unless this result has a matching attempt on its own run.
    """
    if row["kind"] == TEXT_KIND_VERIFICATION_TRANSCRIPT:
        call_types = _QUALITY_ATTEMPT_CALL_TYPES
    elif row["role"] == ROLE_ASR or (
        row["role"] == ROLE_LABEL
        and row["operation"] in (OPERATION_ASR, OPERATION_TIMINGS, OPERATION_VERIFY)
    ):
        call_types = _ASR_ATTEMPT_CALL_TYPES
    else:
        call_types = _TTS_ATTEMPT_CALL_TYPES
    part_uuid = row["part_uuid"]
    candidates = {
        (attempt["provider"], attempt["model"])
        for attempt in attempts
        if attempt["call_type"] in call_types
        and (part_uuid is None or attempt["part_uuid"] == part_uuid)
    }
    if len(candidates) != 1:
        return None, None
    provider, model = next(iter(candidates))
    if provider is None:
        return None, None
    return provider, model


def _can_link_source_audio(row: sqlite3.Row) -> bool:
    """Only recognized speech or an ASR/timing/verify label names source audio."""
    return row["part_uuid"] is None and (
        row["role"] == ROLE_ASR
        or (
            row["role"] == ROLE_LABEL
            and row["operation"] in (OPERATION_ASR, OPERATION_TIMINGS, OPERATION_VERIFY)
        )
    )


def _historical_paid_source(row: sqlite3.Row) -> dict[str, Any] | None:
    """Recover an old paid transcript's source link from its saved run identity.

    Pre-S08 paid reservations saved a validated absolute source path and its
    observed availability in the canonical snapshot, but not an artifact row.
    Use only the exact paid origin and the ASR role; never invent a source from
    arbitrary request options or another run.
    """
    if not _can_link_source_audio(row):
        return None
    raw_snapshot = row["run_snapshot"]
    if not isinstance(raw_snapshot, str):
        return None
    try:
        snapshot = json.loads(raw_snapshot)
    except (ValueError, TypeError):
        return None
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("operation_origin") != PAID_TRANSCRIPTION_ORIGIN
    ):
        return None
    path = snapshot.get("source_audio")
    availability = snapshot.get("source_audio_availability")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or availability not in (AVAILABILITY_PRESENT, AVAILABILITY_MISSING)
    ):
        return None
    return {
        "artifact_uuid": None,
        "role": ARTIFACT_ROLE_SOURCE_AUDIO,
        "path": path,
        "path_kind": PATH_KIND_EXTERNAL_ABSOLUTE,
        "availability": availability,
    }


def _fallback_audio(
    row: sqlite3.Row, candidates: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Link only a completed final, this exact part, or this ASR run's source."""
    if row["run_status"] == RUN_STATUS_COMPLETED:
        for candidate in candidates:
            if candidate["role"] == ARTIFACT_ROLE_FINAL_AUDIO and candidate["part_uuid"] is None:
                return candidate
    part_uuid = row["part_uuid"]
    if part_uuid is not None:
        for candidate in candidates:
            if (
                candidate["part_uuid"] == part_uuid
                and candidate["role"] in _PART_AUDIO_ARTIFACT_ROLES
            ):
                return candidate
        return None
    if _can_link_source_audio(row):
        for candidate in candidates:
            if candidate["role"] == ARTIFACT_ROLE_SOURCE_AUDIO and candidate["part_uuid"] is None:
                return candidate
    return _historical_paid_source(row)


def _result_row(
    row: sqlite3.Row,
    tokens: Sequence[str],
    run_audio: dict[str, list[dict[str, Any]]],
    run_attempts: dict[str, list[sqlite3.Row]],
) -> dict[str, Any]:
    """Project one joined row into a bounded result without leaking raw secrets."""
    run_uuid = row["run_uuid"]
    run_root = row["run_root"]
    provider, model = _attributed_provider_model(row, run_attempts.get(run_uuid, []))
    source_role = row["source_artifact_role"]
    source_part = row["source_artifact_part_uuid"]
    part_uuid = row["part_uuid"]
    source_is_audio = source_role in _AUDIO_ARTIFACT_ROLES and (
        source_role == ARTIFACT_ROLE_FINAL_AUDIO
        or (source_role == ARTIFACT_ROLE_SOURCE_AUDIO and _can_link_source_audio(row))
        or (part_uuid is not None and part_uuid == source_part)
    )
    if source_is_audio:
        audio = {
            "artifact_uuid": row["source_artifact_uuid"],
            "role": source_role,
            "path": _resolved_path(
                run_root, row["source_artifact_path"], row["source_artifact_path_kind"]
            ),
            "path_kind": row["source_artifact_path_kind"],
            "availability": row["source_artifact_availability"],
        }
    else:
        audio = None
        fallback = _fallback_audio(row, run_audio.get(run_uuid, []))
        if fallback is not None:
            audio = {
                "artifact_uuid": fallback["artifact_uuid"],
                "role": fallback["role"],
                "path": _resolved_path(run_root, fallback["path"], fallback["path_kind"]),
                "path_kind": fallback["path_kind"],
                "availability": fallback["availability"],
            }
    if audio is not None and audio["availability"] == AVAILABILITY_PRESENT:
        # The stored observation can grow stale when a saved audio file is later
        # removed. Never promote a persisted missing artifact to present without
        # verifying its bytes, but do mark an absent linked file as missing.
        if not Path(audio["path"]).is_file():
            audio["availability"] = AVAILABILITY_MISSING
    return {
        "chunk_id": row["chunk_id"],
        "run_uuid": run_uuid,
        "user_label": row["user_label"],
        "operation": row["operation"],
        "kind": row["kind"],
        "role": row["role"],
        "text_source_uuid": row["source_uuid"],
        "part_uuid": row["part_uuid"],
        "artifact_uuid": row["artifact_uuid"],
        "chunk_index": row["chunk_index"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "start_ms": row["start_ms"],
        "end_ms": row["end_ms"],
        "text_hash": row["text_hash"],
        "provider": provider,
        "model": model,
        "created_at": row["created_at"],
        "snippet": make_snippet(row["text"], tokens),
        "audio": audio,
    }


def _resolved_path(run_root: str, path: str, path_kind: str) -> str:
    """Return a display path: a managed relative path joined to its run root."""
    if path_kind == PATH_KIND_MANAGED_RELATIVE:
        return str(Path(run_root) / path)
    return path
