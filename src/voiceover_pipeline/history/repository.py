"""Typed entity API over the canonical history SQLite schema.

Plan section 6 makes SQLite the single source of truth for run history. This
module is the write/read seam used by the legacy importer and, later, by the
execution services. It exposes:

* frozen record dataclasses (:class:`RunRecord` ... :class:`TextSourceRecord`);
* :class:`Cost`, a value object that keeps an exact ``Decimal`` string together
  with the provenance that says whether that guarantee really held;
* :class:`HistoryRepository`, whose insert methods each run one short
  transaction and whose :meth:`HistoryRepository.transaction` context manager
  groups several inserts so a legacy run, its parts, attempts, artifacts, and
  text sources land atomically or not at all. :meth:`HistoryRepository.create_legacy_run`
  is only valid inside that caller-managed transaction, so a failed import can
  never leave a committed run without its parts and costs.

Money contract: an attempt's cost is stored as a ``TEXT`` decimal string.
``NULL`` means *unknown*; the string ``"0"`` is a real observed zero. A value
imported from a legacy number keeps its captured decimal text (or the shortest
form of a float) but carries ``source="legacy_import"`` and
``exact_available=False``, so no reader can mistake it for a provider-confirmed
exact amount. There is one cost column per attempt, so an imported legacy total
is never counted twice.

Safety contract: the ingestion boundary itself defends against secrets and
unsafe paths, rather than trusting the caller. ``config_snapshot`` is redacted
fail-closed (API keys, authorization headers, and signed-URL parameters are
removed before storage), a ``managed_relative`` artifact path must be a plain
relative slash path, and an ``external_absolute`` path must actually be
absolute. Run ownership is enforced by the database: a part, attempt, artifact,
or text source may only link to a parent belonging to the same run.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .database import HistoryDatabase, utc_now

MAX_QUERY_LIMIT = 500
DEFAULT_QUERY_LIMIT = 50


class HistoryRepositoryError(RuntimeError):
    """A repository call violated the history write contract."""


# Documented classification values. Writes validate the ones that steer file
# handling, so a typo fails loudly instead of becoming an unclassifiable row.
PATH_KIND_MANAGED_RELATIVE = "managed_relative"
PATH_KIND_EXTERNAL_ABSOLUTE = "external_absolute"
AVAILABILITY_PRESENT = "present"
AVAILABILITY_MISSING = "missing"
TEXT_COMPLETENESS_COMPLETE = "complete"
TEXT_COMPLETENESS_INCOMPLETE = "incomplete"
COST_SOURCE_UNKNOWN = "unknown"
COST_SOURCE_EXACT = "exact"
COST_SOURCE_LEGACY_FLOAT = "legacy_import"
TEXT_KIND_TTS_SCRIPT = "tts_script"
TEXT_KIND_TTS_DIRECTION = "tts_direction"
TEXT_KIND_ASR_TRANSCRIPT = "asr_transcript"
TEXT_KIND_VERIFICATION_TRANSCRIPT = "verification_transcript"
# Placeholder written in place of a string value that looks like a secret.
REDACTED_VALUE = "[redacted]"

_PATH_KINDS = frozenset({PATH_KIND_MANAGED_RELATIVE, PATH_KIND_EXTERNAL_ABSOLUTE})
_AVAILABILITIES = frozenset({AVAILABILITY_PRESENT, AVAILABILITY_MISSING})
_TEXT_COMPLETENESS = frozenset({TEXT_COMPLETENESS_COMPLETE, TEXT_COMPLETENESS_INCOMPLETE})

# Keys whose value must never reach the database. Matching is case-insensitive
# and covers common API-key, authorization, token, secret, and signed-URL names.
_SENSITIVE_KEY_PATTERN = re.compile(
    r"api[_-]?key|apikey|authorization|auth[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"bearer|secret|password|passwd|credential|signature|signed[_-]?url|private[_-]?key",
    re.IGNORECASE,
)
# String values that are themselves secret-bearing, regardless of their key.
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bgsk_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bxai-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(
        r"[?&](x-amz-signature|x-amz-credential|x-amz-security-token|"
        r"x-goog-signature|x-goog-credential|x-goog-date|x-goog-expires|"
        r"access_token|api[_-]?key|signature|sig|token|expires|se)=",
        re.IGNORECASE,
    ),
)
# A signed URL keeps its secret in the query or fragment. Since every provider
# names its parameters differently, the safe default is to redact any http(s)
# URL that carries a query string or fragment at all, even an unfamiliar one.
_URL_WITH_QUERY_OR_FRAGMENT_PATTERN = re.compile(r"^https?://\S*[?#]\S*$", re.IGNORECASE)

_MANAGED_PATH_INVALID_SEGMENTS = frozenset({"", ".", ".."})
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def _dump_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _require_choice(value: str, allowed: frozenset[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _require_uuid(value: str, field_name: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID string") from exc


def _require_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_QUERY_LIMIT}")
    return limit


def _require_offset(offset: int) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return offset


def _decimal_from_value(value: Decimal | int | str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("cost must not be a bool")
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, int):
        decimal_value = Decimal(value)
    elif isinstance(value, str):
        try:
            decimal_value = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"cost string is not a decimal: {value!r}") from exc
    else:
        raise ValueError(
            f"cost must be a Decimal, int, or decimal string, not {type(value).__name__}"
        )
    if not decimal_value.is_finite():
        raise ValueError("cost must be a finite decimal")
    return decimal_value


def _legacy_decimal_text(value: object) -> str | None:
    """Return the decimal text of a legacy cost without binary float coercion.

    A ``str`` or ``Decimal`` keeps its captured lexeme verbatim (so ``0.1000``
    stays ``0.1000``); an ``int`` is exact; a ``float`` falls back to its
    shortest ``repr``. ``None`` means the value is not a usable finite number.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            decimal_value = Decimal(value)
        except InvalidOperation:
            return None
        return value if decimal_value.is_finite() else None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return str(value) if value.is_finite() else None
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else None
    return None


def _is_sensitive_string(value: str) -> bool:
    if _URL_WITH_QUERY_OR_FRAGMENT_PATTERN.match(value) is not None:
        return True
    return any(pattern.search(value) is not None for pattern in _SENSITIVE_VALUE_PATTERNS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("config_snapshot keys must be strings")
            if _SENSITIVE_KEY_PATTERN.search(key) is not None:
                continue
            result[key] = _redact_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return REDACTED_VALUE if _is_sensitive_string(value) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"config_snapshot contains an unsupported value type: {type(value).__name__}")


def _redact_snapshot(snapshot: Any) -> dict[str, Any] | None:
    """Return a fail-closed redacted copy of a run config snapshot.

    The ingestion boundary removes API keys, authorization headers, tokens,
    secrets, and signed-URL parameters instead of trusting the caller to have
    done so: any http(s) URL carrying a query string or fragment is replaced
    even when its parameter name is unfamiliar. Safe metadata is preserved; an
    unsupported value type or a non-mapping snapshot raises rather than being
    stored unchecked. A redacted value is never echoed back in an error or log.
    """
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise ValueError("config_snapshot must be a mapping or None")
    redacted = _redact_value(snapshot)
    if not isinstance(redacted, dict):
        raise ValueError("config_snapshot must be a mapping or None")
    return redacted


def _require_managed_relative_path(path: str) -> str:
    """Reject a managed path that is not a plain relative slash path."""
    if not isinstance(path, str) or not path:
        raise ValueError("managed path must be a non-empty string")
    if path.startswith("/") or path.startswith("\\"):
        raise ValueError(f"managed path must be relative, not absolute: {path!r}")
    if "\\" in path:
        raise ValueError(f"managed path must use forward slashes: {path!r}")
    if _WINDOWS_DRIVE_PATTERN.match(path) is not None:
        raise ValueError(f"managed path must not carry a drive or scheme: {path!r}")
    if any(segment in _MANAGED_PATH_INVALID_SEGMENTS for segment in path.split("/")):
        raise ValueError(f"managed path must not contain empty or traversing segments: {path!r}")
    return path


def _require_external_absolute_path(path: str) -> str:
    """Reject an external path that is not absolute on POSIX or Windows."""
    if not isinstance(path, str) or not path:
        raise ValueError("external path must be a non-empty string")
    if not (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()):
        raise ValueError(f"external path must be absolute: {path!r}")
    return path


def _canonical_legacy_source_root(root: str) -> str:
    """Return the canonical absolute form used to identify a legacy source root."""
    if not isinstance(root, str) or not root.strip():
        raise ValueError("legacy_source_root must be a non-empty string")
    return str(Path(root).expanduser().resolve())


@dataclass(frozen=True)
class Cost:
    """A money amount with explicit provenance.

    ``amount`` is a decimal string (or ``None`` for an unknown cost). ``source``
    labels where the value came from and ``exact_available`` records whether the
    producer guaranteed an exact decimal, which a legacy import cannot. ``raw``
    keeps the original captured representation when it differs from ``amount``.
    """

    amount: str | None = None
    currency: str | None = None
    source: str = COST_SOURCE_UNKNOWN
    exact_available: bool = False
    raw: str | None = None

    @classmethod
    def unknown(cls) -> Cost:
        """Return an unknown cost: no amount, no exact guarantee."""
        return cls()

    @classmethod
    def exact(
        cls,
        value: Decimal | int | str,
        *,
        currency: str | None = None,
        source: str = COST_SOURCE_EXACT,
    ) -> Cost:
        """Return a cost from an exact Decimal-compatible value."""
        amount = str(_decimal_from_value(value))
        return cls(amount=amount, currency=currency, source=source, exact_available=True)

    @classmethod
    def legacy_float(
        cls,
        value: Decimal | int | float | str,
        *,
        currency: str | None = None,
        raw: str | None = None,
    ) -> Cost:
        """Return a cost imported from a legacy number without an exact guarantee.

        The captured decimal text is kept as the ``amount`` (a ``str``/``Decimal``
        lexeme such as ``"0.1000"`` is preserved verbatim; a ``float`` becomes its
        shortest ``repr``). ``raw`` lets an importer pass the original numeric
        token separately. ``exact_available`` stays ``False`` because the
        provider's exact amount was never recorded.
        """
        text = _legacy_decimal_text(value)
        if text is None:
            return cls.unknown()
        return cls(
            amount=text,
            currency=currency,
            source=COST_SOURCE_LEGACY_FLOAT,
            exact_available=False,
            raw=raw if raw is not None else text,
        )

    def as_decimal(self) -> Decimal | None:
        """Return the amount as a Decimal, or ``None`` for an unknown cost."""
        if self.amount is None:
            return None
        return Decimal(self.amount)


@dataclass(frozen=True)
class RunRecord:
    run_uuid: str
    operation: str
    status: str
    run_root: str
    legacy_source_root: str | None = None
    user_label: str | None = None
    parent_uuid: str | None = None
    config_snapshot: dict[str, Any] | None = None
    record_version: int = 1
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class PartRecord:
    part_uuid: str
    run_uuid: str
    position: int
    prepared_text: str | None = None
    voice: str | None = None
    vibe_shared: str | None = None
    vibe_specific: str | None = None
    vibe_effective: str | None = None
    fingerprint: str | None = None
    stage: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class AttemptRecord:
    attempt_uuid: str
    run_uuid: str
    call_type: str
    part_uuid: str | None = None
    provider: str | None = None
    model: str | None = None
    account_alias: str | None = None
    remote_id: str | None = None
    status: str | None = None
    usage: dict[str, Any] | None = None
    cost: Cost = field(default_factory=Cost.unknown)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_uuid: str
    run_uuid: str
    role: str
    path_kind: str
    path: str
    part_uuid: str | None = None
    attempt_uuid: str | None = None
    mime: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    media_metadata: dict[str, Any] | None = None
    availability: str = AVAILABILITY_PRESENT
    created_at: str = ""


@dataclass(frozen=True)
class TextSourceRecord:
    text_source_uuid: str
    run_uuid: str
    kind: str
    origin: str
    content: str | None = None
    part_uuid: str | None = None
    artifact_uuid: str | None = None
    content_hash: str | None = None
    language: str | None = None
    text_completeness: str = TEXT_COMPLETENESS_COMPLETE
    created_at: str = ""


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_uuid=row["run_uuid"],
        operation=row["operation"],
        status=row["status"],
        run_root=row["run_root"],
        legacy_source_root=row["legacy_source_root"],
        user_label=row["user_label"],
        parent_uuid=row["parent_uuid"],
        config_snapshot=_load_json(row["config_snapshot"]),
        record_version=int(row["record_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_part(row: sqlite3.Row) -> PartRecord:
    return PartRecord(
        part_uuid=row["part_uuid"],
        run_uuid=row["run_uuid"],
        position=int(row["position"]),
        prepared_text=row["prepared_text"],
        voice=row["voice"],
        vibe_shared=row["vibe_shared"],
        vibe_specific=row["vibe_specific"],
        vibe_effective=row["vibe_effective"],
        fingerprint=row["fingerprint"],
        stage=row["stage"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_attempt(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        attempt_uuid=row["attempt_uuid"],
        run_uuid=row["run_uuid"],
        call_type=row["call_type"],
        part_uuid=row["part_uuid"],
        provider=row["provider"],
        model=row["model"],
        account_alias=row["account_alias"],
        remote_id=row["remote_id"],
        status=row["status"],
        usage=_load_json(row["usage_json"]),
        cost=Cost(
            amount=row["cost"],
            currency=row["cost_currency"],
            source=row["cost_source"] or COST_SOURCE_UNKNOWN,
            exact_available=bool(row["cost_exact_available"]),
            raw=row["cost_raw"],
        ),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_uuid=row["artifact_uuid"],
        run_uuid=row["run_uuid"],
        role=row["role"],
        path_kind=row["path_kind"],
        path=row["path"],
        part_uuid=row["part_uuid"],
        attempt_uuid=row["attempt_uuid"],
        mime=row["mime"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        media_metadata=_load_json(row["media_metadata_json"]),
        availability=row["availability"],
        created_at=row["created_at"],
    )


def _row_to_text_source(row: sqlite3.Row) -> TextSourceRecord:
    return TextSourceRecord(
        text_source_uuid=row["text_source_uuid"],
        run_uuid=row["run_uuid"],
        kind=row["kind"],
        origin=row["origin"],
        content=row["content"],
        part_uuid=row["part_uuid"],
        artifact_uuid=row["artifact_uuid"],
        content_hash=row["content_hash"],
        language=row["language"],
        text_completeness=row["text_completeness"],
        created_at=row["created_at"],
    )


class HistoryRepository:
    """Typed read/write API over a migrated :class:`HistoryDatabase`.

    Insert methods accept explicit keyword arguments and return the persisted
    frozen record. Each insert opens one short transaction unless the caller has
    already opened one with :meth:`transaction`. To import a legacy run
    atomically, wrap a run, its parts, attempts, artifacts, and text sources in a
    single :meth:`transaction` block: a failure rolls every entity back.
    """

    def __init__(self, database: HistoryDatabase) -> None:
        self._database = database
        self._connection = database.connection

    @contextmanager
    def transaction(self) -> Iterator[HistoryRepository]:
        """Run a group of writes as one transaction, rolling back on failure.

        Nested use inside an already open transaction simply joins that
        transaction, so insert methods can be called from within this block.
        """
        connection = self._connection
        if connection.in_transaction:
            yield self
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def _require_outer_transaction(self, method: str) -> None:
        """Fail closed unless the caller already opened :meth:`transaction`.

        A legacy import must commit the run and all of its parts, attempts,
        artifacts, and text sources together, so a half-finished import can never
        become an idempotent "already imported" result. The check runs before any
        insert and reports no value from the caller's data.
        """
        if not self._connection.in_transaction:
            raise HistoryRepositoryError(
                f"{method} must be called inside `with repository.transaction():` so the run "
                "and all of its parts, attempts, artifacts, and text sources commit or roll "
                "back together"
            )

    # -- writes ---------------------------------------------------------------

    def create_run(
        self,
        *,
        operation: str,
        run_root: str,
        status: str = "running",
        user_label: str | None = None,
        parent_uuid: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        run_uuid: str | None = None,
        record_version: int = 1,
        created_at: str | None = None,
        legacy_source_root: str | None = None,
    ) -> RunRecord:
        """Insert a run and return it.

        ``run_uuid`` is generated when omitted and validated when supplied;
        ``run_root`` scopes the run's source while ``user_label`` stays a
        separate, non-unique field. ``config_snapshot`` is redacted before
        storage. A ``legacy_source_root`` makes this an import write: it must run
        inside the caller's :meth:`transaction` (like :meth:`create_legacy_run`)
        so the run and its import entities commit together.
        """
        if legacy_source_root is not None:
            self._require_outer_transaction("create_run(legacy_source_root=...)")
        record = self._build_run_record(
            operation=operation,
            run_root=run_root,
            status=status,
            user_label=user_label,
            parent_uuid=parent_uuid,
            config_snapshot=config_snapshot,
            run_uuid=run_uuid,
            record_version=record_version,
            created_at=created_at,
            legacy_source_root=legacy_source_root,
        )
        with self.transaction():
            self._insert_run(record)
        return record

    def create_legacy_run(
        self,
        *,
        operation: str,
        run_root: str,
        legacy_source_root: str,
        status: str = "completed",
        user_label: str | None = None,
        parent_uuid: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        run_uuid: str | None = None,
        record_version: int = 1,
        created_at: str | None = None,
    ) -> tuple[RunRecord, bool]:
        """Import one legacy source root idempotently inside the caller's transaction.

        Must be called inside ``with repository.transaction():`` together with
        every part, attempt, artifact, and text source of the import; calling it
        alone raises :class:`HistoryRepositoryError` before any insert. Returns
        ``(record, created)``. The canonical ``legacy_source_root`` is matched
        under a unique index inside that immediate transaction, so a repeated or
        concurrent import of the same directory converges on a single complete
        run, while ordinary runs (which leave ``legacy_source_root`` NULL) may
        still share a ``run_root``.
        """
        self._require_outer_transaction("create_legacy_run")
        canonical_root = _canonical_legacy_source_root(legacy_source_root)
        existing = self._find_by_legacy_source_root(canonical_root)
        if existing is not None:
            return existing, False
        record = self._build_run_record(
            operation=operation,
            run_root=run_root,
            status=status,
            user_label=user_label,
            parent_uuid=parent_uuid,
            config_snapshot=config_snapshot,
            run_uuid=run_uuid,
            record_version=record_version,
            created_at=created_at,
            legacy_source_root=canonical_root,
        )
        self._insert_run(record)
        return record, True

    def add_part(
        self,
        run_uuid: str,
        *,
        position: int,
        prepared_text: str | None = None,
        voice: str | None = None,
        vibe_shared: str | None = None,
        vibe_specific: str | None = None,
        vibe_effective: str | None = None,
        fingerprint: str | None = None,
        stage: str | None = None,
        part_uuid: str | None = None,
        created_at: str | None = None,
    ) -> PartRecord:
        """Insert a part under ``run_uuid`` at a unique ``position``."""
        identifier = (
            _require_uuid(part_uuid, "part_uuid") if part_uuid is not None else str(uuid.uuid4())
        )
        now = created_at or utc_now()
        record = PartRecord(
            part_uuid=identifier,
            run_uuid=run_uuid,
            position=position,
            prepared_text=prepared_text,
            voice=voice,
            vibe_shared=vibe_shared,
            vibe_specific=vibe_specific,
            vibe_effective=vibe_effective,
            fingerprint=fingerprint,
            stage=stage,
            created_at=now,
            updated_at=now,
        )
        with self.transaction():
            self._insert_part(record)
        return record

    def add_attempt(
        self,
        run_uuid: str,
        *,
        call_type: str,
        part_uuid: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        account_alias: str | None = None,
        remote_id: str | None = None,
        status: str | None = None,
        usage: dict[str, Any] | None = None,
        cost: Cost | None = None,
        error: str | None = None,
        attempt_uuid: str | None = None,
        created_at: str | None = None,
    ) -> AttemptRecord:
        """Insert one provider/attempt record with explicit cost provenance.

        When ``part_uuid`` is given it must belong to ``run_uuid``; the database
        composite foreign key rejects a cross-run link.
        """
        identifier = (
            _require_uuid(attempt_uuid, "attempt_uuid")
            if attempt_uuid is not None
            else str(uuid.uuid4())
        )
        now = created_at or utc_now()
        record = AttemptRecord(
            attempt_uuid=identifier,
            run_uuid=run_uuid,
            call_type=call_type,
            part_uuid=part_uuid,
            provider=provider,
            model=model,
            account_alias=account_alias,
            remote_id=remote_id,
            status=status,
            usage=usage,
            cost=cost if cost is not None else Cost.unknown(),
            error=error,
            created_at=now,
            updated_at=now,
        )
        with self.transaction():
            self._insert_attempt(record)
        return record

    def add_artifact(
        self,
        run_uuid: str,
        *,
        role: str,
        path_kind: str,
        path: str,
        part_uuid: str | None = None,
        attempt_uuid: str | None = None,
        mime: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        media_metadata: dict[str, Any] | None = None,
        availability: str = AVAILABILITY_PRESENT,
        artifact_uuid: str | None = None,
        created_at: str | None = None,
    ) -> ArtifactRecord:
        """Insert an artifact reference, managed-relative or explicitly external.

        A ``managed_relative`` path must be a plain relative slash path; an
        ``external_absolute`` path must actually be absolute. Any ``part_uuid``
        or ``attempt_uuid`` must belong to ``run_uuid``.
        """
        _require_choice(path_kind, _PATH_KINDS, "path_kind")
        _require_choice(availability, _AVAILABILITIES, "availability")
        if path_kind == PATH_KIND_MANAGED_RELATIVE:
            _require_managed_relative_path(path)
        else:
            _require_external_absolute_path(path)
        identifier = (
            _require_uuid(artifact_uuid, "artifact_uuid")
            if artifact_uuid is not None
            else str(uuid.uuid4())
        )
        record = ArtifactRecord(
            artifact_uuid=identifier,
            run_uuid=run_uuid,
            role=role,
            path_kind=path_kind,
            path=path,
            part_uuid=part_uuid,
            attempt_uuid=attempt_uuid,
            mime=mime,
            size_bytes=size_bytes,
            sha256=sha256,
            media_metadata=media_metadata,
            availability=availability,
            created_at=created_at or utc_now(),
        )
        with self.transaction():
            self._insert_artifact(record)
        return record

    def add_text_source(
        self,
        run_uuid: str,
        *,
        kind: str,
        origin: str,
        content: str | None = None,
        part_uuid: str | None = None,
        artifact_uuid: str | None = None,
        content_hash: str | None = None,
        language: str | None = None,
        text_completeness: str = TEXT_COMPLETENESS_COMPLETE,
        text_source_uuid: str | None = None,
        created_at: str | None = None,
    ) -> TextSourceRecord:
        """Insert a preserved text source with its completeness and provenance.

        ``content`` is stored verbatim as UTF-8; pass ``None`` with a
        ``text_completeness`` of ``incomplete`` when only a hash survives. Any
        ``part_uuid`` or ``artifact_uuid`` must belong to ``run_uuid``.
        """
        _require_choice(text_completeness, _TEXT_COMPLETENESS, "text_completeness")
        identifier = (
            _require_uuid(text_source_uuid, "text_source_uuid")
            if text_source_uuid is not None
            else str(uuid.uuid4())
        )
        record = TextSourceRecord(
            text_source_uuid=identifier,
            run_uuid=run_uuid,
            kind=kind,
            origin=origin,
            content=content,
            part_uuid=part_uuid,
            artifact_uuid=artifact_uuid,
            content_hash=content_hash,
            language=language,
            text_completeness=text_completeness,
            created_at=created_at or utc_now(),
        )
        with self.transaction():
            self._insert_text_source(record)
        return record

    # -- reads ----------------------------------------------------------------

    def get_run(self, run_uuid: str) -> RunRecord | None:
        """Return one run by UUID, or ``None`` when absent."""
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_uuid = ?", (run_uuid,)
        ).fetchone()
        return _row_to_run(row) if row is not None else None

    def find_run_by_legacy_source_root(self, legacy_source_root: str) -> RunRecord | None:
        """Return the imported run for a legacy source root, or ``None``."""
        return self._find_by_legacy_source_root(_canonical_legacy_source_root(legacy_source_root))

    def list_runs(
        self,
        *,
        user_label: str | None = None,
        operation: str | None = None,
        status: str | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs newest-first, filtered by metadata, with a bounded limit."""
        _require_limit(limit)
        _require_offset(offset)
        conditions: list[str] = []
        parameters: list[Any] = []
        if user_label is not None:
            conditions.append("user_label = ?")
            parameters.append(user_label)
        if operation is not None:
            conditions.append("operation = ?")
            parameters.append(operation)
        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM runs{where} ORDER BY created_at DESC, run_uuid DESC LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])
        rows = self._connection.execute(sql, parameters).fetchall()
        return [_row_to_run(row) for row in rows]

    def find_runs_by_label(
        self, user_label: str, *, limit: int = DEFAULT_QUERY_LIMIT, offset: int = 0
    ) -> list[RunRecord]:
        """Return every run carrying ``user_label`` as candidates, not one pick."""
        return self.list_runs(user_label=user_label, limit=limit, offset=offset)

    def find_runs_by_root(
        self, run_root: str, *, limit: int = DEFAULT_QUERY_LIMIT, offset: int = 0
    ) -> list[RunRecord]:
        """Return every run whose ``run_root`` matches exactly."""
        _require_limit(limit)
        _require_offset(offset)
        rows = self._connection.execute(
            "SELECT * FROM runs WHERE run_root = ? "
            "ORDER BY created_at DESC, run_uuid DESC LIMIT ? OFFSET ?",
            (run_root, limit, offset),
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    def get_parts(self, run_uuid: str) -> list[PartRecord]:
        """Return a run's parts ordered by position."""
        rows = self._connection.execute(
            "SELECT * FROM parts WHERE run_uuid = ? ORDER BY position ASC, part_uuid ASC",
            (run_uuid,),
        ).fetchall()
        return [_row_to_part(row) for row in rows]

    def get_attempts(self, run_uuid: str) -> list[AttemptRecord]:
        """Return a run's attempts ordered by creation time."""
        rows = self._connection.execute(
            "SELECT * FROM attempts WHERE run_uuid = ? ORDER BY created_at ASC, attempt_uuid ASC",
            (run_uuid,),
        ).fetchall()
        return [_row_to_attempt(row) for row in rows]

    def get_attempt(self, attempt_uuid: str) -> AttemptRecord | None:
        """Return one attempt by UUID, or ``None`` when absent."""
        row = self._connection.execute(
            "SELECT * FROM attempts WHERE attempt_uuid = ?", (attempt_uuid,)
        ).fetchone()
        return _row_to_attempt(row) if row is not None else None

    def get_artifacts(self, run_uuid: str) -> list[ArtifactRecord]:
        """Return a run's artifacts ordered by creation time."""
        rows = self._connection.execute(
            "SELECT * FROM artifacts WHERE run_uuid = ? ORDER BY created_at ASC, artifact_uuid ASC",
            (run_uuid,),
        ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def get_text_sources(self, run_uuid: str) -> list[TextSourceRecord]:
        """Return a run's text sources ordered by creation time."""
        rows = self._connection.execute(
            "SELECT * FROM text_sources WHERE run_uuid = ? "
            "ORDER BY created_at ASC, text_source_uuid ASC",
            (run_uuid,),
        ).fetchall()
        return [_row_to_text_source(row) for row in rows]

    # -- inserts used inside a caller-managed transaction ---------------------

    def _build_run_record(
        self,
        *,
        operation: str,
        run_root: str,
        status: str,
        user_label: str | None,
        parent_uuid: str | None,
        config_snapshot: dict[str, Any] | None,
        run_uuid: str | None,
        record_version: int,
        created_at: str | None,
        legacy_source_root: str | None,
    ) -> RunRecord:
        identifier = (
            _require_uuid(run_uuid, "run_uuid") if run_uuid is not None else str(uuid.uuid4())
        )
        now = created_at or utc_now()
        canonical_legacy = (
            _canonical_legacy_source_root(legacy_source_root)
            if legacy_source_root is not None
            else None
        )
        return RunRecord(
            run_uuid=identifier,
            operation=operation,
            status=status,
            run_root=run_root,
            legacy_source_root=canonical_legacy,
            user_label=user_label,
            parent_uuid=parent_uuid,
            config_snapshot=_redact_snapshot(config_snapshot),
            record_version=record_version,
            created_at=now,
            updated_at=now,
        )

    def _find_by_legacy_source_root(self, canonical_root: str) -> RunRecord | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE legacy_source_root = ?", (canonical_root,)
        ).fetchone()
        return _row_to_run(row) if row is not None else None

    def _insert_run(self, record: RunRecord) -> None:
        self._connection.execute(
            "INSERT INTO runs (run_uuid, operation, user_label, parent_uuid, status, run_root, "
            "legacy_source_root, config_snapshot, record_version, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.run_uuid,
                record.operation,
                record.user_label,
                record.parent_uuid,
                record.status,
                record.run_root,
                record.legacy_source_root,
                _dump_json(record.config_snapshot),
                record.record_version,
                record.created_at,
                record.updated_at,
            ),
        )

    def _insert_part(self, record: PartRecord) -> None:
        self._connection.execute(
            "INSERT INTO parts (part_uuid, run_uuid, position, prepared_text, voice, vibe_shared, "
            "vibe_specific, vibe_effective, fingerprint, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.part_uuid,
                record.run_uuid,
                record.position,
                record.prepared_text,
                record.voice,
                record.vibe_shared,
                record.vibe_specific,
                record.vibe_effective,
                record.fingerprint,
                record.stage,
                record.created_at,
                record.updated_at,
            ),
        )

    def _insert_attempt(self, record: AttemptRecord) -> None:
        self._connection.execute(
            "INSERT INTO attempts (attempt_uuid, run_uuid, part_uuid, call_type, provider, model, "
            "account_alias, remote_id, status, usage_json, cost, cost_raw, cost_currency, "
            "cost_source, cost_exact_available, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.attempt_uuid,
                record.run_uuid,
                record.part_uuid,
                record.call_type,
                record.provider,
                record.model,
                record.account_alias,
                record.remote_id,
                record.status,
                _dump_json(record.usage),
                record.cost.amount,
                record.cost.raw,
                record.cost.currency,
                record.cost.source,
                1 if record.cost.exact_available else 0,
                record.error,
                record.created_at,
                record.updated_at,
            ),
        )

    def _insert_artifact(self, record: ArtifactRecord) -> None:
        self._connection.execute(
            "INSERT INTO artifacts (artifact_uuid, run_uuid, part_uuid, attempt_uuid, role, "
            "path_kind, path, mime, size_bytes, sha256, media_metadata_json, availability, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.artifact_uuid,
                record.run_uuid,
                record.part_uuid,
                record.attempt_uuid,
                record.role,
                record.path_kind,
                record.path,
                record.mime,
                record.size_bytes,
                record.sha256,
                _dump_json(record.media_metadata),
                record.availability,
                record.created_at,
            ),
        )

    def _insert_text_source(self, record: TextSourceRecord) -> None:
        self._connection.execute(
            "INSERT INTO text_sources (text_source_uuid, run_uuid, part_uuid, artifact_uuid, kind, "
            "content, content_hash, language, origin, text_completeness, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.text_source_uuid,
                record.run_uuid,
                record.part_uuid,
                record.artifact_uuid,
                record.kind,
                record.content,
                record.content_hash,
                record.language,
                record.origin,
                record.text_completeness,
                record.created_at,
            ),
        )
