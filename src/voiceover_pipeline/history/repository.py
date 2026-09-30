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
  :meth:`HistoryRepository.advance_run_revision` is the typed revision
  compare-and-swap that resume and the compatibility state export build on.
  :meth:`HistoryRepository.reserve_paid_tts_attempt` is the single atomic
  pre-submit seam: it advances that revision and commits one ``submitting``
  attempt marker with no remote id and an unknown cost before the caller may
  make a paid request, and it refuses a second attempt for the same part and
  call type so an unconfirmed paid submit is never repeated. It owns its
  transaction and refuses to run inside a caller's open transaction, on an
  imported legacy run, or on a completed run.
  :meth:`HistoryRepository.record_polza_media_task_accepted` is the matching
  post-submit seam: in one transaction it binds the accepted opaque Polza Media
  task id to that reserved attempt and advances the revision again, so a later
  process may finish the already-paid task with GET calls instead of a second
  POST. It too owns its transaction and refuses to join a caller's.
  :meth:`HistoryRepository.record_polza_media_observed_cost` is the matching
  billing seam: it commits the amount a completed accepted media task reported
  before the caller may download its signed URL. The first observation is a
  compare-and-swap write; a repeated identical amount is idempotent, and a
  different amount is a typed conflict rather than a silent overwrite.
  :meth:`HistoryRepository.record_polza_media_raw_saved` is the linking seam: it
  re-verifies the already-saved paid raw receipt and bytes on disk against the
  run, part, and attempt identity outside any transaction, then links exactly one
  managed ``paid_raw_audio`` artifact and advances the attempt to ``raw_saved``
  in one short transaction. It never holds a transaction across file hashing.

Money contract: an attempt's cost is stored as a ``TEXT`` decimal string.
``NULL`` means *unknown*; the string ``"0"`` is a real observed zero. A value
imported from a legacy number keeps its captured decimal text (or the shortest
form of a float) but carries ``source="legacy_import"`` and
``exact_available=False``, so no reader can mistake it for a provider-confirmed
exact amount. A directly observed amount is ``source="exact"`` when it arrived
as a ``Decimal``, ``int``, or decimal string; a binary float is stored as its
shortest text with ``source="provider_observed_float"`` and
``exact_available=False``. There is one cost column per attempt, so an imported
legacy total is never counted twice.

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
from .raw_receipt import PaidRawReceipt, PaidRawReceiptError, verify_paid_raw_receipt

MAX_QUERY_LIMIT = 500
DEFAULT_QUERY_LIMIT = 50


class HistoryRepositoryError(RuntimeError):
    """A repository call violated the history write contract."""


class HistoryRevisionConflictError(HistoryRepositoryError):
    """A compare-and-swap run update found a revision other than the expected one."""


class HistoryRunNotFoundError(HistoryRepositoryError):
    """A run-scoped update named a UUID that no history run carries."""


class HistoryPartNotFoundError(HistoryRepositoryError):
    """A part-scoped call named a part UUID that does not belong to the run."""


class HistoryPaidAttemptConflictError(HistoryRepositoryError):
    """A paid attempt is already recorded for this run part and call type.

    Raised when a reservation would create a second paid submit for a part that
    already carries an attempt of the same call type, whatever that earlier
    attempt's outcome, so an unconfirmed paid submit is never repeated.
    """


class HistoryPaidMediaTaskConflictError(HistoryRepositoryError):
    """An accepted Polza Media task id cannot be bound to the named attempt.

    Raised when the target is not an existing ``submitting`` ``tts_chunk``
    attempt of the named run, part, and ``polza-tts`` media route whose model
    uses the async ``elevenlabs/`` prefix, when that attempt already carries a
    remote id or a non-``submitting`` outcome, or when the guarded attempt
    update affects no row. The accepted id is immutable, so a second or
    different id is never rebound, and no partial binding is returned.
    """


class HistoryPaidMediaCostConflictError(HistoryRepositoryError):
    """An observed Polza Media cost cannot be recorded on the named attempt.

    Raised when the target is not an accepted (``remote_accepted``) async
    ``elevenlabs/`` ``tts_chunk`` attempt of the named run and part carrying a
    bounded remote task id, or when the attempt already records a different
    billing observation. The first observed amount is immutable, so a different
    price is never silently written over it and no partial observation is
    returned.
    """


class HistoryPaidRawConflictError(HistoryRepositoryError):
    """Verified paid raw evidence cannot be linked to the named attempt.

    Raised when the target is not an accepted (``remote_accepted``) async
    ``elevenlabs/`` ``tts_chunk`` attempt of the named run and part carrying a
    bounded remote task id, when the attempt already carries an unexpected
    status or a partial or conflicting raw artifact, when the on-disk receipt
    and bytes do not verify against the database identity, or when a guarded
    write affects no row. The caller's receipt and its ``raw_path`` are never
    trusted: the evidence is re-read and re-hashed from disk against the run
    ``run_root``, part fingerprint, and attempt remote id before any mutation, so
    a corrupt, missing, or mismatched receipt fails closed without touching the
    database.
    """


class HistoryPaidAttemptInTransactionError(HistoryRepositoryError):
    """A paid attempt transition was attempted inside an already open transaction.

    The reservation and the accepted-id recording each own their transaction so
    the durable attempt marker commits before the caller may make the paid
    request or poll the accepted task. Joining a caller's still-open transaction
    would let a later rollback erase that marker after the request had already
    been sent, so the seam refuses to run.
    """


class HistoryRunNotReservableError(HistoryRepositoryError):
    """A paid attempt cannot be reserved for this run's current state.

    Raised for an imported legacy run, whose paid work may be recorded only as a
    run-level total that no part-linked guard can see, and for any run already
    marked ``completed``, because repeating finished work belongs to a new run.
    """


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
# Provenance for a provider amount that arrived as a binary float. It has no
# exact decimal representation, so only its shortest ``repr`` text is stored and
# no exact guarantee is claimed. It is deliberately not ``legacy_import``: this
# amount was observed during this run, not imported from an old number.
COST_SOURCE_PROVIDER_OBSERVED_FLOAT = "provider_observed_float"
# Polza Media bills ElevenLabs TTS in rubles.
POLZA_MEDIA_COST_CURRENCY = "RUB"
TEXT_KIND_TTS_SCRIPT = "tts_script"
TEXT_KIND_TTS_DIRECTION = "tts_direction"
TEXT_KIND_ASR_TRANSCRIPT = "asr_transcript"
TEXT_KIND_VERIFICATION_TRANSCRIPT = "verification_transcript"
# Call type and stage of the paid TTS attempt marker this repository commits
# before a network submit. The marker shares the legacy importer's chunk call
# type, so imported and native attempts are guarded by one predicate.
ATTEMPT_CALL_TYPE_TTS_CHUNK = "tts_chunk"
ATTEMPT_STATUS_SUBMITTING = "submitting"
# The attempt status once the provider accepted a paid Polza Media submit but its
# result has not been fetched yet. It is written in the same transaction as the
# accepted opaque task id, so a later process may poll the exact accepted task
# with GET calls instead of a second paid POST.
ATTEMPT_STATUS_REMOTE_ACCEPTED = "remote_accepted"
# The attempt status once its accepted paid raw response is preserved on disk and
# linked to a managed history artifact. The raw bytes and bounded receipt are
# already written; this marker records that the canonical history row exists.
ATTEMPT_STATUS_RAW_SAVED = "raw_saved"
# Role and private container MIME of the one managed artifact that links a saved
# paid raw response to its part and attempt. The format set mirrors the bounded
# raw formats of ``history.raw_receipt`` and ``run_state``.
ARTIFACT_ROLE_PAID_RAW_AUDIO = "paid_raw_audio"
_RAW_AUDIO_MIME_BY_FORMAT = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm16": "audio/pcm"}
# Part stage written once its paid raw response is linked, but only while the
# part stage is still unset: an imported or already-staged part keeps its own.
PART_STAGE_RAW_SAVED = "raw_saved"
# The only provider route whose accepted media task id this repository records.
POLZA_TTS_PROVIDER_ID = "polza-tts"
# A run whose work already finished. Repeating finished work creates a new run
# with a link to the previous one, not another paid attempt on the closed run.
RUN_STATUS_COMPLETED = "completed"
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
    re.compile(r"\b(?:proxy-)?authorization\s*:", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bbasic\s+\S+", re.IGNORECASE),
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


def _require_expected_revision(expected_revision: int) -> int:
    """Return a positive, non-bool revision or raise loudly."""
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 1
    ):
        raise ValueError("expected_revision must be a positive integer")
    return expected_revision


def _polza_media_route_model(model: object) -> bool:
    """Whether a Polza TTS model submits through the async ``/media`` route.

    Only the ElevenLabs ``elevenlabs/`` models POST to ``/media`` and can hold a
    recoverable task id; every other ``polza-tts`` model uses the synchronous
    ``/audio/speech`` route, so an accepted marker next to it can never name an
    async media task. This mirrors ``services.recovery.polza_media_route_model``
    as a pure prefix predicate without importing the higher service layer here.
    """
    return isinstance(model, str) and model.startswith("elevenlabs/")


# An accepted paid media task id becomes part of a later ``/media/<id>`` GET
# path, so only an opaque token may be stored. This is the same bounded contract
# as ``run_state._bounded_media_task_id`` and ``polza_tts._safe_media_id``: a URL,
# scheme, slash, query, whitespace, or newline could change the request target.
_MEDIA_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _require_bounded_media_task_id(value: str) -> str:
    """Return a bounded opaque media task id, or raise without echoing the input.

    The accepted id is spliced into a later ``/media/<id>`` GET path, so only an
    opaque token may be stored. The fixed message keeps an unsafe value (for
    example a signed URL or tampered provider output) out of an error report or
    log.
    """
    if not isinstance(value, str) or _MEDIA_TASK_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("remote_task_id is not a bounded opaque media task id")
    return value


def _require_identity_field(value: str, field_name: str) -> str:
    """Return a non-empty identity string, or raise loudly before any write."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


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


# A fixed rejection message for a directly observed amount that is not an exact
# Decimal/int/decimal string or a finite float. It never echoes the value, so an
# unsafe token (for example a signed URL) cannot reach a report or log.
_OBSERVED_MEDIA_COST_REJECTED = (
    "observed media cost must be an exact Decimal, int, or decimal string, or a finite float"
)


def _observed_media_cost(value: Decimal | int | float | str) -> Cost:
    """Normalize one directly observed Polza Media amount to an exact-or-float cost.

    Only a ``Decimal``, an ``int``, a decimal string, or a finite binary ``float``
    is accepted; an unbounded usage mapping or body never reaches this seam. A
    ``Decimal``/``int``/string keeps its captured decimal text (``"0.1000"``
    stays ``"0.1000"``) and is marked exact. A binary float has no exact decimal
    representation, so its shortest ``repr`` text is stored with
    :data:`COST_SOURCE_PROVIDER_OBSERVED_FLOAT` and ``exact_available=False``.
    An observed zero is a real fact and stays distinct from the unknown ``NULL``.
    A ``None``, ``bool``, container, malformed string, or non-finite number is
    rejected with a fixed message that never echoes the value.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(_OBSERVED_MEDIA_COST_REJECTED)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(_OBSERVED_MEDIA_COST_REJECTED)
        text = repr(value)
        return Cost(
            amount=text,
            currency=POLZA_MEDIA_COST_CURRENCY,
            source=COST_SOURCE_PROVIDER_OBSERVED_FLOAT,
            exact_available=False,
            raw=text,
        )
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(_OBSERVED_MEDIA_COST_REJECTED) from exc
        if not parsed.is_finite():
            raise ValueError(_OBSERVED_MEDIA_COST_REJECTED)
        return Cost(
            amount=value,
            currency=POLZA_MEDIA_COST_CURRENCY,
            source=COST_SOURCE_EXACT,
            exact_available=True,
        )
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(_OBSERVED_MEDIA_COST_REJECTED)
        return Cost(
            amount=str(value),
            currency=POLZA_MEDIA_COST_CURRENCY,
            source=COST_SOURCE_EXACT,
            exact_available=True,
        )
    if isinstance(value, int):
        return Cost(
            amount=str(value),
            currency=POLZA_MEDIA_COST_CURRENCY,
            source=COST_SOURCE_EXACT,
            exact_available=True,
        )
    raise ValueError(_OBSERVED_MEDIA_COST_REJECTED)


def _same_observed_cost(stored: Cost, observed: Cost) -> bool:
    """Whether two observed costs are the same billing observation.

    Amount text, currency, provenance, and the exact guarantee must all match, so
    a repeated identical observation is idempotent while a different price or a
    weaker provenance is a conflict that never overwrites the first value.
    """
    return (
        stored.amount == observed.amount
        and stored.currency == observed.currency
        and stored.source == observed.source
        and stored.exact_available == observed.exact_available
    )


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
    revision: int = 1
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
    # A read-only v1 snapshot has no ``revision`` column yet; project the pre-v2
    # default 1 without mutating the schema or requiring a migration.
    revision = int(row["revision"]) if "revision" in row.keys() else 1
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
        revision=revision,
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

    def _require_no_open_transaction(self, method: str) -> None:
        """Fail closed when a paid reservation would join an open transaction.

        The reservation commits the unconfirmed attempt marker before the caller
        may start the paid request. Inside a caller-managed :meth:`transaction` a
        later rollback would erase that marker after the request was already sent,
        so the method refuses to run instead of returning a value the caller could
        act on and then lose.
        """
        if self._connection.in_transaction:
            raise HistoryPaidAttemptInTransactionError(
                f"{method} must not run inside an open transaction; call it on its own so "
                "the attempt marker commits before any paid request"
            )

    def _require_run_paid_transition_allowed(self, run: RunRecord) -> None:
        """Fail closed when a paid transition must not extend this run.

        The guard shared by the pre-submit reservation and the post-submit
        accepted-id transition: an imported legacy run's paid work may live only
        in a run-level total that no part-linked guard can see, and a completed
        run's finished work belongs to a new run, so neither may gain a
        part-linked paid attempt. Both leave the run unchanged.
        """
        if run.legacy_source_root is not None:
            raise HistoryRunNotReservableError(
                f"run {run.run_uuid!r} is an imported legacy run whose paid work cannot be "
                "represented by a part-linked attempt; refusing a new paid transition"
            )
        if run.status == RUN_STATUS_COMPLETED:
            raise HistoryRunNotReservableError(
                f"run {run.run_uuid!r} is already completed; repeat finished work in a new run"
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

    def advance_run_revision(
        self,
        run_uuid: str,
        *,
        expected_revision: int,
        status: str | None = None,
    ) -> RunRecord:
        """Compare-and-swap a run's revision and optionally its status.

        The update matches only ``run_uuid`` at exactly ``expected_revision`` and
        increments the revision by one, so a stale writer cannot advance a run
        that another process already moved. ``updated_at`` is refreshed; ``status``
        is written only when supplied, otherwise the stored value is kept. The
        call joins an open :meth:`transaction` when the caller has one, so it
        rolls back together with the caller's other writes on a later failure.

        ``expected_revision`` must be a positive, non-bool integer. Raises
        :class:`HistoryRevisionConflictError` when the run exists at a different
        revision and :class:`HistoryRunNotFoundError` when no run carries
        ``run_uuid``. Both leave the run unchanged: the compare-and-swap never
        runs a bare ``UPDATE`` without the revision predicate.
        """
        _require_expected_revision(expected_revision)
        now = utc_now()
        with self.transaction():
            cursor = self._connection.execute(
                "UPDATE runs SET revision = revision + 1, updated_at = ?, "
                "status = COALESCE(?, status) "
                "WHERE run_uuid = ? AND revision = ?",
                (now, status, run_uuid, expected_revision),
            )
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_uuid = ?", (run_uuid,)
            ).fetchone()
            if cursor.rowcount != 1 or row is None:
                if row is None:
                    raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
                raise HistoryRevisionConflictError(
                    f"run {run_uuid!r} is not at revision {expected_revision}"
                )
            return _row_to_run(row)

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

    def reserve_paid_tts_attempt(
        self,
        run_uuid: str,
        *,
        part_uuid: str,
        expected_revision: int,
        provider: str,
        model: str,
        account_alias: str,
    ) -> tuple[RunRecord, AttemptRecord]:
        """Atomically reserve one paid TTS attempt for a run part before any submit.

        This is the single pre-submit seam for paid generation. Inside one
        :meth:`transaction` it validates that the run exists and that ``part_uuid``
        belongs to it, refuses a part that already carries any attempt of the same
        call type, compare-and-swaps the run revision through
        :meth:`advance_run_revision`, and inserts one ``submitting`` attempt marker
        with ``remote_id=None`` and :meth:`Cost.unknown`. It returns
        ``(advanced_run, attempt)``, where ``advanced_run.revision`` is the new
        revision. The caller may only start the network request *after* this call
        returns, so a crash between the marker and the response leaves a durable
        unconfirmed marker instead of a silently repeatable submit.

        The marker is committed before any provider call and must stay until the
        outcome is resolved. Any earlier attempt for the same part and call type
        blocks a new reservation even when ``expected_revision`` is still fresh,
        because an ``outcome_unknown`` or accepted paid submit must never turn into
        a second POST. A *known definitive* failure is not yet distinguishable at
        this layer, so every earlier attempt of that call type is refused; a
        narrower recovery policy remains a future slice. ``provider``, ``model``,
        and ``account_alias`` must be non-empty.

        The call owns its transaction and refuses to run inside an already open
        one, so its committed marker cannot be erased by a caller rollback after
        the request was sent. It also refuses a run that already carries a
        ``legacy_source_root`` (an imported run's paid work may live only in a
        run-level total no part-linked guard can see) and any run already in
        :data:`RUN_STATUS_COMPLETED` (repeating finished work belongs to a new
        run), leaving both unchanged.

        Raises :class:`ValueError` for an invalid ``expected_revision`` or an empty
        identity field before any database write, :class:`HistoryPaidAttemptInTransactionError`
        when called inside an open transaction, :class:`HistoryRunNotFoundError`
        for a missing run, :class:`HistoryRunNotReservableError` for a legacy or
        completed run, :class:`HistoryPartNotFoundError` for a foreign or absent
        part, :class:`HistoryPaidAttemptConflictError` for an existing attempt, and
        :class:`HistoryRevisionConflictError` for a stale revision. Every failure
        rolls the revision bump and the insertion back together.
        """
        part_identifier = _require_uuid(part_uuid, "part_uuid")
        _require_identity_field(provider, "provider")
        _require_identity_field(model, "model")
        _require_identity_field(account_alias, "account_alias")
        _require_expected_revision(expected_revision)
        self._require_no_open_transaction("reserve_paid_tts_attempt")
        with self.transaction():
            run = self.get_run(run_uuid)
            if run is None:
                raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
            self._require_run_paid_transition_allowed(run)
            if not self._part_belongs_to_run(part_identifier, run_uuid):
                raise HistoryPartNotFoundError(
                    f"part {part_identifier!r} does not belong to run {run_uuid!r}"
                )
            existing_attempt = self._find_attempt_uuid_for_part(
                run_uuid, part_identifier, ATTEMPT_CALL_TYPE_TTS_CHUNK
            )
            if existing_attempt is not None:
                raise HistoryPaidAttemptConflictError(
                    f"run {run_uuid!r} part {part_identifier!r} already has a paid "
                    f"{ATTEMPT_CALL_TYPE_TTS_CHUNK!r} attempt {existing_attempt!r}; "
                    "refusing a second submit"
                )
            advanced = self.advance_run_revision(run_uuid, expected_revision=expected_revision)
            attempt = self.add_attempt(
                run_uuid,
                call_type=ATTEMPT_CALL_TYPE_TTS_CHUNK,
                part_uuid=part_identifier,
                provider=provider,
                model=model,
                account_alias=account_alias,
                remote_id=None,
                status=ATTEMPT_STATUS_SUBMITTING,
                cost=Cost.unknown(),
            )
            return advanced, attempt

    def record_polza_media_task_accepted(
        self,
        run_uuid: str,
        *,
        attempt_uuid: str,
        part_uuid: str,
        expected_revision: int,
        remote_task_id: str,
    ) -> tuple[RunRecord, AttemptRecord]:
        """Bind an accepted Polza Media task id to a reserved paid attempt durably.

        This is the single seam that records the opaque task id an ElevenLabs
        ``/media`` paid submit returned, before the caller may poll or download
        that task. Inside one :meth:`transaction` it validates that ``attempt_uuid``
        is an existing ``submitting`` ``tts_chunk`` attempt of ``run_uuid`` and
        ``part_uuid`` whose provider is ``polza-tts``, whose model uses the async
        ``elevenlabs/`` media route, and whose ``remote_id`` is still unknown,
        compare-and-swaps the run revision through :meth:`advance_run_revision`,
        and writes the bounded task id together with the
        :data:`ATTEMPT_STATUS_REMOTE_ACCEPTED` status. It returns
        ``(advanced_run, accepted_attempt)``, where ``advanced_run.revision`` is
        the new revision and ``accepted_attempt.remote_id`` is the stored id.

        The transition commits before the caller may issue a GET, so a crash
        between the accepted response and the poll leaves a durable accepted task
        id exactly once. The accepted id is immutable: the attempt's provider,
        model, account alias, and unknown cost are never rewritten, and a second
        or different id is refused rather than rebound. ``remote_task_id`` must be
        a bounded opaque token (``[A-Za-z0-9_-]{1,128}``) matching the contract of
        ``run_state._bounded_media_task_id`` and ``polza_tts._safe_media_id``; a
        URL, secret, or newline is rejected with a fixed message that never echoes
        the unsafe value.

        The guarded attempt update must affect exactly one row. If a trigger,
        constraint, or lost race makes it affect none, the transition raises
        :class:`HistoryPaidMediaTaskConflictError` so the whole transaction,
        including the revision bump, rolls back instead of returning a partial
        accepted-id binding.

        The call owns its transaction and refuses to run inside an already open
        one, so its committed transition cannot be erased by a caller rollback
        after a GET was sent. Like the pre-submit reservation, it also refuses an
        imported legacy run and any run already in :data:`RUN_STATUS_COMPLETED`,
        leaving both unchanged. Raises :class:`ValueError` for an invalid
        ``expected_revision``, a non-UUID attempt or part, or a malformed task id
        before any write; :class:`HistoryPaidAttemptInTransactionError` inside an
        open transaction; :class:`HistoryRunNotFoundError` for a missing run;
        :class:`HistoryRunNotReservableError` for a legacy or completed run;
        :class:`HistoryPartNotFoundError` for a foreign or absent part;
        :class:`HistoryPaidMediaTaskConflictError` when the attempt does not match
        the named run, part, call type, model media route, and ``submitting``
        state, when it already carries a remote id, or when the guarded update
        affects no row; and :class:`HistoryRevisionConflictError` for a stale
        revision. Every failure leaves the run and attempt unchanged.
        """
        attempt_identifier = _require_uuid(attempt_uuid, "attempt_uuid")
        part_identifier = _require_uuid(part_uuid, "part_uuid")
        _require_expected_revision(expected_revision)
        bounded_task_id = _require_bounded_media_task_id(remote_task_id)
        self._require_no_open_transaction("record_polza_media_task_accepted")
        with self.transaction():
            run = self.get_run(run_uuid)
            if run is None:
                raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
            self._require_run_paid_transition_allowed(run)
            if not self._part_belongs_to_run(part_identifier, run_uuid):
                raise HistoryPartNotFoundError(
                    f"part {part_identifier!r} does not belong to run {run_uuid!r}"
                )
            attempt = self.get_attempt(attempt_identifier)
            if (
                attempt is None
                or attempt.run_uuid != run_uuid
                or attempt.part_uuid != part_identifier
            ):
                raise HistoryPaidMediaTaskConflictError(
                    f"no paid attempt {attempt_identifier!r} for run {run_uuid!r} "
                    f"part {part_identifier!r}"
                )
            if (
                attempt.call_type != ATTEMPT_CALL_TYPE_TTS_CHUNK
                or attempt.provider != POLZA_TTS_PROVIDER_ID
            ):
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} is not a {POLZA_TTS_PROVIDER_ID!r} "
                    f"{ATTEMPT_CALL_TYPE_TTS_CHUNK!r} media attempt"
                )
            # Only an ``elevenlabs/`` model submits through the async ``/media``
            # route and can hold a recoverable task id; a synchronous model's
            # accepted marker would name a task that can never be polled.
            if not _polza_media_route_model(attempt.model):
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} model {attempt.model!r} does not use the "
                    "async Polza /media route; it cannot carry a recoverable media task id"
                )
            if attempt.status != ATTEMPT_STATUS_SUBMITTING:
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} is not {ATTEMPT_STATUS_SUBMITTING!r}"
                )
            if attempt.remote_id is not None:
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} already records an accepted remote task id"
                )
            advanced = self.advance_run_revision(run_uuid, expected_revision=expected_revision)
            # Only the accepted id, the status, and the timestamp change: the
            # provider, model, account alias, usage, and unknown cost stay as the
            # reservation wrote them, so no observed cost can be fabricated here.
            cursor = self._connection.execute(
                "UPDATE attempts SET remote_id = ?, status = ?, updated_at = ? "
                "WHERE attempt_uuid = ? AND remote_id IS NULL AND status = ?",
                (
                    bounded_task_id,
                    ATTEMPT_STATUS_REMOTE_ACCEPTED,
                    utc_now(),
                    attempt_identifier,
                    ATTEMPT_STATUS_SUBMITTING,
                ),
            )
            # The guarded write must be the only one that binds the id. A trigger,
            # constraint, or lost race that makes it affect no row cannot yield a
            # successful accepted id with an advanced revision; raise so the whole
            # transaction, including the revision bump, rolls back.
            if cursor.rowcount != 1:
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} was not an unbound "
                    f"{ATTEMPT_STATUS_SUBMITTING!r} attempt during the transition; "
                    "refusing a partial accepted-id binding"
                )
            updated = self.get_attempt(attempt_identifier)
            if updated is None:  # pragma: no cover - the attempt was just updated
                raise HistoryPaidMediaTaskConflictError(
                    f"attempt {attempt_identifier!r} vanished during the transition"
                )
            return advanced, updated

    def record_polza_media_observed_cost(
        self,
        run_uuid: str,
        *,
        attempt_uuid: str,
        part_uuid: str,
        expected_revision: int,
        amount: Decimal | int | float | str,
    ) -> tuple[RunRecord, AttemptRecord]:
        """Commit the amount a completed accepted Polza Media task reported, before download.

        This is the single billing seam for a paid ElevenLabs ``/media`` chunk. It
        must run after :meth:`record_polza_media_task_accepted` and before the
        caller downloads the task's signed URL, so a download that fails cannot
        lose the cost the provider already reported. Inside one :meth:`transaction`
        it validates that ``attempt_uuid`` is an accepted async ``elevenlabs/``
        ``tts_chunk`` attempt of ``run_uuid`` and ``part_uuid`` whose provider is
        ``polza-tts`` and whose bounded remote task id is present, then records the
        first observation, compare-and-swapping the run revision through
        :meth:`advance_run_revision`. It returns ``(current_run, attempt)``, where
        ``current_run.revision`` is the advanced revision for a first observation
        and the unchanged run for an idempotent repeat.

        Only a direct amount value is accepted: an exact ``Decimal``, ``int``, or
        decimal string keeps its captured text and is marked exact with currency
        RUB, while a finite binary ``float`` is stored as its shortest ``repr``
        text with :data:`COST_SOURCE_PROVIDER_OBSERVED_FLOAT` and
        ``exact_available=False``. A ``None``, ``bool``, container, malformed
        string, or non-finite number is rejected with :class:`ValueError` before
        any write, so a missing usage never erases an observed amount and an
        unbounded usage mapping never reaches this seam.

        The first billing observation is immutable. A repeated identical amount
        is idempotent and does not advance the revision when ``expected_revision``
        is the run's current revision, so a GET-only resume may poll the same task
        again safely. A different amount raises
        :class:`HistoryPaidMediaCostConflictError` instead of silently overwriting
        the first observation; reconciling a changed provider price is a separate,
        explicit future concern. The guarded cost update must affect exactly one
        row: if a trigger, constraint, or lost race makes it affect none, the seam
        raises the same conflict so the whole transaction, including the revision
        bump, rolls back instead of returning a partial observation.

        The call owns its transaction and refuses to run inside an already open
        one, so its committed observation cannot be erased by a caller rollback
        after the download starts. Like the other paid seams it refuses an
        imported legacy run and any run already in :data:`RUN_STATUS_COMPLETED`,
        leaving both unchanged. Raises :class:`ValueError` for an invalid
        ``expected_revision``, a non-UUID attempt or part, or an unusable amount
        before any write; :class:`HistoryPaidAttemptInTransactionError` inside an
        open transaction; :class:`HistoryRunNotFoundError` for a missing run;
        :class:`HistoryRunNotReservableError` for a legacy or completed run;
        :class:`HistoryPartNotFoundError` for a foreign or absent part;
        :class:`HistoryPaidMediaCostConflictError` when the attempt does not match
        the named run, part, call type, async media route, acceptance status, and
        bounded remote task id, when it already records a different amount, or when
        the guarded update affects no row; and :class:`HistoryRevisionConflictError`
        for a stale revision on a first or repeated observation. Every failure
        leaves the run and attempt unchanged.
        """
        attempt_identifier = _require_uuid(attempt_uuid, "attempt_uuid")
        part_identifier = _require_uuid(part_uuid, "part_uuid")
        _require_expected_revision(expected_revision)
        observed = _observed_media_cost(amount)
        self._require_no_open_transaction("record_polza_media_observed_cost")
        with self.transaction():
            run = self.get_run(run_uuid)
            if run is None:
                raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
            self._require_run_paid_transition_allowed(run)
            if not self._part_belongs_to_run(part_identifier, run_uuid):
                raise HistoryPartNotFoundError(
                    f"part {part_identifier!r} does not belong to run {run_uuid!r}"
                )
            attempt = self.get_attempt(attempt_identifier)
            if (
                attempt is None
                or attempt.run_uuid != run_uuid
                or attempt.part_uuid != part_identifier
            ):
                raise HistoryPaidMediaCostConflictError(
                    f"no paid attempt {attempt_identifier!r} for run {run_uuid!r} "
                    f"part {part_identifier!r}"
                )
            if (
                attempt.call_type != ATTEMPT_CALL_TYPE_TTS_CHUNK
                or attempt.provider != POLZA_TTS_PROVIDER_ID
                or not _polza_media_route_model(attempt.model)
            ):
                raise HistoryPaidMediaCostConflictError(
                    f"attempt {attempt_identifier!r} is not a {POLZA_TTS_PROVIDER_ID!r} "
                    f"{ATTEMPT_CALL_TYPE_TTS_CHUNK!r} async media attempt"
                )
            if attempt.status != ATTEMPT_STATUS_REMOTE_ACCEPTED:
                raise HistoryPaidMediaCostConflictError(
                    f"attempt {attempt_identifier!r} is not {ATTEMPT_STATUS_REMOTE_ACCEPTED!r}"
                )
            if (
                attempt.remote_id is None
                or _MEDIA_TASK_ID_PATTERN.fullmatch(attempt.remote_id) is None
            ):
                raise HistoryPaidMediaCostConflictError(
                    f"attempt {attempt_identifier!r} does not carry a bounded accepted remote "
                    "task id"
                )
            if attempt.cost.amount is not None:
                # The first billing observation is immutable: repeat the same
                # amount idempotently, but never overwrite it with a different one.
                if not _same_observed_cost(attempt.cost, observed):
                    raise HistoryPaidMediaCostConflictError(
                        f"attempt {attempt_identifier!r} already records a different observed "
                        "cost; refusing to overwrite the first billing observation"
                    )
                # A repeat is idempotent only while the caller's revision is
                # current, so a stale caller cannot treat a completed write as its
                # own and skip a needed reload.
                if run.revision != expected_revision:
                    raise HistoryRevisionConflictError(
                        f"run {run_uuid!r} is not at revision {expected_revision}"
                    )
                return run, attempt
            advanced = self.advance_run_revision(run_uuid, expected_revision=expected_revision)
            cursor = self._connection.execute(
                "UPDATE attempts SET cost = ?, cost_raw = ?, cost_currency = ?, "
                "cost_source = ?, cost_exact_available = ?, updated_at = ? "
                "WHERE attempt_uuid = ? AND status = ? AND cost IS NULL",
                (
                    observed.amount,
                    observed.raw,
                    observed.currency,
                    observed.source,
                    1 if observed.exact_available else 0,
                    utc_now(),
                    attempt_identifier,
                    ATTEMPT_STATUS_REMOTE_ACCEPTED,
                ),
            )
            # The guarded write must be the only one that records the amount. A
            # trigger, constraint, or lost race that makes it affect no row cannot
            # yield a successful observation with an advanced revision; raise so
            # the whole transaction, including the revision bump, rolls back.
            if cursor.rowcount != 1:
                raise HistoryPaidMediaCostConflictError(
                    f"attempt {attempt_identifier!r} did not carry an unobserved accepted cost "
                    "during the transition; refusing a partial cost observation"
                )
            updated = self.get_attempt(attempt_identifier)
            if updated is None:  # pragma: no cover - the attempt was just updated
                raise HistoryPaidMediaCostConflictError(
                    f"attempt {attempt_identifier!r} vanished during the transition"
                )
            return advanced, updated

    def record_polza_media_raw_saved(
        self,
        run_uuid: str,
        *,
        attempt_uuid: str,
        part_uuid: str,
        expected_revision: int,
        receipt: PaidRawReceipt,
    ) -> tuple[RunRecord, AttemptRecord, ArtifactRecord]:
        """Link already-saved paid raw evidence to its accepted attempt durably.

        This is the single seam that turns accepted paid raw audio, already on
        disk under the deterministic ``raw/<chunk-id>.<ext>`` path with its
        bounded emergency receipt, into one canonical ``managed_relative``
        history artifact bound to the run part and attempt, and advances that
        attempt to :data:`ATTEMPT_STATUS_RAW_SAVED`. It must run after
        :meth:`record_polza_media_task_accepted` (and, when a completed poll
        reported it, :meth:`record_polza_media_observed_cost`) so the accepted
        remote id and the exact observed cost already exist and are never
        rewritten.

        The caller-supplied
        :class:`~voiceover_pipeline.history.raw_receipt.PaidRawReceipt` is only a
        locator: its ``raw_path`` is ignored and its own identity is not trusted.
        The on-disk receipt and raw bytes are re-read and re-hashed and their
        bounded identity is checked against the database run ``run_root``, the
        part UUID, synthesis fingerprint, and position, and the attempt UUID and
        bounded remote id *outside* any transaction, so no SQLite transaction is
        ever held across file hashing. Only then does the method own one short
        transaction: it rechecks the run, its ``run_root``, part position, attempt,
        remote id, and receipt identity, applies the current ``expected_revision``
        compare-and-swap,
        inserts exactly one :data:`ARTIFACT_ROLE_PAID_RAW_AUDIO` artifact carrying
        the verified digest, size, MIME, format, and generation id, updates the
        attempt to :data:`ATTEMPT_STATUS_RAW_SAVED`, and leaves the part stage at
        :data:`PART_STAGE_RAW_SAVED` when it was still unset, all atomically.

        A repeated identical call whose ``expected_revision`` is still current is
        an idempotent no-op: it advances no revision and inserts no second
        artifact, so a crash after the database commit but before any later JSON
        compatibility export can be retried safely without a provider call. A
        stale revision, a different remote id, a foreign part, a receipt number
        that does not match the part position, an attempt that is not in an
        accepted state, a partial or conflicting artifact, or a receipt, digest,
        size, or path that does not verify fails closed with no database
        mutation. The already observed exact cost and the accepted
        remote id are never changed, and no provider, network, or FFmpeg path is
        ever opened.

        The call owns its transaction and refuses to run inside an already open
        one. Raises :class:`ValueError` for an invalid ``expected_revision``, a
        non-UUID attempt or part, or a non-receipt before any database write;
        :class:`HistoryPaidAttemptInTransactionError` inside an open transaction;
        :class:`HistoryRunNotFoundError` for a missing run;
        :class:`HistoryRunNotReservableError` for a legacy or completed run;
        :class:`HistoryPartNotFoundError` for a foreign or absent part;
        :class:`HistoryPaidRawConflictError` when the attempt does not match the
        named run, part, call type, async media route, accepted state, and
        bounded remote id, when the verified evidence does not match the database
        identity, when ``runs.run_root`` changed after that verification, when an
        artifact is partial or conflicting, or when a guarded
        write affects no row; and :class:`HistoryRevisionConflictError` for a
        stale revision.
        """
        attempt_identifier = _require_uuid(attempt_uuid, "attempt_uuid")
        part_identifier = _require_uuid(part_uuid, "part_uuid")
        _require_expected_revision(expected_revision)
        if not isinstance(receipt, PaidRawReceipt):
            raise ValueError("receipt must be a PaidRawReceipt")
        self._require_no_open_transaction("record_polza_media_raw_saved")

        # Identity reads and the on-disk verification stay outside any
        # transaction: hashing the raw bytes must never hold the write lock.
        run = self.get_run(run_uuid)
        if run is None:
            raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
        self._require_run_paid_transition_allowed(run)
        part = self._get_part(part_identifier)
        if part is None or part.run_uuid != run_uuid:
            raise HistoryPartNotFoundError(
                f"part {part_identifier!r} does not belong to run {run_uuid!r}"
            )
        attempt = self.get_attempt(attempt_identifier)
        self._require_raw_saved_attempt(attempt, run_uuid, part_identifier)
        assert attempt is not None  # narrowed by the guard above
        if receipt.number != part.position:
            # The caller receipt only locates the raw bytes; its number is still
            # bounded to the generated chunk id. Linking chunk_02 to part
            # position 1 (equal-fingerprint parts or a caller bug) would bind the
            # wrong paid bytes, so refuse before any hashing or database write.
            raise HistoryPaidRawConflictError(
                f"paid raw receipt number does not match part {part_identifier!r} position; "
                "refusing to link it"
            )
        verified = self._verify_media_raw_receipt(run, part, attempt, receipt)

        with self.transaction():
            current_run = self.get_run(run_uuid)
            if current_run is None:  # pragma: no cover - the run existed a moment ago
                raise HistoryRunNotFoundError(f"no history run with UUID {run_uuid!r}")
            self._require_run_paid_transition_allowed(current_run)
            # The evidence was verified on disk against ``run.run_root`` outside
            # this transaction. A concurrent writer can change ``runs.run_root``
            # without a revision bump between that verification and here, so the
            # verified relative path would then point into a root whose bytes were
            # never checked. Re-read it and refuse when it moved.
            if current_run.run_root != run.run_root:
                raise HistoryPaidRawConflictError(
                    f"run {run_uuid!r} run_root changed during the raw-saved transition; "
                    "refusing to link evidence verified against a different root"
                )
            current_part = self._get_part(part_identifier)
            if (
                current_part is None
                or current_part.run_uuid != run_uuid
                or current_part.fingerprint != verified.synthesis_fingerprint
                or current_part.position != verified.number
            ):
                raise HistoryPaidRawConflictError(
                    f"part {part_identifier!r} identity changed during the raw-saved transition"
                )
            current_attempt = self.get_attempt(attempt_identifier)
            self._require_raw_saved_attempt(current_attempt, run_uuid, part_identifier)
            assert current_attempt is not None  # narrowed by the guard above
            if current_attempt.remote_id != verified.remote_task_id:
                raise HistoryPaidRawConflictError(
                    f"attempt {attempt_identifier!r} remote identity changed during the "
                    "raw-saved transition"
                )
            existing = self._raw_saved_artifact(
                run_uuid, part_identifier, attempt_identifier, verified
            )
            if current_attempt.status == ATTEMPT_STATUS_RAW_SAVED:
                # A verified duplicate is idempotent only while the caller's
                # revision is current, so a stale caller cannot treat a
                # committed write as its own and skip a needed reload.
                if existing is None:
                    raise HistoryPaidRawConflictError(
                        f"attempt {attempt_identifier!r} records {ATTEMPT_STATUS_RAW_SAVED!r} "
                        "but its paid raw artifact is missing"
                    )
                if current_run.revision != expected_revision:
                    raise HistoryRevisionConflictError(
                        f"run {run_uuid!r} is not at revision {expected_revision}"
                    )
                return current_run, current_attempt, existing
            if existing is not None:
                raise HistoryPaidRawConflictError(
                    f"attempt {attempt_identifier!r} already carries a paid raw artifact but is "
                    f"not {ATTEMPT_STATUS_RAW_SAVED!r}"
                )
            advanced = self.advance_run_revision(run_uuid, expected_revision=expected_revision)
            artifact = self.add_artifact(
                run_uuid,
                role=ARTIFACT_ROLE_PAID_RAW_AUDIO,
                path_kind=PATH_KIND_MANAGED_RELATIVE,
                path=verified.relative_path,
                part_uuid=part_identifier,
                attempt_uuid=attempt_identifier,
                mime=_RAW_AUDIO_MIME_BY_FORMAT[verified.audio_format],
                size_bytes=verified.size,
                sha256=verified.sha256,
                media_metadata={
                    "format": verified.audio_format,
                    "chunk_id": verified.chunk_id,
                    "number": verified.number,
                    "generation_id": verified.generation_id,
                },
            )
            # Only the status and timestamp change: the provider, model, account
            # alias, accepted remote id, usage, and observed exact cost stay as
            # earlier seams wrote them, so no amount or id can be rewritten here.
            cursor = self._connection.execute(
                "UPDATE attempts SET status = ?, updated_at = ? "
                "WHERE attempt_uuid = ? AND status = ? AND remote_id = ?",
                (
                    ATTEMPT_STATUS_RAW_SAVED,
                    utc_now(),
                    attempt_identifier,
                    ATTEMPT_STATUS_REMOTE_ACCEPTED,
                    verified.remote_task_id,
                ),
            )
            # The guarded write must be the only one that marks the raw saved. A
            # trigger, constraint, or lost race that makes it affect no row
            # cannot yield a linked artifact with an advanced revision: raise so
            # the whole transaction, including the artifact and revision bump,
            # rolls back.
            if cursor.rowcount != 1:
                raise HistoryPaidRawConflictError(
                    f"attempt {attempt_identifier!r} was not an accepted "
                    f"{ATTEMPT_STATUS_REMOTE_ACCEPTED!r} attempt during the raw-saved "
                    "transition; refusing a partial artifact"
                )
            # Best-effort coherence for a part that never had a stage: an
            # imported or already-staged part keeps its own, so a zero rowcount
            # here is legitimate and not an error.
            self._connection.execute(
                "UPDATE parts SET stage = ?, updated_at = ? "
                "WHERE part_uuid = ? AND run_uuid = ? AND stage IS NULL",
                (PART_STAGE_RAW_SAVED, utc_now(), part_identifier, run_uuid),
            )
            updated = self.get_attempt(attempt_identifier)
            if updated is None:  # pragma: no cover - the attempt was just updated
                raise HistoryPaidRawConflictError(
                    f"attempt {attempt_identifier!r} vanished during the transition"
                )
            return advanced, updated, artifact

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

    def _part_belongs_to_run(self, part_uuid: str, run_uuid: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM parts WHERE part_uuid = ? AND run_uuid = ?",
            (part_uuid, run_uuid),
        ).fetchone()
        return row is not None

    def _get_part(self, part_uuid: str) -> PartRecord | None:
        """Return one part by UUID, or ``None`` when absent."""
        row = self._connection.execute(
            "SELECT * FROM parts WHERE part_uuid = ?", (part_uuid,)
        ).fetchone()
        return _row_to_part(row) if row is not None else None

    def _require_raw_saved_attempt(
        self, attempt: AttemptRecord | None, run_uuid: str, part_uuid: str
    ) -> None:
        """Fail closed unless the attempt may carry a linked paid raw artifact.

        The target must be a ``polza-tts`` ``tts_chunk`` attempt of the named run
        and part whose model uses the async ``elevenlabs/`` ``/media`` route, in
        an accepted state with a bounded, immutable remote task id. A
        ``raw_saved`` repeat is handled idempotently by the caller; any other
        status is refused so a partial or already-finished attempt is never
        relinked.
        """
        if attempt is None or attempt.run_uuid != run_uuid or attempt.part_uuid != part_uuid:
            raise HistoryPaidRawConflictError(
                f"no paid attempt for run {run_uuid!r} part {part_uuid!r}"
            )
        if (
            attempt.call_type != ATTEMPT_CALL_TYPE_TTS_CHUNK
            or attempt.provider != POLZA_TTS_PROVIDER_ID
            or not _polza_media_route_model(attempt.model)
        ):
            raise HistoryPaidRawConflictError(
                f"attempt {attempt.attempt_uuid!r} is not a {POLZA_TTS_PROVIDER_ID!r} "
                f"{ATTEMPT_CALL_TYPE_TTS_CHUNK!r} async media attempt"
            )
        if attempt.status not in (ATTEMPT_STATUS_REMOTE_ACCEPTED, ATTEMPT_STATUS_RAW_SAVED):
            raise HistoryPaidRawConflictError(
                f"attempt {attempt.attempt_uuid!r} is {attempt.status!r}, not "
                f"{ATTEMPT_STATUS_REMOTE_ACCEPTED!r} or {ATTEMPT_STATUS_RAW_SAVED!r}"
            )
        if attempt.remote_id is None or _MEDIA_TASK_ID_PATTERN.fullmatch(attempt.remote_id) is None:
            raise HistoryPaidRawConflictError(
                f"attempt {attempt.attempt_uuid!r} does not carry a bounded accepted remote task id"
            )

    def _verify_media_raw_receipt(
        self,
        run: RunRecord,
        part: PartRecord,
        attempt: AttemptRecord,
        receipt: PaidRawReceipt,
    ) -> PaidRawReceipt:
        """Re-read and verify on-disk paid raw evidence against the database identity.

        The caller's receipt is used only to locate the deterministic raw file;
        its ``raw_path`` is ignored and its own identity is not trusted. The
        stored receipt and raw bytes are re-read and re-hashed against the run
        ``run_root``, the part synthesis fingerprint, and the attempt's bounded
        remote id. This runs outside any transaction, so the write lock is never
        held across file hashing. A failing receipt, digest, size, or path is
        reported as :class:`HistoryPaidRawConflictError` with a fixed message.
        """
        try:
            return verify_paid_raw_receipt(
                run_root=run.run_root,
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.part_uuid,
                synthesis_fingerprint=part.fingerprint or "",
                chunk_id=receipt.chunk_id,
                number=receipt.number,
                audio_format=receipt.audio_format,
                remote_task_id=attempt.remote_id,
            )
        except (PaidRawReceiptError, ValueError) as exc:
            raise HistoryPaidRawConflictError(
                f"paid raw evidence for run {run.run_uuid!r} attempt "
                f"{attempt.attempt_uuid!r} did not verify; refusing to link it"
            ) from exc

    def _raw_saved_artifact(
        self, run_uuid: str, part_uuid: str, attempt_uuid: str, verified: PaidRawReceipt
    ) -> ArtifactRecord | None:
        """Return the one matching paid raw artifact, or ``None`` when absent.

        More than one paid raw artifact for the attempt, or an existing one whose
        run, part, or attempt association, availability, MIME, or ``format``,
        ``chunk_id``, ``number``, and ``generation_id`` metadata does not describe
        exactly the verified bytes and receipt, is a conflict: the canonical
        history must link exactly one available managed raw artifact per attempt to
        the verified identity, so a partial or conflicting row is never adopted or
        duplicated.
        """
        rows = self._connection.execute(
            "SELECT * FROM artifacts WHERE attempt_uuid = ? AND role = ?",
            (attempt_uuid, ARTIFACT_ROLE_PAID_RAW_AUDIO),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise HistoryPaidRawConflictError(
                f"attempt {attempt_uuid!r} carries {len(rows)} paid raw artifacts; refusing "
                "to link another"
            )
        artifact = _row_to_artifact(rows[0])
        expected_metadata = {
            "format": verified.audio_format,
            "chunk_id": verified.chunk_id,
            "number": verified.number,
            "generation_id": verified.generation_id,
        }
        if (
            artifact.run_uuid != run_uuid
            or artifact.part_uuid != part_uuid
            or artifact.attempt_uuid != attempt_uuid
            or artifact.path_kind != PATH_KIND_MANAGED_RELATIVE
            or artifact.path != verified.relative_path
            or artifact.mime != _RAW_AUDIO_MIME_BY_FORMAT[verified.audio_format]
            or artifact.size_bytes != verified.size
            or artifact.sha256 != verified.sha256
            or artifact.media_metadata != expected_metadata
            or artifact.availability != AVAILABILITY_PRESENT
        ):
            raise HistoryPaidRawConflictError(
                f"attempt {attempt_uuid!r} already carries a different paid raw artifact; "
                "refusing to link a second"
            )
        return artifact

    def _find_attempt_uuid_for_part(
        self, run_uuid: str, part_uuid: str, call_type: str
    ) -> str | None:
        row = self._connection.execute(
            "SELECT attempt_uuid FROM attempts WHERE run_uuid = ? AND part_uuid = ? "
            "AND call_type = ? LIMIT 1",
            (run_uuid, part_uuid, call_type),
        ).fetchone()
        return row["attempt_uuid"] if row is not None else None

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
