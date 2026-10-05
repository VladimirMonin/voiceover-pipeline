"""Bounded anonymous emergency receipt for already-accepted paid raw audio.

The filesystem and SQLite cannot form one shared transaction (plan sections 5
and 6), so a crash can leave a saved raw audio file with no history row. This
module writes the already-accepted paid bytes to the deterministic run-local
``raw/<chunk-id>.<ext>`` path together with a small, anonymous receipt that
carries only the attempt and part UUIDs, the synthesis fingerprint, a bounded
remote id, the digest, the format, the deterministic relative path, and the
size. A later caller matches that evidence to an existing SQLite run/attempt
identity; the receipt alone never authorizes a new paid POST.

A synchronous paid TTS submit has no accepted remote id, so it also persists
its exact HTTP response *body* here, before any status check, JSON parse, or
audio decode. That private body plus its bounded receipt is bound to the same
already-reserved attempt, part, chunk, synthesis fingerprint, and request
identity, so a malformed or unsupported payload can be replayed locally with
no second POST instead of being lost.

This module is evidence storage, not a second state engine. Nothing here opens
the database, holds a transaction, calls a provider, polls or downloads a media
task, or runs FFmpeg. Every public function validates all inputs before touching
the tree, keeps the raw bytes and their digest intact, rejects a foreign, symlinked,
or group-/world-accessible ``raw`` directory or evidence file, refuses to
overwrite another paid
attempt instead of reusing it, writes private regular files atomically, and fails
closed on tampered, mismatched, or missing evidence. Because a partial write can
leave raw audio without a receipt (or the reverse), both helpers treat that as a
conflict rather than guessing the missing half.

The bounded id, format, digest, and generated-path rules mirror
``run_state``'s raw-audio contract without importing it, exactly as
``history.repository`` mirrors the media task-id contract: ``run_state`` is the
legacy compatibility exporter, while this package is the canonical storage
layer. Concurrent writers to one run root are serialized by the run lock
(``history.locking``); these helpers assume a single writer and reject a
conflicting paid attempt instead of overwriting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Run-local suffix and artifact identity for one synchronous provider response.
# The body and its receipt live beside the decoded raw audio and are named by the
# attempt UUID, so they never collide with the deterministic ``raw/<chunk-id>.*``
# decoded files and a resume never has to guess which attempt a body belongs to.
RESPONSE_RECEIPT_ARTIFACT_TYPE = "voiceover-paid-sync-response-receipt"
RESPONSE_RECEIPT_VERSION = 1
RESPONSE_FILE_SUFFIX = ".response"
# A provider response body is attacker- or proxy-influenceable, so it is bounded
# before it is stored, and it is bounded again before it is read for replay.
MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
# A body a resume may replay must be a successful response. An error status body
# stays private evidence but never becomes a usable paid response, so a server
# error can never be replayed as audio.
_RESPONSE_SUCCESS_MIN = 200
_RESPONSE_SUCCESS_MAX = 299
_MIN_HTTP_STATUS = 100
_MAX_HTTP_STATUS = 599
# Upper bound for the non-secret request identity strings committed with a body.
_MAX_IDENTITY_CHARS = 256

_RESPONSE_RECEIPT_FIELDS = (
    "artifact_type",
    "receipt_version",
    "attempt_uuid",
    "part_uuid",
    "synthesis_fingerprint",
    "chunk_id",
    "number",
    "provider",
    "model",
    "voice",
    "response_format",
    "http_status",
    "generation_id",
    "path",
    "size",
    "sha256",
)
# One additive bounded evidence field: the raw audio container the synchronous provider
# observed in its response, already mapped to a bounded value (never a raw header). It is
# preserved so a local replay decodes the stored bytes with the container they actually
# arrived in instead of the requested one. It stays optional, so every receipt written
# before it existed still loads, verifies, and replays, and the requested
# ``response_format`` stays the separate request-identity field.
_OPTIONAL_RESPONSE_RECEIPT_FIELDS = ("observed_audio_format",)

# Run-local directory that holds raw paid audio and its receipts.
RAW_DIRECTORY_NAME = "raw"
RECEIPT_ARTIFACT_TYPE = "voiceover-paid-raw-receipt"
RECEIPT_VERSION = 1
# Paid audio and its receipt are private, so a newly created directory or file
# never becomes group- or world-readable.
RAW_DIRECTORY_MODE = 0o700
RAW_FILE_MODE = 0o600
# A tampered receipt must not be able to make the reader allocate an unbounded
# amount of memory before it is rejected.
_MAX_RECEIPT_BYTES = 64 * 1024

# Supported paid raw-audio formats mapped to their deterministic file extension.
# This is the same bounded set as ``run_state._RAW_FILE_EXTENSIONS``.
_RAW_FILE_EXTENSIONS = {"mp3": "mp3", "wav": "wav", "pcm16": "pcm"}

# Largest chunk or dialogue-turn number a generated id may carry. Chunks and
# turns are numbered from 1, so a boolean, non-positive, or larger number was not
# generated by this repository.
_MAX_CHUNK_NUMBER = 1_000_000

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# A remote task or generation id becomes part of a later request path, so only an
# opaque token may be stored or reused. This is the same bounded contract as
# ``run_state._bounded_media_task_id`` and ``history.repository``.
_OPAQUE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")

_RECEIPT_FIELDS = (
    "artifact_type",
    "receipt_version",
    "attempt_uuid",
    "part_uuid",
    "synthesis_fingerprint",
    "chunk_id",
    "number",
    "format",
    "path",
    "size",
    "sha256",
    "remote_task_id",
    "generation_id",
)


class PaidRawReceiptError(RuntimeError):
    """Base class for paid raw-receipt contract violations."""


class PaidRawReceiptConflictError(PaidRawReceiptError):
    """Existing raw audio or receipt belongs to a different paid attempt."""


class PaidRawReceiptVerificationError(PaidRawReceiptError):
    """Stored paid raw evidence is missing, malformed, or does not match."""


class PaidSyncResponseError(PaidRawReceiptError):
    """Base class for a synchronous provider response evidence contract violation."""


class PaidSyncResponseConflictError(PaidSyncResponseError):
    """A synchronous response body could not be stored or conflicts with another attempt."""


class PaidSyncResponseVerificationError(PaidSyncResponseError):
    """Stored synchronous response evidence is missing, malformed, or does not match."""


@dataclass(frozen=True)
class PaidRawReceipt:
    """The bounded evidence of one accepted paid raw audio response.

    ``relative_path`` is the deterministic run-relative location and ``raw_path``
    is the absolute path resolved from the supplied run root. Every other field
    is exactly the bounded value written to (or read back from) the receipt; no
    request text, API key, signed URL, or provider body is carried here.
    """

    attempt_uuid: str
    part_uuid: str
    synthesis_fingerprint: str
    chunk_id: str
    number: int
    audio_format: str
    relative_path: str
    size: int
    sha256: str
    remote_task_id: str | None
    generation_id: str | None
    raw_path: Path


def _bounded_uuid(value: Any) -> str | None:
    """Return the canonical UUID string, or None when the value is not one."""
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _bounded_sha256(value: Any) -> str | None:
    """Keep only a lowercase sha256 hex digest."""
    if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None:
        return value
    return None


def _bounded_opaque(value: Any) -> str | None:
    """Keep only a bounded opaque token, else report unknown."""
    if isinstance(value, str) and _OPAQUE_TOKEN_PATTERN.fullmatch(value) is not None:
        return value
    return None


def _bounded_chunk_number(value: Any) -> int | None:
    """Keep only a positive generated chunk number within the bounded maximum."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 1 <= value <= _MAX_CHUNK_NUMBER else None


def _bounded_chunk_id(value: Any, number: int | None) -> str | None:
    """Keep only an id that equals the generated form for this bounded number."""
    if number is None or not isinstance(value, str):
        return None
    if value == f"chunk_{number:02d}" or value == f"turn_{number:04d}":
        return value
    if value == "chunk_01_omnivoice_session" and number == 1:
        return value
    return None


def _bounded_audio_format(value: Any) -> str | None:
    """Keep only a supported raw-audio format."""
    return value if isinstance(value, str) and value in _RAW_FILE_EXTENSIONS else None


def _bounded_size(value: Any) -> int | None:
    """Keep only a non-negative, non-bool byte count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _require_uuid(value: Any, field_name: str) -> str:
    """Return a canonical UUID, or raise without echoing the value."""
    bounded = _bounded_uuid(value)
    if bounded is None:
        raise ValueError(f"{field_name} must be a UUID string")
    return bounded


def _require_sha256(value: Any, field_name: str) -> str:
    """Return a lowercase sha256 hex digest, or raise without echoing the value."""
    bounded = _bounded_sha256(value)
    if bounded is None:
        raise ValueError(f"{field_name} must be a sha256 hex digest")
    return bounded


def _require_chunk_number(value: Any) -> int:
    """Return a bounded generated chunk number, or raise without echoing it."""
    bounded = _bounded_chunk_number(value)
    if bounded is None:
        raise ValueError("number must be a positive generated chunk number")
    return bounded


def _require_chunk_id(value: Any, number: int) -> str:
    """Return the generated id for this number, or raise without echoing it."""
    bounded = _bounded_chunk_id(value, number)
    if bounded is None:
        raise ValueError("chunk_id must equal the generated form for this number")
    return bounded


def _require_audio_format(value: Any) -> str:
    """Return a supported raw-audio format, or raise without echoing the value."""
    bounded = _bounded_audio_format(value)
    if bounded is None:
        raise ValueError("audio_format must be a supported raw audio format")
    return bounded


def _require_optional_opaque(value: Any, field_name: str) -> str | None:
    """Return a bounded opaque token or None, or raise without echoing the value."""
    if value is None:
        return None
    bounded = _bounded_opaque(value)
    if bounded is None:
        raise ValueError(f"{field_name} is not a bounded opaque token")
    return bounded


def _require_absolute_run_root(run_root: Path | str) -> Path:
    """Return an existing absolute run directory, or raise without echoing it.

    A relative run root would follow the current working directory, so it is
    refused instead of quietly resolved. The message never repeats the supplied
    path, so an unsafe value cannot reach a log or report.
    """
    candidate = Path(run_root).expanduser()
    if not candidate.is_absolute():
        raise ValueError("run_root must be an absolute path")
    if not candidate.is_dir():
        raise ValueError("run_root must be an existing directory")
    return candidate


def _raw_file_name(chunk_id: str, audio_format: str) -> str:
    """Return the deterministic file name for validated inputs."""
    return f"{chunk_id}.{_RAW_FILE_EXTENSIONS[audio_format]}"


def _relative_raw_path(chunk_id: str, audio_format: str) -> str:
    """Return the deterministic run-relative raw path for validated inputs."""
    return f"{RAW_DIRECTORY_NAME}/{_raw_file_name(chunk_id, audio_format)}"


def _receipt_name(raw_file_name: str) -> str:
    """Return the deterministic receipt file name beside its raw audio."""
    return f"{raw_file_name}.receipt.json"


def _path_present(path: Path) -> bool:
    """Whether a path names anything, including a dangling symlink."""
    return path.is_symlink() or path.exists()


def _require_regular_file(path: Path, *, error: type[PaidRawReceiptError], label: str) -> None:
    """Require a private non-symlinked regular file, else raise the caller's error."""
    if path.is_symlink():
        raise error(f"{label} must not be a symlink")
    try:
        status = path.stat()
    except OSError as exc:
        # The message never repeats the underlying OSError, whose text normally
        # embeds the absolute pathname.
        raise error(f"cannot stat {label}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise error(f"{label} is not a regular file")
    if os.name == "posix" and stat.S_IMODE(status.st_mode) & 0o077:
        raise error(f"{label} is group- or world-accessible")


def _sha256_file(path: Path, *, error: type[PaidRawReceiptError], label: str) -> str:
    """Digest a run-local raw file in bounded reads, else raise the caller's error."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise error(f"cannot read {label}") from exc
    return digest.hexdigest()


def _file_size(path: Path, *, error: type[PaidRawReceiptError], label: str) -> int:
    """Return the regular file's byte size, else raise the caller's error."""
    try:
        return path.stat().st_size
    except OSError as exc:
        raise error(f"cannot stat {label}") from exc


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a crash where supported."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to a private regular file atomically.

    A unique temp file is created privately, flushed and fsynced, then renamed
    over the target; the temp is always removed, so a failure leaves no partial
    target and no leftover temp. The caller has already rejected any conflicting
    existing file, and the run lock serializes concurrent writers.
    """
    try:
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        raise PaidRawReceiptConflictError("cannot create a temporary paid raw file") from exc
    temp = Path(name)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), RAW_FILE_MODE)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except OSError as exc:
            raise PaidRawReceiptConflictError("cannot write paid raw evidence") from exc
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    _fsync_directory(path.parent)


def _require_private_raw_directory(raw_dir: Path, *, error: type[PaidRawReceiptError]) -> None:
    """Require an existing real raw directory no other user can write into.

    Paid raw audio and its receipts are private regular files, but a group- or
    world-accessible directory lets another user remove or replace them despite
    their modes. As in ``history.paths.ensure_private_directory``, an insecure
    existing directory is rejected instead of chmodded, because it is user-owned.
    Windows does not expose meaningful POSIX mode bits, so the group/world check
    applies only where those bits describe real access.
    """
    try:
        status = raw_dir.stat()
    except OSError as exc:
        raise error("cannot inspect paid raw directory") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise error("paid raw path is not a directory")
    if os.name == "posix" and stat.S_IMODE(status.st_mode) & 0o077:
        raise error(
            "paid raw directory is group- or world-accessible; refusing to store paid "
            "evidence there instead of changing permissions of a user-owned directory"
        )


def _prepare_raw_directory(run_root: Path) -> Path:
    """Return the private ``raw`` directory, creating it or rejecting a foreign one.

    A missing directory is created private. An existing directory is reused only
    when it is a real directory that is not group- or world-accessible; a symlink,
    any other file type, or an insecure existing directory is rejected instead of
    followed, overwritten, or chmodded.
    """
    raw_dir = run_root / RAW_DIRECTORY_NAME
    if raw_dir.is_symlink():
        raise PaidRawReceiptConflictError("paid raw directory must not be a symlink")
    if raw_dir.exists():
        _require_private_raw_directory(raw_dir, error=PaidRawReceiptConflictError)
        return raw_dir
    try:
        os.mkdir(raw_dir, RAW_DIRECTORY_MODE)
        os.chmod(raw_dir, RAW_DIRECTORY_MODE)
    except OSError as exc:
        raise PaidRawReceiptConflictError("cannot create paid raw directory") from exc
    return raw_dir


def _read_receipt_payload(
    receipt_path: Path, *, error: type[PaidRawReceiptError]
) -> dict[str, Any]:
    """Read a bounded receipt as normalized evidence, or raise ``error``.

    The receipt must be a regular non-symlink file of bounded size holding a JSON
    object with exactly the bounded fields and valid values. Every rejection uses
    a fixed message, so a tampered value (for example a signed URL) is never
    echoed into a log or report.
    """
    _require_regular_file(receipt_path, error=error, label="paid raw receipt")
    try:
        with receipt_path.open("rb") as handle:
            # Read one byte past the bound so an oversized receipt is detected
            # without ever allocating its full contents.
            raw = handle.read(_MAX_RECEIPT_BYTES + 1)
    except OSError as exc:
        raise error("cannot read paid raw receipt") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise error("paid raw receipt is larger than the bounded maximum")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        # RecursionError covers a bounded but pathologically deep nesting that
        # CPython's JSON scanner rejects below the size limit; it must never
        # escape as a raw interpreter error.
        raise error("paid raw receipt is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise error("paid raw receipt is not a JSON object")
    if set(payload) != set(_RECEIPT_FIELDS):
        raise error("paid raw receipt does not carry exactly the bounded fields")
    return _normalize_receipt(payload, error=error)


def _normalize_receipt(
    payload: dict[str, Any], *, error: type[PaidRawReceiptError]
) -> dict[str, Any]:
    """Validate and normalize a parsed receipt payload, or raise ``error``."""
    if payload["artifact_type"] != RECEIPT_ARTIFACT_TYPE:
        raise error("paid raw receipt has an unexpected artifact type")
    version = payload["receipt_version"]
    if isinstance(version, bool) or version != RECEIPT_VERSION:
        raise error("paid raw receipt has an unsupported version")
    attempt = _bounded_uuid(payload["attempt_uuid"])
    part = _bounded_uuid(payload["part_uuid"])
    fingerprint = _bounded_sha256(payload["synthesis_fingerprint"])
    number = _bounded_chunk_number(payload["number"])
    chunk_id = _bounded_chunk_id(payload["chunk_id"], number)
    audio_format = _bounded_audio_format(payload["format"])
    size = _bounded_size(payload["size"])
    sha256 = _bounded_sha256(payload["sha256"])
    stored_remote = payload["remote_task_id"]
    stored_generation = payload["generation_id"]
    remote = None if stored_remote is None else _bounded_opaque(stored_remote)
    generation = None if stored_generation is None else _bounded_opaque(stored_generation)
    if (
        attempt is None
        or part is None
        or fingerprint is None
        or number is None
        or chunk_id is None
        or audio_format is None
        or size is None
        or sha256 is None
        or (stored_remote is not None and remote is None)
        or (stored_generation is not None and generation is None)
    ):
        raise error("paid raw receipt does not carry valid bounded evidence")
    expected_path = _relative_raw_path(chunk_id, audio_format)
    if payload["path"] != expected_path:
        raise error("paid raw receipt path is not the deterministic raw path")
    return {
        "artifact_type": RECEIPT_ARTIFACT_TYPE,
        "receipt_version": RECEIPT_VERSION,
        "attempt_uuid": attempt,
        "part_uuid": part,
        "synthesis_fingerprint": fingerprint,
        "chunk_id": chunk_id,
        "number": number,
        "format": audio_format,
        "path": expected_path,
        "size": size,
        "sha256": sha256,
        "remote_task_id": remote,
        "generation_id": generation,
    }


def _receipt_from_payload(run_root: Path, payload: dict[str, Any]) -> PaidRawReceipt:
    """Build the frozen receipt value object from validated evidence."""
    return PaidRawReceipt(
        attempt_uuid=payload["attempt_uuid"],
        part_uuid=payload["part_uuid"],
        synthesis_fingerprint=payload["synthesis_fingerprint"],
        chunk_id=payload["chunk_id"],
        number=payload["number"],
        audio_format=payload["format"],
        relative_path=payload["path"],
        size=payload["size"],
        sha256=payload["sha256"],
        remote_task_id=payload["remote_task_id"],
        generation_id=payload["generation_id"],
        raw_path=run_root / payload["path"],
    )


def bounded_opaque_token(value: Any) -> str | None:
    """Return a bounded opaque token, or ``None`` when the value is not one.

    A receipt can only carry a bounded opaque remote or generation id. A caller
    holding an *optional*, untrusted provider id (for example a synchronous
    ``X-Generation-Id``) uses this to drop an invalid value to ``None`` before
    :func:`write_paid_raw_receipt`, so accepted paid audio is never lost to a
    strict-id rejection and the invalid value never reaches the receipt.
    """
    return _bounded_opaque(value)


def _new_receipt_payload(
    *,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    audio_format: str,
    relative_path: str,
    size: int,
    sha256: str,
    remote_task_id: str | None,
    generation_id: str | None,
) -> dict[str, Any]:
    """Assemble the exact bounded receipt payload for validated inputs."""
    return {
        "artifact_type": RECEIPT_ARTIFACT_TYPE,
        "receipt_version": RECEIPT_VERSION,
        "attempt_uuid": attempt_uuid,
        "part_uuid": part_uuid,
        "synthesis_fingerprint": synthesis_fingerprint,
        "chunk_id": chunk_id,
        "number": number,
        "format": audio_format,
        "path": relative_path,
        "size": size,
        "sha256": sha256,
        "remote_task_id": remote_task_id,
        "generation_id": generation_id,
    }


def write_paid_raw_receipt(
    *,
    run_root: Path | str,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    audio_format: str,
    audio_bytes: bytes,
    remote_task_id: str | None = None,
    generation_id: str | None = None,
) -> PaidRawReceipt:
    """Persist accepted paid bytes and their anonymous receipt before any commit.

    Every input is validated before the tree is touched. The bytes are written to
    the deterministic ``raw/<chunk-id>.<ext>`` path and the bounded receipt beside
    them, both private and atomic. A repeated call with identical bytes and
    identity is an idempotent no-op; any existing raw audio or receipt that does
    not match this exact attempt is a conflict, so another paid attempt is never
    overwritten or adopted. An existing group- or world-accessible ``raw``
    directory is rejected rather than chmodded. No database row is written here,
    and a present receipt never authorizes a new paid POST on its own.
    """
    try:
        root = _require_absolute_run_root(run_root)
        attempt = _require_uuid(attempt_uuid, "attempt_uuid")
        part = _require_uuid(part_uuid, "part_uuid")
        fingerprint = _require_sha256(synthesis_fingerprint, "synthesis_fingerprint")
        bounded_number = _require_chunk_number(number)
        bounded_id = _require_chunk_id(chunk_id, bounded_number)
        bounded_format = _require_audio_format(audio_format)
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise ValueError("audio_bytes must be bytes")
        data = bytes(audio_bytes)
        if not data:
            raise ValueError("audio_bytes must not be empty")
        remote = _require_optional_opaque(remote_task_id, "remote_task_id")
        generation = _require_optional_opaque(generation_id, "generation_id")

        relative_path = _relative_raw_path(bounded_id, bounded_format)
        raw_file_name = _raw_file_name(bounded_id, bounded_format)
        payload = _new_receipt_payload(
            attempt_uuid=attempt,
            part_uuid=part,
            synthesis_fingerprint=fingerprint,
            chunk_id=bounded_id,
            number=bounded_number,
            audio_format=bounded_format,
            relative_path=relative_path,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            remote_task_id=remote,
            generation_id=generation,
        )

        raw_dir = _prepare_raw_directory(root)
        raw_path = raw_dir / raw_file_name
        receipt_path = raw_dir / _receipt_name(raw_file_name)

        if _path_present(raw_path) or _path_present(receipt_path):
            return _existing_evidence(
                run_root=root,
                raw_path=raw_path,
                receipt_path=receipt_path,
                payload=payload,
                data=data,
            )

        _atomic_write_private(raw_path, data)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        _atomic_write_private(receipt_path, serialized.encode("utf-8") + b"\n")
    except OSError as exc:
        # The underlying OSError text normally carries the absolute run path, so
        # the public message is fixed while the cause stays chained for diagnosis.
        raise PaidRawReceiptConflictError(
            "cannot store paid raw evidence in the run directory"
        ) from exc
    return _receipt_from_payload(root, payload)


def _existing_evidence(
    *,
    run_root: Path,
    raw_path: Path,
    receipt_path: Path,
    payload: dict[str, Any],
    data: bytes,
) -> PaidRawReceipt:
    """Return the matching receipt of an identical repeat, else raise a conflict.

    Both halves must be present, non-symlinked regular files, and must describe
    exactly this attempt with exactly these bytes. Anything else — one half
    missing, a different identity, a different digest, or a replaced file — is a
    conflict, because it could be another paid attempt whose evidence must not be
    overwritten or adopted.
    """
    if not (_path_present(raw_path) and _path_present(receipt_path)):
        raise PaidRawReceiptConflictError(
            "incomplete paid raw evidence: raw audio and receipt must both be present"
        )
    _require_regular_file(raw_path, error=PaidRawReceiptConflictError, label="paid raw audio")
    stored = _read_receipt_payload(receipt_path, error=PaidRawReceiptConflictError)
    if stored != payload:
        raise PaidRawReceiptConflictError(
            "existing paid raw receipt belongs to a different paid attempt"
        )
    if _file_size(raw_path, error=PaidRawReceiptConflictError, label="paid raw audio") != len(data):
        raise PaidRawReceiptConflictError("existing paid raw audio has a different size")
    if (
        _sha256_file(raw_path, error=PaidRawReceiptConflictError, label="paid raw audio")
        != payload["sha256"]
    ):
        raise PaidRawReceiptConflictError("existing paid raw audio has a different digest")
    return _receipt_from_payload(run_root, stored)


def verify_paid_raw_receipt(
    *,
    run_root: Path | str,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    audio_format: str,
    remote_task_id: str | None = None,
    generation_id: str | None = None,
) -> PaidRawReceipt:
    """Verify saved paid raw evidence for crash recovery, or raise.

    The receipt and raw audio must both exist as regular non-symlink files, the
    receipt must carry exactly the bounded fields, and its identity must match
    the requested attempt, part, fingerprint, chunk, and format. A caller-supplied
    remote id must match too; a caller that omits it accepts whatever bounded id
    the receipt already holds. The stored relative path must be the deterministic
    one, and the raw file's size and digest must match the receipt, so recoverable
    bytes are never guessed from a file name alone. An existing group- or
    world-accessible ``raw`` directory is rejected rather than chmodded. Any
    tampered, mismatched, missing, or traversal-shaped evidence raises
    :class:`PaidRawReceiptVerificationError`.
    """
    error = PaidRawReceiptVerificationError
    try:
        root = _require_absolute_run_root(run_root)
        attempt = _require_uuid(attempt_uuid, "attempt_uuid")
        part = _require_uuid(part_uuid, "part_uuid")
        fingerprint = _require_sha256(synthesis_fingerprint, "synthesis_fingerprint")
        bounded_number = _require_chunk_number(number)
        bounded_id = _require_chunk_id(chunk_id, bounded_number)
        bounded_format = _require_audio_format(audio_format)
        remote = _require_optional_opaque(remote_task_id, "remote_task_id")
        generation = _require_optional_opaque(generation_id, "generation_id")

        raw_dir = root / RAW_DIRECTORY_NAME
        if raw_dir.is_symlink() or not raw_dir.is_dir():
            raise error("paid raw directory is missing or not a directory")
        _require_private_raw_directory(raw_dir, error=error)
        raw_file_name = _raw_file_name(bounded_id, bounded_format)
        raw_path = raw_dir / raw_file_name
        receipt_path = raw_dir / _receipt_name(raw_file_name)
        if not _path_present(receipt_path):
            raise error("paid raw receipt is missing")
        if not _path_present(raw_path):
            raise error("paid raw audio is missing")

        stored = _read_receipt_payload(receipt_path, error=error)
        expected_identity = {
            "attempt_uuid": attempt,
            "part_uuid": part,
            "synthesis_fingerprint": fingerprint,
            "chunk_id": bounded_id,
            "number": bounded_number,
            "format": bounded_format,
        }
        for key, value in expected_identity.items():
            if stored[key] != value:
                raise error("paid raw receipt does not match the requested identity")
        if remote is not None and stored["remote_task_id"] != remote:
            raise error("paid raw receipt remote identity does not match")
        if generation is not None and stored["generation_id"] != generation:
            raise error("paid raw receipt remote identity does not match")
        _require_regular_file(raw_path, error=error, label="paid raw audio")
        if _file_size(raw_path, error=error, label="paid raw audio") != stored["size"]:
            raise error("paid raw audio size does not match the receipt")
        if _sha256_file(raw_path, error=error, label="paid raw audio") != stored["sha256"]:
            raise error("paid raw audio digest does not match the receipt")
    except OSError as exc:
        # The underlying OSError text normally carries the absolute run path, so
        # the public message is fixed while the cause stays chained for diagnosis.
        raise error("cannot read paid raw evidence from the run directory") from exc
    return _receipt_from_payload(root, stored)


# -- synchronous provider response evidence ------------------------------------


def _bounded_identity(value: Any) -> str | None:
    """Keep only a bounded, control-free, non-empty identity string."""
    if not isinstance(value, str) or not value or len(value) > _MAX_IDENTITY_CHARS:
        return None
    if any(character in value for character in ("\x00", "\r", "\n")):
        return None
    return value


def _require_identity(value: Any, field_name: str) -> str:
    """Return a bounded identity string, or raise without echoing the value."""
    bounded = _bounded_identity(value)
    if bounded is None:
        raise ValueError(f"{field_name} must be a bounded non-empty string")
    return bounded


def _bounded_http_status(value: Any) -> int | None:
    """Keep only a real HTTP status code, else report unknown."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if _MIN_HTTP_STATUS <= value <= _MAX_HTTP_STATUS else None


def _require_http_status(value: Any) -> int:
    """Return a bounded HTTP status code, or raise without echoing it."""
    bounded = _bounded_http_status(value)
    if bounded is None:
        raise ValueError("http_status must be a valid HTTP status code")
    return bounded


def _response_file_name(attempt_uuid: str) -> str:
    """Return the deterministic response body file name for one attempt."""
    return f"{attempt_uuid}{RESPONSE_FILE_SUFFIX}"


def _response_receipt_name(attempt_uuid: str) -> str:
    """Return the deterministic response receipt file name for one attempt."""
    return f"{_response_file_name(attempt_uuid)}.receipt.json"


def _response_relative_path(attempt_uuid: str) -> str:
    """Return the deterministic run-relative response body path for one attempt."""
    return f"{RAW_DIRECTORY_NAME}/{_response_file_name(attempt_uuid)}"


@dataclass(frozen=True)
class SyncResponseReceipt:
    """The bounded evidence of one synchronous paid provider response body.

    ``generation_id`` is the sanitized opaque ``X-Generation-Id`` header value (or
    ``None`` when the provider reported none or a value that is not a bounded
    token). ``http_status`` is the observed status code. ``observed_audio_format``
    is the bounded raw container the provider observed for this body, or ``None``
    when the provider reported none or the stored body describes its own container
    (a JSON body carrying ``contentType``); it never carries a raw header value and
    is not the request identity, which stays ``response_format``. ``relative_path``
    and ``body_path`` point at the private run-local body. No request text, API key,
    signed URL, or provider error body is carried in the receipt itself.
    """

    attempt_uuid: str
    part_uuid: str
    synthesis_fingerprint: str
    chunk_id: str
    number: int
    provider: str
    model: str
    voice: str
    response_format: str
    http_status: int
    generation_id: str | None
    relative_path: str
    size: int
    sha256: str
    body_path: Path
    observed_audio_format: str | None = None


def _new_response_receipt_payload(
    *,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    provider: str,
    model: str,
    voice: str,
    response_format: str,
    http_status: int,
    generation_id: str | None,
    relative_path: str,
    size: int,
    sha256: str,
    observed_audio_format: str | None = None,
) -> dict[str, Any]:
    """Assemble the exact bounded response-receipt payload for validated inputs.

    The optional observed container is written only when one was observed, so a
    provider that reports none keeps the exact pre-existing payload bytes and a
    repeated write of the same evidence stays an idempotent no-op.
    """
    payload: dict[str, Any] = {
        "artifact_type": RESPONSE_RECEIPT_ARTIFACT_TYPE,
        "receipt_version": RESPONSE_RECEIPT_VERSION,
        "attempt_uuid": attempt_uuid,
        "part_uuid": part_uuid,
        "synthesis_fingerprint": synthesis_fingerprint,
        "chunk_id": chunk_id,
        "number": number,
        "provider": provider,
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "http_status": http_status,
        "generation_id": generation_id,
        "path": relative_path,
        "size": size,
        "sha256": sha256,
    }
    if observed_audio_format is not None:
        payload["observed_audio_format"] = observed_audio_format
    return payload


def _normalize_response_receipt(
    payload: dict[str, Any], *, error: type[PaidRawReceiptError]
) -> dict[str, Any]:
    """Validate and normalize a parsed response receipt, or raise ``error``."""
    if payload["artifact_type"] != RESPONSE_RECEIPT_ARTIFACT_TYPE:
        raise error("paid sync response receipt has an unexpected artifact type")
    version = payload["receipt_version"]
    if isinstance(version, bool) or version != RESPONSE_RECEIPT_VERSION:
        raise error("paid sync response receipt has an unsupported version")
    attempt = _bounded_uuid(payload["attempt_uuid"])
    part = _bounded_uuid(payload["part_uuid"])
    fingerprint = _bounded_sha256(payload["synthesis_fingerprint"])
    number = _bounded_chunk_number(payload["number"])
    chunk_id = _bounded_chunk_id(payload["chunk_id"], number)
    provider = _bounded_identity(payload["provider"])
    model = _bounded_identity(payload["model"])
    voice = _bounded_identity(payload["voice"])
    response_format = _bounded_identity(payload["response_format"])
    http_status = _bounded_http_status(payload["http_status"])
    stored_generation = payload["generation_id"]
    generation = None if stored_generation is None else _bounded_opaque(stored_generation)
    stored_observed = payload.get("observed_audio_format")
    observed = None if stored_observed is None else _bounded_identity(stored_observed)
    size = _bounded_size(payload["size"])
    sha256 = _bounded_sha256(payload["sha256"])
    if (
        attempt is None
        or part is None
        or fingerprint is None
        or number is None
        or chunk_id is None
        or provider is None
        or model is None
        or voice is None
        or response_format is None
        or http_status is None
        or size is None
        or size == 0
        or size > MAX_RESPONSE_BODY_BYTES
        or sha256 is None
        or (stored_generation is not None and generation is None)
        or (stored_observed is not None and observed is None)
    ):
        raise error("paid sync response receipt does not carry valid bounded evidence")
    expected_path = _response_relative_path(attempt)
    if payload["path"] != expected_path:
        raise error("paid sync response receipt path is not the deterministic response path")
    normalized: dict[str, Any] = {
        "artifact_type": RESPONSE_RECEIPT_ARTIFACT_TYPE,
        "receipt_version": RESPONSE_RECEIPT_VERSION,
        "attempt_uuid": attempt,
        "part_uuid": part,
        "synthesis_fingerprint": fingerprint,
        "chunk_id": chunk_id,
        "number": number,
        "provider": provider,
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "http_status": http_status,
        "generation_id": generation,
        "path": expected_path,
        "size": size,
        "sha256": sha256,
    }
    if observed is not None:
        normalized["observed_audio_format"] = observed
    return normalized


def _read_response_receipt_payload(
    receipt_path: Path, *, error: type[PaidRawReceiptError]
) -> dict[str, Any]:
    """Read a bounded response receipt as normalized evidence, or raise ``error``."""
    _require_regular_file(receipt_path, error=error, label="paid sync response receipt")
    try:
        with receipt_path.open("rb") as handle:
            raw = handle.read(_MAX_RECEIPT_BYTES + 1)
    except OSError as exc:
        raise error("cannot read paid sync response receipt") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise error("paid sync response receipt is larger than the bounded maximum")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise error("paid sync response receipt is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise error("paid sync response receipt is not a JSON object")
    present = set(payload)
    allowed = set(_RESPONSE_RECEIPT_FIELDS) | set(_OPTIONAL_RESPONSE_RECEIPT_FIELDS)
    if not set(_RESPONSE_RECEIPT_FIELDS) <= present or not present <= allowed:
        raise error("paid sync response receipt does not carry exactly the bounded fields")
    return _normalize_response_receipt(payload, error=error)


def _response_receipt_from_payload(run_root: Path, payload: dict[str, Any]) -> SyncResponseReceipt:
    """Build the frozen response receipt value object from validated evidence."""
    return SyncResponseReceipt(
        attempt_uuid=payload["attempt_uuid"],
        part_uuid=payload["part_uuid"],
        synthesis_fingerprint=payload["synthesis_fingerprint"],
        chunk_id=payload["chunk_id"],
        number=payload["number"],
        provider=payload["provider"],
        model=payload["model"],
        voice=payload["voice"],
        response_format=payload["response_format"],
        http_status=payload["http_status"],
        generation_id=payload["generation_id"],
        relative_path=payload["path"],
        size=payload["size"],
        sha256=payload["sha256"],
        body_path=run_root / payload["path"],
        observed_audio_format=payload.get("observed_audio_format"),
    )


def write_sync_response_receipt(
    *,
    run_root: Path | str,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    provider: str,
    model: str,
    voice: str,
    response_format: str,
    http_status: int,
    body: bytes,
    generation_id: str | None = None,
    observed_audio_format: str | None = None,
) -> SyncResponseReceipt:
    """Persist a synchronous provider response body and its bounded receipt.

    The exact HTTP body is written to the deterministic ``raw/<attempt>.response``
    path together with a bounded receipt, both private and atomic, *before* any
    status check, JSON parse, or audio decode, so an accepted paid response is
    never lost to a malformed or unsupported payload. ``generation_id`` is the
    untrusted ``X-Generation-Id`` header value and is sanitized to a bounded opaque
    token (or ``None``) before it is stored, so an invalid header never fails a
    valid body. ``observed_audio_format`` is the already-mapped bounded container the
    provider observed for this body (or ``None`` when it reported none), recorded
    alongside the requested ``response_format`` so a local replay decodes the stored
    bytes with the container they arrived in; the requested format stays the request
    identity and is never replaced by the observed one. An oversized, empty, or
    unstorable body fails closed. A repeated
    call with identical bytes and identity is an idempotent no-op; any existing
    evidence that does not match this exact attempt is a conflict, so another paid
    attempt is never overwritten or adopted. No database row is written here, and
    a present body never authorizes a new paid POST on its own.
    """
    try:
        root = _require_absolute_run_root(run_root)
        attempt = _require_uuid(attempt_uuid, "attempt_uuid")
        part = _require_uuid(part_uuid, "part_uuid")
        fingerprint = _require_sha256(synthesis_fingerprint, "synthesis_fingerprint")
        bounded_number = _require_chunk_number(number)
        bounded_id = _require_chunk_id(chunk_id, bounded_number)
        bounded_provider = _require_identity(provider, "provider")
        bounded_model = _require_identity(model, "model")
        bounded_voice = _require_identity(voice, "voice")
        bounded_format = _require_identity(response_format, "response_format")
        bounded_status = _require_http_status(http_status)
        if not isinstance(body, (bytes, bytearray)):
            raise ValueError("body must be bytes")
        data = bytes(body)
        if not data:
            raise ValueError("body must not be empty")
        if len(data) > MAX_RESPONSE_BODY_BYTES:
            raise PaidSyncResponseConflictError(
                "the paid sync response body exceeds the bounded maximum"
            )
        bounded_generation = _bounded_opaque(generation_id)
        bounded_observed = (
            None if observed_audio_format is None else _bounded_identity(observed_audio_format)
        )
        if observed_audio_format is not None and bounded_observed is None:
            raise ValueError("observed_audio_format must be a bounded non-empty container")

        relative_path = _response_relative_path(attempt)
        payload = _new_response_receipt_payload(
            attempt_uuid=attempt,
            part_uuid=part,
            synthesis_fingerprint=fingerprint,
            chunk_id=bounded_id,
            number=bounded_number,
            provider=bounded_provider,
            model=bounded_model,
            voice=bounded_voice,
            response_format=bounded_format,
            http_status=bounded_status,
            generation_id=bounded_generation,
            relative_path=relative_path,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            observed_audio_format=bounded_observed,
        )

        raw_dir = _prepare_raw_directory(root)
        body_path = raw_dir / _response_file_name(attempt)
        receipt_path = raw_dir / _response_receipt_name(attempt)

        if _path_present(body_path) or _path_present(receipt_path):
            return _existing_response_evidence(
                run_root=root,
                body_path=body_path,
                receipt_path=receipt_path,
                payload=payload,
                data=data,
            )

        _atomic_write_private(body_path, data)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        _atomic_write_private(receipt_path, serialized.encode("utf-8") + b"\n")
    except OSError as exc:
        # The underlying OSError text normally carries the absolute run path, so
        # the public message is fixed while the cause stays chained for diagnosis.
        raise PaidSyncResponseConflictError(
            "cannot store paid sync response evidence in the run directory"
        ) from exc
    return _response_receipt_from_payload(root, payload)


def _existing_response_evidence(
    *,
    run_root: Path,
    body_path: Path,
    receipt_path: Path,
    payload: dict[str, Any],
    data: bytes,
) -> SyncResponseReceipt:
    """Return the matching receipt of an identical repeat, else raise a conflict.

    Both halves must be present, non-symlinked regular files, and must describe
    exactly this attempt with exactly these bytes. Anything else is a conflict,
    because it could be another paid attempt whose evidence must not be
    overwritten or adopted.
    """
    if not (_path_present(body_path) and _path_present(receipt_path)):
        raise PaidSyncResponseConflictError(
            "incomplete paid sync response evidence: body and receipt must both be present"
        )
    _require_regular_file(
        body_path, error=PaidSyncResponseConflictError, label="paid sync response body"
    )
    stored = _read_response_receipt_payload(receipt_path, error=PaidSyncResponseConflictError)
    if stored != payload:
        raise PaidSyncResponseConflictError(
            "existing paid sync response receipt belongs to a different paid attempt"
        )
    if _file_size(
        body_path, error=PaidSyncResponseConflictError, label="paid sync response body"
    ) != len(data):
        raise PaidSyncResponseConflictError("existing paid sync response body has a different size")
    if (
        _sha256_file(
            body_path, error=PaidSyncResponseConflictError, label="paid sync response body"
        )
        != payload["sha256"]
    ):
        raise PaidSyncResponseConflictError(
            "existing paid sync response body has a different digest"
        )
    return _response_receipt_from_payload(run_root, stored)


def sync_response_evidence_present(*, run_root: Path | str, attempt_uuid: str) -> bool:
    """Whether either private response half exists, including a dangling symlink.

    A pre-feature synchronous attempt may only have a decoded raw receipt. When
    either new response half exists, however, replay must verify it instead of
    taking the older raw-only path and discarding an observed paid cost.
    """
    root = _require_absolute_run_root(run_root)
    attempt = _require_uuid(attempt_uuid, "attempt_uuid")
    raw_dir = root / RAW_DIRECTORY_NAME
    return _path_present(raw_dir / _response_file_name(attempt)) or _path_present(
        raw_dir / _response_receipt_name(attempt)
    )


def verify_sync_response_receipt(
    *,
    run_root: Path | str,
    attempt_uuid: str,
    part_uuid: str,
    synthesis_fingerprint: str,
    chunk_id: str,
    number: int,
    provider: str,
    model: str,
    voice: str,
    response_format: str,
) -> SyncResponseReceipt:
    """Verify stored synchronous response evidence for local replay, or raise.

    The receipt and body must both exist as regular non-symlink files inside a
    private ``raw`` directory, the receipt must carry exactly the bounded evidence
    fields (plus the optional observed audio container), and its attempt, part,
    fingerprint, chunk, number, and request identity must
    match the requested run. The observed status must be a successful response and
    the body's size and digest must match the receipt, so a replayed body is never
    guessed from a file name. Any tampered, mismatched, missing, oversized, or
    traversal-shaped evidence raises :class:`PaidSyncResponseVerificationError`.
    """
    error = PaidSyncResponseVerificationError
    try:
        root = _require_absolute_run_root(run_root)
        attempt = _require_uuid(attempt_uuid, "attempt_uuid")
        part = _require_uuid(part_uuid, "part_uuid")
        fingerprint = _require_sha256(synthesis_fingerprint, "synthesis_fingerprint")
        bounded_number = _require_chunk_number(number)
        bounded_id = _require_chunk_id(chunk_id, bounded_number)
        expected_provider = _require_identity(provider, "provider")
        expected_model = _require_identity(model, "model")
        expected_voice = _require_identity(voice, "voice")
        expected_format = _require_identity(response_format, "response_format")

        raw_dir = root / RAW_DIRECTORY_NAME
        if raw_dir.is_symlink() or not raw_dir.is_dir():
            raise error("paid raw directory is missing or not a directory")
        _require_private_raw_directory(raw_dir, error=error)
        body_path = raw_dir / _response_file_name(attempt)
        receipt_path = raw_dir / _response_receipt_name(attempt)
        if not _path_present(receipt_path):
            raise error("paid sync response receipt is missing")
        if not _path_present(body_path):
            raise error("paid sync response body is missing")

        stored = _read_response_receipt_payload(receipt_path, error=error)
        expected_identity = {
            "attempt_uuid": attempt,
            "part_uuid": part,
            "synthesis_fingerprint": fingerprint,
            "chunk_id": bounded_id,
            "number": bounded_number,
            "provider": expected_provider,
            "model": expected_model,
            "voice": expected_voice,
            "response_format": expected_format,
        }
        for key, value in expected_identity.items():
            if stored[key] != value:
                raise error("paid sync response receipt does not match the requested identity")
        if not _RESPONSE_SUCCESS_MIN <= stored["http_status"] <= _RESPONSE_SUCCESS_MAX:
            raise error("the stored paid sync response is not a successful response")
        _require_regular_file(body_path, error=error, label="paid sync response body")
        if _file_size(body_path, error=error, label="paid sync response body") != stored["size"]:
            raise error("paid sync response body size does not match the receipt")
        if (
            _sha256_file(body_path, error=error, label="paid sync response body")
            != stored["sha256"]
        ):
            raise error("paid sync response body digest does not match the receipt")
    except OSError as exc:
        # The underlying OSError text normally carries the absolute run path, so
        # the public message is fixed while the cause stays chained for diagnosis.
        raise error("cannot read paid sync response evidence from the run directory") from exc
    return _response_receipt_from_payload(root, stored)


def read_sync_response_body(receipt: SyncResponseReceipt) -> bytes:
    """Return the validated private response body of one verified receipt.

    The body is read bounded and re-checked against the receipt's size and digest,
    so a body replaced between verification and read fails closed instead of being
    parsed. At most the bound plus one byte is read, so an oversized body is
    refused before its full contents are allocated.
    """
    error = PaidSyncResponseVerificationError
    path = receipt.body_path
    _require_regular_file(path, error=error, label="paid sync response body")
    if _file_size(path, error=error, label="paid sync response body") != receipt.size:
        raise error("paid sync response body size does not match the receipt")
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_RESPONSE_BODY_BYTES + 1)
    except OSError as exc:
        raise error("cannot read the paid sync response body") from exc
    if len(data) > MAX_RESPONSE_BODY_BYTES:
        raise error("the paid sync response body exceeds the bounded maximum")
    if hashlib.sha256(data).hexdigest() != receipt.sha256:
        raise error("paid sync response body digest does not match the receipt")
    return data
