"""Atomic prepared-TTS snapshot writer for the canonical history database.

Plan sections 5 and 6 make SQLite the single source of truth for run history.
This module is the native counterpart of
:mod:`voiceover_pipeline.history.legacy_import`: it persists one already-prepared
TTS run -- the exact script text, the ordered parts with their effective voice,
the style direction, and the structural script metadata a later resume needs --
in one database transaction, before any provider request exists.

Contract:

* The caller supplies every value. This service reads no ``run_state.json``, no
  script file, no provider, and no network, and writes no JSON. The snapshot
  identity therefore stays stable when the script file changes afterwards,
  because only the caller-provided text is hashed.
* The ordered parts are validated before any insert: each part's chunk number
  must be exactly its 1-based position in ``prepared.parts`` and its id exactly
  the generated form for that number (``chunk_01``, ``turn_0003``, or the single
  ``chunk_01_omnivoice_session``), each part voice must equal its wrapped chunk
  voice, and every part must resolve to a non-empty effective voice (that chunk
  voice, else the run voice). The run provider, model, voice, prompt mode, and
  script format must be non-empty. This position/number/id agreement is what lets
  the later raw-saved seam's ``receipt.number == part.position`` link ever match.
* Fingerprints are canonical UTF-8 SHA-256 over a versioned payload. The run
  identity covers provider, model, voice, style prompt, prompt mode, the opaque
  voice and synthesis identities, the script format, and the full script hash;
  each part identity adds its position and every ``ScriptChunk`` field. Any
  change to voice, text, cast, order, pause, or runtime binding therefore
  produces a new identity.
* A ``run_root`` already owned by any run -- native or imported -- blocks a
  second snapshot before anything is inserted.
* Native parts keep ``stage`` NULL until a real state transition: the existing
  raw-saved seam moves ``NULL -> raw_saved`` later. No ``prepared`` stage is
  invented here, and the per-part vibe columns stay NULL until the speech-parts
  stage defines a value.
* The snapshot owns its transaction: a caller already inside
  ``repository.transaction()`` is refused before any insert, because
  :meth:`HistoryRepository.transaction` would otherwise join that caller's block
  and a later caller rollback could erase a snapshot this function already
  returned as durable.
* The whitelisted structural identity is stored only when the history redaction
  boundary would leave every value unchanged; a secret-looking identity value is
  rejected with a fixed privacy-safe message instead of committing an
  unreconstructable snapshot. Full script and style-direction text is private
  verbatim data and is never suppressed.
* A ``style_prompt`` is stored as a private ``tts_direction`` source only when it
  is a non-empty string. ``None`` means the run has no direction; a supplied empty
  string is rejected before any insert, because the run identity still hashes it
  while no source would record it, and the read view could never rebuild such a run
  from committed rows.
* Run, parts, and text sources commit in one transaction, so a failed insert
  rolls every entity back. No transaction is ever held across a filesystem or
  network call.

Known limits at this foundation stage:

* Only ``polza-tts`` and ``openrouter-tts`` prepared runs are snapshotted; every
  other provider raises :class:`NativeSnapshotValidationError` before any insert.
  Each deferred route omits nonsecret identity a later ``history resume UUID``
  would need, so persisting it now would store an unreconstructable snapshot:
  ``qwen-local`` carries no mode, runtime, clone sample path/hash, or sample
  text; ``omnivoice-local`` carries no ``--voice-bank`` catalog locator, preset
  reference fingerprint, clone reference hashes, or design instruction; and
  ``polza-chat-audio`` carries no ``fallback_voice``. This is a temporary S05
  integration gap, not a disabled live path: the CLI still performs its own run
  and JSON state writes, and this service is not yet wired into execution. A
  route is enabled only once its missing nonsecret inputs (mode, runtime,
  reference locator/hash, sample text, and fallback voice) are captured on the
  prepared run before the CLI cutover.
* The run-level voice must be non-empty, so a deferred route whose identity lives
  only in ``voice_identity`` or a per-part cast voice still needs an effective
  run voice supplied by the execution wiring before the route is enabled.
* The ``run_root`` collision check compares canonical resolved strings, so an
  unusual aliasing mount or a case-insensitive volume can still name one
  directory through two strings; and no inter-process lock is taken here,
  because the single-writer execution wiring that owns the directory does not
  exist yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ScriptChunk
from ..services.prepare import OmniVoiceVoiceBankIdentity, PreparedPart, PreparedRun
from .repository import (
    TEXT_COMPLETENESS_COMPLETE,
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    HistoryRepository,
    PartRecord,
    RunRecord,
    _redact_snapshot,
)

# Operation shared with the legacy importer, so history queries see one TTS kind.
NATIVE_SNAPSHOT_OPERATION = "tts"
# Provenance written on every text source and recorded in the run snapshot. It
# separates a native snapshot from a ``legacy_import`` one.
NATIVE_SNAPSHOT_ORIGIN = "native_snapshot"
# Version of the fingerprint payload. A material change to what participates in
# the identity bumps it instead of silently changing old hashes.
NATIVE_SNAPSHOT_FINGERPRINT_VERSION = 1
# Fixed rejection message used when the history redaction boundary would change a
# whitelisted structural identity value. An unreconstructable identity is worse
# than no snapshot, and the message never echoes the offending value.
_REDACTED_IDENTITY_REJECTED = (
    "config snapshot contains a value the history boundary would redact; refusing to "
    "persist an unreconstructable identity"
)
# Routes whose prepared run already carries every nonsecret identity a later
# resume needs, so only they may be snapshotted at this foundation stage:
#   * ``polza-tts`` -- model and voice.
#   * ``openrouter-tts`` -- model, voice, style prompt, and prompt mode.
#   * ``omnivoice-local`` -- the admitted preset dialogue route, whose
#     ``PreparedRun.voice_bank_identity`` carries the catalog locator and each
#     referenced profile's reference locator, digest, text, and language.
# Every other provider fails closed before any insert, whether it is a known route
# whose resume inputs ``PreparedRun`` does not yet carry (``qwen-local``,
# ``omnivoice-local`` non-dialogue modes, or ``polza-chat-audio``) or an unknown
# identifier. The rejection is one fixed message because an unknown identifier is
# not admitted input and could be a secret, so the message echoes no provider
# value.
_NATIVE_SNAPSHOT_SUPPORTED_TTS_PROVIDERS = frozenset(
    {"polza-tts", "openrouter-tts", "omnivoice-local"}
)
_UNSUPPORTED_TTS_ROUTE_REJECTED = (
    "prepared run provider is not yet supported by the native snapshot writer: its "
    "resume identity inputs are not persisted, so the snapshot would be "
    "unreconstructable"
)


class NativeSnapshotError(RuntimeError):
    """Base class for native prepared-TTS snapshot contract violations."""


class NativeSnapshotValidationError(NativeSnapshotError, ValueError):
    """The prepared run or one of its identities is not a valid snapshot input."""


class NativeSnapshotRunRootConflictError(NativeSnapshotError):
    """The normalized run root already belongs to another history run."""


class NativeSnapshotInTransactionError(NativeSnapshotError):
    """The snapshot was requested inside a caller's already open transaction.

    :meth:`HistoryRepository.transaction` joins an already open transaction, so a
    snapshot written from inside a caller's block would still be part of a
    transaction that caller can roll back after this function returned a
    durable-looking result. The snapshot therefore owns its transaction and
    refuses to join one.
    """


@dataclass(frozen=True)
class PreparedTtsSnapshot:
    """The committed run, its ordered parts, and the run-level fingerprint."""

    run: RunRecord
    parts: tuple[PartRecord, ...]
    fingerprint: str


@dataclass(frozen=True)
class _PlannedPart:
    """One validated part: its stored values, identity, and snapshot entry."""

    position: int
    text: str
    content_hash: str
    effective_voice: str
    fingerprint: str
    config_entry: dict[str, Any]


def _sha256_text(text: str) -> str:
    """Return the lowercase hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    """Return the canonical UTF-8 SHA-256 of a JSON payload."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeSnapshotValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeSnapshotValidationError(f"{field_name} must be a positive integer")
    return value


def _require_standalone_transaction(repository: HistoryRepository) -> None:
    """Fail closed when the caller already holds an open transaction.

    ``HistoryRepository.transaction`` joins an already open transaction, so a
    snapshot written from inside a caller's block would be rolled back with it
    after this function returned a snapshot the caller believed was durable. The
    check reads the same ``in_transaction`` flag the repository's own guards use
    and runs before any insert.
    """
    if repository._connection.in_transaction:
        raise NativeSnapshotInTransactionError(
            "persist_prepared_tts_snapshot must run outside any open transaction; a "
            "caller rollback could erase the returned snapshot"
        )


def _is_generated_chunk_id(chunk_id: str, number: int) -> bool:
    """Whether ``chunk_id`` is the exact generated id for ``number``.

    A later ``run_state`` marker and its paid raw receipt may only echo back the
    generated form for the number: an ordinary chunk ``chunk_01``, a dialogue
    turn ``turn_0003``, or the single OmniVoice session chunk
    ``chunk_01_omnivoice_session`` (always number 1). This mirrors
    ``run_state._bounded_chunk_id`` and ``raw_receipt._bounded_chunk_id`` so the
    raw-saved seam's ``receipt.number == part.position`` link can always match.
    """
    if chunk_id == f"chunk_{number:02d}" or chunk_id == f"turn_{number:04d}":
        return True
    return chunk_id == "chunk_01_omnivoice_session" and number == 1


def _normalize_run_root(run_root: Path) -> str:
    """Return the canonical absolute run root stored in and matched against the DB.

    The resolved form is what the legacy importer already stores for its run
    roots, so a native snapshot correctly collides with an imported run of the
    same directory. Case-insensitive volumes and aliasing mounts can still make
    two different strings name one directory; that residual is recorded rather
    than papered over, and no lock is claimed here.
    """
    return str(Path(run_root).expanduser().resolve())


def _resolve_effective_voice(part: PreparedPart, run_voice: str) -> str:
    """Return the voice a part will be synthesized with, or fail validation.

    A part's own cast voice wins; ``None`` falls back to the run voice. An empty
    cast voice is invalid rather than a silent fallback.
    """
    voice = run_voice if part.voice is None else part.voice
    if not isinstance(voice, str) or not voice.strip():
        raise NativeSnapshotValidationError(f"part {part.chunk.id!r} has no effective voice")
    return voice


def _run_identity_payload(
    *,
    provider: str,
    model: str,
    voice: str,
    style_prompt: str | None,
    prompt_mode: str,
    voice_identity: str | None,
    synthesis_identity: str | None,
    voice_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": NATIVE_SNAPSHOT_FINGERPRINT_VERSION,
        "provider": provider,
        "model": model,
        "voice": voice,
        "style_prompt": style_prompt,
        "prompt_mode": prompt_mode,
        "voice_identity": voice_identity,
        "synthesis_identity": synthesis_identity,
    }
    if voice_bank is not None:
        # Only the ``omnivoice-local`` preset dialogue route records a voice-bank
        # identity. The key is added only when present, so a polza/openrouter run
        # keeps the exact version-1 payload it committed before this route existed
        # and needs no fingerprint-version bump.
        payload["voice_bank"] = voice_bank
    return payload


def _chunk_payload(chunk: ScriptChunk) -> dict[str, Any]:
    return {
        "number": chunk.number,
        "id": chunk.id,
        "text": chunk.text,
        "speaker": chunk.speaker,
        "voice": chunk.voice,
        "voice_fingerprint": chunk.voice_fingerprint,
        "pause_after_ms": chunk.pause_after_ms,
    }


def _part_fingerprint(
    identity: dict[str, Any],
    *,
    chunk: ScriptChunk,
    position: int,
    effective_voice: str,
) -> str:
    payload: dict[str, Any] = dict(identity)
    payload["position"] = position
    payload["chunk"] = _chunk_payload(chunk)
    payload["effective_voice"] = effective_voice
    return _fingerprint(payload)


def _snapshot_fingerprint(
    identity: dict[str, Any],
    *,
    script_format: str,
    script_sha256: str,
    part_fingerprints: list[str],
) -> str:
    payload: dict[str, Any] = dict(identity)
    payload["script_format"] = script_format
    payload["script_sha256"] = script_sha256
    payload["part_fingerprints"] = part_fingerprints
    return _fingerprint(payload)


def _build_snapshot_config(
    *,
    identity: dict[str, Any],
    snapshot_fingerprint: str,
    script_format: str,
    script_sha256: str,
    script_path: str | None,
    part_count: int,
    part_entries: list[dict[str, Any]],
    output_options: dict[str, Any] | None,
    voice_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the whitelisted structural snapshot stored in ``config_snapshot``.

    Only named fields are copied; no ``Namespace``, request, or arbitrary mapping
    is spread in. Full text is deliberately absent -- it lives in ``text_sources``
    -- so the snapshot stays a resume map rather than a second text copy.
    """
    snapshot: dict[str, Any] = {
        "snapshot_version": NATIVE_SNAPSHOT_FINGERPRINT_VERSION,
        "snapshot_fingerprint": snapshot_fingerprint,
        "operation_origin": NATIVE_SNAPSHOT_ORIGIN,
        "script_format": script_format,
        "script_sha256": script_sha256,
        "script_path": script_path,
        "provider": identity["provider"],
        "model": identity["model"],
        "voice": identity["voice"],
        "prompt_mode": identity["prompt_mode"],
        "voice_identity": identity["voice_identity"],
        "synthesis_identity": identity["synthesis_identity"],
        "part_count": part_count,
        "parts": part_entries,
    }
    if voice_bank is not None:
        snapshot["voice_bank"] = voice_bank
    if output_options is not None:
        snapshot["output"] = output_options
    return snapshot


def _require_output_options(value: object) -> dict[str, Any] | None:
    """Validate the bounded output-processing settings a resume must preserve.

    Only a flat mapping of string keys to plain string, integer, or boolean
    values is accepted: it is stored verbatim in the run snapshot so a later
    resume can prove the trimming and assembly semantics have not changed, and a
    nested or non-plain value would be an unreconstructable identity. ``None``
    means the caller records no output settings.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeSnapshotValidationError("output_options must be a mapping or None")
    validated: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise NativeSnapshotValidationError("output option keys must be non-empty strings")
        if not isinstance(item, (str, int, bool)) or isinstance(item, float):
            raise NativeSnapshotValidationError(
                "output option values must be strings, integers, or booleans"
            )
        validated[key] = item
    return validated


def _require_voice_bank_identity(prepared: PreparedRun, provider: str) -> dict[str, Any] | None:
    """Return the committed voice-bank identity payload, or ``None`` when absent.

    Only the admitted ``omnivoice-local`` preset dialogue route carries one, and
    it must always carry one: a non-dialogue ``omnivoice-local`` run (or any other
    provider) that somehow reached this writer would otherwise commit an identity
    whose voice-bank inputs were never captured. The rejection echoes no value.
    """
    identity = prepared.voice_bank_identity
    if provider == "omnivoice-local":
        if not isinstance(identity, OmniVoiceVoiceBankIdentity):
            # A local run without a committed voice-bank identity (a non-dialogue
            # preset/clone/design mode) did not capture its resume inputs, so the
            # same fixed route-rejection message applies and names no value.
            raise NativeSnapshotValidationError(_UNSUPPORTED_TTS_ROUTE_REJECTED)
        return identity.to_payload()
    if identity is not None:
        raise NativeSnapshotValidationError(
            "a voice-bank identity is only valid for the local OmniVoice dialogue route"
        )
    return None


def _require_dialogue_bank_consistency(
    prepared: PreparedRun, identity_payload: dict[str, Any] | None
) -> None:
    """Require every local dialogue turn to match its committed voice-bank profile.

    Each turn's cast profile id must be one of the identity's referenced profiles
    and its bound reference digest must equal that profile's stored
    ``reference_sha256``, so a committed snapshot can never bind a turn to a
    reference the run did not actually clone. The rejection echoes no value.
    """
    if identity_payload is None:
        return
    by_id = {
        item["profile_id"]: item["reference_sha256"]
        for item in identity_payload["profiles"]
        if isinstance(item, dict)
        and isinstance(item.get("profile_id"), str)
        and isinstance(item.get("reference_sha256"), str)
    }
    for part in prepared.parts:
        chunk = part.chunk
        if chunk.voice is None or by_id.get(chunk.voice) != chunk.voice_fingerprint:
            raise NativeSnapshotValidationError(
                "a local OmniVoice dialogue turn does not match its voice-bank identity"
            )


def persist_prepared_tts_snapshot(
    repository: HistoryRepository,
    *,
    prepared: PreparedRun,
    run_root: Path,
    user_label: str | None,
    script_format: str,
    script_text: str,
    script_path: Path | None,
    voice_identity: str | None = None,
    synthesis_identity: str | None = None,
    output_options: dict[str, Any] | None = None,
) -> PreparedTtsSnapshot:
    """Persist one prepared TTS run, its parts, and its text sources atomically.

    The caller owns the ordering contract: ``prepared.parts`` is the synthesis
    order, each part's stored ``position`` is its 1-based index in that tuple, and
    each part keeps its exact ``ScriptChunk`` text, speaker, cast voice, voice
    fingerprint, and pause. Each part must also carry the chunk number equal to
    that position and the generated chunk id for it, and its ``PreparedPart.voice``
    must equal the wrapped ``chunk.voice``; otherwise the stored cast identity and
    the voice actually synthesized could diverge on a resume. Parts are stored
    with their resolved effective voice, a per-part fingerprint, and a NULL stage;
    the run-level fingerprint and the structural resume map go into
    ``config_snapshot``. The full script text and each part text are stored as
    ``tts_script`` sources, and a non-empty style prompt as a private
    ``tts_direction`` source; an empty-string style prompt is rejected with
    :class:`NativeSnapshotValidationError` before any insert, because it would be
    hashed into the run identity while no direction source recorded it, so the view
    could never verify the committed run.

    Validation runs before any insert, and a ``run_root`` already owned by any
    run (including a ``legacy_import`` one) raises
    :class:`NativeSnapshotRunRootConflictError`. The snapshot owns its
    transaction: a caller already inside ``repository.transaction()`` raises
    :class:`NativeSnapshotInTransactionError` before any insert, so a caller
    rollback can never erase a returned snapshot. A whitelisted structural
    identity value the history redaction boundary would change raises
    :class:`NativeSnapshotValidationError` instead of committing an
    unreconstructable snapshot. Run, parts, and text sources commit in one
    transaction, so a failure leaves no partial run behind. No filesystem path,
    JSON state file, lock, or provider is touched.

    Only a ``polza-tts`` or ``openrouter-tts`` provider is accepted at this
    foundation stage. Every other provider -- ``qwen-local``, ``omnivoice-local``,
    ``polza-chat-audio``, or an unknown identifier -- is rejected with
    :class:`NativeSnapshotValidationError` before any insert: its resume identity
    inputs are not persisted on ``PreparedRun``, so the snapshot would be
    unreconstructable.
    """
    _require_standalone_transaction(repository)
    provider = _require_nonempty_text(prepared.provider, "provider")
    if provider not in _NATIVE_SNAPSHOT_SUPPORTED_TTS_PROVIDERS:
        # Fail closed before any insert: only a route whose full nonsecret resume
        # identity is carried on the prepared run is persisted. This covers the
        # known-but-unreconstructable qwen-local, omnivoice-local, and
        # polza-chat-audio routes and any unknown provider; the fixed message
        # never echoes the identifier. See the module docstring's known limits.
        raise NativeSnapshotValidationError(_UNSUPPORTED_TTS_ROUTE_REJECTED)
    model = _require_nonempty_text(prepared.model, "model")
    voice = _require_nonempty_text(prepared.voice, "voice")
    prompt_mode = _require_nonempty_text(prepared.prompt_mode, "prompt_mode")
    resolved_format = _require_nonempty_text(script_format, "script_format")
    if not isinstance(script_text, str):
        raise NativeSnapshotValidationError("script_text must be a string")
    if user_label is not None:
        _require_nonempty_text(user_label, "user_label")
    if not prepared.parts:
        raise NativeSnapshotValidationError("prepared run has no parts to snapshot")
    if script_path is not None and not isinstance(script_path, Path):
        raise NativeSnapshotValidationError("script_path must be a Path or None")
    if voice_identity is not None:
        _require_nonempty_text(voice_identity, "voice_identity")
    if synthesis_identity is not None:
        _require_nonempty_text(synthesis_identity, "synthesis_identity")
    # ``None`` means this run has no style direction, and a non-empty string is
    # stored verbatim below. A supplied empty string is neither: it would be hashed
    # into the run identity while the ``tts_direction`` source is written only for a
    # truthy prompt, so the view -- which rebuilds that identity from committed rows
    # -- could never verify the run. Reject it before any insert instead of
    # committing a snapshot its own reader must refuse.
    style_prompt = prepared.style_prompt
    if style_prompt is not None and (not isinstance(style_prompt, str) or not style_prompt):
        raise NativeSnapshotValidationError("style_prompt must be None or a non-empty string")
    validated_output_options = _require_output_options(output_options)
    # The ``omnivoice-local`` preset dialogue route is the only one that carries a
    # voice-bank identity; every other route must not, so a mismatched pairing
    # fails closed instead of committing an identity the reader cannot rebuild.
    voice_bank_payload = _require_voice_bank_identity(prepared, provider)
    _require_dialogue_bank_consistency(prepared, voice_bank_payload)

    identity = _run_identity_payload(
        provider=provider,
        model=model,
        voice=voice,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        voice_identity=voice_identity,
        synthesis_identity=synthesis_identity,
        voice_bank=voice_bank_payload,
    )

    planned: list[_PlannedPart] = []
    for position, part in enumerate(prepared.parts, start=1):
        chunk = part.chunk
        number = _require_positive_int(chunk.number, "chunk number")
        chunk_id = _require_nonempty_text(chunk.id, "chunk id")
        # The raw-saved seam links paid bytes with ``receipt.number ==
        # part.position`` and a receipt id bounded to the generated form, so a
        # snapshot whose number, position, and id disagree could never be linked.
        if number != position:
            raise NativeSnapshotValidationError(
                f"chunk number {number} must equal its 1-based order {position}"
            )
        if not _is_generated_chunk_id(chunk_id, number):
            # The id is not echoed: a malformed value could be a secret.
            raise NativeSnapshotValidationError(
                f"chunk id is not the generated id for number {number}"
            )
        if part.voice != chunk.voice:
            raise NativeSnapshotValidationError(
                f"part {chunk_id!r} voice must match its wrapped chunk voice"
            )
        if not isinstance(chunk.text, str):
            raise NativeSnapshotValidationError(f"chunk {chunk_id!r} text must be a string")
        effective_voice = _resolve_effective_voice(part, voice)
        part_fingerprint = _part_fingerprint(
            identity,
            chunk=chunk,
            position=position,
            effective_voice=effective_voice,
        )
        planned.append(
            _PlannedPart(
                position=position,
                text=chunk.text,
                content_hash=_sha256_text(chunk.text),
                effective_voice=effective_voice,
                fingerprint=part_fingerprint,
                config_entry={
                    "position": position,
                    "number": number,
                    "id": chunk_id,
                    "speaker": chunk.speaker,
                    "cast_voice": chunk.voice,
                    "voice_fingerprint": chunk.voice_fingerprint,
                    "pause_after_ms": chunk.pause_after_ms,
                    "effective_voice": effective_voice,
                    "fingerprint": part_fingerprint,
                },
            )
        )

    script_sha256 = _sha256_text(script_text)
    snapshot_fingerprint = _snapshot_fingerprint(
        identity,
        script_format=resolved_format,
        script_sha256=script_sha256,
        part_fingerprints=[item.fingerprint for item in planned],
    )
    normalized_root = _normalize_run_root(run_root)
    config_snapshot = _build_snapshot_config(
        identity=identity,
        snapshot_fingerprint=snapshot_fingerprint,
        script_format=resolved_format,
        script_sha256=script_sha256,
        script_path=None if script_path is None else str(script_path.resolve()),
        part_count=len(planned),
        part_entries=[item.config_entry for item in planned],
        output_options=validated_output_options,
        voice_bank=voice_bank_payload,
    )
    # The history boundary redacts secret-looking values in its own copy; if that
    # would change any whitelisted structural identity, persisting it would store
    # an identity that can never be reconstructed, so reject the whole snapshot
    # with a fixed message that never echoes the offending value.
    if _redact_snapshot(config_snapshot) != config_snapshot:
        raise NativeSnapshotValidationError(_REDACTED_IDENTITY_REJECTED)

    with repository.transaction():
        # Inside BEGIN IMMEDIATE the check is atomic with the inserts below, so
        # two writers cannot both claim this run root in the same database.
        if repository.find_runs_by_root(normalized_root, limit=1):
            raise NativeSnapshotRunRootConflictError(
                f"a history run already owns run root {normalized_root!r}"
            )
        run = repository.create_run(
            operation=NATIVE_SNAPSHOT_OPERATION,
            run_root=normalized_root,
            user_label=user_label,
            config_snapshot=config_snapshot,
        )
        repository.add_text_source(
            run.run_uuid,
            kind=TEXT_KIND_TTS_SCRIPT,
            origin=NATIVE_SNAPSHOT_ORIGIN,
            content=script_text,
            content_hash=script_sha256,
            text_completeness=TEXT_COMPLETENESS_COMPLETE,
        )
        if style_prompt:
            repository.add_text_source(
                run.run_uuid,
                kind=TEXT_KIND_TTS_DIRECTION,
                origin=NATIVE_SNAPSHOT_ORIGIN,
                content=style_prompt,
                content_hash=_sha256_text(style_prompt),
                text_completeness=TEXT_COMPLETENESS_COMPLETE,
            )
        parts: list[PartRecord] = []
        for item in planned:
            # No stage: the raw-saved seam later moves NULL -> raw_saved.
            record = repository.add_part(
                run.run_uuid,
                position=item.position,
                prepared_text=item.text,
                voice=item.effective_voice,
                fingerprint=item.fingerprint,
            )
            parts.append(record)
            repository.add_text_source(
                run.run_uuid,
                kind=TEXT_KIND_TTS_SCRIPT,
                origin=NATIVE_SNAPSHOT_ORIGIN,
                content=item.text,
                part_uuid=record.part_uuid,
                content_hash=item.content_hash,
                text_completeness=TEXT_COMPLETENESS_COMPLETE,
            )

    return PreparedTtsSnapshot(
        run=run,
        parts=tuple(parts),
        fingerprint=snapshot_fingerprint,
    )
