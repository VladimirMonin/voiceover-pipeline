"""Read-only, single-snapshot view of one native prepared-TTS run.

Plan sections 5 and 6 make SQLite the single source of truth for run history and
make recovery read that committed state instead of guessing from JSON or the
filesystem. This module is the read counterpart of
:mod:`voiceover_pipeline.history.native_snapshot`: it loads one native
prepared-TTS run and its parts, attempts, artifacts, and text sources under one
short SQLite ``BEGIN DEFERRED`` read transaction, then recomputes the version-1
native fingerprint payload from those committed rows and refuses to return a
partially trusted view when any of them disagrees.

Contract:

* The whole view is read under one deferred read transaction. The first
  ``SELECT`` pins the WAL snapshot; the transaction takes no write lock, and
  every row the caller sees belongs to that same committed revision. A writer on
  another connection may commit while this read transaction is open: this reader
  keeps the old, coherent view tagged with its old revision, and the next call
  sees the new one. The transaction is never held across a filesystem, provider,
  or network call, and a caller already inside ``repository.transaction()`` is
  refused before any read.
* The view is read-only: it opens no JSON file, run directory, provider, or
  network, and it never writes a row or a file. File availability is never
  inferred from a database row, because SQLite and the filesystem do not share a
  transaction.
* The run must be a native prepared-TTS snapshot. A legacy-imported run, a run
  whose operation is not the native snapshot's ``tts``, an unknown snapshot origin
  or version, a malformed or non-redaction-stable config, a provider outside the
  native writer's allowlist, a non-contiguous part position/number/generated id, a
  part row that disagrees with its config entry, a native part source that names a
  part outside the run, a missing or altered run/part/style fingerprint, a
  missing, duplicate, or altered native text or style source, and a text source
  whose non-null part or artifact link names a row outside this run all fail closed
  with a typed error instead of returning a partial view.
* Attempt and artifact rows are returned as plain committed records and their
  run/part/attempt associations within this run are verified, including that an
  artifact naming both a part and an attempt names the part that attempt is bound
  to. Every returned text source's non-null part and artifact links are verified to
  resolve inside this run, and a source naming both a part and an artifact must not
  contradict the part that artifact names. This module never authorizes a paid
  POST/GET, never infers resume permission from a stored marker, and never treats
  that marker as proof that a file still exists. A completed run and an
  unconfirmed ``submitting`` marker are both readable as records; neither grants
  permission to repeat or finish the work.

Known limits at this foundation stage:

* The version-1 identity payload hashes the style prompt itself, but the writer
  stores the prompt as a ``tts_direction`` source only when it is a non-empty
  string, so an empty-string style prompt is not recoverable from committed rows.
  The writer rejects a supplied empty prompt before any insert, and this view still
  fails closed on a run that carries no recoverable direction instead of guessing.
* This module reads only the database. It does not decide whether a run may be
  resumed; that is a separate, explicit recovery decision that the later
  execution wiring must make from this view plus the on-disk evidence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from ..models import ScriptChunk
from ..services.prepare import (
    OmniVoiceModeIdentity,
    OmniVoiceVoiceBankIdentity,
    QwenCloneVoiceIdentity,
    QwenModeVoiceIdentity,
    SpeechPartDirection,
)
from .native_snapshot import (
    _NATIVE_SNAPSHOT_SUPPORTED_TTS_PROVIDERS,
    NATIVE_SNAPSHOT_FINGERPRINT_VERSION,
    NATIVE_SNAPSHOT_OPERATION,
    NATIVE_SNAPSHOT_ORIGIN,
    _is_generated_chunk_id,
    _part_fingerprint,
    _run_identity_payload,
    _sha256_text,
    _snapshot_fingerprint,
)
from .repository import (
    TEXT_KIND_TTS_DIRECTION,
    TEXT_KIND_TTS_SCRIPT,
    ArtifactRecord,
    AttemptRecord,
    HistoryRepository,
    PartRecord,
    RunRecord,
    TextSourceRecord,
    _redact_snapshot,
)


class NativeViewError(RuntimeError):
    """Base class for native view contract violations."""


class NativeViewValidationError(NativeViewError, ValueError):
    """The caller supplied an invalid run UUID or expected revision."""


class NativeViewInTransactionError(NativeViewError):
    """The view was requested from inside a caller's already open transaction.

    The view owns its short read transaction so the first ``SELECT`` pins one
    coherent WAL snapshot. Joining an already open transaction would silently
    read a snapshot that another part of the caller's block has already changed,
    so the call refuses to run instead.
    """


class NativeViewNotFoundError(NativeViewError):
    """No history run carries the requested UUID."""


class NativeViewRevisionConflictError(NativeViewError):
    """The committed run is not at the caller's expected revision."""


class NativeViewIntegrityError(NativeViewError):
    """The run exists but is not a consistent native prepared-TTS snapshot.

    Every integrity failure uses one of these with a fixed message, so a
    tampered text, style, config, fingerprint, or association value is never
    echoed back in an error report or log.
    """


# Fixed rejection message for a run whose committed provider is not one the
# native snapshot writer may persist. The writer's allowlist is the single owner
# of which routes captured enough nonsecret resume identity: a run carrying any
# other provider cannot have come from that writer unchanged, and a recomputed
# identity for such a route would rest on inputs the writer never stored. The
# message echoes no provider value.
_UNSUPPORTED_NATIVE_SNAPSHOT_PROVIDER = (
    "run provider is not a route the native snapshot writer persists; refusing a view "
    "whose identity inputs were never captured"
)


@dataclass(frozen=True)
class NativeTtsPart:
    """One validated part: its committed row plus its reconstructed chunk identity.

    ``text`` is the exact prepared text read back from the verified native
    ``tts_script`` source, and the remaining fields are the reconstructed
    ``ScriptChunk`` identity the part fingerprint commits to.
    """

    record: PartRecord
    number: int
    chunk_id: str
    speaker: str | None
    cast_voice: str | None
    voice_fingerprint: str | None
    pause_after_ms: int
    effective_voice: str
    fingerprint: str
    text: str
    direction: SpeechPartDirection | None = None


@dataclass(frozen=True)
class NativeTtsView:
    """One coherent, committed, read-only snapshot of a native prepared-TTS run.

    ``run.revision`` is the revision this view was read at. ``run_fingerprint``
    is the version-1 run identity recomputed from the committed rows, and each
    ``parts`` entry carries its own recomputed per-part identity. ``attempts``,
    ``artifacts``, and ``text_sources`` are the plain committed records; they
    describe what was recorded, not what is still on disk or what may be
    repeated.
    """

    run: RunRecord
    run_fingerprint: str
    script_format: str
    script_path: str | None
    script_text: str
    style_prompt: str | None
    output_options: dict[str, Any] | None
    voice_bank_identity: dict[str, Any] | None
    omnivoice_mode_identity: dict[str, Any] | None
    qwen_clone_identity: dict[str, Any] | None
    qwen_mode_identity: dict[str, Any] | None
    parts: tuple[NativeTtsPart, ...]
    attempts: tuple[AttemptRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    text_sources: tuple[TextSourceRecord, ...]


def _require_run_uuid(value: object) -> str:
    """Return the canonical UUID string, or raise without echoing the input."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise NativeViewValidationError("run_uuid must be a UUID string") from exc


def _require_expected_revision(value: object) -> int:
    """Return a positive, non-bool revision, or raise with a fixed message."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NativeViewValidationError("expected_revision must be a positive integer")
    return value


def _require_standalone_transaction(repository: HistoryRepository) -> None:
    """Fail closed when the caller already holds an open transaction."""
    if repository._connection.in_transaction:
        raise NativeViewInTransactionError(
            "load_native_tts_view owns its read transaction and must run outside any already "
            "open transaction"
        )


def _config_required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise NativeViewIntegrityError(
            f"native snapshot config field {key!r} is missing or not a non-empty string"
        )
    return value


def _config_optional_str(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise NativeViewIntegrityError(
            f"native snapshot config field {key!r} is not null or a non-empty string"
        )
    return value


def _config_required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeViewIntegrityError(
            f"native snapshot config field {key!r} is missing or not an integer"
        )
    return value


def _config_optional_output(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the committed output-processing settings, or ``None`` when absent.

    The writer stores this mapping only when the run recorded output settings, so
    an older snapshot without the key stays readable. When present it must be a
    flat mapping of plain scalar values, mirroring what the writer accepted, or
    the view fails closed with a fixed message.
    """
    value = config.get("output")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeViewIntegrityError("native snapshot output options are not a mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise NativeViewIntegrityError("native snapshot output option key is not a string")
        if not isinstance(item, (str, int, bool)) or isinstance(item, float):
            raise NativeViewIntegrityError(
                "native snapshot output option value is not a plain string, integer, or boolean"
            )
    return value


def _config_optional_voice_bank(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the committed voice-bank identity, or ``None`` when absent.

    The writer stores this block only for the admitted ``omnivoice-local`` preset
    bank routes (the two-cast dialogue and the single-profile monologue). When
    present it must parse as the same identity the writer
    hash-covered, or the view fails closed with a fixed message; the block is
    returned verbatim so the recomputed run identity matches the committed one
    byte-for-byte.
    """
    value = config.get("voice_bank")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeViewIntegrityError("native snapshot voice-bank identity is not a mapping")
    try:
        OmniVoiceVoiceBankIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeViewIntegrityError("native snapshot voice-bank identity is malformed") from exc
    return value


def _config_optional_omnivoice_mode(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the committed non-preset OmniVoice mode identity, or ``None``.

    The writer stores this block only for the admitted ``omnivoice-local``
    ``auto``/``clone``/``design`` modes. When present it must parse as the same
    identity the writer hash-covered, or the view fails closed with a fixed
    message; the block is returned verbatim so the recomputed run identity matches
    the committed one byte-for-byte.
    """
    value = config.get("omnivoice_mode")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeViewIntegrityError("native snapshot omnivoice mode identity is not a mapping")
    try:
        OmniVoiceModeIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeViewIntegrityError(
            "native snapshot omnivoice mode identity is malformed"
        ) from exc
    return value


def _config_optional_qwen_clone(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the committed clone identity, or ``None`` when absent.

    The writer stores this block only for the admitted ``qwen-local`` clone route.
    When present it must parse as the same identity the writer hash-covered, or the
    view fails closed with a fixed message; the block is returned verbatim so the
    recomputed run identity matches the committed one byte-for-byte.
    """
    value = config.get("qwen_clone")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeViewIntegrityError("native snapshot qwen clone identity is not a mapping")
    try:
        QwenCloneVoiceIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeViewIntegrityError("native snapshot qwen clone identity is malformed") from exc
    return value


def _config_optional_qwen_mode(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the committed instructed-mode identity, or ``None`` when absent.

    The writer stores this block only for the admitted ``qwen-local``
    preset/design routes. When present it must parse as the same identity the
    writer hash-covered, or the view fails closed with a fixed message; the block
    is returned verbatim so the recomputed run identity matches the committed one
    byte-for-byte.
    """
    value = config.get("qwen_mode")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NativeViewIntegrityError("native snapshot qwen mode identity is not a mapping")
    try:
        QwenModeVoiceIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeViewIntegrityError("native snapshot qwen mode identity is malformed") from exc
    return value


def _entry_field(entry: dict[str, Any], key: str) -> Any:
    if key not in entry:
        raise NativeViewIntegrityError(f"native snapshot part entry is missing {key!r}")
    return entry[key]


def _entry_str(entry: dict[str, Any], key: str) -> str:
    value = _entry_field(entry, key)
    if not isinstance(value, str) or not value:
        raise NativeViewIntegrityError(
            f"native snapshot part entry field {key!r} is not a non-empty string"
        )
    return value


def _entry_optional_str(entry: dict[str, Any], key: str) -> str | None:
    value = _entry_field(entry, key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NativeViewIntegrityError(
            f"native snapshot part entry field {key!r} is not null or a string"
        )
    return value


def _entry_int(entry: dict[str, Any], key: str) -> int:
    value = _entry_field(entry, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeViewIntegrityError(
            f"native snapshot part entry field {key!r} is not an integer"
        )
    return value


def _entry_direction(entry: dict[str, Any], part: PartRecord) -> SpeechPartDirection | None:
    """Reconstruct one part's committed ``speech-parts`` direction, or fail closed.

    A part that committed a per-part direction records all three values in its
    config entry and its row; a part that did not records none in both places.
    Either they agree on all three or the run is refused with a fixed message, so a
    direct ``UPDATE`` of a vibe column can never silently change what a resume would
    send while its fingerprint still matched.
    """
    present = any(key in entry for key in ("vibe_shared", "vibe_specific", "vibe_effective"))
    row_values = (part.vibe_shared, part.vibe_specific, part.vibe_effective)
    row_present = any(value is not None for value in row_values)
    if not present:
        if row_present:
            raise NativeViewIntegrityError(
                "native snapshot part vibe disagrees with its config entry"
            )
        return None
    shared = _entry_optional_str(entry, "vibe_shared")
    specific = _entry_optional_str(entry, "vibe_specific")
    effective = _entry_optional_str(entry, "vibe_effective")
    if effective is None:
        raise NativeViewIntegrityError(
            "native snapshot part entry field 'vibe_effective' is not null or a string"
        )
    if (shared, specific, effective) != row_values:
        raise NativeViewIntegrityError("native snapshot part vibe disagrees with its config entry")
    return SpeechPartDirection(shared_vibe=shared, specific_vibe=specific, effective_vibe=effective)


def _require_native_config(run: RunRecord) -> dict[str, Any]:
    """Return the run's committed native config, or fail closed.

    The committed config must be a mapping written by the native snapshot writer:
    the native origin, the version of the fingerprint payload, and a config the
    history redaction boundary would leave unchanged. The last check mirrors the
    writer's own fail-closed guarantee, so a tampered config that smuggled a
    secret-looking value cannot become a trusted view.
    """
    config = run.config_snapshot
    if not isinstance(config, dict):
        raise NativeViewIntegrityError("native snapshot config is missing or not a mapping")
    if config.get("operation_origin") != NATIVE_SNAPSHOT_ORIGIN:
        raise NativeViewIntegrityError("run does not carry the native snapshot origin")
    version = config.get("snapshot_version")
    if isinstance(version, bool) or version != NATIVE_SNAPSHOT_FINGERPRINT_VERSION:
        raise NativeViewIntegrityError("run does not carry a supported native snapshot version")
    try:
        redacted = _redact_snapshot(config)
    except ValueError as exc:
        raise NativeViewIntegrityError("native snapshot config is not redaction-stable") from exc
    if redacted != config:
        raise NativeViewIntegrityError("native snapshot config is not redaction-stable")
    return config


def _exactly_one(sources: list[TextSourceRecord], description: str) -> TextSourceRecord:
    if len(sources) != 1:
        raise NativeViewIntegrityError(f"expected exactly one {description}; found {len(sources)}")
    return sources[0]


def _require_source_text(source: TextSourceRecord, description: str) -> str:
    """Return a native source's text only when its stored hash matches it."""
    content = source.content
    if not isinstance(content, str):
        raise NativeViewIntegrityError(f"{description} has no text content")
    if source.content_hash != _sha256_text(content):
        raise NativeViewIntegrityError(f"{description} content hash does not match its text")
    return content


def _resolve_style_prompt(text_sources: list[TextSourceRecord]) -> str | None:
    """Return the run's style prompt from its committed native direction source.

    The writer stores the prompt only when it is non-empty, so zero direction
    sources means no prompt and more than one is a duplicate. A direction source
    that is not a native, run-level, hash-matching source fails closed.
    """
    directions = [source for source in text_sources if source.kind == TEXT_KIND_TTS_DIRECTION]
    if not directions:
        return None
    if len(directions) != 1:
        raise NativeViewIntegrityError("run carries more than one native tts_direction source")
    source = directions[0]
    if (
        source.origin != NATIVE_SNAPSHOT_ORIGIN
        or source.part_uuid is not None
        or source.artifact_uuid is not None
    ):
        raise NativeViewIntegrityError("run carries a malformed native tts_direction source")
    content = source.content
    if not isinstance(content, str) or not content:
        raise NativeViewIntegrityError("native tts_direction source has no style text")
    if source.content_hash != _sha256_text(content):
        raise NativeViewIntegrityError("native tts_direction source content hash is altered")
    return content


def _verify_relations(
    run: RunRecord,
    parts: list[PartRecord],
    attempts: list[AttemptRecord],
    artifacts: list[ArtifactRecord],
    text_sources: list[TextSourceRecord],
) -> None:
    """Verify every cross-entity association stays inside this run.

    Only the committed associations are checked; no database row is treated as
    proof that an artifact's file exists, and no attempt marker is treated as
    permission to repeat or finish a paid call. An artifact that names both a part
    and an attempt must name the part that attempt is bound to, so a same-run
    artifact can never be read as evidence for a different part's bytes. A
    run-level artifact role carries no part of its own, so no part equality
    applies to it. Every text source's non-null part and artifact links must
    resolve inside this run, because a row inserted with the foreign keys disabled
    can name another run's part or artifact and would otherwise be returned as a
    plain record; only in-run resolution is required, so a source attached to a
    same-run run-level artifact stays readable for a later ASR or timing role.
    """
    part_uuids = {part.part_uuid for part in parts}
    attempts_by_uuid = {attempt.attempt_uuid: attempt for attempt in attempts}
    artifacts_by_uuid = {artifact.artifact_uuid: artifact for artifact in artifacts}
    for attempt in attempts:
        if attempt.run_uuid != run.run_uuid:
            raise NativeViewIntegrityError("attempt row belongs to another run")
        if attempt.part_uuid is not None and attempt.part_uuid not in part_uuids:
            raise NativeViewIntegrityError("attempt references a part outside this run")
    for artifact in artifacts:
        if artifact.run_uuid != run.run_uuid:
            raise NativeViewIntegrityError("artifact row belongs to another run")
        if artifact.part_uuid is not None and artifact.part_uuid not in part_uuids:
            raise NativeViewIntegrityError("artifact references a part outside this run")
        if artifact.attempt_uuid is not None:
            linked_attempt = attempts_by_uuid.get(artifact.attempt_uuid)
            if linked_attempt is None:
                raise NativeViewIntegrityError("artifact references an attempt outside this run")
            if artifact.part_uuid is not None and linked_attempt.part_uuid != artifact.part_uuid:
                raise NativeViewIntegrityError(
                    "artifact part does not match the part bound to its attempt"
                )
    for source in text_sources:
        if source.part_uuid is not None and source.part_uuid not in part_uuids:
            raise NativeViewIntegrityError("text source references a part outside this run")
        if source.artifact_uuid is not None:
            linked_artifact = artifacts_by_uuid.get(source.artifact_uuid)
            if linked_artifact is None:
                raise NativeViewIntegrityError(
                    "text source references an artifact outside this run"
                )
            if (
                source.part_uuid is not None
                and linked_artifact.part_uuid is not None
                and linked_artifact.part_uuid != source.part_uuid
            ):
                raise NativeViewIntegrityError(
                    "text source part does not match the part named by its artifact"
                )


def _build_parts(
    run: RunRecord,
    config: dict[str, Any],
    parts: list[PartRecord],
    text_sources: list[TextSourceRecord],
    identity: dict[str, Any],
) -> tuple[list[NativeTtsPart], list[str]]:
    """Validate the committed parts and return them with their recomputed identity."""
    part_count = _config_required_int(config, "part_count")
    entries = config.get("parts")
    if not isinstance(entries, list):
        raise NativeViewIntegrityError("native snapshot config parts field is missing or malformed")
    if len(entries) != part_count or len(parts) != part_count:
        raise NativeViewIntegrityError(
            "native snapshot part count disagrees with its committed parts"
        )
    if not parts:
        raise NativeViewIntegrityError("native snapshot has no parts")

    # Index this run's native per-part script sources once: each part then resolves
    # its own source in one lookup instead of re-scanning every source, and a native
    # part-level source naming a part outside this run is detected rather than
    # silently ignored. Duplicate and missing sources per part stay a fail-closed
    # ``_exactly_one`` check below.
    part_uuids = {part.part_uuid for part in parts}
    sources_by_part: dict[str, list[TextSourceRecord]] = {}
    for source in text_sources:
        if (
            source.kind != TEXT_KIND_TTS_SCRIPT
            or source.origin != NATIVE_SNAPSHOT_ORIGIN
            or source.artifact_uuid is not None
            or source.part_uuid is None
        ):
            continue
        if source.part_uuid not in part_uuids:
            raise NativeViewIntegrityError(
                "native tts_script source references a part outside this run"
            )
        sources_by_part.setdefault(source.part_uuid, []).append(source)

    native_parts: list[NativeTtsPart] = []
    part_fingerprints: list[str] = []
    for position, part in enumerate(parts, start=1):
        entry = entries[position - 1]
        if not isinstance(entry, dict):
            raise NativeViewIntegrityError("native snapshot part entry is not a mapping")
        entry_position = _entry_int(entry, "position")
        number = _entry_int(entry, "number")
        chunk_id = _entry_str(entry, "id")
        effective_voice = _entry_str(entry, "effective_voice")
        entry_fingerprint = _entry_str(entry, "fingerprint")
        if part.position != position or entry_position != position or number != position:
            raise NativeViewIntegrityError(
                "native snapshot parts are not contiguous in position and number"
            )
        if not _is_generated_chunk_id(chunk_id, number):
            raise NativeViewIntegrityError(
                "native snapshot part id is not the generated id for its number"
            )
        if part.voice != effective_voice or part.fingerprint != entry_fingerprint:
            raise NativeViewIntegrityError(
                "native snapshot part row disagrees with its config entry"
            )
        source = _exactly_one(
            sources_by_part.get(part.part_uuid, []),
            f"native tts_script source for part position {position}",
        )
        text = _require_source_text(
            source, f"native tts_script source for part position {position}"
        )
        if part.prepared_text != text:
            raise NativeViewIntegrityError(
                "native snapshot part text disagrees with its text source"
            )
        chunk = ScriptChunk(
            number=number,
            id=chunk_id,
            text=text,
            speaker=_entry_optional_str(entry, "speaker"),
            voice=_entry_optional_str(entry, "cast_voice"),
            voice_fingerprint=_entry_optional_str(entry, "voice_fingerprint"),
            pause_after_ms=_entry_int(entry, "pause_after_ms"),
        )
        direction = _entry_direction(entry, part)
        recomputed = _part_fingerprint(
            identity,
            chunk=chunk,
            position=position,
            effective_voice=effective_voice,
            direction=direction,
        )
        if recomputed != part.fingerprint:
            raise NativeViewIntegrityError(
                "native snapshot part fingerprint does not match its committed identity"
            )
        part_fingerprints.append(recomputed)
        native_parts.append(
            NativeTtsPart(
                record=part,
                number=number,
                chunk_id=chunk_id,
                speaker=chunk.speaker,
                cast_voice=chunk.voice,
                voice_fingerprint=chunk.voice_fingerprint,
                pause_after_ms=chunk.pause_after_ms,
                effective_voice=effective_voice,
                fingerprint=recomputed,
                text=text,
                direction=direction,
            )
        )
    return native_parts, part_fingerprints


def _read_committed_view(
    repository: HistoryRepository,
    run_uuid: str,
    expected_revision: int | None,
) -> NativeTtsView:
    """Read and verify one run inside the caller's already open read transaction."""
    try:
        # First SELECT: pins the WAL snapshot for the rest of this transaction.
        run = repository.get_run(run_uuid)
    except json.JSONDecodeError as exc:
        raise NativeViewIntegrityError("native snapshot config is not valid JSON") from exc
    if run is None:
        raise NativeViewNotFoundError(f"no history run with UUID {run_uuid!r}")
    if expected_revision is not None and run.revision != expected_revision:
        raise NativeViewRevisionConflictError(f"run {run_uuid!r} is not at the expected revision")
    if run.legacy_source_root is not None:
        raise NativeViewIntegrityError(
            "run is an imported legacy run, not a native prepared-TTS snapshot"
        )
    if run.operation != NATIVE_SNAPSHOT_OPERATION:
        # The native writer owns this column; an operation rewritten by a direct
        # UPDATE leaves every fingerprint intact, so only this check can refuse it.
        raise NativeViewIntegrityError(
            "run operation is not the native prepared-TTS snapshot operation"
        )

    config = _require_native_config(run)
    parts = repository.get_parts(run_uuid)
    try:
        attempts = repository.get_attempts(run_uuid)
        artifacts = repository.get_artifacts(run_uuid)
    except json.JSONDecodeError as exc:
        raise NativeViewIntegrityError(
            "native snapshot attempt or artifact metadata is not valid JSON"
        ) from exc
    text_sources = repository.get_text_sources(run_uuid)

    snapshot_fingerprint = _config_required_str(config, "snapshot_fingerprint")
    script_format = _config_required_str(config, "script_format")
    configured_script_sha256 = _config_required_str(config, "script_sha256")
    script_path = _config_optional_str(config, "script_path")
    provider = _config_required_str(config, "provider")
    if provider not in _NATIVE_SNAPSHOT_SUPPORTED_TTS_PROVIDERS:
        # The writer's allowlist is the single owner of which routes persisted every
        # nonsecret resume input. A run carrying any other provider either never
        # came from that writer or was tampered with its fingerprints rebuilt, and
        # this reader would otherwise trust an identity the writer never captured.
        # The check runs before any part, style, or fingerprint value is trusted.
        raise NativeViewIntegrityError(_UNSUPPORTED_NATIVE_SNAPSHOT_PROVIDER)
    model = _config_required_str(config, "model")
    voice = _config_required_str(config, "voice")
    prompt_mode = _config_required_str(config, "prompt_mode")
    voice_identity = _config_optional_str(config, "voice_identity")
    synthesis_identity = _config_optional_str(config, "synthesis_identity")
    voice_bank = _config_optional_voice_bank(config)
    omnivoice_mode = _config_optional_omnivoice_mode(config)
    if provider == "omnivoice-local":
        # The writer stores exactly one local-OmniVoice identity block per run: the
        # voice-bank block for the preset bank routes, or the mode block for
        # auto/clone/design. A run with neither or both cannot have come from that
        # writer unchanged.
        if voice_bank is None and omnivoice_mode is None:
            raise NativeViewIntegrityError(
                "native snapshot local OmniVoice run is missing its voice-bank or mode identity"
            )
        if voice_bank is not None and omnivoice_mode is not None:
            raise NativeViewIntegrityError(
                "native snapshot local OmniVoice run carries two identity blocks"
            )
    qwen_clone = _config_optional_qwen_clone(config)
    qwen_mode = _config_optional_qwen_mode(config)
    fallback_voice = _config_optional_str(config, "fallback_voice")
    if provider == "polza-chat-audio" and fallback_voice is None:
        # The writer always records the resolved compatibility fallback voice for
        # this route, so a run without one did not come from that writer unchanged.
        raise NativeViewIntegrityError(
            "native snapshot polza-chat-audio run is missing its fallback voice"
        )
    if provider == "qwen-local":
        # The writer stores exactly one local-Qwen identity block per run: the clone
        # block for clone mode, or the instructed-mode block for preset/design. A run
        # with neither or both cannot have come from that writer unchanged.
        if qwen_clone is None and qwen_mode is None:
            raise NativeViewIntegrityError(
                "native snapshot local Qwen run is missing its mode identity"
            )
        if qwen_clone is not None and qwen_mode is not None:
            raise NativeViewIntegrityError(
                "native snapshot local Qwen run carries two identity blocks"
            )

    style_prompt = _resolve_style_prompt(text_sources)
    identity = _run_identity_payload(
        provider=provider,
        model=model,
        voice=voice,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        voice_identity=voice_identity,
        synthesis_identity=synthesis_identity,
        voice_bank=voice_bank,
        omnivoice_mode=omnivoice_mode,
        qwen_clone=qwen_clone,
        qwen_mode=qwen_mode,
        fallback_voice=fallback_voice,
    )
    native_parts, part_fingerprints = _build_parts(run, config, parts, text_sources, identity)

    script_source = _exactly_one(
        [
            source
            for source in text_sources
            if source.kind == TEXT_KIND_TTS_SCRIPT
            and source.origin == NATIVE_SNAPSHOT_ORIGIN
            and source.part_uuid is None
            and source.artifact_uuid is None
        ],
        "run-level native tts_script source",
    )
    script_text = _require_source_text(script_source, "run-level native tts_script source")
    script_sha256 = _sha256_text(script_text)
    if script_sha256 != configured_script_sha256:
        raise NativeViewIntegrityError(
            "native snapshot script hash does not match its committed text source"
        )

    run_fingerprint = _snapshot_fingerprint(
        identity,
        script_format=script_format,
        script_sha256=script_sha256,
        part_fingerprints=part_fingerprints,
    )
    if run_fingerprint != snapshot_fingerprint:
        raise NativeViewIntegrityError(
            "native snapshot run fingerprint does not match its committed identity"
        )

    _verify_relations(run, parts, attempts, artifacts, text_sources)

    return NativeTtsView(
        run=run,
        run_fingerprint=run_fingerprint,
        script_format=script_format,
        script_path=script_path,
        script_text=script_text,
        style_prompt=style_prompt,
        output_options=_config_optional_output(config),
        voice_bank_identity=voice_bank,
        omnivoice_mode_identity=omnivoice_mode,
        qwen_clone_identity=qwen_clone,
        qwen_mode_identity=qwen_mode,
        parts=tuple(native_parts),
        attempts=tuple(attempts),
        artifacts=tuple(artifacts),
        text_sources=tuple(text_sources),
    )


def load_native_tts_view(
    repository: HistoryRepository,
    run_uuid: str,
    *,
    expected_revision: int | None = None,
) -> NativeTtsView:
    """Load one coherent, verified, read-only view of a native prepared-TTS run.

    The whole view is read under one short ``BEGIN DEFERRED`` read transaction:
    the first ``SELECT`` pins the WAL snapshot, the transaction takes no write
    lock, and every returned row belongs to that one committed revision. A writer
    on another connection that commits while this read is open cannot tear it:
    this call returns the old, coherent view tagged with its old ``run.revision``,
    and the next call sees the new revision.

    ``run_uuid`` must be a UUID string and ``expected_revision``, when given, must
    be a positive integer; both are validated before any read. ``expected_revision``
    is compared against the pinned revision, so a stale caller reloads instead of
    trusting a view read from a newer revision.

    The run must be a native prepared-TTS snapshot: a legacy-imported run, a run
    whose operation is not the native snapshot's ``tts``, an unsupported origin or
    version, a malformed or non-redaction-stable config, a provider outside the
    native writer's allowlist, a non-contiguous part position/number/generated id, a
    part row that disagrees with its config entry, a native part source that names a
    part outside the run, a missing, duplicate, or altered native text or style
    source, a text source whose non-null part or artifact link names a row outside
    this run, or a run/part fingerprint that does not match the committed identity
    all raise :class:`NativeViewIntegrityError` with a
    fixed privacy-safe message. Attempt and artifact rows are returned as plain
    records with their in-run associations verified, including that an artifact
    naming both a part and an attempt names the part that attempt is bound to, and
    every text source's non-null part and artifact links are verified to resolve
    inside this run. Neither the records nor the associations
    authorize a paid call, infer resume permission, or prove that a file exists.

    The call owns its read transaction and refuses to run inside an already open
    one. Raises :class:`NativeViewValidationError` for an invalid ``run_uuid`` or
    ``expected_revision`` before any read, :class:`NativeViewInTransactionError`
    inside an open transaction, :class:`NativeViewNotFoundError` for a missing run,
    :class:`NativeViewRevisionConflictError` for a stale ``expected_revision``, and
    :class:`NativeViewIntegrityError` for any inconsistent snapshot. No error
    message echoes a text, style, config, or fingerprint value.
    """
    resolved_uuid = _require_run_uuid(run_uuid)
    if expected_revision is not None:
        _require_expected_revision(expected_revision)
    _require_standalone_transaction(repository)

    connection = repository._connection
    connection.execute("BEGIN DEFERRED")
    try:
        view = _read_committed_view(repository, resolved_uuid, expected_revision)
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
    return view
