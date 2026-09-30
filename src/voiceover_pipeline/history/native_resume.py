"""Read-only synthesis-identity preflight for resuming a native prepared-TTS run.

Plan sections 5 and 6 make SQLite the single source of truth for run history, so a
later ``generate --resume`` must prove that the run it is about to continue was
prepared for exactly the synthesis identity it would submit now. This module is
that proof and nothing else: the read-only identity check that sits between
:func:`voiceover_pipeline.history.native_view.load_native_tts_view` and the
execution wiring that does not exist yet.

Contract:

* The committed view is loaded and verified through
  :func:`~voiceover_pipeline.history.native_view.load_native_tts_view` before any
  stored value is trusted, so a missing run, a stale ``expected_revision``, a
  malformed config, and every other inconsistent snapshot fail closed with that
  reader's typed errors. The returned
  :class:`~voiceover_pipeline.history.native_view.NativeTtsView` is tagged with the
  revision the caller asked for, so the caller can hand ``view.run.revision`` to
  the later revision compare-and-swap that a paid reservation uses.
* Every dimension a resume would change is compared against the committed rows: the
  canonical resolved run root, the script format, the run provider, model, voice,
  style prompt, prompt mode, opaque voice identity, and opaque synthesis identity,
  then the candidate part count and each part's version-1 fingerprint. The
  candidate fingerprints are recomputed with the same pure helpers the native
  writer and the native view use, so a candidate part must carry a contiguous
  ``chunk.number == position``, the generated chunk id for that number, a
  ``PreparedPart.voice`` equal to its wrapped ``chunk.voice``, and the exact text,
  speaker, cast voice, voice fingerprint, pause, effective voice, number, and count
  the committed part carries. Any difference is a typed identity conflict.
* Nothing else is compared. The raw source script file, its path, and a hash read
  from it are deliberately not part of this check: the legacy ``generate --resume``
  hashes the prepared chunks, so an edited comment or whitespace that leaves every
  prepared chunk identical still passes, while a changed spoken text, cast voice,
  speaker, pause, voice fingerprint, ordering, or part count fails here before any
  network call exists.
* The check is read-only and self-contained. It writes no database row and no file,
  takes no lock, opens no JSON state, reads no source script, and makes no provider,
  model, or network call. Its only filesystem access is resolving the candidate run
  root, and that resolution runs after the read transaction has already committed.
* The check grants no permission. It never authorizes a paid POST or a GET, never
  decides whether an attempt may be reserved, repeated, or finished, never infers
  that an artifact file exists, and never treats a stored attempt marker as resume
  permission or as a reason to refuse. A caller must still own the run lock and a
  standalone revision compare-and-swap paid reservation before any paid submit;
  this preflight only answers whether the committed synthesis identity still
  matches.
* Conflict messages are fixed and privacy-safe: no stored or candidate text, style
  prompt, run root, or opaque identity value is echoed into an error report or log.

Known limits at this foundation stage:

* The preflight is not wired into the CLI, the executor, or any JSON output: it is a
  library seam for the later cutover, and the CLI still performs its own
  ``run_state.json`` script-hash and dialogue synthesis-identity check.
* It verifies the committed synthesis identity only. Reconstructing a run from
  stored history alone (``history resume UUID``) and deciding what to do about
  on-disk artifacts and attempts remain future work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ScriptChunk
from ..services.prepare import PreparedPart, PreparedRun
from .native_snapshot import (
    _is_generated_chunk_id,
    _normalize_run_root,
    _part_fingerprint,
    _run_identity_payload,
)
from .native_view import (
    NativeTtsView,
    _config_optional_str,
    _config_required_str,
    load_native_tts_view,
)
from .repository import HistoryRepository


class NativeResumeError(RuntimeError):
    """Base class for native resume preflight contract violations."""


class NativeResumeValidationError(NativeResumeError, ValueError):
    """The caller supplied a candidate the preflight cannot compare safely."""


class NativeResumeIdentityConflictError(NativeResumeError):
    """The candidate synthesis identity no longer matches the committed run.

    Every conflict uses one of the fixed messages below, so a tampered or merely
    different text, style prompt, run root, or opaque identity value is never
    echoed back in an error report or log.
    """


# Fixed conflict messages. Each names only the compared dimension, never the stored
# or candidate value, so an error report or log stays privacy-safe.
_RUN_ROOT_CONFLICT = "native resume identity conflict: run root changed"
_SCRIPT_FORMAT_CONFLICT = "native resume identity conflict: script format changed"
_RUN_IDENTITY_CONFLICT = (
    "native resume identity conflict: run provider, model, voice, style prompt, prompt mode, "
    "or opaque identity changed"
)
_PART_COUNT_CONFLICT = "native resume identity conflict: part count changed"
_PART_ORDER_CONFLICT = (
    "native resume identity conflict: candidate parts are not contiguous with the generated "
    "chunk ids and numbers the committed run stores"
)
_PART_IDENTITY_CONFLICT = (
    "native resume identity conflict: part text, cast voice, voice fingerprint, speaker, pause, "
    "or effective voice changed"
)


def _require_identity_text(value: object, field_name: str) -> str:
    """Return a non-empty identity string, or raise a fixed validation error."""
    if not isinstance(value, str) or not value.strip():
        raise NativeResumeValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_optional_identity(value: object, field_name: str) -> str | None:
    """Return ``None`` or a non-empty identity string, mirroring the writer."""
    if value is None:
        return None
    return _require_identity_text(value, field_name)


def _require_prepared(prepared: object) -> PreparedRun:
    """Return the candidate prepared run, or raise before any database read.

    The candidate shape is checked here rather than trusted, so an unusable
    candidate never opens a read transaction. The field rules mirror the native
    writer: provider, model, voice, and prompt mode must be non-empty, and a style
    prompt must be ``None`` or a non-empty string.
    """
    if not isinstance(prepared, PreparedRun):
        raise NativeResumeValidationError("prepared must be a PreparedRun")
    _require_identity_text(prepared.provider, "prepared provider")
    _require_identity_text(prepared.model, "prepared model")
    _require_identity_text(prepared.voice, "prepared voice")
    _require_identity_text(prepared.prompt_mode, "prepared prompt mode")
    style_prompt = prepared.style_prompt
    if style_prompt is not None and (not isinstance(style_prompt, str) or not style_prompt):
        raise NativeResumeValidationError(
            "prepared style prompt must be None or a non-empty string"
        )
    if not isinstance(prepared.parts, (tuple, list)) or not prepared.parts:
        raise NativeResumeValidationError("prepared parts must be a non-empty sequence")
    for part in prepared.parts:
        if not isinstance(part, PreparedPart) or not isinstance(part.chunk, ScriptChunk):
            raise NativeResumeValidationError("prepared part must wrap a ScriptChunk")
        chunk = part.chunk
        if (
            not isinstance(chunk.number, int)
            or isinstance(chunk.number, bool)
            or not isinstance(chunk.id, str)
            or not isinstance(chunk.text, str)
            or any(
                value is not None and not isinstance(value, str)
                for value in (chunk.speaker, chunk.voice, chunk.voice_fingerprint, part.voice)
            )
            or not isinstance(chunk.pause_after_ms, int)
            or isinstance(chunk.pause_after_ms, bool)
        ):
            raise NativeResumeValidationError("prepared chunk fields have invalid types")
    return prepared


def _require_run_root(run_root: object) -> Path:
    if not isinstance(run_root, Path):
        raise NativeResumeValidationError("run_root must be a Path")
    return run_root


def _require_script_format(script_format: object) -> str:
    if not isinstance(script_format, str) or not script_format.strip():
        raise NativeResumeValidationError("script_format must be a non-empty string")
    return script_format


def _view_config(view: NativeTtsView) -> dict[str, Any]:
    """Return the view's committed native config.

    :func:`load_native_tts_view` has already verified this mapping, so the guard
    only narrows the type. Its message names no value.
    """
    config = view.run.config_snapshot
    if not isinstance(config, dict):
        raise NativeResumeError("native resume preflight requires a committed native snapshot")
    return config


def preflight_native_tts_resume(
    repository: HistoryRepository,
    run_uuid: str,
    *,
    expected_revision: int,
    prepared: PreparedRun,
    script_format: str,
    run_root: Path,
    voice_identity: str | None = None,
    synthesis_identity: str | None = None,
) -> NativeTtsView:
    """Prove that ``prepared`` is the exact synthesis identity the run stores.

    This is a read-only identity check for a later ``generate --resume``. It loads
    one coherent verified view through
    :func:`~voiceover_pipeline.history.native_view.load_native_tts_view` and then
    requires the committed run to agree with the candidate on the canonical run
    root, the script format, the run provider, model, voice, style prompt, prompt
    mode, opaque voice identity, and opaque synthesis identity, plus the part count
    and every part's version-1 fingerprint. The candidate fingerprints are
    recomputed with the same pure helpers the native writer uses, so a candidate
    part that differs in text, speaker, cast voice, voice fingerprint, pause,
    effective voice, number, or order is rejected. On success the returned
    :class:`~voiceover_pipeline.history.native_view.NativeTtsView` is the committed
    view at ``expected_revision``, whose ``run.revision`` the caller can pass to the
    later revision compare-and-swap.

    The candidate shape is validated before the read, so an unusable candidate
    never opens a transaction. The view is loaded and fully verified before any
    stored value is compared, so a missing run, a stale ``expected_revision``, a
    malformed config, or an inconsistent snapshot fails closed with the reader's own
    typed errors (:class:`NativeViewNotFoundError`,
    :class:`NativeViewRevisionConflictError`, :class:`NativeViewValidationError`,
    :class:`NativeViewIntegrityError`) and an already open caller transaction raises
    :class:`NativeViewInTransactionError`. Every identity mismatch raises
    :class:`NativeResumeIdentityConflictError` and every unusable candidate raises
    :class:`NativeResumeValidationError`, both with fixed messages that echo no
    text, style prompt, run root, or identity value.

    The call writes no row and no file, opens no JSON state, reads no source script,
    takes no lock, and makes no provider, model, or network call; its only
    filesystem access is resolving the candidate run root after the read transaction
    has committed. It never authorizes a paid POST or a GET, never decides whether an
    attempt may be reserved or finished, and never infers that a file exists. The
    caller still owns the run lock and a standalone revision compare-and-swap paid
    reservation before any paid submit.
    """
    if expected_revision is None:
        raise NativeResumeValidationError("expected_revision must be a positive integer")
    candidate_root = _require_run_root(run_root)
    candidate_format = _require_script_format(script_format)
    candidate_voice_identity = _require_optional_identity(voice_identity, "voice_identity")
    candidate_synthesis_identity = _require_optional_identity(
        synthesis_identity, "synthesis_identity"
    )
    candidate = _require_prepared(prepared)

    view = load_native_tts_view(repository, run_uuid, expected_revision=expected_revision)
    config = _view_config(view)

    # The run root is resolved only after the view's read transaction committed, so
    # no transaction is ever held across a filesystem call.
    try:
        resolved_root = _normalize_run_root(candidate_root)
    except (OSError, RuntimeError, ValueError):
        raise NativeResumeValidationError("run_root cannot be resolved") from None
    if view.run.run_root != resolved_root:
        raise NativeResumeIdentityConflictError(_RUN_ROOT_CONFLICT)
    if view.script_format != candidate_format:
        raise NativeResumeIdentityConflictError(_SCRIPT_FORMAT_CONFLICT)
    if (
        _config_required_str(config, "provider") != candidate.provider
        or _config_required_str(config, "model") != candidate.model
        or _config_required_str(config, "voice") != candidate.voice
        or _config_required_str(config, "prompt_mode") != candidate.prompt_mode
        or view.style_prompt != candidate.style_prompt
        or _config_optional_str(config, "voice_identity") != candidate_voice_identity
        or _config_optional_str(config, "synthesis_identity") != candidate_synthesis_identity
    ):
        raise NativeResumeIdentityConflictError(_RUN_IDENTITY_CONFLICT)

    if len(candidate.parts) != len(view.parts):
        raise NativeResumeIdentityConflictError(_PART_COUNT_CONFLICT)

    # Every run identity field above already equals the committed value, so this
    # payload is the same version-1 identity the view recomputed from its rows and
    # the per-part fingerprints below are directly comparable.
    identity = _run_identity_payload(
        provider=candidate.provider,
        model=candidate.model,
        voice=candidate.voice,
        style_prompt=candidate.style_prompt,
        prompt_mode=candidate.prompt_mode,
        voice_identity=candidate_voice_identity,
        synthesis_identity=candidate_synthesis_identity,
    )
    for position, part in enumerate(candidate.parts, start=1):
        chunk = part.chunk
        committed = view.parts[position - 1]
        if chunk.number != position or committed.number != chunk.number:
            raise NativeResumeIdentityConflictError(_PART_ORDER_CONFLICT)
        if not _is_generated_chunk_id(chunk.id, chunk.number):
            raise NativeResumeIdentityConflictError(_PART_ORDER_CONFLICT)
        if part.voice != chunk.voice:
            raise NativeResumeIdentityConflictError(_PART_IDENTITY_CONFLICT)
        effective_voice = candidate.voice if part.voice is None else part.voice
        if not isinstance(effective_voice, str) or not effective_voice.strip():
            raise NativeResumeIdentityConflictError(_PART_IDENTITY_CONFLICT)
        fingerprint = _part_fingerprint(
            identity, chunk=chunk, position=position, effective_voice=effective_voice
        )
        if fingerprint != committed.fingerprint:
            raise NativeResumeIdentityConflictError(_PART_IDENTITY_CONFLICT)
    return view
