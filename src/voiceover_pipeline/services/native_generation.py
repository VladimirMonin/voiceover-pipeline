"""Native DB-first generation and resume for the ordinary Polza Media TTS route.

Plan sections 5 and 6 make SQLite the single source of truth for run history:
one native run's prepared snapshot, paid attempts, raw evidence, converted
chunks, and final assembly live in the database, and the legacy JSON files become
compatibility exports. This module owns the bounded executor for the admitted
routes -- an ordinary, non-dialogue ``polza-tts`` run without integrated timing
processing, either its synchronous ``/audio/speech`` model or its async
``elevenlabs/`` ``/media`` model, the synchronous ``openrouter-tts`` route, and
the validated ``openrouter-tts`` Gemini two-speaker dialogue route, and the
``omnivoice-local`` preset two-profile bank dialogue route, and the ordinary
non-dialogue ``qwen-local`` local routes (clone and the instructed preset/design
modes) -- and their recovery decisions. Two
orthogonal integrated steps are admitted for that same
ordinary route: the recorded silence-trimming semantics (``--no-trim`` vs the
default trim) and one local post-audio step -- either local ``faster-whisper``
timings or, separately, a local ASR quality verification.

The admitted ``openrouter-tts`` dialogue route is deliberately narrower. It is one
turn per request, keeps each turn's cast voice, and requires an installed local
``--tts-quality-provider``: every converted turn is transcribed and checked before
the next paid turn and before the final concat, and the observed PASS/FAIL verdict
together with the private transcript is persisted on the TTS run, linked to that
exact turn part. It records no integrated timing and no ``--no-trim`` mixture; the
final audio is assembled with the recorded 250/600/0 ms pause plan. The admitted
``omnivoice-local`` preset dialogue route is local: one turn per request with its
cast voice-bank profile, no paid POST or paid attempt marker, no quality gate, and no
``--no-trim``/``--with-timings`` mixture. Each real local invocation reserves its own
cost-free ``local_tts_chunk`` attempt before the local model runs and links that
turn's raw bytes to it before conversion, so a crash converts from them with no
second model run and an interrupted or failed invocation stays truthful and
repeatable on an explicit resume. Every other dialogue route -- every
``polza-tts`` dialogue -- stays on the legacy executor.

Contract:

* Ownership is resolved from two independent pieces of evidence: the committed
  run row found by canonical ``run_root`` and a tiny run-local ownership
  descriptor. A committed native row always wins, so the crash window between the
  snapshot commit and the descriptor write stays native-owned. A descriptor whose
  database row is missing, an unreadable or foreign database next to native
  evidence, or a malformed descriptor fails closed; there is no silent fallback
  to the legacy JSON writer and no replacement database is created.
* Selection and mutation are serialized by the same cross-process run lock the
  history layer already owns, so two writers can never execute one run root.
* A fresh run commits its prepared snapshot and then its descriptor *before* any
  provider request exists. A resume proves the exact synthesis identity through
  :func:`~voiceover_pipeline.history.native_resume.preflight_native_tts_resume`
  and the recorded output-processing settings before any paid action.
* The paid ordering is inherited unchanged from the reservation seam: reserve a
  never-attempted part, submit once, record the accepted opaque task id, record
  the exact observed cost, persist the accepted bytes and their bounded receipt,
  link them, and only then convert and commit. No prior marker ever triggers a
  second POST; a known accepted task id is finished with GET calls only, and a
  verified raw receipt is rebuilt locally with no provider work at all. A
  synchronous route has no task id: it links the inline bytes and the exact
  ``polza-tts`` cost it reported in one transaction (``openrouter-tts`` reports
  none, so its cost stays unknown rather than being invented), and a lost or
  uncertain synchronous response stays a durable ``submitting`` marker that
  blocks every later resume with no repeat POST or GET.
* Recovery decisions read and hash on-disk evidence *outside* every database
  transaction. A completed part is skipped only after its committed digest, size,
  and file presence verify; a mismatch fails closed instead of regenerating and
  re-billing. Final assembly uses the ordered database parts, never a directory
  glob; a dialogue run concatenates the same ordered parts through the dialogue
  turn seam so each recorded pause is preserved.
* The provider is constructed lazily, only for a fresh unattempted submit or a
  known-id GET recovery. A local raw rebuild or a completed-run export repair
  never reads an API key and never builds a provider.
* A dialogue run's per-turn verdict is durable: a recorded FAIL is re-reported by
  ``generate --resume``, ``history resume``, and ``history sync`` without a second
  paid POST or local model run, and the paid raw bytes, converted audio, and cost
  stay on disk.

Known limits: this slice admits the non-dialogue ``polza-tts`` and
``openrouter-tts`` routes with the recorded trimming/timing/quality semantics in
the snapshot, the validated ``openrouter-tts`` Gemini dialogue route, the
``omnivoice-local`` preset two-profile bank dialogue route, and the ordinary
non-dialogue ``qwen-local`` local routes (clone and the instructed preset/design
modes). The
crash window between a synchronous raw receipt and its database link (covered by
local reconciliation) is the only place a synchronous observed cost cannot be
rebuilt, because the receipt carries no cost. Cloud ASR, cloud timing, the other
speech providers, and every ``polza-tts`` dialogue route stay on the legacy
executor until their own identity and processing paths are supported.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from ..artifacts import build_run_paths, build_srt, build_timing_manifest
from ..config import DEFAULT_TIMING_MODEL
from ..execution_identity import build_execution_identity
from ..gemini_dialogue import is_dialogue_format
from ..history.database import (
    HistoryDatabase,
    HistoryDatabaseError,
    connect_readonly_consistent,
)
from ..history.locking import HistoryRunLockedError, acquire_run_lock
from ..history.native_asr import (
    ARTIFACT_ROLE_SRT,
    ARTIFACT_ROLE_TIMINGS_JSON,
    ATTEMPT_CALL_TYPE_TIMING,
    ATTEMPT_CALL_TYPE_VERIFY,
    NATIVE_ASR_ORIGIN,
    OPERATION_TIMINGS,
    OPERATION_VERIFY,
    AsrHistoryArtifact,
    AsrHistorySave,
    AsrHistoryText,
    persist_asr_history,
    sha256_text,
)
from ..history.native_export import build_native_export, write_native_export
from ..history.native_resume import NativeResumeError, preflight_native_tts_resume
from ..history.native_snapshot import (
    NATIVE_SNAPSHOT_ORIGIN,
    NativeSnapshotError,
    persist_prepared_tts_snapshot,
)
from ..history.native_view import (
    NativeTtsPart,
    NativeTtsView,
    NativeViewError,
    load_native_tts_view,
)
from ..history.paths import HistoryPathsError, history_database_path
from ..history.raw_receipt import (
    PaidRawReceipt,
    PaidRawReceiptError,
    bounded_opaque_token,
    write_paid_raw_receipt,
)
from ..history.repository import (
    ARTIFACT_ROLE_CHUNK_AUDIO,
    ARTIFACT_ROLE_FINAL_AUDIO,
    ARTIFACT_ROLE_LOCAL_RAW_AUDIO,
    ARTIFACT_ROLE_PAID_RAW_AUDIO,
    ARTIFACT_ROLE_TTS_TURN_QUALITY,
    ATTEMPT_CALL_TYPE_TTS_CHUNK,
    ATTEMPT_STATUS_COMPLETED,
    ATTEMPT_STATUS_RAW_SAVED,
    ATTEMPT_STATUS_REMOTE_ACCEPTED,
    ATTEMPT_STATUS_SUBMITTING,
    MAX_QUERY_LIMIT,
    PATH_KIND_MANAGED_RELATIVE,
    RUN_STATUS_COMPLETED,
    TEXT_COMPLETENESS_COMPLETE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
    ArtifactRecord,
    AttemptRecord,
    HistoryRepository,
    HistoryRepositoryError,
    RunRecord,
)
from ..models import ASRResult, RunPaths, ScriptChunk, SynthesisResult
from ..run_state import LOG_FILE, GenerationLogger
from ..tts_quality import TTSQualityResult, evaluate_tts_transcript
from . import costs
from .prepare import (
    OmniVoiceVoiceBankIdentity,
    PreparedPart,
    PreparedRun,
    QwenCloneVoiceIdentity,
    QwenModeVoiceIdentity,
)
from .recovery import polza_media_route_model
from .synthesis import synthesize_part

# Numeric exit codes duplicated from ``cli.py`` because the CLI imports this
# service; only the documented stable codes are used here.
_EXIT_ARGS = 2
_EXIT_MISSING_DEP = 10
_EXIT_PROVIDER = 30
_EXIT_OUTPUT = 50
_EXIT_QUALITY = 60

# Run-local ownership descriptor. It identifies the committed native run and
# carries no progress and no execution permission; the run lock serializes writers
# and the database remains the only progress truth.
OWNERSHIP_FILE_NAME = ".voiceover-native-history.json"
OWNERSHIP_ARTIFACT_TYPE = "voiceover-native-history-ownership"
OWNERSHIP_VERSION = 1
# Version of the fixed trimming/assembly semantics recorded in the snapshot. A
# later change bumps it instead of silently resuming with different processing.
OUTPUT_PROCESSING_VERSION = 1
# Bounded account label for a single paid attempt; the provider identity itself is
# the model, and no user secret ever enters this field.
PAID_ACCOUNT_ALIAS = "default"

# Stable machine-readable error codes carried in the CLI error envelope.
_ERROR_OWNERSHIP_UNVERIFIABLE = "NATIVE_OWNERSHIP_UNVERIFIABLE"
_ERROR_OWNERSHIP_RUN_MISSING = "NATIVE_OWNERSHIP_RUN_MISSING"
_ERROR_PROCESSING_CHANGED = "NATIVE_PROCESSING_UNSUPPORTED"
_ERROR_EVIDENCE_INCONSISTENT = "NATIVE_EVIDENCE_INCONSISTENT"
_ERROR_RAWS_EVIDENCE_INVALID = "NATIVE_RAW_EVIDENCE_INVALID"
_ERROR_SUBMIT_UNCONFIRMED = "PAID_SUBMIT_UNCONFIRMED"
_ERROR_SYNTHESIS_FAILED = "NATIVE_SYNTHESIS_FAILED"
# Local synthesis (the admitted ``omnivoice-local`` preset dialogue route) has no
# paid POST, so it has its own bounded codes: a local model failure, verified
# local raw evidence that no longer matches, and a local reference profile that
# changed or disappeared before any local model call.
_ERROR_LOCAL_SYNTHESIS_FAILED = "NATIVE_LOCAL_SYNTHESIS_FAILED"
_ERROR_LOCAL_RAW_EVIDENCE_INVALID = "NATIVE_LOCAL_RAW_EVIDENCE_INVALID"
_ERROR_LOCAL_REFERENCE_UNAVAILABLE = "NATIVE_LOCAL_REFERENCE_UNAVAILABLE"
# The installed local runtime or its cached model for an admitted local route is
# unavailable offline; the probe runs before the first local synthesis so the
# route never downloads a model implicitly.
_ERROR_LOCAL_MODEL_UNAVAILABLE = "NATIVE_LOCAL_MODEL_UNAVAILABLE"
_ERROR_CONVERSION_FAILED = "NATIVE_CONVERSION_FAILED"
_ERROR_ASSEMBLY_FAILED = "NATIVE_ASSEMBLY_FAILED"
_ERROR_EXPORT_FAILED = "NATIVE_EXPORT_FAILED"
_ERROR_OUTPUT_UNAVAILABLE = "NATIVE_OUTPUT_UNAVAILABLE"
_ERROR_HISTORY_UNAVAILABLE = "NATIVE_HISTORY_UNAVAILABLE"
_ERROR_HISTORY_CONFLICT = "NATIVE_HISTORY_CONFLICT"
# ``history resume``/``history sync`` reconstruct one committed run from its
# snapshot alone, so a stored identity they cannot rebuild, a run root that no
# longer resolves to the requested committed run, a completed run whose final
# audio is gone, and an unattempted part that ``history sync`` must not submit all
# fail closed with their own stable code. A local ``omnivoice-local`` part that
# ``history sync`` cannot finish without running the local model reports its own
# fixed code, so a sync error never claims a paid submit was refused.
_ERROR_HISTORY_INCOMPLETE = "NATIVE_HISTORY_RECONSTRUCTION_FAILED"
_ERROR_HISTORY_RUN_MISMATCH = "NATIVE_HISTORY_RUN_MISMATCH"
_ERROR_FINAL_AUDIO_MISSING = "NATIVE_FINAL_AUDIO_MISSING"
_ERROR_SYNC_SUBMIT_REQUIRED = "NATIVE_SYNC_PAID_SUBMIT_REQUIRED"
_ERROR_SYNC_LOCAL_SYNTHESIS_REQUIRED = "NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED"
# Local timing (after the committed final audio) has its own bounded codes: a
# timing extraction failure, a timing-history persistence failure, a malformed
# recorded timing block, an unsupported timing provider, an unavailable local
# timing model, a timing artifact write failure, and duplicate linked timing
# evidence each report their own stable code.
_ERROR_TIMING_FAILED = "NATIVE_TIMING_FAILED"
_ERROR_TIMING_HISTORY_FAILED = "NATIVE_TIMING_HISTORY_FAILED"
_ERROR_TIMING_OPTIONS_INVALID = "NATIVE_TIMING_OPTIONS_INVALID"
_ERROR_TIMING_PROVIDER_UNSUPPORTED = "NATIVE_TIMING_PROVIDER_UNSUPPORTED"
_ERROR_TIMING_MODEL_UNAVAILABLE = "NATIVE_TIMING_MODEL_UNAVAILABLE"
_ERROR_TIMING_ARTIFACT_FAILED = "NATIVE_TIMING_ARTIFACT_FAILED"
_ERROR_TIMING_CONFLICT = "NATIVE_TIMING_CONFLICT"
# Local quality verification (after the committed final audio) has its own bounded
# codes: a recorded-quality block a run cannot rebuild, an unsupported/unregistered
# provider, an unavailable local model, a failed local transcription, a comparison
# mismatch, a linked-history persistence failure, and duplicate linked verification
# evidence each report their own stable code.
_ERROR_QUALITY_OPTIONS_INVALID = "NATIVE_QUALITY_OPTIONS_INVALID"
_ERROR_QUALITY_PROVIDER_UNSUPPORTED = "NATIVE_QUALITY_PROVIDER_UNSUPPORTED"
_ERROR_QUALITY_MODEL_UNAVAILABLE = "NATIVE_QUALITY_MODEL_UNAVAILABLE"
_ERROR_QUALITY_ASR_FAILED = "NATIVE_QUALITY_ASR_FAILED"
_ERROR_QUALITY_FAILED = "NATIVE_QUALITY_FAILED"
_ERROR_QUALITY_HISTORY_FAILED = "NATIVE_QUALITY_HISTORY_FAILED"
_ERROR_QUALITY_CONFLICT = "NATIVE_QUALITY_CONFLICT"
# The per-turn dialogue gate has its own bounded code for a run whose local turn
# verification never ran or could not be repaired without a model (``history
# sync``): the committed audio and cost stay, and an explicit resume is required
# to run the pending local model.
_ERROR_QUALITY_INCOMPLETE = "NATIVE_QUALITY_INCOMPLETE"

# Stable mode names for the two history entry points.
_MODE_RESUME = "resume"
_MODE_SYNC = "sync"

# Recovery route of one part.
_ROUTE_LEGACY = "legacy"
_ROUTE_NATIVE_EXISTING = "native_existing"
_ROUTE_NATIVE_NEW = "native_new"
_ROUTE_BLOCKED = "blocked"


class NativeGenerationError(RuntimeError):
    """A native generation step failed with a stable exit code and error code."""

    def __init__(self, message: str, *, code: int, error_code: str) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code


@dataclass(frozen=True)
class NativeOwnership:
    """Where a run root belongs before any provider or deletion step.

    ``route`` is one of the bounded routes above; ``run_uuid`` and ``revision``
    identify an existing committed native run, and ``error_code`` plus ``reason``
    describe a fail-closed block. ``reason`` never echoes a stored value.
    """

    route: str
    run_uuid: str | None = None
    revision: int | None = None
    error_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NativeExecutionHooks:
    """The CLI's local media presentation seams and streaming digest helper."""

    write_audio_as_mp3: Callable[[str, bytes, str, Path], None]
    trim_final_silence: Callable[[str, str, Path], None]
    mp3_duration_ms: Callable[[str, Path], int]
    concat_audio_files: Callable[[str, list[Path], Path], None]
    concat_dialogue_turns: Callable[[str, list[tuple[Path, int]], Path], None]
    sha256_file: Callable[[Path], str]
    progress: Callable[[str], None]


# Flat keys recorded in a native run's output options so a resume can prove the
# exact local timing settings instead of inferring them from the current command
# line. They are plain scalars, so the snapshot's redaction boundary and the
# verified view accept them unchanged, and a run that recorded no timing keeps
# its original output options byte-for-byte.
_TIMING_ENABLED_KEY = "timing_enabled"
_TIMING_PROVIDER_KEY = "timing_provider"
_TIMING_MODEL_KEY = "timing_model"
_TIMING_DEVICE_KEY = "timing_device"
_TIMING_COMPUTE_KEY = "timing_compute"
_TIMING_LANGUAGE_KEY = "timing_language"
_TIMING_WORD_TIMESTAMPS_KEY = "timing_word_timestamps"


@dataclass(frozen=True)
class NativeTimingOptions:
    """The local timing settings a native run records and a resume must match.

    Only the local ``faster-whisper`` provider is admitted for native timings;
    cloud timing providers stay on the legacy executor. ``model`` is ``None``
    when the command left it unset, in which case the adapter's default applies.
    """

    provider: str
    model: str | None
    device: str
    compute: str
    language: str
    word_timestamps: bool


def _timing_output_fields(timing: NativeTimingOptions | None) -> dict[str, Any]:
    """Return the flat output-option fields for a timing run, or nothing."""
    if timing is None:
        return {}
    return {
        _TIMING_ENABLED_KEY: True,
        _TIMING_PROVIDER_KEY: timing.provider,
        _TIMING_MODEL_KEY: timing.model or "",
        _TIMING_DEVICE_KEY: timing.device,
        _TIMING_COMPUTE_KEY: timing.compute,
        _TIMING_LANGUAGE_KEY: timing.language,
        _TIMING_WORD_TIMESTAMPS_KEY: timing.word_timestamps,
    }


def timing_options_from_output_options(
    options: dict[str, Any] | None,
) -> NativeTimingOptions | None:
    """Return the recorded local timing settings, or ``None`` when timing is off.

    A run that recorded no timing, or an older snapshot written before timing was
    part of the output options, reports ``None`` and a resume performs no local
    timing. A present but malformed timing block fails closed instead of silently
    dropping the requested work.
    """
    if not isinstance(options, dict) or not options.get(_TIMING_ENABLED_KEY):
        return None
    provider = options.get(_TIMING_PROVIDER_KEY)
    model = options.get(_TIMING_MODEL_KEY)
    device = options.get(_TIMING_DEVICE_KEY)
    compute = options.get(_TIMING_COMPUTE_KEY)
    language = options.get(_TIMING_LANGUAGE_KEY)
    word_timestamps = options.get(_TIMING_WORD_TIMESTAMPS_KEY)
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not isinstance(device, str)
        or not device
        or not isinstance(compute, str)
        or not compute
        or not isinstance(language, str)
        or not language
        or not isinstance(word_timestamps, bool)
    ):
        raise NativeGenerationError(
            "the recorded local timing settings are malformed; refusing to run a partial "
            "timing step.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_TIMING_OPTIONS_INVALID,
        )
    return NativeTimingOptions(
        provider=provider,
        model=model or None,
        device=device,
        compute=compute,
        language=language,
        word_timestamps=word_timestamps,
    )


def _preflight_local_timing(options: NativeTimingOptions) -> None:
    """Fail closed before any paid submit when local timing cannot run offline.

    Only the local ``faster-whisper`` provider is admitted. The probe inspects the
    installed package and its model cache without downloading, so an unavailable
    dependency or an uncached model stops the command before the paid TTS POST
    instead of after the audio is already paid for.
    """
    if options.provider != "faster-whisper":
        raise NativeGenerationError(
            "only the local faster-whisper provider is admitted for native timings; cloud "
            "timing providers stay on the legacy executor.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_TIMING_PROVIDER_UNSUPPORTED,
        )
    from ..providers.faster_whisper import faster_whisper_availability

    availability = faster_whisper_availability(options.model or DEFAULT_TIMING_MODEL)
    if not availability.available:
        raise NativeGenerationError(
            availability.remediation or "the local faster-whisper model is unavailable.",
            code=_EXIT_MISSING_DEP,
            error_code=_ERROR_TIMING_MODEL_UNAVAILABLE,
        )


# Flat keys recorded in a native run's output options so a resume can prove the
# exact local quality-verification settings instead of inferring them from the
# current command line. Like the timing fields they are plain scalars, so the
# snapshot's redaction boundary and the verified view accept them unchanged, and a
# run that requested no quality keeps its original output options byte-for-byte.
_QUALITY_ENABLED_KEY = "quality_enabled"
_QUALITY_PROVIDER_KEY = "quality_provider"
_QUALITY_MODEL_KEY = "quality_model"
_QUALITY_DEVICE_KEY = "quality_device"
_QUALITY_COMPUTE_KEY = "quality_compute"
_QUALITY_RUNTIME_KEY = "quality_runtime"
_QUALITY_LANGUAGE_KEY = "quality_language"

# The only quality providers a native run admits: the installed local ASR routes
# whose dependency probe inspects local assets and never downloads. A cloud ASR
# provider keeps the legacy executor, exactly as it did before this step.
NATIVE_LOCAL_QUALITY_PROVIDERS = frozenset({"qwen-local", "nemotron-local"})


@dataclass(frozen=True)
class NativeQualityOptions:
    """The local quality-verification settings a native run records and must match.

    Only an installed local ASR provider is admitted for native quality; a cloud
    ASR provider stays on the legacy executor. ``model`` and ``language`` are
    ``None`` when the command left them unset, in which case the provider default
    and the provider's own language detection apply.
    """

    provider: str
    model: str | None
    device: str
    compute: str
    runtime: str
    language: str | None


def _quality_output_fields(quality: NativeQualityOptions | None) -> dict[str, Any]:
    """Return the flat output-option fields for a quality run, or nothing."""
    if quality is None:
        return {}
    return {
        _QUALITY_ENABLED_KEY: True,
        _QUALITY_PROVIDER_KEY: quality.provider,
        _QUALITY_MODEL_KEY: quality.model or "",
        _QUALITY_DEVICE_KEY: quality.device,
        _QUALITY_COMPUTE_KEY: quality.compute,
        _QUALITY_RUNTIME_KEY: quality.runtime,
        _QUALITY_LANGUAGE_KEY: quality.language or "",
    }


def quality_options_from_output_options(
    options: dict[str, Any] | None,
) -> NativeQualityOptions | None:
    """Return the recorded local quality settings, or ``None`` when quality is off.

    A run that recorded no quality, or an older snapshot written before quality was
    part of the output options, reports ``None`` and a resume performs no local
    verification. A present but malformed quality block fails closed instead of
    silently dropping the requested work.
    """
    if not isinstance(options, dict) or not options.get(_QUALITY_ENABLED_KEY):
        return None
    provider = options.get(_QUALITY_PROVIDER_KEY)
    model = options.get(_QUALITY_MODEL_KEY)
    device = options.get(_QUALITY_DEVICE_KEY)
    compute = options.get(_QUALITY_COMPUTE_KEY)
    runtime = options.get(_QUALITY_RUNTIME_KEY)
    language = options.get(_QUALITY_LANGUAGE_KEY)
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not isinstance(device, str)
        or not device
        or not isinstance(compute, str)
        or not compute
        or not isinstance(runtime, str)
        or not runtime
        or not isinstance(language, str)
    ):
        raise NativeGenerationError(
            "the recorded local quality settings are malformed; refusing to run a partial "
            "verification step.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_QUALITY_OPTIONS_INVALID,
        )
    return NativeQualityOptions(
        provider=provider,
        model=model or None,
        device=device,
        compute=compute,
        runtime=runtime,
        language=language or None,
    )


def _preflight_local_quality(options: NativeQualityOptions) -> None:
    """Fail closed before any paid submit when local quality cannot run offline.

    Only an installed local ASR provider is admitted. The probe inspects the
    registered provider's installed package and local model assets without
    downloading or calling a provider, so an unavailable dependency or an absent
    local model stops the command before the paid TTS POST instead of after the
    audio is already paid for.
    """
    if options.provider not in NATIVE_LOCAL_QUALITY_PROVIDERS:
        raise NativeGenerationError(
            "only an installed local ASR provider is admitted for native quality "
            "verification; cloud ASR providers stay on the legacy executor.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_QUALITY_PROVIDER_UNSUPPORTED,
        )
    from ..providers.asr_registry import ASRProviderNotFoundError, get_asr_provider_spec

    try:
        spec = get_asr_provider_spec(options.provider)
    except ASRProviderNotFoundError:
        raise NativeGenerationError(
            "the selected local quality provider is not a registered ASR provider.",
            code=_EXIT_ARGS,
            error_code=_ERROR_QUALITY_PROVIDER_UNSUPPORTED,
        ) from None
    health = spec.dependency_probe()
    if not health.available:
        raise NativeGenerationError(
            health.remediation or "the local quality ASR model is unavailable.",
            code=_EXIT_MISSING_DEP,
            error_code=_ERROR_QUALITY_MODEL_UNAVAILABLE,
        )


@dataclass(frozen=True)
class NativeLinkedQuality:
    """One verified linked local quality-verification run and its verdict."""

    run_uuid: str
    passed: bool


@dataclass(frozen=True)
class NativeLinkedTiming:
    """One verified linked local-timing run and its on-disk artifacts."""

    run_uuid: str
    timings_json: Path
    srt: Path
    segment_count: int | None


@dataclass(frozen=True)
class NativeGenerationSummary:
    """What the CLI turns into its success JSON or human output."""

    run_uuid: str
    revision: int
    files: dict[str, str]
    duration_ms: int
    segment_count: int | None
    cost_total: float | None
    cost_currency: str | None
    timing_requested: bool = False
    timing_complete: bool = False
    quality_requested: bool = False
    quality_complete: bool = False
    quality_passed: bool | None = None


class _ReadOnlyConnection:
    """Minimal handle exposing ``.connection`` to :class:`HistoryRepository`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


# ── ownership descriptor ──────────────────────────────────────────────────────


def _descriptor_path(run_root: Path) -> Path:
    return run_root / OWNERSHIP_FILE_NAME


def _read_descriptor(run_root: Path) -> str | None:
    """Return the committed run UUID the descriptor names, or fail closed.

    A missing descriptor is ``None``. A symlinked descriptor is refused before
    its target is read, including a dangling link, so a user-controlled alias can
    never redirect the ownership probe. A present descriptor that is not a
    bounded ownership document raises :class:`NativeGenerationError`, so a
    malformed or tampered marker never becomes a legacy run.
    """
    path = _descriptor_path(run_root)
    if path.is_symlink():
        raise NativeGenerationError(
            "native run ownership descriptor must not be a symlink; refusing to trust it.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise NativeGenerationError(
            "native run ownership descriptor is unreadable; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type",
        "ownership_version",
        "run_uuid",
    }:
        raise NativeGenerationError(
            "native run ownership descriptor is malformed; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    if (
        payload["artifact_type"] != OWNERSHIP_ARTIFACT_TYPE
        or payload["ownership_version"] != OWNERSHIP_VERSION
    ):
        raise NativeGenerationError(
            "native run ownership descriptor has an unsupported version; refusing legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    run_uuid = payload["run_uuid"]
    if not isinstance(run_uuid, str) or not run_uuid:
        raise NativeGenerationError(
            "native run ownership descriptor names no run; refusing to fall back to legacy JSON.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    return run_uuid


def _write_descriptor(run_root: Path, run_uuid: str) -> None:
    """Atomically write the tiny ownership descriptor for a committed native run.

    The temporary file is created uniquely and exclusively beside the descriptor
    with ``mkstemp``, flushed and fsynced, then renamed over it. A fixed
    ``*.json.tmp`` name is never used, so a pre-existing symlink at that name
    cannot redirect the write to an external file. Only this call's own temporary
    file is removed on failure.
    """
    payload = {
        "artifact_type": OWNERSHIP_ARTIFACT_TYPE,
        "ownership_version": OWNERSHIP_VERSION,
        "run_uuid": run_uuid,
    }
    path = _descriptor_path(run_root)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def _ensure_descriptor(run_root: Path, run_uuid: str) -> None:
    """Restore a missing descriptor for a committed run without rewriting one."""
    existing = _read_descriptor(run_root)
    if existing is None:
        _write_descriptor(run_root, run_uuid)
    elif existing != run_uuid:
        raise NativeGenerationError(
            "native run ownership descriptor names a different run than the committed history.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )


def _native_local_evidence(run_root: Path) -> bool:
    """Whether the run root carries any marker the legacy writer must not own.

    A descriptor, a compatibility ``run_state.json`` that carries the native
    history marker, or a ``raw/`` directory with at least one paid receipt are all
    evidence that this directory belongs to canonical history. When the database
    cannot confirm that, the caller fails closed instead of letting a legacy
    writer mutate the directory.
    """
    if _descriptor_path(run_root).exists():
        return True
    state_path = run_root / "run_state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            payload = None
        if isinstance(payload, dict) and "native_history" in payload:
            return True
    raw_dir = run_root / "raw"
    if raw_dir.is_dir() and any(raw_dir.glob("*.receipt.json")):
        return True
    return False


def _find_native_run(
    canonical_root: str, database_path: Path | None
) -> tuple[RunRecord | None, bool]:
    """Return the committed native run for a root and whether the database was usable.

    The read is read-only and writes nothing: an absent database means no native
    ownership, while an unreadable, foreign, or newer database reports itself
    unusable so the caller can fail closed when native evidence exists. The reader
    is WAL-consistent, so a native row another live writer just committed is not
    missed as a stale main-file snapshot. Only a run carrying the native snapshot
    origin counts; an imported run that happens to share the directory does not.
    """
    try:
        db_path = history_database_path() if database_path is None else Path(database_path)
    except HistoryPathsError:
        return None, False
    if not db_path.exists():
        return None, True
    connection: sqlite3.Connection | None = None
    try:
        connection = connect_readonly_consistent(db_path)
        repository = HistoryRepository(cast(HistoryDatabase, _ReadOnlyConnection(connection)))
        rows = repository.find_runs_by_root(canonical_root, limit=MAX_QUERY_LIMIT)
    except (HistoryDatabaseError, sqlite3.Error, OSError):
        return None, False
    finally:
        if connection is not None:
            connection.close()
    native = [
        run
        for run in rows
        if run.legacy_source_root is None
        and isinstance(run.config_snapshot, dict)
        and run.config_snapshot.get("operation_origin") == NATIVE_SNAPSHOT_ORIGIN
    ]
    if len(native) > 1:
        # Two native runs on one root cannot be told apart; refuse instead of
        # choosing one.
        return None, False
    return (native[0] if native else None), True


def _fresh_run_root_available(run_root: Path) -> bool:
    """Whether a fresh native run may claim this root without another writer's state.

    The early ownership resolution already routes a native-owned root away from a
    fresh start; this is the re-check under the run lock. If a concurrent legacy
    or native writer created local state between the early decision and this
    point, the root is refused instead of letting a second writer interleave one
    directory. Any descriptor, native run state, or paid raw receipt, and any
    legacy ``run_state.json``, counts as another writer's state.
    """
    return not _native_local_evidence(run_root) and not (run_root / "run_state.json").exists()


def resolve_native_ownership(
    output_root: Path, *, database_path: Path | None = None
) -> NativeOwnership:
    """Resolve one run root's route without writing anything.

    A committed native run row, or failing that native local evidence, makes the
    root native-owned. Missing or unreadable database state next to native
    evidence fails closed with a stable error code. Every other root reports
    ``legacy``, which keeps existing legacy runs and unswitched configurations on
    the exact behavior they had.
    """
    canonical_root = str(Path(output_root).resolve())
    descriptor_uuid = _read_descriptor(output_root)
    run, database_usable = _find_native_run(canonical_root, database_path)
    if run is not None:
        return NativeOwnership(
            route=_ROUTE_NATIVE_EXISTING, run_uuid=run.run_uuid, revision=run.revision
        )
    if not database_usable:
        if descriptor_uuid is not None or _native_local_evidence(output_root):
            return NativeOwnership(
                route=_ROUTE_BLOCKED,
                error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
                reason=(
                    "this run directory carries native history evidence but the history "
                    "database could not be read; refusing to treat it as a legacy JSON run. "
                    "Restore the history database or choose a different --run-id."
                ),
            )
        return NativeOwnership(route=_ROUTE_LEGACY)
    if descriptor_uuid is not None or _native_local_evidence(output_root):
        return NativeOwnership(
            route=_ROUTE_BLOCKED,
            error_code=_ERROR_OWNERSHIP_RUN_MISSING,
            reason=(
                "this run directory is owned by native history, but its committed history run "
                "no longer exists; refusing to treat it as a legacy JSON run or to recreate "
                "history silently. Choose a different --run-id."
            ),
        )
    return NativeOwnership(route=_ROUTE_LEGACY)


def build_output_options(
    no_trim: bool,
    *,
    timing: NativeTimingOptions | None = None,
    quality: NativeQualityOptions | None = None,
) -> dict[str, Any]:
    """Return the fixed output-processing settings the native route records.

    The recorded ``trim_final_silence`` is the exact semantics this run uses, so a
    later resume proves it did not change instead of inferring it from the current
    command line: the default trims each converted part while ``--no-trim`` keeps
    the provider's final silence. When the run requested local timings or local
    quality verification, those effective settings are recorded alongside so a
    resume proves they did not change either.
    """
    options: dict[str, Any] = {
        "trim_final_silence": not no_trim,
        "processing_version": OUTPUT_PROCESSING_VERSION,
    }
    options.update(_timing_output_fields(timing))
    options.update(_quality_output_fields(quality))
    return options


# ── database helpers ──────────────────────────────────────────────────────────


@contextmanager
def _open_writable_history(database_path: Path | None) -> Iterator[HistoryRepository]:
    """Open the private history database for writing, migrating it if needed."""
    try:
        path = history_database_path() if database_path is None else Path(database_path)
    except HistoryPathsError as exc:
        raise NativeGenerationError(
            "the history home is not a usable absolute directory.",
            code=_EXIT_ARGS,
            error_code="NATIVE_HISTORY_HOME_INVALID",
        ) from exc
    try:
        database = HistoryDatabase(path)
        database.connect()
        database.migrate()
    except (HistoryDatabaseError, sqlite3.Error, OSError) as exc:
        raise NativeGenerationError(
            "the local history database could not be opened for this run.",
            code=_EXIT_OUTPUT,
            error_code=_ERROR_HISTORY_UNAVAILABLE,
        ) from exc
    try:
        yield HistoryRepository(database)
    finally:
        database.close()


# ── part evidence ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _PartEvidence:
    """One part's committed attempt and artifacts, read from a verified view."""

    part: NativeTtsPart
    attempt: AttemptRecord | None
    raw_artifact: ArtifactRecord | None
    local_raw_artifact: ArtifactRecord | None
    chunk_artifact: ArtifactRecord | None


def _collect_evidence(view: NativeTtsView) -> list[_PartEvidence]:
    evidence: list[_PartEvidence] = []
    for part in view.parts:
        part_uuid = part.record.part_uuid
        # Only a paid ``tts_chunk`` attempt is paid evidence. A local route records
        # its own ``local_tts_chunk`` attempt per real local invocation, and a part
        # may legitimately carry several after local retries, so those rows never
        # count here and are never mistaken for a paid reservation.
        attempts = [
            attempt
            for attempt in view.attempts
            if attempt.part_uuid == part_uuid and attempt.call_type == ATTEMPT_CALL_TYPE_TTS_CHUNK
        ]
        raws = [
            artifact
            for artifact in view.artifacts
            if artifact.part_uuid == part_uuid and artifact.role == ARTIFACT_ROLE_PAID_RAW_AUDIO
        ]
        local_raws = [
            artifact
            for artifact in view.artifacts
            if artifact.part_uuid == part_uuid and artifact.role == ARTIFACT_ROLE_LOCAL_RAW_AUDIO
        ]
        chunks = [
            artifact
            for artifact in view.artifacts
            if artifact.part_uuid == part_uuid and artifact.role == ARTIFACT_ROLE_CHUNK_AUDIO
        ]
        if len(attempts) > 1 or len(raws) > 1 or len(local_raws) > 1 or len(chunks) > 1:
            # A part with more than one attempt or artifact cannot be told apart, so
            # the executor must not silently pick one and continue to a paid call.
            raise NativeGenerationError(
                "refusing to continue: a committed part carries duplicate attempts or "
                "artifacts, so its history cannot be reconstructed.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        evidence.append(
            _PartEvidence(
                part=part,
                attempt=attempts[0] if attempts else None,
                raw_artifact=raws[0] if raws else None,
                local_raw_artifact=local_raws[0] if local_raws else None,
                chunk_artifact=chunks[0] if chunks else None,
            )
        )
    return evidence


def _has_paid_evidence(evidence: _PartEvidence) -> bool:
    return (
        evidence.attempt is not None
        or evidence.raw_artifact is not None
        or evidence.chunk_artifact is not None
    )


def _has_any_evidence(evidence: _PartEvidence) -> bool:
    """Whether a part carries any committed paid or local evidence.

    Used only for the contiguous-prefix invariant, so a local raw or converted
    chunk is treated as committed work exactly like a paid one.
    """
    return _has_paid_evidence(evidence) or evidence.local_raw_artifact is not None


def _verified_chunk_file(
    evidence: _PartEvidence, run_root: Path, sha256_file: Callable[[Path], str]
) -> Path | None:
    """Return the verified converted chunk path, or ``None`` when it is not usable.

    A committed converted artifact is skipped only when its file still exists and
    its size and digest match the committed evidence. Any mismatch returns
    ``None`` so the caller re-derives from retained raw bytes instead of trusting
    a foreign file.
    """
    artifact = evidence.chunk_artifact
    if artifact is None or artifact.sha256 is None or artifact.size_bytes is None:
        return None
    path = run_root / artifact.path
    if not path.is_file():
        return None
    try:
        if path.stat().st_size != artifact.size_bytes:
            return None
        if sha256_file(path) != artifact.sha256:
            return None
    except OSError:
        return None
    return path


def _verified_raw_bytes(
    raw: ArtifactRecord, run_root: Path, sha256_file: Callable[[Path], str]
) -> bytes:
    """Return retained raw paid bytes only when they still match their artifact.

    A crash between writing the raw bytes and committing the database row, or any
    later tampering, is caught here: the file must exist and match the committed
    size and digest, or the caller fails closed rather than converting foreign
    bytes.
    """
    if raw.sha256 is None or raw.size_bytes is None:
        raise NativeGenerationError(
            "the retained paid raw artifact is missing its digest or size.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        )
    path = run_root / raw.path
    if not path.is_file():
        raise NativeGenerationError(
            "the retained paid raw audio is missing; refusing to regenerate paid work.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        )
    try:
        if path.stat().st_size != raw.size_bytes or sha256_file(path) != raw.sha256:
            raise NativeGenerationError(
                "the retained paid raw audio does not match its committed evidence.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        return path.read_bytes()
    except OSError as exc:
        raise NativeGenerationError(
            "the retained paid raw audio could not be read.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_RAWS_EVIDENCE_INVALID,
        ) from exc


# ── per-turn dialogue quality evidence ──────────────────────────────────────────


@dataclass(frozen=True)
class _TurnQuality:
    """One committed per-turn dialogue verdict read back from the verified view.

    ``passed`` is the recorded PASS/FAIL the run must re-report, and
    ``transcript`` is the private observed ASR text used only to re-verify the
    stored hash. Neither value is ever projected into a public receipt or export.
    """

    passed: bool
    transcript: str


def _turn_quality_evidence(view: NativeTtsView) -> dict[str, _TurnQuality]:
    """Return each part's committed per-turn dialogue verdict, keyed by part UUID.

    The verdict is the pairing of one ``tts_turn_quality_receipt`` artifact and one
    private ``verification_transcript`` source that names the same part, so a
    half-written pair is treated as absent instead of trusted. A duplicate receipt
    or transcript per part, a missing or non-boolean verdict, and a transcript
    whose stored hash no longer matches its text all fail closed rather than
    silently dropping a recorded failure. A part without committed evidence is
    simply absent from the mapping, which the caller reads as pending work.
    """
    receipts: dict[str, ArtifactRecord] = {}
    for artifact in view.artifacts:
        if artifact.role != ARTIFACT_ROLE_TTS_TURN_QUALITY:
            continue
        if artifact.part_uuid is None or artifact.part_uuid in receipts:
            raise NativeGenerationError(
                "refusing to continue: the dialogue turn quality evidence is not unique per part.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        receipts[artifact.part_uuid] = artifact
    transcripts: dict[str, str] = {}
    for source in view.text_sources:
        if source.kind != TEXT_KIND_VERIFICATION_TRANSCRIPT or source.part_uuid is None:
            continue
        if source.part_uuid in transcripts:
            raise NativeGenerationError(
                "refusing to continue: the dialogue turn verification transcript is not unique "
                "per part.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        content = source.content
        if not isinstance(content, str) or source.content_hash != sha256_text(content):
            raise NativeGenerationError(
                "refusing to continue: a committed dialogue turn verification transcript does "
                "not match its stored hash.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        transcripts[source.part_uuid] = content
    evidence: dict[str, _TurnQuality] = {}
    for part in view.parts:
        part_uuid = part.record.part_uuid
        receipt = receipts.get(part_uuid)
        transcript = transcripts.get(part_uuid)
        if receipt is None or transcript is None:
            continue
        metadata = receipt.media_metadata if isinstance(receipt.media_metadata, dict) else {}
        passed = metadata.get("quality_passed")
        if not isinstance(passed, bool):
            raise NativeGenerationError(
                "refusing to continue: a committed dialogue turn quality receipt carries no "
                "boolean verdict.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        evidence[part_uuid] = _TurnQuality(passed=passed, transcript=transcript)
    return evidence


# ── executor ──────────────────────────────────────────────────────────────────


def _sync_observed_amount(result: SynthesisResult) -> str | float | None:
    """Return the exact cost a synchronous submit reported, or ``None`` when unknown.

    Only a ``polza-tts`` ``/audio/speech`` response carries usage, and only its
    recognized cost fields are read; ``openrouter-tts`` returns its audio inline
    with no synchronous usage, so the amount stays unknown rather than being
    fetched with a new remote GET or invented from a number. The exact decimal
    text is preferred over its binary float so the stored billing fact stays exact
    where the provider supplied one.
    """
    metadata = result.raw_metadata if isinstance(result.raw_metadata, dict) else {}
    cost, cost_exact = costs.media_observed_cost(metadata.get("usage_direct"))
    if cost is None:
        return None
    return cost_exact if cost_exact is not None else cost


def _new_staging_path(target: Path) -> Path:
    """Return a fresh private staging path beside ``target`` for one output file.

    The file lives in the output's own directory so publishing is an atomic
    same-filesystem rename, and ``mkstemp`` creates it exclusively with private
    mode, so a crash can neither reuse a foreign file nor leave a world-readable
    partial artifact. The original suffix is preserved because the media layer
    selects its codec and bitrate from it.
    """
    descriptor, name = tempfile.mkstemp(
        prefix=".vo-native-", suffix=target.suffix, dir=str(target.parent)
    )
    os.close(descriptor)
    os.chmod(name, 0o600)
    return Path(name)


def _publish_staged(
    staged: Path,
    target: Path,
    *,
    error_code: str,
    size_bytes: int,
    sha256: str,
    sha256_file: Callable[[Path], str],
) -> None:
    """Publish once without replacing an existing output or following an alias.

    A byte-identical file already published just before a crash can be adopted
    without clobbering it; conflicting or symlinked files remain untouched. The
    hard link is an exclusive, same-filesystem publication of the private staging
    file, unlike a rename that silently replaces an existing paid artifact.
    """
    if target.parent.is_symlink() or target.is_symlink() or staged.is_symlink():
        raise NativeGenerationError(
            "refusing to publish audio through a symlink.",
            code=_EXIT_OUTPUT,
            error_code=error_code,
        )
    try:
        os.link(staged, target)
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            raise NativeGenerationError(
                "refusing to replace an existing non-regular audio output.",
                code=_EXIT_OUTPUT,
                error_code=error_code,
            ) from None
        try:
            same_bytes = target.stat().st_size == size_bytes and sha256_file(target) == sha256
        except OSError:
            same_bytes = False
        if not same_bytes:
            raise NativeGenerationError(
                "refusing to replace an existing conflicting audio output.",
                code=_EXIT_OUTPUT,
                error_code=error_code,
            ) from None
    staged.unlink()


class _Executor:
    """One locked native run's committed state, provider factory, and media hooks."""

    def __init__(
        self,
        *,
        repository: HistoryRepository,
        view: NativeTtsView,
        paths,
        run_root: Path,
        prepared: PreparedRun,
        chunks: list[ScriptChunk],
        script_format: str,
        script_path: Path,
        output_options: dict[str, Any],
        ffmpeg_path: str,
        ffprobe_path: str,
        provider_factory: Callable[[], Any],
        hooks: NativeExecutionHooks,
        logger: GenerationLogger,
        resume: bool,
        paid_submit_allowed: bool = True,
        local_timing_allowed: bool = True,
        local_quality_allowed: bool = True,
    ) -> None:
        self.repository = repository
        self.view = view
        self.paths = paths
        self.run_root = run_root
        self.prepared = prepared
        self.chunks = chunks
        self.script_format = script_format
        self.dialogue = is_dialogue_format(script_format)
        # The admitted local routes have no paid POST and no paid-marker guard: the
        # ``omnivoice-local`` preset dialogue and every ``qwen-local`` local route
        # (clone and the instructed preset/design modes). Each
        # reserves its own cost-free ``local_tts_chunk`` attempt per real invocation.
        self.local = prepared.provider in {"omnivoice-local", "qwen-local"}
        self.dialogue_quality_gate = self.dialogue and prepared.provider == "openrouter-tts"
        self.script_path = script_path
        self.output_options = output_options
        self.timing_options = timing_options_from_output_options(output_options)
        self.quality_options = quality_options_from_output_options(output_options)
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.provider_factory = provider_factory
        self.hooks = hooks
        self.logger = logger
        self.resume = resume
        self.paid_submit_allowed = paid_submit_allowed
        self.local_timing_allowed = local_timing_allowed
        self.local_quality_allowed = local_quality_allowed
        # The recorded semantics, never the current command line: a run that
        # recorded ``trim_final_silence`` false keeps every converted part's
        # provider-issued silence, and a resume proved that setting unchanged
        # before reaching this executor.
        self.trim_final_silence = bool(output_options.get("trim_final_silence", True))
        self._revision = view.run.revision
        self._provider: Any = None

    # -- provider laziness ----------------------------------------------------

    def _provider_instance(self) -> Any:
        """Build the provider on first real need; never for a local/export path."""
        if self._provider is None:
            self._provider = self.provider_factory()
        return self._provider

    def _run_uuid(self) -> str:
        return self.view.run.run_uuid

    def _model(self) -> str:
        return self.prepared.model

    def _uses_media_route(self) -> bool:
        """Whether this run submits to the async Polza ``elevenlabs/`` ``/media`` route.

        A synchronous ``polza-tts`` model (``/audio/speech``) or an
        ``openrouter-tts`` submit returns its audio inline and never stores a
        remote task id, so it must not bind the media callbacks or ever reach the
        GET-only media recovery path.
        """
        return self.prepared.provider == "polza-tts" and polza_media_route_model(
            self.prepared.model
        )

    # -- per-part processing --------------------------------------------------

    def _bind_media_callbacks(
        self,
        provider: Any,
        *,
        attempt: AttemptRecord,
        part: NativeTtsPart,
        accepted_id: dict[str, str],
    ) -> None:
        """Bind DB task-id and observed-cost transitions onto the provider callbacks.

        The accepted id is committed before the first poll or download, and the
        exact observed cost before the signed-URL download, exactly as the paid
        reservation order requires. A failure here propagates out of the provider
        call so the outcome stays unconfirmed and no second POST is ever sent.
        """
        run_uuid = self._run_uuid()
        part_uuid = part.record.part_uuid
        attempt_uuid = attempt.attempt_uuid

        def on_accepted(task_id: str) -> None:
            run, _updated = self.repository.record_polza_media_task_accepted(
                run_uuid,
                attempt_uuid=attempt_uuid,
                part_uuid=part_uuid,
                expected_revision=self._revision,
                remote_task_id=task_id,
            )
            self._revision = run.revision
            accepted_id["id"] = task_id
            self.logger.event("info", "remote_accepted", chunk=part.number, id=part.chunk_id)

        def on_completed(_task_id: str, usage: dict | None, _generation_id: str | None) -> None:
            cost, cost_exact = costs.media_observed_cost(usage)
            if cost is None:
                return
            amount: Decimal | float | str = cost_exact if cost_exact is not None else cost
            run, _updated = self.repository.record_polza_media_observed_cost(
                run_uuid,
                attempt_uuid=attempt_uuid,
                part_uuid=part_uuid,
                expected_revision=self._revision,
                amount=amount,
            )
            self._revision = run.revision
            self.logger.event("info", "cost_observed", chunk=part.number, id=part.chunk_id)

        provider.on_media_task_accepted = on_accepted
        provider.on_media_completed = on_completed

    def _convert_and_commit(
        self, evidence: _PartEvidence, result: SynthesisResult, attempt: AttemptRecord
    ) -> None:
        """Convert one accepted audio result into the canonical chunk artifact.

        The file is written and trimmed with the recorded semantics, hashed, and
        then committed with the digest, size, and bounded processing metadata in
        one short transaction. A conversion failure propagates as an output error
        while the raw and accepted-id evidence stay committed, so a later resume
        rebuilds locally instead of re-submitting.
        """
        part = evidence.part
        output_path = self.paths.chunks_dir / f"{part.chunk_id}.mp3"
        staged = _new_staging_path(output_path)
        try:
            self.hooks.write_audio_as_mp3(
                self.ffmpeg_path, result.audio_bytes, result.audio_format, staged
            )
            if self.trim_final_silence:
                self.hooks.trim_final_silence(self.ffmpeg_path, self.ffprobe_path, staged)
            duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, staged)
            sha256 = self.hooks.sha256_file(staged)
            size_bytes = staged.stat().st_size
            _publish_staged(
                staged,
                output_path,
                error_code=_ERROR_CONVERSION_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            self.logger.event("error", "chunk_conversion_failed", chunk=part.number)
            raise NativeGenerationError(
                f"Failed to write chunk audio {output_path.name}: {exc}",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_CONVERSION_FAILED,
            ) from exc
        metadata = {
            "duration_ms": duration_ms,
            "generation_id": result.generation_id,
            "processing_version": OUTPUT_PROCESSING_VERSION,
        }
        try:
            run, _artifact = self.repository.record_tts_part_completed(
                self._run_uuid(),
                part_uuid=part.record.part_uuid,
                attempt_uuid=attempt.attempt_uuid,
                expected_revision=self._revision,
                path=f"chunks/{part.chunk_id}.mp3",
                mime="audio/mpeg",
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata=metadata,
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this part conflicts with the converted chunk; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        self.logger.event("info", "chunk_state_saved", chunk=part.number, id=part.chunk_id)
        if self.dialogue_quality_gate:
            # The required per-turn local quality gate runs on this just-committed
            # turn audio, before any later turn's paid submit and before concat.
            self._run_turn_quality(evidence)

    # -- local synthesis (admitted omnivoice-local preset dialogue) -------------

    def _local_raw_path(self, part: NativeTtsPart, audio_format: str) -> str:
        """Return the managed-relative raw path for one local turn's bytes."""
        extension = "wav" if audio_format == "wav" else audio_format
        return f"raw/{part.chunk_id}.{extension}"

    def _link_local_raw(
        self, part: NativeTtsPart, result: SynthesisResult, attempt_uuid: str
    ) -> ArtifactRecord:
        """Persist and link locally synthesized raw bytes before any conversion.

        The bytes are written to their managed ``raw/<id>.<ext>`` path and linked as
        the one :data:`ARTIFACT_ROLE_LOCAL_RAW_AUDIO` artifact of this invocation's
        ``attempt_uuid``, so a later FFmpeg crash can be rebuilt from these bytes
        with no second local model run. A conflicting existing file (a different
        synthesis for the same turn) fails closed instead of being silently
        replaced.
        """
        relative = self._local_raw_path(part, result.audio_format)
        target = self.run_root / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise NativeGenerationError(
                "the native run raw directory could not be created.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_OUTPUT_UNAVAILABLE,
            ) from exc
        staged = _new_staging_path(target)
        try:
            staged.write_bytes(result.audio_bytes)
            size_bytes = staged.stat().st_size
            sha256 = self.hooks.sha256_file(staged)
            _publish_staged(
                staged,
                target,
                error_code=_ERROR_LOCAL_RAW_EVIDENCE_INVALID,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except OSError as exc:
            staged.unlink(missing_ok=True)
            raise NativeGenerationError(
                "the locally synthesized raw audio could not be stored.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_LOCAL_RAW_EVIDENCE_INVALID,
            ) from exc
        mime = "audio/wav" if result.audio_format == "wav" else f"audio/{result.audio_format}"
        try:
            run, artifact = self.repository.record_local_tts_raw_saved(
                self._run_uuid(),
                attempt_uuid=attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                path=relative,
                mime=mime,
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata={
                    "format": result.audio_format,
                    "chunk_id": part.chunk_id,
                    "number": part.number,
                },
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this part conflicts with its local raw audio; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        return artifact

    def _submit_local_part(self, evidence: _PartEvidence) -> None:
        """Synthesize one part locally, link its raw bytes, then convert it.

        A local route has no paid reservation and no network, so this reserves one
        durable ``local_tts_chunk`` attempt before calling the local model, stores
        the accepted bytes linked to that attempt, and converts. An unconfirmed
        local attempt is safe to repeat on an explicit resume, so a failed or
        interrupted invocation never blocks a new one, but its outcome is recorded
        (``local_failed``, or a pending row when the process was interrupted) so
        ``history costs`` counts every real local invocation.
        """
        part = evidence.part
        provider = self._provider_instance()
        run, attempt = self.repository.reserve_local_tts_attempt(
            self._run_uuid(),
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            provider=self.prepared.provider,
            model=self._model(),
        )
        self._revision = run.revision
        self.logger.event("info", "local_synthesis_started", chunk=part.number, id=part.chunk_id)
        try:
            result = self._invoke_provider(provider, part)
        except Exception:
            self.logger.event("error", "local_synthesis_failed", chunk=part.number)
            self._record_local_attempt_failure(part, attempt)
            raise NativeGenerationError(
                f"Failed to synthesize {part.chunk_id} locally.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_SYNTHESIS_FAILED,
            ) from None
        self._link_local_raw(part, result, attempt.attempt_uuid)
        self._convert_local(evidence, result, attempt.attempt_uuid)

    def _record_local_attempt_failure(self, part: NativeTtsPart, attempt: AttemptRecord) -> None:
        """Record a failed local invocation's outcome without masking the failure.

        The synthesis error is the caller's primary outcome; a persistence failure
        while labelling the attempt is logged and swallowed so the same fixed
        synthesis error still reaches the user. A pending attempt a crash left
        behind needs no write here.
        """
        try:
            run, _updated = self.repository.record_local_tts_attempt_failed(
                self._run_uuid(),
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                error=_ERROR_LOCAL_SYNTHESIS_FAILED,
            )
        except HistoryRepositoryError:
            self.logger.event("warning", "local_attempt_outcome_unsaved", chunk=part.number)
            return
        self._revision = run.revision

    def _verified_local_raw_bytes(self, raw: ArtifactRecord) -> bytes:
        """Return linked local raw bytes only while they still match their row."""
        return _verified_raw_bytes(raw, self.run_root, self.hooks.sha256_file)

    def _convert_local_from_raw(self, evidence: _PartEvidence) -> None:
        """Rebuild one local part from its linked raw bytes with no model run."""
        raw = evidence.local_raw_artifact
        assert raw is not None  # narrowed by the caller
        attempt_uuid = raw.attempt_uuid
        if attempt_uuid is None:
            raise NativeGenerationError(
                "the linked local raw artifact carries no local attempt.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_RAW_EVIDENCE_INVALID,
            )
        metadata = raw.media_metadata if isinstance(raw.media_metadata, dict) else {}
        audio_format = metadata.get("format")
        if not isinstance(audio_format, str) or not audio_format:
            raise NativeGenerationError(
                "the linked local raw artifact is missing its audio format.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_RAW_EVIDENCE_INVALID,
            )
        audio_bytes = self._verified_local_raw_bytes(raw)
        result = SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            transcript=evidence.part.text,
            client_path="local",
        )
        self.logger.event(
            "info", "local_raw_recovery", chunk=evidence.part.number, id=evidence.part.chunk_id
        )
        self._convert_local(evidence, result, attempt_uuid)

    def _convert_local(
        self, evidence: _PartEvidence, result: SynthesisResult, attempt_uuid: str
    ) -> None:
        """Convert one locally synthesized turn into its canonical chunk artifact."""
        part = evidence.part
        output_path = self.paths.chunks_dir / f"{part.chunk_id}.mp3"
        staged = _new_staging_path(output_path)
        try:
            self.hooks.write_audio_as_mp3(
                self.ffmpeg_path, result.audio_bytes, result.audio_format, staged
            )
            if self.trim_final_silence:
                self.hooks.trim_final_silence(self.ffmpeg_path, self.ffprobe_path, staged)
            duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, staged)
            sha256 = self.hooks.sha256_file(staged)
            size_bytes = staged.stat().st_size
            _publish_staged(
                staged,
                output_path,
                error_code=_ERROR_CONVERSION_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            self.logger.event("error", "chunk_conversion_failed", chunk=part.number)
            raise NativeGenerationError(
                f"Failed to write chunk audio {output_path.name}: {exc}",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_CONVERSION_FAILED,
            ) from exc
        metadata = {
            "duration_ms": duration_ms,
            "generation_id": result.generation_id,
            "processing_version": OUTPUT_PROCESSING_VERSION,
        }
        try:
            run, _artifact = self.repository.record_local_tts_part_completed(
                self._run_uuid(),
                attempt_uuid=attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                path=f"chunks/{part.chunk_id}.mp3",
                mime="audio/mpeg",
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata=metadata,
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this part conflicts with the converted chunk; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        self.logger.event("info", "chunk_state_saved", chunk=part.number, id=part.chunk_id)

    def _has_pending_local_synthesis(self) -> bool:
        """Whether any local part still needs a real local model invocation.

        A part with a verified converted chunk, or a part whose raw bytes are
        already linked (so it only needs a local FFmpeg rebuild), needs no model.
        """
        for item in _collect_evidence(self.view):
            if item.local_raw_artifact is not None:
                continue
            if _verified_chunk_file(item, self.run_root, self.hooks.sha256_file) is not None:
                continue
            return True
        return False

    def preflight_local_synthesis(self) -> None:
        """Fail closed before the first local model call when its inputs are gone.

        Any admitted ``qwen-local`` local route needs this: the executor proves the
        committed identity's runtime still matches the environment and the installed
        runtime plus its cached model are available without a download before any
        local model runs. The clone route additionally proves its reference sample
        still matches the committed digest and size. It runs only when a real local
        invocation is pending and only when this execution may start new synthesis
        (``generate`` and ``history resume``; never ``history sync``), so a
        completed-run export repair and a local raw rebuild neither read a reference
        nor probe the runtime.
        """
        if self.prepared.provider != "qwen-local":
            return
        if not self.paid_submit_allowed or not self._has_pending_local_synthesis():
            return
        self._preflight_qwen_local_identity()

    def _preflight_qwen_local_identity(self) -> None:
        """Verify the committed local Qwen identity and runtime before any model.

        The committed run carries exactly one local-Qwen identity block: the clone
        block (whose external reference sample is re-checked) or the instructed-mode
        block. The block's runtime must still match the environment and the installed
        runtime plus its cached model must be available offline; either failure stops
        the run before the local model, with no implicit download.
        """
        clone_identity = self.prepared.qwen_clone_identity
        mode_identity = self.prepared.qwen_mode_identity
        if isinstance(clone_identity, QwenCloneVoiceIdentity):
            runtime = clone_identity.runtime
            model = clone_identity.model
            self._preflight_qwen_clone_reference(clone_identity)
        elif isinstance(mode_identity, QwenModeVoiceIdentity):
            runtime = mode_identity.runtime
            model = mode_identity.model
        else:
            raise NativeGenerationError(
                "this local Qwen run records no mode identity; refusing to run the local model.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_REFERENCE_UNAVAILABLE,
            )
        current_runtime = os.environ.get("VOICEOVER_QWEN_TTS_RUNTIME", "python").strip()
        if current_runtime != runtime:
            raise NativeGenerationError(
                "the local Qwen runtime changed since this run was prepared; refusing to "
                "run the local model.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_REFERENCE_UNAVAILABLE,
            )
        from ..providers.qwen_local import qwen_local_tts_availability

        availability = qwen_local_tts_availability(runtime, model)
        if not availability.available:
            raise NativeGenerationError(
                availability.remediation or "the local qwen-local runtime or model is unavailable.",
                code=_EXIT_MISSING_DEP,
                error_code=_ERROR_LOCAL_MODEL_UNAVAILABLE,
            )

    def _preflight_qwen_clone_reference(self, identity: QwenCloneVoiceIdentity) -> None:
        """Verify the committed clone reference still matches before any local model."""
        sample = Path(identity.sample_path)
        try:
            unchanged = (
                sample.is_file()
                and sample.stat().st_size == identity.sample_size
                and self.hooks.sha256_file(sample) == identity.sample_sha256
            )
        except OSError:
            unchanged = False
        if not unchanged:
            raise NativeGenerationError(
                "the local clone reference audio is missing or changed; refusing to run the "
                "local model.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_LOCAL_REFERENCE_UNAVAILABLE,
            )

    def _run_local_parts(self, evidence: list[_PartEvidence]) -> None:
        """Execute or recover every local part, stopping before any inconsistent gap.

        A part with no verified converted chunk is processed exactly once: a part
        with linked local raw audio is converted from those bytes with no model run,
        and a part without any evidence is synthesized locally. A local attempt is
        safe to repeat, so no paid-marker guard applies; ``history sync`` may not run
        a local model and fails closed on an uncommitted part instead. Paid evidence
        on a local run is a mixed-route conflict and always fails closed.
        """
        first_incomplete_done = False
        for index, item in enumerate(evidence):
            if _verified_chunk_file(item, self.run_root, self.hooks.sha256_file) is not None:
                self.hooks.progress(
                    f"Skipping {item.part.chunk_id}/{len(evidence):02d}: already committed"
                )
                continue
            if any(_has_any_evidence(later) for later in evidence[index + 1 :]):
                raise NativeGenerationError(
                    "refusing to resume: a later part carries committed evidence before this "
                    "part completed.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            if first_incomplete_done and _has_any_evidence(item):
                raise NativeGenerationError(
                    "refusing to resume: committed evidence does not form a contiguous prefix.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            if item.attempt is not None or item.raw_artifact is not None:
                raise NativeGenerationError(
                    "refusing to continue: a local run carries paid evidence on one of its parts.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            self.hooks.progress(
                f"Generating {item.part.chunk_id}/{len(evidence):02d}: {item.part.chunk_id}.mp3"
            )
            if item.local_raw_artifact is not None:
                self._convert_local_from_raw(item)
            elif self.paid_submit_allowed:
                self._submit_local_part(item)
            else:
                raise NativeGenerationError(
                    "refusing to sync: an unattempted local part remains and history sync "
                    "never runs a local model. Use history resume to synthesize it.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_SYNC_LOCAL_SYNTHESIS_REQUIRED,
                )
            first_incomplete_done = True

    # -- per-turn dialogue quality gate ---------------------------------------

    def _run_turn_quality(self, evidence: _PartEvidence) -> None:
        """Run the required local ASR quality check for one converted dialogue turn.

        The check reuses the existing installed local ASR adapter and the existing
        strict dialogue thresholds -- the same comparison the legacy gate applies
        to every turn -- so no new model or threshold is invented. The observed
        transcript and the content-free PASS/FAIL verdict are persisted together,
        linked to the exact turn part, before any outcome is reported. A mismatch
        keeps the paid raw bytes, converted audio, and cost intact and raises the
        existing quality exit, while a local transcription failure raises the
        fixed partial error; neither path ever repeats the paid submit.
        """
        options = self.quality_options
        if options is None:
            # The dialogue route always records its local quality settings; a run
            # without them cannot be verified and must not continue to a paid call.
            raise NativeGenerationError(
                "the dialogue native route requires an installed local --tts-quality-provider.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_QUALITY_OPTIONS_INVALID,
            )
        part = evidence.part
        audio_path = self.paths.chunks_dir / f"{part.chunk_id}.mp3"
        from .transcription import transcribe_local_asr_quality

        try:
            result = transcribe_local_asr_quality(
                provider_id=options.provider,
                audio_path=audio_path,
                model=options.model,
                device=options.device,
                compute=options.compute,
                runtime=options.runtime,
                language=options.language,
            )
        except Exception as exc:
            self.logger.event(
                "error", "dialogue_quality_transcription_failed", error=type(exc).__name__
            )
            raise NativeGenerationError(
                "the turn audio is complete, but the local per-turn quality transcription failed.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_QUALITY_ASR_FAILED,
            ) from None
        quality = evaluate_tts_transcript(
            expected_text=part.text,
            actual_transcript=result.transcript,
            minimum_similarity=1.0,
            maximum_missing_ratio=0.0,
            maximum_unexpected_ratio=0.0,
            maximum_repeated_ngram_excess=0,
            strip_audio_tags=True,
        )
        self._persist_turn_quality(part, result, quality, audio_path)
        if not quality.passed:
            self.logger.event("warning", "dialogue_turn_quality_failed", chunk=part.number)
            raise NativeGenerationError(
                "the turn audio is complete, but the local per-turn quality verification did "
                "not pass; the paid audio and cost were kept and the failed verification was "
                "recorded.",
                code=_EXIT_QUALITY,
                error_code=_ERROR_QUALITY_FAILED,
            )
        self.logger.event("info", "dialogue_turn_quality_verified", chunk=part.number)

    def _persist_turn_quality(
        self,
        part: NativeTtsPart,
        result: ASRResult,
        quality: TTSQualityResult,
        audio_path: Path,
    ) -> None:
        """Persist one turn's verdict and private transcript, linked to its part.

        The content-free receipt artifact and the private transcript source commit
        in one transaction, so a crash can never leave a verdict without its
        transcript or the reverse. The receipt references the managed converted
        turn audio (not the raw paid bytes) and carries the observed ASR identity
        and the verdict the run re-reports; the transcript is stored verbatim and
        is never projected into a receipt, machine output, or export. A write
        failure reports the fixed partial error and keeps the paid evidence.
        """
        execution = result.execution
        metadata: dict[str, Any] = quality.public_receipt(
            audio_sha256=self.hooks.sha256_file(audio_path),
            asr_provider=result.provider_id,
            asr_model=result.model_id,
            asr_runtime=execution.runtime,
            asr_model_revision=execution.model_revision,
        )
        metadata["turn_index"] = part.number
        metadata["quality_passed"] = bool(quality.passed)
        try:
            size_bytes = audio_path.stat().st_size
            with self.repository.transaction():
                artifact = self.repository.add_artifact(
                    self._run_uuid(),
                    role=ARTIFACT_ROLE_TTS_TURN_QUALITY,
                    path_kind=PATH_KIND_MANAGED_RELATIVE,
                    path=f"chunks/{part.chunk_id}.mp3",
                    part_uuid=part.record.part_uuid,
                    mime="audio/mpeg",
                    size_bytes=size_bytes,
                    sha256=metadata["audio_sha256"],
                    media_metadata=metadata,
                )
                self.repository.add_text_source(
                    self._run_uuid(),
                    kind=TEXT_KIND_VERIFICATION_TRANSCRIPT,
                    origin=NATIVE_ASR_ORIGIN,
                    content=result.transcript,
                    part_uuid=part.record.part_uuid,
                    artifact_uuid=artifact.artifact_uuid,
                    content_hash=sha256_text(result.transcript),
                    language=result.language or None,
                    text_completeness=(
                        TEXT_COMPLETENESS_COMPLETE
                        if result.transcript.strip()
                        else TEXT_COMPLETENESS_INCOMPLETE
                    ),
                )
        except (HistoryRepositoryError, OSError, ValueError):
            self.logger.event("error", "dialogue_quality_history_failed", chunk=part.number)
            raise NativeGenerationError(
                "the local per-turn quality verification ran, but the linked verification "
                "history could not be stored; the paid audio and cost were kept.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_QUALITY_HISTORY_FAILED,
            ) from None
        self._reload_view()

    def _dialogue_quality_summary(self) -> tuple[bool, bool | None]:
        """Return whether every turn carries a verdict and whether all passed."""
        evidence = _turn_quality_evidence(self.view)
        complete = all(part.record.part_uuid in evidence for part in self.view.parts)
        if not complete:
            return False, None
        return True, all(item.passed for item in evidence.values())

    def _reject_recorded_dialogue_failure(self) -> None:
        """Re-report a durable dialogue FAIL before any paid or local work exists.

        A recorded failure can never be repaired by another paid submit and its
        verdict is already durable, so ``generate --resume``, ``history resume``,
        and ``history sync`` all re-report it without running the model or sending
        another POST. The paid raw bytes, converted audio, and cost stay on disk.
        """
        if not self.dialogue_quality_gate:
            return
        evidence = _turn_quality_evidence(self.view)
        if any(not item.passed for item in evidence.values()):
            raise NativeGenerationError(
                "the run recorded a failed local per-turn quality verification; the paid audio "
                "and cost were kept. Use a different --run-id for a new attempt.",
                code=_EXIT_QUALITY,
                error_code=_ERROR_QUALITY_FAILED,
            )

    def _link_raw(
        self,
        part: NativeTtsPart,
        attempt: AttemptRecord,
        *,
        audio_bytes: bytes,
        audio_format: str,
        remote_task_id: str | None,
        generation_id: str | None,
    ) -> AttemptRecord:
        """Persist accepted bytes and their receipt, then link them to the attempt."""
        receipt: PaidRawReceipt = write_paid_raw_receipt(
            run_root=self.run_root,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            synthesis_fingerprint=part.fingerprint,
            chunk_id=part.chunk_id,
            number=part.number,
            audio_format=audio_format,
            audio_bytes=audio_bytes,
            remote_task_id=remote_task_id,
            generation_id=generation_id,
        )
        run, updated, _artifact = self.repository.record_polza_media_raw_saved(
            self._run_uuid(),
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            receipt=receipt,
        )
        self._revision = run.revision
        return updated

    def _link_sync_raw(
        self, part: NativeTtsPart, attempt: AttemptRecord, *, result: SynthesisResult
    ) -> AttemptRecord:
        """Link synchronous inline bytes and their observed cost with no remote id.

        The bounded receipt is written first (with a known ``generation_id`` but
        ``remote_task_id=None``), then the exact ``polza-tts`` cost the same
        response reported is committed together with the raw artifact, before any
        FFmpeg conversion. A crash between the receipt and this link leaves a
        ``submitting`` marker with valid on-disk evidence that a later resume
        reconciles locally. An ``openrouter-tts`` response reports no synchronous
        usage, so the amount stays unknown.
        """
        receipt: PaidRawReceipt = write_paid_raw_receipt(
            run_root=self.run_root,
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            synthesis_fingerprint=part.fingerprint,
            chunk_id=part.chunk_id,
            number=part.number,
            audio_format=result.audio_format,
            audio_bytes=result.audio_bytes,
            remote_task_id=None,
            generation_id=result.generation_id,
        )
        run, updated, _artifact = self.repository.record_polza_sync_raw_saved(
            self._run_uuid(),
            attempt_uuid=attempt.attempt_uuid,
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            receipt=receipt,
            amount=_sync_observed_amount(result),
        )
        self._revision = run.revision
        return updated

    def _submit_fresh_part(self, evidence: _PartEvidence) -> None:
        """Reserve, submit once, persist raw evidence, and commit the conversion."""
        part = evidence.part
        run, attempt = self.repository.reserve_paid_tts_attempt(
            self._run_uuid(),
            part_uuid=part.record.part_uuid,
            expected_revision=self._revision,
            provider=self.prepared.provider,
            model=self._model(),
            account_alias=PAID_ACCOUNT_ALIAS,
        )
        self._revision = run.revision
        provider = self._provider_instance()
        uses_media = self._uses_media_route()
        accepted_id: dict[str, str] = {}
        if uses_media:
            self._bind_media_callbacks(
                provider, attempt=attempt, part=part, accepted_id=accepted_id
            )
        self.logger.event("info", "paid_submit_started", chunk=part.number, id=part.chunk_id)
        try:
            result = self._invoke_provider(provider, part)
        except Exception:
            # The provider exception body is untrusted (it may carry an echoed
            # Authorization header), so it never reaches the public error message
            # or a chained cause the machine envelope could reveal.
            self.logger.event("error", "chunk_provider_failed", chunk=part.number)
            raise NativeGenerationError(
                f"Failed to synthesize {part.chunk_id}.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SYNTHESIS_FAILED,
            ) from None
        if uses_media:
            updated = self._link_raw(
                part,
                attempt,
                audio_bytes=result.audio_bytes,
                audio_format=result.audio_format,
                remote_task_id=accepted_id.get("id"),
                generation_id=result.generation_id,
            )
        else:
            # The synchronous provider's optional generation id is untrusted and
            # is not required to be a bounded opaque token (a dotted, overlong, or
            # non-ASCII ``X-Generation-Id`` is possible). The bounded receipt can
            # only carry such a token, so an invalid id is dropped to ``None``
            # before any receipt is written: the accepted paid audio is never
            # lost to a strict-id rejection. The sanitized result feeds the
            # receipt, the raw artifact link, and the conversion metadata alike, so
            # every later verification and the export see the same dropped id.
            result = replace(result, generation_id=bounded_opaque_token(result.generation_id))
            updated = self._link_sync_raw(part, attempt, result=result)
        self._convert_and_commit(evidence, result, updated)

    def _invoke_provider(self, provider: Any, part: NativeTtsPart) -> SynthesisResult:
        """Submit one part once, applying the per-turn dialogue cast voice as legacy did.

        A non-dialogue native run keeps its run-level voice and calls the provider
        directly. A dialogue turn goes through the same ``synthesize_part`` seam the
        legacy executor uses, so an ``OpenRouterTTSProvider`` receives its own cast
        voice as the ``voice`` keyword while a provider that does not accept one is
        called unchanged. The provider is invoked exactly once; its error propagates.
        """
        if not self.dialogue:
            return provider.synthesize_chunk(part.text, part.chunk_id)
        chunk = ScriptChunk(
            number=part.number,
            id=part.chunk_id,
            text=part.text,
            speaker=part.speaker,
            voice=part.cast_voice,
            voice_fingerprint=part.voice_fingerprint,
            pause_after_ms=part.pause_after_ms,
        )
        return synthesize_part(provider, PreparedPart(chunk=chunk, voice=part.effective_voice))

    def _recover_via_get(self, evidence: _PartEvidence, attempt: AttemptRecord) -> None:
        """Finish a known accepted task with GET calls only, then convert locally."""
        part = evidence.part
        assert attempt.remote_id is not None  # narrowed by the caller
        provider = self._provider_instance()
        accepted_id: dict[str, str] = {}
        self._bind_media_callbacks(provider, attempt=attempt, part=part, accepted_id=accepted_id)
        self.logger.event("info", "paid_media_recovery", chunk=part.number, id=part.chunk_id)
        try:
            result = provider.recover_media_task(attempt.remote_id, part.text, part.chunk_id)
        except Exception:
            # The provider exception body is untrusted and never becomes the public
            # error message or a chained cause.
            raise NativeGenerationError(
                f"Failed to recover media task for {part.chunk_id}.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SYNTHESIS_FAILED,
            ) from None
        updated = self._link_raw(
            part,
            attempt,
            audio_bytes=result.audio_bytes,
            audio_format=result.audio_format,
            remote_task_id=attempt.remote_id,
            generation_id=result.generation_id,
        )
        self._convert_and_commit(evidence, result, updated)

    def _convert_from_raw(self, evidence: _PartEvidence, attempt: AttemptRecord) -> None:
        """Rebuild one chunk from retained raw paid bytes with no provider call."""
        raw = evidence.raw_artifact
        assert raw is not None  # narrowed by the caller
        metadata = raw.media_metadata if isinstance(raw.media_metadata, dict) else {}
        audio_format = metadata.get("format")
        if not isinstance(audio_format, str):
            raise NativeGenerationError(
                "the retained paid raw artifact is missing its audio format.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        generation_id = metadata.get("generation_id")
        audio_bytes = _verified_raw_bytes(raw, self.run_root, self.hooks.sha256_file)
        result = SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            transcript=evidence.part.text,
            generation_id=generation_id if isinstance(generation_id, str) else None,
            client_path="requests",
        )
        self.logger.event(
            "info", "paid_raw_recovery", chunk=evidence.part.number, id=evidence.part.chunk_id
        )
        self._convert_and_commit(evidence, result, attempt)

    def _reconcile_raw_receipt(
        self, evidence: _PartEvidence, attempt: AttemptRecord
    ) -> ArtifactRecord | None:
        """Link an already-written raw receipt that has no database artifact yet.

        A crash between writing the paid bytes plus their receipt and committing
        the database row leaves valid evidence on disk. Re-verifying it and
        linking it here recovers that window without a provider request; the
        verification hashes the file outside any transaction. ``None`` means the
        on-disk evidence is absent or does not match, so the caller falls back to
        GET-only recovery; a return value is the newly committed raw artifact.
        """
        from ..history.raw_receipt import verify_paid_raw_receipt

        part = evidence.part
        receipt_format = _raw_format_from_receipt(self.run_root, part.chunk_id, part.number)
        if receipt_format is None:
            return None
        try:
            receipt = verify_paid_raw_receipt(
                run_root=self.run_root,
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                synthesis_fingerprint=part.fingerprint,
                chunk_id=part.chunk_id,
                number=part.number,
                audio_format=receipt_format,
                remote_task_id=attempt.remote_id,
            )
        except (PaidRawReceiptError, ValueError):
            return None
        try:
            run, _updated, artifact = self.repository.record_polza_media_raw_saved(
                self._run_uuid(),
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                receipt=receipt,
            )
        except HistoryRepositoryError:
            return None
        self._revision = run.revision
        return artifact

    def _reconcile_sync_raw_receipt(
        self, evidence: _PartEvidence, attempt: AttemptRecord
    ) -> ArtifactRecord | None:
        """Link a synchronous raw receipt left by a crash between bytes and database.

        A synchronous submit has no remote task id, so a crash after the response
        was received (bytes plus receipt written) but before the database link
        leaves the attempt in ``submitting`` with valid evidence on disk. Re-verifying
        that receipt and linking it under CAS here rebuilds the part locally with no
        POST and no GET. ``None`` means the evidence is absent or does not match, so
        the caller keeps the documented unconfirmed-submit block and never falls back
        to a paid submit. The exact cost the response reported is not in the receipt,
        so a cost observed in this crash window is lost and stays unknown.
        """
        from ..history.raw_receipt import verify_paid_raw_receipt

        part = evidence.part
        receipt_format = _raw_format_from_receipt(self.run_root, part.chunk_id, part.number)
        if receipt_format is None:
            return None
        try:
            receipt = verify_paid_raw_receipt(
                run_root=self.run_root,
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                synthesis_fingerprint=part.fingerprint,
                chunk_id=part.chunk_id,
                number=part.number,
                audio_format=receipt_format,
                remote_task_id=None,
            )
        except (PaidRawReceiptError, ValueError):
            return None
        try:
            run, _updated, artifact = self.repository.record_polza_sync_raw_saved(
                self._run_uuid(),
                attempt_uuid=attempt.attempt_uuid,
                part_uuid=part.record.part_uuid,
                expected_revision=self._revision,
                receipt=receipt,
                amount=None,
            )
        except HistoryRepositoryError:
            return None
        self._revision = run.revision
        return artifact

    def _process_first_incomplete(self, evidence: _PartEvidence) -> None:
        """Recover or submit exactly the first part without a verified chunk file."""
        attempt = evidence.attempt
        if attempt is None:
            # A converted artifact without an attempt cannot be reconciled safely.
            raise NativeGenerationError(
                "a committed part carries partial history without its paid attempt.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        status = attempt.status
        if status == ATTEMPT_STATUS_SUBMITTING:
            if not self._uses_media_route():
                # A synchronous POST with no remote task id can still have been
                # accepted: if its response was received, the raw bytes and their
                # receipt are already on disk. Reconcile exactly that same-attempt
                # local evidence and rebuild the part with no POST and no GET.
                linked = self._reconcile_sync_raw_receipt(evidence, attempt)
                if linked is not None:
                    self._convert_from_raw(replace(evidence, raw_artifact=linked), attempt)
                    return
            # No recoverable evidence exists, so the paid outcome is unknown and
            # must never be repeated automatically.
            raise NativeGenerationError(
                "refusing to resume: the previous paid submit for this part was never confirmed. "
                "A paid outcome must not be repeated automatically.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_SUBMIT_UNCONFIRMED,
            )
        if status not in (
            ATTEMPT_STATUS_REMOTE_ACCEPTED,
            ATTEMPT_STATUS_RAW_SAVED,
            ATTEMPT_STATUS_COMPLETED,
        ):
            raise NativeGenerationError(
                "refusing to resume: this part carries an unrecognized paid attempt state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_EVIDENCE_INCONSISTENT,
            )
        if evidence.raw_artifact is not None:
            self._convert_from_raw(evidence, attempt)
            return
        if status == ATTEMPT_STATUS_REMOTE_ACCEPTED and attempt.remote_id is not None:
            linked = self._reconcile_raw_receipt(evidence, attempt)
            if linked is not None:
                # The crash window between raw and database left valid evidence on
                # disk; rebuild that part locally with no provider request at all.
                self._convert_from_raw(replace(evidence, raw_artifact=linked), attempt)
                return
            self._recover_via_get(evidence, attempt)
            return
        if status == ATTEMPT_STATUS_RAW_SAVED:
            raise NativeGenerationError(
                "refusing to resume: this part records saved raw audio but its linked raw "
                "artifact is missing.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_RAWS_EVIDENCE_INVALID,
            )
        raise NativeGenerationError(
            "refusing to resume: this part's paid evidence cannot be reconstructed locally.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_EVIDENCE_INCONSISTENT,
        )

    def _reload_view(self) -> None:
        """Reload the verified view so it reflects the parts this process just committed."""
        self.view = load_native_tts_view(self.repository, self._run_uuid())

    def run_parts(self) -> None:
        """Execute or recover every part, stopping before any inconsistent gap.

        Only the first part without a verified converted chunk may carry
        recoverable evidence. Any later part that already carries paid evidence
        while an earlier part is unfinished fails closed, so a partially written
        history can never be extended by a new paid submit. An executor created by
        ``history sync`` (``paid_submit_allowed`` is ``False``) fails closed on an
        unattempted part instead of submitting it, so sync never starts new paid
        work.

        A dialogue run adds the required per-turn local quality gate: a recorded
        FAIL is re-reported before any work, a verified chunk whose verdict is still
        pending has that one check repaired locally with no provider request, and a
        fresh or recovered turn is checked right after its conversion.
        """
        evidence = _collect_evidence(self.view)
        if self.local:
            # The admitted local route has no paid attempt and no paid-marker guard;
            # its own loop handles a retryable local attempt and raw recovery.
            self._run_local_parts(evidence)
            return
        if self.dialogue_quality_gate:
            # A recorded per-turn FAIL is durable and can never be repaired by a
            # paid submit, so every verb re-reports it before any work.
            self._reject_recorded_dialogue_failure()
        first_incomplete_done = False
        for index, item in enumerate(evidence):
            if _verified_chunk_file(item, self.run_root, self.hooks.sha256_file) is not None:
                if (
                    self.dialogue_quality_gate
                    and self.local_quality_allowed
                    and item.part.record.part_uuid not in _turn_quality_evidence(self.view)
                ):
                    # A crash between a turn's conversion and its local check leaves
                    # the chunk verified but the verdict pending; an explicit resume
                    # repairs exactly that turn with no provider request at all.
                    self._run_turn_quality(item)
                self.hooks.progress(
                    f"Skipping {item.part.chunk_id}/{len(evidence):02d}: already committed"
                )
                continue
            if any(_has_paid_evidence(later) for later in evidence[index + 1 :]):
                raise NativeGenerationError(
                    "refusing to resume: a later part carries paid evidence before this part "
                    "completed.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            if first_incomplete_done and _has_paid_evidence(item):
                raise NativeGenerationError(
                    "refusing to resume: paid evidence does not form a contiguous prefix.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_EVIDENCE_INCONSISTENT,
                )
            self.hooks.progress(
                f"Generating {item.part.chunk_id}/{len(evidence):02d}: {item.part.chunk_id}.mp3"
            )
            if _has_paid_evidence(item):
                self._process_first_incomplete(item)
            elif self.paid_submit_allowed:
                self._submit_fresh_part(item)
            else:
                # ``history sync`` retrieves the state/result of known operations
                # only. An unattempted part needs a new paid submit, which sync
                # never performs, so it fails closed here before any provider or
                # key exists instead of running the rest of the run.
                raise NativeGenerationError(
                    "refusing to sync: an unattempted part remains and history sync never starts "
                    "a new paid submit. Use history resume to synthesize it.",
                    code=_EXIT_PROVIDER,
                    error_code=_ERROR_SYNC_SUBMIT_REQUIRED,
                )
            first_incomplete_done = True

    # -- assembly and export --------------------------------------------------

    def _all_chunks_verified(self) -> list[Path]:
        """Return the ordered, verified chunk paths, or fail closed."""
        evidence = _collect_evidence(self.view)
        ordered: list[Path] = []
        for item in evidence:
            verified = _verified_chunk_file(item, self.run_root, self.hooks.sha256_file)
            if verified is None:
                raise NativeGenerationError(
                    "refusing to assemble: a committed chunk file is missing or does not match "
                    "its history.",
                    code=_EXIT_OUTPUT,
                    error_code=_ERROR_ASSEMBLY_FAILED,
                )
            ordered.append(verified)
        return ordered

    def _ordered_dialogue_turns(self) -> list[tuple[Path, int]]:
        """Return the ordered (turn audio, pause) pairs for the dialogue concat.

        The pairs follow the committed part order, never a directory glob, so the
        250 ms between turns, the 600 ms after a section delimiter, and the 0 ms
        after the final turn are preserved exactly as the validated plan recorded
        them.
        """
        evidence = _collect_evidence(self.view)
        turns: list[tuple[Path, int]] = []
        for item in evidence:
            verified = _verified_chunk_file(item, self.run_root, self.hooks.sha256_file)
            if verified is None:
                raise NativeGenerationError(
                    "refusing to assemble: a committed dialogue turn file is missing or does "
                    "not match its history.",
                    code=_EXIT_OUTPUT,
                    error_code=_ERROR_ASSEMBLY_FAILED,
                )
            turns.append((verified, item.part.pause_after_ms))
        return turns

    def _require_dialogue_quality_before_concat(self) -> None:
        """Fail closed until every turn carries a verdict, and re-report a FAIL.

        The pending local check is run only when this execution may run a local
        model (``generate``/``history resume``); ``history sync`` never runs one and
        reports the incomplete state instead. A recorded FAIL is always re-reported
        with the existing quality exit and never re-runs the model.
        """
        if self.quality_options is None:
            raise NativeGenerationError(
                "the dialogue native route requires an installed local --tts-quality-provider.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_QUALITY_OPTIONS_INVALID,
            )
        evidence = _collect_evidence(self.view)
        verdicts = _turn_quality_evidence(self.view)
        pending = [item for item in evidence if item.part.record.part_uuid not in verdicts]
        if pending and self.local_quality_allowed:
            for item in pending:
                self._run_turn_quality(item)
            verdicts = _turn_quality_evidence(self.view)
            pending = [item for item in evidence if item.part.record.part_uuid not in verdicts]
        if pending:
            raise NativeGenerationError(
                "the dialogue turns are complete, but their local quality verification has not "
                "run; an explicit resume runs the pending local check before concat.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_QUALITY_INCOMPLETE,
            )
        if any(not item.passed for item in verdicts.values()):
            raise NativeGenerationError(
                "the dialogue turns are complete, but a local per-turn quality verification "
                "recorded a failure; the paid audio and cost were kept.",
                code=_EXIT_QUALITY,
                error_code=_ERROR_QUALITY_FAILED,
            )

    def assemble_and_complete(self) -> None:
        """Concatenate the ordered committed parts and close the run."""
        output_path = self.paths.full_mp3
        staged = _new_staging_path(output_path)
        try:
            if self.dialogue:
                self.hooks.concat_dialogue_turns(
                    self.ffmpeg_path, self._ordered_dialogue_turns(), staged
                )
            else:
                self.hooks.concat_audio_files(self.ffmpeg_path, self._all_chunks_verified(), staged)
            duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, staged)
            sha256 = self.hooks.sha256_file(staged)
            size_bytes = staged.stat().st_size
            _publish_staged(
                staged,
                output_path,
                error_code=_ERROR_ASSEMBLY_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            raise NativeGenerationError(
                f"Failed to concat MP3 chunks: {exc}",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_ASSEMBLY_FAILED,
            ) from exc
        try:
            run, _artifact = self.repository.record_tts_run_completed(
                self._run_uuid(),
                expected_revision=self._revision,
                path=self.paths.full_mp3.name,
                mime="audio/mpeg",
                size_bytes=size_bytes,
                sha256=sha256,
                media_metadata={
                    "duration_ms": duration_ms,
                    "execution_source": build_execution_identity(),
                },
            )
        except HistoryRepositoryError as exc:
            raise NativeGenerationError(
                "the committed history for this run conflicts with the assembled final audio; "
                "refusing to overwrite committed state.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_HISTORY_CONFLICT,
            ) from exc
        self._revision = run.revision
        self.logger.event("info", "run_completed", duration_ms=duration_ms)

    def _final_artifact_verified(self) -> bool:
        finals = [
            artifact
            for artifact in self.view.artifacts
            if artifact.role == ARTIFACT_ROLE_FINAL_AUDIO
        ]
        if len(finals) != 1:
            return False
        final = finals[0]
        if final.sha256 is None or final.size_bytes is None:
            return False
        path = self.run_root / final.path
        if not path.is_file():
            return False
        try:
            return (
                path.stat().st_size == final.size_bytes
                and self.hooks.sha256_file(path) == final.sha256
            )
        except OSError:
            return False

    def ensure_complete(self) -> None:
        """Complete the run only from verified committed parts.

        A run whose final artifact is already present and verified is left
        untouched, so an export-only repair neither reassembles audio nor reads an
        API key. Otherwise the ordered committed parts are assembled and the run
        is closed.
        """
        self._reload_view()
        if self.view.run.status == "completed" and self._final_artifact_verified():
            return
        if self.dialogue_quality_gate:
            # The required per-turn gate runs before the final concat, so a
            # recorded or still-pending turn verdict stops here instead of
            # assembling audio the project cannot claim is verified.
            self._require_dialogue_quality_before_concat()
        self.assemble_and_complete()

    # -- linked local timing --------------------------------------------------

    def _final_audio_path(self) -> Path:
        """Return the committed final audio path, or fail closed before timing."""
        finals = [
            artifact
            for artifact in self.view.artifacts
            if artifact.role == ARTIFACT_ROLE_FINAL_AUDIO
        ]
        if len(finals) != 1:
            raise NativeGenerationError(
                "the completed run has no single committed final audio artifact to time.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_FINAL_AUDIO_MISSING,
            )
        path = self.run_root / finals[0].path
        if not path.is_file():
            raise NativeGenerationError(
                "the committed final audio is missing; refusing to run local timings.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_FINAL_AUDIO_MISSING,
            )
        return path

    def _verified_timing_artifact(self, artifact: ArtifactRecord) -> Path | None:
        """Return a linked timing artifact's path only when it still matches its row."""
        if artifact.sha256 is None or artifact.size_bytes is None:
            return None
        path = Path(artifact.path)
        if not path.is_file():
            return None
        try:
            if path.stat().st_size != artifact.size_bytes:
                return None
            if self.hooks.sha256_file(path) != artifact.sha256:
                return None
        except OSError:
            return None
        return path

    def _linked_timing(self) -> NativeLinkedTiming | None:
        """Return the verified linked local-timing evidence for this run, else ``None``.

        Timing is persisted as a separate ``timings`` run whose ``parent_uuid`` is
        this TTS run, so the link is read from committed rows rather than a JSON
        marker. The evidence counts only when exactly one child run exists and
        both its ``timings_json`` and ``srt`` artifacts still match their committed
        digest and size; a duplicate, altered, or missing artifact reports no
        usable timing, and two child runs fail closed instead of being told apart.
        """
        children = self.repository.find_runs_by_parent(self._run_uuid(), limit=MAX_QUERY_LIMIT)
        timing_runs = [
            run
            for run in children
            if run.operation == OPERATION_TIMINGS and run.legacy_source_root is None
        ]
        if len(timing_runs) > 1:
            raise NativeGenerationError(
                "more than one linked local timing run belongs to this run; refusing to "
                "choose one.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_TIMING_CONFLICT,
            )
        if not timing_runs:
            return None
        run = timing_runs[0]
        if run.status != RUN_STATUS_COMPLETED:
            return None
        artifacts = self.repository.get_artifacts(run.run_uuid)
        json_artifacts = [item for item in artifacts if item.role == ARTIFACT_ROLE_TIMINGS_JSON]
        srt_artifacts = [item for item in artifacts if item.role == ARTIFACT_ROLE_SRT]
        if len(json_artifacts) != 1 or len(srt_artifacts) != 1:
            return None
        timings_json = self._verified_timing_artifact(json_artifacts[0])
        srt = self._verified_timing_artifact(srt_artifacts[0])
        if timings_json is None or srt is None:
            return None
        snapshot = run.config_snapshot if isinstance(run.config_snapshot, dict) else {}
        segment_count = snapshot.get("segment_count")
        return NativeLinkedTiming(
            run_uuid=run.run_uuid,
            timings_json=timings_json,
            srt=srt,
            segment_count=(
                segment_count
                if isinstance(segment_count, int) and not isinstance(segment_count, bool)
                else None
            ),
        )

    def preflight_timing(self) -> None:
        """Fail closed before any paid submit when the pending local timing cannot run.

        A run whose timing is already linked, or one that runs under ``history
        sync`` (where local model inference is not allowed), needs no local
        timing and is skipped. Otherwise the availability probe runs before
        ``run_parts`` so an unavailable or uncached model stops the command before
        the paid TTS POST.
        """
        options = self.timing_options
        if options is None or not self.local_timing_allowed:
            return
        if self._linked_timing() is not None:
            return
        _preflight_local_timing(options)

    def _publish_timing_artifact(self, target: Path, text: str) -> None:
        """Publish one timing artifact through the shared private staging seam."""
        staged = _new_staging_path(target)
        try:
            staged.write_text(text, encoding="utf-8")
            size_bytes = staged.stat().st_size
            sha256 = self.hooks.sha256_file(staged)
            _publish_staged(
                staged,
                target,
                error_code=_ERROR_TIMING_ARTIFACT_FAILED,
                size_bytes=size_bytes,
                sha256=sha256,
                sha256_file=self.hooks.sha256_file,
            )
        except NativeGenerationError:
            staged.unlink(missing_ok=True)
            raise
        except Exception as exc:
            staged.unlink(missing_ok=True)
            raise NativeGenerationError(
                f"Failed to write the local timing artifact {target.name}.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_TIMING_ARTIFACT_FAILED,
            ) from exc

    def _timing_history_save(
        self,
        options: NativeTimingOptions,
        manifest: dict[str, Any],
        audio_path: Path,
        timings_json_path: Path,
        srt_path: Path,
    ) -> AsrHistorySave:
        """Build the linked timing-history evidence from the durable timing JSON.

        The transcript is read back from the just-published timing JSON, so the
        stored text is exactly what the artifact holds and no span is invented.
        The save carries this TTS run root as ``parent_run_root`` so
        :func:`persist_asr_history` links it through ``parent_uuid`` instead of
        fabricating TTS parts or a second TTS attempt.
        """
        payload = json.loads(timings_json_path.read_text(encoding="utf-8"))
        segments = payload.get("segments") or []
        transcript = " ".join(str(segment.get("text") or "") for segment in segments).strip()
        snapshot: dict[str, Any] = {
            "operation_origin": NATIVE_ASR_ORIGIN,
            "provider": manifest.get("provider") or options.provider,
            "model": manifest.get("model"),
            "backend": manifest.get("backend"),
            "device": manifest.get("device"),
            "compute_type": manifest.get("compute_type"),
            "language": manifest.get("language") or None,
            "timestamp_basis": "provider_segment_timestamps",
            "word_timestamps_requested": options.word_timestamps,
            "segment_count": len(segments),
            "total_duration_ms": manifest.get("total_duration_ms"),
            "source_audio": str(audio_path.resolve()),
            "timings_json": str(timings_json_path),
            "srt": str(srt_path),
        }
        return AsrHistorySave(
            operation=OPERATION_TIMINGS,
            attempt_call_type=ATTEMPT_CALL_TYPE_TIMING,
            provider=manifest.get("provider") or options.provider,
            model=manifest.get("model") or options.model,
            config_snapshot=snapshot,
            text_sources=(
                AsrHistoryText(
                    kind=TEXT_KIND_ASR_TRANSCRIPT,
                    content=transcript,
                    language=manifest.get("language") or None,
                    text_completeness=(
                        TEXT_COMPLETENESS_COMPLETE
                        if transcript.strip()
                        else TEXT_COMPLETENESS_INCOMPLETE
                    ),
                ),
            ),
            artifacts=(
                AsrHistoryArtifact(
                    role=ARTIFACT_ROLE_TIMINGS_JSON,
                    path=timings_json_path,
                    mime="application/json",
                ),
                AsrHistoryArtifact(
                    role=ARTIFACT_ROLE_SRT,
                    path=srt_path,
                    mime="application/x-subrip",
                ),
            ),
            parent_run_root=self.run_root,
            source_audio=audio_path,
            source_audio_mime="audio/mpeg",
        )

    def ensure_timing(self) -> None:
        """Run the pending local timing step for a completed run, once only.

        The step runs only when the run recorded local timing, the timing is not
        already linked, and this execution may run a local model (``generate``,
        ``generate --resume``, and ``history resume``; never ``history sync``). A
        timing failure or a timing-history persistence failure keeps the completed
        TTS audio, its paid raw bytes, and its cost untouched: no legacy JSON
        writer runs, the just-written timing artifacts are removed so no export can
        reference unlinked timing, and the caller reports the fixed partial error.
        """
        options = self.timing_options
        if options is None:
            return
        if self._linked_timing() is not None:
            return
        if not self.local_timing_allowed:
            return
        self._reload_view()
        audio_path = self._final_audio_path()
        timings_json_path = self.paths.output_root / f"{self.paths.prefix}.timings.json"
        srt_path = self.paths.output_root / f"{self.paths.prefix}.srt"
        from .transcription import transcribe_timing_audio

        try:
            timing = transcribe_timing_audio(
                audio_path=audio_path,
                timing_provider=options.provider,
                model=options.model,
                device=options.device,
                compute_type=options.compute,
                language=options.language,
                word_timestamps=options.word_timestamps,
                quiet=True,
                local_files_only=True,
            )
        except Exception as exc:
            self.logger.event("error", "timings_failed", error=type(exc).__name__)
            raise NativeGenerationError(
                "the voiceover audio is complete, but local timing extraction failed.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_TIMING_FAILED,
            ) from None
        duration_ms = self.hooks.mp3_duration_ms(self.ffprobe_path, audio_path)
        manifest = build_timing_manifest(timing, duration_ms)
        published: list[Path] = []
        try:
            self._publish_timing_artifact(
                timings_json_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            published.append(timings_json_path)
            self._publish_timing_artifact(srt_path, build_srt(timing))
            published.append(srt_path)
        except NativeGenerationError:
            for path in published:
                path.unlink(missing_ok=True)
            raise
        try:
            save = self._timing_history_save(
                options, manifest, audio_path, timings_json_path, srt_path
            )
            persist_asr_history(save)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            self.logger.event("error", "timings_history_failed")
            raise NativeGenerationError(
                "the voiceover audio and timings files were produced, but the linked timing "
                "history could not be stored.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_TIMING_HISTORY_FAILED,
            ) from None
        if self._linked_timing() is None:
            raise NativeGenerationError(
                "the linked timing history could not be verified after it was written.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_TIMING_HISTORY_FAILED,
            )
        self.logger.event("info", "timings_complete", segments=len(timing.segments))

    # -- linked local quality verification ------------------------------------

    def _linked_quality(self) -> NativeLinkedQuality | None:
        """Return the verified linked local quality evidence for this run, else ``None``.

        Quality is persisted as a separate ``verify`` run whose ``parent_uuid`` is
        this TTS run, so the link is read from committed rows rather than a JSON
        marker. The evidence counts only when exactly one child run exists, it
        completed, and its one private ``verification_transcript`` source still
        matches its stored hash next to a recorded boolean verdict; a duplicate or
        altered source reports no usable verification, and two child runs fail
        closed instead of being told apart.
        """
        children = self.repository.find_runs_by_parent(self._run_uuid(), limit=MAX_QUERY_LIMIT)
        quality_runs = [
            run
            for run in children
            if run.operation == OPERATION_VERIFY and run.legacy_source_root is None
        ]
        if len(quality_runs) > 1:
            raise NativeGenerationError(
                "more than one linked local quality-verification run belongs to this run; "
                "refusing to choose one.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_QUALITY_CONFLICT,
            )
        if not quality_runs:
            return None
        run = quality_runs[0]
        if run.status != RUN_STATUS_COMPLETED:
            return None
        snapshot = run.config_snapshot if isinstance(run.config_snapshot, dict) else {}
        passed = snapshot.get("quality_passed")
        if not isinstance(passed, bool):
            return None
        transcripts = [
            source
            for source in self.repository.get_text_sources(run.run_uuid)
            if source.kind == TEXT_KIND_VERIFICATION_TRANSCRIPT
        ]
        if len(transcripts) != 1:
            return None
        transcript_content = transcripts[0].content
        if not isinstance(transcript_content, str):
            return None
        if transcripts[0].content_hash != sha256_text(transcript_content):
            return None
        return NativeLinkedQuality(run_uuid=run.run_uuid, passed=passed)

    def preflight_quality(self) -> None:
        """Fail closed before any paid submit when the pending local quality cannot run.

        A run whose verification is already linked, or one that runs under
        ``history sync`` (where local model inference is not allowed), needs no
        local model and is skipped. Otherwise the dependency and local-asset probe
        runs before ``run_parts`` so an unavailable or absent local model stops the
        command before the paid TTS POST.
        """
        options = self.quality_options
        if options is None or not self.local_quality_allowed:
            return
        if self._linked_quality() is not None:
            return
        _preflight_local_quality(options)

    def _quality_history_save(
        self, result: ASRResult, quality: TTSQualityResult, audio_path: Path
    ) -> AsrHistorySave:
        """Build the linked quality-history evidence from one observed verification.

        The observed verification transcript is stored as its own private
        ``verification_transcript`` role and the expected script text is never
        re-stored, so a check can never overwrite or impersonate the run's own
        script. The run snapshot is the content-free receipt: the observed ASR
        identity plus the verdict, similarity, and failure reasons, with no
        transcript text. ``parent_run_root`` links the child run to this TTS run
        through ``parent_uuid`` instead of fabricating TTS parts or a second TTS
        attempt.
        """
        execution = result.execution
        snapshot: dict[str, Any] = {
            "operation_origin": NATIVE_ASR_ORIGIN,
            "provider": result.provider_id,
            "model": result.model_id,
            "language": result.language or None,
            "runtime": execution.runtime,
            "model_revision": execution.model_revision,
            "device": execution.resolved_device,
            "compute": execution.resolved_compute,
            "quality_passed": bool(quality.passed),
            "similarity": float(quality.similarity),
            "failure_reasons": [str(reason) for reason in quality.failure_reasons],
            "expected_text_persisted": False,
        }
        return AsrHistorySave(
            operation=OPERATION_VERIFY,
            attempt_call_type=ATTEMPT_CALL_TYPE_VERIFY,
            provider=result.provider_id,
            model=result.model_id,
            config_snapshot=snapshot,
            text_sources=(
                AsrHistoryText(
                    kind=TEXT_KIND_VERIFICATION_TRANSCRIPT,
                    content=result.transcript,
                    language=result.language or None,
                    text_completeness=(
                        TEXT_COMPLETENESS_COMPLETE
                        if result.transcript.strip()
                        else TEXT_COMPLETENESS_INCOMPLETE
                    ),
                ),
            ),
            parent_run_root=self.run_root,
            source_audio=audio_path,
            source_audio_mime="audio/mpeg",
        )

    def ensure_quality(self) -> None:
        """Run the pending local quality verification for a completed run, once only.

        The step runs only when the run recorded local quality, the verification is
        not already linked, and this execution may run a local model (``generate``,
        ``generate --resume``, and ``history resume``; never ``history sync``). The
        comparison is the existing ``verify-tts`` one -- the same default similarity
        and word-ratio thresholds against the run's own committed script text -- so
        no new threshold is invented. The observed verdict -- PASS or FAIL -- is
        persisted through the same linked ``verify`` helper before the outcome is
        reported, so a failed check is durable: a later ``generate --resume`` or
        ``history resume`` re-reports the recorded failure as the same quality exit
        with no second local model run, and ``history sync`` reports the persisted
        verdict without running the model at all. A mismatch keeps the completed TTS
        audio, its paid raw bytes, and its cost untouched; a local transcription
        failure or a linked-history persistence failure keeps the same evidence and
        reports the fixed partial error instead of claiming a stored verdict.
        """
        options = self.quality_options
        if options is None:
            return
        if self.dialogue:
            # A dialogue run verifies each turn before concat and re-reports any
            # recorded failure through ``ensure_complete``; there is no single
            # linked ``verify`` child to run here.
            return
        linked = self._linked_quality()
        if linked is not None:
            # A recorded failure is re-surfaced by every command that may run a
            # local model; ``history sync`` is a state reader, so it reports the
            # persisted verdict through the summary instead of failing.
            if not linked.passed and self.local_quality_allowed:
                raise NativeGenerationError(
                    "the voiceover audio is complete, but the linked local TTS quality "
                    "verification recorded a failure.",
                    code=_EXIT_QUALITY,
                    error_code=_ERROR_QUALITY_FAILED,
                )
            return
        if not self.local_quality_allowed:
            return
        self._reload_view()
        audio_path = self._final_audio_path()
        from .transcription import transcribe_local_asr_quality

        try:
            result = transcribe_local_asr_quality(
                provider_id=options.provider,
                audio_path=audio_path,
                model=options.model,
                device=options.device,
                compute=options.compute,
                runtime=options.runtime,
                language=options.language,
            )
        except Exception as exc:
            self.logger.event("error", "quality_transcription_failed", error=type(exc).__name__)
            raise NativeGenerationError(
                "the voiceover audio is complete, but the local quality transcription failed.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_QUALITY_ASR_FAILED,
            ) from None
        quality = evaluate_tts_transcript(
            expected_text=self.view.script_text,
            actual_transcript=result.transcript,
        )
        # Persist the observed verification before any outcome is reported, so the
        # verdict is durable and a failed check is never silently claimed. A write
        # failure reports the fixed partial error and keeps the provider response.
        try:
            persist_asr_history(self._quality_history_save(result, quality, audio_path))
        except Exception:
            self.logger.event("error", "quality_history_failed")
            raise NativeGenerationError(
                "the local quality verification ran, but the linked verification history "
                "could not be stored; the paid audio and cost were kept.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_QUALITY_HISTORY_FAILED,
            ) from None
        if self._linked_quality() is None:
            raise NativeGenerationError(
                "the linked quality-verification history could not be verified after it was "
                "written.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_QUALITY_HISTORY_FAILED,
            )
        if not quality.passed:
            self.logger.event("warning", "quality_failed", reasons=len(quality.failure_reasons))
            raise NativeGenerationError(
                "the voiceover audio is complete, but the local TTS quality verification "
                "did not pass; the paid audio and cost were kept and the failed verification "
                "was recorded.",
                code=_EXIT_QUALITY,
                error_code=_ERROR_QUALITY_FAILED,
            )
        self.logger.event("info", "quality_verified", similarity=float(quality.similarity))

    def export(self) -> NativeGenerationSummary:
        """Project one verified committed view onto the compatibility JSON files."""
        self._reload_view()
        view = self.view
        try:
            export = build_native_export(
                view,
                self.paths,
                script_path=self.script_path,
                ffmpeg_path=self.ffmpeg_path,
                ffprobe_path=self.ffprobe_path,
            )
            write_native_export(export, self.paths)
        except OSError as exc:
            raise NativeGenerationError(
                "Failed to write the compatibility JSON exports; the history database and audio "
                "are intact.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_EXPORT_FAILED,
            ) from exc
        except NativeGenerationError:
            raise
        except Exception as exc:
            raise NativeGenerationError(
                "Failed to project the committed history into the compatibility JSON exports; "
                "the history database and audio are intact.",
                code=_EXIT_OUTPUT,
                error_code=_ERROR_EXPORT_FAILED,
            ) from exc
        duration_ms = int(export.run_manifest.get("main_duration_ms") or 0)
        files = {
            "full_mp3": str(self.paths.full_mp3),
            "run_json": str(self.paths.run_json),
            "chunks_json": str(self.paths.chunks_json),
            "manifest_json": str(self.paths.output_root / "manifest.json"),
        }
        # Only a linked, still-verified timing run adds timing references to the
        # result; a run that requested timing but has none stays incomplete.
        timing = self._linked_timing()
        segment_count: int | None = None
        if timing is not None:
            files["timings_json"] = str(timing.timings_json)
            files["srt"] = str(timing.srt)
            segment_count = timing.segment_count
        # The private verification transcript is never projected into a result, the
        # JSON exports, or the compatibility manifests: only the content-free
        # verdict and whether a linked verification exists at all.
        if self.dialogue_quality_gate:
            quality_complete, quality_passed = self._dialogue_quality_summary()
        else:
            quality = self._linked_quality()
            quality_complete = quality is not None
            quality_passed = quality.passed if quality is not None else None
        return NativeGenerationSummary(
            run_uuid=view.run.run_uuid,
            revision=view.run.revision,
            files=files,
            duration_ms=duration_ms,
            segment_count=segment_count,
            cost_total=export.chunks_manifest.get("cost_total"),
            cost_currency=export.chunks_manifest.get("cost_currency"),
            timing_requested=self.timing_options is not None,
            timing_complete=timing is not None,
            quality_requested=self.quality_options is not None,
            quality_complete=quality_complete,
            quality_passed=quality_passed,
        )


def _raw_format_from_receipt(run_root: Path, chunk_id: str, number: int) -> str | None:
    """Return the bounded audio format an on-disk receipt names, or ``None``.

    Only the deterministic ``raw/<chunk-id>.<ext>.receipt.json`` files are read,
    and only their bounded ``format`` field is returned. A missing, unreadable,
    or malformed receipt reports ``None`` so the caller falls back to GET-only
    recovery instead of guessing a format.
    """
    raw_dir = run_root / "raw"
    if not raw_dir.is_dir():
        return None
    for candidate in sorted(raw_dir.glob(f"{chunk_id}.*.receipt.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            candidate_format = payload.get("format")
            if isinstance(candidate_format, str) and candidate_format:
                return candidate_format
    return None


def _find_native_run_in_repository(
    repository: HistoryRepository, canonical_root: str
) -> RunRecord | None:
    """Return the committed native run for a canonical root inside an open database.

    The locked executor re-verifies ownership against its own writable connection
    instead of trusting the pre-lock read. Two native runs on one root cannot be
    told apart, so that is a fail-closed conflict rather than a silent pick.
    """
    rows = repository.find_runs_by_root(canonical_root, limit=MAX_QUERY_LIMIT)
    native = [
        run
        for run in rows
        if run.legacy_source_root is None
        and isinstance(run.config_snapshot, dict)
        and run.config_snapshot.get("operation_origin") == NATIVE_SNAPSHOT_ORIGIN
    ]
    if len(native) > 1:
        raise NativeGenerationError(
            "more than one native history run owns this run directory; refusing to choose one.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_UNVERIFIABLE,
        )
    return native[0] if native else None


def _ensure_native_run_dirs(paths) -> None:
    """Create the native run's output directories after the lock and ownership check.

    The native route must not create a directory tree in a root before it has taken
    the run lock and re-verified ownership, so directory creation happens here
    instead of in the caller before ``run_native_generation`` takes the lock. A
    failure is an output error and leaves the database and audio untouched.
    """
    try:
        paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NativeGenerationError(
            "the native run output directory could not be created.",
            code=_EXIT_OUTPUT,
            error_code=_ERROR_OUTPUT_UNAVAILABLE,
        ) from exc


def execute_native_tts(
    *,
    repository: HistoryRepository,
    run_root: Path,
    paths,
    prepared: PreparedRun,
    chunks: list[ScriptChunk],
    script_format: str,
    script_path: Path,
    output_options: dict[str, Any],
    ffmpeg_path: str,
    ffprobe_path: str,
    user_label: str | None,
    resume: bool,
    provider_factory: Callable[[], Any],
    hooks: NativeExecutionHooks,
    logger: GenerationLogger,
    expected_run_uuid: str | None = None,
    paid_submit_allowed: bool = True,
    local_timing_allowed: bool = True,
    local_quality_allowed: bool = True,
) -> NativeGenerationSummary:
    """Run one locked native TTS generation, resume, or completed-run export.

    The caller has already taken the run lock and opened the writable history
    database. A fresh run commits its prepared snapshot and ownership descriptor
    before any provider exists; a resume proves the exact committed synthesis
    identity and output settings before any paid action; a completed run is only
    re-exported. The provider factory is invoked lazily and never for a local raw
    rebuild or an export-only repair.

    ``expected_run_uuid`` binds the call to one already-committed run: the run
    found for this run directory must be exactly that run, so ``history
    resume``/``history sync`` can neither create a new run nor silently switch to
    another one. ``paid_submit_allowed`` is ``False`` for ``history sync``, which
    then fails closed on an unattempted part instead of submitting it.
    ``local_timing_allowed`` is ``False`` for ``history sync`` so it never runs a
    local timing model; the pending timing work stays for an explicit resume.
    ``local_quality_allowed`` is likewise ``False`` for ``history sync`` so it never
    runs a local quality model either.
    """
    if (
        is_dialogue_format(script_format)
        and prepared.provider == "openrouter-tts"
        and quality_options_from_output_options(output_options) is None
    ):
        # The admitted OpenRouter dialogue route always records the local quality
        # settings it verifies each turn with; a run without them cannot satisfy
        # the required gate. The omnivoice-local preset dialogue route records no
        # quality gate (its legacy route never had one), so it is exempt.
        raise NativeGenerationError(
            "the dialogue native route requires an installed local --tts-quality-provider.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_QUALITY_OPTIONS_INVALID,
        )
    canonical_root = str(run_root.resolve())
    existing = _find_native_run_in_repository(repository, canonical_root)
    if expected_run_uuid is not None and (
        existing is None or existing.run_uuid != expected_run_uuid
    ):
        raise NativeGenerationError(
            "the committed history run for this run directory does not match the run being "
            "resumed or synced; refusing to create or switch a history run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_RUN_MISMATCH,
        )
    if existing is None and not _fresh_run_root_available(run_root):
        raise NativeGenerationError(
            "the run directory already carries local run state; refusing to start a "
            "native run over it.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_OWNERSHIP_RUN_MISSING,
        )
    _ensure_native_run_dirs(paths)
    run_uuid: str
    view: NativeTtsView
    if existing is None:
        script_text = "\n".join(chunk.text for chunk in chunks)
        try:
            snapshot = persist_prepared_tts_snapshot(
                repository,
                prepared=prepared,
                run_root=run_root,
                user_label=user_label,
                script_format=script_format,
                script_text=script_text,
                script_path=script_path,
                output_options=output_options,
            )
            run_uuid = snapshot.run.run_uuid
            _write_descriptor(run_root, run_uuid)
            view = load_native_tts_view(repository, run_uuid)
        except (NativeSnapshotError, NativeViewError) as exc:
            raise NativeGenerationError(
                f"refusing to start a native run: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_SNAPSHOT_FAILED",
            ) from exc
        resume = False
    else:
        run_uuid = existing.run_uuid
        _ensure_descriptor(run_root, run_uuid)
        try:
            view = load_native_tts_view(repository, run_uuid)
        except NativeViewError as exc:
            raise NativeGenerationError(
                f"refusing to resume: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_HISTORY_INTEGRITY",
            ) from exc
        if not resume:
            raise NativeGenerationError(
                "this run is owned by native history and already exists. Use --resume to "
                "continue it, or a different --run-id for a new run.",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_RUN_ALREADY_EXISTS",
            )
        if view.output_options is not None and view.output_options != output_options:
            raise NativeGenerationError(
                "refusing to resume: the recorded output-processing settings differ from this "
                "command.",
                code=_EXIT_PROVIDER,
                error_code=_ERROR_PROCESSING_CHANGED,
            )
        try:
            preflight_native_tts_resume(
                repository,
                run_uuid,
                expected_revision=view.run.revision,
                prepared=prepared,
                script_format=script_format,
                run_root=run_root,
            )
        except (NativeResumeError, NativeViewError) as exc:
            raise NativeGenerationError(
                f"refusing to resume: {exc}",
                code=_EXIT_PROVIDER,
                error_code="NATIVE_RESUME_IDENTITY_CHANGED",
            ) from exc

    executor = _Executor(
        repository=repository,
        view=view,
        paths=paths,
        run_root=run_root,
        prepared=prepared,
        chunks=chunks,
        script_format=script_format,
        script_path=script_path,
        output_options=output_options,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        provider_factory=provider_factory,
        hooks=hooks,
        logger=logger,
        resume=resume,
        paid_submit_allowed=paid_submit_allowed,
        local_timing_allowed=local_timing_allowed,
        local_quality_allowed=local_quality_allowed,
    )
    executor.preflight_timing()
    executor.preflight_quality()
    executor.preflight_local_synthesis()
    executor.run_parts()
    executor.ensure_complete()
    executor.ensure_timing()
    executor.ensure_quality()
    return executor.export()


def run_native_generation(
    *,
    paths,
    prepared: PreparedRun,
    chunks: list[ScriptChunk],
    script_format: str,
    script_path: Path,
    output_options: dict[str, Any],
    ffmpeg_path: str,
    ffprobe_path: str,
    user_label: str | None,
    resume: bool,
    provider_factory: Callable[[], Any],
    hooks: NativeExecutionHooks,
    database_path: Path | None = None,
    expected_run_uuid: str | None = None,
    paid_submit_allowed: bool = True,
    local_timing_allowed: bool = True,
    local_quality_allowed: bool = True,
) -> NativeGenerationSummary:
    """Hold the run lock and the history database around one native execution.

    The generation logger is constructed here, after the run lock is taken, so the
    native route creates no file in the run root before the lock and the ownership
    recheck inside :func:`execute_native_tts`. ``local_timing_allowed`` and
    ``local_quality_allowed`` are threaded to :func:`execute_native_tts` so an
    explicit resume may run the pending local timing or quality verification while
    ``history sync`` never runs a local model.
    """
    try:
        with acquire_run_lock(paths.output_root):
            logger = GenerationLogger(paths.output_root / LOG_FILE)
            with _open_writable_history(database_path) as repository:
                return execute_native_tts(
                    repository=repository,
                    run_root=paths.output_root,
                    paths=paths,
                    prepared=prepared,
                    chunks=chunks,
                    script_format=script_format,
                    script_path=script_path,
                    output_options=output_options,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    user_label=user_label,
                    resume=resume,
                    provider_factory=provider_factory,
                    hooks=hooks,
                    logger=logger,
                    expected_run_uuid=expected_run_uuid,
                    paid_submit_allowed=paid_submit_allowed,
                    local_timing_allowed=local_timing_allowed,
                    local_quality_allowed=local_quality_allowed,
                )
    except HistoryRunLockedError as exc:
        raise NativeGenerationError(
            "another process is already writing this run directory; refusing to run two "
            "writers for one native run.",
            code=_EXIT_PROVIDER,
            error_code="NATIVE_RUN_LOCKED",
        ) from exc


# ── reconstruction from committed history ─────────────────────────────────────


def _stored_identity_str(config: dict[str, Any], key: str) -> str:
    """Return one required stored identity field, or fail closed.

    The verified view already guarantees these fields, so this only narrows the
    type and keeps the failure a bounded native error rather than an unexpected
    exception. The message echoes no stored value.
    """
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise NativeGenerationError(
            "the committed snapshot is missing an identity field a resume needs; refusing to "
            "reconstruct the run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        )
    return value


@dataclass(frozen=True)
class NativeRunReconstruction:
    """The prepared synthesis inputs rebuilt from one committed native view.

    ``prepared`` and ``chunks`` are the exact identity the run's own fingerprint
    commits to, so a later identity preflight either passes unchanged or fails
    closed. ``paths`` is rebuilt with the same run-path builder the fresh route
    uses. ``script_path`` is the stored source path only: it is never read, so a
    deleted or moved original script does not matter.
    """

    prepared: PreparedRun
    chunks: list[ScriptChunk]
    paths: RunPaths
    script_path: Path


def _stored_voice_bank_identity(config: dict[str, Any]) -> OmniVoiceVoiceBankIdentity | None:
    """Return the committed voice-bank identity, or ``None`` when the run has none.

    The verified view already guarantees the committed block parses, so this only
    narrows the type; a run without one (every non-omnivoice route) reports
    ``None`` and its reconstruction stays exactly as before.
    """
    value = config.get("voice_bank")
    if value is None:
        return None
    try:
        return OmniVoiceVoiceBankIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeGenerationError(
            "the committed snapshot voice-bank identity is malformed; refusing to "
            "reconstruct the run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        ) from exc


def _stored_qwen_clone_identity(config: dict[str, Any]) -> QwenCloneVoiceIdentity | None:
    """Return the committed clone identity, or ``None`` when the run has none.

    The verified view already guarantees the committed block parses, so this only
    narrows the type; a run without one (every non-qwen route) reports ``None`` and
    its reconstruction stays exactly as before.
    """
    value = config.get("qwen_clone")
    if value is None:
        return None
    try:
        return QwenCloneVoiceIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeGenerationError(
            "the committed snapshot qwen clone identity is malformed; refusing to "
            "reconstruct the run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        ) from exc


def _stored_qwen_mode_identity(config: dict[str, Any]) -> QwenModeVoiceIdentity | None:
    """Return the committed instructed-mode identity, or ``None`` when absent.

    The verified view already guarantees the committed block parses, so this only
    narrows the type; a run without one (the clone route or every non-qwen route)
    reports ``None`` and its reconstruction stays exactly as before.
    """
    value = config.get("qwen_mode")
    if value is None:
        return None
    try:
        return QwenModeVoiceIdentity.from_payload(value)
    except ValueError as exc:
        raise NativeGenerationError(
            "the committed snapshot qwen mode identity is malformed; refusing to "
            "reconstruct the run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        ) from exc


def reconstruct_native_run(view: NativeTtsView) -> NativeRunReconstruction:
    """Rebuild one committed native run's synthesis inputs from its verified view.

    Every value comes from the committed rows the view already verified and
    fingerprinted: the ordered parts with their exact text, cast voice, speaker,
    and pause, plus the run provider, model, voice, style prompt, and prompt mode.
    The run paths are rebuilt with :func:`voiceover_pipeline.artifacts.build_run_paths`
    from the stored canonical run root, so they match the fresh route exactly. The
    original script file is never opened: only its stored path string is returned
    as the compatibility ``script`` field, and a snapshot that records no source
    path at all fails closed instead of guessing one.
    """
    config = view.run.config_snapshot
    if not isinstance(config, dict):  # pragma: no cover - a verified view guarantees this
        raise NativeGenerationError(
            "the committed snapshot is not a mapping; refusing to reconstruct the run.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        )
    chunks = [
        ScriptChunk(
            number=part.number,
            id=part.chunk_id,
            text=part.text,
            speaker=part.speaker,
            voice=part.cast_voice,
            voice_fingerprint=part.voice_fingerprint,
            pause_after_ms=part.pause_after_ms,
        )
        for part in view.parts
    ]
    model = _stored_identity_str(config, "model")
    voice_bank_identity = _stored_voice_bank_identity(config)
    qwen_clone_identity = _stored_qwen_clone_identity(config)
    qwen_mode_identity = _stored_qwen_mode_identity(config)
    prepared = PreparedRun(
        provider=_stored_identity_str(config, "provider"),
        model=model,
        voice=_stored_identity_str(config, "voice"),
        style_prompt=view.style_prompt,
        prompt_mode=_stored_identity_str(config, "prompt_mode"),
        parts=tuple(PreparedPart(chunk=chunk, voice=chunk.voice) for chunk in chunks),
        voice_bank_identity=voice_bank_identity,
        qwen_clone_identity=qwen_clone_identity,
        qwen_mode_identity=qwen_mode_identity,
    )
    run_root = Path(view.run.run_root)
    paths = build_run_paths(run_root.parent, model, run_root.name)
    script_path_value = view.script_path
    if script_path_value is None:
        raise NativeGenerationError(
            "the committed snapshot records no source script path for its compatibility export; "
            "refusing to guess one.",
            code=_EXIT_PROVIDER,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        )
    return NativeRunReconstruction(
        prepared=prepared,
        chunks=chunks,
        paths=paths,
        script_path=Path(script_path_value),
    )


def _completed_final_verified(view: NativeTtsView, hooks: NativeExecutionHooks) -> bool:
    """Whether a completed run's one final audio artifact is present and unchanged.

    The export repair is the only work a completed history run may do, so a
    completed run whose final audio is missing or tampered must fail closed
    instead of re-running FFmpeg assembly over committed evidence.
    """
    finals = [artifact for artifact in view.artifacts if artifact.role == ARTIFACT_ROLE_FINAL_AUDIO]
    if len(finals) != 1:
        return False
    final = finals[0]
    if final.sha256 is None or final.size_bytes is None:
        return False
    path = Path(view.run.run_root) / final.path
    if not path.is_file():
        return False
    try:
        return path.stat().st_size == final.size_bytes and hooks.sha256_file(path) == final.sha256
    except OSError:
        return False


def run_native_history_generation(
    *,
    view: NativeTtsView,
    mode: str,
    provider_builder: Callable[[PreparedRun], Any],
    hooks: NativeExecutionHooks,
    ffmpeg_path: str,
    ffprobe_path: str,
    database_path: Path | None = None,
) -> NativeGenerationSummary:
    """Resume or sync one committed native run reconstructed from its snapshot.

    ``mode`` is ``"resume"`` or ``"sync"``. The synthesis inputs, run paths, script
    format, and script path are rebuilt from ``view`` alone through
    :func:`reconstruct_native_run`, so the original script file is never re-read.
    ``provider_builder`` is invoked lazily and only when the executor actually needs
    a provider, so a completed-run export repair or a local raw rebuild reads no API
    key.

    ``history resume`` may submit a truly unattempted part, so it is potentially
    paid; it still skips every verified part, finishes a known Media task id with
    GET calls only, rebuilds a part locally from verified raw evidence, and may run
    the pending local timing or quality verification.
    ``history sync`` never starts a new paid submit: it repairs the compatibility
    exports of a completed run, finishes a known Media id with GET calls only, and
    rebuilds a part locally from verified raw evidence, but fails closed on an
    unattempted part or an unconfirmed submit before any provider or key exists, and
    runs no local model, so a pending timing or quality verification stays
    incomplete for an explicit resume. A quality verification already recorded as a
    failure is reported by ``history sync`` as that persisted verdict
    (``quality.passed: false``) without running the model.

    A completed run whose committed final audio is missing or changed fails closed
    before any write, so the export repair never re-runs assembly over committed
    evidence. The call is bound to ``view.run.run_uuid`` so it can neither create a
    new run nor switch to another run on the same run root.
    """
    if mode not in (_MODE_RESUME, _MODE_SYNC):
        raise NativeGenerationError(
            "history mode must be 'resume' or 'sync'.",
            code=_EXIT_ARGS,
            error_code=_ERROR_HISTORY_INCOMPLETE,
        )
    reconstruction = reconstruct_native_run(view)
    if view.run.status == "completed" and not _completed_final_verified(view, hooks):
        raise NativeGenerationError(
            "refusing to resume or sync: this run is recorded as completed but its final audio "
            "is missing or does not match the committed history. Restore the final audio and "
            "retry.",
            code=_EXIT_OUTPUT,
            error_code=_ERROR_FINAL_AUDIO_MISSING,
        )
    provider_cache: list[Any] = []

    def provider_factory() -> Any:
        if not provider_cache:
            provider_cache.append(provider_builder(reconstruction.prepared))
        return provider_cache[0]

    output_options = (
        view.output_options
        if view.output_options is not None
        else build_output_options(no_trim=False)
    )
    return run_native_generation(
        paths=reconstruction.paths,
        prepared=reconstruction.prepared,
        chunks=reconstruction.chunks,
        script_format=view.script_format,
        script_path=reconstruction.script_path,
        output_options=output_options,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        user_label=view.run.user_label,
        resume=True,
        provider_factory=provider_factory,
        hooks=hooks,
        database_path=database_path,
        expected_run_uuid=view.run.run_uuid,
        paid_submit_allowed=(mode == _MODE_RESUME),
        local_timing_allowed=(mode == _MODE_RESUME),
        local_quality_allowed=(mode == _MODE_RESUME),
    )
