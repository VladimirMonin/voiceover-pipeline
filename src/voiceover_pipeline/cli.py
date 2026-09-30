import argparse
import glob as glob_mod
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time  # noqa: F401 - shared sleep seam tests patch via cli.time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

from . import settings as settings_module
from .artifacts import (
    build_run_paths,
    build_srt,
    build_timing_manifest,
    write_json,
)
from .asr_longform import (
    LongFormASRMediaError,
    transcribe_prerecorded_long_form,
    uses_long_form_orchestration,
)
from .commands import history as history_commands
from .commands.split import ScriptNotFoundError, prepare_split_chunks
from .config import (
    DEFAULT_ASR_COMPUTE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_FALLBACK_VOICE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVIDER,
    DEFAULT_SCRIPT_DIR,
    DEFAULT_TIMING_COMPUTE,
    DEFAULT_TIMING_DEVICE,
    DEFAULT_TIMING_LANGUAGE,
    DEFAULT_TIMING_PROVIDER,
    ELEVENLABS_TTS_VOICES,
    GEMINI_TTS_VOICES,
    OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
    OMNIVOICE_DEFAULT_LANGUAGE,
    OMNIVOICE_DEFAULT_SEED,
    OMNIVOICE_DEFAULT_STEPS,
    OMNIVOICE_LOCAL_MODEL_ID,
    OMNIVOICE_STYLE_CONDITION,
    OPENAI_TTS_VOICES,
    OPENROUTER_TTS_MODELS,
    PODCAST_NARRATION_PROMPT,
    POLZA_TTS_MODELS,
    PROVIDER_DEFAULT_MODELS,
    QWEN_INSTRUCT,
    QWEN_LANGUAGE,
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
    QWEN_MODEL_VOICE_DESIGN,
    QWEN_PRESET_SPEAKERS,
    read_groq_key,
    read_openrouter_key,
    read_polza_key,
    read_xai_key,
)
from .gemini_dialogue import (
    DIALOGUE_FORMAT,
    GEMINI_DIALOGUE_FORMAT,
    GEMINI_TTS_MODEL,
    chunks_from_validation,
    dialogue_turns_from_validation,
    is_dialogue_format,
    validate_gemini_dialogue_file,
)
from .history import paid_transcription as paid_transcription_history
from .history.locking import HistoryRunLockedError, acquire_run_lock
from .history.native_asr import (
    ARTIFACT_ROLE_QUALITY_RECEIPT,
    ARTIFACT_ROLE_SRT,
    ARTIFACT_ROLE_TIMINGS_JSON,
    ATTEMPT_CALL_TYPE_ASR,
    ATTEMPT_CALL_TYPE_TIMING,
    ATTEMPT_CALL_TYPE_VERIFY,
    NATIVE_ASR_ORIGIN,
    OPERATION_ASR,
    OPERATION_TIMINGS,
    OPERATION_VERIFY,
    AsrHistoryArtifact,
    AsrHistorySave,
    AsrHistorySaveResult,
    AsrHistoryText,
    failed_history_result,
    inactive_history_result,
    persist_asr_history,
    saved_history_result,
)
from .history.repository import (
    DEFAULT_QUERY_LIMIT,
    RUN_STATUS_COMPLETED,
    TEXT_COMPLETENESS_COMPLETE,
    TEXT_COMPLETENESS_INCOMPLETE,
    TEXT_KIND_ASR_CONTEXT,
    TEXT_KIND_ASR_TRANSCRIPT,
    TEXT_KIND_VERIFICATION_TRANSCRIPT,
)
from .local_runtime.contracts import OmniVoiceRequest
from .local_tts_text import merge_omnivoice_session_fragments, prepare_local_tts_chunks
from .media import (
    check_media_tools,
    concat_audio_files,
    concat_dialogue_turns,
    concat_mp3_chunks,
    mp3_duration_ms,
    trim_final_silence,
    write_audio_as_mp3,
)
from .models import (
    ASRContextHints,
    ASRResult,
    ChunkArtifact,
    ScriptChunk,
    SynthesisResult,
)
from .omnivoice_design import (
    OMNIVOICE_LONG_FORM_THRESHOLD_SECONDS,
    evaluate_omnivoice_design_route,
    normalize_omnivoice_design_instruction,
)
from .omnivoice_voice_bank import (
    VoiceBankCatalog,
    VoiceBankError,
    load_voice_bank,
)
from .pricing import (
    fetch_openrouter_generation_detail,
    fetch_openrouter_model_pricing,
    fetch_polza_generation_detail,
    fetch_polza_model_pricing,
)
from .providers import (
    OmniVoiceLocalTTSProvider,
    PolzaTTSProvider,
    TTSProvider,
)
from .providers.asr_registry import (
    ASRProviderNotFoundError,
    get_asr_provider_spec,
    list_asr_provider_specs,
)
from .providers.audio_cpp_omnivoice_tts import omnivoice_local_dependency_probe
from .retry import is_retryable_error
from .run_state import (
    ATTEMPT_FAILED,
    ATTEMPT_OUTCOME_UNKNOWN,
    LOG_FILE,
    STATE_FILE,
    GenerationLogger,
    atomic_write_json,
    completed_numbers,
    load_state,
    script_hash,
    unconfirmed_attempt,
    upsert_completed_chunk,
)
from .script_splitter import split_markdown_by_delimiter
from .services import (
    cost_enrichment,
    costs,
    execution,
    finalization,
    native_generation,
    provider_factory,
    recovery,
    transcription,
)
from .services.prepare import (
    OMNIVOICE_AUTO_MODE,
    OMNIVOICE_CLONE_MODE,
    OMNIVOICE_DESIGN_MODE,
    OMNIVOICE_MODE_IDENTITY_MODES,
    PreparationError,
    PreparedRun,
    bind_omnivoice_dialogue_fingerprints,
    build_omnivoice_mode_identity,
    build_qwen_clone_identity,
    build_qwen_mode_identity,
    default_voice,
    prepare_generation_identity,
    prepare_run,
    prepare_runtime_chunks,
    prepare_script_fragments,
    resolve_script_format,
)
from .services.synthesis import synthesize_part
from .services.transcription import build_asr_request
from .tts_prompting import read_style_prompt_from_file
from .tts_quality import evaluate_tts_transcript
from .voiceover_script import (
    VOICEOVER_FORMAT,
    chunks_from_voiceover_report,
    validate_voiceover_file,
)

_EXIT_OK = 0
_EXIT_ARGS = 2
_EXIT_MISSING_DEP = 10
_EXIT_NO_FFMPEG = 11
_EXIT_NO_KEY = 20
_EXIT_PROVIDER = 30
_EXIT_WHISPER = 40
_EXIT_OUTPUT = 50
_EXIT_QUALITY = 60


# ═══════════════════════════════════════════════════════════════════════════════
# CliError
# ═══════════════════════════════════════════════════════════════════════════════


class CliError(RuntimeError):
    def __init__(self, message: str, code: int, *, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details


def gemini_chunks_from_validation(report: dict[str, Any]) -> list[ScriptChunk]:
    """Compatibility alias for callers that still inspect section chunks."""
    return chunks_from_validation(report)


def fail(message: str, code: int, *, details: dict[str, object] | None = None) -> NoReturn:
    raise CliError(message, code, details=details)


def _find_default_script() -> Path:
    candidates = [
        DEFAULT_SCRIPT_DIR / "podcast_script_raw.txt",
        DEFAULT_SCRIPT_DIR / "script.md",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = build_parser()
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        if exc.code == _EXIT_ARGS and "--json" in sys.argv[1:]:
            _json_error("Invalid command-line arguments", _EXIT_ARGS)
        raise

    try:
        if args.command == "generate":
            generate(args)
        elif args.command == "split":
            split_cmd(args)
        elif args.command == "transcribe":
            transcribe_cmd(args)
        elif args.command == "verify-tts":
            verify_tts_cmd(args)
        elif args.command == "timings":
            run_timings(args)
        elif args.command == "status":
            status_cmd(args)
        elif args.command == "concat":
            concat_cmd(args)
        elif args.command == "doctor":
            doctor_cmd(args)
        elif args.command == "validate":
            validate_cmd(args)
        elif args.command == "list":
            list_cmd(args)
        elif args.command == "history":
            history_cmd(args)
    except CliError as exc:
        _emit_error(args, str(exc), exc.code, details=exc.details)
    except SystemExit:
        raise
    except Exception as exc:
        _emit_error(args, str(exc), _EXIT_PROVIDER)


# ═══════════════════════════════════════════════════════════════════════════════
# parser
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voiceover + Whisper timing CLI for agents.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # --------------- generate ---------------
    gen = subparsers.add_parser(
        "generate", help="Generate chunk MP3 + full MP3 + optional timings."
    )
    gen.add_argument(
        "--provider",
        choices=[
            "polza-chat-audio",
            "polza-tts",
            "openrouter-tts",
            "qwen-local",
            "omnivoice-local",
        ],
        default=None,
    )
    gen.add_argument("--model", default=argparse.SUPPRESS)
    gen.add_argument("--script", type=Path, default=_find_default_script())
    gen.add_argument("--delimiter", default="******")
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gen.add_argument("--run-id", default="")
    gen.add_argument("--voice", default=None)
    gen.add_argument(
        "--format",
        choices=["markdown", VOICEOVER_FORMAT, DIALOGUE_FORMAT, GEMINI_DIALOGUE_FORMAT],
        default="markdown",
    )
    gen.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Validation limit for voiceover metadata scripts.",
    )
    gen.add_argument(
        "--speaker-voice",
        action="append",
        default=[],
        help="Gemini dialogue voice mapping, e.g. Speaker1=Puck. Can repeat.",
    )
    gen.add_argument("--fallback-voice", default=DEFAULT_FALLBACK_VOICE)
    gen.add_argument("--style-prompt", default=None)
    gen.add_argument("--style-prompt-file", type=Path, default=None)
    gen.add_argument("--no-style-prompt", action="store_true")
    gen.add_argument("--no-trim", action="store_true")
    gen.add_argument(
        "--json", dest="json_output", action="store_true", help="Output JSON to stdout."
    )
    gen.add_argument("--json-events", action="store_true", help="Emit progress events as NDJSON.")
    gen.add_argument("--overwrite", action="store_true", help="Overwrite existing run folder.")
    gen.add_argument(
        "--confirm-delete-paid-audio",
        action="store_true",
        help="Allow --overwrite to delete existing chunk audio.",
    )
    gen.add_argument("--skip-existing", action="store_true", help="Skip if run folder exists.")
    gen.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run without regenerating completed chunks.",
    )
    gen.add_argument(
        "--retries", type=int, default=3, help="Attempts per chunk for retryable provider errors."
    )
    gen.add_argument(
        "--retry-delay", type=float, default=2.0, help="Initial retry delay in seconds."
    )
    gen.add_argument(
        "--retry-max-delay", type=float, default=30.0, help="Maximum retry delay in seconds."
    )
    gen.add_argument("--no-retry", action="store_true", help="Disable retry attempts.")
    gen.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Generate only the first N chunks for a test run.",
    )
    gen.add_argument(
        "--dry-run-cost",
        action="store_true",
        help="Validate and estimate generation scope without TTS calls.",
    )

    qwen = gen.add_argument_group("qwen-local options")
    qwen.add_argument("--mode", choices=["preset", "auto", "clone", "design"], default="preset")
    qwen.add_argument(
        "--qwen-instruct",
        default=None,
        help="Per-run speaking style instruction for qwen-local preset or design mode.",
    )
    qwen.add_argument("--sample", type=str, default=None)
    qwen.add_argument("--sample-text", type=str, default=None)

    omnivoice = gen.add_argument_group("omnivoice-local options")
    omnivoice.add_argument("--reference-audio", type=Path, default=None)
    omnivoice.add_argument("--reference-text", type=str, default=None)
    omnivoice.add_argument(
        "--design-instruction",
        type=str,
        default=None,
        help=(
            "Voice Design instruction. Non-English/non-Chinese speech is experimental up to "
            "the 30-second threshold and rejected above it; choose clone, preset, short "
            "experimental clips, or another provider."
        ),
    )
    omnivoice.add_argument(
        "--voice-bank",
        type=Path,
        default=None,
        help="Path to voice bank catalog.json for omnivoice-local --mode preset.",
    )

    tim = gen.add_argument_group("Whisper timing (optional)")
    tim.add_argument("--with-timings", action="store_true")
    tim.add_argument(
        "--timing-provider",
        default=DEFAULT_TIMING_PROVIDER,
        choices=["faster-whisper", "openrouter-whisper", "groq-whisper", "xai-stt"],
        help="Transcription provider (default: faster-whisper)",
    )
    tim.add_argument(
        "--timing-model", default=None, help="Provider-specific model for transcription"
    )
    tim.add_argument(
        "--timing-device", default=DEFAULT_TIMING_DEVICE, choices=["auto", "cpu", "cuda"]
    )
    tim.add_argument(
        "--timing-compute",
        default=DEFAULT_TIMING_COMPUTE,
        choices=["auto", "int8", "int8_float16", "float16", "float32"],
    )
    tim.add_argument("--timing-language", default=DEFAULT_TIMING_LANGUAGE)
    tim.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps (faster-whisper + groq-whisper; openrouter-whisper ignores with a warning).",
    )

    quality = gen.add_argument_group("Dialogue TTS quality gate")
    quality.add_argument(
        "--tts-quality-provider",
        default=None,
        help="Explicit ASR provider used to verify every dialogue turn before concat.",
    )
    quality.add_argument("--tts-quality-model", default=None)
    quality.add_argument("--tts-quality-language", default=None)
    quality.add_argument("--tts-quality-device", default=DEFAULT_ASR_DEVICE)
    quality.add_argument("--tts-quality-compute", default=DEFAULT_ASR_COMPUTE)
    quality.add_argument(
        "--tts-quality-runtime", choices=["auto", "python", "audio-cpp"], default="auto"
    )

    # --------------- split ---------------
    spl = subparsers.add_parser("split", help="Print chunk ids and character counts.")
    spl.add_argument("--script", type=Path, default=_find_default_script())
    spl.add_argument("--delimiter", default="******")
    spl.add_argument("--json", dest="json_output", action="store_true")

    # --------------- transcribe ---------------
    asr = subparsers.add_parser(
        "transcribe", help="Transcribe finite audio with a registered local ASR provider."
    )
    asr.add_argument("--audio", type=str, required=True)
    asr.add_argument("--provider", required=True, help="Registered ASR provider ID.")
    asr.add_argument("--model", default=None, help="Provider-specific ASR model ID.")
    asr.add_argument("--language", default=None, help="Optional forced language.")
    asr.add_argument(
        "--device",
        default=DEFAULT_ASR_DEVICE,
        help="Requested device, validated against provider capabilities.",
    )
    asr.add_argument(
        "--compute",
        default=DEFAULT_ASR_COMPUTE,
        help="Requested compute mode, validated against provider capabilities.",
    )
    asr.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Require validated word timestamps from the selected ASR provider.",
    )
    context_group = asr.add_mutually_exclusive_group()
    context_group.add_argument("--context", default=None, help="Optional ASR context text.")
    context_group.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Read optional ASR context text from a file.",
    )
    asr.add_argument(
        "--runtime",
        choices=["auto", "python", "audio-cpp"],
        default="auto",
        help="Requested ASR runtime route.",
    )
    asr.add_argument("--json", dest="json_output", action="store_true")

    # --------------- verify-tts ---------------
    verify_tts = subparsers.add_parser(
        "verify-tts",
        help="Fail closed on major TTS omissions, unexpected speech, or repetition.",
    )
    verify_tts.add_argument("--audio", type=str, required=True)
    expected_group = verify_tts.add_mutually_exclusive_group(required=True)
    expected_group.add_argument("--expected-text", default=None)
    expected_group.add_argument("--expected-file", type=Path, default=None)
    verify_tts.add_argument("--provider", required=True, help="Registered ASR provider ID.")
    verify_tts.add_argument("--model", default=None, help="Provider-specific ASR model ID.")
    verify_tts.add_argument("--language", default=None, help="Optional forced language.")
    verify_tts.add_argument("--device", default=DEFAULT_ASR_DEVICE)
    verify_tts.add_argument("--compute", default=DEFAULT_ASR_COMPUTE)
    verify_tts.add_argument("--runtime", choices=["auto", "python", "audio-cpp"], default="auto")
    verify_tts.add_argument("--receipt", type=Path, default=None)
    verify_tts.add_argument("--json", dest="json_output", action="store_true")

    # --------------- timings ---------------
    timp = subparsers.add_parser("timings", help="Extract Whisper timings from audio.")
    timp.add_argument("--audio", type=str, required=True)
    timp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    timp.add_argument("--run-id", default="")
    timp.add_argument(
        "--timing-provider",
        default=DEFAULT_TIMING_PROVIDER,
        choices=["faster-whisper", "openrouter-whisper", "groq-whisper", "xai-stt"],
        help="Transcription provider (default: faster-whisper)",
    )
    timp.add_argument(
        "--model", default=None, help="Provider-specific model (e.g. openai/whisper-large-v3-turbo)"
    )
    timp.add_argument("--device", default=DEFAULT_TIMING_DEVICE, choices=["auto", "cpu", "cuda"])
    timp.add_argument(
        "--compute",
        default=None,
        choices=["auto", "int8", "int8_float16", "float16", "float32", "bfloat16"],
    )
    timp.add_argument("--language", default=DEFAULT_TIMING_LANGUAGE)
    timp.add_argument(
        "--asr-provider",
        default=None,
        help="Optional registered ASR provider for generic word-timestamp artifacts.",
    )
    timp.add_argument("--json", dest="json_output", action="store_true")
    timp.add_argument("--word-timestamps", action="store_true")
    timp.add_argument("--overwrite", action="store_true")
    timp.add_argument("--skip-existing", action="store_true", help="Skip if output dir exists.")

    # --------------- status ---------------
    stat = subparsers.add_parser("status", help="Show resumable generation status for a run.")
    stat.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    stat.add_argument("--run-id", required=True)
    stat.add_argument("--json", dest="json_output", action="store_true")

    # --------------- concat ---------------
    con = subparsers.add_parser(
        "concat", help="Concatenate existing chunks, including partial runs."
    )
    con.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    con.add_argument("--run-id", required=True)
    con.add_argument("--format", choices=["mp3", "ogg"], default="ogg")
    con.add_argument("--json", dest="json_output", action="store_true")

    # --------------- doctor ---------------
    doc = subparsers.add_parser("doctor", help="Check environment and dependencies.")
    doc.add_argument("--json", dest="json_output", action="store_true")
    doc.add_argument(
        "--provider",
        default=None,
        choices=[
            "polza-chat-audio",
            "polza-tts",
            "openrouter-tts",
            "qwen-local",
            "omnivoice-local",
        ],
        help="Check provider-specific requirements.",
    )
    doc.add_argument("--with-timings", action="store_true", help="Check timing dependencies.")
    doc.add_argument(
        "--timing-provider",
        default="faster-whisper",
        choices=["faster-whisper", "openrouter-whisper", "groq-whisper", "xai-stt"],
        help="Timing provider to check (default: faster-whisper)",
    )
    doc.add_argument(
        "--timing-device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Requested timing device for dependency check.",
    )
    doc.add_argument(
        "--with-asr",
        action="store_true",
        help="Check a registered local ASR provider dependency boundary.",
    )
    doc.add_argument(
        "--asr-provider", default=None, help="ASR provider ID to check with --with-asr."
    )
    doc.add_argument(
        "--asr-device",
        default=DEFAULT_ASR_DEVICE,
        help="Requested ASR device for the selected provider.",
    )
    doc.add_argument(
        "--asr-compute",
        default=DEFAULT_ASR_COMPUTE,
        help="Requested ASR compute mode for the selected provider.",
    )

    # --------------- validate ---------------
    val = subparsers.add_parser("validate", help="Validate script for generation.")
    val.add_argument("--script", type=Path, required=True)
    val.add_argument("--delimiter", default="******")
    val.add_argument(
        "--format",
        choices=["markdown", VOICEOVER_FORMAT, DIALOGUE_FORMAT, GEMINI_DIALOGUE_FORMAT],
        default="markdown",
    )
    val.add_argument(
        "--provider",
        choices=[
            "polza-chat-audio",
            "polza-tts",
            "openrouter-tts",
            "qwen-local",
            "omnivoice-local",
        ],
        default=None,
    )
    val.add_argument("--model", default=None)
    val.add_argument("--voice", default=None)
    val.add_argument(
        "--speaker-voice",
        action="append",
        default=[],
        help="Gemini dialogue voice mapping, e.g. Speaker1=Puck. Can repeat.",
    )
    val.add_argument(
        "--agent", action="store_true", help="Include agent-oriented snippets and suggested fixes."
    )
    val.add_argument("--max-chunk-chars", type=int, default=None)
    val.add_argument("--json", dest="json_output", action="store_true")

    # --------------- list ---------------
    lst = subparsers.add_parser("list", help="List available providers, voices, or timing models.")
    lst.add_argument(
        "target",
        choices=["providers", "voices", "timing-models", "timing-providers", "asr-providers"],
    )
    lst.add_argument("--provider", default=None, help="Filter voices by provider.")
    lst.add_argument(
        "--voice-bank",
        type=Path,
        default=None,
        help="Path to voice bank catalog.json for omnivoice-local voice listing.",
    )
    lst.add_argument("--json", dest="json_output", action="store_true")

    # --------------- history ---------------
    hist = subparsers.add_parser(
        "history", help="Read and import the local SQLite run history (not CWD-bound)."
    )
    hist_sub = hist.add_subparsers(dest="history_command")
    hist_sub.required = True

    hist_list = hist_sub.add_parser("list", help="List run metadata from local history.")
    hist_list.add_argument("--label", default=None, help="Exact user-label filter.")
    hist_list.add_argument("--operation", default=None, help="Exact operation filter.")
    hist_list.add_argument("--status", default=None, help="Exact status filter.")
    hist_list.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    hist_list.add_argument("--offset", type=int, default=0)
    hist_list.add_argument("--json", dest="json_output", action="store_true")

    hist_show = hist_sub.add_parser("show", help="Show one run metadata by UUID or label.")
    hist_show.add_argument("run", metavar="ID")
    hist_show.add_argument("--json", dest="json_output", action="store_true")

    hist_resume = hist_sub.add_parser(
        "resume",
        help=(
            "Resume one committed native TTS run from its stored snapshot (potentially paid: "
            "an unattempted part may be submitted)."
        ),
        description=(
            "Resume one committed native TTS run from its stored snapshot. The original script "
            "file is not re-read. This is potentially paid: an unattempted part may be submitted "
            "in the normal order, while a known remote id is finished with GET calls only and a "
            "verified raw receipt is rebuilt locally. --overwrite is never accepted."
        ),
    )
    hist_resume.add_argument("run", metavar="ID", help="Internal UUID of a native TTS run.")
    hist_resume.add_argument("--json", dest="json_output", action="store_true")

    hist_sync = hist_sub.add_parser(
        "sync",
        help=(
            "Retrieve the stored state/result of a known native TTS run without a new paid submit."
        ),
        description=(
            "Retrieve the stored state/result of a known native TTS run without a new paid "
            "submit: repair the compatibility JSON exports of a completed run, rebuild a part "
            "from verified raw evidence, or finish a known remote id with GET calls only. An "
            "unattempted part or an unconfirmed submit blocks before any provider or key."
        ),
    )
    hist_sync.add_argument("run", metavar="ID", help="Internal UUID of a native TTS run.")
    hist_sync.add_argument("--json", dest="json_output", action="store_true")

    hist_import = hist_sub.add_parser("import", help="Import legacy out/<run-id> trees.")
    hist_import.add_argument("source", metavar="DIR")
    hist_import.add_argument(
        "--dry-run", action="store_true", help="Preview an import without writing anything."
    )
    hist_import.add_argument("--json", dest="json_output", action="store_true")

    hist_costs = hist_sub.add_parser(
        "costs", help="Read-only money totals grouped by currency and operation."
    )
    hist_costs.add_argument("--json", dest="json_output", action="store_true")

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# generate
# ═══════════════════════════════════════════════════════════════════════════════

_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _validate_run_id(run_id: str) -> str:
    stripped = run_id.strip()
    if not stripped:
        fail("--run-id must not be empty or whitespace-only", _EXIT_ARGS)
    if run_id != stripped:
        fail("--run-id must not have leading or trailing whitespace", _EXIT_ARGS)
    if run_id[-1] in (" ", "."):
        fail("--run-id must not end with a space or dot", _EXIT_ARGS)
    for ch in run_id:
        if ch in '<>:"|?*\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f':
            fail(f"Invalid --run-id: illegal character '{ch}'", _EXIT_ARGS)
    if "/" in run_id or "\\" in run_id:
        fail(f"Invalid --run-id: path separators not allowed: {run_id}", _EXIT_ARGS)
    if run_id in (".", ".."):
        fail("Invalid --run-id: '.' and '..' not allowed", _EXIT_ARGS)
    if Path(run_id).is_absolute():
        fail(f"Invalid --run-id: absolute paths are not allowed: {run_id}", _EXIT_ARGS)
    normalized = run_id.rstrip(" .").upper()
    if normalized in _RESERVED_WINDOWS_NAMES:
        fail(f"Invalid --run-id: Windows reserved name not allowed: {run_id}", _EXIT_ARGS)
    return run_id


def _safe_remove_run_dir(directory: Path, output_dir: Path | None = None) -> None:
    resolved = directory.resolve()
    if resolved == Path.cwd().resolve():
        fail(f"Refusing to remove current working directory: {resolved}", _EXIT_OUTPUT)
    root = resolved.anchor or "C:\\"
    if str(resolved).rstrip("\\/") == root.rstrip("\\/"):
        fail(f"Refusing to remove drive root: {resolved}", _EXIT_OUTPUT)
    if resolved == Path.home().resolve():
        fail(f"Refusing to remove home directory: {resolved}", _EXIT_OUTPUT)
    if output_dir is not None:
        base = output_dir.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            fail(
                f"Refusing to remove directory outside output-dir: {resolved} (output-dir: {base})",
                _EXIT_OUTPUT,
            )
    shutil.rmtree(resolved)


def _ensure_run_dirs(paths) -> None:
    try:
        paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Failed to create output directory {paths.chunks_dir}: {e}", _EXIT_OUTPUT)


def _looks_like_windows_drive_root(path: Path) -> bool:
    """Detect values like ``C:\\`` even when running on POSIX.

    On Linux, ``Path("C:\\")`` is a relative directory named ``C:\\``, so
    ``Path.anchor`` is empty and the normal root check does not catch it.
    The CLI still treats Windows drive roots as unsafe because scripts and
    tests may be authored cross-platform.
    """
    value = str(path).strip().replace("/", "\\")
    return len(value) == 3 and value[1:] == ":\\" and value[0].isalpha()


def _validate_output_dir(output_dir: Path) -> Path:
    if _looks_like_windows_drive_root(output_dir):
        fail(f"--output-dir cannot be a drive root: {output_dir}", _EXIT_ARGS)

    resolved = output_dir.resolve()
    if resolved == Path.cwd().resolve():
        fail(f"--output-dir cannot be the current working directory: {resolved}", _EXIT_ARGS)
    home = Path.home().resolve()
    if resolved == home:
        fail(f"--output-dir cannot be your home directory: {resolved}", _EXIT_ARGS)
    root = resolved.anchor or "C:\\\\"
    if str(resolved).rstrip("\\\\/") == root.rstrip("\\\\/"):
        fail(f"--output-dir cannot be a drive root: {resolved}", _EXIT_ARGS)
    return output_dir


def _resolve_style_prompt(args: argparse.Namespace) -> str | None:
    if getattr(args, "no_style_prompt", False):
        return None
    if getattr(args, "style_prompt_file", None):
        return read_style_prompt_from_file(args.style_prompt_file)
    if getattr(args, "style_prompt", None) is not None:
        return args.style_prompt
    return PODCAST_NARRATION_PROMPT


def _validate_omnivoice_voice_bank(args: argparse.Namespace) -> VoiceBankCatalog:
    """Load the preset-mode voice bank and resolve the requested profile.

    Attaches the resolved catalog and profile to ``args`` so later CLI
    stages reuse the same identity without re-reading the bank.
    """
    voice_bank_arg = getattr(args, "voice_bank", None)
    if voice_bank_arg is None:
        fail(
            "omnivoice-local preset mode requires --voice-bank catalog.json",
            _EXIT_ARGS,
        )
    catalog_path = Path(voice_bank_arg)
    try:
        catalog = load_voice_bank(catalog_path)
    except VoiceBankError as exc:
        fail(str(exc), _EXIT_ARGS)
    requested_voice = getattr(args, "voice", None)
    voice_id = requested_voice if requested_voice is not None else catalog.default_voice
    profile = next((item for item in catalog.profiles if item.id == voice_id), None)
    if profile is None:
        fail(f"voice '{voice_id}' not found in the voice bank", _EXIT_ARGS)
    args.voice_bank_catalog = catalog
    args.voice_bank_profile = profile
    return catalog


def _resolve_provider_style_prompt(args: argparse.Namespace) -> str | None:
    if getattr(args, "provider", None) == "openrouter-tts":
        return None
    if getattr(args, "provider", None) == "omnivoice-local":
        return None
    if getattr(args, "provider", None) == "qwen-local":
        instruct = getattr(args, "qwen_instruct", None)
        return QWEN_INSTRUCT if instruct is None else instruct
    return _resolve_style_prompt(args)


def _validate_omnivoice_options(args: argparse.Namespace) -> None:
    if getattr(args, "provider", None) != "omnivoice-local":
        return

    mode = getattr(args, "mode", "preset")
    if mode not in ("preset", "auto", "clone", "design"):
        fail("omnivoice-local mode must be preset, auto, clone, or design", _EXIT_ARGS)

    reference_audio = getattr(args, "reference_audio", None)
    reference_text = getattr(args, "reference_text", None)
    design_instruction = getattr(args, "design_instruction", None)
    unsupported: list[str] = []
    if getattr(args, "sample", None) is not None:
        unsupported.append("--sample")
    if getattr(args, "sample_text", None) is not None:
        unsupported.append("--sample-text")
    if getattr(args, "qwen_instruct", None) is not None:
        unsupported.append("--qwen-instruct")
    if getattr(args, "style_prompt", None) is not None:
        unsupported.append("--style-prompt")
    if getattr(args, "style_prompt_file", None) is not None:
        unsupported.append("--style-prompt-file")
    if getattr(args, "no_style_prompt", False):
        unsupported.append("--no-style-prompt")
    voice = getattr(args, "voice", None)
    if mode != "preset" and voice is not None:
        unsupported.append("--voice")
    if getattr(args, "fallback_voice", DEFAULT_FALLBACK_VOICE) != DEFAULT_FALLBACK_VOICE:
        unsupported.append("--fallback-voice")
    dialogue_format = is_dialogue_format(getattr(args, "format", "markdown"))
    if getattr(args, "speaker_voice", []) and not dialogue_format:
        unsupported.append("--speaker-voice")
    if unsupported:
        fail(
            "omnivoice-local rejects unsupported Qwen options, voice controls, and style controls: "
            + ", ".join(unsupported),
            _EXIT_ARGS,
        )

    if mode in ("preset", "auto"):
        fields = [
            flag
            for flag, value in (
                ("--reference-audio", reference_audio),
                ("--reference-text", reference_text),
                ("--design-instruction", design_instruction),
            )
            if value is not None
        ]
        if fields:
            label = "auto" if mode == "auto" else "preset/fixed-style"
            fail(
                f"omnivoice-local {label} rejects clone/design fields: " + ", ".join(fields),
                _EXIT_ARGS,
            )
        if mode == "preset":
            if dialogue_format:
                voice_bank_arg = getattr(args, "voice_bank", None)
                if voice_bank_arg is None:
                    fail(
                        "omnivoice-local dialogue preset mode requires --voice-bank catalog.json",
                        _EXIT_ARGS,
                    )
                try:
                    args.voice_bank_catalog = load_voice_bank(Path(voice_bank_arg))
                except VoiceBankError as exc:
                    fail(str(exc), _EXIT_ARGS)
            else:
                _validate_omnivoice_voice_bank(args)
            try:
                OmniVoiceRequest(mode="fixed-style", style_condition=OMNIVOICE_STYLE_CONDITION)
            except ValueError as exc:
                fail(str(exc), _EXIT_ARGS)
            return
        try:
            if mode == "auto":
                OmniVoiceRequest(mode="auto")
            else:
                OmniVoiceRequest(mode="fixed-style", style_condition=OMNIVOICE_STYLE_CONDITION)
        except ValueError as exc:
            fail(str(exc), _EXIT_ARGS)
        return

    if mode == "clone":
        if reference_audio is None or reference_text is None or not reference_text.strip():
            fail(
                "omnivoice-local clone mode requires a readable reference audio file via "
                "--reference-audio and non-empty --reference-text.",
                _EXIT_ARGS,
            )
        if design_instruction is not None:
            fail("omnivoice-local clone mode rejects --design-instruction", _EXIT_ARGS)
        reference_audio_path = Path(reference_audio)
        if not reference_audio_path.is_file():
            fail(
                f"OmniVoice reference audio is not a readable file: {reference_audio_path}",
                _EXIT_ARGS,
            )
        try:
            with reference_audio_path.open("rb"):
                pass
        except (OSError, ValueError):
            fail(
                f"OmniVoice reference audio is not a readable file: {reference_audio_path}",
                _EXIT_ARGS,
            )
        try:
            OmniVoiceRequest(
                mode="clone",
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
            )
        except ValueError as exc:
            fail(str(exc), _EXIT_ARGS)
        return

    if reference_audio is not None or reference_text is not None:
        fail(
            "omnivoice-local design mode rejects --reference-audio and --reference-text",
            _EXIT_ARGS,
        )
    if design_instruction is None or not design_instruction.strip():
        fail(
            "omnivoice-local design mode requires non-empty --design-instruction",
            _EXIT_ARGS,
        )
    try:
        normalize_omnivoice_design_instruction(
            design_instruction, language=OMNIVOICE_DEFAULT_LANGUAGE
        )
    except ValueError as exc:
        fail(str(exc), _EXIT_ARGS)
    try:
        OmniVoiceRequest(mode="design", instruction=design_instruction)
    except ValueError as exc:
        fail(str(exc), _EXIT_ARGS)


def _enforce_omnivoice_design_route(args: argparse.Namespace, chunks: list[ScriptChunk]) -> None:
    """Reject unsupported long design before provider/GPU admission."""
    if getattr(args, "provider", None) != "omnivoice-local":
        return
    policy = evaluate_omnivoice_design_route(
        " ".join(chunk.text for chunk in chunks),
        language=OMNIVOICE_DEFAULT_LANGUAGE,
        mode=getattr(args, "mode", "preset"),
    )
    if policy.status == "allowed":
        return
    if policy.status == "experimental":
        assert policy.warning is not None
        print(f"Warning: {policy.warning}", file=sys.stderr)
        return
    fail(
        "OmniVoice Voice Design is unreliable for long Russian speech: upstream trains "
        "this mode only on Chinese and English. The text is valid, but this mode is "
        f"unsupported above the {OMNIVOICE_LONG_FORM_THRESHOLD_SECONDS:g}-second long-form "
        "threshold. Choose Russian reference cloning, an available accepted preset, "
        "separately accepted short design clips (experimental), or another TTS provider; "
        "no mode or voice was changed.",
        _EXIT_ARGS,
        details=policy.error_details(),
    )


def _resolve_script_format(script_path: Path, requested_format: str) -> str:
    """Compatibility wrapper for ``services.prepare.resolve_script_format``."""
    return resolve_script_format(script_path, requested_format)


def generate(args: argparse.Namespace) -> None:
    if getattr(args, "json_output", False) and getattr(args, "json_events", False):
        fail(
            "--json and --json-events are mutually exclusive; use one machine output mode.",
            _EXIT_ARGS,
        )
    if getattr(args, "resume", False) and getattr(args, "overwrite", False):
        fail(
            "--resume and --overwrite are mutually exclusive: --resume continues an existing run "
            "while --overwrite deletes it first. Use --resume to continue, or --overwrite with "
            "--confirm-delete-paid-audio for an explicitly new run.",
            _EXIT_ARGS,
        )
    script_format = _resolve_script_format(args.script, args.format)
    args.format = script_format
    _validate_omnivoice_options(args)
    try:
        ffmpeg_path, ffprobe_path = check_media_tools()
    except RuntimeError as e:
        fail(str(e), _EXIT_NO_FFMPEG)

    gemini_report = None
    voiceover_report = None
    if is_dialogue_format(script_format):
        args.provider = args.provider or "openrouter-tts"
        _resolve_model(args)
        allowed_voices = None
        if args.provider == "omnivoice-local":
            catalog = getattr(args, "voice_bank_catalog", None)
            if catalog is None:
                fail("omnivoice-local dialogue requires an admitted voice bank", _EXIT_ARGS)
            allowed_voices = {profile.id for profile in catalog.profiles}
        gemini_report = validate_gemini_dialogue_file(
            args.script,
            delimiter=args.delimiter,
            model=args.model,
            speaker_voice_overrides=args.speaker_voice,
            agent=True,
            provider=args.provider,
            allowed_voices=allowed_voices,
        )
        if not gemini_report["valid"]:
            if args.json_output:
                first = gemini_report["errors"][0] if gemini_report["errors"] else {}
                message = first.get("message", "Gemini dialogue validation failed.")
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "error": message,
                            "code": _EXIT_ARGS,
                            "details": gemini_report,
                        },
                        ensure_ascii=False,
                    )
                )
                sys.exit(_EXIT_ARGS)
            for item in gemini_report["errors"]:
                print(f"ERROR {item['code']}: {item['message']}", file=sys.stderr)
            sys.exit(_EXIT_ARGS)
        chunks = dialogue_turns_from_validation(gemini_report)
        if args.voice is not None and args.provider != "omnivoice-local":
            derived_voice = next(iter(gemini_report["speaker_voice_map"].values()))
            if args.voice != derived_voice:
                fail(
                    "Explicit --voice conflicts with the dialogue cast; remove --voice or align it with the first speaker voice.",
                    _EXIT_ARGS,
                )
    elif script_format == VOICEOVER_FORMAT:
        voiceover_report = validate_voiceover_file(
            args.script,
            delimiter=args.delimiter,
            provider_override=args.provider,
            model_override=getattr(args, "model", None),
            voice_override=args.voice,
            max_chunk_chars=args.max_chunk_chars,
            agent=True,
        )
        if not voiceover_report["valid"]:
            if args.json_output:
                print(json.dumps(voiceover_report, ensure_ascii=False))
                sys.exit(_EXIT_ARGS)
            for item in voiceover_report["errors"]:
                print(f"ERROR {item['code']}: {item['message']}", file=sys.stderr)
            sys.exit(_EXIT_ARGS)
        effective = voiceover_report["effective_config"]
        args.provider = effective["provider"]
        args.model = effective["model"]
        args.voice = effective["voice"]
        if effective.get("fallback_voice"):
            args.fallback_voice = effective["fallback_voice"]
        if (
            effective.get("style_prompt")
            and args.style_prompt is None
            and args.style_prompt_file is None
            and not args.no_style_prompt
        ):
            args.style_prompt = effective["style_prompt"]
        chunks = chunks_from_voiceover_report(voiceover_report)
    else:
        args.provider = args.provider or DEFAULT_PROVIDER
        _resolve_model(args)
        chunks = split_markdown_by_delimiter(args.script, args.delimiter)

    _resolve_qwen_mode_identity(args)
    _validate_model_for_provider(args.provider, args.model)
    _validate_omnivoice_options(args)
    if args.provider == "openrouter-tts" and (
        getattr(args, "style_prompt", None) is not None
        or getattr(args, "style_prompt_file", None) is not None
    ):
        fail(
            "OpenRouter /audio/speech does not support style prompts in synthesis input; "
            "remove --style-prompt/--style-prompt-file and select delivery with --voice only.",
            _EXIT_ARGS,
        )
    try:
        fragment_preparation = prepare_script_fragments(
            args,
            chunks,
            script_format,
            local_prepare=prepare_local_tts_chunks,
        )
    except ValueError as exc:
        fail(str(exc), _EXIT_ARGS)
    chunks = fragment_preparation.chunks
    original_chunk_count = fragment_preparation.original_count
    requested_fragment_count = fragment_preparation.requested_count
    _enforce_omnivoice_design_route(args, chunks)
    chunks = prepare_runtime_chunks(
        args,
        chunks,
        script_format,
        merge_session_fragments=merge_omnivoice_session_fragments,
    )
    runtime_session_count = len(chunks)
    if args.run_id:
        _validate_run_id(args.run_id)
    _validate_output_dir(args.output_dir)
    paths = build_run_paths(args.output_dir, args.model, args.run_id or None)

    if getattr(args, "dry_run_cost", False):
        _json_ok(
            {
                "status": "success",
                "dry_run": True,
                "provider": args.provider,
                "model": args.model,
                "voice": args.voice or _default_voice(args),
                "script_format": script_format,
                "chunks": requested_fragment_count,
                "original_chunks": original_chunk_count,
                "runtime_sessions": runtime_session_count,
                "total_characters": sum(len(chunk.text) for chunk in chunks),
                "estimated_cost": None,
                "estimate_note": "Exact pre-generation cost is unavailable for this provider/model without usage data.",
            }
        )

    # Native history ownership is resolved before the legacy JSON recovery guards,
    # provider construction, pricing, or any directory deletion. A committed
    # native run row or native local evidence decides the route; a fail-closed
    # classification stops here instead of falling back to the legacy writer.
    native_ownership = native_generation.resolve_native_ownership(paths.output_root)
    if native_ownership.route == "blocked":
        fail(
            native_ownership.reason or "native run ownership could not be verified.",
            _EXIT_PROVIDER,
            details={"error_code": native_ownership.error_code or "NATIVE_OWNERSHIP_UNVERIFIABLE"},
        )
    # The paid timings route owns the same canonical output root namespace. A
    # committed paid run -- pending or completed, even one whose descriptor write
    # failed -- or its local evidence means this directory belongs to that paid
    # submit, so no generate route may select, overwrite, or admit it.
    _reject_paid_owned_root_for_generate(paths)
    if native_ownership.route == "native_existing" or (
        _native_route_eligible(args, script_format) and not paths.output_root.exists()
    ):
        _run_native_route(
            args,
            chunks,
            script_format,
            paths,
            native_ownership,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            gemini_report=gemini_report,
        )
        return

    # The native executor takes this same one-writer run lock, so holding it here
    # for the whole mutable legacy tail keeps ownership selection and mutation from
    # interleaving with a concurrent native writer for this run root.
    with _legacy_run_lock(paths):
        if args.resume and not args.skip_existing and paths.output_root.exists():
            # A resume whose previous paid submit is unconfirmed must fail closed before
            # any optional quality/timing preflight, key read, provider build, identity
            # check, or pricing I/O: the documented PAID_SUBMIT_UNCONFIRMED envelope
            # cannot be replaced by a key or dependency error, and no request may leave.
            # The one exception is a paid attempt the same command can recover without
            # a new submit: saved raw audio rebuilt locally, or a stored paid Polza
            # media id finished with GET calls only. Both still match this exact
            # provider/model/voice/script chunk, where a validated dialogue's voice is
            # its first cast voice rather than the provider default. Every other marker
            # keeps the block.
            # ``--skip-existing`` keeps its precedence and reports an existing folder as
            # skipped without loading run state. The same guard runs inside
            # ``_generate_step`` for direct callers.
            pending_attempt = _unconfirmed_paid_attempt_in_existing_run(paths)
            if pending_attempt is not None:
                resume_state, _state_unreadable = _load_status_state(paths.output_root / STATE_FILE)
                if not _recoverable_paid_attempts(
                    resume_state,
                    provider=args.provider,
                    model=args.model,
                    voice=_resume_guard_voice(args, gemini_report),
                    chunks=chunks,
                    chunks_dir=paths.chunks_dir,
                    run_root=paths.output_root,
                ):
                    _reject_unconfirmed_paid_resume(
                        pending_attempt, GenerationLogger(paths.output_root / LOG_FILE)
                    )

        if (
            is_dialogue_format(script_format)
            and args.provider == "openrouter-tts"
            and not getattr(args, "tts_quality_provider", None)
        ):
            fail(
                "OpenRouter dialogue requires --tts-quality-provider so every paid turn is "
                "transcribed and checked before concat.",
                _EXIT_ARGS,
            )

        if is_dialogue_format(script_format) and getattr(args, "tts_quality_provider", None):
            _preflight_tts_quality_provider(args)

        if getattr(args, "with_timings", False):
            _preflight_timing_dependency(
                getattr(args, "timing_provider", "faster-whisper"),
            )

        # The ownership decision was made before the run lock was taken, so a
        # concurrent native writer can commit between that read and this write.
        # Re-resolve with the WAL-consistent reader and fail closed, rather than let
        # the legacy writer create, resume, overwrite, or delete a native-owned root.
        _reject_native_owned_root_for_legacy(paths)
        _reject_paid_owned_root_for_generate(paths)

        if paths.output_root.exists():
            if args.skip_existing:
                files = _list_artifact_files(paths)
                _json_ok(
                    {
                        "status": "skipped",
                        "reason": "run folder exists",
                        "run_id": paths.prefix,
                        "files": files,
                    }
                )
                return
            if args.resume:
                pass
            elif not args.overwrite:
                fail(
                    f"Run folder already exists: {paths.output_root}. Use --resume to continue, --skip-existing, or --overwrite.",
                    _EXIT_PROVIDER,
                )
            if args.overwrite:
                # The marker check precedes the paid-MP3 confirmation: a run that
                # already saved one MP3 and later left an unconfirmed attempt must
                # report the documented PAID_SUBMIT_UNCONFIRMED envelope instead of
                # the generic paid-audio delete error.
                pending_attempt = _unconfirmed_paid_attempt_in_existing_run(paths)
                if pending_attempt is not None:
                    fail(
                        "Refusing --overwrite: "
                        f"{paths.output_root} holds an unconfirmed paid submit "
                        f"({pending_attempt.get('id')}, status {pending_attempt.get('status')}). "
                        "Keep this run as evidence and use a different --run-id for an explicitly "
                        "new attempt.",
                        _EXIT_PROVIDER,
                        details=_paid_submit_unconfirmed_details(pending_attempt),
                    )
                if _has_paid_chunk_audio(paths) and not args.confirm_delete_paid_audio:
                    fail(
                        "Refusing to delete existing paid chunk audio. Use --resume, or add --confirm-delete-paid-audio with --overwrite.",
                        _EXIT_PROVIDER,
                    )
                _safe_remove_run_dir(paths.output_root, args.output_dir)

        _ensure_run_dirs(paths)

        try:
            generation_identity = prepare_generation_identity(
                args,
                chunks,
                gemini_report,
                resolve_style_prompt=_resolve_provider_style_prompt,
            )
        except PreparationError as exc:
            fail(str(exc), _EXIT_ARGS)
        chunks = generation_identity.chunks
        style_prompt = generation_identity.style_prompt
        prompt_mode = generation_identity.prompt_mode
        _preflight_dialogue_resume(args, chunks, paths, style_prompt, prompt_mode)
        api_key = read_api_key(args)
        provider_for_generation: Any = build_provider(args, api_key, style_prompt, prompt_mode)
        if args.provider == "omnivoice-local" and gemini_report:
            provider_for_generation = _bind_dialogue_voice_bank_providers(
                provider_for_generation,
                args.voice_bank_catalog,
                gemini_report["speaker_voice_map"],
            )
        pricing_snapshot = fetch_pricing_snapshot(args.provider, api_key, args.model)

        _generate_step(
            args,
            provider_for_generation,
            ffmpeg_path,
            ffprobe_path,
            chunks,
            api_key,
            pricing_snapshot,
            paths,
            style_prompt,
            prompt_mode,
        )


def _native_route_eligible(args: argparse.Namespace, script_format: str) -> bool:
    """Whether this command is an admitted DB-first native TTS route.

    An ordinary, non-dialogue ``polza-tts`` run on either its async ``elevenlabs/``
    ``/media`` model route or its synchronous ``/audio/speech`` model route, an
    ordinary non-dialogue ``polza-chat-audio`` chat-audio run, and an
    ordinary ``openrouter-tts`` run, are executed by the native executor when they
    use no unsupported option mixture. Such a run is admitted for a plain Markdown
    script and for a ``format: voiceover`` script: the voiceover validator has
    already resolved the one provider, model, and voice the whole script speaks
    with, and the snapshot hashes the prepared chunk text, so no per-chunk provider
    or voice is invented. Two bounded integrated steps and the recorded trimming
    semantics are admitted for that same route: an installed *local*
    ``--tts-quality-provider`` (``qwen-local``/``nemotron-local``),
    ``--with-timings`` for the local ``faster-whisper`` or the paid cloud
    ``groq-whisper``/``xai-stt`` provider, and ``--no-trim``. The timing and local
    quality steps may be requested together. Every other route and mixture -- a
    cloud quality provider on a non-dialogue route and an unregistered quality
    provider, plus the ``openrouter-whisper`` timing provider -- keeps the legacy
    executor and its JSON writer unchanged.

    One dialogue route is admitted separately: the existing validated
    ``openrouter-tts`` Gemini two-speaker script with the required installed local
    ``--tts-quality-provider`` (``qwen-local``/``nemotron-local``) or the paid
    cloud ``xai-stt`` provider. It runs each turn's required quality gate before
    the final concat and before the next paid turn. A second dialogue route, the
    existing ``omnivoice-local`` preset two-profile bank script, is likewise
    admitted and runs its optional quality gate before concat only when it recorded
    a local or paid cloud ``--tts-quality-provider``. Both dialogue routes also
    admit the recorded trimming semantics of ``--no-trim`` and an integrated
    ``--with-timings`` step for the local or paid cloud timing providers. Every ordinary
    non-dialogue ``qwen-local`` mode (clone, and the instructed preset/design) and
    every non-dialogue ``omnivoice-local`` mode (preset bank, and
    ``auto``/``clone``/``design``) is admitted as a local route with the same
    recorded trimming, integrated local timing, and installed local quality steps.
    Both local families also admit a ``format: voiceover`` script for the
    combinations that already function on the legacy executor: every
    ``qwen-local`` mode, and the ``omnivoice-local`` preset bank route. The
    voiceover validator resolves one voice for the whole script and each mode then
    applies its own effective voice (see the helpers below), so no new voice
    semantics is invented. An ``omnivoice-local`` non-preset mode keeps the legacy
    executor for a ``format: voiceover`` script because that script always
    supplies a voice the mode rejects. Every other dialogue route -- every
    ``polza-tts`` dialogue -- stays on the legacy executor.
    """
    if is_dialogue_format(script_format):
        return _native_dialogue_route_eligible(args)
    if getattr(args, "provider", None) == "qwen-local":
        # The local Qwen routes admit a plain Markdown script and a
        # ``format: voiceover`` script: the voiceover validator resolves one voice
        # for the whole script and the CLI's ``_resolve_qwen_mode_identity`` then
        # selects the mode's own effective voice -- ``preset`` keeps that validated
        # voice, while ``clone``/``design`` replace it with their mode marker exactly
        # as the legacy executor does. ``auto`` is a usage error before this gate.
        if script_format not in ("markdown", VOICEOVER_FORMAT):
            return False
        return _native_qwen_local_route_eligible(args)
    if getattr(args, "provider", None) == "omnivoice-local":
        if script_format == VOICEOVER_FORMAT:
            # Only the preset bank route accepts a voiceover script: that script
            # always supplies a voice, which a non-preset mode rejects as a usage
            # error, and the validator permits only the default bank marker voice,
            # which the admitted catalog must resolve into the committed identity.
            if (getattr(args, "mode", None) or "preset") != "preset":
                return False
        elif script_format != "markdown":
            return False
        return _native_omnivoice_monologue_route_eligible(args)
    if script_format not in ("markdown", VOICEOVER_FORMAT):
        return False
    if not _native_local_output_steps_admitted(args):
        return False
    if not isinstance(getattr(args, "model", None), str):
        return False
    return getattr(args, "provider", None) in (
        "polza-tts",
        "polza-chat-audio",
        "openrouter-tts",
    )


def _native_timing_step_admitted(args: argparse.Namespace) -> bool:
    """Whether the requested integrated timing step may run on the native route.

    The recorded trimming semantics of ``--no-trim`` are always part of the native
    route and need no gate here. An integrated ``--with-timings`` run is admitted
    for the local ``faster-whisper`` provider and for the paid cloud
    ``groq-whisper``/``xai-stt`` providers, whose one POST runs through the shared
    paid-transcription boundary. ``openrouter-whisper`` is deliberately absent: it
    returns no real timestamps and keeps the legacy route's explicit rejection.
    """
    if not getattr(args, "with_timings", False):
        return True
    return getattr(args, "timing_provider", "faster-whisper") in (
        native_generation.NATIVE_TIMING_PROVIDERS
    )


def _native_quality_step_admitted(args: argparse.Namespace, *, dialogue: bool) -> bool:
    """Whether the requested integrated quality step may run on the native route.

    The installed local ``qwen-local``/``nemotron-local`` providers are always
    admitted. The paid cloud ``xai-stt`` provider is admitted only for a dialogue
    route, where it verifies each turn on the same boundary: a non-dialogue run
    keeps the exact legacy behavior of ignoring the cloud quality flag.
    """
    quality_provider = getattr(args, "tts_quality_provider", None)
    if quality_provider is None:
        return True
    if quality_provider in native_generation.NATIVE_LOCAL_QUALITY_PROVIDERS:
        return True
    return dialogue and quality_provider in native_generation.NATIVE_PAID_QUALITY_PROVIDERS


def _native_local_output_steps_admitted(args: argparse.Namespace) -> bool:
    """Whether the requested non-dialogue post-audio steps may run natively.

    This is the non-dialogue combination of :func:`_native_timing_step_admitted`
    (local or paid cloud timing) and :func:`_native_quality_step_admitted` with
    ``dialogue=False`` (installed local quality only). A cloud quality provider on a
    non-dialogue route keeps the legacy executor, exactly as before.
    """
    return _native_timing_step_admitted(args) and _native_quality_step_admitted(
        args, dialogue=False
    )


def _bind_native_omnivoice_monologue_voice(args: argparse.Namespace) -> None:
    """Bind the effective voice of the admitted preset OmniVoice bank monologue.

    That route's effective voice is its selected bank profile id. The native
    snapshot needs a non-empty run voice and the run identity records it; the legacy
    executor keeps its unchanged ``None`` voice. A non-preset mode has no CLI voice
    at all (it rejects one) and records its mode marker on the prepared run instead.
    """
    if getattr(args, "voice_bank_profile", None) is not None and args.voice is None:
        args.voice = args.voice_bank_profile.id


def _native_omnivoice_monologue_route_eligible(args: argparse.Namespace) -> bool:
    """Whether this command is an admitted non-dialogue OmniVoice route.

    Two families are admitted:

    * the existing ``--mode preset`` run with an admitted ``--voice-bank`` catalog
      and a resolved profile, which merges into one OmniVoice session part that
      clones the selected bank profile; and
    * the existing ``auto``, ``clone``, and ``design`` modes, whose identity is the
      mode plus, for clone, a readable ``--reference-audio`` with a non-empty
      ``--reference-text``, and, for design, a non-empty ``--design-instruction``.

    ``--no-trim``, ``--with-timings --timing-provider faster-whisper``, and an
    installed local quality provider are admitted by
    :func:`_native_local_output_steps_admitted`; an unexpected model and an unknown
    mode keep the legacy executor, because the native route records the mode, its
    reference/instruction inputs, the model, and the effective voice as part of its
    identity. The route gate admits the preset bank route for a ``format:
    voiceover`` script too; a non-preset mode is never admitted there, because such
    a script always supplies the ``--voice`` those modes reject.
    """
    if not _native_local_output_steps_admitted(args):
        return False
    if getattr(args, "model", None) != OMNIVOICE_LOCAL_MODEL_ID:
        return False
    mode = getattr(args, "mode", None) or "preset"
    if mode == "preset":
        if getattr(args, "voice_bank_catalog", None) is None:
            return False
        return getattr(args, "voice_bank_profile", None) is not None
    if mode == OMNIVOICE_AUTO_MODE:
        return True
    if mode == OMNIVOICE_CLONE_MODE:
        if getattr(args, "reference_audio", None) is None:
            return False
        reference_text = getattr(args, "reference_text", None)
        return isinstance(reference_text, str) and bool(reference_text.strip())
    if mode == OMNIVOICE_DESIGN_MODE:
        design_instruction = getattr(args, "design_instruction", None)
        return isinstance(design_instruction, str) and bool(design_instruction.strip())
    return False


def _qwen_tts_runtime() -> str:
    """Return the selected Qwen TTS runtime, exactly as the provider factory reads it."""
    return os.environ.get("VOICEOVER_QWEN_TTS_RUNTIME", "python").strip()


def _native_qwen_local_route_eligible(args: argparse.Namespace) -> bool:
    """Whether this command is an admitted local Qwen native route.

    Three ordinary non-dialogue ``qwen-local`` modes are admitted:

    * ``clone`` with a reference ``--sample``;
    * ``preset`` with the CustomVoice model; and
    * ``design`` with the VoiceDesign model.

    The instructed ``preset``/``design`` modes additionally require a non-empty
    instruction (the ``--qwen-instruct`` value, or the ``QWEN_INSTRUCT`` default),
    because the runtime needs one and the snapshot stores it as the run's style
    identity. The route gate admits these for a plain Markdown script and for a
    ``format: voiceover`` script; the CLI's ``_resolve_qwen_mode_identity`` has
    already selected the exact effective voice each mode speaks with. ``auto``
    (rejected as a usage error before this gate, because the runtime implements no
    automatic mode selection), an unexpected model, a missing sample, and an
    unrecognized ``VOICEOVER_QWEN_TTS_RUNTIME`` all keep the legacy executor,
    because the native route records the mode, model, voice, instruction, and
    runtime as part of its identity. The recorded trimming semantics, an integrated
    ``--with-timings`` step for the local or paid cloud timing providers, and an
    installed local quality provider are admitted by
    :func:`_native_local_output_steps_admitted`; a cloud quality provider keeps the
    legacy executor, exactly as before.
    """
    if not _native_local_output_steps_admitted(args):
        return False
    if _qwen_tts_runtime() not in ("python", "audio-cpp"):
        return False
    mode = getattr(args, "mode", None)
    if mode == "clone":
        if getattr(args, "model", None) != QWEN_MODEL_BASE:
            return False
        sample = getattr(args, "sample", None)
        return isinstance(sample, str) and bool(sample.strip())
    if mode == "preset":
        if getattr(args, "model", None) != QWEN_MODEL_CUSTOMVOICE:
            return False
        return _qwen_instruct_admissible(args)
    if mode == "design":
        if getattr(args, "model", None) != QWEN_MODEL_VOICE_DESIGN:
            return False
        return _qwen_instruct_admissible(args)
    return False


def _qwen_instruct_admissible(args: argparse.Namespace) -> bool:
    """Whether an admitted instructed ``qwen-local`` mode has a usable instruction.

    An unset ``--qwen-instruct`` means the ``QWEN_INSTRUCT`` default; an explicit
    empty or whitespace instruction keeps the run on the legacy executor instead of
    reaching the runtime or the snapshot's non-empty style-prompt rule.
    """
    instruct = getattr(args, "qwen_instruct", None)
    return instruct is None or (isinstance(instruct, str) and bool(instruct.strip()))


def _native_dialogue_route_eligible(args: argparse.Namespace) -> bool:
    """Whether this command is one of the two admitted native dialogue routes.

    The existing validated OpenRouter Gemini two-speaker dialogue is admitted only
    with the required installed local ``--tts-quality-provider``; its per-turn
    quality gate runs before the next paid turn and before the final concat. The
    existing ``omnivoice-local`` preset dialogue is admitted with its admitted
    voice bank and runs the same per-turn local quality gate before concat only
    when it recorded an installed local ``--tts-quality-provider``; without one it
    keeps the exact legacy behavior of no gate. Both dialogue routes also admit the
    recorded trimming semantics of ``--no-trim`` and an integrated
    ``--with-timings`` step for the local ``faster-whisper`` or the paid cloud
    ``groq-whisper``/``xai-stt`` provider; the OpenRouter route's required gate also
    accepts the paid cloud ``xai-stt`` quality provider. Every ``polza-tts``
    dialogue stays on the legacy executor.
    """
    provider = getattr(args, "provider", None)
    if provider == "omnivoice-local":
        if getattr(args, "model", None) != OMNIVOICE_LOCAL_MODEL_ID:
            return False
        if (getattr(args, "mode", None) or "preset") != "preset":
            return False
        if not _native_timing_step_admitted(args):
            return False
        if not _native_quality_step_admitted(args, dialogue=True):
            return False
        if getattr(args, "voice_bank_catalog", None) is None:
            return False
        return True
    if provider != "openrouter-tts":
        return False
    if getattr(args, "model", None) != GEMINI_TTS_MODEL:
        return False
    if getattr(args, "tts_quality_provider", None) not in (
        native_generation.NATIVE_LOCAL_QUALITY_PROVIDERS
        | native_generation.NATIVE_PAID_QUALITY_PROVIDERS
    ):
        return False
    # The optional timing step and the recorded trimming semantics are admitted
    # alongside the required per-turn quality gate; every admitted timing provider
    # (local or paid cloud) and the paid xAI quality provider are covered.
    return _native_timing_step_admitted(args)


@contextmanager
def _legacy_run_lock(paths) -> Iterator[None]:
    """Serialize the legacy writer against every other writer of one run root.

    Ownership was resolved before this lock is taken, so a concurrent native
    writer could otherwise commit between that read and the legacy write. The
    native executor takes this same one-writer run lock, so holding it from the
    legacy ownership recheck through the whole legacy generation means the two
    writers of one run root can never interleave. Contention fails closed with the
    stable provider exit code and a bounded message that does not echo the lock
    path or the history home.
    """
    try:
        with acquire_run_lock(paths.output_root):
            yield
    except HistoryRunLockedError:
        fail(
            "another process is already writing this run directory; refusing to run two "
            "writers for one run.",
            _EXIT_PROVIDER,
            details={"error_code": "NATIVE_RUN_LOCKED"},
        )


def _reject_native_owned_root_for_legacy(paths) -> None:
    """Fail closed when a root the legacy writer is about to touch is native-owned.

    Ownership is resolved before the run lock is taken, so a concurrent native
    writer can commit between that early read and the legacy write. Re-resolving
    here with the WAL-consistent reader keeps the legacy selection and its write
    from interleaving a native owner: a native-owned or unverifiable root stops
    with the stable ownership envelope instead of being created, resumed,
    overwritten, or deleted by the legacy writer. A genuinely legacy root is
    unchanged.
    """
    decision = native_generation.resolve_native_ownership(paths.output_root)
    if decision.route == "legacy":
        return
    fail(
        decision.reason or "this run directory is owned by native history.",
        _EXIT_PROVIDER,
        details={"error_code": decision.error_code or "NATIVE_OWNERSHIP_UNVERIFIABLE"},
    )


def _reject_paid_owned_root_for_generate(paths) -> None:
    """Fail closed when a root the generate writer is about to touch is paid-owned.

    The paid timings route binds ownership to the same canonical output root, so a
    committed paid timing run -- still ``submitting`` or completed -- or a paid
    timing ownership descriptor/local evidence means the directory belongs to that
    paid submit. A generate route must not select it, overwrite it, or freshly
    admit it, because a second paid submit or a deleted raw response is not
    recoverable. This runs under the shared run-root lock so a concurrent paid
    writer cannot commit between the ownership read and the generate write.
    """
    try:
        ownership = paid_transcription_history.resolve_paid_timing_ownership(paths.output_root)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_PROVIDER, details={"error_code": exc.error_code})
    if ownership.owned:
        fail(
            "This run directory is owned by a paid timing run; a generate invocation or "
            "--overwrite cannot replace its paid evidence. Use `history resume`/`history "
            "sync` with its UUID, or choose a different --run-id.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TIMING_OUTPUT_OWNED", "run_uuid": ownership.run_uuid},
        )


def _run_native_route(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    script_format: str,
    paths,
    ownership,
    *,
    ffmpeg_path: str,
    ffprobe_path: str,
    gemini_report: dict[str, Any] | None = None,
) -> None:
    """Execute, resume, or re-export one native-owned run and print the result.

    The provider is built lazily inside ``provider_factory`` and therefore only
    for a fresh unattempted submit (async media or synchronous) or a known-id
    media GET recovery: a local raw rebuild or a completed-run export repair never
    reads an API key or constructs a provider. ``--overwrite`` is rejected for a
    native run so accepted paid evidence is never deleted; ``--skip-existing``
    keeps its usual precedence. ``gemini_report`` carries the validated dialogue
    cast so a dialogue run resolves its first-cast voice identity exactly as the
    legacy executor does; a non-dialogue OmniVoice route instead resolves its
    effective voice before the snapshot from the selected bank profile (preset) or
    the mode marker (``auto``/``clone``/``design``).
    """
    if ownership.route == "native_existing" and not _native_route_eligible(args, script_format):
        fail(
            "This run directory is owned by native history. Only an admitted native "
            "polza-tts, polza-chat-audio, openrouter-tts, local Qwen clone/preset/design, "
            "or local OmniVoice auto/clone/design/bank-mono/dialogue run may continue it; "
            "choose a different --run-id for other options.",
            _EXIT_PROVIDER,
            details={"error_code": "NATIVE_OPTIONS_UNSUPPORTED"},
        )
    if paths.output_root.exists() and getattr(args, "skip_existing", False):
        files = _list_artifact_files(paths)
        _json_ok(
            {
                "status": "skipped",
                "reason": "run folder exists",
                "run_id": paths.prefix,
                "files": files,
            }
        )
    if ownership.route == "native_existing":
        if getattr(args, "overwrite", False):
            fail(
                "Refusing --overwrite: this run is owned by the native history database and its "
                "accepted paid evidence must be kept as-is. Use a different --run-id for an "
                "explicitly new attempt.",
                _EXIT_PROVIDER,
                details={"error_code": "NATIVE_OVERWRITE_UNSUPPORTED"},
            )
        if not getattr(args, "resume", False):
            fail(
                f"Run folder already exists: {paths.output_root}. Use --resume to continue it, "
                "or a different --run-id.",
                _EXIT_PROVIDER,
            )
    if args.provider == "omnivoice-local" and not is_dialogue_format(script_format):
        _bind_native_omnivoice_monologue_voice(args)
    try:
        generation_identity = prepare_generation_identity(
            args,
            chunks,
            gemini_report,
            resolve_style_prompt=_resolve_provider_style_prompt,
        )
    except PreparationError as exc:
        fail(str(exc), _EXIT_ARGS)
    chunks = generation_identity.chunks
    qwen_clone_identity = None
    qwen_mode_identity = None
    omnivoice_mode_identity = None
    if args.provider == "qwen-local":
        # A clone reference is read and hashed here, before the snapshot, so a
        # missing sample fails as a usage error and the committed identity always
        # carries a locator and digest this run actually read. The instructed
        # preset/design routes carry no external file: their identity is the
        # resolved mode/model/voice and the exact style instruction the run speaks
        # with, captured here before any provider exists.
        try:
            if args.mode == "clone":
                qwen_clone_identity = build_qwen_clone_identity(
                    model=args.model,
                    sample_path=args.sample,
                    sample_text=getattr(args, "sample_text", None) or "",
                    runtime=_qwen_tts_runtime(),
                    language=QWEN_LANGUAGE,
                )
            else:
                qwen_mode_identity = build_qwen_mode_identity(
                    mode=args.mode,
                    model=args.model,
                    voice=args.voice,
                    instruct=generation_identity.style_prompt,
                    runtime=_qwen_tts_runtime(),
                    language=QWEN_LANGUAGE,
                )
        except PreparationError as exc:
            fail(str(exc), _EXIT_ARGS)
    elif args.provider == "omnivoice-local" and not is_dialogue_format(script_format):
        # A non-preset mode's identity is its mode plus, for clone, the reference
        # file read and hashed here (so a missing file is a usage error and the
        # committed identity always carries a digest this run actually read) and,
        # for design, the exact instruction the runtime speaks with.
        mode = getattr(args, "mode", "preset")
        if mode in OMNIVOICE_MODE_IDENTITY_MODES:
            try:
                omnivoice_mode_identity = build_omnivoice_mode_identity(
                    mode=mode,
                    model=args.model,
                    reference_audio=getattr(args, "reference_audio", None),
                    reference_text=getattr(args, "reference_text", None),
                    design_instruction=getattr(args, "design_instruction", None),
                )
            except PreparationError as exc:
                fail(str(exc), _EXIT_ARGS)
    prepared = prepare_run(
        args,
        chunks,
        generation_identity.style_prompt,
        generation_identity.prompt_mode,
        qwen_clone_identity=qwen_clone_identity,
        qwen_mode_identity=qwen_mode_identity,
        omnivoice_mode_identity=omnivoice_mode_identity,
    )
    provider_cache: list[Any] = []

    def provider_factory() -> Any:
        if not provider_cache:
            api_key = read_api_key(args)
            if args.provider == "omnivoice-local":
                # Every local OmniVoice bank route verifies its committed reference
                # (and, for dialogue, clones one provider per cast voice-bank
                # profile) before any local model call; a changed or missing
                # reference file fails closed with a bounded error.
                try:
                    provider: Any = build_provider(
                        args,
                        api_key,
                        generation_identity.style_prompt,
                        generation_identity.prompt_mode,
                    )
                    if gemini_report:
                        provider = _bind_dialogue_voice_bank_providers(
                            provider, args.voice_bank_catalog, gemini_report["speaker_voice_map"]
                        )
                except VoiceBankError as exc:
                    fail(
                        str(exc),
                        _EXIT_PROVIDER,
                        details={
                            "error_code": native_generation._ERROR_LOCAL_REFERENCE_UNAVAILABLE
                        },
                    )
            else:
                provider = build_provider(
                    args, api_key, generation_identity.style_prompt, generation_identity.prompt_mode
                )
            provider_cache.append(provider)
        return provider_cache[0]

    hooks = _native_execution_hooks(args)
    output_options = native_generation.build_output_options(
        args.no_trim,
        timing=_native_timing_options(args),
        quality=_native_quality_options(args),
    )
    try:
        summary = native_generation.run_native_generation(
            paths=paths,
            prepared=prepared,
            chunks=chunks,
            script_format=script_format,
            script_path=args.script,
            output_options=output_options,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            user_label=args.run_id or None,
            resume=bool(args.resume),
            provider_factory=provider_factory,
            hooks=hooks,
        )
    except native_generation.NativeGenerationError as exc:
        fail(str(exc), exc.code, details={"error_code": exc.error_code})
    if args.json_output:
        payload: dict[str, Any] = {
            "status": "success",
            "provider": args.provider,
            "model": args.model,
            "run_id": paths.prefix,
            "files": summary.files,
            "duration_ms": summary.duration_ms,
            "segment_count": summary.segment_count,
            "cost": {"total": summary.cost_total, "currency": summary.cost_currency},
        }
        if summary.timing_requested:
            payload["timing"] = {"complete": summary.timing_complete}
        if summary.quality_requested:
            payload["quality"] = {
                "complete": summary.quality_complete,
                "passed": summary.quality_passed if summary.quality_complete else None,
            }
        _json_ok(payload)
    print(f"Full MP3: {paths.full_mp3}")
    print(f"Run manifest: {paths.run_json}")
    print(f"Manifest: {paths.output_root / 'manifest.json'}")


def _preflight_dialogue_resume(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    paths,
    style_prompt: str | None,
    prompt_mode: str,
) -> None:
    """Reject unsafe dialogue resumes before provider construction or pricing I/O."""
    if not args.resume or not is_dialogue_format(getattr(args, "format", "markdown")):
        return
    state = load_state(paths.output_root / STATE_FILE)
    if state is None:
        if any(paths.chunks_dir.glob("*.mp3")):
            fail(
                "Cannot resume: orphan dialogue audio exists without trusted run state.",
                _EXIT_PROVIDER,
            )
        return
    if state.get("script_hash") != script_hash(chunks):
        fail(
            "Cannot resume: script chunks do not match the previous run_state.json.", _EXIT_PROVIDER
        )
    synthesis_identity = _dialogue_synthesis_identity(args, style_prompt, prompt_mode, chunks)
    if "synthesis_identity" not in state:
        fail(
            "Cannot resume: run state predates dialogue synthesis identity; start a fresh run instead of mixing artifacts.",
            _EXIT_PROVIDER,
        )
    if state.get("synthesis_identity") != synthesis_identity:
        fail("Cannot resume: dialogue synthesis identity changed.", _EXIT_PROVIDER)


def _state_entry_number(entry: dict[str, Any]) -> int | None:
    """Compatibility wrapper for ``services.recovery.state_entry_number``."""
    return recovery.state_entry_number(entry)


def _merge_attached_costs_into_state(state: dict[str, Any], artifacts: list[ChunkArtifact]) -> None:
    """Compatibility wrapper for ``services.cost_enrichment.merge_attached_costs_into_state``.

    The service owns the trusted-state merge and reports a missing or ambiguous
    chunk match as ``AttachedCostStateMismatchError``; this wrapper keeps the
    former name and the provider exit envelope that callers already expect.
    """
    try:
        cost_enrichment.merge_attached_costs_into_state(state, artifacts)
    except cost_enrichment.AttachedCostStateMismatchError as exc:
        fail(str(exc), _EXIT_PROVIDER)


def _media_observed_cost(usage: Any) -> tuple[float | None, str | None]:
    """Compatibility wrapper for ``services.costs.media_observed_cost``."""
    return costs.media_observed_cost(usage)


def _bind_polza_media_attempt(
    provider: PolzaTTSProvider,
    *,
    state: dict[str, Any],
    state_path: Path,
    logger: GenerationLogger,
    chunk: ScriptChunk,
) -> None:
    """Compatibility wrapper for ``services.execution.bind_polza_media_attempt``."""
    execution.bind_polza_media_attempt(
        provider, state=state, state_path=state_path, logger=logger, chunk=chunk
    )


def _persist_paid_raw_audio(
    result,
    *,
    state: dict[str, Any],
    state_path: Path,
    paths,
    chunk: ScriptChunk,
    logger: GenerationLogger,
) -> None:
    """Compatibility wrapper for ``services.execution.persist_paid_raw_audio``."""
    execution.persist_paid_raw_audio(
        result, state=state, state_path=state_path, paths=paths, chunk=chunk, logger=logger
    )


def _raw_recovery_result(
    args, chunk: ScriptChunk, run_root: Path, recovery: dict[str, Any]
) -> SynthesisResult:
    """Compatibility wrapper for ``services.execution.raw_recovery_result``."""
    return execution.raw_recovery_result(args, chunk, run_root, recovery)


def _generate_step(
    args,
    provider,
    ffmpeg_path,
    ffprobe_path,
    chunks,
    api_key,
    pricing_snapshot,
    paths,
    style_prompt,
    prompt_mode,
) -> None:
    run_started_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    logger = GenerationLogger(paths.output_root / LOG_FILE)
    state_path = paths.output_root / STATE_FILE
    logger.event(
        "info",
        "run_started",
        run_id=paths.prefix,
        provider=args.provider,
        model=args.model,
        chunks=len(chunks),
    )
    _emit_json_event(args, "run_started", run_id=paths.prefix, chunks=len(chunks))

    setup = execution.prepare_generation_state(
        args,
        chunks=chunks,
        style_prompt=style_prompt,
        prompt_mode=prompt_mode,
        paths=paths,
        ffprobe_path=ffprobe_path,
        state_path=state_path,
        logger=logger,
        hooks=execution.GenerationSetupHooks(
            # The setup seam runs in ``services.execution``; these are the
            # CLI-bound identity, recovery, and failure callables tests patch.
            voice_identity=_omnivoice_voice_identity,
            dialogue_synthesis_identity=_dialogue_synthesis_identity,
            recover_existing_chunks=_recover_existing_chunks,
            reject_unconfirmed_paid_resume=_reject_unconfirmed_paid_resume,
            fail_provider=lambda message: fail(message, _EXIT_PROVIDER),
        ),
    )
    state = setup.state
    prepared = setup.prepared
    paid_submit = setup.paid_submit
    retry_policy = setup.retry_policy
    dialogue_run = setup.dialogue_run
    recoverable_attempts = setup.recoverable_attempts
    completed = setup.completed
    loop_state = setup.loop_state
    chunk_artifacts_by_number = loop_state.chunk_artifacts_by_number

    hooks = execution.PartExecutionHooks(
        # The loop runs in ``services.execution``; these are the CLI-bound
        # callables existing tests patch and the CLI presentation seams.
        synthesize_part=synthesize_part,
        write_audio_as_mp3=write_audio_as_mp3,
        trim_final_silence=trim_final_silence,
        mp3_duration_ms=mp3_duration_ms,
        fail_provider=lambda message: fail(message, _EXIT_PROVIDER),
        fail_output=lambda message: fail(message, _EXIT_OUTPUT),
        emit_json_event=_emit_json_event,
        log_retry=_log_retry,
        reject_unconfirmed_paid_resume=_reject_unconfirmed_paid_resume,
        paid_submit_attempt_status=_paid_submit_attempt_status,
        public_projection=_public_artifact_projection,
        progress=print,
    )
    execution.execute_prepared_parts(
        args=args,
        prepared=prepared,
        chunks=chunks,
        paths=paths,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        state=state,
        state_path=state_path,
        logger=logger,
        provider=provider,
        paid_submit=paid_submit,
        retry_policy=retry_policy,
        recoverable_attempts=recoverable_attempts,
        dialogue_run=dialogue_run,
        completed=completed,
        loop_state=loop_state,
        hooks=hooks,
    )

    summary = finalization.finalize_generation(
        args=args,
        paths=paths,
        state=state,
        state_path=state_path,
        logger=logger,
        chunks=chunks,
        chunk_artifacts_by_number=chunk_artifacts_by_number,
        prepared=prepared,
        pricing_snapshot=pricing_snapshot,
        api_key=api_key,
        run_started_at=run_started_at,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        dialogue_run=dialogue_run,
        hooks=finalization.FinalizationHooks(
            # The finalization seam runs in ``services.finalization``; these are
            # the CLI-bound callables existing tests patch and the failure codes.
            attach_costs=attach_costs,
            verify_dialogue_turns_before_concat=_verify_dialogue_turns_before_concat,
            concat_dialogue_turns=concat_dialogue_turns,
            concat_mp3_chunks=concat_mp3_chunks,
            mp3_duration_ms=mp3_duration_ms,
            extract_timings=_extract_timings,
            fail_provider=lambda message: fail(message, _EXIT_PROVIDER),
            fail_output=lambda message: fail(message, _EXIT_OUTPUT),
            fail_missing_dep=lambda message: fail(message, _EXIT_MISSING_DEP),
            fail_whisper=lambda message: fail(message, _EXIT_WHISPER),
        ),
    )
    _emit_json_event(args, "run_complete", run_id=paths.prefix, duration_ms=summary.duration_ms)
    if args.json_output:
        _json_ok(
            {
                "status": "success",
                "provider": args.provider,
                "model": args.model,
                "run_id": paths.prefix,
                "files": summary.files,
                "duration_ms": summary.duration_ms,
                "segment_count": summary.segment_count,
                "cost": {"total": summary.cost_total, "currency": summary.cost_currency},
            }
        )
    else:
        print(f"Full MP3: {paths.full_mp3}")
        print(f"Run manifest: {paths.run_json}")
        print(f"Manifest: {paths.output_root / 'manifest.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# split / timings / doctor / validate / list
# ═══════════════════════════════════════════════════════════════════════════════


def split_cmd(args: argparse.Namespace) -> None:
    try:
        chunks = prepare_split_chunks(Path(args.script), args.delimiter)
    except ScriptNotFoundError as exc:
        fail(str(exc), _EXIT_ARGS)
    if args.json_output:
        _json_ok(
            {"status": "success", "chunks": [{"id": c.id, "chars": len(c.text)} for c in chunks]}
        )
    else:
        for chunk in chunks:
            print(f"{chunk.id}: {len(chunk.text)} chars")


def _validate_asr_request_options(
    args: argparse.Namespace, spec, hints: ASRContextHints | None = None
) -> None:
    runtime = getattr(args, "runtime", "auto")
    if runtime not in ("auto", "python", "audio-cpp"):
        fail(f"ASR runtime choice is not supported: {runtime}", _EXIT_ARGS)
    if runtime == "audio-cpp" and args.device != "cuda":
        fail(
            f"ASR provider {spec.provider_id} requires device=cuda for runtime=audio-cpp; "
            "the native route is CUDA-only",
            _EXIT_ARGS,
        )
    if (
        spec.provider_id == "nemotron-local"
        and hints is not None
        and hints.context_text is not None
    ):
        fail(
            f"ASR provider {spec.provider_id} does not support context text; "
            "only its model-owned language prompt selection is supported",
            _EXIT_ARGS,
        )
    capabilities = spec.capabilities
    if not capabilities.batch_audio:
        fail(
            f"ASR provider {spec.provider_id} does not support finite batch audio",
            _EXIT_ARGS,
        )
    if args.device not in capabilities.device_modes:
        fail(
            f"ASR provider {spec.provider_id} does not support device={args.device}",
            _EXIT_ARGS,
        )
    if args.compute not in capabilities.compute_modes:
        fail(
            f"ASR provider {spec.provider_id} does not support compute={args.compute}",
            _EXIT_ARGS,
        )
    model_ids = {model["id"] for model in spec.models if "id" in model}
    if args.model and model_ids and args.model not in model_ids:
        fail(
            f"ASR provider {spec.provider_id} does not support model={args.model}",
            _EXIT_ARGS,
        )
    if args.language and not capabilities.forced_language:
        fail(
            f"ASR provider {spec.provider_id} does not support forced language selection",
            _EXIT_ARGS,
        )
    if getattr(args, "word_timestamps", False) and not capabilities.word_timestamps:
        fail(
            f"ASR provider {spec.provider_id} does not support word timestamps",
            _EXIT_ARGS,
        )


def _asr_result_payload(result, source_audio: Path) -> dict:
    execution = {
        "runtime": result.execution.runtime,
        "runtime_version": result.execution.runtime_version,
        "model_revision": result.execution.model_revision,
        "device": result.execution.resolved_device,
        "compute": result.execution.resolved_compute,
        "measurements": dict(result.execution.measurements),
    }
    if result.execution.raw_timestamp_entries:
        execution["raw_timestamp_entries"] = [
            dict(entry) for entry in result.execution.raw_timestamp_entries
        ]
    if result.execution.long_form is not None:
        execution["long_form"] = dict(result.execution.long_form)
    return {
        "status": "success",
        "provider": result.provider_id,
        "model": result.model_id,
        "transcript": result.transcript,
        "language": result.language,
        "duration_s": result.duration_s,
        "source_audio": str(source_audio.resolve()),
        "timestamp_mode": result.alignment_origin or "none",
        "segments": [
            {"text": segment.text, "start_s": segment.start_s, "end_s": segment.end_s}
            for segment in result.segments
        ],
        "words": [
            {
                "text": word.text,
                "start_s": word.start_s,
                "end_s": word.end_s,
                "confidence": word.confidence,
            }
            for word in result.words
        ],
        "execution": execution,
    }


def _resolve_asr_context(args: argparse.Namespace) -> ASRContextHints:
    context_text = getattr(args, "context", None)
    context_file = getattr(args, "context_file", None)
    if context_file is not None:
        context_path = Path(context_file)
        try:
            context_text = context_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            fail(f"Unable to read ASR context file: {context_path}", _EXIT_ARGS)
    if context_text is not None and not context_text.strip():
        fail("ASR context must not be blank", _EXIT_ARGS)
    return ASRContextHints(context_text=context_text)


def _transcribe_result(args: argparse.Namespace) -> tuple[ASRResult, Path]:
    audio_path = _resolve_audio(args.audio)
    if not audio_path.exists():
        fail(f"Audio file not found: {audio_path}", _EXIT_ARGS)
    hints = _resolve_asr_context(args)
    try:
        spec = get_asr_provider_spec(args.provider)
    except ASRProviderNotFoundError as exc:
        fail(str(exc), _EXIT_ARGS)

    _validate_asr_request_options(args, spec, hints)
    request = build_asr_request(args, spec, hints, audio_path)
    try:
        provider = transcription.resolve_asr_provider(spec, request)
    except transcription.ASRDependencyUnavailableError as exc:
        fail(str(exc), _EXIT_MISSING_DEP)
    try:
        result = transcription.transcribe_asr_request(
            provider,
            spec,
            request,
            long_form=uses_long_form_orchestration(spec.provider_id),
            long_form_transcribe=transcribe_prerecorded_long_form,
            capability_check=True,
        )
    except LongFormASRMediaError as exc:
        fail(str(exc), _EXIT_NO_FFMPEG)
    except ModuleNotFoundError as exc:
        fail(f"Missing dependency for ASR provider {spec.provider_id}: {exc}", _EXIT_MISSING_DEP)
    except transcription.ASRCapabilityError as exc:
        fail(str(exc), _EXIT_PROVIDER)
    except Exception as exc:
        fail(f"ASR provider {spec.provider_id} failed: {exc}", _EXIT_PROVIDER)

    return result, audio_path


def _persist_history(
    build_save: Callable[[], AsrHistorySave | None],
) -> AsrHistorySaveResult:
    """Persist one completed local result without ever raising into the command.

    ``build_save`` returns the evidence to store, or ``None`` when history does
    not apply to this command or route (history disabled in ``settings.toml``, or
    a cloud timing route that stays legacy). Every failure -- an unreadable
    settings file, a managed-home or database failure, or a validation error --
    reports the same fixed machine-visible persistence failure instead of
    claiming a save. The provider response the command already produced is kept
    and no provider is called again.
    """
    try:
        if not settings_module.load_history_settings().enabled:
            return inactive_history_result()
        save = build_save()
        if save is None:
            return inactive_history_result()
        return saved_history_result(persist_asr_history(save))
    except Exception:
        return failed_history_result()


def _report_persistence_failure(*, json_output: bool) -> None:
    """Exit with the fixed persistence failure code after the result was emitted."""
    if not json_output:
        print(
            "History was not saved; the observed result above is unchanged "
            "(HISTORY_PERSISTENCE_FAILED).",
            file=sys.stderr,
        )
    sys.exit(_EXIT_OUTPUT)


def _transcript_completeness(transcript: str) -> str:
    """Mark a blank transcript ``incomplete`` so it never reads as full text."""
    return TEXT_COMPLETENESS_COMPLETE if transcript.strip() else TEXT_COMPLETENESS_INCOMPLETE


def _context_provenance(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Return the ASR context prompt and how it was supplied, if at all.

    ``(None, None)`` means the caller supplied no context. The text is re-resolved
    from the already validated arguments so the persisted prompt is exactly the
    one the request carried; a resolved blank value is treated as absent.
    """
    if getattr(args, "context", None) is not None:
        source = "inline"
    elif getattr(args, "context_file", None) is not None:
        source = "file"
    else:
        return None, None
    context_text = _resolve_asr_context(args).context_text
    if context_text is None or not context_text.strip():
        return None, None
    return context_text, source


def _asr_run_snapshot(
    result: ASRResult,
    *,
    context_source: str | None,
    context_prompt_present: bool,
) -> dict[str, Any]:
    """Build the bounded non-secret provenance map stored on an ASR run.

    Only observed values are copied: the reported timestamp origin, the actual
    segments with their own ``null`` bounds, and the runtime receipt. A text-only
    result therefore carries no span it did not report.
    """
    execution = result.execution
    snapshot: dict[str, Any] = {
        "operation_origin": NATIVE_ASR_ORIGIN,
        "provider": result.provider_id,
        "model": result.model_id,
        "language": result.language or None,
        "runtime": execution.runtime,
        "runtime_version": execution.runtime_version,
        "model_revision": execution.model_revision,
        "device": execution.resolved_device,
        "compute": execution.resolved_compute,
        "timestamp_mode": result.alignment_origin or "none",
        "duration_s": result.duration_s,
        "segment_count": len(result.segments),
        "segments_with_timestamps": sum(
            1 for segment in result.segments if segment.start_s is not None
        ),
        "segments": [
            {"text": segment.text, "start_s": segment.start_s, "end_s": segment.end_s}
            for segment in result.segments
        ],
        "word_count": len(result.words),
        "context_source": context_source,
        "context_prompt_present": context_prompt_present,
        "runtime_measurements": dict(execution.measurements) or None,
    }
    if execution.long_form is not None:
        # The long-form receipt is already JSON-shaped; normalize it through JSON
        # so the history boundary can redact a copy of a plain structure only.
        snapshot["long_form"] = json.loads(json.dumps(dict(execution.long_form)))
    return snapshot


def _transcribe_history_save(
    args: argparse.Namespace, result: ASRResult, audio_path: Path
) -> AsrHistorySave:
    """Build the evidence one completed local ``transcribe`` run contributes."""
    context_text, context_source = _context_provenance(args)
    texts = [
        AsrHistoryText(
            kind=TEXT_KIND_ASR_TRANSCRIPT,
            content=result.transcript,
            language=result.language or None,
            text_completeness=_transcript_completeness(result.transcript),
        )
    ]
    if context_text is not None:
        texts.append(
            AsrHistoryText(
                kind=TEXT_KIND_ASR_CONTEXT,
                content=context_text,
                text_completeness=TEXT_COMPLETENESS_COMPLETE,
            )
        )
    return AsrHistorySave(
        operation=OPERATION_ASR,
        attempt_call_type=ATTEMPT_CALL_TYPE_ASR,
        provider=result.provider_id,
        model=result.model_id,
        config_snapshot=_asr_run_snapshot(
            result,
            context_source=context_source,
            context_prompt_present=context_text is not None,
        ),
        text_sources=tuple(texts),
        source_audio=audio_path,
    )


def transcribe_cmd(args: argparse.Namespace) -> None:
    result, audio_path = _transcribe_result(args)

    data = _asr_result_payload(result, audio_path)
    save = _persist_history(lambda: _transcribe_history_save(args, result, audio_path))
    history_block = save.metadata()
    if history_block is not None:
        data["history"] = history_block
        if not save.saved:
            data["status"] = "partial"
    if args.json_output:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(result.transcript)
    if history_block is not None and not save.saved:
        _report_persistence_failure(json_output=args.json_output)
    if args.json_output:
        sys.exit(_EXIT_OK)


def _expected_tts_text(args: argparse.Namespace) -> str:
    expected_text = getattr(args, "expected_text", None)
    expected_file = getattr(args, "expected_file", None)
    if expected_file is not None:
        try:
            expected_text = Path(expected_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            fail(f"Unable to read expected TTS text file: {expected_file}", _EXIT_ARGS)
    if expected_text is None or not expected_text.strip():
        fail("Expected TTS text must not be blank", _EXIT_ARGS)
    return expected_text


def _verify_history_save(
    result: ASRResult,
    audio_path: Path,
    receipt_path: Path | None,
    quality,
) -> AsrHistorySave:
    """Build the evidence one ``verify-tts`` run contributes.

    The observed verification transcript is stored as its own private
    ``verification_transcript`` role; the expected TTS text is deliberately never
    stored, so a check can never overwrite or impersonate the source script. The
    audio's parent directory is offered as the parent run root so audio verified
    inside a native TTS run root links to that run, and the content-free quality
    receipt is referenced only when the caller asked for and got one.
    """
    execution = result.execution
    artifacts: list[AsrHistoryArtifact] = []
    if receipt_path is not None:
        artifacts.append(
            AsrHistoryArtifact(role=ARTIFACT_ROLE_QUALITY_RECEIPT, path=Path(receipt_path))
        )
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
        "receipt_recorded": receipt_path is not None,
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
                text_completeness=_transcript_completeness(result.transcript),
            ),
        ),
        artifacts=tuple(artifacts),
        source_audio=audio_path,
        parent_run_root=audio_path.parent,
    )


def verify_tts_cmd(args: argparse.Namespace) -> None:
    args.context = None
    args.context_file = None
    args.word_timestamps = False
    expected_text = _expected_tts_text(args)
    result, audio_path = _transcribe_result(args)
    quality = evaluate_tts_transcript(
        expected_text=expected_text,
        actual_transcript=result.transcript,
    )
    receipt = quality.public_receipt(
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        asr_provider=result.provider_id,
        asr_model=result.model_id,
        asr_runtime=result.execution.runtime,
        asr_model_revision=result.execution.model_revision,
    )
    receipt["status"] = "success" if quality.passed else "quality_failed"
    receipt_path = getattr(args, "receipt", None)
    if receipt_path is not None:
        atomic_write_json(Path(receipt_path), receipt)
    save = _persist_history(lambda: _verify_history_save(result, audio_path, receipt_path, quality))
    history_block = save.metadata()
    if history_block is not None:
        receipt["history"] = history_block
    if args.json_output:
        print(json.dumps(receipt, ensure_ascii=False))
    else:
        print("TTS quality PASS" if quality.passed else "TTS quality FAIL")
    if history_block is not None and not save.saved:
        _report_persistence_failure(json_output=args.json_output)
    sys.exit(_EXIT_OK if quality.passed else _EXIT_QUALITY)


def _preflight_tts_quality_provider(args: argparse.Namespace) -> None:
    """Validate the selected ASR route before any paid dialogue request."""
    provider_id = args.tts_quality_provider
    if provider_id == "xai-stt":
        read_xai_key()
        return
    try:
        spec = get_asr_provider_spec(provider_id)
    except ASRProviderNotFoundError as exc:
        fail(str(exc), _EXIT_ARGS)
    health = spec.dependency_probe()
    if not health.available:
        fail(health.remediation, _EXIT_MISSING_DEP)


def _transcribe_dialogue_quality_audio(
    args: argparse.Namespace, audio_path: Path
) -> tuple[str, str, str | None, str, str | None]:
    """Return transcript plus content-free ASR identity for a quality check."""
    return transcription.transcribe_dialogue_quality_audio(
        args=args,
        audio_path=audio_path,
        transcribe_result=_transcribe_result,
    )


def _verify_dialogue_turns_before_concat(
    args: argparse.Namespace,
    chunks: list[ScriptChunk],
    chunk_artifacts: list[ChunkArtifact],
    paths,
) -> dict[str, Any] | None:
    """Transcribe and strictly verify every dialogue turn before final concat."""
    return transcription.verify_dialogue_turns_before_concat(
        args=args,
        chunks=chunks,
        chunk_artifacts=chunk_artifacts,
        paths=paths,
        transcribe_quality_audio=_transcribe_dialogue_quality_audio,
        sha256_file=_sha256_file,
        fail_quality=lambda message, receipt: fail(message, _EXIT_QUALITY, details=receipt),
    )


def _timings_history_save(
    args: argparse.Namespace, audio_path: Path, files: dict[str, str]
) -> AsrHistorySave | None:
    """Build the evidence one completed local ``timings`` run contributes.

    Only the local routes are persisted here: the registered (therefore local) ASR
    provider route and ``faster-whisper``. Any other ``--timing-provider`` returns
    ``None``; the standalone cloud timing route is not written by this helper at
    all -- it commits its paid attempt, private raw body, and transcript through
    the paid-transcription boundary, so this writer never claims a cloud outcome it
    cannot prove. The transcript and provenance are read back from the timings JSON
    artifact the command just wrote, so the stored text is exactly what the durable
    artifact holds and a text-only route never gains an invented span.
    """
    if args.asr_provider:
        provider_id = args.asr_provider
        timestamp_basis = "asr_word_spans"
    else:
        if args.timing_provider != "faster-whisper":
            return None
        provider_id = args.timing_provider
        timestamp_basis = "provider_segment_timestamps"
    manifest = json.loads(Path(files["timings_json"]).read_text(encoding="utf-8"))
    segments = manifest.get("segments") or []
    transcript = " ".join(str(segment.get("text") or "") for segment in segments).strip()
    snapshot: dict[str, Any] = {
        "operation_origin": NATIVE_ASR_ORIGIN,
        "provider": manifest.get("provider") or provider_id,
        "model": manifest.get("model"),
        "backend": manifest.get("backend"),
        "device": manifest.get("device"),
        "compute_type": manifest.get("compute_type"),
        "language": manifest.get("language") or None,
        "timestamp_basis": timestamp_basis,
        "word_timestamps_requested": bool(getattr(args, "word_timestamps", False)),
        "segment_count": len(segments),
        "total_duration_ms": manifest.get("total_duration_ms"),
        "source_audio": str(audio_path.resolve()),
        "timings_json": files["timings_json"],
        "srt": files["srt"],
    }
    return AsrHistorySave(
        operation=OPERATION_TIMINGS,
        attempt_call_type=ATTEMPT_CALL_TYPE_TIMING,
        provider=provider_id,
        model=manifest.get("model") or args.model,
        config_snapshot=snapshot,
        text_sources=(
            AsrHistoryText(
                kind=TEXT_KIND_ASR_TRANSCRIPT,
                content=transcript,
                language=manifest.get("language") or None,
                text_completeness=_transcript_completeness(transcript),
            ),
        ),
        artifacts=(
            AsrHistoryArtifact(
                role=ARTIFACT_ROLE_TIMINGS_JSON,
                path=Path(files["timings_json"]),
                mime="application/json",
            ),
            AsrHistoryArtifact(
                role=ARTIFACT_ROLE_SRT,
                path=Path(files["srt"]),
                mime="application/x-subrip",
            ),
        ),
        source_audio=audio_path,
    )


_PAID_TIMING_PROVIDERS = frozenset({"groq-whisper", "xai-stt"})

# The adapter default model each paid timing provider uses when the caller omits
# ``--model``; kept in sync with ``services.transcription.transcribe_timing_audio``.
_PAID_TIMING_DEFAULT_MODELS = {"groq-whisper": "whisper-large-v3-turbo", "xai-stt": "grok-stt"}


def _history_configured_enabled() -> bool:
    """Whether canonical history is enabled, failing closed on a bad setting."""
    try:
        return bool(settings_module.load_history_settings().enabled)
    except Exception:
        return False


def _paid_timing_model(timing_provider: str, model: str | None) -> str:
    """Return the effective model a paid timing request will send."""
    return model or _PAID_TIMING_DEFAULT_MODELS.get(timing_provider, "")


def _paid_timing_request_options(timing_provider: str, word_timestamps: bool) -> dict[str, Any]:
    """Return the exact provider request shape the adapter will send."""
    if timing_provider == "groq-whisper":
        granularities = ["segment", "word"] if word_timestamps else ["segment"]
        return {"response_format": "verbose_json", "timestamp_granularities": granularities}
    return {"format": "true"}


def _reserve_paid_timing(
    args: argparse.Namespace, audio_path: Path, output_dir: Path, run_id: str
) -> paid_transcription_history.PaidTranscriptionReservation:
    """Commit the paid timing attempt marker before the adapter may POST."""
    provider = args.timing_provider
    request = paid_transcription_history.PaidTranscriptionRequest(
        call_type=paid_transcription_history.ATTEMPT_CALL_TYPE_PAID_TIMING,
        provider=provider,
        model=_paid_timing_model(provider, args.model),
        language=args.language,
        word_timestamps=bool(args.word_timestamps),
        source_audio=audio_path,
        output_root=output_dir,
        request_options={
            **_paid_timing_request_options(provider, bool(args.word_timestamps)),
            "timestamp_granularity": "word" if args.word_timestamps else "segment",
            "output_dir": str(output_dir),
            "output_prefix": run_id,
        },
        user_label=run_id,
    )
    try:
        return paid_transcription_history.reserve_paid_transcription(request)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})


def _complete_paid_timing(
    reservation: paid_transcription_history.PaidTranscriptionReservation, files: dict[str, str]
) -> None:
    """Record the observed transcript and timing artifacts and close the run."""
    manifest = json.loads(Path(files["timings_json"]).read_text(encoding="utf-8"))
    segments = manifest.get("segments") or []
    transcript = " ".join(str(segment.get("text") or "") for segment in segments).strip()
    artifacts = (
        AsrHistoryArtifact(
            role=ARTIFACT_ROLE_TIMINGS_JSON,
            path=Path(files["timings_json"]),
            mime="application/json",
            media_metadata={
                "timestamp_basis": manifest.get("timestamp_basis") or "unknown",
                "segment_count": len(segments),
                "total_duration_ms": manifest.get("total_duration_ms"),
            },
        ),
        AsrHistoryArtifact(
            role=ARTIFACT_ROLE_SRT,
            path=Path(files["srt"]),
            mime="application/x-subrip",
        ),
    )
    text_sources = (
        AsrHistoryText(
            kind=TEXT_KIND_ASR_TRANSCRIPT,
            content=transcript,
            language=manifest.get("language") or None,
            text_completeness=_transcript_completeness(transcript),
        ),
    )
    metadata = {
        "timestamp_basis": manifest.get("timestamp_basis") or "unknown",
        "segment_count": len(segments),
        "total_duration_ms": manifest.get("total_duration_ms"),
        "backend": manifest.get("backend"),
        "cost_known": False,
    }
    try:
        paid_transcription_history.complete_paid_transcription(
            reservation,
            artifacts=artifacts,
            text_sources=text_sources,
            result_metadata=metadata,
            required_artifact_roles=(ARTIFACT_ROLE_TIMINGS_JSON, ARTIFACT_ROLE_SRT),
        )
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})


def run_timings(args: argparse.Namespace) -> None:
    try:
        check_media_tools()
    except RuntimeError as e:
        fail(str(e), _EXIT_NO_FFMPEG)
    audio_path = _resolve_audio(args.audio)
    if not audio_path.exists():
        fail(f"Audio file not found: {audio_path}", _EXIT_ARGS)
    if args.run_id:
        _validate_run_id(args.run_id)
    _validate_output_dir(args.output_dir)
    run_id = args.run_id or audio_path.stem
    # Keep the user-supplied output leaf lexically, before ``resolve()`` follows
    # it. A symlinked leaf is refused here instead of being silently redirected to
    # whatever target it points at, and the same locator is rechecked under the
    # run lock before any reservation or publication.
    lexical_output = Path(args.output_dir).expanduser() / run_id
    _reject_symlinked_output_leaf(lexical_output)
    output_dir = lexical_output.resolve()

    cloud_timing = (
        not args.asr_provider
        and getattr(args, "timing_provider", "faster-whisper") in _PAID_TIMING_PROVIDERS
    )
    if cloud_timing:
        _run_cloud_timing(args, audio_path, lexical_output, output_dir, run_id)
        return
    _run_local_timings(args, audio_path, lexical_output, output_dir, run_id)


def _reject_symlinked_output_leaf(output_leaf: Path) -> None:
    """Refuse a user-supplied output leaf that is a symlink, before it is followed."""
    if output_leaf.is_symlink():
        fail(
            "Refusing to use a symlinked timings output directory.",
            _EXIT_OUTPUT,
            details={"error_code": "PAID_TIMING_OWNERSHIP_UNVERIFIABLE"},
        )


def _guard_foreign_timings_root(output_leaf: Path) -> None:
    """Fail closed when a timings output root belongs to another route's writer.

    Ownership is bound to the canonical output root, so *every* fresh timings
    invocation -- including ``--overwrite`` -- against a paid-owned root (a still
    ``submitting`` run or a completed one), or against a native-history-owned
    root, stops before any deletion, key access, or POST. The caller resuming or
    syncing keeps using that run's UUID; the lexical leaf is checked first so a
    leaf swapped to a symlink after the command started is refused rather than
    followed to a fresh target.
    """
    _reject_symlinked_output_leaf(output_leaf)
    try:
        ownership = paid_transcription_history.resolve_paid_timing_ownership(output_leaf)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    if ownership.owned:
        fail(
            "This output directory is owned by a paid timing run; a fresh invocation or "
            "--overwrite cannot replace it. Use `history resume`/`history sync` with its "
            "UUID, or choose a different --run-id.",
            _EXIT_OUTPUT,
            details={"error_code": "PAID_TIMING_OUTPUT_OWNED", "run_uuid": ownership.run_uuid},
        )
    try:
        native_decision = native_generation.resolve_native_ownership(output_leaf)
    except native_generation.NativeGenerationError as exc:
        fail(str(exc), exc.code, details={"error_code": exc.error_code})
    if native_decision.route != "legacy":
        fail(
            "This output directory is owned by native history; a timings run cannot replace "
            "its committed run or local evidence. Choose a different --run-id.",
            _EXIT_OUTPUT,
            details={
                "error_code": native_decision.error_code or "NATIVE_TIMING_OUTPUT_OWNED",
                "run_uuid": native_decision.run_uuid,
            },
        )


def _reject_source_inside_output(audio_path: Path, output_dir: Path) -> None:
    """Refuse to delete an output directory that contains the source audio."""
    try:
        resolved_audio = audio_path.resolve()
        resolved_output = output_dir.resolve()
        resolved_audio.relative_to(resolved_output)
    except ValueError:
        return
    fail(
        f"Refusing to remove output directory {resolved_output}: the source audio is inside it.",
        _EXIT_OUTPUT,
        details={"error_code": "PAID_SOURCE_INSIDE_OUTPUT"},
    )


def _handle_timings_output_dir(
    args: argparse.Namespace, output_dir: Path, run_id: str, audio_path: Path
) -> bool:
    """Handle an already-existing timings output dir; return True when handled.

    ``--skip-existing`` reports the existing run and stops. Without ``--overwrite``
    an existing directory is a usage error. With ``--overwrite`` the source audio
    is proven not to live inside the directory before it is removed.
    """
    if not output_dir.exists():
        return False
    if args.skip_existing:
        timing_json = output_dir / f"{run_id}.timings.json"
        srt_path = output_dir / f"{run_id}.srt"
        files = {"timings_json": str(timing_json), "srt": str(srt_path)}
        if args.json_output:
            _json_ok(
                {
                    "status": "skipped",
                    "reason": "output dir exists",
                    "run_id": run_id,
                    "files": files,
                }
            )
        print(f"Skipping: output dir exists: {output_dir}")
        return True
    if not args.overwrite:
        fail(
            f"Output dir exists: {output_dir}. Use --overwrite or --skip-existing.",
            _EXIT_PROVIDER,
        )
    _reject_source_inside_output(audio_path, output_dir)
    _safe_remove_run_dir(output_dir, args.output_dir)
    return False


def _print_timings_result(
    args: argparse.Namespace,
    files: dict[str, str],
    timing: dict[str, Any],
    history_block: dict[str, Any] | None,
    persistence_ok: bool,
) -> None:
    if args.json_output:
        payload: dict[str, Any] = {
            "status": "success" if persistence_ok else "partial",
            "files": files,
            "segment_count": timing["segment_count"],
            "duration_ms": timing["total_duration_ms"],
        }
        if history_block is not None:
            payload["history"] = history_block
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"Timings JSON: {files['timings_json']}")
        print(f"SRT: {files['srt']}")
        print(f"Segments: {timing['segment_count']}")
    if not persistence_ok:
        _report_persistence_failure(json_output=args.json_output)
    if args.json_output:
        sys.exit(_EXIT_OK)


def _run_local_timings(
    args: argparse.Namespace,
    audio_path: Path,
    output_leaf: Path,
    output_dir: Path,
    run_id: str,
) -> None:
    """Run a local (or legacy cloud) timings route under the shared run lock.

    The local route deletes and writes the canonical output root, so it holds the
    same cross-process run lock the paid and native routes take. Ownership is
    re-resolved under that lock, so a paid reservation committed after the early
    lexical check -- even with ``--overwrite`` -- is never deleted or overwritten.
    The local history writer takes no lock of its own, so this does not double-lock.
    """
    try:
        with acquire_run_lock(output_dir):
            _run_local_timings_locked(args, audio_path, output_leaf, output_dir, run_id)
    except HistoryRunLockedError:
        fail(
            "Another process is already writing this timings run directory; refusing to run "
            "two writers for one output root.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TIMING_RUN_LOCKED"},
        )


def _run_local_timings_locked(
    args: argparse.Namespace,
    audio_path: Path,
    output_leaf: Path,
    output_dir: Path,
    run_id: str,
) -> None:
    """Body of the local timings route, already holding the output-root lock."""
    _guard_foreign_timings_root(output_leaf)
    if _handle_timings_output_dir(args, output_dir, run_id, audio_path):
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Failed to create output directory {output_dir}: {e}", _EXIT_OUTPUT)
    try:
        if args.asr_provider:
            timing = _extract_asr_timings(
                audio_path=audio_path,
                output_dir=output_dir,
                prefix=run_id,
                provider_id=args.asr_provider,
                model=args.model,
                device=args.device,
                compute=args.compute or "auto",
                language=args.language,
            )
        else:
            timing = _extract_timings(
                audio_path=audio_path,
                output_dir=output_dir,
                prefix=run_id,
                timing_provider=args.timing_provider,
                model=args.model,
                device=args.device,
                compute_type=args.compute or DEFAULT_TIMING_COMPUTE,
                language=args.language,
                word_timestamps=args.word_timestamps,
                quiet=args.json_output,
            )
    except CliError:
        raise
    except ModuleNotFoundError as exc:
        fail(
            f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --extra timing-whisper",
            _EXIT_MISSING_DEP,
        )
    except Exception as exc:
        fail(f"Whisper timing failed: {exc}", _EXIT_WHISPER)

    files = {
        "timings_json": str(output_dir / f"{run_id}.timings.json"),
        "srt": str(output_dir / f"{run_id}.srt"),
    }
    save = _persist_history(lambda: _timings_history_save(args, audio_path, files))
    persistence_ok = save.saved or not save.active
    _print_timings_result(args, files, timing, save.metadata(), persistence_ok)


def _run_cloud_timing(
    args: argparse.Namespace,
    audio_path: Path,
    output_leaf: Path,
    output_dir: Path,
    run_id: str,
) -> None:
    """Run the standalone paid cloud timing route under the shared run lock.

    The history-disabled and readable-source checks run before any lock, directory
    creation, deletion, or key access. The canonical output root's cross-process
    lock is then held across ownership selection, reservation, POST, persistence,
    and publication, so two invocations can never issue two paid submits. The
    lexical output leaf is rechecked under that lock so a leaf swapped to a
    symlink after the command started is refused rather than redirected.
    """
    if not _history_configured_enabled():
        fail(
            "Cloud timing providers require canonical history so their paid-submit marker "
            "can be committed before the request; history is disabled in settings.toml.",
            _EXIT_OUTPUT,
            details={"error_code": "PAID_TIMING_HISTORY_REQUIRED"},
        )
    try:
        source_identity = paid_transcription_history.require_readable_source(audio_path)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_ARGS, details={"error_code": exc.error_code})
    try:
        with acquire_run_lock(output_dir):
            _run_cloud_timing_locked(
                args, audio_path, output_leaf, output_dir, run_id, source_identity
            )
    except HistoryRunLockedError:
        fail(
            "Another process is already writing this timings run directory; refusing to run "
            "two writers for one output root.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TIMING_RUN_LOCKED"},
        )


def _run_cloud_timing_locked(
    args: argparse.Namespace,
    audio_path: Path,
    output_leaf: Path,
    output_dir: Path,
    run_id: str,
    source_identity: tuple[str, int, str],
) -> None:
    """Body of the paid cloud timing route, already holding the output-root lock."""
    _guard_foreign_timings_root(output_leaf)
    if _handle_timings_output_dir(args, output_dir, run_id, audio_path):
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Failed to create output directory {output_dir}: {e}", _EXIT_OUTPUT)

    reservation = _reserve_paid_timing(args, audio_path, output_dir, run_id)
    raw_sink = lambda body, content_type: (  # noqa: E731 - bound once for the adapter sink
        paid_transcription_history.record_paid_transcription_raw(
            reservation, body=body, content_type=content_type
        )
    )
    artifact_writer = lambda audio, out, prefix, timing: _write_paid_timing_artifacts(  # noqa: E731
        audio, out, prefix, timing, source_identity
    )
    try:
        timing = _extract_timings(
            audio_path=audio_path,
            output_dir=output_dir,
            prefix=run_id,
            timing_provider=args.timing_provider,
            model=args.model,
            device=args.device,
            compute_type=args.compute or DEFAULT_TIMING_COMPUTE,
            language=args.language,
            word_timestamps=args.word_timestamps,
            quiet=args.json_output,
            on_raw_response=raw_sink,
            artifact_writer=artifact_writer,
        )
    except CliError:
        raise
    except ModuleNotFoundError as exc:
        fail(
            f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --extra timing-whisper",
            _EXIT_MISSING_DEP,
        )
    except Exception as exc:
        # A cloud POST that failed or timed out leaves the attempt marker in
        # ``submitting``; the command reports the failure and never resubmits.
        fail(f"Whisper timing failed: {exc}", _EXIT_WHISPER)

    files = {
        "timings_json": str(output_dir / f"{run_id}.timings.json"),
        "srt": str(output_dir / f"{run_id}.srt"),
    }
    _complete_paid_timing(reservation, files)
    history_block: dict[str, Any] | None = {"saved": True, "run_uuid": reservation.run_uuid}
    _print_timings_result(args, files, timing, history_block, True)


def status_cmd(args: argparse.Namespace) -> None:
    if args.run_id:
        _validate_run_id(args.run_id)
    run_dir = (Path(args.output_dir) / args.run_id).resolve()
    state, state_unreadable = _load_status_state(run_dir / STATE_FILE)
    chunks_dir = run_dir / "chunks"
    chunk_files = _continuous_chunk_files(chunks_dir)
    total = int(state.get("chunk_count", len(chunk_files)) if state else len(chunk_files))
    ready = int(state.get("completed_count", len(chunk_files)) if state else len(chunk_files))
    completed = completed_numbers(state)
    if state:
        ready = len(completed)
    next_chunk = ready + 1 if total == 0 or ready < total else None
    full_audio = _find_full_audio(run_dir)
    timings_json = list(run_dir.glob("*.timings.json"))
    errors = state.get("errors", []) if state else []
    can_resume = run_dir.exists() and ready < total and bool(state or chunk_files)
    resume_block_reason = None
    if state_unreadable:
        can_resume = False
        resume_block_reason = _RUN_STATE_UNREADABLE_RESUME_BLOCK_REASON
    elif unconfirmed_attempt(state) is not None:
        # A marker holding saved raw audio or a stored paid media id resumes
        # without a new submit, but only when every earlier chunk is genuinely
        # finished: state-completed and still on disk. A chunk whose MP3 is
        # missing is regenerated, and that paid submit would collide with the
        # stored attempt, so status reports the same block --resume would hit
        # instead of promising a resume.
        raw_recovery = _known_raw_recovery(state, run_dir, chunks_dir)
        media_recovery = _known_media_recovery(state)
        prior_ids = _state_completed_chunk_ids(state)
        raw_ready = raw_recovery is not None and _preceding_chunks_ready(
            prior_ids, chunks_dir, completed, raw_recovery["number"]
        )
        media_ready = media_recovery is not None and _preceding_chunks_ready(
            prior_ids, chunks_dir, completed, media_recovery["number"]
        )
        if not raw_ready and not media_ready:
            can_resume = False
            resume_block_reason = _PAID_SUBMIT_RESUME_BLOCK_REASON
    data = {
        "status": "success",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "total_chunks": total,
        "completed_chunks": ready,
        "next_chunk": next_chunk,
        "full_audio_exists": bool(full_audio),
        "full_audio": str(full_audio) if full_audio else None,
        "timings_exists": bool(timings_json),
        "errors": errors,
        "can_resume": can_resume,
        "resume_block_reason": resume_block_reason,
    }
    if args.json_output:
        _json_ok(data)
    print(f"Run: {args.run_id}")
    print(f"Chunks: {ready} of {total}")
    print(f"Next chunk: {next_chunk if next_chunk is not None else 'none'}")
    print(f"Full audio: {'yes' if full_audio else 'no'}")
    print(f"Timings: {'yes' if timings_json else 'no'}")
    print(f"Errors: {len(errors)}")
    print(f"Can resume: {'yes' if can_resume else 'no'}")
    if resume_block_reason is not None:
        print(f"Resume blocked: {resume_block_reason}")


def concat_cmd(args: argparse.Namespace) -> None:
    try:
        ffmpeg_path, _ffprobe_path = check_media_tools()
    except RuntimeError as e:
        fail(str(e), _EXIT_NO_FFMPEG)
    _validate_run_id(args.run_id)
    run_dir = (Path(args.output_dir) / args.run_id).resolve()
    state = load_state(run_dir / STATE_FILE)
    chunks_dir = run_dir / "chunks"
    chunk_files = _continuous_chunk_files(chunks_dir)
    if not chunk_files:
        fail(f"No contiguous chunk_*.mp3 files found in {chunks_dir}", _EXIT_ARGS)
    total = int(state.get("chunk_count", len(chunk_files)) if state else len(chunk_files))
    ready = len(chunk_files)
    kind = "full" if ready >= total else "partial"
    output_path = run_dir / f"{kind}-{ready}-of-{total}.{args.format}"
    try:
        concat_audio_files(ffmpeg_path, chunk_files, output_path)
    except Exception as e:
        fail(f"Failed to concat existing chunks: {e}", _EXIT_OUTPUT)
    data = {
        "status": "success",
        "run_id": args.run_id,
        "partial": ready < total,
        "completed_chunks": ready,
        "total_chunks": total,
        "file": str(output_path),
    }
    if args.json_output:
        _json_ok(data)
    print(f"Wrote {output_path}")
    if ready < total:
        print(f"Partial file: {ready} of {total} chunks")


def doctor_cmd(args: argparse.Namespace) -> None:
    results: dict[str, dict] = {}

    results["python"] = {"ok": True, "version": sys.version.split()[0], "required": True}

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    results["ffmpeg"] = {"ok": bool(ffmpeg), "path": ffmpeg, "required": True}
    results["ffprobe"] = {"ok": bool(ffprobe), "path": ffprobe, "required": True}

    env_file = Path.cwd() / ".env"
    results["env_file"] = {"ok": env_file.exists(), "path": str(env_file), "required": True}

    need_polza = (args.provider or DEFAULT_PROVIDER) in ("polza-chat-audio", "polza-tts")
    polza_ok = False
    try:
        read_polza_key()
        polza_ok = True
    except Exception:
        pass
    results["polza_key"] = {"ok": polza_ok, "required": need_polza}

    need_or = (args.provider or "") == "openrouter-tts"
    or_ok = False
    try:
        read_openrouter_key()
        or_ok = True
    except Exception:
        pass
    results["openrouter_key"] = {"ok": or_ok, "required": need_or}

    omnivoice_health = None
    if args.provider == "omnivoice-local":
        omnivoice_health = omnivoice_local_dependency_probe()
        results["omnivoice_local"] = {
            "ok": omnivoice_health.available,
            "required": True,
        }
        if not omnivoice_health.available:
            results["omnivoice_local"]["reason_code"] = (
                omnivoice_health.reason_code or "invalid_native_package"
            )

    need_whisper = bool(args.with_timings)
    timing_provider = getattr(args, "timing_provider", "faster-whisper")
    need_asr = bool(getattr(args, "with_asr", False))
    asr_health = None

    if need_asr:
        if not args.asr_provider:
            fail("--asr-provider is required with --with-asr", _EXIT_ARGS)
        try:
            asr_spec = get_asr_provider_spec(args.asr_provider)
        except ASRProviderNotFoundError as exc:
            fail(str(exc), _EXIT_ARGS)
        if args.asr_device not in asr_spec.capabilities.device_modes:
            fail(
                f"ASR provider {asr_spec.provider_id} does not support device={args.asr_device}",
                _EXIT_ARGS,
            )
        if args.asr_compute not in asr_spec.capabilities.compute_modes:
            fail(
                f"ASR provider {asr_spec.provider_id} does not support compute={args.asr_compute}",
                _EXIT_ARGS,
            )
        asr_health = asr_spec.dependency_probe()
        results["asr_provider"] = {
            "ok": asr_health.available,
            "provider": asr_spec.provider_id,
            "required": True,
        }
        if not asr_health.available:
            results["asr_provider"]["reason_code"] = asr_health.reason_code or "unavailable"

    if timing_provider == "openrouter-whisper":
        whisper_ok = False
        try:
            read_openrouter_key()
            whisper_ok = True
        except Exception:
            pass
        results["openrouter_whisper_key"] = {"ok": whisper_ok, "required": need_whisper}
    elif timing_provider == "groq-whisper":
        whisper_ok = False
        try:
            read_groq_key()
            whisper_ok = True
        except Exception:
            pass
        results["groq_whisper_key"] = {"ok": whisper_ok, "required": need_whisper}
    elif timing_provider == "xai-stt":
        whisper_ok = False
        try:
            read_xai_key()
            whisper_ok = True
        except Exception:
            pass
        results["xai_stt_key"] = {"ok": whisper_ok, "required": need_whisper}
    else:
        try:
            whisper_ok = importlib.util.find_spec("faster_whisper") is not None
        except ValueError:
            # Lightweight test doubles can exist in sys.modules without a module spec.
            whisper_ok = "faster_whisper" in sys.modules
        results["faster_whisper"] = {"ok": whisper_ok, "required": need_whisper}

    need_cuda = args.provider in {"qwen-local", "omnivoice-local"} or args.timing_device == "cuda"
    cuda_available = False
    if args.provider == "omnivoice-local":
        from .local_runtime.lifecycle import probe_local_gpu_state

        cuda_available = probe_local_gpu_state().probe_error is None
    else:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
    results["cuda"] = {"ok": cuda_available, "required": need_cuda}

    required_ok = all(info.get("ok", False) for info in results.values() if info.get("required"))
    optional_ok = all(
        info.get("ok", False) for info in results.values() if not info.get("required")
    )
    workflow_ok = required_ok

    warnings: list[str] = []
    if not cuda_available and not need_cuda:
        warnings.append(
            "CUDA is unavailable: qwen-local, omnivoice-local, and cuda timings will not work, but cloud TTS and CPU timings are OK."
        )
    if not cuda_available and need_cuda:
        warnings.append(
            "CUDA is unavailable but required for the selected provider or timing device."
        )
    if not whisper_ok and need_whisper:
        if timing_provider == "openrouter-whisper":
            warnings.append(
                "OPENROUTER_API_KEY is missing. Set it in .env: OPENROUTER_API_KEY=sk-or-v1-..."
            )
        elif timing_provider == "groq-whisper":
            warnings.append("GROQ_API_KEY is missing. Set it in .env: GROQ_API_KEY=gsk_...")
        elif timing_provider == "xai-stt":
            warnings.append("X_AI_API_KEY is missing. Set it in .env: X_AI_API_KEY=xai-...")
        else:
            warnings.append(
                "faster-whisper is not installed. Install with: uv sync --extra timing-whisper"
            )
    if asr_health is not None and not asr_health.available:
        warnings.append(asr_health.remediation)
    if omnivoice_health is not None and not omnivoice_health.available:
        warnings.append(omnivoice_health.remediation)
    if not polza_ok and need_polza:
        warnings.append("POLZA_API_KEY is missing. Set it in .env: POLZA_API_KEY=...")
    if not or_ok and need_or:
        warnings.append("OPENROUTER_API_KEY is missing. Set it in .env: OPENROUTER_API_KEY=...")

    if args.json_output:
        _json_ok(
            {
                "status": "success",
                "required_ok": required_ok,
                "optional_ok": optional_ok,
                "workflow_ok": workflow_ok,
                "checks": results,
                "warnings": warnings,
            }
        )
    else:
        for name, info in results.items():
            status = "OK" if info.get("ok") else "MISSING"
            req = "*required" if info.get("required") else "optional"
            print(f"  {name}: {status} ({req})")
        for w in warnings:
            print(f"  WARNING: {w}")


def validate_cmd(args: argparse.Namespace) -> None:
    script = Path(args.script)
    if not script.exists():
        fail("Script file not found", _EXIT_ARGS)

    script_format = _resolve_script_format(script, args.format)

    if is_dialogue_format(script_format):
        report = validate_gemini_dialogue_file(
            script,
            delimiter=args.delimiter,
            model=args.model,
            speaker_voice_overrides=args.speaker_voice,
            agent=args.agent,
        )
        if args.json_output:
            print(json.dumps(report, ensure_ascii=False))
            sys.exit(_EXIT_OK)
        print(f"Script: {script}")
        print(f"Format: {DIALOGUE_FORMAT}")
        print(f"Chunks: {report['chunks']}, Valid: {report['valid']}")
        for item in report["errors"]:
            loc = f" line {item.get('line') or item.get('line_start', '')}".rstrip()
            print(f"  ERROR {item['code']}{loc}: {item['message']}")
        for item in report["warnings"]:
            loc = f" line {item.get('line') or item.get('line_start', '')}".rstrip()
            print(f"  WARNING {item['code']}{loc}: {item['message']}")
        return

    if script_format == VOICEOVER_FORMAT:
        report = validate_voiceover_file(
            script,
            delimiter=args.delimiter,
            provider_override=args.provider,
            model_override=args.model,
            voice_override=args.voice,
            max_chunk_chars=args.max_chunk_chars,
            agent=args.agent,
        )
        if args.json_output:
            print(json.dumps(report, ensure_ascii=False))
            sys.exit(_EXIT_OK)
        print(f"Script: {script}")
        print(f"Format: {VOICEOVER_FORMAT}")
        print(f"Chunks: {report['chunks']}, Valid: {report['valid']}")
        for item in report["errors"]:
            loc = f" line {item.get('line') or item.get('line_start', '')}".rstrip()
            print(f"  ERROR {item['code']}{loc}: {item['message']}")
        for item in report["warnings"]:
            loc = f" line {item.get('line') or item.get('line_start', '')}".rstrip()
            print(f"  WARNING {item['code']}{loc}: {item['message']}")
        return

    text = script.read_text(encoding="utf-8-sig")
    parts = [p.strip() for p in text.split(args.delimiter)]
    chunk_list = [(i, p) for i, p in enumerate(parts, start=1) if p]

    max_chunk_chars = 2000 if args.max_chunk_chars is None else args.max_chunk_chars
    issues: list[dict] = []
    total_chars = 0
    for idx, chunk_text in chunk_list:
        chars = len(chunk_text)
        total_chars += chars
        if chars > max_chunk_chars:
            issues.append(
                {"chunk": idx, "type": "too_long", "chars": chars, "limit": max_chunk_chars}
            )

    warnings = []
    for idx, chunk_text in chunk_list:
        has_digits = any(ch.isdigit() for ch in chunk_text)
        if has_digits:
            warnings.append({"chunk": idx, "type": "contains_digits"})

    ok = len(issues) == 0

    if args.json_output:
        _json_ok(
            {
                "status": "success" if ok else "warning",
                "valid": ok,
                "chunks": len(chunk_list),
                "total_chars": total_chars,
                "issues": issues,
                "warnings": warnings,
            }
        )
    else:
        print(f"Script: {script}")
        print(f"Chunks: {len(chunk_list)}, Total chars: {total_chars}")
        print(f"Valid: {ok}")
        for issue in issues:
            print(
                f"  ISSUE chunk {issue['chunk']}: {issue['type']} ({issue['chars']} chars > {issue['limit']})"
            )


def list_cmd(args: argparse.Namespace) -> None:
    data: dict[str, Any]
    if args.target == "providers":
        data = {
            "providers": [
                {
                    "id": "polza-chat-audio",
                    "models": ["openai/gpt-audio-mini", "openai/gpt-audio"],
                    "currency": "RUB",
                },
                {"id": "polza-tts", "models": POLZA_TTS_MODELS, "currency": "RUB"},
                {"id": "openrouter-tts", "models": OPENROUTER_TTS_MODELS, "currency": "USD"},
                {
                    "id": "qwen-local",
                    "modes": ["preset", "clone"],
                    "currency": "RUB",
                    "cost": "free",
                },
                {
                    "id": "omnivoice-local",
                    "models": [OMNIVOICE_LOCAL_MODEL_ID],
                    "modes": ["auto", "preset", "clone", "design"],
                    "license": "CC-BY-NC-4.0 upstream weights; local noncommercial research only",
                },
            ]
        }
    elif args.target == "voices":
        provider = args.provider or "polza-chat-audio"
        voices_flat = {
            "polza-chat-audio": [
                "ash",
                "ballad",
                "coral",
                "verse",
                "marin",
                "cedar",
                "echo",
                "sage",
                "shimmer",
                "onyx",
            ],
            "polza-tts": OPENAI_TTS_VOICES + ELEVENLABS_TTS_VOICES,
            "openrouter-tts": GEMINI_TTS_VOICES,
            "qwen-local": QWEN_PRESET_SPEAKERS,
            "omnivoice-local": [],
        }
        voice_categories = {
            "polza-tts": {
                "openai": OPENAI_TTS_VOICES,
                "elevenlabs": ELEVENLABS_TTS_VOICES,
            },
            "openrouter-tts": {
                "gemini": GEMINI_TTS_VOICES,
            },
        }
        data = {"provider": provider, "voices": voices_flat.get(provider, [])}
        if provider == "omnivoice-local":
            bank_arg = getattr(args, "voice_bank", None)
            if bank_arg is not None:
                try:
                    catalog = load_voice_bank(Path(bank_arg))
                except VoiceBankError as exc:
                    fail(str(exc), _EXIT_ARGS)
                data["voices"] = [profile.id for profile in catalog.profiles]
                data["profiles"] = [
                    {
                        "id": profile.id,
                        "display_name": profile.display_name,
                        "description": profile.description,
                        "language": profile.language,
                    }
                    for profile in catalog.profiles
                ]
            data["voice_selection"] = {
                "kind": "built-in-style-condition",
                "condition": OMNIVOICE_STYLE_CONDITION,
                "named_preset": False,
                "voice_cloning": False,
                "voice_design": False,
            }
        if provider in voice_categories:
            data["voice_categories"] = voice_categories[provider]
    elif args.target == "timing-models":
        data = {
            "timing_models": [
                {"id": "base", "parameters_m": 74, "disk_mb": 148, "speed": "fastest"},
                {
                    "id": "small",
                    "parameters_m": 244,
                    "disk_mb": 486,
                    "speed": "fast",
                    "default": True,
                },
                {"id": "medium", "parameters_m": 769, "disk_mb": 1536, "speed": "balanced"},
                {"id": "large-v3-turbo", "parameters_m": 809, "disk_mb": 1620, "speed": "slow"},
                {"id": "large-v3", "parameters_m": 1550, "disk_mb": 3090, "speed": "slowest"},
            ]
        }
    elif args.target == "timing-providers":
        data = {
            "timing_providers": [
                {
                    "id": "faster-whisper",
                    "type": "local",
                    "models": [
                        {"id": "base", "parameters_m": 74, "disk_mb": 148, "speed": "fastest"},
                        {
                            "id": "small",
                            "parameters_m": 244,
                            "disk_mb": 486,
                            "speed": "fast",
                            "default": True,
                        },
                        {"id": "medium", "parameters_m": 769, "disk_mb": 1536, "speed": "balanced"},
                        {
                            "id": "large-v3-turbo",
                            "parameters_m": 809,
                            "disk_mb": 1620,
                            "speed": "slow",
                        },
                        {
                            "id": "large-v3",
                            "parameters_m": 1550,
                            "disk_mb": 3090,
                            "speed": "slowest",
                        },
                    ],
                },
                {
                    "id": "openrouter-whisper",
                    "type": "cloud",
                    "currency": "USD",
                    "models": [
                        {
                            "id": "openai/whisper-large-v3-turbo",
                            "description": "Optimized Whisper Large V3 — fast, 99+ languages",
                        },
                        {
                            "id": "openai/whisper-large-v3",
                            "description": "Whisper Large V3 — highest accuracy",
                        },
                        {"id": "openai/whisper-1", "description": "Whisper v1 — legacy, cheapest"},
                    ],
                },
                {
                    "id": "groq-whisper",
                    "type": "cloud",
                    "currency": "USD",
                    "timestamps": ["segment", "word"],
                    "models": [
                        {
                            "id": "whisper-large-v3-turbo",
                            "description": "Optimized Whisper Large V3 Turbo — fast, 99+ languages",
                            "default": True,
                        },
                        {
                            "id": "whisper-large-v3",
                            "description": "Whisper Large V3 — highest accuracy",
                        },
                    ],
                },
                {
                    "id": "xai-stt",
                    "type": "cloud",
                    "currency": "USD",
                    "timestamps": ["word"],
                    "models": [
                        {
                            "id": "grok-stt",
                            "description": "Grok STT — word-level timestamps, 12 formats, multichannel, diarization",
                            "default": True,
                        },
                    ],
                },
            ]
        }
    elif args.target == "asr-providers":
        data = {"asr_providers": [spec.listing() for spec in list_asr_provider_specs()]}
    else:
        data = {}
    if args.json_output:
        _json_ok({"status": "success", **data})
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# history
# ═══════════════════════════════════════════════════════════════════════════════


def history_cmd(args: argparse.Namespace) -> None:
    """Dispatch a ``history`` subcommand to its handler.

    ``list``/``show``/``import``/``costs`` are read-only or import-only;
    ``resume``/``sync`` reconstruct one committed native run and run the shared
    native executor, so they are the only branches that can open the history
    database for writing.
    """
    try:
        if args.history_command == "list":
            payload = history_commands.list_history(
                label=args.label,
                operation=args.operation,
                status=args.status,
                limit=args.limit,
                offset=args.offset,
            )
        elif args.history_command == "show":
            payload = history_commands.show_history(args.run)
        elif args.history_command == "import":
            if args.dry_run:
                payload = history_commands.preview_history_import(args.source)
            else:
                payload = history_commands.run_history_import(args.source)
        elif args.history_command == "costs":
            payload = history_commands.costs_history()
        elif args.history_command in ("resume", "sync"):
            payload = _history_native_command(args, args.history_command)
        else:
            fail("Unknown history subcommand.", _EXIT_ARGS)
    except history_commands.HistoryCommandError as exc:
        fail(str(exc), exc.code, details=exc.details)

    if args.json_output:
        _json_ok(payload)
    _print_history(args.history_command, payload)


def _print_history(subcommand: str, payload: dict[str, Any]) -> None:
    if subcommand == "list":
        database = payload["database"]
        runs = payload["runs"]
        if not runs:
            print(f"No history runs found. Database: {database['path']}")
            return
        print(f"History runs ({payload['count']}):")
        for run in runs:
            label = run["user_label"] if run["user_label"] is not None else "-"
            print(
                f"  {run['run_uuid']}  operation={run['operation']}  label={label}  "
                f"status={run['status']}  created={run['created_at']}"
            )
        return

    if subcommand == "show":
        run = payload["run"]
        print(f"Run {run['run_uuid']}")
        print(f"  operation: {run['operation']}")
        print(f"  status: {run['status']}")
        print(f"  label: {run['user_label']}")
        print(f"  root: {run['run_root']}")
        print(f"  created: {run['created_at']}")
        print(f"  updated: {run['updated_at']}")
        if run["legacy_source_root"]:
            print(f"  legacy_source_root: {run['legacy_source_root']}")
        print(
            f"  parts: {len(payload['parts'])}  attempts: {len(payload['attempts'])}  "
            f"artifacts: {len(payload['artifacts'])}  text_sources: {len(payload['text_sources'])}"
        )
        for attempt in payload["attempts"]:
            cost = attempt["cost"]
            currency = cost["currency"] or ""
            print(
                f"  attempt {attempt['attempt_uuid']} {attempt['call_type']} "
                f"status={attempt['status']} cost={cost['amount']} {currency} "
                f"({cost['source']}, exact={cost['exact_available']})"
            )
        return

    if subcommand == "costs":
        print(f"History costs (database: {payload['database']['path']})")
        if not payload["totals"]:
            print("  No costed attempts found.")
        else:
            for total in payload["totals"]:
                currency = total["currency"] if total["currency"] is not None else "unknown"
                amount = total["known_amount"] if total["known_amount"] is not None else "unknown"
                print(
                    f"  {currency}: known={amount} "
                    f"(exact={total['exact_attempts']}, non_exact={total['non_exact_attempts']}) "
                    f"unknown_attempts={total['unknown_attempts']}"
                )
        for row in payload["operations"]:
            operation = row["operation"] if row["operation"] is not None else "unknown"
            currency = row["currency"] if row["currency"] is not None else "unknown"
            amount = row["known_amount"] if row["known_amount"] is not None else "unknown"
            print(f"  {operation}/{currency}: known={amount} unknown={row['unknown_attempts']}")
        print(
            f"  completeness: {payload['completeness']}  attempts: {payload['attempts']}  "
            f"local_attempts_without_api_charge: {payload['local_attempts_without_api_charge']}"
        )
        return

    if subcommand in ("resume", "sync"):
        verb = "Resumed" if subcommand == "resume" else "Synced"
        if payload.get("operation") == OPERATION_TIMINGS:
            print(f"{verb} timings run {payload['run_uuid']} (revision {payload['revision']})")
            print(f"  status: {payload['status']}")
            files = payload.get("files") or {}
            if files.get("timings_json"):
                print(f"  Timings JSON: {files['timings_json']}")
            if files.get("srt"):
                print(f"  SRT: {files['srt']}")
            return
        files = payload["files"]
        print(f"{verb} run {payload['run_uuid']} (revision {payload['revision']})")
        print(f"  Full MP3: {files['full_mp3']}")
        print(f"  Run manifest: {files['run_json']}")
        print(f"  Manifest: {files['manifest_json']}")
        return

    if payload["dry_run"]:
        print(f"Dry run for {payload['source']} (database: {payload['database']['path']})")
        print(
            f"  discovered={payload['discovered_count']} "
            f"importable={payload['importable_count']} "
            f"already_imported={payload['already_imported_count']} "
            f"missing_text={payload['missing_text_count']} "
            f"missing_audio={payload['missing_audio_count']}"
        )
        if payload["scan_conflicts"]:
            print(f"  scan_conflicts: {', '.join(payload['scan_conflicts'])}")
        return
    print(f"Imported from {payload['source']} (database: {payload['database']['path']})")
    print(
        f"  imported={payload['imported_count']} skipped={payload['skipped_count']} "
        f"rejected={payload['rejected_count']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _native_timing_options(
    args: argparse.Namespace,
) -> native_generation.NativeTimingOptions | None:
    """Return the recorded timing settings for a native run, or ``None``.

    The local ``faster-whisper`` provider and the paid cloud
    ``groq-whisper``/``xai-stt`` providers reach a native run (``_native_route_eligible``
    already sent ``openrouter-whisper`` to the legacy executor); the effective
    settings are recorded in the run snapshot so a resume proves they did not
    change. A run without ``--with-timings`` records nothing and keeps its output
    options byte-for-byte, so a completed run written before this route stays
    resumable.
    """
    if not getattr(args, "with_timings", False):
        return None
    return native_generation.NativeTimingOptions(
        provider=args.timing_provider,
        model=args.timing_model,
        device=args.timing_device,
        compute=args.timing_compute,
        language=args.timing_language,
        word_timestamps=bool(args.word_timestamps),
    )


def _native_quality_options(
    args: argparse.Namespace,
) -> native_generation.NativeQualityOptions | None:
    """Return the recorded quality settings for a native run, or ``None``.

    An installed local ASR provider (``qwen-local`` or ``nemotron-local``) reaches
    a native run on any admitted route; the paid cloud ``xai-stt`` provider reaches
    only the two admitted dialogue routes and is per-turn. ``_native_route_eligible``
    already sent every other quality provider to the legacy executor. A run without
    ``--tts-quality-provider`` records nothing and keeps its output options
    byte-for-byte, so a completed run written before this route stays resumable and
    a non-dialogue legacy run keeps ignoring the flag exactly as before.
    """
    provider_id = getattr(args, "tts_quality_provider", None)
    if not provider_id:
        return None
    return native_generation.NativeQualityOptions(
        provider=provider_id,
        model=getattr(args, "tts_quality_model", None),
        device=args.tts_quality_device,
        compute=args.tts_quality_compute,
        runtime=args.tts_quality_runtime,
        language=getattr(args, "tts_quality_language", None),
    )


def _native_execution_hooks(args: argparse.Namespace) -> native_generation.NativeExecutionHooks:
    """The local media seams both native routes present to the executor.

    ``generate`` and ``history resume``/``history sync`` share the exact same
    presentation and hashing seams, so a reconstructed run behaves identically to
    the route that created it.
    """
    return native_generation.NativeExecutionHooks(
        write_audio_as_mp3=write_audio_as_mp3,
        trim_final_silence=trim_final_silence,
        mp3_duration_ms=mp3_duration_ms,
        concat_audio_files=concat_audio_files,
        concat_dialogue_turns=concat_dialogue_turns,
        sha256_file=_sha256_file,
        progress=print if not args.json_output else (lambda _message: None),
    )


def _omnivoice_dialogue_cast_map(prepared: PreparedRun) -> dict[str, str]:
    """Return the speaker -> profile-id cast map from a reconstructed prepared run."""
    cast: dict[str, str] = {}
    for part in prepared.parts:
        chunk = part.chunk
        if chunk.speaker and chunk.voice:
            cast.setdefault(chunk.speaker, chunk.voice)
    return cast


def _build_native_omnivoice_provider(prepared: PreparedRun, api_key: str) -> Any:
    """Rebuild the OmniVoice provider(s) for a reconstructed local OmniVoice run.

    The committed identity selects the admitted route and supplies its exact
    settings. A non-preset ``auto``/``clone``/``design`` run is rebuilt from its
    committed mode, reference path/text, or design instruction, and its mode-marker
    voice; the executor has already proved a clone reference still matches its stored
    digest before any local model. A preset bank run re-loads the committed catalog
    locator and every referenced profile's current ``reference_sha256`` must match
    the digest this run stored before any local model is constructed, so a changed or
    missing reference fails closed instead of silently synthesizing a different
    voice; a dialogue run (every part carries a cast profile) is cloned per cast
    profile by the same ``bind_dialogue_voice_bank_providers`` seam the fresh route
    uses, and a monologue run's one referenced profile is bound as the run's clone
    reference.
    """
    mode_identity = prepared.omnivoice_mode_identity
    if mode_identity is not None:
        # No ``voice`` here: these modes reject any voice control, and the provider
        # takes its voice from the committed mode, so the rebuilt arguments carry
        # only the inputs the mode actually speaks with.
        mode_args = argparse.Namespace(
            provider="omnivoice-local",
            model=prepared.model,
            mode=mode_identity.mode,
            format="markdown",
            reference_audio=mode_identity.reference_audio,
            reference_text=mode_identity.reference_text,
            design_instruction=mode_identity.design_instruction,
        )
        return build_provider(mode_args, api_key, prepared.style_prompt, prepared.prompt_mode)
    identity = prepared.voice_bank_identity
    if identity is None:
        fail(
            "this local OmniVoice run records no voice-bank identity; refusing to resume it.",
            _EXIT_PROVIDER,
        )
    try:
        catalog = load_voice_bank(Path(identity.catalog_path))
    except VoiceBankError as exc:
        fail(str(exc), _EXIT_PROVIDER)
    stored = {profile.profile_id: profile for profile in identity.profiles}
    cast = _omnivoice_dialogue_cast_map(prepared)
    referenced = (
        list(cast.values()) if cast else [profile.profile_id for profile in identity.profiles]
    )
    catalog_profiles = {profile.id: profile for profile in catalog.profiles}
    for voice_id in referenced:
        profile = catalog_profiles.get(voice_id)
        bound = stored.get(voice_id)
        if profile is None or bound is None or bound.reference_sha256 != profile.reference_sha256:
            fail(
                "the voice-bank reference for this run changed or is missing; refusing to "
                "resume the local OmniVoice run.",
                _EXIT_PROVIDER,
                details={"error_code": native_generation._ERROR_LOCAL_REFERENCE_UNAVAILABLE},
            )
    if cast:
        identity_args = argparse.Namespace(
            provider="omnivoice-local",
            model=prepared.model,
            voice=prepared.voice,
            mode="preset",
            format="dialogue",
        )
        base = build_provider(identity_args, api_key, prepared.style_prompt, prepared.prompt_mode)
        return _bind_dialogue_voice_bank_providers(base, catalog, cast)
    # Monologue: the run's one referenced profile is its clone reference, bound
    # through the committed catalog locator exactly as the fresh route does.
    identity_args = argparse.Namespace(
        provider="omnivoice-local",
        model=prepared.model,
        voice=prepared.voice,
        mode="preset",
        format="markdown",
        voice_bank=identity.catalog_path,
        voice_bank_catalog=catalog,
    )
    try:
        return build_provider(identity_args, api_key, prepared.style_prompt, prepared.prompt_mode)
    except VoiceBankError as exc:
        fail(
            str(exc),
            _EXIT_PROVIDER,
            details={"error_code": native_generation._ERROR_LOCAL_REFERENCE_UNAVAILABLE},
        )


def _build_native_qwen_local_provider(prepared: PreparedRun) -> Any:
    """Rebuild the local Qwen provider for a reconstructed run.

    The committed identity selects the admitted route and supplies its exact
    settings. A clone run re-reads the reference locator and text it committed to;
    a preset/design run rebuilds the mode, model, voice, and instruction it spoke
    with. The executor already proved the identity still matches and the runtime is
    available offline before any local model runs, so this only reconstructs the
    already-admitted route. A run without a committed identity cannot be resumed and
    fails closed.
    """
    clone_identity = prepared.qwen_clone_identity
    if clone_identity is not None:
        identity_args = argparse.Namespace(
            provider="qwen-local",
            model=prepared.model,
            voice=prepared.voice,
            mode=clone_identity.mode,
            sample=clone_identity.sample_path,
            sample_text=clone_identity.sample_text,
            qwen_instruct=prepared.style_prompt,
        )
    else:
        mode_identity = prepared.qwen_mode_identity
        if mode_identity is None:
            fail(
                "this local Qwen run records no mode identity; refusing to resume it.",
                _EXIT_PROVIDER,
            )
        identity_args = argparse.Namespace(
            provider="qwen-local",
            model=prepared.model,
            voice=prepared.voice,
            mode=mode_identity.mode,
            sample=None,
            sample_text=None,
            qwen_instruct=mode_identity.instruct,
        )
    api_key = read_api_key(identity_args)
    return build_provider(identity_args, api_key, prepared.style_prompt, prepared.prompt_mode)


def _native_history_provider_builder(prepared: PreparedRun) -> Any:
    """Build the provider a reconstructed native run needs, lazily from its snapshot.

    ``history resume`` and ``history sync`` recover the provider identity from the
    committed snapshot, so an admitted ``polza-tts``, ``polza-chat-audio``,
    ``openrouter-tts``, ``omnivoice-local``, or ``qwen-local`` provider is built from
    those stored values only. The ``polza-chat-audio`` route rebuilds its resolved
    compatibility ``fallback_voice`` from the snapshot. The API key is read when the
    executor first calls the factory -- for a
    fresh unattempted submit or a known-id GET recovery -- and never for an export
    repair or a local raw rebuild.
    """
    if prepared.provider == "qwen-local":
        return _build_native_qwen_local_provider(prepared)
    identity_args = argparse.Namespace(
        provider=prepared.provider,
        model=prepared.model,
        voice=prepared.voice,
        fallback_voice=prepared.fallback_voice,
    )
    api_key = read_api_key(identity_args)
    if prepared.provider == "omnivoice-local":
        return _build_native_omnivoice_provider(prepared, api_key)
    return build_provider(identity_args, api_key, prepared.style_prompt, prepared.prompt_mode)


def _paid_timing_files(state: paid_transcription_history.PaidTranscriptionState) -> dict[str, str]:
    """Return the timings/SRT paths one paid timing run recorded, or an empty map."""
    options = state.snapshot.get("request_options")
    opts = options if isinstance(options, dict) else {}
    output_dir_value = opts.get("output_dir")
    prefix = opts.get("output_prefix")
    if isinstance(output_dir_value, str) and isinstance(prefix, str):
        output_dir = Path(output_dir_value)
        return {
            "timings_json": str(output_dir / f"{prefix}.timings.json"),
            "srt": str(output_dir / f"{prefix}.srt"),
        }
    return {}


def _paid_timing_payload(
    state: paid_transcription_history.PaidTranscriptionState, mode: str
) -> dict[str, Any]:
    completed = state.status == RUN_STATUS_COMPLETED
    return {
        "dry_run": False,
        "mode": mode,
        "run_uuid": state.run_uuid,
        "revision": state.revision,
        "operation": OPERATION_TIMINGS,
        "status": state.status,
        "complete": completed,
        "files": _paid_timing_files(state),
    }


def _paid_transcription_history_command(
    args: argparse.Namespace, mode: str
) -> dict[str, Any] | None:
    """Handle ``history resume``/``sync`` for one paid cloud-timing run.

    Returns ``None`` only when the UUID is not a paid transcription timing run, so
    the ordinary native-TTS reader handles it unchanged; a corrupt, unreadable, or
    otherwise malformed paid run fails closed with its own fixed error instead of
    being reclassified as “not this handler.” ``history sync`` never touches a
    provider, a model, or a writer. ``history resume`` holds the same canonical
    output-root lock, reloads the state after acquiring it, reconciles a crash
    window the receipt validates, and otherwise replays the exact attempt's stored
    response locally (no second POST); while the attempt stays an unconfirmed
    ``submitting`` marker it fails closed with ``PAID_SUBMIT_UNCONFIRMED``.
    """
    try:
        state = paid_transcription_history.load_paid_transcription_state(args.run)
    except paid_transcription_history.PaidTranscriptionError as exc:
        if exc.error_code in (
            "PAID_TRANSCRIPTION_NOT_FOUND",
            "PAID_TRANSCRIPTION_NOT_PAID",
        ):
            # Unknown or non-paid UUIDs dispatch onward to the native-TTS reader.
            return None
        fail(str(exc), _EXIT_PROVIDER, details={"error_code": exc.error_code})
    if state.operation != OPERATION_TIMINGS:
        return None
    if mode == "sync" or state.status == RUN_STATUS_COMPLETED:
        return _paid_timing_payload(state, mode)
    try:
        # A renamed ancestor can be replaced with a symlink while the saved raw
        # body remains readable. Check the DB-bound canonical root before even
        # allocating a lock for a retargeted path, then re-check after locking and
        # reloading the run so neither receipt recovery nor publication follows it.
        output_root = paid_transcription_history.require_paid_output_root(state)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    try:
        with acquire_run_lock(output_root):
            fresh = paid_transcription_history.load_paid_transcription_state(state.run_uuid)
            try:
                locked_root = paid_transcription_history.require_paid_output_root(fresh)
            except paid_transcription_history.PaidTranscriptionError as exc:
                fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
            return _resume_paid_timing(fresh, locked_root)
    except HistoryRunLockedError:
        fail(
            "Another process is already writing this paid timing run directory; refusing to "
            "resume two writers for one output root.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TIMING_RUN_LOCKED"},
        )


def _resume_paid_timing(
    state: paid_transcription_history.PaidTranscriptionState, output_root: Path
) -> dict[str, Any]:
    """Replay or reconcile one paid timing run under the already-held run lock."""
    if state.status == RUN_STATUS_COMPLETED:
        return _paid_timing_payload(state, "resume")
    files = _paid_timing_files(state)
    if state.attempt_status == "submitting":
        try:
            recovered = paid_transcription_history.recover_paid_transcription_after_crash(
                state.run_uuid
            )
        except paid_transcription_history.PaidTranscriptionError:
            recovered = None
        if recovered is None:
            fail(
                "Refusing to resume: this paid timing request has an unconfirmed submit; a "
                "new attempt is never made automatically.",
                _EXIT_PROVIDER,
                details={"error_code": _PAID_SUBMIT_UNCONFIRMED_ERROR_CODE},
            )
        state = paid_transcription_history.load_paid_transcription_state(state.run_uuid)
    if not state.raw_saved:
        fail(
            "Refusing to resume: this paid timing run has no saved response to replay.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TRANSCRIPTION_UNSUPPORTED"},
        )
    _replay_paid_timing(state, files, output_root)
    payload = _paid_timing_payload(state, "resume")
    payload["complete"] = True
    payload["status"] = RUN_STATUS_COMPLETED
    return payload


def _replay_paid_timing(
    state: paid_transcription_history.PaidTranscriptionState,
    files: dict[str, str],
    output_root: Path,
) -> None:
    """Re-parse the saved raw response and republish the timing artifacts."""
    if not files:
        fail(
            "Refusing to replay: this paid timing run does not record its output location.",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TRANSCRIPTION_UNSUPPORTED"},
        )
    try:
        audio_path = paid_transcription_history.verify_source_identity(state.snapshot)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    try:
        body = paid_transcription_history.read_paid_transcription_raw(state.run_uuid)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    try:
        raw = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        fail(
            f"The saved paid timing response is not valid JSON: {exc}",
            _EXIT_PROVIDER,
            details={"error_code": "PAID_TRANSCRIPTION_UNSUPPORTED"},
        )
    snapshot = state.snapshot
    language = snapshot.get("language") or ""
    word_timestamps = bool(snapshot.get("word_timestamps_requested"))
    if state.provider == "groq-whisper":
        from .providers.groq_whisper import parse_groq_timing_result

        timing = parse_groq_timing_result(
            raw,
            model=state.model,
            language=str(language),
            word_timestamps=word_timestamps,
            audio_path=audio_path,
        )
    else:
        from .providers.xai_stt import parse_xai_timing_result

        timing = parse_xai_timing_result(
            raw,
            model=state.model,
            language=str(language),
            word_timestamps=word_timestamps,
            audio_path=audio_path,
        )
    output_dir = output_root
    prefix = Path(files["timings_json"]).name[: -len(".timings.json")]
    try:
        # A resume only publishes into an existing committed root; never create
        # a replacement directory if that root was moved after admission.
        paid_transcription_history.require_paid_output_root(state)
        source_identity = paid_transcription_history.require_readable_source(audio_path)
        _write_paid_timing_artifacts(audio_path, output_dir, prefix, timing, source_identity)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    reservation = paid_transcription_history.reservation_from_state(state)
    _complete_paid_timing(reservation, files)


def _history_native_command(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    """Resume or sync one committed native TTS run from its stored snapshot.

    The verified view is loaded first with a read-only seam, so an unknown or
    legacy UUID, an absent database, and a non-native snapshot all fail before any
    write. The snapshot alone supplies the prepared parts, run paths, script
    format, and script path; the original script file is never re-read. The
    provider is built only when the executor actually needs one, so a completed-run
    export repair and a local raw rebuild read no API key. ``history resume`` may
    submit a truly unattempted part and is therefore potentially paid; it may also
    run the pending local faster-whisper timing or the pending local quality
    verification for the completed audio. ``history sync`` never starts a new paid
    submit and never runs a local model, so a missing timing or verification stays
    incomplete (``quality.complete`` is ``false``) for an explicit resume.
    """
    paid_payload = _paid_transcription_history_command(args, mode)
    if paid_payload is not None:
        return paid_payload
    view = history_commands.load_native_history_view(args.run)
    try:
        ffmpeg_path, ffprobe_path = check_media_tools()
    except RuntimeError as exc:
        fail(str(exc), _EXIT_NO_FFMPEG)
    try:
        summary = native_generation.run_native_history_generation(
            view=view,
            mode=mode,
            provider_builder=_native_history_provider_builder,
            hooks=_native_execution_hooks(args),
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )
    except native_generation.NativeGenerationError as exc:
        fail(str(exc), exc.code, details={"error_code": exc.error_code})
    payload: dict[str, Any] = {
        "dry_run": False,
        "mode": mode,
        "run_uuid": summary.run_uuid,
        "revision": summary.revision,
        "files": summary.files,
        "duration_ms": summary.duration_ms,
        "cost": {"total": summary.cost_total, "currency": summary.cost_currency},
    }
    if summary.timing_requested:
        payload["timing"] = {"complete": summary.timing_complete}
    if summary.quality_requested:
        payload["quality"] = {
            "complete": summary.quality_complete,
            "passed": summary.quality_passed if summary.quality_complete else None,
        }
    return payload


def _preflight_timing_dependency(timing_provider: str = "faster-whisper") -> None:
    if timing_provider == "openrouter-whisper":
        try:
            read_openrouter_key()
        except RuntimeError as exc:
            fail(str(exc), _EXIT_NO_KEY)
    elif timing_provider == "groq-whisper":
        try:
            read_groq_key()
        except RuntimeError as exc:
            fail(str(exc), _EXIT_NO_KEY)
    elif timing_provider == "xai-stt":
        try:
            read_xai_key()
        except RuntimeError as exc:
            fail(str(exc), _EXIT_NO_KEY)
    else:
        try:
            import faster_whisper  # noqa: F401
        except ModuleNotFoundError as exc:
            fail(
                f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --extra timing-whisper",
                _EXIT_MISSING_DEP,
            )


def _has_paid_chunk_audio(paths) -> bool:
    return paths.chunks_dir.exists() and any(paths.chunks_dir.glob("chunk_*.mp3"))


_PAID_SUBMIT_UNCONFIRMED_ERROR_CODE = "PAID_SUBMIT_UNCONFIRMED"

# Machine reason reported by ``status --json`` for a run whose resume is blocked.
_PAID_SUBMIT_RESUME_BLOCK_REASON = "paid_submit_unconfirmed"

# Machine reason reported by ``status --json`` when an existing run_state.json
# cannot prove the run is resumable: invalid JSON, or valid JSON that is not an
# object. Distinct from an absent state, which stays an ordinary empty run.
_RUN_STATE_UNREADABLE_RESUME_BLOCK_REASON = "run_state_unreadable"


# Bounded marker reported when an existing run cannot prove no paid submit is
# pending: an unreadable state file, or valid JSON that is not an object.
_UNKNOWN_PAID_ATTEMPT: dict[str, Any] = {
    "id": None,
    "number": None,
    "status": "unknown",
}


def _paid_submit_unconfirmed_details(marker: dict[str, Any]) -> dict[str, object]:
    """Bounded, JSON-friendly details for a blocked resume or overwrite."""
    return {
        "error_code": _PAID_SUBMIT_UNCONFIRMED_ERROR_CODE,
        "chunk_id": marker.get("id"),
        "chunk_number": marker.get("number"),
        "attempt_status": marker.get("status"),
    }


def _reject_unconfirmed_paid_resume(marker: dict[str, Any], logger: GenerationLogger) -> NoReturn:
    """Log and fail a resume blocked by an unconfirmed paid submit.

    Shared by the early ``generate`` preflight (which runs before any provider or
    pricing work) and the ``_generate_step`` guard kept for direct callers, so
    both report the same bounded ``PAID_SUBMIT_UNCONFIRMED`` details and exit code.
    """
    logger.event(
        "error",
        "resume_rejected",
        reason="unconfirmed_paid_submit",
        chunk=marker.get("number"),
        id=marker.get("id"),
        status=marker.get("status"),
    )
    fail(
        f"Cannot resume: chunk {marker.get('id')} has an unconfirmed paid submit "
        f"({marker.get('status')}). No provider request was sent; repeating that "
        "submit needs an explicit new run instead of --resume.",
        _EXIT_PROVIDER,
        details=_paid_submit_unconfirmed_details(marker),
    )


def _load_status_state(state_path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Load run state for ``status`` without crashing on unusable content.

    ``run_state.json`` is user-editable. Invalid JSON, or valid JSON that is not
    an object, carries no trusted run evidence, so it is reported as unreadable
    instead of raising or being mistaken for an empty, resumable run. An absent
    file is not unreadable: it stays an ordinary no-state run.
    """
    if not state_path.exists():
        return None, False
    try:
        state = load_state(state_path)
    except (OSError, ValueError):
        return None, True
    if not isinstance(state, dict):
        return None, True
    return state, False


def _unconfirmed_paid_attempt_in_existing_run(paths) -> dict[str, Any] | None:
    """Return a pending paid-attempt marker found in an existing run folder.

    ``--overwrite`` deletes the run folder, so the marker is checked before any
    deletion: a run whose paid submit outcome is unconfirmed is evidence to
    keep, not replaceable scratch space. An unreadable state file also fails
    closed because it cannot prove that no paid submit is pending. Valid JSON
    whose top level is not an object (``null``, a list, or a scalar) is treated
    the same way: it carries no trusted marker and cannot prove the run is safe
    to delete.
    """
    state_path = paths.output_root / STATE_FILE
    if not state_path.exists():
        return None
    try:
        state = load_state(state_path)
    except (OSError, ValueError):
        return dict(_UNKNOWN_PAID_ATTEMPT)
    if not isinstance(state, dict):
        return dict(_UNKNOWN_PAID_ATTEMPT)
    return unconfirmed_attempt(state)


def _polza_media_route_model(model: object) -> bool:
    """Compatibility wrapper for ``services.recovery.polza_media_route_model``."""
    return recovery.polza_media_route_model(model)


def _known_media_recovery(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compatibility wrapper for ``services.recovery.known_media_recovery``."""
    return recovery.known_media_recovery(state)


def _preceding_chunks_ready(
    prior_ids: dict[int, str], chunks_dir: Path, completed: set[int], number: int
) -> bool:
    """Compatibility wrapper for ``services.recovery.preceding_chunks_ready``."""
    return recovery.preceding_chunks_ready(prior_ids, chunks_dir, completed, number)


def _state_completed_chunk_ids(state: dict[str, Any] | None) -> dict[int, str]:
    """Compatibility wrapper for ``services.recovery.state_completed_chunk_ids``."""
    return recovery.state_completed_chunk_ids(state)


def _known_raw_recovery(
    state: dict[str, Any] | None, run_root: Path, chunks_dir: Path
) -> dict[str, Any] | None:
    """Compatibility wrapper for ``services.recovery.known_raw_recovery``."""
    return recovery.known_raw_recovery(state, run_root, chunks_dir)


def _recoverable_raw_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
    run_root: Path,
) -> dict[int, dict[str, Any]]:
    """Compatibility wrapper for ``services.recovery.recoverable_raw_attempts``."""
    return recovery.recoverable_raw_attempts(
        state,
        provider=provider,
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_root,
    )


def _recoverable_paid_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
    run_root: Path,
) -> dict[int, dict[str, Any]]:
    """Compatibility wrapper for ``services.recovery.recoverable_paid_attempts``."""
    return recovery.recoverable_paid_attempts(
        state,
        provider=provider,
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
        run_root=run_root,
    )


def _recoverable_media_attempts(
    state: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    voice: str | None,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Compatibility wrapper for ``services.recovery.recoverable_media_attempts``."""
    return recovery.recoverable_media_attempts(
        state,
        provider=provider,
        model=model,
        voice=voice,
        chunks=chunks,
        chunks_dir=chunks_dir,
    )


def _paid_submit_attempt_status(error: BaseException) -> str:
    """Label a failed paid attempt from its failure class, not its error text.

    ``outcome_unknown`` covers exactly the failures that used to trigger a retry
    — timeouts, connection breaks, transient HTTP responses, still-pending media
    tasks — because they leave the paid outcome unconfirmed. Everything else is a
    definite failure.
    """
    return ATTEMPT_OUTCOME_UNKNOWN if is_retryable_error(error) else ATTEMPT_FAILED


def _emit_json_event(args, event: str, **fields) -> None:
    if not getattr(args, "json_events", False):
        return
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def _log_retry(
    logger: GenerationLogger,
    args,
    chunk: ScriptChunk,
    attempt: int,
    error: BaseException,
    delay: float,
) -> None:
    logger.event(
        "warn",
        "chunk_retry",
        chunk=chunk.number,
        id=chunk.id,
        attempt=attempt,
        delay_sec=round(delay, 2),
        error=str(error),
    )
    _emit_json_event(
        args, "chunk_retry", chunk=chunk.number, id=chunk.id, attempt=attempt, error=str(error)
    )


def _recover_existing_chunks(
    state: dict,
    chunks: list[ScriptChunk],
    chunks_dir: Path,
    ffprobe_path: str,
    model: str,
    voice: str,
) -> None:
    total_duration_ms = 0
    for chunk in chunks:
        path = chunks_dir / f"{chunk.id}.mp3"
        if not path.exists():
            break
        duration_ms = mp3_duration_ms(ffprobe_path, path)
        start_ms = total_duration_ms
        end_ms = start_ms + duration_ms
        total_duration_ms = end_ms
        artifact = ChunkArtifact(
            number=chunk.number,
            id=chunk.id,
            file=path.name,
            duration_ms=duration_ms,
            duration_sec=round(duration_ms / 1000, 3),
            start_ms=start_ms,
            end_ms=end_ms,
            text_characters=len(chunk.text),
            transcript=None,
            client_path="recovered-from-existing-file",
            generation_id=None,
        )
        upsert_completed_chunk(state, artifact=artifact, model=model, voice=voice, text=chunk.text)


def _continuous_chunk_files(chunks_dir: Path) -> list[Path]:
    files = []
    number = 1
    while True:
        path = chunks_dir / f"chunk_{number:02d}.mp3"
        if not path.exists():
            break
        files.append(path)
        number += 1
    return files


def _find_full_audio(run_dir: Path) -> Path | None:
    for path in sorted(run_dir.glob("*-voiceover-*.mp3")):
        if path.is_file():
            return path
    return None


def _resolve_audio(raw: str) -> Path:
    if "*" in raw or "?" in raw:
        matches = sorted(glob_mod.glob(raw))
        if not matches:
            fail(f"No files match: {raw}", _EXIT_ARGS)
        if len(matches) > 1:
            fail(f"Multiple files match: {raw}. Provide exact path.", _EXIT_ARGS)
        return Path(matches[0])
    return Path(raw)


def _write_timing_artifacts(audio_path, output_dir, prefix, timing):
    ffprobe_path = shutil.which("ffprobe")
    duration_ms = (
        mp3_duration_ms(ffprobe_path, audio_path)
        if ffprobe_path
        else sum(seg.duration_ms for seg in timing.segments)
    )
    timing_json = output_dir / f"{prefix}.timings.json"
    write_json(timing_json, build_timing_manifest(timing, duration_ms))
    srt_path = output_dir / f"{prefix}.srt"
    srt_path.write_text(build_srt(timing), encoding="utf-8")
    return {"segment_count": len(timing.segments), "total_duration_ms": duration_ms}


def _extract_asr_timings(
    audio_path, output_dir, prefix, provider_id, model, device, compute, language
):
    spec = get_asr_provider_spec(provider_id)
    args = argparse.Namespace(
        device=device,
        compute=compute,
        model=model,
        language=language,
        word_timestamps=True,
    )
    _validate_asr_request_options(args, spec)
    health = spec.dependency_probe()
    if not health.available:
        raise ModuleNotFoundError(health.remediation)
    timing = transcription.transcribe_generic_asr_timing(
        spec,
        audio_path=audio_path,
        model=model,
        device=device,
        compute=compute,
        language=language,
        mp3_duration_ms=mp3_duration_ms,
    )
    return _write_timing_artifacts(audio_path, output_dir, prefix, timing)


def _write_paid_timing_artifacts(audio_path, output_dir, prefix, timing, source_identity):
    """Publish the paid timing artifacts atomically after re-checking the source.

    The paid route re-verifies the source audio immediately after the provider
    response and before probing or publishing, so a mid-request replacement never
    yields a transcript with misleading provenance. Each artifact is written via
    an fsynced temporary file and an atomic replacement inside the validated
    output root, so an attacker-planted symlink leaf is never followed.
    """
    try:
        verified_source = paid_transcription_history.require_readable_source(audio_path)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    if verified_source != tuple(source_identity):
        fail(
            "The source audio changed during the paid timing request; refusing to publish "
            "misleading timing provenance.",
            _EXIT_OUTPUT,
            details={"error_code": "PAID_SOURCE_CHANGED"},
        )
    ffprobe_path = shutil.which("ffprobe")
    duration_ms = (
        mp3_duration_ms(ffprobe_path, audio_path)
        if ffprobe_path
        else sum(seg.duration_ms for seg in timing.segments)
    )
    manifest = build_timing_manifest(timing, duration_ms)
    srt_text = build_srt(timing)
    paid_transcription_history.atomic_write_artifact(
        output_dir,
        f"{prefix}.timings.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    paid_transcription_history.atomic_write_artifact(
        output_dir, f"{prefix}.srt", srt_text.encode("utf-8")
    )
    return {"segment_count": len(timing.segments), "total_duration_ms": duration_ms}


def _extract_timings(
    audio_path,
    output_dir,
    prefix,
    timing_provider,
    model,
    device,
    compute_type,
    language,
    word_timestamps=False,
    quiet=False,
    on_raw_response=None,
    artifact_writer=None,
):
    if timing_provider == "openrouter-whisper":
        fail(
            "openrouter-whisper does NOT return segment or word-level timestamps. "
            "The API only returns full text — one segment covering the entire audio. "
            "This would produce useless timings (one entry for the whole file) and waste your money.\n\n"
            "Use a provider that supports real timestamps:\n"
            "  • faster-whisper (local) — free, segments + words\n"
            "  • groq-whisper (cloud) — GROQ_API_KEY, segments + words\n"
            "  • xai-stt (cloud) — X_AI_API_KEY, words + confidence\n\n"
            "For plain transcription (text only, no timings), use OpenRouter directly.",
            _EXIT_PROVIDER,
        )

    timing = transcription.transcribe_timing_audio(
        audio_path=audio_path,
        timing_provider=timing_provider,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        word_timestamps=word_timestamps,
        quiet=quiet,
        on_raw_response=on_raw_response,
    )

    writer = artifact_writer or _write_timing_artifacts
    try:
        result = writer(audio_path, output_dir, prefix, timing)
    except paid_transcription_history.PaidTranscriptionError as exc:
        fail(str(exc), _EXIT_OUTPUT, details={"error_code": exc.error_code})
    except Exception as e:
        fail(f"Failed to write timing artifacts: {e}", _EXIT_OUTPUT)

    if not quiet:
        print(f"Timings JSON: {output_dir / f'{prefix}.timings.json'}")
        print(f"SRT: {output_dir / f'{prefix}.srt'}")

    return result


def _list_artifact_files(paths) -> dict:
    """Compatibility wrapper for ``services.finalization.list_artifact_files``."""
    return finalization.list_artifact_files(paths)


def _json_ok(data: dict) -> NoReturn:
    data.setdefault("status", "success")
    print(json.dumps(data, ensure_ascii=False))
    sys.exit(_EXIT_OK)


def _json_error(message: str, code: int, *, details: dict[str, object] | None = None) -> NoReturn:
    payload: dict[str, object] = {"status": "error", "error": message, "code": code}
    if details is not None:
        payload["details"] = details
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(code)


def _emit_error(
    args,
    message: str,
    code: int,
    *,
    details: dict[str, object] | None = None,
) -> NoReturn:
    if getattr(args, "json_output", True):
        _json_error(message, code, details=details)
    else:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(code)


_VALID_MODELS_BY_PROVIDER = {
    "polza-chat-audio": [
        "openai/gpt-audio-mini",
        "openai/gpt-audio",
    ],
    "polza-tts": POLZA_TTS_MODELS,
    "openrouter-tts": OPENROUTER_TTS_MODELS,
    "omnivoice-local": [OMNIVOICE_LOCAL_MODEL_ID],
}


def _resolve_model(args: argparse.Namespace) -> None:
    if not hasattr(args, "model") or args.model is None:
        args.model = PROVIDER_DEFAULT_MODELS.get(args.provider, DEFAULT_MODEL)


def _resolve_qwen_mode_identity(args: argparse.Namespace) -> None:
    """Resolve the admitted ``qwen-local`` mode's model and effective voice.

    ``auto`` is rejected here as a usage error before any provider, model, or
    snapshot exists: the CLI offers the choice but the runtime implements no
    automatic mode selection, so silently substituting ``preset`` would speak with
    a mode this run was never asked for.
    """
    if args.provider != "qwen-local":
        return
    mode = getattr(args, "mode", "preset")
    if mode == "preset":
        args.model = QWEN_MODEL_CUSTOMVOICE
    elif mode == "clone":
        args.model = QWEN_MODEL_BASE
        args.voice = "clone"
    elif mode == "design":
        args.model = QWEN_MODEL_VOICE_DESIGN
        args.voice = "design"
    else:
        fail(
            "qwen-local --mode auto is not implemented: choose --mode preset, clone, or "
            "design. No mode was substituted and no provider was selected.",
            _EXIT_ARGS,
        )


def _validate_model_for_provider(provider: str, model: str) -> None:
    valid = _VALID_MODELS_BY_PROVIDER.get(provider, [])
    if not valid:
        return
    if model not in valid:
        fail(
            f"Model '{model}' is not valid for provider '{provider}'. Valid models: {valid}",
            _EXIT_ARGS,
        )


def _omnivoice_voice_identity(args: argparse.Namespace) -> str | None:
    if getattr(args, "provider", None) != "omnivoice-local":
        return None
    mode = getattr(args, "mode", "preset")
    if mode == "preset":
        profile = getattr(args, "voice_bank_profile", None)
        catalog = getattr(args, "voice_bank_catalog", None)
        if profile is None or catalog is None:
            return None
        return f"preset:{profile.id}:{profile.reference_sha256}"
    if mode == "clone":
        reference_audio_path = getattr(args, "reference_audio", None)
        reference_text = getattr(args, "reference_text", None)
        if reference_audio_path is None or reference_text is None:
            return None
        return f"clone:{_sha256_file(Path(reference_audio_path))}:{_sha256_text(reference_text)}"
    if mode == "design":
        design_instruction = getattr(args, "design_instruction", None)
        if design_instruction is None:
            return None
        return f"design:{_sha256_text(design_instruction)}"
    return "auto"


def _bind_omnivoice_dialogue_fingerprints(
    chunks: list[ScriptChunk], catalog: VoiceBankCatalog
) -> list[ScriptChunk]:
    """Compatibility wrapper for ``services.prepare`` fingerprint binding."""
    try:
        return bind_omnivoice_dialogue_fingerprints(chunks, catalog)
    except PreparationError as exc:
        fail(str(exc), _EXIT_ARGS)


def _dialogue_synthesis_identity(
    args: argparse.Namespace,
    style_prompt: str | None,
    prompt_mode: str,
    chunks: list[ScriptChunk] | None = None,
) -> str | None:
    if not is_dialogue_format(getattr(args, "format", None)):
        return None
    speaker_voice_map = getattr(args, "speaker_voice_map", None) or {}
    provider = getattr(args, "provider", None)
    catalog = getattr(args, "voice_bank_catalog", None)
    profiles = (
        {profile.id: profile for profile in catalog.profiles}
        if provider == "omnivoice-local" and isinstance(catalog, VoiceBankCatalog)
        else {}
    )
    if provider == "omnivoice-local" and not profiles:
        fail("Cannot build dialogue identity without an admitted OmniVoice voice bank.", _EXIT_ARGS)
    cast: dict[str, dict[str, str | None]] = {}
    for alias, voice in sorted(speaker_voice_map.items()):
        profile = profiles.get(voice)
        if provider == "omnivoice-local" and profile is None:
            fail(f"voice '{voice}' not found in the voice bank", _EXIT_ARGS)
        cast[alias] = {
            "voice": voice,
            "profile_id": profile.id if profile is not None else None,
            "voice_fingerprint": profile.reference_sha256 if profile is not None else None,
        }
    payload = {
        "format": DIALOGUE_FORMAT,
        "execution_strategy": "turn-by-turn-v1",
        "provider": provider,
        "model": args.model,
        "cast": cast,
        "style_prompt_sha256": _sha256_text(style_prompt or ""),
        "prompt_mode": prompt_mode,
        "trim_speech": not getattr(args, "no_trim", False),
        "omnivoice": {
            "mode": getattr(args, "mode", None),
            "seed": OMNIVOICE_DEFAULT_SEED if provider == "omnivoice-local" else None,
            "steps": OMNIVOICE_DEFAULT_STEPS if provider == "omnivoice-local" else None,
            "guidance": OMNIVOICE_DEFAULT_GUIDANCE_SCALE if provider == "omnivoice-local" else None,
        },
        "turns": [
            {
                "turn_index": chunk.number,
                "id": chunk.id,
                "speaker": chunk.speaker,
                "voice": chunk.voice,
                "voice_fingerprint": chunk.voice_fingerprint
                or (profiles[chunk.voice].reference_sha256 if chunk.voice in profiles else None),
                "text_sha256": _sha256_text(chunk.text),
                "pause_after_ms": chunk.pause_after_ms,
            }
            for chunk in (chunks or [])
        ]
        if chunks and any(chunk.speaker is not None for chunk in chunks)
        else [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _gemini_dialogue_identity(
    args: argparse.Namespace,
    style_prompt: str | None,
    prompt_mode: str,
    chunks: list[ScriptChunk] | None = None,
) -> str | None:
    """Compatibility wrapper for callers from the former OpenRouter-only path."""
    if getattr(args, "provider", None) != "openrouter-tts":
        return None
    return _dialogue_synthesis_identity(args, style_prompt, prompt_mode, chunks)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _polza_direct_cost_kwargs(cost: float, cost_exact: str | None) -> dict:
    """Compatibility wrapper for ``services.costs.polza_direct_cost_kwargs``."""
    return costs.polza_direct_cost_kwargs(cost, cost_exact)


def _recovered_attempt_cost_kwargs(recovery: dict[str, Any]) -> dict:
    """Compatibility wrapper for ``services.costs.recovered_attempt_cost_kwargs``."""
    return costs.recovered_attempt_cost_kwargs(recovery)


def _direct_cost_kwargs(provider: str, result) -> dict:
    """Compatibility wrapper for ``services.costs.direct_cost_kwargs``."""
    return costs.direct_cost_kwargs(provider, result)


def _public_runtime_receipt(result) -> dict[str, str] | None:
    receipt = (result.raw_metadata or {}).get("runtime_receipt")
    if receipt is None:
        return None
    required_keys = {"model_id", "sha256", "quantization", "license", "provenance"}
    if not isinstance(receipt, dict) or set(receipt) != required_keys:
        raise RuntimeError("Local TTS provider returned an invalid public runtime receipt")
    if not all(isinstance(value, str) and value for value in receipt.values()):
        raise RuntimeError("Local TTS provider returned an invalid public runtime receipt")
    return dict(receipt)


def _public_voice_selection(result) -> dict[str, object] | None:
    selection = (result.raw_metadata or {}).get("voice_selection")
    if selection is None:
        return None
    if not isinstance(selection, dict):
        raise RuntimeError("Local TTS provider returned an invalid public voice selection")
    kind = selection.get("kind")
    if kind == "auto-voice":
        expected = {
            "kind": "auto-voice",
            "named_preset": False,
            "voice_cloning": False,
            "voice_design": False,
        }
        if selection != expected:
            raise RuntimeError("Local TTS provider returned an invalid public voice selection")
        return dict(expected)
    if kind == "bank-preset":
        expected = {
            "kind": "bank-preset",
            "voice_id": selection.get("voice_id"),
            "voice_fingerprint": selection.get("voice_fingerprint"),
        }
        if (
            not isinstance(expected["voice_id"], str)
            or not expected["voice_id"]
            or not isinstance(expected["voice_fingerprint"], str)
            or len(expected["voice_fingerprint"]) != 64
        ):
            raise RuntimeError("Local TTS provider returned an invalid public voice selection")
        if selection != expected:
            raise RuntimeError("Local TTS provider returned an invalid public voice selection")
        return dict(expected)
    if kind == "built-in-style-condition":
        expected = {
            "kind": "built-in-style-condition",
            "condition": OMNIVOICE_STYLE_CONDITION,
            "named_preset": False,
            "voice_cloning": False,
            "voice_design": False,
        }
        if selection != expected:
            raise RuntimeError("Local TTS provider returned an invalid public voice selection")
        return dict(expected)
    if kind in ("reference-clone", "design-instruction"):
        expected = {
            "kind": kind,
            "named_preset": False,
            "voice_cloning": kind == "reference-clone",
            "voice_design": kind == "design-instruction",
        }
        if selection != expected:
            raise RuntimeError("Local TTS provider returned an invalid public voice selection")
        return dict(expected)
    raise RuntimeError("Local TTS provider returned an invalid public voice selection")


def _public_voice_session(result) -> dict[str, object] | None:
    session = (result.raw_metadata or {}).get("voice_session")
    if session is None:
        return None
    if (
        not isinstance(session, dict)
        or set(session) != {"strategy", "seed", "internal_text_chunk_size"}
        or session.get("strategy")
        not in {
            "auto-voice-native-session",
            "bank-preset-native-session",
            "single-native-invocation-internal-text-chunking",
            "reference-isolated-native-session",
            "design-instruction-native-session",
        }
        or isinstance(session.get("seed"), bool)
        or not isinstance(session.get("seed"), int)
        or isinstance(session.get("internal_text_chunk_size"), bool)
        or not isinstance(session.get("internal_text_chunk_size"), int)
    ):
        raise RuntimeError("Local TTS provider returned an invalid public voice session")
    return dict(session)


def _public_artifact_projection(result) -> dict[str, Any]:
    """The three public OmniVoice projection fields for one saved chunk artifact.

    Bundled as the single call-time seam the moved part loop uses, so the exact
    validation and output shape of each projection stays in ``cli``.
    """
    return {
        "runtime_receipt": _public_runtime_receipt(result),
        "voice_selection": _public_voice_selection(result),
        "voice_session": _public_voice_session(result),
    }


def _default_voice(args: argparse.Namespace) -> str | None:
    """Compatibility wrapper for ``services.prepare.default_voice``."""
    return default_voice(args)


def _resume_guard_voice(
    args: argparse.Namespace, gemini_report: dict[str, Any] | None
) -> str | None:
    """Voice the early paid-marker resume guard must match for this exact command.

    A validated dialogue run without ``--voice`` binds its first cast voice as the
    run identity later in ``services.prepare.prepare_generation_identity``, not the
    provider default. Mirroring that choice here lets a saved raw attempt recover
    locally instead of being falsely reported as an unconfirmed paid submit; a
    genuine provider/model/voice mismatch still fails closed.
    """
    requested_voice = args.voice
    if not requested_voice and gemini_report:
        return next(iter(gemini_report["speaker_voice_map"].values()))
    return requested_voice or _default_voice(args)


def read_api_key(args: argparse.Namespace) -> str:
    if args.provider in ("polza-chat-audio", "polza-tts"):
        try:
            return read_polza_key()
        except RuntimeError as e:
            fail(str(e), _EXIT_NO_KEY)
    if args.provider == "openrouter-tts":
        try:
            return read_openrouter_key()
        except RuntimeError as e:
            fail(str(e), _EXIT_NO_KEY)
    if args.provider in {"qwen-local", "omnivoice-local"}:
        return ""
    raise RuntimeError(f"Unsupported provider: {args.provider}")


def build_provider(
    args: argparse.Namespace, api_key: str, style_prompt: str | None, prompt_mode: str
) -> TTSProvider:
    """Compatibility wrapper for ``services.provider_factory.build_tts_provider``.

    OmniVoice argument validation still runs here before any construction, so
    ``cli.build_provider`` stays the external entry point and its exact usage
    errors keep their existing strings and codes. Only the typed configuration
    errors are translated; the unknown-provider ``RuntimeError`` propagates.
    """
    if args.provider == "omnivoice-local":
        _validate_omnivoice_options(args)
    try:
        return provider_factory.build_tts_provider(args, api_key, style_prompt, prompt_mode)
    except provider_factory.ProviderConfigurationError as exc:
        fail(str(exc), _EXIT_ARGS)


def _bind_dialogue_voice_bank_providers(
    provider: Any,
    catalog: VoiceBankCatalog,
    speaker_voice_map: dict[str, str],
) -> dict[str, OmniVoiceLocalTTSProvider]:
    """Compatibility wrapper for ``services.provider_factory`` dialogue binding."""
    try:
        return provider_factory.bind_dialogue_voice_bank_providers(
            provider, catalog, speaker_voice_map
        )
    except provider_factory.DialogueProviderBindingError as exc:
        fail(str(exc), _EXIT_PROVIDER)


def fetch_pricing_snapshot(provider: str, api_key: str, model: str) -> dict | None:
    """Compatibility wrapper for ``services.cost_enrichment.fetch_pricing_snapshot``.

    Forwards the CLI's currently bound (and monkeypatched) pricing lookups, so
    ``cli.fetch_pricing_snapshot`` stays the patchable paid-preflight seam while
    the provider-to-lookup selection lives in the service.
    """
    return cost_enrichment.fetch_pricing_snapshot(
        provider,
        api_key,
        model,
        fetch_polza_pricing=fetch_polza_model_pricing,
        fetch_openrouter_pricing=fetch_openrouter_model_pricing,
    )


def json_safe_metadata(value: Any) -> Any:
    """Compatibility wrapper for ``services.costs.json_safe_metadata``."""
    return costs.json_safe_metadata(value)


def attach_costs(provider, api_key, model, run_started_at, chunks):
    """Compatibility wrapper for ``services.cost_enrichment.attach_costs``.

    Forwards the CLI's currently bound history lookups so tests that patch
    ``cli.fetch_polza_generation_detail`` / ``cli.fetch_openrouter_generation_detail``
    keep steering the extracted body.
    """
    return cost_enrichment.attach_costs(
        provider,
        api_key,
        model,
        run_started_at,
        chunks,
        fetch_polza_detail=fetch_polza_generation_detail,
        fetch_openrouter_detail=fetch_openrouter_generation_detail,
    )


def summarize_costs(provider: str, chunks: list[ChunkArtifact]) -> tuple:
    """Compatibility wrapper for ``services.cost_enrichment.summarize_costs``."""
    return cost_enrichment.summarize_costs(provider, chunks)


def generation_source(provider: str) -> str:
    """Compatibility wrapper for ``services.cost_enrichment.generation_source``."""
    return cost_enrichment.generation_source(provider)


if __name__ == "__main__":
    main()
