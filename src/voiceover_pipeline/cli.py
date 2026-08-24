import argparse
import glob as glob_mod
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn, cast

from .artifacts import (
    build_chunks_manifest,
    build_manifest_json,
    build_run_manifest,
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
from .asr_timing_bridge import asr_result_to_timing
from .config import (
    DEFAULT_ASR_COMPUTE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_FALLBACK_VOICE,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_OPENROUTER_TTS_VOICE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLZA_TTS_VOICE,
    DEFAULT_PROVIDER,
    DEFAULT_QWEN_VOICE,
    DEFAULT_SCRIPT_DIR,
    DEFAULT_TIMING_COMPUTE,
    DEFAULT_TIMING_DEVICE,
    DEFAULT_TIMING_LANGUAGE,
    DEFAULT_TIMING_MODEL,
    DEFAULT_TIMING_PROVIDER,
    DEFAULT_VOICE,
    ELEVENLABS_TTS_VOICES,
    GEMINI_TTS_VOICES,
    OMNIVOICE_DEFAULT_GUIDANCE_SCALE,
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
    chunks_from_validation,
    dialogue_turns_from_validation,
    is_dialogue_format,
    validate_gemini_dialogue_file,
)
from .local_runtime.contracts import OmniVoiceRequest
from .local_runtime.transports.audio_cpp_cli import NATIVE_AUDIO_CPP_EXECUTABLE_ENV
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
from .models import ASRContextHints, ASRRequest, ASRRuntimeChoice, ChunkArtifact, ScriptChunk
from .omnivoice_design import normalize_omnivoice_design_instruction
from .omnivoice_voice_bank import (
    VoiceBankCatalog,
    VoiceBankError,
    load_voice_bank,
    resolve_bank_profile,
)
from .pricing import (
    cost_from_generation,
    fetch_openrouter_generation_detail,
    fetch_openrouter_model_pricing,
    fetch_polza_generation_costs,
    fetch_polza_model_pricing,
)
from .providers import (
    OmniVoiceLocalTTSProvider,
    OpenRouterTTSProvider,
    PolzaChatAudioProvider,
    PolzaTTSProvider,
    QwenLocalTTSProvider,
    TTSProvider,
)
from .providers.asr_registry import (
    ASRProviderNotFoundError,
    get_asr_provider_spec,
    list_asr_provider_specs,
)
from .providers.audio_cpp_omnivoice_tts import omnivoice_local_dependency_probe
from .providers.base import TranscriptionProvider, validate_asr_response
from .retry import RetryPolicy, run_with_retry
from .run_state import (
    LOG_FILE,
    STATE_FILE,
    GenerationLogger,
    append_error,
    atomic_write_json,
    completed_numbers,
    initial_state,
    load_state,
    script_hash,
    state_chunks_as_artifacts,
    upsert_completed_chunk,
)
from .script_splitter import split_markdown_by_delimiter
from .tts_prompting import read_style_prompt_from_file, resolve_prompt_mode
from .voiceover_script import (
    VOICEOVER_FORMAT,
    chunks_from_voiceover_report,
    detect_frontmatter_format,
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


# ═══════════════════════════════════════════════════════════════════════════════
# CliError
# ═══════════════════════════════════════════════════════════════════════════════


class CliError(RuntimeError):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def gemini_chunks_from_validation(report: dict[str, Any]) -> list[ScriptChunk]:
    """Compatibility alias for callers that still inspect section chunks."""
    return chunks_from_validation(report)


def fail(message: str, code: int) -> NoReturn:
    raise CliError(message, code)


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
    except CliError as exc:
        _emit_error(args, str(exc), exc.code)
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
    omnivoice.add_argument("--design-instruction", type=str, default=None)
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
        normalize_omnivoice_design_instruction(design_instruction)
    except ValueError as exc:
        fail(str(exc), _EXIT_ARGS)
    try:
        OmniVoiceRequest(mode="design", instruction=design_instruction)
    except ValueError as exc:
        fail(str(exc), _EXIT_ARGS)


def _resolve_script_format(script_path: Path, requested_format: str) -> str:
    detected_format = detect_frontmatter_format(script_path)
    script_format = requested_format
    if (
        detected_format is not None
        and script_format == "markdown"
        and (detected_format == VOICEOVER_FORMAT or is_dialogue_format(detected_format))
    ):
        script_format = detected_format
    if script_format is None:
        return "markdown"
    return DIALOGUE_FORMAT if is_dialogue_format(script_format) else script_format


def generate(args: argparse.Namespace) -> None:
    if getattr(args, "json_output", False) and getattr(args, "json_events", False):
        fail(
            "--json and --json-events are mutually exclusive; use one machine output mode.",
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
    if not is_dialogue_format(script_format):
        try:
            chunks = prepare_local_tts_chunks(chunks, args.provider, args.model)
        except ValueError as exc:
            fail(str(exc), _EXIT_ARGS)
    if not chunks:
        fail("Script produced no chunks. Check delimiter and content.", _EXIT_ARGS)
    original_chunk_count = len(chunks)
    if args.limit_chunks is not None:
        if args.limit_chunks <= 0:
            fail("--limit-chunks must be greater than zero", _EXIT_ARGS)
        chunks = chunks[: args.limit_chunks]
    requested_fragment_count = len(chunks)
    if (
        args.provider == "omnivoice-local"
        and args.model == OMNIVOICE_LOCAL_MODEL_ID
        and not is_dialogue_format(script_format)
    ):
        bank_profile = getattr(args, "voice_bank_profile", None)
        bank_catalog = getattr(args, "voice_bank_catalog", None)
        if getattr(args, "mode", "preset") == "preset" and bank_profile is not None:
            reference_audio_path = (
                str(bank_catalog.root / bank_profile.reference_audio)
                if bank_catalog is not None
                else str(Path(bank_profile.reference_audio))
            )
            chunks = merge_omnivoice_session_fragments(
                chunks,
                mode="clone",
                reference_audio_path=reference_audio_path,
                reference_text=bank_profile.reference_text,
            )
        else:
            chunks = merge_omnivoice_session_fragments(
                chunks,
                mode=getattr(args, "mode", "preset"),
                reference_audio_path=getattr(args, "reference_audio", None),
                reference_text=getattr(args, "reference_text", None),
                design_instruction=getattr(args, "design_instruction", None),
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

    if getattr(args, "with_timings", False):
        _preflight_timing_dependency(
            getattr(args, "timing_provider", "faster-whisper"),
        )

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
        elif _has_paid_chunk_audio(paths) and not args.confirm_delete_paid_audio:
            fail(
                "Refusing to delete existing paid chunk audio. Use --resume, or add --confirm-delete-paid-audio with --overwrite.",
                _EXIT_PROVIDER,
            )
        if args.overwrite:
            _safe_remove_run_dir(paths.output_root, args.output_dir)

    _ensure_run_dirs(paths)

    requested_voice = args.voice
    if gemini_report:
        args.speaker_voice_map = gemini_report["speaker_voice_map"]
        args.voice = requested_voice or next(iter(args.speaker_voice_map.values()))
        if args.provider == "omnivoice-local":
            chunks = _bind_omnivoice_dialogue_fingerprints(chunks, args.voice_bank_catalog)
    else:
        args.speaker_voice_map = {}
        args.voice = requested_voice or _default_voice(args)

    style_prompt = _resolve_provider_style_prompt(args)
    if (
        gemini_report
        and not args.no_style_prompt
        and args.style_prompt is None
        and args.style_prompt_file is None
    ):
        style_prompt = gemini_report["style_prompt"]
    prompt_mode = resolve_prompt_mode(args.provider, args.model)
    _preflight_dialogue_resume(args, chunks, paths, style_prompt, prompt_mode)
    api_key = read_api_key(args)
    provider_for_generation: Any = build_provider(args, api_key, style_prompt, prompt_mode)
    if args.provider == "omnivoice-local" and gemini_report:
        catalog = args.voice_bank_catalog
        if not isinstance(provider_for_generation, OmniVoiceLocalTTSProvider):
            fail("omnivoice-local dialogue did not build an OmniVoice provider", _EXIT_PROVIDER)
        providers: dict[str, OmniVoiceLocalTTSProvider] = {}
        for voice_id in gemini_report["speaker_voice_map"].values():
            profile, reference_path = resolve_bank_profile(catalog, voice_id)
            providers[voice_id] = provider_for_generation.for_voice_bank_profile(
                profile, reference_path
            )
        provider_for_generation = providers
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

    current_hash = script_hash(chunks)
    dialogue_run = is_dialogue_format(getattr(args, "format", "markdown"))
    current_voice_identity = None if dialogue_run else _omnivoice_voice_identity(args)
    synthesis_identity = (
        _dialogue_synthesis_identity(args, style_prompt, prompt_mode, chunks)
        if dialogue_run
        else None
    )
    state = load_state(state_path)
    if state and args.resume:
        if state.get("script_hash") != current_hash:
            logger.event("error", "resume_rejected", reason="script_hash_mismatch")
            fail(
                "Cannot resume: script chunks do not match the previous run_state.json.",
                _EXIT_PROVIDER,
            )
        if dialogue_run:
            if "synthesis_identity" not in state:
                logger.event("error", "resume_rejected", reason="synthesis_identity_missing")
                fail(
                    "Cannot resume: run state predates dialogue synthesis identity; start a fresh run instead of mixing artifacts.",
                    _EXIT_PROVIDER,
                )
            if state.get("synthesis_identity") != synthesis_identity:
                logger.event("error", "resume_rejected", reason="synthesis_identity_mismatch")
                fail("Cannot resume: dialogue synthesis identity changed.", _EXIT_PROVIDER)
        elif "voice_identity" in state and current_voice_identity is not None:
            if state.get("voice_identity") != current_voice_identity:
                logger.event("error", "resume_rejected", reason="voice_identity_mismatch")
                fail("Cannot resume: voice identity changed.", _EXIT_PROVIDER)
        logger.event("info", "resume_detected", completed=state.get("completed_count", 0))
    elif state and not args.resume:
        logger.event("info", "state_replaced", reason="fresh_run")
        state = initial_state(
            provider=args.provider,
            model=args.model,
            voice=args.voice,
            script_path=args.script,
            chunks=chunks,
            script_format=getattr(args, "format", "markdown"),
            run_id=paths.prefix,
            limited_to_chunks=getattr(args, "limit_chunks", None),
            voice_identity=current_voice_identity,
            synthesis_identity=synthesis_identity,
        )
    elif args.resume:
        state = initial_state(
            provider=args.provider,
            model=args.model,
            voice=args.voice,
            script_path=args.script,
            chunks=chunks,
            script_format=getattr(args, "format", "markdown"),
            run_id=paths.prefix,
            limited_to_chunks=getattr(args, "limit_chunks", None),
            voice_identity=current_voice_identity,
            synthesis_identity=synthesis_identity,
        )
        if dialogue_run:
            if any(paths.chunks_dir.glob("*.mp3")):
                logger.event("error", "resume_rejected", reason="orphan_dialogue_audio")
                fail(
                    "Cannot resume: orphan dialogue audio exists without trusted run state.",
                    _EXIT_PROVIDER,
                )
        else:
            _recover_existing_chunks(
                state, chunks, paths.chunks_dir, ffprobe_path, args.model, args.voice
            )
            logger.event("info", "resume_recovered", completed=state.get("completed_count", 0))
    else:
        state = initial_state(
            provider=args.provider,
            model=args.model,
            voice=args.voice,
            script_path=args.script,
            chunks=chunks,
            script_format=getattr(args, "format", "markdown"),
            run_id=paths.prefix,
            limited_to_chunks=getattr(args, "limit_chunks", None),
            voice_identity=current_voice_identity,
            synthesis_identity=synthesis_identity,
        )
    atomic_write_json(state_path, state)

    chunk_artifacts: list[ChunkArtifact] = (
        state_chunks_as_artifacts(state, paths.chunks_dir) if args.resume else []
    )
    chunk_artifacts_by_number = {artifact.number: artifact for artifact in chunk_artifacts}
    completed = completed_numbers(state)
    total_duration_ms = max((artifact.end_ms for artifact in chunk_artifacts), default=0)
    if dialogue_run and chunk_artifacts:
        total_duration_ms += chunk_artifacts[-1].pause_after_ms
    retry_policy = RetryPolicy(
        attempts=args.retries,
        delay_seconds=args.retry_delay,
        max_delay_seconds=args.retry_max_delay,
        enabled=not args.no_retry,
    )

    for chunk in chunks:
        output_path = paths.chunks_dir / f"{chunk.id}.mp3"
        if chunk.number in completed and output_path.exists():
            if dialogue_run:
                completed_artifact = chunk_artifacts_by_number.get(chunk.number)
                if (
                    completed_artifact is None
                    or completed_artifact.audio_sha256 is None
                    or completed_artifact.audio_sha256 != _sha256_file(output_path)
                ):
                    logger.event("error", "resume_rejected", reason="dialogue_audio_hash_mismatch")
                    fail(
                        "Cannot resume: dialogue audio does not match the trusted run state.",
                        _EXIT_PROVIDER,
                    )
            logger.event(
                "info",
                "chunk_skipped_resume",
                chunk=chunk.number,
                id=chunk.id,
                file=output_path.name,
            )
            _emit_json_event(
                args, "chunk_skipped", chunk=chunk.number, id=chunk.id, reason="resume"
            )
            continue
        if not args.json_output:
            print(f"Generating {chunk.id}/{len(chunks):02d}: {output_path.name}")
        logger.event(
            "info", "chunk_started", chunk=chunk.number, id=chunk.id, file=output_path.name
        )
        _emit_json_event(args, "chunk_started", chunk=chunk.number, id=chunk.id)

        def synthesize_current_chunk():
            selected_provider: Any = provider
            if isinstance(provider, dict):
                selected_provider = provider[chunk.voice]
            if not isinstance(selected_provider, OpenRouterTTSProvider):
                return selected_provider.synthesize_chunk(chunk.text, chunk.id)
            try:
                return selected_provider.synthesize_chunk(chunk.text, chunk.id, voice=chunk.voice)
            except TypeError as error:
                if "unexpected keyword argument 'voice'" not in str(error):
                    raise
                return selected_provider.synthesize_chunk(chunk.text, chunk.id)

        try:
            result = run_with_retry(
                synthesize_current_chunk,
                policy=retry_policy,
                on_retry=lambda attempt, error, delay: _log_retry(
                    logger, args, chunk, attempt, error, delay
                ),
            )
        except Exception as e:
            append_error(state, chunk_id=chunk.id, message=str(e))
            atomic_write_json(state_path, state)
            logger.event("error", "chunk_failed", chunk=chunk.number, id=chunk.id, error=str(e))
            _emit_json_event(args, "chunk_failed", chunk=chunk.number, id=chunk.id, error=str(e))
            fail(f"Failed to synthesize {chunk.id}: {e}", _EXIT_PROVIDER)
        logger.event(
            "info",
            "chunk_provider_response",
            chunk=chunk.number,
            id=chunk.id,
            generation_id=result.generation_id,
        )
        try:
            write_audio_as_mp3(ffmpeg_path, result.audio_bytes, result.audio_format, output_path)
        except Exception as e:
            append_error(state, chunk_id=chunk.id, message=str(e))
            atomic_write_json(state_path, state)
            logger.event(
                "error", "chunk_write_failed", chunk=chunk.number, id=chunk.id, error=str(e)
            )
            fail(f"Failed to write chunk audio {output_path}: {e}", _EXIT_OUTPUT)
        logger.event(
            "info", "chunk_file_saved", chunk=chunk.number, id=chunk.id, file=output_path.name
        )
        if not args.no_trim:
            try:
                trim_final_silence(ffmpeg_path, ffprobe_path, output_path)
                logger.event("info", "chunk_trimmed", chunk=chunk.number, id=chunk.id)
            except Exception as e:
                append_error(state, chunk_id=chunk.id, message=str(e))
                atomic_write_json(state_path, state)
                logger.event(
                    "error", "chunk_trim_failed", chunk=chunk.number, id=chunk.id, error=str(e)
                )
                fail(f"Failed to trim chunk audio {output_path}: {e}", _EXIT_OUTPUT)

        duration_ms = mp3_duration_ms(ffprobe_path, output_path)
        start_ms = total_duration_ms
        end_ms = start_ms + duration_ms
        total_duration_ms = end_ms + (chunk.pause_after_ms if dialogue_run else 0)

        artifact = ChunkArtifact(
            number=chunk.number,
            id=chunk.id,
            file=output_path.name,
            duration_ms=duration_ms,
            duration_sec=round(duration_ms / 1000, 3),
            start_ms=start_ms,
            end_ms=end_ms,
            text_characters=len(chunk.text),
            transcript=None if dialogue_run else result.transcript,
            client_path=result.client_path,
            generation_id=result.generation_id,
            speaker=chunk.speaker,
            voice=chunk.voice or args.voice,
            voice_fingerprint=chunk.voice_fingerprint,
            turn_index=chunk.number if dialogue_run else None,
            speech_duration_ms=duration_ms if dialogue_run else None,
            audio_sha256=_sha256_file(output_path) if dialogue_run else None,
            pause_after_ms=chunk.pause_after_ms,
            runtime_receipt=_public_runtime_receipt(result),
            voice_selection=_public_voice_selection(result),
            voice_session=_public_voice_session(result),
            **_direct_cost_kwargs(args.provider, result),
        )
        chunk_artifacts_by_number[chunk.number] = artifact
        upsert_completed_chunk(
            state,
            artifact=artifact,
            model=args.model,
            voice=chunk.voice or args.voice,
            text=chunk.text,
            include_text=not dialogue_run,
            include_transcript=not dialogue_run,
        )
        atomic_write_json(state_path, state)
        logger.event(
            "info", "chunk_state_saved", chunk=chunk.number, id=chunk.id, state=state_path.name
        )
        _emit_json_event(
            args,
            "chunk_saved",
            chunk=chunk.number,
            id=chunk.id,
            file=output_path.name,
            duration_ms=duration_ms,
        )
        if not args.json_output:
            print(f"Saved {output_path.name}: {duration_ms} ms")

    chunk_artifacts = [
        chunk_artifacts_by_number[number] for number in sorted(chunk_artifacts_by_number)
    ]

    time.sleep(2)
    chunk_artifacts = attach_costs(
        args.provider, api_key, args.model, run_started_at, chunk_artifacts
    )
    cost_total, cost_total_exact, cost_currency, cost_source = summarize_costs(
        args.provider, chunk_artifacts
    )

    chunks_manifest = build_chunks_manifest(
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        style_prompt=style_prompt,
        script=args.script,
        chunks_dir=paths.chunks_dir,
        pricing_snapshot=pricing_snapshot,
        cost_exact_available=cost_total is not None,
        cost_total=cost_total,
        cost_total_exact=cost_total_exact,
        cost_currency=cost_currency,
        cost_source=cost_source,
        chunk_artifacts=chunk_artifacts,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        prompt_mode=prompt_mode,
        script_format=getattr(args, "format", "markdown"),
        speaker_voice_map=getattr(args, "speaker_voice_map", None) or None,
    )
    try:
        write_json(paths.chunks_json, chunks_manifest)
    except Exception as e:
        fail(f"Failed to write {paths.chunks_json}: {e}", _EXIT_OUTPUT)

    try:
        logger.event("info", "concat_started", output=paths.full_mp3.name)
        if dialogue_run:
            concat_dialogue_turns(
                ffmpeg_path,
                [
                    (paths.chunks_dir / artifact.file, artifact.pause_after_ms)
                    for artifact in chunk_artifacts
                ],
                paths.full_mp3,
            )
        else:
            concat_mp3_chunks(ffmpeg_path, paths.chunks_dir, paths.full_mp3)
    except Exception as e:
        append_error(state, chunk_id=None, message=str(e))
        atomic_write_json(state_path, state)
        logger.event("error", "concat_failed", error=str(e))
        fail(f"Failed to concat MP3 chunks: {e}", _EXIT_OUTPUT)
    main_duration_ms = mp3_duration_ms(ffprobe_path, paths.full_mp3)
    logger.event(
        "info", "concat_complete", output=paths.full_mp3.name, duration_ms=main_duration_ms
    )
    run_manifest = build_run_manifest(chunks_manifest, paths, main_duration_ms)
    try:
        write_json(paths.run_json, run_manifest)
    except Exception as e:
        fail(f"Failed to write {paths.run_json}: {e}", _EXIT_OUTPUT)

    timing_info = None
    if getattr(args, "with_timings", False):
        try:
            logger.event("info", "timings_started", audio=paths.full_mp3.name)
            timing_info = _extract_timings(
                audio_path=paths.full_mp3,
                output_dir=paths.output_root,
                prefix=paths.prefix,
                timing_provider=getattr(args, "timing_provider", "faster-whisper"),
                model=args.timing_model,
                device=args.timing_device,
                compute_type=args.timing_compute,
                language=args.timing_language,
                word_timestamps=args.word_timestamps,
                quiet=args.json_output,
            )
        except ModuleNotFoundError as exc:
            logger.event("error", "timings_failed", error=str(exc))
            fail(
                f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --extra timing-whisper",
                _EXIT_MISSING_DEP,
            )
        except Exception as exc:
            logger.event("error", "timings_failed", error=str(exc))
            fail(
                f"Voiceover generated but timing extraction failed: {exc}",
                _EXIT_WHISPER,
            )

    manifest_json = build_manifest_json(paths, main_duration_ms)
    try:
        write_json(paths.output_root / "manifest.json", manifest_json)
    except Exception as e:
        fail(f"Failed to write manifest.json: {e}", _EXIT_OUTPUT)

    files = _list_artifact_files(paths)
    state["status"] = "completed"
    state["full_mp3"] = str(paths.full_mp3)
    state["main_duration_ms"] = main_duration_ms
    atomic_write_json(state_path, state)
    logger.event("info", "run_complete", run_id=paths.prefix, duration_ms=main_duration_ms)
    _emit_json_event(args, "run_complete", run_id=paths.prefix, duration_ms=main_duration_ms)
    if args.json_output:
        _json_ok(
            {
                "status": "success",
                "provider": args.provider,
                "model": args.model,
                "run_id": paths.prefix,
                "files": files,
                "duration_ms": main_duration_ms,
                "segment_count": timing_info["segment_count"] if timing_info else None,
                "cost": {"total": cost_total, "currency": cost_currency},
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
    script = Path(args.script)
    if not script.exists():
        fail(f"Script file not found: {script}", _EXIT_ARGS)
    chunks = split_markdown_by_delimiter(script, args.delimiter)
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


def transcribe_cmd(args: argparse.Namespace) -> None:
    audio_path = _resolve_audio(args.audio)
    if not audio_path.exists():
        fail(f"Audio file not found: {audio_path}", _EXIT_ARGS)
    hints = _resolve_asr_context(args)
    try:
        spec = get_asr_provider_spec(args.provider)
    except ASRProviderNotFoundError as exc:
        fail(str(exc), _EXIT_ARGS)

    _validate_asr_request_options(args, spec, hints)
    runtime = cast(ASRRuntimeChoice, getattr(args, "runtime", "auto"))
    if spec.provider_id == "nemotron-local" and runtime == "audio-cpp":
        from .providers.audio_cpp_nemotron_asr import audio_cpp_nemotron_asr_dependency_probe
        from .providers.nemotron_asr_local import nemotron_asr_audio_cpp_provider_factory

        health = audio_cpp_nemotron_asr_dependency_probe()
        provider_factory = nemotron_asr_audio_cpp_provider_factory
    elif spec.provider_id == "nemotron-local" and runtime == "python":
        from .providers.nemotron_asr_local import (
            nemotron_asr_python_dependency_probe,
            nemotron_asr_python_provider_factory,
        )

        health = nemotron_asr_python_dependency_probe()
        provider_factory = nemotron_asr_python_provider_factory
    elif runtime == "audio-cpp":
        if not os.environ.get(NATIVE_AUDIO_CPP_EXECUTABLE_ENV, "").strip():
            fail(
                f"ASR provider {spec.provider_id} does not support runtime=audio-cpp "
                "without a native audio.cpp package",
                _EXIT_MISSING_DEP,
            )
        health = spec.dependency_probe()
        provider_factory = spec.factory
    else:
        health = spec.dependency_probe()
        provider_factory = spec.factory
    if not health.available:
        fail(health.remediation, _EXIT_MISSING_DEP)

    model_id = args.model
    if model_id is None:
        model_id = next((model["id"] for model in spec.models if model.get("default")), None)
    request = ASRRequest(
        audio_path=audio_path,
        model_id=model_id,
        language=args.language,
        device=args.device,
        compute=args.compute,
        hints=hints,
        timestamp_mode="word" if args.word_timestamps else "none",
        runtime_choice=runtime,
    )
    provider = provider_factory()
    try:
        raw_result = (
            transcribe_prerecorded_long_form(provider, request)
            if uses_long_form_orchestration(spec.provider_id)
            else provider.transcribe(request)
        )
        result = validate_asr_response(request, raw_result)
    except LongFormASRMediaError as exc:
        fail(str(exc), _EXIT_NO_FFMPEG)
    except ModuleNotFoundError as exc:
        fail(f"Missing dependency for ASR provider {spec.provider_id}: {exc}", _EXIT_MISSING_DEP)
    except Exception as exc:
        fail(f"ASR provider {spec.provider_id} failed: {exc}", _EXIT_PROVIDER)

    capabilities = spec.capabilities
    if result.provider_id != spec.provider_id:
        fail(
            f"ASR provider {spec.provider_id} returned provider ID {result.provider_id}",
            _EXIT_PROVIDER,
        )
    if (
        any(segment.start_s is not None for segment in result.segments)
        and not capabilities.segment_timestamps
    ):
        fail(
            f"ASR provider {spec.provider_id} returned undeclared segment timestamps",
            _EXIT_PROVIDER,
        )
    if result.words and not capabilities.word_timestamps:
        fail(f"ASR provider {spec.provider_id} returned undeclared word timestamps", _EXIT_PROVIDER)
    if result.alignment_origin == "forced" and not capabilities.forced_alignment:
        fail(
            f"ASR provider {spec.provider_id} returned undeclared forced alignment", _EXIT_PROVIDER
        )

    data = _asr_result_payload(result, audio_path)
    if args.json_output:
        _json_ok(data)
    else:
        print(result.transcript)


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
    output_dir = (Path(args.output_dir) / run_id).resolve()

    if output_dir.exists():
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
            else:
                print(f"Skipping: output dir exists: {output_dir}")
            return
        if not args.overwrite:
            fail(
                f"Output dir exists: {output_dir}. Use --overwrite or --skip-existing.",
                _EXIT_PROVIDER,
            )
        _safe_remove_run_dir(output_dir, args.output_dir)

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
    if args.json_output:
        _json_ok(
            {
                "status": "success",
                "files": files,
                "segment_count": timing["segment_count"],
                "duration_ms": timing["total_duration_ms"],
            }
        )
    else:
        print(f"Timings JSON: {files['timings_json']}")
        print(f"SRT: {files['srt']}")
        print(f"Segments: {timing['segment_count']}")


def status_cmd(args: argparse.Namespace) -> None:
    if args.run_id:
        _validate_run_id(args.run_id)
    run_dir = (Path(args.output_dir) / args.run_id).resolve()
    state = load_state(run_dir / STATE_FILE)
    chunks_dir = run_dir / "chunks"
    chunk_files = _continuous_chunk_files(chunks_dir)
    total = int(state.get("chunk_count", len(chunk_files)) if state else len(chunk_files))
    ready = int(state.get("completed_count", len(chunk_files)) if state else len(chunk_files))
    if state:
        ready = len(completed_numbers(state))
    next_chunk = ready + 1 if total == 0 or ready < total else None
    full_audio = _find_full_audio(run_dir)
    timings_json = list(run_dir.glob("*.timings.json"))
    errors = state.get("errors", []) if state else []
    can_resume = run_dir.exists() and ready < total and bool(state or chunk_files)
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
            "openrouter-tts": GEMINI_TTS_VOICES + OPENAI_TTS_VOICES,
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
                "openai": OPENAI_TTS_VOICES,
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
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


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
    model_id = model or next((item["id"] for item in spec.models if item.get("default")), None)
    request = ASRRequest(
        audio_path=audio_path,
        model_id=model_id,
        language=language,
        device=device,
        compute=compute,
        timestamp_mode="word",
    )
    result = validate_asr_response(request, spec.factory().transcribe(request))
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("FFprobe is required to validate generic ASR timestamp bounds")
    source_duration_s = mp3_duration_ms(ffprobe_path, audio_path) / 1000
    timing = asr_result_to_timing(
        result,
        source_audio=str(audio_path.resolve()),
        source_duration_s=source_duration_s,
    )
    return _write_timing_artifacts(audio_path, output_dir, prefix, timing)


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
):
    provider: TranscriptionProvider
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
    elif timing_provider == "groq-whisper":
        from .providers.groq_whisper import GroqWhisperProvider

        effective_model = model or "whisper-large-v3-turbo"
        provider = GroqWhisperProvider(model=effective_model)
    elif timing_provider == "xai-stt":
        from .providers.xai_stt import XAISttProvider

        effective_model = model or "grok-stt"
        provider = XAISttProvider(model=effective_model)
    else:
        from .providers.faster_whisper import FasterWhisperProvider

        effective_model = model or DEFAULT_TIMING_MODEL
        provider = FasterWhisperProvider(
            model_size=effective_model,
            device=device,
            compute_type=compute_type,
        )

    timing = provider.transcribe(
        audio_path=audio_path,
        language=language,
        word_timestamps=word_timestamps,
        quiet=quiet,
    )

    try:
        result = _write_timing_artifacts(audio_path, output_dir, prefix, timing)
    except Exception as e:
        fail(f"Failed to write timing artifacts: {e}", _EXIT_OUTPUT)

    if not quiet:
        print(f"Timings JSON: {output_dir / f'{prefix}.timings.json'}")
        print(f"SRT: {output_dir / f'{prefix}.srt'}")

    return result


def _list_artifact_files(paths) -> dict:
    files = {
        "full_mp3": str(paths.full_mp3),
        "run_json": str(paths.run_json),
        "chunks_json": str(paths.chunks_json),
        "manifest_json": str(paths.output_root / "manifest.json"),
    }
    timings_json = paths.output_root / f"{paths.prefix}.timings.json"
    srt_path = paths.output_root / f"{paths.prefix}.srt"
    if timings_json.exists():
        files["timings_json"] = str(timings_json)
    if srt_path.exists():
        files["srt"] = str(srt_path)
    return files


def _json_ok(data: dict) -> NoReturn:
    data.setdefault("status", "success")
    print(json.dumps(data, ensure_ascii=False))
    sys.exit(_EXIT_OK)


def _json_error(message: str, code: int) -> NoReturn:
    print(json.dumps({"status": "error", "error": message, "code": code}, ensure_ascii=False))
    sys.exit(code)


def _emit_error(args, message: str, code: int) -> NoReturn:
    if getattr(args, "json_output", True):
        _json_error(message, code)
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
    profiles = {profile.id: profile for profile in catalog.profiles}
    bound: list[ScriptChunk] = []
    for chunk in chunks:
        if chunk.voice is None:
            fail("OmniVoice dialogue turn is missing a voice-bank profile", _EXIT_ARGS)
        profile = profiles.get(chunk.voice)
        if profile is None:
            fail(f"voice '{chunk.voice}' not found in the voice bank", _EXIT_ARGS)
        bound.append(replace(chunk, voice_fingerprint=profile.reference_sha256))
    return bound


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


def _direct_cost_kwargs(provider: str, result) -> dict:
    if provider != "polza-tts":
        return {}
    usage = (result.raw_metadata or {}).get("usage_direct")
    if not isinstance(usage, dict):
        return {}
    cost_rub = usage.get("cost_rub") or usage.get("cost")
    if cost_rub is None:
        return {}
    return {
        "cost": float(cost_rub),
        "cost_exact": str(cost_rub),
        "cost_currency": "RUB",
        "cost_rub": float(cost_rub),
        "cost_rub_exact": str(cost_rub),
        "usage": usage,
        "generation_detail_source": "Polza API usage.cost_rub (direct)",
    }


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


def _default_voice(args: argparse.Namespace) -> str | None:
    if args.provider == "polza-tts":
        if args.model and args.model.startswith("elevenlabs/"):
            return DEFAULT_ELEVENLABS_VOICE
        return DEFAULT_POLZA_TTS_VOICE
    if args.provider == "openrouter-tts":
        if args.model and args.model.startswith("openai/"):
            return DEFAULT_OPENAI_TTS_VOICE
        return DEFAULT_OPENROUTER_TTS_VOICE
    if args.provider == "qwen-local":
        return DEFAULT_QWEN_VOICE
    if args.provider == "omnivoice-local":
        return None
    return DEFAULT_VOICE


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
    if args.provider == "polza-chat-audio":
        return PolzaChatAudioProvider(
            api_key=api_key, model=args.model, voice=args.voice, fallback_voice=args.fallback_voice
        )
    if args.provider == "polza-tts":
        return PolzaTTSProvider(api_key=api_key, model=args.model, voice=args.voice)
    if args.provider == "openrouter-tts":
        return OpenRouterTTSProvider(
            api_key=api_key,
            model=args.model,
            voice=args.voice,
            style_prompt=style_prompt,
            prompt_mode=prompt_mode,
        )
    if args.provider == "qwen-local":
        instruct = getattr(args, "qwen_instruct", None)
        provider_kwargs = {
            "mode": args.mode,
            "voice": None if args.mode == "design" else args.voice,
            "instruct": QWEN_INSTRUCT if instruct is None else instruct,
            "sample_path": args.sample,
            "sample_text": getattr(args, "sample_text", None) or "",
        }
        runtime = os.environ.get("VOICEOVER_QWEN_TTS_RUNTIME", "python").strip()
        if runtime == "audio-cpp":
            from .providers.audio_cpp_qwen_tts import AudioCppQwenTTSProvider

            return AudioCppQwenTTSProvider.from_environment(**provider_kwargs)
        if runtime != "python":
            fail(
                "VOICEOVER_QWEN_TTS_RUNTIME must be either 'python' or 'audio-cpp'.",
                _EXIT_ARGS,
            )
        return QwenLocalTTSProvider(
            **provider_kwargs,
        )
    if args.provider == "omnivoice-local":
        _validate_omnivoice_options(args)
        omni_kwargs: dict[str, Any] = {}
        mode = getattr(args, "mode", "preset")
        if mode == "auto":
            omni_kwargs.update({"mode": "auto"})
        elif mode == "preset":
            catalog = getattr(args, "voice_bank_catalog", None)
            profile = getattr(args, "voice_bank_profile", None)
            if catalog is None:
                fail(
                    "omnivoice-local preset mode requires --voice-bank catalog.json",
                    _EXIT_ARGS,
                )
            if profile is None:
                if not is_dialogue_format(getattr(args, "format", "markdown")):
                    fail(
                        "omnivoice-local preset mode requires a resolved voice-bank profile",
                        _EXIT_ARGS,
                    )
                omni_kwargs.update({"mode": "preset"})
            else:
                reference_path = resolve_bank_profile(catalog, profile.id)[1]
                omni_kwargs.update(
                    {
                        "mode": "preset",
                        "voice_bank": (profile, reference_path),
                    }
                )
        elif mode == "clone":
            omni_kwargs.update(
                {
                    "mode": "clone",
                    "reference_audio_path": args.reference_audio,
                    "reference_text": args.reference_text,
                }
            )
        elif mode == "design":
            omni_kwargs.update(
                {
                    "mode": "design",
                    "design_instruction": args.design_instruction,
                }
            )
        return OmniVoiceLocalTTSProvider.from_environment(**omni_kwargs)
    raise RuntimeError(f"Unsupported provider: {args.provider}")


def fetch_pricing_snapshot(provider: str, api_key: str, model: str) -> dict | None:
    if provider in ("polza-chat-audio", "polza-tts"):
        return fetch_polza_model_pricing(api_key, model)
    if provider == "openrouter-tts":
        return fetch_openrouter_model_pricing(model)
    return None


def attach_costs(provider, api_key, model, run_started_at, chunks):
    if provider in {"qwen-local", "omnivoice-local"}:
        enriched = []
        for chunk in chunks:
            enriched.append(
                ChunkArtifact(
                    **{**chunk.__dict__, "cost": 0.0, "cost_exact": "0.0", "cost_currency": "RUB"}
                )
            )
        return enriched
    generations: list[dict[str, Any] | None] = []
    if provider in ("polza-chat-audio", "polza-tts"):
        generations.extend(
            fetch_polza_generation_costs(api_key, model, run_started_at, len(chunks))
        )
    else:
        generations = []
        for chunk in chunks:
            detail = None
            for _ in range(4):
                detail = fetch_openrouter_generation_detail(api_key, chunk.generation_id)
                if detail:
                    break
                time.sleep(3)
            generations.append(detail)
    if len(generations) != len(chunks):
        return chunks
    enriched = []
    for chunk, generation in zip(chunks, generations):
        cost, cost_exact, currency = cost_from_generation(provider, generation)
        enriched.append(
            ChunkArtifact(
                **{
                    **chunk.__dict__,
                    "generation_id": generation.get("id") or chunk.generation_id
                    if generation
                    else chunk.generation_id,
                    "cost_rub": cost if currency == "RUB" else None,
                    "cost_rub_exact": cost_exact if currency == "RUB" else None,
                    "cost": cost,
                    "cost_exact": cost_exact,
                    "cost_currency": currency,
                    "usage": generation.get("usage") if generation else None,
                    "generation_time_ms": generation.get("generationTimeMs")
                    or generation.get("generation_time")
                    if generation
                    else None,
                    "generated_at": generation.get("createdAt") or generation.get("created_at")
                    if generation
                    else None,
                    "generation_detail_source": generation_source(provider) if generation else None,
                }
            )
        )
    return enriched


def summarize_costs(provider: str, chunks: list[ChunkArtifact]) -> tuple:
    if not chunks or any(chunk.cost is None for chunk in chunks):
        return None, None, None, None
    total = sum(float(chunk.cost or 0) for chunk in chunks)
    currency = chunks[0].cost_currency
    source = generation_source(provider)
    return round(total, 8), str(round(total, 8)), currency, source


def generation_source(provider: str) -> str:
    return {
        "polza-chat-audio": "Polza GET /api/v1/history/generations/{id}",
        "polza-tts": "Polza API usage.cost_rub or GET /api/v1/history/generations/{id}",
        "openrouter-tts": "OpenRouter GET /api/v1/generation?id=...",
        "qwen-local": "qwen-local (free)",
        "omnivoice-local": "omnivoice-local (local model; no billing request)",
    }.get(provider, "unknown")


if __name__ == "__main__":
    main()
