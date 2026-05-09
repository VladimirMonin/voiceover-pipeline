import argparse
import glob as glob_mod
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NoReturn

from .artifacts import (
    build_chunks_manifest,
    build_manifest_json,
    build_run_manifest,
    build_run_paths,
    build_srt,
    build_timing_manifest,
    write_json,
)
from .config import (
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
    DEFAULT_VOICE,
    ELEVENLABS_TTS_VOICES,
    GEMINI_TTS_VOICES,
    OPENAI_TTS_VOICES,
    OPENROUTER_TTS_MODELS,
    PODCAST_NARRATION_PROMPT,
    POLZA_TTS_MODELS,
    PROVIDER_DEFAULT_MODELS,
    QWEN_PRESET_SPEAKERS,
    read_openrouter_key,
    read_polza_key,
)
from .media import check_media_tools, concat_mp3_chunks, mp3_duration_ms, trim_final_silence, write_audio_as_mp3
from .models import ChunkArtifact
from .pricing import (
    cost_from_generation,
    fetch_openrouter_generation_detail,
    fetch_openrouter_model_pricing,
    fetch_polza_generation_costs,
    fetch_polza_model_pricing,
)
from .providers import OpenRouterTTSProvider, PolzaChatAudioProvider, PolzaTTSProvider, QwenLocalTTSProvider, TTSProvider
from .script_splitter import split_markdown_by_delimiter
from .tts_prompting import read_style_prompt_from_file, resolve_prompt_mode

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
    args = parser.parse_args()

    try:
        if args.command == "generate":
            generate(args)
        elif args.command == "split":
            split_cmd(args)
        elif args.command == "timings":
            run_timings(args)
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
    gen = subparsers.add_parser("generate", help="Generate chunk MP3 + full MP3 + optional timings.")
    gen.add_argument("--provider", choices=["polza-chat-audio", "polza-tts", "openrouter-tts", "qwen-local"], default=DEFAULT_PROVIDER)
    gen.add_argument("--model", default=argparse.SUPPRESS)
    gen.add_argument("--script", type=Path, default=_find_default_script())
    gen.add_argument("--delimiter", default="******")
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gen.add_argument("--run-id", default="")
    gen.add_argument("--voice", default=None)
    gen.add_argument("--fallback-voice", default=DEFAULT_FALLBACK_VOICE)
    gen.add_argument("--style-prompt", default=None)
    gen.add_argument("--style-prompt-file", type=Path, default=None)
    gen.add_argument("--no-style-prompt", action="store_true")
    gen.add_argument("--no-trim", action="store_true")
    gen.add_argument("--json", dest="json_output", action="store_true", help="Output JSON to stdout.")
    gen.add_argument("--overwrite", action="store_true", help="Overwrite existing run folder.")
    gen.add_argument("--skip-existing", action="store_true", help="Skip if run folder exists.")

    qwen = gen.add_argument_group("qwen-local options")
    qwen.add_argument("--mode", choices=["preset", "clone"], default="preset")
    qwen.add_argument("--sample", type=str, default=None)
    qwen.add_argument("--sample-text", type=str, default="")

    tim = gen.add_argument_group("Whisper timing (optional)")
    tim.add_argument("--with-timings", action="store_true")
    tim.add_argument("--timing-model", default=DEFAULT_TIMING_MODEL, choices=["base", "small", "medium", "large-v3-turbo", "large-v3"])
    tim.add_argument("--timing-device", default=DEFAULT_TIMING_DEVICE, choices=["auto", "cpu", "cuda"])
    tim.add_argument("--timing-compute", default=DEFAULT_TIMING_COMPUTE, choices=["auto", "int8", "int8_float16", "float16", "float32"])
    tim.add_argument("--timing-language", default=DEFAULT_TIMING_LANGUAGE)
    tim.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps.")

    # --------------- split ---------------
    spl = subparsers.add_parser("split", help="Print chunk ids and character counts.")
    spl.add_argument("--script", type=Path, default=_find_default_script())
    spl.add_argument("--delimiter", default="******")
    spl.add_argument("--json", dest="json_output", action="store_true")

    # --------------- timings ---------------
    timp = subparsers.add_parser("timings", help="Extract Whisper timings from audio.")
    timp.add_argument("--audio", type=str, required=True)
    timp.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    timp.add_argument("--run-id", default="")
    timp.add_argument("--model", default=DEFAULT_TIMING_MODEL, choices=["base", "small", "medium", "large-v3-turbo", "large-v3"])
    timp.add_argument("--device", default=DEFAULT_TIMING_DEVICE, choices=["auto", "cpu", "cuda"])
    timp.add_argument("--compute", default=DEFAULT_TIMING_COMPUTE, choices=["auto", "int8", "int8_float16", "float16", "float32"])
    timp.add_argument("--language", default=DEFAULT_TIMING_LANGUAGE)
    timp.add_argument("--json", dest="json_output", action="store_true")
    timp.add_argument("--word-timestamps", action="store_true")
    timp.add_argument("--overwrite", action="store_true")
    timp.add_argument("--skip-existing", action="store_true", help="Skip if output dir exists.")

    # --------------- doctor ---------------
    doc = subparsers.add_parser("doctor", help="Check environment and dependencies.")
    doc.add_argument("--json", dest="json_output", action="store_true")
    doc.add_argument("--provider", default=None, choices=["polza-chat-audio", "polza-tts", "openrouter-tts", "qwen-local"], help="Check provider-specific requirements.")
    doc.add_argument("--with-timings", action="store_true", help="Check timing dependencies.")
    doc.add_argument("--timing-device", default="cpu", choices=["auto", "cpu", "cuda"], help="Requested timing device for dependency check.")

    # --------------- validate ---------------
    val = subparsers.add_parser("validate", help="Validate script for generation.")
    val.add_argument("--script", type=Path, required=True)
    val.add_argument("--delimiter", default="******")
    val.add_argument("--max-chunk-chars", type=int, default=2000)
    val.add_argument("--json", dest="json_output", action="store_true")

    # --------------- list ---------------
    lst = subparsers.add_parser("list", help="List available providers, voices, or timing models.")
    lst.add_argument("target", choices=["providers", "voices", "timing-models"])
    lst.add_argument("--provider", default=None, help="Filter voices by provider.")
    lst.add_argument("--json", dest="json_output", action="store_true")

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# generate
# ═══════════════════════════════════════════════════════════════════════════════

_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
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
        fail(f"Invalid --run-id: '.' and '..' not allowed", _EXIT_ARGS)
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
            fail(f"Refusing to remove directory outside output-dir: {resolved} (output-dir: {base})", _EXIT_OUTPUT)
    shutil.rmtree(resolved)


def _ensure_run_dirs(paths) -> None:
    try:
        paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Failed to create output directory {paths.chunks_dir}: {e}", _EXIT_OUTPUT)


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved == Path.cwd().resolve():
        fail(f"--output-dir cannot be the current working directory: {resolved}", _EXIT_ARGS)
    home = Path.home().resolve()
    if resolved == home:
        fail(f"--output-dir cannot be your home directory: {resolved}", _EXIT_ARGS)
    root = resolved.anchor or "C:\\"
    if str(resolved).rstrip("\\/") == root.rstrip("\\/"):
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


def generate(args: argparse.Namespace) -> None:
    try:
        ffmpeg_path, ffprobe_path = check_media_tools()
    except RuntimeError as e:
        fail(str(e), _EXIT_NO_FFMPEG)
    _resolve_model(args)
    _validate_model_for_provider(args.provider, args.model)
    chunks = split_markdown_by_delimiter(args.script, args.delimiter)
    if not chunks:
        fail("Script produced no chunks. Check delimiter and content.", _EXIT_ARGS)
    if args.run_id:
        _validate_run_id(args.run_id)
    _validate_output_dir(args.output_dir)
    paths = build_run_paths(args.output_dir, args.model, args.run_id or None)

    if paths.output_root.exists():
        if args.skip_existing:
            files = _list_artifact_files(paths)
            _json_ok({"status": "skipped", "reason": "run folder exists", "run_id": paths.prefix, "files": files})
            return
        if not args.overwrite:
            fail(
                f"Run folder already exists: {paths.output_root}. Use --overwrite to replace or --skip-existing.",
                _EXIT_PROVIDER,
            )
        _safe_remove_run_dir(paths.output_root, args.output_dir)

    _ensure_run_dirs(paths)

    api_key = read_api_key(args)
    args.voice = args.voice or _default_voice(args)
    style_prompt = _resolve_style_prompt(args)
    prompt_mode = resolve_prompt_mode(args.provider, args.model)
    provider = build_provider(args, api_key, style_prompt, prompt_mode)
    pricing_snapshot = fetch_pricing_snapshot(args.provider, api_key, args.model)

    _generate_step(args, provider, ffmpeg_path, ffprobe_path, chunks, api_key, pricing_snapshot, paths, style_prompt, prompt_mode)


def _generate_step(args, provider, ffmpeg_path, ffprobe_path, chunks, api_key, pricing_snapshot, paths, style_prompt, prompt_mode) -> None:
    run_started_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    chunk_artifacts: list[ChunkArtifact] = []
    total_duration_ms = 0

    for chunk in chunks:
        output_path = paths.chunks_dir / f"{chunk.id}.mp3"
        if not args.json_output:
            print(f"Generating {chunk.id}/{len(chunks):02d}: {output_path.name}")

        result = provider.synthesize_chunk(chunk.text, chunk.id)
        try:
            write_audio_as_mp3(ffmpeg_path, result.audio_bytes, result.audio_format, output_path)
        except Exception as e:
            fail(f"Failed to write chunk audio {output_path}: {e}", _EXIT_OUTPUT)
        if not args.no_trim:
            try:
                trim_final_silence(ffmpeg_path, ffprobe_path, output_path)
            except Exception as e:
                fail(f"Failed to trim chunk audio {output_path}: {e}", _EXIT_OUTPUT)

        duration_ms = mp3_duration_ms(ffprobe_path, output_path)
        start_ms = total_duration_ms
        end_ms = start_ms + duration_ms
        total_duration_ms = end_ms

        chunk_artifacts.append(ChunkArtifact(
            number=chunk.number, id=chunk.id, file=output_path.name,
            duration_ms=duration_ms, duration_sec=round(duration_ms / 1000, 3),
            start_ms=start_ms, end_ms=end_ms, text_characters=len(chunk.text),
            transcript=result.transcript, client_path=result.client_path,
            generation_id=result.generation_id,
            **_direct_cost_kwargs(args.provider, result),
        ))
        if not args.json_output:
            print(f"Saved {output_path.name}: {duration_ms} ms")

    time.sleep(2)
    chunk_artifacts = attach_costs(args.provider, api_key, args.model, run_started_at, chunk_artifacts)
    cost_total, cost_total_exact, cost_currency, cost_source = summarize_costs(args.provider, chunk_artifacts)

    chunks_manifest = build_chunks_manifest(
        provider=args.provider, model=args.model, voice=args.voice,
        style_prompt=style_prompt,
        script=args.script, chunks_dir=paths.chunks_dir, pricing_snapshot=pricing_snapshot,
        cost_exact_available=cost_total is not None, cost_total=cost_total,
        cost_total_exact=cost_total_exact, cost_currency=cost_currency, cost_source=cost_source,
        chunk_artifacts=chunk_artifacts, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
        prompt_mode=prompt_mode,
    )
    try:
        write_json(paths.chunks_json, chunks_manifest)
    except Exception as e:
        fail(f"Failed to write {paths.chunks_json}: {e}", _EXIT_OUTPUT)

    try:
        concat_mp3_chunks(ffmpeg_path, paths.chunks_dir, paths.full_mp3)
    except Exception as e:
        fail(f"Failed to concat MP3 chunks: {e}", _EXIT_OUTPUT)
    main_duration_ms = mp3_duration_ms(ffprobe_path, paths.full_mp3)
    run_manifest = build_run_manifest(chunks_manifest, paths, main_duration_ms)
    try:
        write_json(paths.run_json, run_manifest)
    except Exception as e:
        fail(f"Failed to write {paths.run_json}: {e}", _EXIT_OUTPUT)

    timing_info = None
    if getattr(args, "with_timings", False):
        try:
            timing_info = _extract_timings(
                audio_path=paths.full_mp3, output_dir=paths.output_root, prefix=paths.prefix,
                model=args.timing_model, device=args.timing_device,
                compute_type=args.timing_compute, language=args.timing_language,
                word_timestamps=args.word_timestamps, quiet=args.json_output,
            )
        except ModuleNotFoundError as exc:
            fail(f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --group timing-whisper", _EXIT_MISSING_DEP)
        except Exception as exc:
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
    if args.json_output:
        _json_ok({
            "status": "success", "provider": args.provider, "model": args.model,
            "run_id": paths.prefix, "files": files, "duration_ms": main_duration_ms,
            "segment_count": timing_info["segment_count"] if timing_info else None,
            "cost": {"total": cost_total, "currency": cost_currency},
        })
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
        _json_ok({"status": "success", "chunks": [{"id": c.id, "chars": len(c.text)} for c in chunks]})
    else:
        for chunk in chunks:
            print(f"{chunk.id}: {len(chunk.text)} chars")


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
                _json_ok({"status": "skipped", "reason": "output dir exists", "run_id": run_id, "files": files})
            else:
                print(f"Skipping: output dir exists: {output_dir}")
            return
        if not args.overwrite:
            fail(f"Output dir exists: {output_dir}. Use --overwrite or --skip-existing.", _EXIT_PROVIDER)
        _safe_remove_run_dir(output_dir, args.output_dir)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"Failed to create output directory {output_dir}: {e}", _EXIT_OUTPUT)

    try:
        timing = _extract_timings(
            audio_path=audio_path, output_dir=output_dir, prefix=run_id,
            model=args.model, device=args.device, compute_type=args.compute,
            language=args.language, word_timestamps=args.word_timestamps,
            quiet=args.json_output,
        )
    except ModuleNotFoundError as exc:
        fail(f"Missing dependency for Whisper timing: {exc}. Install with: uv sync --group timing-whisper", _EXIT_MISSING_DEP)
    except Exception as exc:
        fail(f"Whisper timing failed: {exc}", _EXIT_WHISPER)

    files = {"timings_json": str(output_dir / f"{run_id}.timings.json"), "srt": str(output_dir / f"{run_id}.srt")}
    if args.json_output:
        _json_ok({"status": "success", "files": files, "segment_count": timing["segment_count"],
                   "duration_ms": timing["total_duration_ms"]})
    else:
        print(f"Timings JSON: {files['timings_json']}")
        print(f"SRT: {files['srt']}")
        print(f"Segments: {timing['segment_count']}")


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

    need_whisper = bool(args.with_timings)
    whisper_ok = False
    try:
        import faster_whisper
        whisper_ok = True
    except ImportError:
        pass
    results["faster_whisper"] = {"ok": whisper_ok, "required": need_whisper}

    need_cuda = (args.provider == "qwen-local") or (args.timing_device == "cuda")
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    results["cuda"] = {"ok": cuda_available, "required": need_cuda}

    required_ok = all(info.get("ok", False) for info in results.values() if info.get("required"))
    optional_ok = all(info.get("ok", False) for info in results.values() if not info.get("required"))
    workflow_ok = required_ok

    warnings: list[str] = []
    if not cuda_available and not need_cuda:
        warnings.append("CUDA is unavailable: qwen-local and cuda timings will not work, but cloud TTS and CPU timings are OK.")
    if not cuda_available and need_cuda:
        warnings.append("CUDA is unavailable but required for the selected provider or timing device.")
    if not whisper_ok and need_whisper:
        warnings.append("faster-whisper is not installed. Install with: uv sync --group timing-whisper")
    if not polza_ok and need_polza:
        warnings.append("POLZA_API_KEY is missing. Set it in .env: POLZA_API_KEY=...")
    if not or_ok and need_or:
        warnings.append("OPENROUTER_API_KEY is missing. Set it in .env: OPENROUTER_API_KEY=...")

    if args.json_output:
        _json_ok({
            "status": "success",
            "required_ok": required_ok,
            "optional_ok": optional_ok,
            "workflow_ok": workflow_ok,
            "checks": results,
            "warnings": warnings,
        })
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

    text = script.read_text(encoding="utf-8-sig")
    parts = [p.strip() for p in text.split(args.delimiter)]
    chunk_list = [(i, p) for i, p in enumerate(parts, start=1) if p]

    issues: list[dict] = []
    total_chars = 0
    for idx, chunk_text in chunk_list:
        chars = len(chunk_text)
        total_chars += chars
        if chars > args.max_chunk_chars:
            issues.append({"chunk": idx, "type": "too_long", "chars": chars, "limit": args.max_chunk_chars})

    warnings = []
    for idx, chunk_text in chunk_list:
        has_digits = any(ch.isdigit() for ch in chunk_text)
        if has_digits:
            warnings.append({"chunk": idx, "type": "contains_digits"})

    ok = len(issues) == 0

    if args.json_output:
        _json_ok({"status": "success" if ok else "warning", "valid": ok, "chunks": len(chunk_list),
                   "total_chars": total_chars, "issues": issues, "warnings": warnings})
    else:
        print(f"Script: {script}")
        print(f"Chunks: {len(chunk_list)}, Total chars: {total_chars}")
        print(f"Valid: {ok}")
        for issue in issues:
            print(f"  ISSUE chunk {issue['chunk']}: {issue['type']} ({issue['chars']} chars > {issue['limit']})")


def list_cmd(args: argparse.Namespace) -> None:
    if args.target == "providers":
        data = {
            "providers": [
                {"id": "polza-chat-audio", "models": ["openai/gpt-audio-mini", "openai/gpt-audio"], "currency": "RUB"},
                {"id": "polza-tts", "models": POLZA_TTS_MODELS, "currency": "RUB"},
                {"id": "openrouter-tts", "models": OPENROUTER_TTS_MODELS, "currency": "USD"},
                {"id": "qwen-local", "modes": ["preset", "clone"], "currency": "RUB", "cost": "free"},
            ]
        }
    elif args.target == "voices":
        provider = args.provider or "polza-chat-audio"
        voices_flat = {
            "polza-chat-audio": ["ash", "ballad", "coral", "verse", "marin", "cedar", "echo", "sage", "shimmer", "onyx"],
            "polza-tts": OPENAI_TTS_VOICES + ELEVENLABS_TTS_VOICES,
            "openrouter-tts": GEMINI_TTS_VOICES + OPENAI_TTS_VOICES,
            "qwen-local": QWEN_PRESET_SPEAKERS,
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
        if provider in voice_categories:
            data["voice_categories"] = voice_categories[provider]
    elif args.target == "timing-models":
        data = {
            "timing_models": [
                {"id": "base", "parameters_m": 74, "disk_mb": 148, "speed": "fastest"},
                {"id": "small", "parameters_m": 244, "disk_mb": 486, "speed": "fast", "default": True},
                {"id": "medium", "parameters_m": 769, "disk_mb": 1536, "speed": "balanced"},
                {"id": "large-v3-turbo", "parameters_m": 809, "disk_mb": 1620, "speed": "slow"},
                {"id": "large-v3", "parameters_m": 1550, "disk_mb": 3090, "speed": "slowest"},
            ]
        }
    else:
        data = {}
    if args.json_output:
        _json_ok({"status": "success", **data})
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_audio(raw: str) -> Path:
    if "*" in raw or "?" in raw:
        matches = sorted(glob_mod.glob(raw))
        if not matches:
            fail(f"No files match: {raw}", _EXIT_ARGS)
        if len(matches) > 1:
            fail(f"Multiple files match: {raw}. Provide exact path.", _EXIT_ARGS)
        return Path(matches[0])
    return Path(raw)


def _extract_timings(audio_path, output_dir, prefix, model, device, compute_type, language, word_timestamps=False, quiet=False):
    from .whisper_timing import transcribe_audio

    timing = transcribe_audio(
        audio_path=audio_path, model_size=model, device=device,
        compute_type=compute_type, language=language,
        word_timestamps=word_timestamps, quiet=quiet,
    )

    ffprobe_path = shutil.which("ffprobe")
    duration_ms = mp3_duration_ms(ffprobe_path, audio_path) if ffprobe_path else sum(
        seg.duration_ms for seg in timing.segments
    )

    timing_json = output_dir / f"{prefix}.timings.json"
    try:
        write_json(timing_json, build_timing_manifest(timing, duration_ms))
    except Exception as e:
        fail(f"Failed to write {timing_json}: {e}", _EXIT_OUTPUT)

    srt_path = output_dir / f"{prefix}.srt"
    try:
        srt_path.write_text(build_srt(timing), encoding="utf-8")
    except OSError as e:
        fail(f"Failed to write {srt_path}: {e}", _EXIT_OUTPUT)

    if not quiet:
        print(f"Timings JSON: {timing_json}")
        print(f"SRT: {srt_path}")

    return {"segment_count": len(timing.segments), "total_duration_ms": duration_ms}


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
}


def _resolve_model(args: argparse.Namespace) -> None:
    if not hasattr(args, "model") or args.model is None:
        args.model = PROVIDER_DEFAULT_MODELS.get(args.provider, DEFAULT_MODEL)


def _validate_model_for_provider(provider: str, model: str) -> None:
    valid = _VALID_MODELS_BY_PROVIDER.get(provider, [])
    if not valid:
        return
    if model not in valid:
        fail(
            f"Model '{model}' is not valid for provider '{provider}'. "
            f"Valid models: {valid}",
            _EXIT_ARGS,
        )


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


def _default_voice(args: argparse.Namespace) -> str:
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
    if args.provider == "qwen-local":
        return ""
    raise RuntimeError(f"Unsupported provider: {args.provider}")


def build_provider(args: argparse.Namespace, api_key: str, style_prompt: str | None, prompt_mode: str) -> TTSProvider:
    if args.provider == "polza-chat-audio":
        return PolzaChatAudioProvider(api_key=api_key, model=args.model, voice=args.voice, fallback_voice=args.fallback_voice)
    if args.provider == "polza-tts":
        return PolzaTTSProvider(api_key=api_key, model=args.model, voice=args.voice)
    if args.provider == "openrouter-tts":
        return OpenRouterTTSProvider(api_key=api_key, model=args.model, voice=args.voice, style_prompt=style_prompt, prompt_mode=prompt_mode)
    if args.provider == "qwen-local":
        return QwenLocalTTSProvider(mode=args.mode, voice=args.voice, sample_path=args.sample, sample_text=args.sample_text)
    raise RuntimeError(f"Unsupported provider: {args.provider}")


def fetch_pricing_snapshot(provider: str, api_key: str, model: str) -> dict | None:
    if provider in ("polza-chat-audio", "polza-tts"):
        return fetch_polza_model_pricing(api_key, model)
    if provider == "openrouter-tts":
        return fetch_openrouter_model_pricing(model)
    return None


def attach_costs(provider, api_key, model, run_started_at, chunks):
    if provider == "qwen-local":
        enriched = []
        for chunk in chunks:
            enriched.append(ChunkArtifact(**{**chunk.__dict__, "cost": 0.0, "cost_exact": "0.0", "cost_currency": "RUB"}))
        return enriched
    if provider in ("polza-chat-audio", "polza-tts"):
        generations = fetch_polza_generation_costs(api_key, model, run_started_at, len(chunks))
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
        enriched.append(ChunkArtifact(**{
            **chunk.__dict__,
            "generation_id": generation.get("id") or chunk.generation_id if generation else chunk.generation_id,
            "cost_rub": cost if currency == "RUB" else None,
            "cost_rub_exact": cost_exact if currency == "RUB" else None,
            "cost": cost, "cost_exact": cost_exact, "cost_currency": currency,
            "usage": generation.get("usage") if generation else None,
            "generation_time_ms": generation.get("generationTimeMs") or generation.get("generation_time") if generation else None,
            "generated_at": generation.get("createdAt") or generation.get("created_at") if generation else None,
            "generation_detail_source": generation_source(provider) if generation else None,
        }))
    return enriched


def summarize_costs(provider: str, chunks: list[ChunkArtifact]) -> tuple:
    if not chunks or any(chunk.cost is None for chunk in chunks):
        return None, None, None, None
    total = sum(float(chunk.cost or 0) for chunk in chunks)
    currency = chunks[0].cost_currency
    source = generation_source(provider) if provider != "qwen-local" else "qwen-local (free)"
    return round(total, 8), str(round(total, 8)), currency, source


def generation_source(provider: str) -> str:
    return {
        "polza-chat-audio": "Polza GET /api/v1/history/generations/{id}",
        "polza-tts": "Polza API usage.cost_rub or GET /api/v1/history/generations/{id}",
        "openrouter-tts": "OpenRouter GET /api/v1/generation?id=...",
        "qwen-local": "qwen-local (free)",
    }.get(provider, "unknown")


if __name__ == "__main__":
    main()
