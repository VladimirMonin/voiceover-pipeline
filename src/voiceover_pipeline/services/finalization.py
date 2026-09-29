"""Post-loop finalization for the CLI generation step.

This module owns everything ``cli._generate_step`` does after the prepared-part
execution loop returns and before it reports success: it reconciles the
late-observed run cost into trusted state, builds and writes the chunks, run,
and manifest artifacts, runs the dialogue quality gate before the final concat,
concatenates the chunk audio, optionally extracts timings, and writes the
completed state. The provider history GET for the observed cost happens exactly
once here, after every chunk is already saved, and never submits.

It calls the CLI's call-time compatibility and presentation seams through an
explicit ``FinalizationHooks`` object rather than importing ``cli``: tests patch
``cli.attach_costs``, ``cli._verify_dialogue_turns_before_concat``,
``cli.concat_mp3_chunks``, ``cli.concat_dialogue_turns``, and
``cli.mp3_duration_ms``, and the provider/output/missing-dependency/whisper fail
envelopes plus the timing writer stay CLI-owned. The cost merge and Decimal
summary are owned by ``services.cost_enrichment``; a mismatch raises its typed
``AttachedCostStateMismatchError`` and is translated to the CLI provider exit
without writing a partial state.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from ..artifacts import (
    build_chunks_manifest,
    build_manifest_json,
    build_run_manifest,
    write_json,
)
from ..models import ChunkArtifact
from ..run_state import GenerationLogger, append_error, atomic_write_json
from . import cost_enrichment


@dataclass(frozen=True)
class FinalizationSummary:
    """The values the CLI turns into its success JSON or human output.

    Only the artifact file path map, full duration, optional timing segment
    count, and run cost total/currency cross back to the CLI presentation layer.
    """

    files: dict[str, str]
    duration_ms: int
    segment_count: int | None
    cost_total: float | None
    cost_currency: str | None


@dataclass
class FinalizationHooks:
    """The CLI's call-time compatibility and presentation callables.

    These stay the functions bound in ``cli`` at call time: tests patch the cost
    attach, dialogue quality, concat, and duration seams, and the timing writer
    plus the provider/output/missing-dependency/whisper ``fail`` envelopes are
    CLI-owned presentation rather than finalization logic.
    """

    attach_costs: Callable[..., list[ChunkArtifact]]
    verify_dialogue_turns_before_concat: Callable[..., dict[str, Any] | None]
    concat_dialogue_turns: Callable[..., None]
    concat_mp3_chunks: Callable[..., None]
    mp3_duration_ms: Callable[[str, Path], int]
    extract_timings: Callable[..., Any]
    fail_provider: Callable[[str], NoReturn]
    fail_output: Callable[[str], NoReturn]
    fail_missing_dep: Callable[[str], NoReturn]
    fail_whisper: Callable[[str], NoReturn]


def list_artifact_files(paths) -> dict[str, str]:
    """Return the run's artifact file map, adding timings files when present."""
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


def finalize_generation(
    *,
    args,
    paths,
    state: dict[str, Any],
    state_path: Path,
    logger: GenerationLogger,
    chunks,
    chunk_artifacts_by_number: dict[int, ChunkArtifact],
    prepared,
    pricing_snapshot,
    api_key,
    run_started_at,
    ffmpeg_path: str,
    ffprobe_path: str,
    dialogue_run: bool,
    hooks: FinalizationHooks,
) -> FinalizationSummary:
    """Reconcile costs, write artifacts, concat, and mark the run completed.

    The order is the behavior contract: the trusted cost state write precedes the
    dialogue quality gate, that gate precedes the final concat, and the completed
    state write precedes the ``run_complete`` log. Every failure keeps its own
    envelope, event, and exit code.
    """
    chunk_artifacts = [
        chunk_artifacts_by_number[number] for number in sorted(chunk_artifacts_by_number)
    ]

    time.sleep(2)
    chunk_artifacts = hooks.attach_costs(
        args.provider, api_key, args.model, run_started_at, chunk_artifacts
    )
    try:
        cost_enrichment.merge_attached_costs_into_state(state, chunk_artifacts)
    except cost_enrichment.AttachedCostStateMismatchError as exc:
        hooks.fail_provider(str(exc))
    atomic_write_json(state_path, state)
    cost_total, cost_total_exact, cost_currency, cost_source = cost_enrichment.summarize_costs(
        args.provider, chunk_artifacts
    )

    tts_quality_receipt = (
        hooks.verify_dialogue_turns_before_concat(args, chunks, chunk_artifacts, paths)
        if dialogue_run
        else None
    )

    chunks_manifest = build_chunks_manifest(
        provider=prepared.provider,
        model=prepared.model,
        voice=prepared.voice,
        style_prompt=prepared.style_prompt,
        script=args.script,
        chunks_dir=paths.chunks_dir,
        pricing_snapshot=pricing_snapshot,
        cost_exact_available=cost_total_exact is not None,
        cost_total=cost_total,
        cost_total_exact=cost_total_exact,
        cost_currency=cost_currency,
        cost_source=cost_source,
        chunk_artifacts=chunk_artifacts,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        prompt_mode=prepared.prompt_mode,
        script_format=getattr(args, "format", "markdown"),
        speaker_voice_map=getattr(args, "speaker_voice_map", None) or None,
        execution_source=state.get("execution_source"),
        tts_quality_receipt=tts_quality_receipt,
    )
    try:
        write_json(paths.chunks_json, chunks_manifest)
    except Exception as e:
        hooks.fail_output(f"Failed to write {paths.chunks_json}: {e}")

    try:
        logger.event("info", "concat_started", output=paths.full_mp3.name)
        if dialogue_run:
            hooks.concat_dialogue_turns(
                ffmpeg_path,
                [
                    (paths.chunks_dir / artifact.file, artifact.pause_after_ms)
                    for artifact in chunk_artifacts
                ],
                paths.full_mp3,
            )
        else:
            hooks.concat_mp3_chunks(ffmpeg_path, paths.chunks_dir, paths.full_mp3)
    except Exception as e:
        append_error(state, chunk_id=None, message=str(e))
        atomic_write_json(state_path, state)
        logger.event("error", "concat_failed", error=str(e))
        hooks.fail_output(f"Failed to concat MP3 chunks: {e}")
    main_duration_ms = hooks.mp3_duration_ms(ffprobe_path, paths.full_mp3)
    logger.event(
        "info", "concat_complete", output=paths.full_mp3.name, duration_ms=main_duration_ms
    )
    run_manifest = build_run_manifest(chunks_manifest, paths, main_duration_ms)
    try:
        write_json(paths.run_json, run_manifest)
    except Exception as e:
        hooks.fail_output(f"Failed to write {paths.run_json}: {e}")

    timing_info: Any = None
    if getattr(args, "with_timings", False):
        try:
            logger.event("info", "timings_started", audio=paths.full_mp3.name)
            timing_info = hooks.extract_timings(
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
            hooks.fail_missing_dep(
                f"Missing dependency for Whisper timing: {exc}. "
                "Install with: uv sync --extra timing-whisper"
            )
        except Exception as exc:
            logger.event("error", "timings_failed", error=str(exc))
            hooks.fail_whisper(f"Voiceover generated but timing extraction failed: {exc}")

    manifest_json = build_manifest_json(paths, main_duration_ms)
    try:
        write_json(paths.output_root / "manifest.json", manifest_json)
    except Exception as e:
        hooks.fail_output(f"Failed to write manifest.json: {e}")

    files = list_artifact_files(paths)
    state["status"] = "completed"
    state["full_mp3"] = str(paths.full_mp3)
    state["main_duration_ms"] = main_duration_ms
    atomic_write_json(state_path, state)
    logger.event("info", "run_complete", run_id=paths.prefix, duration_ms=main_duration_ms)

    return FinalizationSummary(
        files=files,
        duration_ms=main_duration_ms,
        segment_count=timing_info["segment_count"] if timing_info else None,
        cost_total=cost_total,
        cost_currency=cost_currency,
    )
