import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import model_slug
from .models import ChunkArtifact, RunPaths, TimingResult


def build_run_paths(output_dir: Path, model: str, run_id: str | None = None) -> RunPaths:
    root = (output_dir / (run_id or model_slug(model))).resolve()
    chunks_dir = root / "chunks"

    slug = model_slug(model)
    prefix = run_id or slug
    return RunPaths(
        output_root=root,
        chunks_dir=chunks_dir,
        full_mp3=root / f"{prefix}-voiceover-{slug}.mp3",
        chunks_json=chunks_dir / "chunks.json",
        run_json=root / f"{prefix}-voiceover-{slug}.json",
        prefix=prefix,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to write {path}: {e}") from e


def chunk_artifacts_to_dicts(chunks: list[ChunkArtifact]) -> list[dict[str, Any]]:
    return [drop_none(asdict(chunk)) for chunk in chunks]


def drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def build_chunks_manifest(
    *,
    provider: str,
    model: str,
    voice: str,
    style_prompt: str | None,
    script: Path,
    chunks_dir: Path,
    pricing_snapshot: dict[str, Any] | None,
    cost_exact_available: bool,
    cost_total: float | None,
    cost_total_exact: str | None,
    cost_currency: str | None,
    cost_source: str | None,
    chunk_artifacts: list[ChunkArtifact],
    ffmpeg_path: str,
    ffprobe_path: str,
    prompt_mode: str = "auto",
    script_format: str = "markdown",
    speaker_voice_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    total_duration_ms = chunk_artifacts[-1].end_ms if chunk_artifacts else 0
    return drop_none(
        {
            "artifact_type": "voiceover-chunks",
            "provider": provider,
            "model": model,
            "voice": voice,
            "style_prompt": style_prompt,
            "prompt_mode": prompt_mode,
            "script_format": script_format,
            "speaker_voice_map": speaker_voice_map,
            "format": "mp3",
            "script": str(script.resolve()),
            "chunks_dir": str(chunks_dir),
            "chunk_count": len(chunk_artifacts),
            "total_duration_ms": total_duration_ms,
            "total_duration_sec": round(total_duration_ms / 1000, 3),
            "cost_exact_available": cost_exact_available,
            "cost_total": cost_total,
            "cost_total_exact": cost_total_exact,
            "cost_currency": cost_currency,
            "cost_source": cost_source,
            "cost_per_minute": cost_per_minute(cost_total, total_duration_ms),
            "pricing_snapshot": pricing_snapshot,
            "duration_source": "ffprobe_after_ffmpeg_final_silence_trim",
            "ffmpeg": ffmpeg_path,
            "ffprobe": ffprobe_path,
            "chunks": chunk_artifacts_to_dicts(chunk_artifacts),
        }
    )


def build_run_manifest(chunks_manifest: dict[str, Any], paths: RunPaths, main_duration_ms: int) -> dict[str, Any]:
    cost_total = chunks_manifest.get("cost_total")
    return drop_none(
        {
            **chunks_manifest,
            "artifact_type": "voiceover-run",
            "output_dir": str(paths.output_root),
            "chunks_dir": str(paths.chunks_dir),
            "main_file": paths.full_mp3.name,
            "main_duration_ms": main_duration_ms,
            "main_duration_sec": round(main_duration_ms / 1000, 3),
            "cost_per_minute": cost_per_minute(cost_total, main_duration_ms),
            "concat_method": "ffmpeg concat demuxer with libmp3lame re-encode at 128k",
        }
    )


def cost_per_minute(cost_total: float | None, duration_ms: int) -> float | None:
    if cost_total is None or duration_ms <= 0:
        return None
    return round(float(cost_total) / (duration_ms / 60000), 8)


def build_timing_manifest(timing: TimingResult, duration_ms: int) -> dict[str, Any]:
    return drop_none(
        {
            "artifact_type": "voiceover-timings",
            "source_audio": timing.source_audio,
            "model": timing.model,
            "backend": timing.backend,
            "device": timing.device,
            "compute_type": timing.compute_type,
            "language": timing.language,
            "total_duration_ms": duration_ms,
            "total_duration_sec": round(duration_ms / 1000, 3),
            "segment_count": len(timing.segments),
            "segments": [seg_to_dict(seg) for seg in timing.segments],
        }
    )


def seg_to_dict(seg: Any) -> dict[str, Any]:
    return drop_none(asdict(seg))


def build_srt(timing: TimingResult) -> str:
    lines: list[str] = []
    for i, seg in enumerate(timing.segments, start=1):
        start_ts = _ms_to_srt(seg.start_ms)
        end_ts = _ms_to_srt(seg.end_ms)
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _ms_to_srt(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    r = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{r:03d}"


def build_manifest_json(paths: Any, duration_ms: int) -> dict[str, Any]:
    data: dict[str, Any] = {
        "artifact_type": "voiceover-production-bundle",
        "run_id": paths.prefix,
        "full_mp3": str(paths.full_mp3),
        "run_json": str(paths.run_json),
        "chunks_json": str(paths.chunks_json),
        "duration_ms": duration_ms,
    }
    timings_json = paths.output_root / f"{paths.prefix}.timings.json"
    srt_path = paths.output_root / f"{paths.prefix}.srt"
    if timings_json.exists():
        data["timings_json"] = str(timings_json)
    if srt_path.exists():
        data["srt"] = str(srt_path)
    return data
