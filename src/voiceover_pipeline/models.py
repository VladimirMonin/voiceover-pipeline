from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScriptChunk:
    number: int
    id: str
    text: str


@dataclass(frozen=True)
class SynthesisResult:
    audio_bytes: bytes
    audio_format: str
    transcript: str | None = None
    generation_id: str | None = None
    client_path: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkArtifact:
    number: int
    id: str
    file: str
    duration_ms: int
    duration_sec: float
    start_ms: int
    end_ms: int
    text_characters: int
    transcript: str | None
    client_path: str | None
    generation_id: str | None
    cost_rub: float | None = None
    cost_rub_exact: str | None = None
    cost: float | None = None
    cost_exact: str | None = None
    cost_currency: str | None = None
    usage: dict[str, Any] | None = None
    generation_time_ms: int | None = None
    generated_at: str | None = None
    generation_detail_source: str | None = None


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    chunks_dir: Path
    full_mp3: Path
    chunks_json: Path
    run_json: Path
    prefix: str = ""


@dataclass(frozen=True)
class TimingSegment:
    id: int
    start_sec: float
    end_sec: float
    start_ms: int
    end_ms: int
    duration_ms: int
    text: str
    words: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class TimingResult:
    segments: list[TimingSegment]
    model: str
    backend: str
    device: str
    compute_type: str
    language: str
    source_audio: str
