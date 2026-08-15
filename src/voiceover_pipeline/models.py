from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping


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
    provider: str | None = None
    device: str = ""
    compute_type: str = ""
    language: str = ""
    source_audio: str = ""


ASRAlignmentOrigin = Literal["native", "forced"]
ASRPhraseStrength = Literal["mild", "normal", "strong"]


@dataclass(frozen=True)
class ASRPhraseHint:
    text: str
    strength: ASRPhraseStrength = "normal"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("ASR phrase hints must not be blank")
        if self.strength not in ("mild", "normal", "strong"):
            raise ValueError("ASR phrase hint strength must be mild, normal, or strong")


@dataclass(frozen=True)
class ASRGlossaryHint:
    profile_id: str
    digest: str
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("ASR glossary profile_id must not be blank")
        if not self.digest.strip():
            raise ValueError("ASR glossary digest must not be blank")
        object.__setattr__(self, "terms", tuple(self.terms))


@dataclass(frozen=True)
class ASRContextHints:
    context_text: str | None = None
    glossary: ASRGlossaryHint | None = None
    phrase_hints: tuple[ASRPhraseHint, ...] = ()
    initial_prompt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phrase_hints", tuple(self.phrase_hints))


@dataclass(frozen=True)
class ASRRequest:
    audio_path: Path | str
    model_id: str | None = None
    language: str | None = None
    device: str = "cpu"
    compute: str = "auto"
    hints: ASRContextHints = field(default_factory=ASRContextHints)

    def __post_init__(self) -> None:
        if not str(self.audio_path):
            raise ValueError("ASR audio_path must not be blank")
        if not self.device.strip():
            raise ValueError("ASR device must not be blank")
        if not self.compute.strip():
            raise ValueError("ASR compute must not be blank")


@dataclass(frozen=True)
class ASRExecutionReceipt:
    runtime: str
    runtime_version: str | None = None
    model_revision: str | None = None
    resolved_device: str = "cpu"
    resolved_compute: str = "auto"
    measurements: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.runtime.strip():
            raise ValueError("ASR runtime must not be blank")
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))


@dataclass(frozen=True)
class ASRSegment:
    text: str
    start_s: float | None = None
    end_s: float | None = None

    def __post_init__(self) -> None:
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("ASR segment timestamps must include both start and end")
        if self.start_s is not None and self.start_s < 0:
            raise ValueError("ASR timestamps must be non-negative")
        if self.end_s is not None and self.start_s is not None and self.end_s < self.start_s:
            raise ValueError("ASR timestamp end must not be before start")


@dataclass(frozen=True)
class ASRWordSpan:
    text: str
    start_s: float
    end_s: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("ASR timestamps must be non-negative")
        if self.end_s < self.start_s:
            raise ValueError("ASR timestamp end must not be before start")


@dataclass(frozen=True)
class ASRCapabilities:
    batch_audio: bool = True
    streaming: bool = False
    forced_language: bool = False
    contextual_bias: bool = False
    phrase_boosting: bool = False
    segment_timestamps: bool = False
    word_timestamps: bool = False
    forced_alignment: bool = False
    confidence: bool = False
    device_modes: tuple[str, ...] = ("cpu",)
    compute_modes: tuple[str, ...] = ("auto",)


def _validate_monotonic_spans(spans: tuple[ASRSegment | ASRWordSpan, ...]) -> None:
    previous_start: float | None = None
    for span in spans:
        start = span.start_s
        if start is None:
            continue
        if previous_start is not None and start < previous_start:
            raise ValueError("ASR timestamps must be monotonic")
        previous_start = start


@dataclass(frozen=True)
class ASRResult:
    transcript: str
    provider_id: str
    model_id: str
    execution: ASRExecutionReceipt
    language: str = ""
    duration_s: float | None = None
    segments: tuple[ASRSegment, ...] = ()
    words: tuple[ASRWordSpan, ...] = ()
    alignment_origin: ASRAlignmentOrigin | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "words", tuple(self.words))
        if self.alignment_origin not in (None, "native", "forced"):
            raise ValueError("ASR alignment origin must be native or forced")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("ASR duration must be non-negative")
        _validate_monotonic_spans(self.segments)
        _validate_monotonic_spans(self.words)
        has_timestamps = any(segment.start_s is not None for segment in self.segments) or bool(self.words)
        if has_timestamps and self.alignment_origin is None:
            raise ValueError("ASR timestamps require an alignment origin")
