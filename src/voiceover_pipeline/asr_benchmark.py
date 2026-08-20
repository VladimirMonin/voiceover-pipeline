"""Offline, reproducible evaluation for local ASR adapters.

The harness executes adapters against an explicit corpus manifest. It records
aggregate metrics and resource observations but intentionally never writes
reference text, transcripts, or prompt contents to its reports.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Protocol

from voiceover_pipeline.artifacts import write_json
from voiceover_pipeline.models import (
    ASRContextHints,
    ASRExecutionReceipt,
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
)
from voiceover_pipeline.providers.asr_registry import get_asr_provider_spec
from voiceover_pipeline.providers.faster_whisper import FasterWhisperProvider

try:
    resource: Any = importlib.import_module("resource")
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None


REPORT_SCHEMA_VERSION = 2
_REPORT_JSON_NAME = "asr-benchmark.json"
_REPORT_MARKDOWN_NAME = "asr-benchmark.md"


class BenchmarkAdapter(Protocol):
    provider_id: str
    prompt_supported: bool
    timestamp_supported: bool

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        prompt: str | None,
    ) -> ASRResult: ...


@dataclass(frozen=True)
class ResourceSnapshot:
    """Best-effort resource measurements made while one request runs."""

    peak_ram_bytes: int | None
    peak_vram_bytes: int | None


class BenchmarkResourceMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> ResourceSnapshot: ...


class ResourceMonitor:
    """Sample process RAM and NVIDIA memory without making benchmark calls fail."""

    def __init__(self, sample_interval_s: float = 0.1) -> None:
        self._sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_ram_bytes: int | None = None
        self._peak_vram_bytes: int | None = None

    def start(self) -> None:
        self._sample_once()
        if shutil.which("nvidia-smi"):
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop(self) -> ResourceSnapshot:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._sample_interval_s * 2))
        self._sample_once()
        return ResourceSnapshot(
            peak_ram_bytes=self._peak_ram_bytes,
            peak_vram_bytes=self._peak_vram_bytes,
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        self._record_ram()
        self._record_vram()

    def _record_ram(self) -> None:
        if resource is None:
            return
        try:
            raw_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (AttributeError, OSError):
            return
        peak = int(raw_value if sys.platform == "darwin" else raw_value * 1024)
        self._peak_ram_bytes = max(self._peak_ram_bytes or 0, peak)

    def _record_vram(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            values_mib = [
                int(line.strip())
                for line in completed.stdout.splitlines()
                if line.strip().isdigit()
            ]
        except (OSError, subprocess.SubprocessError):
            return
        if values_mib:
            peak = max(values_mib) * 1024 * 1024
            self._peak_vram_bytes = max(self._peak_vram_bytes or 0, peak)


class RegistryBenchmarkAdapter:
    """Adapter for text ASR providers registered in the local ASR registry."""

    def __init__(
        self,
        provider_id: str,
        *,
        model_id: str | None,
        device: str,
        compute: str,
    ) -> None:
        self._spec = get_asr_provider_spec(provider_id)
        self.provider_id = self._spec.provider_id
        self.prompt_supported = self._spec.capabilities.contextual_bias
        self.timestamp_supported = (
            self._spec.capabilities.segment_timestamps or self._spec.capabilities.word_timestamps
        )
        self._model_id = model_id
        self._device = device
        self._compute = compute
        self._provider = self._spec.factory()

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        prompt: str | None,
    ) -> ASRResult:
        hints = ASRContextHints(context_text=prompt) if prompt is not None else ASRContextHints()
        request = ASRRequest(
            audio_path=audio_path,
            model_id=self._model_id,
            language=language,
            device=self._device,
            compute=self._compute,
            hints=hints,
            timestamp_mode="word" if self._spec.capabilities.word_timestamps else "none",
        )
        return self._provider.transcribe(request)


class FasterWhisperBenchmarkAdapter:
    """Map the existing faster-whisper timing provider into the ASR benchmark contract."""

    provider_id = "faster-whisper"
    prompt_supported = False
    timestamp_supported = True

    def __init__(self, *, model_id: str | None, device: str, compute: str) -> None:
        self._model_id = model_id or "small"
        self._device = device
        self._compute = compute

    def transcribe(
        self,
        *,
        audio_path: Path,
        language: str,
        prompt: str | None,
    ) -> ASRResult:
        if prompt is not None:
            raise ValueError("faster-whisper benchmark adapter does not declare prompt support")
        timing = FasterWhisperProvider(
            model_size=self._model_id,
            device=self._device,
            compute_type=self._compute,
        ).transcribe(
            audio_path=audio_path,
            language=language,
            word_timestamps=True,
            quiet=True,
        )
        segments = tuple(
            ASRSegment(
                text=segment.text,
                start_s=segment.start_sec,
                end_s=segment.end_sec,
            )
            for segment in timing.segments
        )
        return ASRResult(
            transcript=" ".join(segment.text for segment in segments).strip(),
            provider_id=self.provider_id,
            model_id=timing.model,
            language=timing.language,
            execution=ASRExecutionReceipt(
                runtime=timing.backend,
                resolved_device=timing.device,
                resolved_compute=timing.compute_type,
            ),
            segments=segments,
            alignment_origin="native" if segments else None,
        )


def create_benchmark_adapter(
    provider_id: str,
    *,
    model_id: str | None,
    device: str,
    compute: str,
) -> BenchmarkAdapter:
    """Create a deferred-load adapter; this never downloads or loads a model."""

    if provider_id == "whisper":
        return FasterWhisperBenchmarkAdapter(
            model_id=model_id,
            device=device,
            compute=compute,
        )
    if provider_id in {"qwen-local", "nemotron-local"}:
        return RegistryBenchmarkAdapter(
            provider_id,
            model_id=model_id,
            device=device,
            compute=compute,
        )
    raise ValueError(
        "Unsupported benchmark provider "
        f"{provider_id!r}; expected whisper, qwen-local, or nemotron-local"
    )


def normalize_transcript(text: str) -> str:
    """Normalize Unicode text without language-specific external dependencies."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return " ".join(tokens)


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, reference_item in enumerate(reference, start=1):
        current = [index]
        for candidate_index, hypothesis_item in enumerate(hypothesis, start=1):
            substitution = previous[candidate_index - 1] + (reference_item != hypothesis_item)
            insertion = current[candidate_index - 1] + 1
            deletion = previous[candidate_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _error_counts(reference_text: str, transcript: str) -> dict[str, int]:
    normalized_reference = normalize_transcript(reference_text)
    normalized_transcript = normalize_transcript(transcript)
    reference_words = normalized_reference.split()
    transcript_words = normalized_transcript.split()
    reference_characters = list(normalized_reference.replace(" ", ""))
    transcript_characters = list(normalized_transcript.replace(" ", ""))
    return {
        "word_errors": _levenshtein(reference_words, transcript_words),
        "word_count": len(reference_words),
        "character_errors": _levenshtein(reference_characters, transcript_characters),
        "character_count": len(reference_characters),
    }


def _rate(errors: int, count: int) -> float | None:
    if count == 0:
        return 0.0 if errors == 0 else None
    return round(errors / count, 9)


def _relative_audio_path(
    manifest_path: Path,
    raw_path: str,
    *,
    corpus_root: Path | None = None,
) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"Benchmark manifest audio path must be relative: {raw_path}")
    root = corpus_root.resolve() if corpus_root is not None else manifest_path.parent.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark corpus root not found: {root}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Benchmark manifest audio path escapes its corpus: {raw_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Benchmark audio file not found: {candidate}")
    return candidate


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load benchmark manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("corpus_id"), str):
        raise ValueError("Benchmark manifest must declare schema_version=1 and corpus_id")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark manifest must contain a non-empty cases list")
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Benchmark manifest case must be an object")
        required = {"case_id", "audio_path", "expected_text", "duration_s", "language", "sha256"}
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Benchmark manifest case is missing: {', '.join(missing)}")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in identifiers:
            raise ValueError("Benchmark case IDs must be unique non-empty strings")
        identifiers.add(case_id)
        if not isinstance(case["expected_text"], str) or not case["expected_text"].strip():
            raise ValueError(f"Benchmark case {case_id} has an empty expected_text")
        if not isinstance(case["duration_s"], (int, float)) or case["duration_s"] <= 0:
            raise ValueError(f"Benchmark case {case_id} has an invalid duration_s")
        expected_words = case.get("expected_word_timestamps")
        if expected_words is not None:
            if not isinstance(expected_words, list):
                raise ValueError(
                    f"Benchmark case {case_id} expected_word_timestamps must be a list"
                )
            for index, expected_word in enumerate(expected_words):
                if not isinstance(expected_word, dict):
                    raise ValueError(
                        f"Benchmark case {case_id} expected_word_timestamps entries must be objects"
                    )
                text = expected_word.get("text")
                start_s = expected_word.get("start_s")
                end_s = expected_word.get("end_s")
                if (
                    not isinstance(text, str)
                    or not text
                    or isinstance(start_s, bool)
                    or isinstance(end_s, bool)
                    or not isinstance(start_s, (int, float))
                    or not isinstance(end_s, (int, float))
                    or not isfinite(float(start_s))
                    or not isfinite(float(end_s))
                    or not 0 <= float(start_s) <= float(end_s) <= float(case["duration_s"])
                ):
                    raise ValueError(
                        f"Benchmark case {case_id} expected_word_timestamps entry {index} has invalid bounds"
                    )
    return manifest, cases


def _as_number(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return int(value)


def _boundary_metrics(
    result: ASRResult,
    expected_window: Any,
    *,
    timestamp_supported: bool,
) -> dict[str, Any]:
    if not timestamp_supported:
        return {
            "available": False,
            "end_absolute_error_s": None,
            "mean_absolute_error_s": None,
            "reason": "adapter does not declare timestamp support",
            "start_absolute_error_s": None,
            "status": "NOT_APPLICABLE",
            "supported": False,
        }
    if not isinstance(expected_window, list) or len(expected_window) != 2:
        return {
            "available": False,
            "end_absolute_error_s": None,
            "mean_absolute_error_s": None,
            "reason": "benchmark case has no trusted speech-window reference",
            "start_absolute_error_s": None,
            "status": "NOT_APPLICABLE",
            "supported": True,
        }
    spans = [
        (segment.start_s, segment.end_s)
        for segment in result.segments
        if segment.start_s is not None and segment.end_s is not None
    ]
    if not spans:
        return {
            "available": False,
            "end_absolute_error_s": None,
            "mean_absolute_error_s": None,
            "reason": "provider returned no segment timestamps",
            "start_absolute_error_s": None,
            "status": "NOT_RUN",
            "supported": True,
        }
    actual_start = min(start for start, _ in spans)
    actual_end = max(end for _, end in spans)
    start_error = abs(actual_start - float(expected_window[0]))
    end_error = abs(actual_end - float(expected_window[1]))
    return {
        "available": True,
        "end_absolute_error_s": round(end_error, 9),
        "mean_absolute_error_s": round((start_error + end_error) / 2, 9),
        "reason": None,
        "start_absolute_error_s": round(start_error, 9),
        "status": "MEASURED",
        "supported": True,
    }


def _phrase_timing_metrics(
    result: ASRResult,
    *,
    duration_s: float,
    timestamp_supported: bool,
) -> dict[str, Any]:
    """Summarize usable phrase timing without requiring manual boundary truth."""

    transcript_tokens = normalize_transcript(result.transcript).split()
    if not transcript_tokens:
        return {
            "coverage": None,
            "finite": None,
            "in_bounds": None,
            "interval_count": 0,
            "monotonic": None,
            "non_negative": None,
            "reason": "provider returned no transcript",
            "sensible_coverage": None,
            "source": None,
            "status": "NOT_APPLICABLE",
        }
    if not timestamp_supported:
        return {
            "coverage": None,
            "finite": None,
            "in_bounds": None,
            "interval_count": 0,
            "monotonic": None,
            "non_negative": None,
            "reason": "adapter does not declare timestamp support",
            "sensible_coverage": None,
            "source": None,
            "status": "NOT_APPLICABLE",
        }

    spans = result.segments
    source = "segments"
    if not spans and result.words:
        spans = (
            ASRSegment(
                text=result.transcript,
                start_s=min(word.start_s for word in result.words),
                end_s=max(word.end_s for word in result.words),
            ),
        )
        source = "word_envelope"
    if not spans:
        return {
            "coverage": None,
            "finite": None,
            "in_bounds": None,
            "interval_count": 0,
            "monotonic": None,
            "non_negative": None,
            "reason": "provider returned no phrase or word timestamps for speech output",
            "sensible_coverage": None,
            "source": None,
            "status": "NOT_RUN",
        }

    bounds = [(span.start_s, span.end_s) for span in spans]
    finite = all(
        isinstance(start, (int, float))
        and not isinstance(start, bool)
        and isinstance(end, (int, float))
        and not isinstance(end, bool)
        and isfinite(float(start))
        and isfinite(float(end))
        for start, end in bounds
    )
    if not finite:
        return {
            "coverage": None,
            "finite": False,
            "in_bounds": False,
            "interval_count": len(spans),
            "monotonic": False,
            "non_negative": False,
            "reason": "provider returned a non-finite phrase timestamp",
            "sensible_coverage": None,
            "source": source,
            "status": "MEASURED",
        }

    numeric_bounds = [
        (float(start), float(end)) for start, end in bounds if start is not None and end is not None
    ]
    coverage = round(
        sum(len(normalize_transcript(span.text).split()) for span in spans)
        / len(transcript_tokens),
        9,
    )
    non_negative = all(start >= 0 and end >= 0 for start, end in numeric_bounds)
    monotonic = all(
        current_start >= previous_end
        for (_, previous_end), (current_start, _) in zip(numeric_bounds, numeric_bounds[1:])
    )
    in_bounds = all(start <= end <= duration_s for start, end in numeric_bounds)
    return {
        "coverage": coverage,
        "finite": True,
        "in_bounds": in_bounds,
        "interval_count": len(spans),
        "monotonic": monotonic,
        "non_negative": non_negative,
        "reason": None,
        "sensible_coverage": 0.5 <= coverage <= 1.5,
        "source": source,
        "status": "MEASURED",
    }


def _word_timing_metrics(result: ASRResult, case: dict[str, Any]) -> dict[str, Any]:
    """Return transcript-free timing quality evidence for a single ASR result."""

    words = result.words
    transcript_tokens = normalize_transcript(result.transcript).split()
    timed_tokens = sum(len(normalize_transcript(word.text).split()) for word in words)
    coverage = round(timed_tokens / len(transcript_tokens), 9) if transcript_tokens else None
    duration_s = float(case["duration_s"])
    monotonic = all(
        current.start_s >= previous.end_s for previous, current in zip(words, words[1:])
    )
    in_bounds = all(0 <= word.start_s <= word.end_s <= duration_s for word in words)
    references = case.get("expected_word_timestamps")
    if references is None:
        references = []
    if not isinstance(references, list):
        raise ValueError("expected_word_timestamps must be a list when provided")

    errors: list[float] = []
    matched_pairs: list[tuple[ASRWordSpan, dict[str, Any]]] = []
    for actual, expected in zip(words, references):
        if not isinstance(expected, dict):
            raise ValueError("expected_word_timestamps entries must be objects")
        expected_text = expected.get("text")
        expected_start = expected.get("start_s")
        expected_end = expected.get("end_s")
        if (
            not isinstance(expected_text, str)
            or not isinstance(expected_start, (int, float))
            or not isinstance(expected_end, (int, float))
            or isinstance(expected_start, bool)
            or isinstance(expected_end, bool)
        ):
            raise ValueError("expected_word_timestamps entries require text, start_s, and end_s")
        if normalize_transcript(actual.text) != normalize_transcript(expected_text):
            break
        matched_pairs.append((actual, expected))
        errors.extend((actual.start_s - float(expected_start), actual.end_s - float(expected_end)))

    absolute_errors = [abs(error) for error in errors]
    p95_index = max(0, (95 * len(absolute_errors) + 99) // 100 - 1)
    drift = (
        matched_pairs[-1][0].end_s - float(matched_pairs[-1][1]["end_s"]) if matched_pairs else None
    )
    return {
        "alignment_origin": result.alignment_origin,
        "boundary_mae_s": round(sum(absolute_errors) / len(absolute_errors), 9)
        if absolute_errors
        else None,
        "boundary_p95_s": round(sorted(absolute_errors)[p95_index], 9) if absolute_errors else None,
        "coverage": coverage,
        "drift_s": round(drift, 9) if drift is not None else None,
        "in_bounds": in_bounds,
        "monotonic": monotonic,
        "reference_word_count": len(references),
        "reason": None if references else "benchmark case has no trusted word-boundary reference",
        "signed_boundary_error_s": round(sum(errors) / len(errors), 9) if errors else None,
        "status": "MEASURED" if references else "NOT_APPLICABLE",
        "word_count": len(words),
        "zero_duration_count": sum(1 for word in words if word.start_s == word.end_s),
    }


def _case_record(
    *,
    case: dict[str, Any],
    result: ASRResult,
    wall_s: float,
    resources: ResourceSnapshot,
    timestamp_supported: bool,
) -> dict[str, Any]:
    counts = _error_counts(case["expected_text"], result.transcript)
    duration_s = float(case["duration_s"])
    peak_ram = (
        max(
            value
            for value in (
                resources.peak_ram_bytes,
                _as_number(result.execution.measurements.get("peak_ram_bytes")),
            )
            if value is not None
        )
        if any(
            value is not None
            for value in (
                resources.peak_ram_bytes,
                _as_number(result.execution.measurements.get("peak_ram_bytes")),
            )
        )
        else None
    )
    peak_vram = (
        max(
            value
            for value in (
                resources.peak_vram_bytes,
                _as_number(result.execution.measurements.get("peak_vram_bytes")),
            )
            if value is not None
        )
        if any(
            value is not None
            for value in (
                resources.peak_vram_bytes,
                _as_number(result.execution.measurements.get("peak_vram_bytes")),
            )
        )
        else None
    )
    return {
        "audio_sha256": case["sha256"],
        "case_id": case["case_id"],
        "category": case.get("category", "unknown"),
        "cer": _rate(counts["character_errors"], counts["character_count"]),
        "character_count": counts["character_count"],
        "character_errors": counts["character_errors"],
        "duration_s": duration_s,
        "language": result.language or case["language"],
        "peak_ram_bytes": peak_ram,
        "peak_vram_bytes": peak_vram,
        "phrase_timing": _phrase_timing_metrics(
            result,
            duration_s=duration_s,
            timestamp_supported=timestamp_supported,
        ),
        "rtf": round(wall_s / duration_s, 9),
        "speech_detected": bool(normalize_transcript(result.transcript)),
        "timestamp_boundary": _boundary_metrics(
            result,
            case.get("expected_speech_window_s"),
            timestamp_supported=timestamp_supported,
        ),
        "word_timing": _word_timing_metrics(result, case),
        "wall_s": round(wall_s, 9),
        "wer": _rate(counts["word_errors"], counts["word_count"]),
        "word_count": counts["word_count"],
        "word_errors": counts["word_errors"],
    }


def _subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    word_errors = sum(record["word_errors"] for record in records)
    word_count = sum(record["word_count"] for record in records)
    character_errors = sum(record["character_errors"] for record in records)
    character_count = sum(record["character_count"] for record in records)
    total_wall_s = sum(record["wall_s"] for record in records)
    total_duration_s = sum(record["duration_s"] for record in records)
    ram_values = [
        record["peak_ram_bytes"] for record in records if record["peak_ram_bytes"] is not None
    ]
    vram_values = [
        record["peak_vram_bytes"] for record in records if record["peak_vram_bytes"] is not None
    ]
    boundaries = [
        record["timestamp_boundary"]["mean_absolute_error_s"]
        for record in records
        if record["timestamp_boundary"]["available"]
    ]
    supported_count = sum(1 for record in records if record["timestamp_boundary"]["supported"])
    phrase_timings = [record["phrase_timing"] for record in records]
    phrase_coverages = [
        timing["coverage"] for timing in phrase_timings if timing["coverage"] is not None
    ]
    word_timings = [record["word_timing"] for record in records]
    word_coverages = [
        timing["coverage"] for timing in word_timings if timing["coverage"] is not None
    ]
    word_boundary_maes = [
        timing["boundary_mae_s"] for timing in word_timings if timing["boundary_mae_s"] is not None
    ]
    word_boundary_p95s = [
        timing["boundary_p95_s"] for timing in word_timings if timing["boundary_p95_s"] is not None
    ]
    word_signed_errors = [
        timing["signed_boundary_error_s"]
        for timing in word_timings
        if timing["signed_boundary_error_s"] is not None
    ]
    word_drifts = [timing["drift_s"] for timing in word_timings if timing["drift_s"] is not None]
    return {
        "case_count": len(records),
        "cer": _rate(character_errors, character_count),
        "character_count": character_count,
        "character_errors": character_errors,
        "peak_ram_bytes": max(ram_values) if ram_values else None,
        "peak_vram_bytes": max(vram_values) if vram_values else None,
        "phrase_timing": {
            "available_case_count": sum(
                1 for timing in phrase_timings if timing["status"] == "MEASURED"
            ),
            "finite_case_count": sum(1 for timing in phrase_timings if timing["finite"] is True),
            "in_bounds_case_count": sum(
                1 for timing in phrase_timings if timing["in_bounds"] is True
            ),
            "mean_coverage": round(sum(phrase_coverages) / len(phrase_coverages), 9)
            if phrase_coverages
            else None,
            "monotonic_case_count": sum(
                1 for timing in phrase_timings if timing["monotonic"] is True
            ),
            "non_negative_case_count": sum(
                1 for timing in phrase_timings if timing["non_negative"] is True
            ),
            "not_applicable_case_count": sum(
                1 for timing in phrase_timings if timing["status"] == "NOT_APPLICABLE"
            ),
            "not_run_case_count": sum(
                1 for timing in phrase_timings if timing["status"] == "NOT_RUN"
            ),
            "sensible_coverage_case_count": sum(
                1 for timing in phrase_timings if timing["sensible_coverage"] is True
            ),
        },
        "rtf": round(total_wall_s / total_duration_s, 9) if total_duration_s else None,
        "subsets": {},
        "timestamp_boundary": {
            "available_case_count": len(boundaries),
            "mean_absolute_error_s": round(sum(boundaries) / len(boundaries), 9)
            if boundaries
            else None,
            "not_applicable_case_count": sum(
                1
                for record in records
                if record["timestamp_boundary"]["status"] == "NOT_APPLICABLE"
            ),
            "not_run_case_count": sum(
                1 for record in records if record["timestamp_boundary"]["status"] == "NOT_RUN"
            ),
            "supported_case_count": supported_count,
        },
        "word_timing": {
            "boundary_mae_s": round(sum(word_boundary_maes) / len(word_boundary_maes), 9)
            if word_boundary_maes
            else None,
            "boundary_p95_s": max(word_boundary_p95s) if word_boundary_p95s else None,
            "coverage": round(sum(word_coverages) / len(word_coverages), 9)
            if word_coverages
            else None,
            "drift_s": round(sum(word_drifts) / len(word_drifts), 9) if word_drifts else None,
            "in_bounds_case_count": sum(1 for timing in word_timings if timing["in_bounds"]),
            "monotonic_case_count": sum(1 for timing in word_timings if timing["monotonic"]),
            "reference_case_count": sum(
                1 for timing in word_timings if timing["reference_word_count"] > 0
            ),
            "signed_boundary_error_s": round(sum(word_signed_errors) / len(word_signed_errors), 9)
            if word_signed_errors
            else None,
            "timed_word_count": sum(timing["word_count"] for timing in word_timings),
            "zero_duration_count": sum(timing["zero_duration_count"] for timing in word_timings),
        },
        "wall_s": round(total_wall_s, 9),
        "wer": _rate(word_errors, word_count),
        "word_count": word_count,
        "word_errors": word_errors,
    }


def _run_summary(records: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _subset_summary(records)
    by_id = {case["case_id"]: case for case in cases}
    for name, predicate in (
        ("noise", lambda case: case.get("noise", {}).get("category") != "none"),
        (
            "pause",
            lambda case: (
                case.get("pause", {}).get("leading_s", 0) > 0
                or case.get("pause", {}).get("trailing_s", 0) > 0
            ),
        ),
    ):
        subset = [record for record in records if predicate(by_id[record["case_id"]])]
        summary["subsets"][name] = _subset_summary(subset)
        summary["subsets"][name].pop("subsets")
    return summary


def _run_benchmark_mode(
    cases: list[dict[str, Any]],
    manifest_path: Path,
    adapter: BenchmarkAdapter,
    *,
    corpus_root: Path | None,
    prompt: str | None,
    require_phrase_timing: bool,
    monitor_factory: Callable[[], BenchmarkResourceMonitor],
    clock: Callable[[], float],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    runtime_receipts: set[tuple[str, str | None, str | None, str, str]] = set()
    model_ids: set[str] = set()
    for case in cases:
        audio_path = _relative_audio_path(
            manifest_path,
            case["audio_path"],
            corpus_root=corpus_root,
        )
        monitor = monitor_factory()
        monitor.start()
        started = clock()
        try:
            result = adapter.transcribe(
                audio_path=audio_path,
                language=case["language"],
                prompt=prompt,
            )
        finally:
            wall_s = max(clock() - started, 0.0)
            resources = monitor.stop()
        if result.provider_id != adapter.provider_id:
            raise ValueError(
                f"Benchmark adapter {adapter.provider_id} returned {result.provider_id}"
            )
        model_ids.add(result.model_id)
        runtime_receipts.add(
            (
                result.execution.runtime,
                result.execution.runtime_version,
                result.execution.model_revision,
                result.execution.resolved_device,
                result.execution.resolved_compute,
            )
        )
        records.append(
            _case_record(
                case=case,
                result=result,
                wall_s=wall_s,
                resources=resources,
                timestamp_supported=adapter.timestamp_supported,
            )
        )
    if require_phrase_timing:
        invalid_cases = [
            record["case_id"]
            for record in records
            if record["speech_detected"]
            and adapter.timestamp_supported
            and (
                record["phrase_timing"]["status"] != "MEASURED"
                or record["phrase_timing"]["finite"] is not True
                or record["phrase_timing"]["non_negative"] is not True
                or record["phrase_timing"]["monotonic"] is not True
                or record["phrase_timing"]["in_bounds"] is not True
                or record["phrase_timing"]["sensible_coverage"] is not True
            )
        ]
        if invalid_cases:
            raise ValueError(
                "Phrase timing requirement failed for benchmark cases: " + ", ".join(invalid_cases)
            )
    receipts = [
        {
            "device": device,
            "model_revision": model_revision,
            "runtime": runtime,
            "runtime_version": runtime_version,
            "compute": compute,
        }
        for runtime, runtime_version, model_revision, device, compute in sorted(runtime_receipts)
    ]
    return {
        "cases": records,
        "execution": receipts,
        "model_ids": sorted(model_ids),
        "prompt_enabled": prompt is not None,
        "summary": _run_summary(records, cases),
    }


def run_benchmark(
    manifest_path: Path | str,
    adapter: BenchmarkAdapter,
    *,
    corpus_root: Path | str | None = None,
    prompt_text: str | None = None,
    require_phrase_timing: bool = False,
    monitor_factory: Callable[[], BenchmarkResourceMonitor] = ResourceMonitor,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute a local adapter over a manifest without writing any score itself."""

    path = Path(manifest_path)
    resolved_corpus_root = Path(corpus_root) if corpus_root is not None else None
    manifest, cases = _load_manifest(path)
    if prompt_text is not None and not prompt_text.strip():
        raise ValueError("Benchmark prompt text must not be blank")
    modes: list[str | None] = [None]
    skipped_reason: str | None = None
    if prompt_text is not None:
        if adapter.prompt_supported:
            modes.append(prompt_text)
        else:
            skipped_reason = "adapter does not declare prompt support"
    runs = [
        _run_benchmark_mode(
            cases,
            path,
            adapter,
            corpus_root=resolved_corpus_root,
            prompt=prompt,
            require_phrase_timing=require_phrase_timing,
            monitor_factory=monitor_factory,
            clock=clock,
        )
        for prompt in modes
    ]
    prompt_summary: dict[str, Any] = {
        "enabled_run_count": sum(1 for run in runs if run["prompt_enabled"]),
        "provided": prompt_text is not None,
    }
    if skipped_reason is not None:
        prompt_summary["skipped_reason"] = skipped_reason
    return {
        "corpus": {
            "case_count": len(cases),
            "id": manifest["corpus_id"],
            "schema_version": manifest["schema_version"],
        },
        "prompt": prompt_summary,
        "timing": {
            "phrase_timing_required": require_phrase_timing,
            "word_timing_is_optional": True,
        },
        "provider": {
            "id": adapter.provider_id,
            "prompt_supported": adapter.prompt_supported,
            "timestamp_supported": adapter.timestamp_supported,
        },
        "runs": runs,
        "schema_version": REPORT_SCHEMA_VERSION,
    }


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# ASR benchmark report",
        "",
        f"- Corpus: `{report['corpus']['id']}` ({report['corpus']['case_count']} cases)",
        f"- Provider: `{report['provider']['id']}`",
        "- Phrase timing is required."
        if report["timing"]["phrase_timing_required"]
        else "- Phrase timing is best-effort.",
        "- Word-level timing is optional evidence when the provider emits it.",
        "- Reports omit reference text, transcripts, and prompt contents.",
        "- RAM is the local process high-water mark; VRAM is best-effort `nvidia-smi` sampling.",
        "",
        "| Prompt enabled | WER | CER | RTF | Peak RAM bytes | Peak VRAM bytes | Phrase timing coverage | Timestamp boundary MAE s | Word timing MAE s | Timed words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        summary = run["summary"]
        boundary = summary["timestamp_boundary"]["mean_absolute_error_s"]
        phrase_timing = summary["phrase_timing"]
        word_timing = summary["word_timing"]
        prompt_state = "yes" if run["prompt_enabled"] else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    prompt_state,
                    _format_metric(summary["wer"]),
                    _format_metric(summary["cer"]),
                    _format_metric(summary["rtf"]),
                    _format_metric(summary["peak_ram_bytes"]),
                    _format_metric(summary["peak_vram_bytes"]),
                    _format_metric(phrase_timing["mean_coverage"]),
                    _format_metric(boundary),
                    _format_metric(word_timing["boundary_mae_s"]),
                    _format_metric(word_timing["timed_word_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Noise and pause subsets",
            "",
            "| Prompt enabled | Subset | Cases | WER | CER | RTF | Timestamp boundary MAE s | Word timing MAE s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        prompt_state = "yes" if run["prompt_enabled"] else "no"
        for subset_name in ("noise", "pause"):
            subset = run["summary"]["subsets"][subset_name]
            boundary = subset["timestamp_boundary"]["mean_absolute_error_s"]
            word_timing = subset["word_timing"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        prompt_state,
                        subset_name,
                        _format_metric(subset["case_count"]),
                        _format_metric(subset["wer"]),
                        _format_metric(subset["cer"]),
                        _format_metric(subset["rtf"]),
                        _format_metric(boundary),
                        _format_metric(word_timing["boundary_mae_s"]),
                    ]
                )
                + " |"
            )
    if report["prompt"].get("skipped_reason"):
        lines.extend(
            [
                "",
                "## Prompt comparison",
                "",
                f"Prompt-on run was skipped: {report['prompt']['skipped_reason']}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_reports(report: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    """Write deterministic report paths with the project's existing JSON writer."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / _REPORT_JSON_NAME
    markdown_path = directory / _REPORT_MARKDOWN_NAME
    write_json(json_path, report)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path
