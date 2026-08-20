import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import fixture_path

import voiceover_pipeline.asr_benchmark as asr_benchmark
from voiceover_pipeline.asr_benchmark import (
    ResourceSnapshot,
    create_benchmark_adapter,
    normalize_transcript,
    run_benchmark,
    write_benchmark_reports,
)
from voiceover_pipeline.models import (
    ASRCapabilities,
    ASRExecutionReceipt,
    ASRResult,
    ASRSegment,
    ASRWordSpan,
)

CORPUS_MANIFEST = fixture_path("asr_evaluation") / "manifest.json"


_WINDOWS = {
    "clean-short": (0.0, 2.469375),
    "leading-pause": (0.75, 3.041375),
    "trailing-pause": (0.0, 2.0255),
    "white-noise": (0.0, 2.803375),
    "pause-and-white-noise": (0.5, 3.366875),
}


class FixtureBenchmarkAdapter:
    provider_id = "fixture-qwen"
    prompt_supported = True
    timestamp_supported = True

    def __init__(self) -> None:
        self.requests = []

    def transcribe(self, *, audio_path: Path, language: str, prompt: str | None) -> ASRResult:
        self.requests.append((audio_path.name, language, prompt))
        case_id = audio_path.stem
        transcript = {
            "clean-short": "The quick brown fox checks one two three.",
            "leading-pause": "A pause comes before this sentence.",
            "trailing-pause": "This sentence ends before a pause.",
            "white-noise": "White noise makes speech recognition harder.",
            "pause-and-white-noise": "A long pause and white noise test the boundary.",
        }[case_id]
        if case_id == "white-noise" and prompt is None:
            transcript = "White noise makes recognition harder."
        start_s, end_s = _WINDOWS[case_id]
        return ASRResult(
            transcript=transcript,
            provider_id=self.provider_id,
            model_id="fixture-model",
            language=language,
            execution=ASRExecutionReceipt(
                runtime="fixture-runtime",
                runtime_version="1.0",
                model_revision="fixture-revision",
                resolved_device="cpu",
                resolved_compute="float32",
            ),
            segments=(ASRSegment(text=transcript, start_s=start_s, end_s=end_s),),
            alignment_origin="native",
        )


class FixtureTextOnlyAdapter(FixtureBenchmarkAdapter):
    provider_id = "fixture-text-only"
    prompt_supported = False
    timestamp_supported = False

    def transcribe(self, *, audio_path: Path, language: str, prompt: str | None) -> ASRResult:
        assert prompt is None
        result = super().transcribe(audio_path=audio_path, language=language, prompt=prompt)
        return ASRResult(
            transcript=result.transcript,
            provider_id=self.provider_id,
            model_id=result.model_id,
            language=result.language,
            execution=result.execution,
        )


class FixtureWordTimingAdapter(FixtureBenchmarkAdapter):
    def transcribe(self, *, audio_path: Path, language: str, prompt: str | None) -> ASRResult:
        result = super().transcribe(audio_path=audio_path, language=language, prompt=prompt)
        words = tuple(
            ASRWordSpan(text=f"{word} ", start_s=index * 0.2, end_s=index * 0.2 + 0.1)
            for index, word in enumerate(result.transcript.rstrip(".").split())
        )
        return ASRResult(
            transcript=result.transcript,
            provider_id=self.provider_id,
            model_id=result.model_id,
            language=result.language,
            duration_s=4.0,
            execution=result.execution,
            words=words,
            alignment_origin="forced",
        )


class FixtureMonitor:
    def start(self) -> None:
        return None

    def stop(self) -> ResourceSnapshot:
        return ResourceSnapshot(peak_ram_bytes=12_345, peak_vram_bytes=67_890)


def _fixture_monitor() -> FixtureMonitor:
    return FixtureMonitor()


def test_benchmark_emits_prompt_matrix_metrics_subsets_and_stable_reports(tmp_path):
    adapter = FixtureBenchmarkAdapter()

    report = run_benchmark(
        CORPUS_MANIFEST,
        adapter,
        prompt_text="Use the product glossary.",
        monitor_factory=_fixture_monitor,
    )

    assert report["schema_version"] == 2
    assert report["corpus"] == {
        "case_count": 5,
        "id": "synthetic-asr-evaluation-v1",
        "schema_version": 1,
    }
    assert len(report["runs"]) == 2
    assert [run["prompt_enabled"] for run in report["runs"]] == [False, True]
    assert all("prompt_text" not in json.dumps(run) for run in report["runs"])
    assert "Use the product glossary." not in json.dumps(report)

    prompt_off, prompt_on = report["runs"]
    assert prompt_off["summary"]["subsets"]["noise"]["case_count"] == 2
    assert prompt_off["summary"]["subsets"]["pause"]["case_count"] == 3
    assert prompt_off["summary"]["wer"] > 0
    assert prompt_on["summary"]["wer"] == 0
    assert prompt_on["summary"]["cer"] == 0
    assert prompt_on["summary"]["peak_ram_bytes"] == 12_345
    assert prompt_on["summary"]["peak_vram_bytes"] == 67_890
    assert prompt_on["model_ids"] == ["fixture-model"]
    assert prompt_on["execution"][0]["model_revision"] == "fixture-revision"
    assert prompt_on["summary"]["timestamp_boundary"] == {
        "available_case_count": 5,
        "mean_absolute_error_s": 0.0,
        "not_applicable_case_count": 0,
        "not_run_case_count": 0,
        "supported_case_count": 5,
    }
    assert prompt_on["summary"]["phrase_timing"] == {
        "available_case_count": 5,
        "finite_case_count": 5,
        "in_bounds_case_count": 5,
        "mean_coverage": 1.0,
        "monotonic_case_count": 5,
        "non_negative_case_count": 5,
        "not_applicable_case_count": 0,
        "not_run_case_count": 0,
        "sensible_coverage_case_count": 5,
    }
    assert [request[2] is not None for request in adapter.requests] == [False] * 5 + [True] * 5

    json_path, markdown_path = write_benchmark_reports(report, tmp_path)
    first_json = json_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    second_json_path, second_markdown_path = write_benchmark_reports(report, tmp_path)

    assert second_json_path == json_path
    assert second_markdown_path == markdown_path
    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    assert json.loads(first_json) == report
    assert "# ASR benchmark report" in first_markdown
    assert "Prompt enabled" in first_markdown
    assert "## Noise and pause subsets" in first_markdown


def test_benchmark_honestly_skips_prompt_on_and_timestamp_metric_when_unsupported():
    report = run_benchmark(
        CORPUS_MANIFEST,
        FixtureTextOnlyAdapter(),
        prompt_text="This must not reach a text-only adapter.",
        monitor_factory=_fixture_monitor,
    )

    assert len(report["runs"]) == 1
    assert report["prompt"] == {
        "enabled_run_count": 0,
        "provided": True,
        "skipped_reason": "adapter does not declare prompt support",
    }
    assert report["runs"][0]["summary"]["timestamp_boundary"] == {
        "available_case_count": 0,
        "mean_absolute_error_s": None,
        "not_applicable_case_count": 5,
        "not_run_case_count": 0,
        "supported_case_count": 0,
    }


def test_benchmark_records_word_timing_quality_fields_without_exposing_transcripts():
    report = run_benchmark(
        CORPUS_MANIFEST,
        FixtureWordTimingAdapter(),
        monitor_factory=_fixture_monitor,
    )

    timing = report["runs"][0]["cases"][0]["word_timing"]

    assert timing == {
        "alignment_origin": "forced",
        "boundary_mae_s": None,
        "boundary_p95_s": None,
        "coverage": 1.0,
        "drift_s": None,
        "in_bounds": True,
        "monotonic": True,
        "reference_word_count": 0,
        "reason": "benchmark case has no trusted word-boundary reference",
        "signed_boundary_error_s": None,
        "status": "NOT_APPLICABLE",
        "word_count": 8,
        "zero_duration_count": 0,
    }


def test_benchmark_without_trusted_boundary_truth_completes_with_explicit_statuses(tmp_path):
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["cases"] = [manifest["cases"][0]]
    manifest["cases"][0].pop("expected_speech_window_s")
    manifest["cases"][0].pop("expected_word_timestamps", None)
    manifest_path = tmp_path / "transcript-resource-lane.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_benchmark(
        manifest_path,
        FixtureBenchmarkAdapter(),
        corpus_root=CORPUS_MANIFEST.parent,
        require_phrase_timing=True,
        monitor_factory=_fixture_monitor,
    )

    case = report["runs"][0]["cases"][0]
    assert report["timing"] == {
        "phrase_timing_required": True,
        "word_timing_is_optional": True,
    }
    assert case["speech_detected"] is True
    assert case["timestamp_boundary"]["status"] == "NOT_APPLICABLE"
    assert case["word_timing"]["status"] == "NOT_APPLICABLE"
    assert case["phrase_timing"] == {
        "coverage": 1.0,
        "finite": True,
        "in_bounds": True,
        "interval_count": 1,
        "monotonic": True,
        "non_negative": True,
        "reason": None,
        "sensible_coverage": True,
        "source": "segments",
        "status": "MEASURED",
    }


def test_phrase_timing_requirement_allows_a_no_speech_result(tmp_path):
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["cases"] = [manifest["cases"][0]]
    manifest_path = tmp_path / "no-speech-lane.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FixtureNoSpeechAdapter(FixtureBenchmarkAdapter):
        def transcribe(self, *, audio_path: Path, language: str, prompt: str | None) -> ASRResult:
            return ASRResult(
                transcript="",
                provider_id=self.provider_id,
                model_id="fixture-model",
                language=language,
                execution=ASRExecutionReceipt(
                    runtime="fixture-runtime",
                    resolved_device="cpu",
                    resolved_compute="float32",
                ),
            )

    report = run_benchmark(
        manifest_path,
        FixtureNoSpeechAdapter(),
        corpus_root=CORPUS_MANIFEST.parent,
        require_phrase_timing=True,
        monitor_factory=_fixture_monitor,
    )

    case = report["runs"][0]["cases"][0]
    assert case["speech_detected"] is False
    assert case["phrase_timing"]["status"] == "NOT_APPLICABLE"
    assert "quick brown fox" not in json.dumps(report)


def test_phrase_timing_requirement_rejects_supported_speech_without_timing(tmp_path):
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["cases"] = [manifest["cases"][0]]
    manifest_path = tmp_path / "missing-timing-lane.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FixtureUntimedSpeechAdapter(FixtureBenchmarkAdapter):
        def transcribe(self, *, audio_path: Path, language: str, prompt: str | None) -> ASRResult:
            return ASRResult(
                transcript="timing is required",
                provider_id=self.provider_id,
                model_id="fixture-model",
                language=language,
                execution=ASRExecutionReceipt(
                    runtime="fixture-runtime",
                    resolved_device="cpu",
                    resolved_compute="float32",
                ),
            )

    with pytest.raises(ValueError, match="Phrase timing requirement failed.*clean-short"):
        run_benchmark(
            manifest_path,
            FixtureUntimedSpeechAdapter(),
            corpus_root=CORPUS_MANIFEST.parent,
            require_phrase_timing=True,
            monitor_factory=_fixture_monitor,
        )


def test_benchmark_rejects_malformed_manual_word_timing_references(tmp_path):
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["cases"] = [manifest["cases"][0]]
    manifest["cases"][0]["expected_word_timestamps"] = [
        {"text": "The", "start_s": 0.8, "end_s": 0.1}
    ]
    manifest_path = tmp_path / "manual-timing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_word_timestamps.*bounds"):
        run_benchmark(
            manifest_path,
            FixtureWordTimingAdapter(),
            corpus_root=CORPUS_MANIFEST.parent,
            monitor_factory=_fixture_monitor,
        )


def test_registry_benchmark_adapter_requests_word_timestamps_for_timestamp_capable_provider(
    monkeypatch,
):
    captured: list[object] = []

    class FixtureProvider:
        def transcribe(self, request):
            captured.append(request)
            return ASRResult(
                transcript="fixture",
                provider_id="qwen-local",
                model_id="fixture-model",
                execution=ASRExecutionReceipt(runtime="fixture", resolved_compute="float32"),
                words=(ASRWordSpan(text="fixture", start_s=0.0, end_s=0.1),),
                alignment_origin="forced",
            )

    spec = SimpleNamespace(
        provider_id="qwen-local",
        capabilities=ASRCapabilities(word_timestamps=True, forced_alignment=True),
        factory=FixtureProvider,
    )
    monkeypatch.setattr(asr_benchmark, "get_asr_provider_spec", lambda _provider_id: spec)

    adapter = asr_benchmark.RegistryBenchmarkAdapter(
        "qwen-local", model_id=None, device="cpu", compute="float32"
    )
    result = adapter.transcribe(audio_path=Path("fixture.wav"), language="ru", prompt=None)

    assert result.alignment_origin == "forced"
    assert captured[0].timestamp_mode == "word"


def test_benchmark_resolves_external_corpus_paths_from_an_explicit_root(tmp_path):
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["cases"] = manifest["cases"][:1]
    manifest["corpus_id"] = "external-root-fixture-v1"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        run_benchmark(manifest_path, FixtureBenchmarkAdapter(), monitor_factory=_fixture_monitor)
    except FileNotFoundError as error:
        assert "Benchmark audio file not found" in str(error)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("expected local manifest-relative resolution to fail")

    report = run_benchmark(
        manifest_path,
        FixtureBenchmarkAdapter(),
        corpus_root=CORPUS_MANIFEST.parent,
        monitor_factory=_fixture_monitor,
    )

    assert report["corpus"] == {
        "case_count": 1,
        "id": "external-root-fixture-v1",
        "schema_version": 1,
    }

    try:
        run_benchmark(
            manifest_path,
            FixtureBenchmarkAdapter(),
            corpus_root=tmp_path / "missing-root",
            monitor_factory=_fixture_monitor,
        )
    except FileNotFoundError as error:
        assert "Benchmark corpus root not found" in str(error)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("expected a missing explicit corpus root to fail")


def test_normalization_and_offline_adapter_factories_cover_whisper_qwen_and_nemotron():
    assert normalize_transcript(" Привет,   МИР! ") == "привет мир"

    whisper = create_benchmark_adapter("whisper", model_id="small", device="cpu", compute="int8")
    qwen = create_benchmark_adapter("qwen-local", model_id=None, device="cpu", compute="float32")
    nemotron = create_benchmark_adapter(
        "nemotron-local", model_id=None, device="cpu", compute="auto"
    )

    assert (whisper.provider_id, whisper.prompt_supported, whisper.timestamp_supported) == (
        "faster-whisper",
        False,
        True,
    )
    assert (qwen.provider_id, qwen.prompt_supported, qwen.timestamp_supported) == (
        "qwen-local",
        True,
        True,
    )
    assert (nemotron.provider_id, nemotron.prompt_supported, nemotron.timestamp_supported) == (
        "nemotron-local",
        False,
        True,
    )


def test_whisper_adapter_maps_existing_timing_provider_without_loading_a_model(monkeypatch):
    calls = []

    class FakeFasterWhisperProvider:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def transcribe(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                segments=[
                    SimpleNamespace(text="fixture transcript", start_sec=0.25, end_sec=1.5),
                ],
                model="small",
                backend="faster-whisper",
                device="cpu",
                compute_type="int8",
                language="en",
            )

    monkeypatch.setattr(asr_benchmark, "FasterWhisperProvider", FakeFasterWhisperProvider)
    result = create_benchmark_adapter(
        "whisper", model_id="small", device="cpu", compute="int8"
    ).transcribe(
        audio_path=CORPUS_MANIFEST.parent / "audio" / "clean-short.wav",
        language="en",
        prompt=None,
    )

    assert result.transcript == "fixture transcript"
    assert result.segments[0].start_s == 0.25
    assert result.segments[0].end_s == 1.5
    assert calls == [
        {"model_size": "small", "device": "cpu", "compute_type": "int8"},
        {
            "audio_path": CORPUS_MANIFEST.parent / "audio" / "clean-short.wav",
            "language": "en",
            "word_timestamps": True,
            "quiet": True,
        },
    ]


def test_benchmark_tool_help_is_available_without_initializing_any_provider():
    tool_path = Path(__file__).resolve().parent.parent / "tools" / "run_asr_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(tool_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "qwen-local" in completed.stdout
    assert "nemotron-local" in completed.stdout
    assert "whisper" in completed.stdout
    assert "--corpus-root" in completed.stdout
    assert "--require-phrase-timing" in completed.stdout
