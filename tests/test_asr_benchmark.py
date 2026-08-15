import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from conftest import fixture_path
import voiceover_pipeline.asr_benchmark as asr_benchmark
from voiceover_pipeline.asr_benchmark import (
    ResourceSnapshot,
    create_benchmark_adapter,
    normalize_transcript,
    run_benchmark,
    write_benchmark_reports,
)
from voiceover_pipeline.models import ASRExecutionReceipt, ASRResult, ASRSegment


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

    assert report["schema_version"] == 1
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
        "supported_case_count": 5,
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
        "supported_case_count": 0,
    }


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
    nemotron = create_benchmark_adapter("nemotron-local", model_id=None, device="cpu", compute="auto")

    assert (whisper.provider_id, whisper.prompt_supported, whisper.timestamp_supported) == (
        "faster-whisper",
        False,
        True,
    )
    assert (qwen.provider_id, qwen.prompt_supported, qwen.timestamp_supported) == (
        "qwen-local",
        True,
        False,
    )
    assert (nemotron.provider_id, nemotron.prompt_supported, nemotron.timestamp_supported) == (
        "nemotron-local",
        False,
        False,
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
