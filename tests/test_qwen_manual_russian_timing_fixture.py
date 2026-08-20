from __future__ import annotations

from pathlib import Path

from voiceover_pipeline.asr_benchmark import (
    ResourceSnapshot,
    _case_record,
    _load_manifest,
    normalize_transcript,
)
from voiceover_pipeline.models import ASRExecutionReceipt, ASRResult, ASRWordSpan

MANIFEST = Path(__file__).parent / "fixtures" / "qwen_manual_russian_timing" / "manifest.json"


def test_manual_russian_qwen_timing_subset_is_valid_and_direct_audio_annotated():
    manifest, cases = _load_manifest(MANIFEST)

    assert manifest["corpus_id"] == "wvm-qwen-manual-russian-timing-v1"
    assert len(cases) == 2
    annotation = manifest["annotation"]
    assert annotation["accuracy_limit_ms"] == 100
    assert "manual direct-audio inspection" in annotation["method"]
    assert "no ASR output was consulted" in annotation["method"]

    for case in cases:
        assert case["category"] == "positive_speech"
        assert case["language"] == "ru"
        assert case["audio_path"].startswith("tests/assets/whisper/slice5/ru_")
        assert case["source_case_id"].startswith(("p0_ru_", "p1_ru_"))
        expected_words = case["expected_word_timestamps"]
        assert normalize_transcript(case["expected_text"]) == normalize_transcript(
            " ".join(word["text"] for word in expected_words)
        )
        previous_end_s = 0.0
        for word in expected_words:
            assert previous_end_s <= word["start_s"] <= word["end_s"] <= case["duration_s"]
            previous_end_s = word["end_s"]


def test_manual_russian_qwen_timing_subset_runs_through_case_metrics():
    _manifest, cases = _load_manifest(MANIFEST)

    for case in cases:
        expected_words = case["expected_word_timestamps"]
        assert case["expected_speech_window_s"] == [
            expected_words[0]["start_s"],
            expected_words[-1]["end_s"],
        ]
        result = ASRResult(
            transcript=case["expected_text"],
            provider_id="qwen-local",
            model_id="fixture-model",
            language="Russian",
            execution=ASRExecutionReceipt(runtime="fixture"),
            words=tuple(ASRWordSpan(**word) for word in expected_words),
            alignment_origin="forced",
        )

        record = _case_record(
            case=case,
            result=result,
            wall_s=1.0,
            resources=ResourceSnapshot(peak_ram_bytes=1, peak_vram_bytes=2),
            timestamp_supported=True,
        )

        assert record["timestamp_boundary"] == {
            "available": False,
            "end_absolute_error_s": None,
            "mean_absolute_error_s": None,
            "reason": "provider returned no segment timestamps",
            "start_absolute_error_s": None,
            "status": "NOT_RUN",
            "supported": True,
        }
        assert record["word_timing"] == {
            "alignment_origin": "forced",
            "boundary_mae_s": 0.0,
            "boundary_p95_s": 0.0,
            "coverage": 1.0,
            "drift_s": 0.0,
            "in_bounds": True,
            "monotonic": True,
            "reference_word_count": len(expected_words),
            "reason": None,
            "signed_boundary_error_s": 0.0,
            "status": "MEASURED",
            "word_count": len(expected_words),
            "zero_duration_count": 0,
        }
