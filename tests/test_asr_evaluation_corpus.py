import hashlib
import json
import wave
from pathlib import Path

from conftest import fixture_path

CORPUS_ROOT = fixture_path("asr_evaluation")
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
REQUIRED_CASE_KEYS = {
    "audio_path",
    "case_id",
    "category",
    "duration_s",
    "expected_speech_window_s",
    "expected_text",
    "language",
    "noise",
    "pause",
    "provenance",
    "sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asr_evaluation_corpus_manifest_loads_with_deterministic_case_order():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["corpus_id"] == "synthetic-asr-evaluation-v1"
    assert manifest["privacy"] == {
        "contains_private_recordings": False,
        "contains_personal_data": False,
        "source": "locally-generated synthetic speech only",
    }
    assert manifest["provenance"]["generator"] == "tools/generate_asr_evaluation_corpus.py"
    assert manifest["provenance"]["wvm_assets_used"] is False

    cases = manifest["cases"]
    assert [case["case_id"] for case in cases] == [
        "clean-short",
        "leading-pause",
        "trailing-pause",
        "white-noise",
        "pause-and-white-noise",
    ]
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_asr_evaluation_corpus_cases_are_hashed_pcm_wav_with_complete_metadata():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for case in manifest["cases"]:
        assert REQUIRED_CASE_KEYS <= case.keys()
        assert case["expected_text"]
        assert case["language"] == "en"
        assert case["category"] in {"clean", "pause", "noise", "pause_noise"}
        assert case["noise"]["category"] in {"none", "white"}
        assert set(case["pause"]) == {"leading_s", "trailing_s"}
        assert case["pause"]["leading_s"] >= 0
        assert case["pause"]["trailing_s"] >= 0

        audio_path = CORPUS_ROOT / case["audio_path"]
        assert audio_path.is_file()
        assert _sha256(audio_path) == case["sha256"]

        with wave.open(str(audio_path), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16_000
            assert audio.getcomptype() == "NONE"
            duration_s = audio.getnframes() / audio.getframerate()

        assert duration_s == case["duration_s"]
        speech_start, speech_end = case["expected_speech_window_s"]
        assert 0 <= speech_start < speech_end <= duration_s


def test_asr_evaluation_corpus_documents_provenance_and_license_boundaries():
    docs_path = Path(__file__).resolve().parent.parent / "docs" / "asr-evaluation-corpus.md"
    docs = docs_path.read_text(encoding="utf-8")

    assert "No WVM asset is copied or referenced by the corpus." in docs
    assert "NOASSERTION" in docs
    assert "private recording" in docs.lower()
