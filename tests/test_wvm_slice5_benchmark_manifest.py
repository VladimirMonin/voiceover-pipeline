import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from unittest import SkipTest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "wvm_slice5_benchmark" / "manifest.json"
REQUIRED_CASE_KEYS = {
    "audio_path",
    "case_id",
    "category",
    "duration_s",
    "expected_text",
    "forbidden_anchors",
    "language",
    "noise",
    "pause",
    "prompt_sentinels",
    "reference_path",
    "reference_sha256",
    "required_anchors",
    "sha256",
    "state",
    "tier",
}
EXPECTED_CATEGORY_COUNTS = {
    "chunked_prompt_tail": 6,
    "degraded_audio": 7,
    "hallucination_phrase": 6,
    "language_detection": 7,
    "no_speech": 6,
    "positive_speech": 6,
    "prompt_echo": 6,
    "prompt_quality": 6,
}
EXPECTED_TIER_COUNTS = {"P0": 16, "P1": 22, "P2": 12}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wvm_root() -> Path:
    configured = os.environ.get("WVM_ROOT")
    if not configured:
        raise SkipTest("WVM_ROOT is not configured; local WVM reference validation is skipped")
    root = Path(configured)
    if not root.is_dir():
        raise SkipTest(f"WVM_ROOT does not exist: {root}")
    return root.resolve()


def test_wvm_slice5_manifest_is_canonical_and_category_balanced():
    raw_manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest = json.loads(raw_manifest)

    assert raw_manifest == json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert manifest["schema_version"] == 1
    assert manifest["corpus_id"] == "wvm-slice5-local-reference-v1"
    assert manifest["license"]["audio_and_references"] == "NOASSERTION"
    assert manifest["privacy"] == {
        "contains_personal_data": False,
        "contains_private_recordings": False,
        "source": "owner-approved local synthetic WVM Slice 5 evaluation assets",
    }
    assert manifest["local_reference"]["asset_root"] == "tests/assets/whisper/slice5"
    assert "--corpus-root" in manifest["local_reference"]["root_resolution"]
    assert "no WVM audio binary or code is copied" in manifest["provenance"]["note"]

    cases = manifest["cases"]
    assert len(cases) == 50
    assert len({case["case_id"] for case in cases}) == 50
    assert Counter(case["category"] for case in cases) == EXPECTED_CATEGORY_COUNTS
    assert Counter(case["tier"] for case in cases) == EXPECTED_TIER_COUNTS
    assert manifest["selection"]["category_counts"] == EXPECTED_CATEGORY_COUNTS
    assert manifest["selection"]["tier_counts"] == EXPECTED_TIER_COUNTS

    for case in cases:
        assert REQUIRED_CASE_KEYS <= case.keys()
        assert not Path(case["audio_path"]).is_absolute()
        assert not Path(case["reference_path"]).is_absolute()
        assert Path(case["audio_path"]).suffix == ".ogg"
        assert Path(case["reference_path"]).suffix == ".txt"
        assert case["reference_path"] == str(Path(case["audio_path"]).with_suffix(".txt"))
        assert case["expected_text"]
        assert case["duration_s"] > 0
        assert case["language"] in {None, "de", "en", "es", "ru"}
        assert case["state"] in {
            "diagnostic",
            "filtered_or_rejected",
            "no_speech",
            "non_empty",
        }
        assert len(case["sha256"]) == len(case["reference_sha256"]) == 64
        assert set(case["noise"]) == {"category"}
        assert case["noise"]["category"]
        assert set(case["pause"]) == {"leading_s", "trailing_s"}
        assert case["pause"]["leading_s"] >= 0
        assert case["pause"]["trailing_s"] >= 0


def test_wvm_slice5_manifest_matches_the_local_external_pairs_when_configured():
    root = _wvm_root()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for case in manifest["cases"]:
        audio_path = (root / case["audio_path"]).resolve()
        reference_path = (root / case["reference_path"]).resolve()
        assert audio_path.is_relative_to(root)
        assert reference_path.is_relative_to(root)
        assert audio_path.is_file(), case["case_id"]
        assert reference_path.is_file(), case["case_id"]
        assert _sha256(audio_path) == case["sha256"]
        assert _sha256(reference_path) == case["reference_sha256"]
        assert reference_path.read_text(encoding="utf-8").strip() == case["expected_text"]


def test_wvm_slice5_manifest_documentation_covers_local_root_and_redistribution_boundary():
    documentation = (PROJECT_ROOT / "docs" / "wvm-slice5-local-reference-benchmark.md").read_text(
        encoding="utf-8"
    )

    assert "WVM_ROOT" in documentation
    assert "--corpus-root" in documentation
    assert "NOASSERTION" in documentation
    assert "redistribution" in documentation.lower()
    assert "missing" in documentation.lower()
    assert "No WVM audio binary" in documentation
