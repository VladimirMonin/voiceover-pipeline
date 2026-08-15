#!/usr/bin/env python3
"""Generate and verify the privacy-safe synthetic ASR evaluation corpus."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "asr_evaluation"
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
VOICE = "kal"

CASE_SPECS = (
    {
        "case_id": "clean-short",
        "category": "clean",
        "expected_text": "The quick brown fox checks one two three.",
        "pause": {"leading_s": 0.0, "trailing_s": 0.0},
        "noise": {"category": "none", "amplitude": 0.0, "seed": None},
    },
    {
        "case_id": "leading-pause",
        "category": "pause",
        "expected_text": "A pause comes before this sentence.",
        "pause": {"leading_s": 0.75, "trailing_s": 0.0},
        "noise": {"category": "none", "amplitude": 0.0, "seed": None},
    },
    {
        "case_id": "trailing-pause",
        "category": "pause",
        "expected_text": "This sentence ends before a pause.",
        "pause": {"leading_s": 0.0, "trailing_s": 0.75},
        "noise": {"category": "none", "amplitude": 0.0, "seed": None},
    },
    {
        "case_id": "white-noise",
        "category": "noise",
        "expected_text": "White noise makes speech recognition harder.",
        "pause": {"leading_s": 0.0, "trailing_s": 0.0},
        "noise": {"category": "white", "amplitude": 0.012, "seed": 4_104},
    },
    {
        "case_id": "pause-and-white-noise",
        "category": "pause_noise",
        "expected_text": "A long pause and white noise test the boundary.",
        "pause": {"leading_s": 0.5, "trailing_s": 0.5},
        "noise": {"category": "white", "amplitude": 0.015, "seed": 5_105},
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _render_speech(text: str, output_path: Path) -> None:
    text_path = output_path.with_suffix(".source-text.txt")
    _write_text_file(text_path, text)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=textfile={text_path}:voice={VOICE}",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                str(CHANNELS),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
        )
    finally:
        text_path.unlink(missing_ok=True)


def _read_pcm_samples(path: Path) -> array:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != CHANNELS
            or audio.getsampwidth() != SAMPLE_WIDTH
            or audio.getframerate() != SAMPLE_RATE
            or audio.getcomptype() != "NONE"
        ):
            raise RuntimeError(f"Unexpected intermediate audio format: {path}")
        frames = audio.readframes(audio.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _xorshift32(state: int) -> tuple[int, int]:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF, state & 0xFFFF


def _add_white_noise(samples: array, amplitude: float, seed: int) -> None:
    state = seed
    scale = int(amplitude * 32_767)
    for index, sample in enumerate(samples):
        state, value = _xorshift32(state)
        noise = ((value - 32_768) * scale) // 32_768
        samples[index] = max(-32_768, min(32_767, sample + noise))


def _write_wav(path: Path, samples: array) -> None:
    raw = array("h", samples)
    if sys.byteorder != "little":
        raw.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(CHANNELS)
        audio.setsampwidth(SAMPLE_WIDTH)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(raw.tobytes())


def _case_manifest_entry(spec: dict[str, object], audio_path: Path, speech_frames: int) -> dict[str, object]:
    pause = spec["pause"]
    assert isinstance(pause, dict)
    leading_s = float(pause["leading_s"])
    trailing_s = float(pause["trailing_s"])
    with wave.open(str(audio_path), "rb") as audio:
        total_frames = audio.getnframes()
    duration_s = total_frames / SAMPLE_RATE
    speech_start = leading_s
    speech_end = speech_start + speech_frames / SAMPLE_RATE
    return {
        "case_id": spec["case_id"],
        "audio_path": str(Path("audio") / audio_path.name),
        "sha256": _sha256(audio_path),
        "duration_s": duration_s,
        "expected_text": spec["expected_text"],
        "language": "en",
        "category": spec["category"],
        "pause": pause,
        "noise": spec["noise"],
        "expected_speech_window_s": [speech_start, speech_end],
        "provenance": {
            "source": "locally-generated synthetic speech",
            "source_recording": False,
            "text_authoring": "project-owned fixture text",
            "rendering": "FFmpeg flite filter with voice=kal",
            "noise_transform": "deterministic xorshift32 white-noise mixer",
        },
    }


def build_corpus(output_root: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg with the flite filter is required to generate this corpus")

    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for spec in CASE_SPECS:
        case_id = str(spec["case_id"])
        raw_path = audio_root / f".{case_id}.raw.wav"
        output_path = audio_root / f"{case_id}.wav"
        _render_speech(str(spec["expected_text"]), raw_path)
        speech_samples = _read_pcm_samples(raw_path)
        raw_path.unlink(missing_ok=True)

        pause = spec["pause"]
        noise = spec["noise"]
        assert isinstance(pause, dict)
        assert isinstance(noise, dict)
        leading_frames = int(float(pause["leading_s"]) * SAMPLE_RATE)
        trailing_frames = int(float(pause["trailing_s"]) * SAMPLE_RATE)
        samples = array("h", [0]) * leading_frames
        samples.extend(speech_samples)
        samples.extend(array("h", [0]) * trailing_frames)
        if noise["category"] == "white":
            _add_white_noise(samples, float(noise["amplitude"]), int(noise["seed"]))
        _write_wav(output_path, samples)
        cases.append(_case_manifest_entry(spec, output_path, len(speech_samples)))

    manifest = {
        "schema_version": 1,
        "corpus_id": "synthetic-asr-evaluation-v1",
        "privacy": {
            "contains_private_recordings": False,
            "contains_personal_data": False,
            "source": "locally-generated synthetic speech only",
        },
        "license": {
            "audio": "NOASSERTION",
            "metadata_and_generator": "MIT (repository license)",
            "reason": "FFmpeg/flite voice-output redistribution has not been separately audited.",
        },
        "provenance": {
            "generator": "tools/generate_asr_evaluation_corpus.py",
            "audio_renderer": "FFmpeg flite filter with voice=kal",
            "noise_renderer": "in-process deterministic xorshift32 mixer",
            "wvm_assets_used": False,
        },
        "cases": cases,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_corpus(output_root: Path) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported corpus manifest schema")
    for case in manifest["cases"]:
        audio_path = output_root / case["audio_path"]
        if not audio_path.is_file() or _sha256(audio_path) != case["sha256"]:
            raise RuntimeError(f"Integrity check failed: {audio_path}")
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
                raise RuntimeError(f"Unexpected WAV format: {audio_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Verify the existing manifest and WAV hashes")
    args = parser.parse_args()
    if args.check:
        validate_corpus(args.output)
        print(f"validated {args.output}")
    else:
        build_corpus(args.output)
        validate_corpus(args.output)
        print(f"generated {args.output}")


if __name__ == "__main__":
    main()
