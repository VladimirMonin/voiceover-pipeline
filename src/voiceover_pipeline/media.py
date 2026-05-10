import re
import shutil
import subprocess
from pathlib import Path

from .config import CHANNELS, MP3_BITRATE, SAMPLE_RATE


def check_media_tools() -> tuple[str, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required but was not found in PATH.")

    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("FFprobe is required but was not found in PATH.")

    for tool_path in (ffmpeg_path, ffprobe_path):
        subprocess.run(
            [tool_path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return ffmpeg_path, ffprobe_path


def write_audio_as_mp3(ffmpeg_path: str, audio_bytes: bytes, audio_format: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_format == "mp3":
        output_path.write_bytes(audio_bytes)
        return

    if audio_format == "wav":
        temp_wav = output_path.with_suffix(".temp.wav")
        temp_wav.write_bytes(audio_bytes)
        try:
            _wav_to_mp3(ffmpeg_path, temp_wav, output_path)
        finally:
            temp_wav.unlink(missing_ok=True)
        return

    if audio_format not in ("pcm16", "pcm"):
        raise RuntimeError(f"Unsupported source audio format: {audio_format}")

    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(output_path),
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


def _wav_to_mp3(ffmpeg_path: str, wav_path: Path, mp3_path: Path) -> None:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(wav_path),
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(mp3_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


def mp3_duration_ms(ffprobe_path: str, input_path: Path) -> int:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return round(float(result.stdout.strip()) * 1000)


def trim_final_silence(ffmpeg_path: str, ffprobe_path: str, input_path: Path) -> None:
    duration_sec = mp3_duration_ms(ffprobe_path, input_path) / 1000
    detect = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-i",
            str(input_path),
            "-af",
            "silencedetect=noise=-45dB:d=1",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if detect.returncode != 0:
        raise RuntimeError(detect.stderr)

    silence_ranges: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in detect.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            current_start = float(start_match.group(1))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and current_start is not None:
            silence_ranges.append((current_start, float(end_match.group(1))))
            current_start = None

    if not silence_ranges:
        return

    final_start, final_end = silence_ranges[-1]
    if duration_sec - final_end > 0.35 or duration_sec - final_start < 1.5:
        return

    temp_path = input_path.with_suffix(".trimmed.mp3")
    result = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-to",
            f"{max(final_start + 0.2, 0.5):.3f}",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(temp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))

    temp_path.replace(input_path)


def concat_mp3_chunks(ffmpeg_path: str, chunks_dir: Path, output_path: Path) -> None:
    concat_audio_files(ffmpeg_path, sorted(chunks_dir.glob("chunk_*.mp3")), output_path)


def concat_audio_files(ffmpeg_path: str, chunk_paths: list[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise RuntimeError("No audio chunk files provided for concat")

    concat_list = output_path.with_suffix(".concat.txt")
    concat_list.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-codec:a",
                _concat_codec(output_path),
                *(_concat_bitrate_args(output_path)),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    finally:
        concat_list.unlink(missing_ok=True)


def _concat_codec(output_path: Path) -> str:
    if output_path.suffix.lower() == ".ogg":
        return "libopus"
    return "libmp3lame"


def _concat_bitrate_args(output_path: Path) -> list[str]:
    if output_path.suffix.lower() == ".ogg":
        return ["-b:a", "96k"]
    return ["-b:a", MP3_BITRATE]
