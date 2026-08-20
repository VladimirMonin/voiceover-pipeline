import shutil
import sys
from pathlib import Path
from typing import Any

from voiceover_pipeline.config import (
    DEFAULT_TIMING_COMPUTE,
    DEFAULT_TIMING_DEVICE,
    DEFAULT_TIMING_MODEL,
    WHISPER_HF_REPOS,
)
from voiceover_pipeline.models import TimingResult, TimingSegment
from voiceover_pipeline.providers.base import TranscriptionProvider


def _detect_device(requested: str) -> str:
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested


def _detect_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cpu":
        return "int8"
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 12:
                return "float16"
            if cap[0] >= 7:
                return "int8_float16"
            return "int8"
    except Exception:
        pass
    return "int8"


TIMING_MODEL_SPEEDS: dict[str, str] = {
    "base": "fastest",
    "small": "fast",
    "medium": "balanced",
    "large-v3-turbo": "slow",
    "large-v3": "slowest",
}

TIMING_MODEL_SIZES: dict[str, tuple[int, int]] = {
    "base": (74, 148),
    "small": (244, 486),
    "medium": (769, 1536),
    "large-v3-turbo": (809, 1620),
    "large-v3": (1550, 3090),
}


class FasterWhisperProvider(TranscriptionProvider):
    provider_id = "faster-whisper"

    def __init__(
        self,
        model_size: str = DEFAULT_TIMING_MODEL,
        device: str = DEFAULT_TIMING_DEVICE,
        compute_type: str = DEFAULT_TIMING_COMPUTE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for model_id in WHISPER_HF_REPOS:
            params_m, disk_mb = TIMING_MODEL_SIZES.get(model_id, (0, 0))
            entry: dict[str, Any] = {
                "id": model_id,
                "parameters_m": params_m,
                "disk_mb": disk_mb,
                "speed": TIMING_MODEL_SPEEDS.get(model_id, "unknown"),
            }
            if model_id == DEFAULT_TIMING_MODEL:
                entry["default"] = True
            models.append(entry)
        return models

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "ru",
        word_timestamps: bool = False,
        quiet: bool = False,
    ) -> TimingResult:
        def _log(msg: str) -> None:
            if not quiet:
                print(msg, file=sys.stderr)

        if not shutil.which("ffprobe"):
            raise RuntimeError("FFprobe is required for audio transcription.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        resolved_device = _detect_device(self.device)
        resolved_compute = _detect_compute_type(self.compute_type, resolved_device)

        hf_repo = WHISPER_HF_REPOS.get(self.model_size, WHISPER_HF_REPOS["small"])

        _log(
            f"Loading Whisper model {self.model_size} ({hf_repo}) "
            f"on {resolved_device}/{resolved_compute} ..."
        )

        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(
                model_size_or_path=hf_repo,
                device=resolved_device,
                compute_type=resolved_compute,
            )
        except Exception as first_error:
            fallback_compute = "float32" if resolved_device == "cpu" else "float16"
            _log(
                f"  First attempt failed ({first_error}), retrying with "
                f"compute_type={fallback_compute}"
            )
            resolved_compute = fallback_compute
            model = WhisperModel(
                model_size_or_path=hf_repo,
                device=resolved_device,
                compute_type=resolved_compute,
            )

        segments_iter, info = model.transcribe(
            audio=str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=word_timestamps,
        )

        timing_segments: list[TimingSegment] = []
        for idx, seg in enumerate(segments_iter):
            start_ms_val = round(seg.start * 1000)
            end_ms_val = round(seg.end * 1000)
            duration_ms_val = end_ms_val - start_ms_val
            words_list = None
            if word_timestamps and seg.words:
                words_list = [
                    {
                        "word": w.word.strip(),
                        "start_ms": round(w.start * 1000),
                        "end_ms": round(w.end * 1000),
                    }
                    for w in seg.words
                ]
            timing_segments.append(
                TimingSegment(
                    id=idx,
                    start_sec=round(seg.start, 3),
                    end_sec=round(seg.end, 3),
                    start_ms=start_ms_val,
                    end_ms=end_ms_val,
                    duration_ms=duration_ms_val,
                    text=seg.text.strip(),
                    words=words_list,
                )
            )

        return TimingResult(
            segments=timing_segments,
            model=self.model_size,
            backend="faster-whisper",
            provider=self.provider_id,
            device=resolved_device,
            compute_type=resolved_compute,
            language=language,
            source_audio=str(audio_path.resolve()),
        )
